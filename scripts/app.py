"""Lancer avec python -m streamlit run scripts/app.py."""
import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import psycopg
import pymupdf
import streamlit as st

from indexer import ROOT, connexion, index
from question import GenerationError, repondre_document

EXEMPLE = ROOT / "data" / "1706.03762v7.pdf"


def deja_indexe(digest):
    with connexion() as conn:
        if not conn.execute("SELECT to_regclass('maintidoc_fragments')").fetchone()[0]:
            return False
        return conn.execute(
            "SELECT 1 FROM maintidoc_fragments WHERE document_sha256=%s LIMIT 1",
            (digest,),
        ).fetchone() is not None


def apercu(data):
    with pymupdf.open(stream=data) as doc:
        if not doc.is_pdf or doc.needs_pass or not len(doc):
            raise ValueError
        page = doc[0]
        scale = 600 / max(page.rect.width, page.rect.height)
        preview = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png")
        return len(doc), preview


st.set_page_config(page_title="MaintiDoc | Lecture documentaire", page_icon=":material/menu_book:", layout="wide")
st.html("""
<style>
/* Hallmark v2 | shape: app screen | theme: Cobalt | signature: original PDF page */
/* Critique: one reading surface, original page facing the question. */
:root { --rule:#ccd4e2; --focus:#234bc0; --paper:#fff; --panel:#e9edf4; }
html,body {overflow-x:clip}
.stMainBlockContainer {max-width:72rem;padding-top:2.5rem}
h1,h2,h3 {overflow-wrap:break-word;min-width:0}
[data-testid="stImage"] {background:var(--panel);padding:1rem;border:1px solid var(--rule)}
[data-testid="stImage"] img {max-height:22rem;object-fit:contain;background:var(--paper)}
:focus-visible {outline:2px solid var(--focus);outline-offset:2px}
@media(max-width:640px){.stMainBlockContainer{padding:1.5rem 1rem}[data-testid="stImage"] img{max-height:10rem}}
</style>
""")
st.title("MaintiDoc")
st.write("Interrogez votre document et retrouvez la page citée.")
st.divider()
left, right = st.columns([1, 1.4], gap="large")
with left:
    st.subheader("Document")
    upload = st.file_uploader("Charger un PDF", type=["pdf"], max_upload_size=20)
    name = upload.name if upload else EXEMPLE.name
    identity = upload.file_id if upload else "exemple"
    if st.session_state.get("document") != identity:
        st.session_state.pop("resultat", None)
        st.session_state.historique = []
        data = upload.getvalue() if upload else EXEMPLE.read_bytes()
        try:
            pages, preview = apercu(data)
        except (RuntimeError, ValueError, pymupdf.FileDataError):
            st.error("Impossible de lire ce PDF. Choisissez un fichier valide et sans mot de passe.")
            st.stop()
        st.session_state.update(document=identity, pages=pages, preview=preview,
                                digest=hashlib.sha256(data).hexdigest())
    pages = st.session_state.pages
    st.text(f"{name}  ·  {pages} page{'s' if pages > 1 else ''}")
    st.image(st.session_state.preview, alt="Première page du PDF sélectionné", width="stretch")

with right:
    st.subheader("Question au document")
    with st.form(f"form-{identity}"):
        question = st.text_area("Votre question", placeholder="Quelle information cherchez-vous ?", max_chars=1500)
        envoyer = st.form_submit_button("Rechercher la réponse", type="primary")
    if envoyer:
        st.session_state.pop("resultat", None)
        if not question.strip():
            st.warning("Saisissez une question.")
        else:
            try:
                digest = st.session_state.digest
                if not deja_indexe(digest):
                    with st.spinner("Indexation du PDF…"), TemporaryDirectory() as directory:
                        path = EXEMPLE
                        if upload:
                            path = Path(directory) / "document.pdf"
                            path.write_bytes(upload.getbuffer())
                        index(path)
                with st.spinner("Recherche des passages et rédaction…"):
                    trace = []
                    historique = st.session_state.get("historique", [])
                    answer = repondre_document(question, digest, name, historique, trace)
                st.session_state.historique = (historique + [(question, answer)])[-2:]
                st.session_state.outils = trace
                st.session_state.resultat = (question, answer)
            except (psycopg.Error, httpx.HTTPError, OSError):
                logging.exception("Service local inaccessible")
                st.error("Vérifiez PostgreSQL, le fichier .env et Ollama, puis réessayez.")
            except GenerationError as exc:
                st.error(str(exc))
            except (ValueError, RuntimeError, KeyError):
                logging.exception("Traitement du document impossible")
                st.error("La lecture du contenu ou la génération a échoué. Vérifiez que le PDF contient du texte et réessayez.")
    if "resultat" in st.session_state:
        asked, answer = st.session_state.resultat
        st.caption("Outils consultés : " + ", ".join(st.session_state.get("outils", [])))
        st.subheader("Réponse")
        st.text(asked)
        body, _, sources = answer.partition("\n\nSources\n")
        st.text(body)
        if sources:
            st.subheader("Sources")
            st.text(sources)
