"""Routage LangChain vers les index du PDF sélectionné."""
import json
import re

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from ollama import ResponseError

from figures import OLLAMA, embed_text
from indexer import connexion

GENERATEUR = "qwen3.5:4b"


def retrouver(client, digest, question, historique, trace):
    def outil(categorie, description):
        def chercher(requete: str, numero: int = 0):

            with connexion() as conn:
                if numero and categorie != "texte":
                    return conn.execute("""SELECT contenu, metadata FROM maintidoc_fragments
                        WHERE document_sha256=%s AND metadata->>'type'=%s
                        AND (metadata->>'numero')::int=%s ORDER BY position""",
                        (digest, categorie, numero)).fetchall()
                vecteur = json.dumps(embed_text(client, [requete])[0])
                jointure = ""
                distance = "f.embedding <=> %(v)s::vector"
                if categorie == "figure" and conn.execute("SELECT to_regclass('maintidoc_figures')").fetchone()[0]:
                    jointure = """LEFT JOIN maintidoc_figures i ON i.document_sha256=f.document_sha256
                                  AND i.numero=(f.metadata->>'numero')::int"""
                    distance = f"least({distance}, i.embedding <=> %(v)s::vector)"
                return conn.execute(f"""SELECT f.contenu, f.metadata FROM maintidoc_fragments f {jointure}
                    WHERE f.document_sha256=%(digest)s AND f.metadata->>'type'=%(type)s
                    ORDER BY {distance}, f.position LIMIT 3""",
                    dict(digest=digest, type=categorie, v=vecteur)).fetchall()
        return StructuredTool.from_function(chercher, name=f"chercher_{categorie}", description=description)

    outils = [
        outil("texte", "Explications et faits dans les paragraphes du PDF."),
        outil("tableau", "Résultats chiffrés, scores et comparaisons dans les tableaux."),
        outil("figure", "Schémas, illustrations et légendes des figures."),
    ]
    disponibles = {o.name: o for o in outils}
    modele = ChatOllama(model=GENERATEUR, base_url=OLLAMA.removesuffix("/api"),
                        temperature=0, reasoning=False, num_ctx=8192, num_predict=256,
                        client_kwargs={"timeout": 180, "trust_env": False})
    demande = json.dumps(dict(question=question, historique=historique), ensure_ascii=False)
    try:
        choix = modele.bind_tools(outils).invoke([
            ("system", "Appelle les outils documentaires utiles. Formule une requête autonome par outil. "
             "Pour une comparaison qui poursuit la question précédente, passe dans numero "
             "le numéro du tableau ou de la figure cité dans sa source. Si la demande désigne "
             "une autre figure ou un autre tableau, prends son numéro. Pour une nouvelle "
             "recherche sans numéro explicite, utilise 0. Le numéro est celui de l'objet, pas la page."),
            ("human", demande)])
    except ResponseError as exc:
        raise RuntimeError("Le modèle local ne peut pas sélectionner un outil.") from exc

    recherches = {}
    for categorie, motif in (("tableau", r"tableaux?|tables?"), ("figure", r"figures?")):
        for groupe in re.findall(rf"\b(?:{motif})\s+(\d{{1,6}}\b(?:\s*(?:,|et|and|&)\s*\d{{1,6}}\b)*)", question, re.I):
            for numero in re.findall(r"\d+", groupe):
                recherches[(f"chercher_{categorie}", int(numero))] = question
    explicites = {nom for nom, _ in recherches}
    for appel in choix.tool_calls:
        nom = appel["name"]
        requete = appel["args"].get("requete")
        if nom in disponibles and nom not in explicites and isinstance(requete, str) and requete.strip():
            numero = appel["args"].get("numero", 0)
            if type(numero) is not int or not 0 <= numero <= 2147483647:
                numero = 0
            categorie = nom.removeprefix("chercher_")
            if historique and re.search(r"\b(même|mêmes|ce tableau|cette figure)\b|^\s*et (pour|avec|en)\b",
                                          question, re.I):
                # Une comparaison peut viser le même test alors que deux tableaux
                # donnent des scores différents (jeu de développement et jeu de test).
                sources = historique[-1][1].partition("\n\nSources\n")[2]
                precedents = set(re.findall(rf"\({categorie} (\d+)\)$", sources, re.M))
                if len(precedents) == 1 and categorie in ("tableau", "figure"):
                    numero = int(precedents.pop())
            recherches.setdefault((nom, numero if categorie != "texte" else 0), requete[:1500])
    passages = []
    for (nom, numero), requete in recherches.items():
        passages.extend(disponibles[nom].invoke({"requete": requete, "numero": numero}))
        if nom not in trace:
            trace.append(nom)
    if not passages and "chercher_texte" not in trace:
        passages = disponibles["chercher_texte"].invoke({"requete": question})
        trace.append("chercher_texte (repli)")
    return passages
