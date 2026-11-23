"""Indexer les figures du PDF ou les rechercher par description ou par image."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import httpx


from indexer import MODEL, ROOT, connexion

CLIP = "Qdrant/clip-ViT-B-32-vision"
VISION = "qwen3.5:4b"
OLLAMA = "http://127.0.0.1:11434/api"


def clip_model():
    from fastembed import ImageEmbedding

    return ImageEmbedding(model_name=CLIP, cache_dir=str(ROOT / ".modeles"), threads=2)


def post(client, endpoint, payload):
    response = client.post(f"{OLLAMA}/{endpoint}", json=payload)
    response.raise_for_status()
    return response.json()


def embed_text(client, texts):
    return post(client, "embed", dict(model=MODEL, input=texts, truncate=False))["embeddings"]


def traiter(digest, client):
    with connexion() as conn:
        figures = conn.execute("""SELECT metadata, contenu FROM maintidoc_fragments
            WHERE document_sha256=%s AND metadata->>'type'='figure'
            ORDER BY (metadata->>'numero')::int""", (digest,)).fetchall()
    if not figures:
        raise SystemExit("Aucune figure pour ce PDF. Exécutez d’abord indexer.py.")
    paths = [ROOT / meta["fichier"] for meta, _ in figures]
    visual = list(clip_model().embed(paths))
    descriptions = []
    for (meta, caption), path in zip(figures, paths, strict=True):
        prompt = ("Décris cette figure en français en deux phrases utiles à la recherche. "
                  "Identifie les éléments visibles et leurs relations. La légende de la source est : " + caption)
        result = post(client, "chat", dict(model=VISION, stream=False, think=False,
                      messages=[dict(role="user", content=prompt,
                      images=[base64.b64encode(path.read_bytes()).decode("ascii")])],
                      options=dict(temperature=0, num_predict=220)))
        description = result["message"]["content"].strip()
        if not description:
            raise ValueError(f"Description vide pour {path.name}")
        descriptions.append(description)
        print(f"Figure {meta['numero']} décrite", flush=True)
    vectors = embed_text(client, descriptions)
    with connexion() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(170603762)")
        conn.execute("""CREATE TABLE IF NOT EXISTS maintidoc_figures (
            document_sha256 text NOT NULL, numero integer NOT NULL, page integer NOT NULL,
            fichier text NOT NULL, description text NOT NULL, clip vector(512) NOT NULL,
            embedding vector(1024) NOT NULL, modele_clip text NOT NULL, modele_vision text NOT NULL,
            modele_texte text NOT NULL, PRIMARY KEY(document_sha256, numero))""")
        conn.execute("DELETE FROM maintidoc_figures WHERE document_sha256=%s", (digest,))
        for (meta, _), desc, img, vec in zip(figures, descriptions, visual, vectors, strict=True):
            conn.execute("""INSERT INTO maintidoc_figures VALUES
                (%s,%s,%s,%s,%s,%s::vector,%s::vector,%s,%s,%s)""",
                (digest, meta["numero"], meta["page"], meta["fichier"], desc,
                 json.dumps(img.tolist(), allow_nan=False), json.dumps(vec, allow_nan=False),
                 CLIP, VISION, MODEL))
    print(f"{len(figures)} figures indexées.")


def rechercher(digest, client, question, image):
    if image:
        vector = next(iter(clip_model().embed([image]))).tolist()
        distance = "clip <=> %s::vector"
    else:
        vector = embed_text(client, [question])[0]
        distance = "embedding <=> %s::vector"
    with connexion() as conn:
        rows = conn.execute(f"""SELECT numero, page, description FROM maintidoc_figures
              WHERE document_sha256=%s ORDER BY {distance}, numero LIMIT 3""",
              (digest, json.dumps(vector))).fetchall()
    for numero, page, desc in rows:
        print(f"Figure {numero}, page {page} — {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "1706.03762v7.pdf")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--question")
    mode.add_argument("--image", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    with httpx.Client(timeout=300, trust_env=False) as client:
        if args.question or args.image:
            rechercher(digest, client, args.question, args.image)
        else:
            traiter(digest, client)
