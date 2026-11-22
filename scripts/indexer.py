"""Indexation du PDF de travail MaintiDoc, avec références aux pages physiques."""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import httpx
import psycopg
import pymupdf
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from psycopg.types.json import Jsonb

MODEL = "qwen3-embedding:0.6b"
ROOT = Path(__file__).resolve().parent.parent


class ArticleReader(BaseReader):
    """Lecteur LlamaIndex pour les figures légendées et tableaux réglés de cet article."""

    def load_data(self, path, digest):
        docs = []
        folder = ROOT / "extraits" / digest
        folder.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(path) as pdf:
            for page in pdf:
                blocks = page.get_text("blocks", sort=True)
                occupied = []
                base = dict(source=path.name, page=page.number + 1, sha256=digest)
                for block in blocks:
                    caption = " ".join(block[4].split())
                    match = re.match(r"^(Table|Figure) (\d+):", caption)
                    if not match:
                        continue
                    label, number = match.groups()
                    if label == "Table":
                        # Les quatre tableaux sont délimités par de longs filets horizontaux.
                        rules = [s["rect"] for s in page.get_drawings()
                                 if s["rect"].width > page.rect.width * .35
                                 and s["rect"].height < 2 and s["rect"].y0 >= block[3]]
                        if len(rules) < 2:
                            raise ValueError(f"Limites du tableau {number} à vérifier")
                        box = pymupdf.Rect(min(r.x0 for r in rules)-1, min(r.y0 for r in rules)-1,
                                           max(r.x1 for r in rules)+1, max(r.y1 for r in rules)+1)
                        body = page.get_text(clip=box, sort=True).strip()
                        kind = "tableau"
                    else:
                        # La figure 2 comporte deux images. Les figures en annexe sont vectorielles.
                        images = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()
                                  if i["bbox"][3] <= block[1]+2]
                        if images:
                            box = pymupdf.Rect(images[0])
                            for rect in images[1:]:
                                box |= rect
                        else:
                            box = pymupdf.Rect(0, 48, page.rect.width, block[1])
                        body = ""
                        kind = "figure"
                    asset = folder / f"{kind}-{number}.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=box).save(asset)
                    content = (caption + "\n" + body).strip()
                    if kind == "tableau":
                        asset.with_suffix(".txt").write_text(content, encoding="utf-8")
                    docs.append(Document(text=content, metadata=dict(base, type=kind, numero=int(number),
                                         fichier=asset.relative_to(ROOT).as_posix())))
                    occupied.extend([box, pymupdf.Rect(block[:4])])
                parts = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
                         and not any(rect.contains(pymupdf.Rect(b[:4]).tl + (2, 2)) for rect in occupied)]
                if parts:
                    docs.append(Document(text="\n\n".join(parts), metadata=dict(base, type="texte")))
        return docs


def prepare(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    docs = ArticleReader().load_data(path, digest)
    # Compter les mots localement évite un téléchargement de tokenizer.
    splitter = SentenceSplitter(chunk_size=250, chunk_overlap=30, tokenizer=str.split)
    records = []
    for doc in docs:
        chunks = splitter.split_text(doc.text) if doc.metadata["type"] == "texte" else [doc.text]
        for i, chunk in enumerate(chunks, 1):
            records.append((chunk, dict(doc.metadata, fragment=i)))
    return digest, records


def connexion():
    password = os.getenv("POSTGRES_PASSWORD")
    if not password and not os.getenv("DATABASE_URL"):
        password = next(line.split("=", 1)[1] for line in (ROOT / ".env").read_text().splitlines()
                        if line.startswith("POSTGRES_PASSWORD="))
    options = {} if os.getenv("DATABASE_URL") else dict(host="127.0.0.1", port=55432,
              dbname="maintidoc", user="postgres", password=password)
    return psycopg.connect(os.getenv("DATABASE_URL", ""), **options)


def index(path):
    digest, records = prepare(path)
    print("Extraction", dict(Counter(m["type"] for _, m in records)), flush=True)
    vectors = []
    with httpx.Client(timeout=300, trust_env=False) as client:
        for offset in range(0, len(records), 8):
            batch = [t for t, _ in records[offset:offset+8]]
            response = client.post("http://127.0.0.1:11434/api/embed",
                                   json=dict(model=MODEL, input=batch, truncate=False))
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
    rows = [(digest, i, text, Jsonb(meta), MODEL, json.dumps(vec, allow_nan=False))
            for i, ((text, meta), vec) in enumerate(zip(records, vectors, strict=True))]
    with connexion() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("SELECT pg_advisory_xact_lock(170603762)")
        conn.execute("""CREATE TABLE IF NOT EXISTS maintidoc_fragments (
            document_sha256 text NOT NULL, position integer NOT NULL, contenu text NOT NULL,
            metadata jsonb NOT NULL, modele text NOT NULL, embedding vector(1024) NOT NULL,
            PRIMARY KEY(document_sha256, position))""")
        conn.execute("DELETE FROM maintidoc_fragments WHERE document_sha256=%s", (digest,))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO maintidoc_fragments VALUES (%s,%s,%s,%s,%s,%s::vector)", rows)
        print("Vecteurs et dimension", conn.execute("""SELECT count(*), min(vector_dims(embedding))
              FROM maintidoc_fragments WHERE document_sha256=%s""", (digest,)).fetchone())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="?", default=ROOT / "data" / "1706.03762v7.pdf")
    index(parser.parse_args().pdf)

