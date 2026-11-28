import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import agent
import indexer
import question


class RegressionTests(unittest.TestCase):
    def test_compare_tables_even_if_router_repeats_first_call(self):
        call = {"name": "chercher_tableau", "args": {"requete": "BLEU", "numero": 2}}
        with patch.object(agent, "ChatOllama") as model, patch.object(agent, "connexion") as db:
            model.return_value.bind_tools.return_value.invoke.return_value = SimpleNamespace(tool_calls=[call, call])
            execute = db.return_value.__enter__.return_value.execute
            execute.side_effect = lambda sql, params: SimpleNamespace(fetchall=lambda: [params])
            for prompt in ("Compare le tableau 2 et le tableau 3", "Compare les tableaux 2, 3 et 2"):
                rows = agent.retrouver(None, "pdf", prompt, [], [])
                self.assertEqual(rows, [("pdf", "tableau", 2), ("pdf", "tableau", 3)])

    def test_invalid_generation_is_retryable(self):
        with patch.object(question, "retrouver", return_value=[("texte", {"type": "texte"})]):
            for content in ("[]", "null", "{", '{"reponse":"x","sources":[9]}'):
                with self.subTest(content=content), patch.object(question, "post", return_value={"message":{"content":content}}):
                    with self.assertRaisesRegex(question.GenerationError, "Réessayez"):
                        question.repondre_document("test", "hash", "test.pdf")

    def test_visual_evidence_sent_with_caption(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure.png"
            Image.new("RGB", (10, 10), "blue").save(path)
            row = ("Légende", dict(type="figure", numero=1, page=2, fragment=1, fichier=path.name))
            with patch.object(question, "ROOT", path.parent), patch.object(question, "retrouver", return_value=[row]), patch.object(question, "post") as post:
                post.return_value = {"message":{"content":'{"reponse":"Bleu","sources":[1]}'}}
                self.assertIn("page 2", question.repondre_document("Couleur ?", "hash", "test.pdf"))
                msg = post.call_args.args[2]["messages"][-1]
                self.assertTrue(msg["images"][0].startswith("iVBOR"))
                self.assertEqual(json.loads(msg["content"])["extraits"][0]["image"], 1)

    def test_separate_figures_keep_panels_together(self):
        with TemporaryDirectory() as tmp, patch.object(indexer, "ROOT", Path(tmp)):
            path = Path(tmp) / "diagrams.pdf"
            with pymupdf.open() as doc:
                page = doc.new_page()
                for x, y, color in [(30, 60, "red"), (60, 60, "blue"), (30, 150, "green")]:
                    buf = io.BytesIO()
                    Image.new("RGB", (20, 20), color).save(buf, format="PNG")
                    page.insert_image((x, y, x+20, y+20), stream=buf.getvalue())
                page.insert_text((30, 100), "Figure 1: Two panels")
                page.insert_text((30, 190), "Figure 2: Single panel")
                doc.save(path)
            docs = indexer.ArticleReader().load_data(path, "hash")
            sizes = []
            for d in docs:
                if d.metadata["type"] == "figure":
                    with Image.open(Path(tmp) / d.metadata["fichier"]) as img:
                        sizes.append(img.size)
            self.assertEqual(sizes, [(100, 40), (40, 40)])

    def test_manual_without_numbered_captions(self):
        with TemporaryDirectory() as tmp, patch.object(indexer, "ROOT", Path(tmp)):
            path = Path(tmp) / "manual.pdf"
            with pymupdf.open() as doc:
                page = doc.new_page()
                page.insert_text((40, 65), "Fan fault indicator")
                buf = io.BytesIO()
                Image.new("RGB", (100, 80), "blue").save(buf, format="PNG")
                page.insert_image((40, 75, 140, 155), stream=buf.getvalue())
                for x in (40, 160, 300):
                    page.draw_line((x, 200), (x, 260))
                for y in (200, 230, 260):
                    page.draw_line((40, y), (300, y))
                for x, y, text in [(45, 220, "Card"), (165, 220, "PSID"),
                                   (45, 250, "Storage"), (165, 250, "MT_0000000892")]:
                    page.insert_text((x, y), text)
                doc.save(path)
            docs = indexer.ArticleReader().load_data(path, "manual")
            objects = {d.metadata["type"]: d for d in docs}
            self.assertIn("Fan fault indicator", objects["figure"].text)
            self.assertRegex(objects["tableau"].text, r"Storage[^\n]*MT_0000000892")
            for kind in ("figure", "tableau"):
                self.assertTrue((Path(tmp) / objects[kind].metadata["fichier"]).is_file())

    def test_new_upload_clears_followup_history(self):
        histories = []
        def answer(q, digest, name, history, trace, visuels):
            histories.append(list(history))
            return "Réponse"
        with patch.object(question, "repondre_document", side_effect=answer), patch("figures.traiter"), patch.object(indexer, "connexion"):
            app = AppTest.from_file(str(ROOT / "scripts/app.py")).run()
            for q in ("Première question", "Et pour le modèle de base ?"):
                app.text_area[0].set_value(q)
                app.button[0].click().run()
            with pymupdf.open() as doc:
                doc.new_page().insert_text((30, 70), "Autre document")
                app.file_uploader[0].upload("autre.pdf", doc.tobytes(), "application/pdf").run()
            app.text_area[0].set_value("Nouvelle question")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(histories, [[], [("Première question", "Réponse")], []])
