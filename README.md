# MaintiDoc

MaintiDoc est un projet de cours autour d'un assistant pour les techniciens de maintenance. L'idée est de retrouver une information dans un manuel, une procédure, un schéma ou une photo, puis de répondre en citant la source. Plus tard, l'assistant pourra consulter l'historique des interventions via MCP et préparer un ticket. Un technicien habilité garde la main sur les équipements et valide la création du ticket.

Exemple de demande : « Voici une photo du panneau de commande. Que signifie ce voyant orange ? Consulte les dernières interventions et prépare un ticket si nécessaire. »

## Installation

Prévoir Python 3.12, Docker et Ollama. Depuis la racine du projet, dans PowerShell :

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b
~~~

Pour une nouvelle installation, copier `.env.example` vers `.env` et renseigner `POSTGRES_PASSWORD` avec un mot de passe local. Démarrer Docker et Ollama, puis lancer PostgreSQL et les scripts dans cet ordre :

~~~powershell
docker compose -p maintidoc_devoir up -d --wait
.\.venv\Scripts\python.exe scripts\indexer.py
.\.venv\Scripts\python.exe scripts\figures.py
~~~

PostgreSQL écoute sur `127.0.0.1:55432`. Docker conserve les données dans un volume. Relancer un script remplace les résultats du même PDF dans une transaction.

## Tester la recherche

~~~powershell
.\.venv\Scripts\python.exe scripts\figures.py --question "Quel schéma montre un encodeur et un décodeur ?"
.\.venv\Scripts\python.exe scripts\figures.py --image chemin\vers\image.png
~~~

Chaque commande affiche les trois figures les plus proches, avec leur page et leur description. La première cherche dans les descriptions. La seconde compare les images par distance cosinus entre leurs vecteurs CLIP.

## Organisation des fichiers

- `data/1706.03762v7.pdf` contient l'article *Attention Is All You Need*, utilisé pour les premiers essais. Le document cible du devoir est un PDF technique de maintenance avec du texte, des images et des tableaux.
- `scripts/indexer.py` extrait le contenu avec un lecteur LlamaIndex adapté à l'article. Il indexe les fragments, les tableaux et les légendes dans pgvector avec `qwen3-embedding:0.6b`.
- `scripts/figures.py` encode les PNG avec CLIP ViT-B/32 (512 dimensions). Qwen3.5:4b décrit chaque figure en français à partir de son PNG et de sa légende. `qwen3-embedding:0.6b` encode les descriptions (1 024 dimensions). La table `maintidoc_figures` conserve les deux vecteurs, la description, les modèles, la page et le chemin du PNG.
- `extraits/` reçoit les PNG et le texte des tableaux. Les PNG gardent les symboles et les exposants visibles pour vérification. `.modeles/` contient le cache CLIP. Les scripts créent ces deux dossiers au besoin.

## Avancement de la première partie

- [x] Sélection du PDF de test.
- [x] Extraction et indexation. Sur l'article fourni, PostgreSQL contient 40 vecteurs de 1 024 dimensions (31 fragments de texte, 5 légendes de figures et 4 tableaux). Une recherche sur la complexité des couches a retrouvé le tableau 1 (page 6) en premier.
- [x] Traitement des figures. La question sur l'encodeur et le décodeur retrouve la figure 1 (page 3) en premier. Une miniature de la figure 2 (240 × 152 pixels) retrouve cette figure (page 4) en premier avec CLIP.
- [ ] Questions et réponses. Récupérer les passages utiles et produire une réponse fondée sur ces passages avec un modèle open source comme Qwen via Ollama.
- [ ] Interface Streamlit. Charger un PDF, poser une question et afficher la réponse.
- [ ] Agent LangChain. Choisir les outils de recherche, interroger l'index et utiliser les résultats pour répondre.

La suite du projet prévoit le fine-tuning, l'orchestration avec MCP, puis les clients Web et Telegram ou Discord, la sécurité et le déploiement.
