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

## Interroger un PDF dans le navigateur

Avec PostgreSQL et Ollama démarrés, lancer depuis la racine du projet :

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run scripts/app.py
~~~

Ouvrir http://127.0.0.1:8501. L'article du devoir est disponible dès l'ouverture. Pour changer de document, charger un PDF de 20 Mo maximum, saisir une question et cliquer sur **Rechercher la réponse**. La première question déclenche l'indexation si nécessaire. Une réponse sourcée indique le fichier et la page. L'index reste dans PostgreSQL et les extraits dans `extraits/`.

Le lecteur utilise le texte sélectionnable du PDF. La reconnaissance des figures et des tableaux est adaptée à la mise en page de l'article. Pour enrichir un autre PDF indexé avec les descriptions visuelles, utiliser `scripts/figures.py --pdf chemin\vers\document.pdf`.

## Poser une question au PDF

Depuis la racine, avec Ollama et PostgreSQL démarrés et les étapes 2 et 3 exécutées :

~~~powershell
.\.venv\Scripts\python.exe scripts\question.py "Quels scores BLEU le Transformer big obtient-il en anglais-allemand et en anglais-français ?"
~~~

La réponse attendue est 28,4 en anglais-allemand et 41,8 en anglais-français (tableau 2, page 8). Qwen choisit avec LangChain où chercher : dans le texte, les tableaux ou les figures. Chaque outil encode sa requête avec `qwen3-embedding:0.6b` et récupère jusqu'à trois extraits du PDF sélectionné dans pgvector. Qwen rédige ensuite sa réponse à partir de ces extraits. Les pages affichées proviennent des métadonnées des sources citées. L'option `--pdf` sélectionne un autre PDF déjà indexé.

Dans Streamlit, poser « Quel score BLEU obtient le Transformer big en anglais-allemand ? », puis « Et pour le modèle de base sur la même traduction ? ». Les réponses attendues sont 28,4 et 27,3, toutes deux dans le tableau 2, page 8. L'interface affiche les outils appelés et garde les deux derniers échanges pour comprendre les relances. Changer de document efface cet historique.

Sur une question portant sur le tableau 2, les modèles locaux 4B et 9B ont donné les deux bons scores. Ollama a mesuré 116 jetons/s pour le 4B et 69 pour le 9B pendant la génération. Le 4B occupe aussi moins de place sur disque (3,32 Go contre 6,55 Go).

## Vérifier les corrections

Pour comparer les résultats, essayer « Compare les tableaux 2 et 3 pour le Transformer de base en anglais-allemand ». Le tableau 2 donne 27,3 sur newstest2014 et le tableau 3 donne 25,8 sur newstest2013. Pour une question sur une figure, le modèle reçoit aussi le PNG avec sa légende. Après une mise à jour de l'extraction, relancer l'indexation puis le traitement des figures pour actualiser les fichiers déjà indexés.

Les tests de régression couvrent le routage des tableaux, les réponses mal formées, les PNG envoyés au modèle, le découpage des figures et le changement de PDF après une relance. Ils fonctionnent sans PostgreSQL ni Ollama :

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

## Organisation des fichiers

- `data/1706.03762v7.pdf` contient l'article *Attention Is All You Need*, utilisé pour les premiers essais. Le document cible du devoir est un PDF technique de maintenance avec du texte, des images et des tableaux.
- `scripts/indexer.py` extrait le contenu avec un lecteur LlamaIndex adapté à l'article. Il indexe les fragments, les tableaux et les légendes dans pgvector avec `qwen3-embedding:0.6b`.
- `scripts/figures.py` encode les PNG avec CLIP ViT-B/32 (512 dimensions). Qwen3.5:4b décrit chaque figure en français à partir de son PNG et de sa légende. `qwen3-embedding:0.6b` encode les descriptions (1 024 dimensions). La table `maintidoc_figures` conserve les deux vecteurs, la description, les modèles, la page et le chemin du PNG.
- `extraits/` reçoit les PNG et le texte des tableaux. Les PNG gardent les symboles et les exposants visibles pour vérification. `.modeles/` contient le cache CLIP. Les scripts créent ces deux dossiers au besoin.

## Avancement de la première partie

- [x] Sélection du PDF de test.
- [x] Extraction et indexation. Sur l'article fourni, PostgreSQL contient 40 vecteurs de 1 024 dimensions (31 fragments de texte, 5 légendes de figures et 4 tableaux). Une recherche sur la complexité des couches a retrouvé le tableau 1 (page 6) en premier.
- [x] Traitement des figures. La question sur l'encodeur et le décodeur retrouve la figure 1 (page 3) en premier. Une miniature de la figure 2 (240 × 152 pixels) retrouve cette figure (page 4) en premier avec CLIP.
- [x] Questions et réponses avec Qwen via Ollama. Tests sur le tableau 2 (page 8), la figure 1 (page 3) et une question hors sujet sur un pneu de tracteur.
- [x] Interface Streamlit. Chargement et question testés dans le navigateur sur l'article et sur une fiche PDF d'une page. La réponse cite la page du document sélectionné.
- [x] Agent LangChain. Sélection des outils texte, tableau et figure vérifiée sur l'article, avec une relance comparative et une question hors sujet.

La suite du projet prévoit le fine-tuning, l'orchestration avec MCP, puis les clients Web et Telegram ou Discord, la sécurité et le déploiement.
