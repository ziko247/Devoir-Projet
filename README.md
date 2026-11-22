MaintiDoc est un assistant destiné aux techniciens de maintenance. Il exploite des
manuels, des procédures, des schémas et des photographies pour répondre à des questions techniques avec des références vérifiables. Il peut consulter l’historique des interventions et préparer un ticket de maintenance. La création effective du ticket nécessite une validation humaine.

Exemple
de demande : « Voici une photo du panneau de commande. Que signifie ce voyant orange ?  Consulte les dernières interventions et prépare un ticket si nécessaire. »

Objectif général

Construire progressivement un assistant documentaire destiné aux techniciens de maintenance, capable d’exploiter des documents et des photographies, de produire des réponses sourcées, de consulter un historique via MCP et de préparer des tickets soumis à validation humaine. Le système ne pilote aucun équipement et ne remplace pas un technicien habilité.
Découpage :
- Étape 1 — RAG Agentique Multi Modal
- Étape 2 — Fine-Tuning
- Étape 3 — Orchestration agentique et MCP
- Étape 4 — Client Web, Telegram/Discord, sécurité et déploiement

Première partie : Travail demandé
1.   Sélection du document — Choisir un fichier PDF quelconque
contenant du texte, des images et des tableaux. Un document technique dans le Domaine
de la maintenance des machines. Pour ne pas se perdre, je vous recommande, dans
un premier temps, d’utiliser le fichier pdf de l’article “Attention is
all you need.pdf” ci joint


2.   Extraction et indexation — Utiliser LlamaIndex pour analyser le PDF et extraire le texte, les
images et les tableaux. Produire des embeddings avec un modèle open source,
puis stocker les vecteurs et métadonnées dans PGVector (PostgreSQL).


3.   Traitement des images — Utiliser
CLIP pour représenter les images et/ou faciliter leur recherche. Générer des
descriptions textuelles des images et les indexer afin de les rendre
interrogeables.


4.    Question-réponse et génération — Construire un pipeline RAG avec un LLM open source, par exemple Qwen
exécuté via Ollama. Récupérer les passages pertinents et produire une réponse
fondée sur le contexte retrouvé.


5.    Interface utilisateur — Créer
une interface Streamlit permettant le chargement du PDF, la saisie des
questions et l’affichage des réponses.


6.    Orchestration agentique — Rendre
le RAG agentique avec LangChain : sélection d’outils de recherche,
interrogation de l’index, gestion du contexte et génération de la réponse.
Avancement du devoir
-------------------

[x] Étape 1. Document sélectionné : 1706.03762v7.pdf.
Étape 2. Le script indexer.py utilise un lecteur LlamaIndex adapté aux légendes
et à la mise en page de cet article. Les figures sont exportées en PNG et les
tableaux en texte et en PNG dans extraits/. Le texte des légendes sert à
indexer les figures en attendant le traitement visuel de l'étape 3.
La version PNG des tableaux conserve les symboles et les exposants.

Rejouer depuis ce dossier dans PowerShell, avec Docker et Ollama démarrés :
    docker compose -p maintidoc_devoir up -d --wait
    .\.venv\Scripts\python.exe indexer.py

Pour une nouvelle installation, créer un environnement Python 3.12, installer
requirements.txt, télécharger le modèle avec ollama pull qwen3-embedding:0.6b
et créer le fichier .env avec une ligne POSTGRES_PASSWORD=<mot de passe local>.
Le service écoute sur 127.0.0.1:55432. La base maintidoc et le volume Docker
conservent les données entre deux lancements. Réindexer le même PDF remplace
ses fragments dans une transaction.

[x] Étape 2 vérifiée sur le PDF fourni. PostgreSQL contient 40 vecteurs de 1 024 dimensions (31 fragments, 5 figures, 4 tableaux). Une recherche sur la complexité des couches retrouve le tableau 1, page 6, en premier.

Étape 3 — Traitement et recherche des figures
--------------------------------------------
Après l'étape 2, lancer depuis ce dossier :
    ollama pull qwen3.5:4b
    .\.venv\Scripts\python.exe figures.py

CLIP ViT-B/32 encode les pixels de chaque figure en 512 dimensions. Qwen3.5:4b
lit le PNG et sa légende pour rédiger une description en français.
qwen3-embedding:0.6b encode cette description en 1 024 dimensions.
La table maintidoc_figures conserve les deux vecteurs, la description, les
modèles utilisés, la page et le chemin du PNG. Une nouvelle exécution remplace
les résultats de ce PDF après le traitement de toutes ses figures.

Recherche textuelle sur les descriptions :
    .\.venv\Scripts\python.exe figures.py --question "Quel schéma montre un encodeur et un décodeur ?"

Recherche visuelle par distance cosinus entre vecteurs CLIP :
    .\.venv\Scripts\python.exe figures.py --image chemin\vers\image.png

Les deux commandes affichent les trois figures les plus proches avec leurs
pages et descriptions. Les descriptions sont générées automatiquement.

[x] Étape 3 vérifiée. La question sur l'encodeur et le décodeur retrouve la figure 1 (page 3). Une miniature de 240 × 152 pixels retrouve la figure 2 (page 4) par CLIP. Les deux arrivent en première position.
