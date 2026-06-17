# BVMT Intelligence

BVMT Intelligence est une application RAG (Retrieval-Augmented Generation) avancée conçue pour analyser, interroger et extraire des informations à partir de documents PDF et de données de cotations (fichiers CSV/TXT).

## Structure du Projet

- `data/raw/` : Dossier contenant les documents bruts (PDF, CSV).
- `vectorstore/` : Dossier où l'index FAISS est sauvegardé après ingestion.
- `src/ingestion/` : Pipeline de traitement et de vectorisation des documents.
- `app.py` : Application Streamlit (interface utilisateur temporaire).

## Prérequis

- Python 3.9+
- Pip

## Étapes d'installation

1. **Cloner ou créer le dossier du projet.**
2. **Créer un environnement virtuel (recommandé) :**
   ```bash
   python -m venv venv
   # Sur Windows :
   venv\Scripts\activate
   # Sur Linux/Mac :
   source venv/bin/activate
   ```
3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configuration :**
   Copiez le fichier `.env.example` en `.env` et ajoutez vos clés API (par exemple, `GROQ_API_KEY`).
   ```bash
   # Sur Windows :
   copy .env.example .env
   # Sur Linux/Mac :
   cp .env.example .env
   ```

## Commande pour lancer l'ingestion

Assurez-vous que vos fichiers PDF et CSV sont dans le dossier `data/raw/`, puis exécutez le script d'ingestion à la racine du projet :

```bash
python -m src.ingestion.ingest
```

Ce script va :
- Charger tous les PDF (via PyPDF) et fichiers de cotations (via Pandas).
- Découper les textes en morceaux (chunks de 900 de taille avec 200 de chevauchement).
- Créer un index FAISS avec le modèle d'embeddings BGE-M3.
- Sauvegarder l'index dans le dossier `vectorstore/`.

## Comment lancer l'application

Une fois l'ingestion terminée, vous pouvez démarrer l'interface utilisateur Streamlit :

```bash
streamlit run app.py
```