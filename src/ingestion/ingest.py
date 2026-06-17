import os
import argparse
import logging
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.document_loader import load_all_documents

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constantes
DATA_DIR = os.path.join("data", "raw")
VECTORSTORE_DIR = "vectorstore"

# Définition des modèles d'embeddings
MODELS = {
    "bge": "BAAI/bge-m3",
    "light": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

def init_environment():
    """Charge les variables d'environnement."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY n'est pas défini dans l'environnement.")

def get_embeddings(model_choice="bge"):
    """Initialise le modèle d'embeddings."""
    model_name = MODELS.get(model_choice, MODELS["bge"])
    logger.info(f"Initialisation du modèle d'embeddings : {model_name}")
    
    # Configuration optimisée pour éviter les crashes de RAM
    # Le batch_size réduit limite la consommation mémoire de HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 8  # Optimisation mémoire: batch plus petit pour ne pas crasher
        }
    )
    return embeddings

def create_vectorstore(model_choice="bge"):
    """Pipeline complet d'ingestion des documents optimisé."""
    init_environment()
    
    logger.info(f"Recherche de documents dans {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        logger.error(f"Le répertoire {DATA_DIR} n'existe pas.")
        os.makedirs(DATA_DIR, exist_ok=True)
        return

    documents = load_all_documents(DATA_DIR)
    
    if not documents:
        logger.warning("Aucun document trouvé à ingérer.")
        return
        
    logger.info(f"{len(documents)} pages/documents chargés au total.")

    # Chunking avec tailles réduites pour accélérer et limiter la mémoire
    logger.info("Début du découpage (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    # Réduction stricte de la taille (au lieu de 800-1000)
        chunk_overlap=100, # Chevauchement de 100
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents découpés en {len(chunks)} chunks au total.")

    if not chunks:
        logger.warning("Aucun chunk généré.")
        return

    logger.info(f"Génération des embeddings et création de l'index FAISS (mode {model_choice})...")
    embeddings = get_embeddings(model_choice)
    
    # Ajout au Vectorstore par lots (batches)
    # Permet d'afficher la progression via tqdm et de lisser l'usage de la mémoire
    batch_size_faiss = 1000  # Ajout par lots plus petits
    vectorstore = None
    
    for i in tqdm(range(0, len(chunks), batch_size_faiss), desc="Vectorisation FAISS"):
        batch_chunks = chunks[i:i + batch_size_faiss]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch_chunks, embeddings)
        else:
            vectorstore.add_documents(batch_chunks)
    
    # Sauvegarde
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    logger.info(f"Sauvegarde du vectorstore dans {VECTORSTORE_DIR}...")
    vectorstore.save_local(VECTORSTORE_DIR)
    
    logger.info("Pipeline d'ingestion terminé avec succès !")
    return vectorstore

def load_vectorstore(model_choice="bge"):
    """Charge l'index FAISS s'il existe."""
    embeddings = get_embeddings(model_choice)
    if os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        logger.info(f"Chargement du vectorstore depuis {VECTORSTORE_DIR}...")
        return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.error("Aucun vectorstore trouvé. Exécutez le script d'ingestion d'abord.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion BVMT Intelligence")
    parser.add_argument(
        "--embeddings", 
        choices=["bge", "light"], 
        default="bge",
        help="Choix du modèle : 'bge' pour prod (précis mais lourd), 'light' pour dev (rapide et léger)."
    )
    args = parser.parse_args()
    
    create_vectorstore(model_choice=args.embeddings)

if __name__ == "__main__":
    main()
