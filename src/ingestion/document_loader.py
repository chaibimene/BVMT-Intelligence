import os
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.utils.helpers import extract_metadata_from_filename

import logging
logger = logging.getLogger(__name__)

def load_pdfs(data_dir: str, max_pages: int = 50) -> List[Document]:
    """Charge les fichiers PDF dans le répertoire donné avec gestion d'erreurs et limite de pages."""
    documents = []
    pdf_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
                
    for file_path in tqdm(pdf_files, desc="Chargement des PDFs"):
        try:
            loader = PyPDFLoader(file_path)
            # On charge le document, et on ne garde que les premières pages pour éviter l'explosion
            docs = loader.load()
            
            file_name = os.path.basename(file_path)
            metadata = extract_metadata_from_filename(file_name)
            
            if len(docs) > max_pages:
                logger.info(f"Fichier {file_name} limité à {max_pages} pages sur {len(docs)}.")
                docs = docs[:max_pages]

            for doc in docs:
                doc.metadata.update(metadata)
            
            documents.extend(docs)
        except Exception as e:
            logger.warning(f"Impossible de charger le PDF {file_path} (ignoré) : {e}")
            
    return documents

def load_cotations(data_dir: str, max_rows: int = 200) -> List[Document]:
    """Charge les fichiers de cotations avec parsing tolérant et limitation stricte de taille."""
    documents = []
    cotation_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".csv", ".txt")):
                cotation_files.append(os.path.join(root, file))
                
    for file_path in tqdm(cotation_files, desc="Chargement des cotations"):
        file_name = os.path.basename(file_path)
        metadata = extract_metadata_from_filename(file_name)
        
        try:
            # Parsing plus tolérant
            if file_name.lower().endswith(".txt"):
                df = pd.read_csv(file_path, sep=r'\s+', engine='python', on_bad_lines='skip')
            else:
                df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
                
            # Limitation très stricte pour éviter de générer un nombre gigantesque de chunks
            if len(df) > max_rows:
                logger.info(f"Fichier {file_name} limité à {max_rows} lignes sur {len(df)}.")
                df = df.head(max_rows)
                
            text_content = f"Données de cotations - Fichier: {file_name}\n\n" + df.to_string(index=False)
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            logger.warning(f"Erreur pandas pour {file_name}. Tentative avec TextLoader : {e}")
            try:
                # Fallback brutal si pandas échoue (ex: mauvaise structuration persistante)
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update(metadata)
                    lines = doc.page_content.split('\n')
                    if len(lines) > max_rows:
                        doc.page_content = '\n'.join(lines[:max_rows]) + "\n...[DONNÉES TRONQUÉES]"
                documents.extend(docs)
            except Exception as e2:
                logger.error(f"Echec total pour le chargement de {file_path} (ignoré) : {e2}")
                
    return documents

def load_all_documents(data_dir: str) -> List[Document]:
    """Charge tous les documents (PDF, CSV, TXT) depuis le répertoire."""
    docs = []
    docs.extend(load_pdfs(data_dir))
    docs.extend(load_cotations(data_dir))
    return docs
