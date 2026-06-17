import os
import re

def clean_text(text):
    """Nettoie le texte extrait des documents."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_metadata_from_filename(filename):
    """Extrait des métadonnées basiques à partir du nom du fichier."""
    metadata = {
        "nom_fichier": filename,
        "source": "BVMT",
    }
    
    # Extraction de l'année si présente (4 chiffres entre 1900 et 2099)
    year_match = re.search(r'(19|20)\d{2}', filename)
    if year_match:
        metadata["annee"] = year_match.group(0)
    else:
        metadata["annee"] = "Inconnue"
        
    # Détermination du type de document
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        metadata["type_document"] = "pdf"
    elif lower_name.endswith(".csv") or lower_name.endswith(".txt"):
        metadata["type_document"] = "cotations"
    else:
        metadata["type_document"] = "autre"
        
    return metadata
