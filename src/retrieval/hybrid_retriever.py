import logging
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

def get_hybrid_retriever(model_choice="bge", use_multi_query=True):
    """
    Crée un retriever hybride combinant FAISS (recherche vectorielle) et BM25 (recherche lexicale).
    """
    logger.info("Chargement du VectorStore FAISS...")
    from src.ingestion.ingest import load_vectorstore
    vectorstore = load_vectorstore(model_choice)
    
    if not vectorstore:
        logger.error("Impossible de créer le retriever : VectorStore introuvable.")
        return None

    # Extraction des documents pour construire l'index BM25
    logger.info("Extraction des documents depuis le docstore pour BM25...")
    documents = list(vectorstore.docstore._dict.values())
    
    if not documents:
        logger.error("Aucun document trouvé dans le docstore.")
        return None

    logger.info("Initialisation de BM25Retriever...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Combine les deux avec un poids égal
    logger.info("Création de l'EnsembleRetriever (FAISS + BM25)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    if use_multi_query:
        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever
            from langchain_groq import ChatGroq
            logger.info("Enrobage avec MultiQueryRetriever...")
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=llm
            )
            return multi_query_retriever
        except Exception as e:
            logger.warning(f"MultiQueryRetriever non disponible: {e}. Utilisation de l'EnsembleRetriever simple.")
            return ensemble_retriever
    
    return ensemble_retriever