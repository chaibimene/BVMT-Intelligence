from typing import List, TypedDict, Dict, Any

class GraphState(TypedDict):
    """
    Représente l'état de notre graphe d'agents.
    """
    question: str
    generation: str
    documents: List[str]
    iterations: int
    error: str
    query_type: str
    chat_history: List[Dict[str, Any]]
