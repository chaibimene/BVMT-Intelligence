from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Représente l'état de notre graphe d'agents.
    """
    question: str
    generation: str
    documents: List[str]
    iterations: int
    error: str
