import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def get_hallucination_grader():
    """
    Évalue si la génération est fondée sur les documents fournis (détection d'hallucination).
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    system_prompt = """Vous êtes un évaluateur strict. Votre rôle est d'évaluer si la RÉPONSE fournie est entièrement fondée sur les DOCUMENTS.
    
Répondez uniquement par 'oui' ou 'non'.
- 'oui' : la réponse est supportée par les documents ou la réponse indique qu'elle ne connait pas l'information.
- 'non' : la réponse contient des informations non présentes dans les documents (hallucination)."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "DOCUMENTS:\n{documents}\n\nRÉPONSE:\n{generation}"),
    ])
    
    return prompt | llm | StrOutputParser()

def get_answer_relevance_grader():
    """
    Évalue si la génération répond effectivement à la question de l'utilisateur.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    system_prompt = """Vous êtes un évaluateur strict. Votre rôle est d'évaluer si la RÉPONSE adresse utilement la QUESTION posée.
    
Répondez uniquement par 'oui' ou 'non'.
- 'oui' : la réponse adresse la question.
- 'non' : la réponse est hors sujet ou ne répond pas du tout à la question."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "QUESTION:\n{question}\n\nRÉPONSE:\n{generation}"),
    ])
    
    return prompt | llm | StrOutputParser()
