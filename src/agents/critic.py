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
    
    system_prompt = """Vous êtes un évaluateur. Votre rôle est d'évaluer si la RÉPONSE est globalement cohérente avec les DOCUMENTS fournis.

Répondez uniquement par 'oui' ou 'non'.
- 'oui' : la réponse s'appuie sur le contenu des documents, même si elle fait une synthèse, formule des recommandations ou déduit une tendance. Une reformulation intelligente est acceptable.
- 'non' : la réponse contient des faits SPÉCIFIQUES clairement absents des documents (noms, chiffres, dates entièrement inventés).

IMPORTANT : Ne rejetez PAS une réponse simplement parce qu'elle synthétise ou reformule les informations. Seule une fabrication complète de faits est une hallucination."""
    
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
- 'oui' : la réponse contient des faits/noms concrets extraits du contexte. Tous ces formats sont acceptés : puces Markdown, tableau avec recommandations (Surpondérer/Neutre/Sous-pondérer), analyse structurée en gras.
- 'non' : la réponse dit "je n'ai pas les données" alors qu'il y a des informations disponibles, ou la réponse n'est pas du tout liée à la question."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "QUESTION:\n{question}\n\nRÉPONSE:\n{generation}"),
    ])
    
    return prompt | llm | StrOutputParser()
