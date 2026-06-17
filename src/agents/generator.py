import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def get_generator():
    """
    Crée le générateur (RAG LLM) qui produit une réponse très structurée et concise basée sur les documents fournis.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un analyste financier expert de la Bourse des Valeurs Mobilières de Tunis (BVMT).
Votre rôle est de répondre de manière concise, directe et très professionnelle à la question de l'utilisateur EN FRANÇAIS, en utilisant UNIQUEMENT le contexte fourni.

RÈGLES STRICTES :
1. Structurez votre réponse ainsi :
   - Introduction : Une phrase d'accroche qui répond directement à la question.
   - Points clés : Utilisez des puces (bullet points) pour synthétiser l'information sans paraphraser inutilement. Ne collez pas de longs extraits bruts.
   - Conclusion courte : Une courte phrase résumant l'idée générale.
2. Si la réponse ne se trouve pas clairement dans le contexte, dites poliment que les informations fournies ne permettent pas de répondre précisément, et donnez uniquement ce que vous savez. N'inventez RIEN.

Contexte:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question : {question}"),
    ])

    generator = prompt | llm | StrOutputParser()
    return generator

def get_general_generator():
    """
    Générateur pour les questions générales ne nécessitant pas de contexte.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un assistant IA officiel de la BVMT (Bourse des Valeurs Mobilières de Tunis). 
Répondez de manière professionnelle, très courte et directe (en français). 
Si l'utilisateur pose une question hors finance, rappelez-lui gentiment que votre domaine d'expertise se limite au marché financier et aux entreprises tunisiennes cotées."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()

def get_investment_generator():
    """
    Générateur spécialisé pour repousser les demandes de conseils d'investissement directs.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un assistant IA de la BVMT. L'utilisateur vous demande un conseil en investissement ou une recommandation d'achat/vente.
    
Votre rôle :
- Expliquez clairement et professionnellement que vous êtes une IA d'information et que vous n'êtes pas habilité à donner des conseils en investissement personnalisés.
- Donnez des conseils très généraux sur l'investissement (ex: diversifier son portefeuille, consulter les états financiers, contacter un intermédiaire en bourse agréé).
- Soyez concis, rassurant, mais ferme sur le fait que la décision finale appartient à l'investisseur.
- Rédigez en français."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
