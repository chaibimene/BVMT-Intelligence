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
Votre rôle est d'analyser le contexte fourni et de donner une réponse CONCRÈTE et UTILE.

RÈGLES STRICTES ET ABSOLUES :
1. SOYEZ ASSERTIF : Interdiction formelle de dire "Les informations ne permettent pas de répondre" ou "Je n'ai pas les données". Si l'année exacte manque, utilisez les informations des années proches ou les données disponibles pour les entreprises évoquées (ex: BIAT, Attijari, Amen Bank, etc.).
2. DONNEZ DE LA VALEUR : S'il y a des noms d'entreprises, des chiffres, ou des tendances dans le contexte qui se rapprochent de la question, citez-les explicitement.
3. FORMAT OBLIGATOIRE (Vous devez respecter exactement cette structure Markdown) :

**[Une phrase directe de réponse]**

- [Point clé concret 1 avec des faits]
- [Point clé concret 2 avec des faits]
- [Point clé concret 3 (si nécessaire)]

**[Une phrase courte de conclusion ou de tendance globale]**

4. Ne mentionnez pas "Voici les sources :" car l'interface s'en occupe.

Contexte :
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
Répondez de manière professionnelle, très courte et directe (en français). Limite: 50 mots.
Si l'utilisateur pose une question hors finance, rappelez-lui gentiment que votre domaine d'expertise se limite au marché financier et aux entreprises tunisiennes cotées."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()

def get_investment_generator():
    """
    Générateur spécialisé pour les demandes de conseils d'investissement directs.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un assistant IA de la BVMT. L'utilisateur demande un conseil en investissement.
    
RÈGLES STRICTES :
1. Mise en garde (Short-circuit) : Commencez par UNE SEULE phrase courte (ex: "Je ne donne pas de conseils d'investissement, mais voici les faits.").
2. SOYEZ ASSERTIF : Interdiction de dire "je n'ai pas de données". Utilisez intelligemment le CONTEXTE pour extraire des noms de sociétés, des performances ou des tendances, même si la date exacte n'est pas trouvée.
3. FORMAT OBLIGATOIRE (Respectez exactement ce Markdown) :

**[Mise en garde courte]**

- [Point clé concret 1 basé sur le contexte]
- [Point clé concret 2 basé sur le contexte]

**[Conclusion ou Tendance globale]**

Contexte (utilisez-le pour répondre factuellement) :
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question : {question}"),
    ])

    return prompt | llm | StrOutputParser()
