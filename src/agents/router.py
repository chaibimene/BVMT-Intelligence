import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def get_router():
    """
    Crée un routeur qui décide si la question de l'utilisateur nécessite :
    - 'vectorstore' : si la question concerne des faits, des données, des sociétés cotées de la BVMT, ou l'historique financier.
    - 'investment_advice' : si la question demande explicitement dans quoi investir, quelle action acheter, ou des recommandations financières personnalisées.
    - 'general' : si c'est une question générale (salutations, questions basiques hors finance).
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un expert en routage pour un assistant de la Bourse des Valeurs Mobilières de Tunis (BVMT).
Analysez la question de l'utilisateur et classez-la strictement dans l'une des 3 catégories suivantes :

1. 'investment_advice' : L'utilisateur demande des conseils pour investir, demande quelle action acheter, ou cherche une recommandation financière directe (ex: "Dans quoi dois-je investir ?", "Est-ce le moment d'acheter la BT ?").
2. 'vectorstore' : L'utilisateur cherche des informations factuelles sur la bourse, des entreprises, des bilans, ou des cotations passées (ex: "Quel est le résultat de la SFBT en 2022 ?", "Comment fonctionne la BVMT ?").
3. 'general' : L'utilisateur pose une question basique, dit bonjour, ou pose une question totalement hors sujet de la finance.

Répondez UNIQUEMENT par le nom de la catégorie ('investment_advice', 'vectorstore', ou 'general'), sans aucun autre mot ou ponctuation.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    router = prompt | llm | StrOutputParser()
    return router
