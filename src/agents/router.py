import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

def get_router():
    """
    Crée un routeur qui décide du type de traitement à appliquer.
    Retourne un objet JSON : "route" et "query_type".
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un expert en routage pour un assistant de la Bourse des Valeurs Mobilières de Tunis (BVMT).
Analysez la question de l'utilisateur et générez une réponse strictement en format JSON contenant deux clés : "route" et "query_type".

1. Choix pour "route" :
   - "report" : L'utilisateur demande explicitement de générer un rapport complet ou une synthèse longue (ex: "Génère un rapport sur...", "Fais une synthèse sectorielle").
   - "vectorstore" : Toute question liée à la finance, aux entreprises, aux banques, aux marchés, aux cotations, ou aux conseils d'investissement.
   - "general" : Salutations, questions basiques totalement hors finance.

2. Choix pour "query_type" :
   - "investment" : L'utilisateur demande des recommandations, conseils, ou comparaisons orientées vers l'achat/vente (ex: "Dans quoi investir ?", "Quelle banque choisir ?", "Conseille-moi des actions").
   - "resume" : L'utilisateur veut un aperçu général ou une synthèse globale.
   - "performance" : L'utilisateur demande des chiffres, des résultats, ou qui est le meilleur/le pire.
   - "comparaison" : L'utilisateur veut comparer deux entités ou plus.
   - "tendance" : L'utilisateur demande une évolution dans le temps ou des perspectives.
   - "autre" : Si aucun de ces types ne correspond.

Votre réponse DOIT être un JSON valide, sans bloc de code Markdown, par exemple :
{{"route": "vectorstore", "query_type": "investment"}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    router = prompt | llm | JsonOutputParser()
    return router
