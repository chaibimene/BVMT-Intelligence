import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

def get_router():
    """
    Crée un routeur qui décide du type de traitement à appliquer.
    Retourne un objet JSON contenant :
    - 'route' : 'investment_advice', 'vectorstore' ou 'general'
    - 'query_type' : 'resume', 'performance', 'comparaison', 'tendance', ou 'autre'
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un expert en routage pour un assistant de la Bourse des Valeurs Mobilières de Tunis (BVMT).
Analysez la question de l'utilisateur et générez une réponse strictement en format JSON contenant deux clés : "route" et "query_type".

1. Choix pour "route" :
   - "investment_advice" : Conseils pour investir, quelle action acheter, recommandation d'achat/vente.
   - "vectorstore" : Informations factuelles (performances, entreprises, bilans, cotations passées).
   - "general" : Salutations, questions basiques hors finance.

2. Choix pour "query_type" :
   - "resume" : L'utilisateur veut un aperçu général ou une synthèse globale.
   - "performance" : L'utilisateur demande des chiffres, des résultats, ou qui est le meilleur/le pire.
   - "comparaison" : L'utilisateur veut comparer deux entités ou plus.
   - "tendance" : L'utilisateur demande une évolution dans le temps ou des perspectives.
   - "autre" : Si aucun de ces types ne correspond.

Votre réponse DOIT être un JSON valide, sans bloc de code Markdown, par exemple :
{{"route": "vectorstore", "query_type": "performance"}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    router = prompt | llm | JsonOutputParser()
    return router
