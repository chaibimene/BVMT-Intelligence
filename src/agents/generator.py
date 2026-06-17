import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def format_history(history):
    if not history:
        return "Aucun historique."
    formatted = []
    # Prendre les 3-4 derniers messages
    recent = history[-4:] if len(history) >= 4 else history
    for msg in recent:
        role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
        formatted.append(f"{role}: {msg.get('content')}")
    return "\n".join(formatted)

def get_generator():
    """
    Crée le générateur (RAG LLM) qui produit une réponse très structurée et concise basée sur les documents fournis.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un analyste financier expert de la Bourse des Valeurs Mobilières de Tunis (BVMT).
Votre rôle est d'analyser le contexte fourni et de donner une réponse CONCRÈTE et UTILE.

Historique récent :
{chat_history}

RÈGLES STRICTES ET ABSOLUES :
1. SOYEZ ASSERTIF : Interdiction de dire "Les informations ne permettent pas de répondre" ou "Je n'ai pas les données". Si l'année exacte manque, utilisez les données des années proches.
2. Si le type de question est [investment], appliquez le FORMAT INVESTISSEMENT ci-dessous :

   **Analyse et recommandations sectorielles**
   - [Société 1] : [Surpondérer / Neutre / Sous-pondérer] – [Justification courte basée sur le contexte]
   - [Société 2] : [Surpondérer / Neutre / Sous-pondérer] – [Justification courte]
   (ajoutez autant de lignes que le contexte le permet)

   **Synthèse**
   [Une phrase de tendance globale]

   > ⚠️ *Ceci est une analyse générale basée sur les rapports BVMT. Ce n'est pas un conseil financier personnalisé. Faites vos propres recherches et consultez un professionnel agréé.*

3. Pour tous les autres types de questions, utilisez ce FORMAT STANDARD :

   **[Une phrase directe de réponse]**
   - [Point clé concret 1 avec faits]
   - [Point clé concret 2 avec faits]
   **[Conclusion courte]**

4. Ne mentionnez jamais "Voici les sources :".

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
Prenez en compte l'historique si nécessaire :
{chat_history}

Si l'utilisateur pose une question hors finance, rappelez-lui gentiment que votre domaine d'expertise se limite au marché financier et aux entreprises tunisiennes cotées."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()

def get_investment_generator():
    """
    Générateur d'analyse sectorielle et de recommandations d'investissement
    basées sur les données réelles des rapports BVMT.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un analyste financier senior spécialisé dans les marchés émergents et la BVMT (Bourse des Valeurs Mobilières de Tunis).
L'utilisateur vous demande une analyse et des recommandations d'investissement.

Historique récent :
{chat_history}

RÈGLES ABSOLUES :
1. ANALYSEZ le contexte fourni pour extraire des données sur les sociétés : PNB, résultats nets, croissance, rendement, liquidité, valorisation.
2. DONNEZ des recommandations claires au format suivant pour chaque société ou secteur pertinent présent dans le contexte :
   - **Surpondérer** : Position favorable (croissance solide, valorisation attractive, bons fondamentaux)
   - **Neutre** : Position mitigée (potentiel mais risques équilibrés)
   - **Sous-pondérer** : Position défavorable (faible croissance, valorisation élevée, risques importants)
3. Si des données spécifiques manquent, utilisez les tendances disponibles pour motiver votre recommandation. Ne refusez jamais de répondre.
4. FORMAT OBLIGATOIRE :

**Analyse et recommandations — [Secteur ou thème demandé]**

| Société | Recommandation | Points forts | Risques |
|---------|---------------|-------------|--------|
| [Nom]   | Surpondérer ✅ | [Fait concret] | [Risque] |
| [Nom]   | Neutre ⚖️     | [Fait concret] | [Risque] |
| [Nom]   | Sous-pondérer ❌ | [Fait concret] | [Risque] |

**Synthèse sectorielle**
[Tendance générale et perspective du secteur en 2-3 phrases]

> ⚠️ *Cette analyse est basée sur les rapports et données BVMT disponibles. Elle ne constitue pas un conseil financier personnalisé. Faites vos propres recherches et consultez un professionnel agréé avant toute décision d'investissement.*

Contexte extrait des rapports BVMT :
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question d'investissement : {question}"),
    ])

    return prompt | llm | StrOutputParser()

def get_report_generator():
    """
    Générateur pour produire de longs rapports détaillés.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = """Vous êtes un analyste financier senior de la BVMT. L'utilisateur a demandé la génération d'un rapport complet ou d'une synthèse détaillée.

RÈGLES POUR LE RAPPORT :
1. Le rapport doit être long, détaillé et professionnel.
2. Structure obligatoire (utilisez le Markdown de manière élégante) :
   # [Titre du rapport clair et accrocheur]
   
   ## Résumé Exécutif
   [Un paragraphe synthétique donnant la vue d'ensemble]
   
   ## Points Clés et Analyse
   ### [Secteur ou Entreprise 1]
   - [Analyse détaillée avec chiffres]
   ### [Secteur ou Entreprise 2]
   - [Analyse détaillée avec chiffres]
   (Ajoutez autant de sous-sections que le contexte le permet)
   
   ## Conclusion et Perspectives
   [Résumé des tendances futures ou bilan global]

3. Vous DEVEZ utiliser toutes les données chiffrées pertinentes présentes dans le contexte. N'hésitez pas à étoffer l'analyse avec vos connaissances sur la macroéconomie tunisienne si nécessaire, tout en vous appuyant majoritairement sur le contexte.

Contexte extrait des documents :
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Demande de rapport : {question}"),
    ])

    return prompt | llm | StrOutputParser()
