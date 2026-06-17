import logging
from langgraph.graph import END, StateGraph
from src.agents.state import GraphState
from src.agents.router import get_router
from src.agents.generator import get_generator, get_general_generator, get_investment_generator, get_report_generator, format_history
from src.agents.critic import get_hallucination_grader, get_answer_relevance_grader

logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self, retriever):
        self.retriever = retriever
        self.router = get_router()
        self.generator = get_generator()
        self.general_generator = get_general_generator()
        self.investment_generator = get_investment_generator()
        self.report_generator = get_report_generator()
        self.hallucination_grader = get_hallucination_grader()
        self.answer_relevance_grader = get_answer_relevance_grader()

    def analyze_intent(self, state: GraphState):
        logger.info("--- ANALYSE DE L'INTENTION ---")
        question = state["question"]
        try:
            source = self.router.invoke({"question": question})
            route = source.get("route", "vectorstore")
            query_type = source.get("query_type", "autre")
        except Exception as e:
            logger.error(f"Erreur de routage: {e}")
            route = "vectorstore"
            query_type = "autre"
            
        logger.info(f"-> Décision: Route={route}, Type={query_type}")
        return {"query_type": query_type, "error": route}
        
    def route_question(self, state: GraphState):
        route = state.get("error", "vectorstore")
        
        if "report" in route.lower():
            logger.info("-> Route vers Rapport complet (avec RAG large)")
            return "retrieve_report"
        elif "vectorstore" in route.lower():
            logger.info("-> Route vers RAG (Vectorstore)")
            return "retrieve"
        else:
            logger.info("-> Route vers LLM Général")
            return "general_llm"

    def retrieve(self, state: GraphState):
        logger.info("--- RETRIEVAL ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        documents = documents[:5]
        logger.info(f"-> {len(documents)} documents récupérés (standard).")
        return {"documents": documents, "question": question, "iterations": state.get("iterations", 0)}

    def retrieve_report(self, state: GraphState):
        logger.info("--- RETRIEVAL REPORT ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        documents = documents[:10]  # Plus de contexte pour un rapport complet
        logger.info(f"-> {len(documents)} documents récupérés (large).")
        return {"documents": documents, "question": question, "iterations": state.get("iterations", 0)}

    def generate(self, state: GraphState):
        logger.info("--- GENERATION RAG ---")
        question = state["question"]
        documents = state.get("documents", [])
        chat_history = format_history(state.get("chat_history", []))
        query_type = state.get("query_type", "autre")
        context = "\n\n".join([doc.page_content for doc in documents])

        # Aiguillage vers le générateur spécialisé selon le type
        if query_type == "investment":
            logger.info("-> Mode Investment : utilisation du générateur d'analyse sectorielle")
            generation = self.investment_generator.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history
            })
        else:
            enriched_question = f"[Type: {query_type}] {question}"
            generation = self.generator.invoke({
                "context": context,
                "question": enriched_question,
                "chat_history": chat_history
            })

        iterations = state.get("iterations", 0) + 1
        return {"generation": generation, "iterations": iterations}

    def generate_general(self, state: GraphState):
        logger.info("--- GENERATION GENERALE ---")
        question = state["question"]
        chat_history = format_history(state.get("chat_history", []))
        generation = self.general_generator.invoke({"question": question, "chat_history": chat_history})
        return {"generation": generation, "documents": []}

    def generate_investment(self, state: GraphState):
        logger.info("--- GENERATION INVESTMENT ---")
        question = state["question"]
        documents = state.get("documents", [])
        chat_history = format_history(state.get("chat_history", []))
        context = "\n\n".join([doc.page_content for doc in documents])
        generation = self.investment_generator.invoke({"context": context, "question": question, "chat_history": chat_history})
        iterations = state.get("iterations", 0) + 1
        return {"generation": generation, "iterations": iterations}

    def generate_report(self, state: GraphState):
        logger.info("--- GENERATION REPORT ---")
        question = state["question"]
        documents = state.get("documents", [])
        context = "\n\n".join([doc.page_content for doc in documents])
        generation = self.report_generator.invoke({"context": context, "question": question})
        # On ne fait pas de retry sur le rapport pour éviter des boucles longues
        return {"generation": generation, "iterations": 10}

    def grade_generation(self, state: GraphState):
        logger.info("--- CRITIC EVALUATION ---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        iterations = state.get("iterations", 0)
        
        if iterations >= 3:
            logger.warning("-> Max itérations atteintes, fin.")
            return "useful"
            
        context = "\n\n".join([doc.page_content for doc in documents])
        
        score_hallucination = self.hallucination_grader.invoke({"documents": context, "generation": generation})
        if "oui" in score_hallucination.lower():
            logger.info("-> Pas d'hallucination détectée.")
            score_relevance = self.answer_relevance_grader.invoke({"question": question, "generation": generation})
            if "oui" in score_relevance.lower():
                logger.info("-> Réponse pertinente.")
                return "useful"
            else:
                logger.warning("-> Réponse non pertinente ou trop longue.")
                return "not_useful"
        else:
            logger.warning("-> Hallucination détectée, régénération nécessaire.")
            return "not_supported"

    def build_graph(self):
        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node("analyze", self.analyze_intent)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("retrieve_report", self.retrieve_report)
        workflow.add_node("generate", self.generate)
        workflow.add_node("general_llm", self.generate_general)
        workflow.add_node("report_generation", self.generate_report)

        workflow.set_entry_point("analyze")

        # Edges
        workflow.add_conditional_edges(
            "analyze",
            self.route_question,
            {
                "retrieve": "retrieve",
                "general_llm": "general_llm",
                "retrieve_report": "retrieve_report"
            }
        )

        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("retrieve_report", "report_generation")
        workflow.add_edge("general_llm", END)
        workflow.add_edge("report_generation", END)

        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "useful": END,
                "not_useful": "generate",
                "not_supported": "generate"
            }
        )

        return workflow.compile()
