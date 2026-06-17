import streamlit as st
import logging
from src.ingestion.ingest import create_vectorstore
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.agents.graph import WorkflowManager
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="BVMT Intelligence", layout="wide", page_icon="📈")

# --- Configuration et Initialisation ---
@st.cache_resource(show_spinner=False)
def init_system(model_choice):
    """Initialise le retriever et le graphe LangGraph. Cache le résultat."""
    retriever = get_hybrid_retriever(model_choice=model_choice, use_multi_query=True)
    if not retriever:
        return None
    manager = WorkflowManager(retriever)
    return manager.build_graph()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Ingestion des documents")
    model_choice = st.selectbox(
        "Modèle d'embedding :",
        ["light", "bge"],
        index=0,
        help="'light' est recommandé pour le développement local (rapide). 'bge' pour la production."
    )
    
    if st.button("🚀 Relancer l'ingestion"):
        with st.spinner(f"Ingestion en cours (modèle: {model_choice})... Cela peut prendre quelques minutes."):
            try:
                create_vectorstore(model_choice=model_choice)
                st.success("✅ Ingestion terminée avec succès !")
                # Forcer le rechargement du retriever
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur lors de l'ingestion : {str(e)}")

# --- Main Interface ---
st.title("📈 BVMT Intelligence")
st.markdown("Assistant conversationnel intelligent sur la Bourse des Valeurs Mobilières de Tunis.")

# Initialiser le workflow
with st.spinner("Chargement de l'index et des agents..."):
    workflow = init_system(model_choice)

if not workflow:
    st.warning("⚠️ L'index n'est pas encore disponible ou n'a pas pu être chargé. Veuillez (re)lancer l'ingestion depuis le menu de gauche.")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources consultées (Top 5)"):
                for i, doc in enumerate(msg["sources"]):
                    import os
                    source_path = doc.metadata.get('source', 'Inconnue')
                    filename = os.path.basename(source_path) if source_path != 'Inconnue' else 'Inconnue'
                    st.markdown(f"**📄 {filename}**")
                    clean_content = doc.page_content.replace('\n', ' ').strip()
                    st.caption(f'"{clean_content[:250]}..."')
                    st.divider()

# Input utilisateur
if prompt := st.chat_input("Posez votre question sur la bourse tunisienne..."):
    # Ajouter la question à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Analyse et génération de la réponse..."):
            initial_state = {"question": prompt}
            
            try:
                # Exécution du graphe
                final_state = workflow.invoke(initial_state)
                
                response_text = final_state.get("generation", "Désolé, je n'ai pas pu générer de réponse.")
                documents = final_state.get("documents", [])
                
                st.markdown(response_text)
                
                # Affichage des sources
                if documents:
                    with st.expander("📚 Sources consultées (Top 5)"):
                        for i, doc in enumerate(documents):
                            import os
                            source_path = doc.metadata.get('source', 'Inconnue')
                            filename = os.path.basename(source_path) if source_path != 'Inconnue' else 'Inconnue'
                            
                            st.markdown(f"**📄 {filename}**")
                            # Nettoyer un peu l'extrait
                            clean_content = doc.page_content.replace('\n', ' ').strip()
                            st.caption(f'"{clean_content[:250]}..."')
                            st.divider()
                            
                # Sauvegarder dans l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": documents
                })
                
            except Exception as e:
                st.error(f"Une erreur s'est produite lors du traitement : {str(e)}")

