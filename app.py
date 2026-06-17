import os
import streamlit as st
import logging
from src.ingestion.ingest import create_vectorstore
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.agents.graph import WorkflowManager
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BVMT Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { color: #a8b2d8; margin: 0.4rem 0 0 0; font-size: 0.95rem; }

    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }

    [data-testid="stSidebar"] { background: #0f0f1a; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e94560; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color: #a8b2d8; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(233,69,96,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ────────────────────────────────────────────────────
def render_sources(documents):
    """Affiche les sources dans un expander Streamlit propre."""
    if not documents:
        return
    with st.expander("📚 Sources principales"):
        for doc in documents:
            source_path = doc.metadata.get('source', 'Inconnue')
            filename = os.path.basename(source_path) if source_path != 'Inconnue' else 'Inconnue'
            clean_content = doc.page_content.replace('\n', ' ').strip()
            st.markdown(f"**📄 {filename}**")
            st.markdown(f"> *{clean_content[:220]}...*")
            st.divider()


# ─── Init Système (avec cache) ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_system(model_choice):
    """Initialise le retriever et le graphe LangGraph. Mis en cache."""
    retriever = get_hybrid_retriever(model_choice=model_choice, use_multi_query=True)
    if not retriever:
        return None
    manager = WorkflowManager(retriever)
    return manager.build_graph()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 BVMT Intelligence")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox(
        "Modèle d'embedding :",
        ["light", "bge"],
        index=0,
        help="'light' est recommandé pour le développement. 'bge' pour la production (plus précis)."
    )

    st.markdown("---")
    st.markdown("### 📂 Ajouter des documents")
    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Formats acceptés : PDF, TXT, CSV"
    )

    if uploaded_files:
        data_dir = os.path.join("data", "raw")
        os.makedirs(data_dir, exist_ok=True)
        for uf in uploaded_files:
            dest = os.path.join(data_dir, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"✅ {len(uploaded_files)} fichier(s) sauvegardé(s) dans `data/raw/`")
        st.info("👇 Cliquez sur **Relancer l'ingestion** pour les indexer.")

    st.markdown("---")
    st.markdown("### 🔄 Index Vectoriel")
    if st.button("🚀 Relancer l'ingestion", use_container_width=True):
        with st.spinner(f"Ingestion en cours (modèle: {model_choice})..."):
            try:
                create_vectorstore(model_choice=model_choice)
                st.success("✅ Ingestion terminée avec succès !")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur lors de l'ingestion : {str(e)}")

    st.markdown("---")
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("BVMT Intelligence v2.0 · LangGraph + Groq + FAISS/BM25")


# ─── En-tête Principal ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 BVMT Intelligence</h1>
    <p>Accédez instantanément à la connaissance de la BVMT : rapports, entreprises, tendances et analyses intelligentes.</p>
</div>
""", unsafe_allow_html=True)

# ─── Chargement du Workflow ───────────────────────────────────────────────────
with st.spinner("⚙️ Chargement de l'index et des agents..."):
    workflow = init_system(model_choice)

if not workflow:
    st.warning(
        "⚠️ L'index n'est pas encore disponible ou n'a pas pu être chargé. "
        "Veuillez (re)lancer l'ingestion depuis le menu de gauche."
    )
    st.stop()

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Historique de conversation ───────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ─── Input utilisateur ────────────────────────────────────────────────────────
if prompt := st.chat_input("💬 Posez votre question sur la bourse tunisienne..."):
    # Affichage immédiat de la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyse en cours..."):
            # Préparer l'historique pour le graphe (4 derniers échanges, hors le message courant)
            history_for_graph = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-5:-1]
            ]

            initial_state = {
                "question": prompt,
                "chat_history": history_for_graph,
                "iterations": 0,
                "documents": [],
                "generation": "",
                "error": "",
                "query_type": ""
            }

            try:
                final_state = workflow.invoke(initial_state)

                response_text = final_state.get("generation", "Désolé, je n'ai pas pu générer de réponse.")
                documents = final_state.get("documents", [])

                st.markdown(response_text)
                render_sources(documents)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": documents
                })

            except Exception as e:
                logger.error(f"Erreur pipeline : {e}", exc_info=True)
                st.error(f"❌ Une erreur s'est produite : {str(e)}")
