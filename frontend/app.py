import os
import shutil
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# Ensure project root is on sys.path so we can import QA, loaders, etc.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from QA.rag_pipeline import ConversationalRAG  # noqa: E402


# Load environment variables (works on Streamlit Cloud and locally)
load_dotenv()


@st.cache_resource(show_spinner=True)
def get_rag():
    return ConversationalRAG()


st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📚 AI-Powered Research Assistant")
st.markdown("Ask questions about your documents or websites. Powered by RAG + LangChain + Gemini 2.5 Flash.")


# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Ensure upload directory exists (under project root)
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Upload documents
uploaded_files = st.file_uploader(
    "📤 Upload files (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

rag_chat = get_rag()

if uploaded_files:
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with st.spinner(f"Uploading and indexing {file.name}..."):
            try:
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file, f)
                # Rebuild index after new uploads
                rag_chat.rebuild_index()
                st.success(f"✅ {file.name} uploaded and indexed successfully!")
            except Exception as e:
                st.error(f"❌ Upload/indexing failed: {e}")


# Ask question
question = st.text_input("💬 Ask a question:")

if st.button("Ask") and question.strip():
    try:
        result = rag_chat.ask(question)
        answer = result.get("answer", "")
        sources = result.get("source_documents", [])

        if answer.strip():
            st.session_state.chat_history.append(
                (question, answer, sources),
            )
        else:
            st.warning("⚠️ No valid answer returned.")
    except Exception as e:
        st.error(f"❌ Error: {e}")


# Show chat history
st.divider()
st.subheader("🧠 Conversation History")

for q, a, sources in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    if sources:
        with st.expander("📄 Sources"):
            for s in sources:
                # Each `s` is a Document; show a snippet of its content
                try:
                    content = getattr(s, "page_content", str(s))
                except Exception:
                    content = str(s)
                st.code(content[:500], language="text")
    st.markdown("---")


# Clear history
if st.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
