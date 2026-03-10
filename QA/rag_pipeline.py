import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from loaders.all_loaders import custom_loader
from utils.splitter import split_text as custom_text_splitter

# Load API key from environment / .env file
load_dotenv()

# Project root is one level above this file's directory (QA/)
ROOT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
DATA_DIR = ROOT_DIR / "data"


class ConversationalRAG:
    def __init__(self):
        # Expect GEMINI_API_KEY (configured in Streamlit / Vercel or local .env)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        # langchain-google-genai expects GOOGLE_API_KEY to be set
        os.environ["GOOGLE_API_KEY"] = gemini_key

        # Use local HuggingFace sentence-transformer embeddings (no external API)
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-2.5-flash")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        index_path = VECTORSTORE_DIR / "index.faiss"
        if index_path.exists():
            print("🔄 Loading existing FAISS index...")
            self.db = FAISS.load_local(str(VECTORSTORE_DIR), self.embedding, allow_dangerous_deserialization=True)
        else:
            print("⚙️ No index found, creating new one...")
            self._build_index()

        self._create_qa_chain()

    def _create_qa_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.db.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

    def _build_index(self):
        documents = custom_loader(str(DATA_DIR))
        split_docs = custom_text_splitter(documents)
        split_docs = [doc for doc in split_docs if doc.page_content.strip()]
        self.db = FAISS.from_documents(split_docs, self.embedding)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        self.db.save_local(str(VECTORSTORE_DIR))

    def rebuild_index(self):
        print("♻️ Rebuilding FAISS index...")
        self._build_index()
        self.memory.clear()  # 🧠 Clear old conversation context
        self._create_qa_chain()
        print("✅ Index rebuilt and memory cleared.")

    def ask(self, question: str):
        return self.qa_chain.invoke({"question": question})
