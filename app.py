import os
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- ENV + MODELS ---
load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"]
)

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="ISRO Bot 🚀", layout="wide")
st.title("🛰️ ISRO Help Bot")
st.markdown("Ask anything about **MOSDAC** or upload your own **PDF/TXT file** for Q&A!")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "custom_qa_chain" not in st.session_state:
    st.session_state.custom_qa_chain = None

# --- VECTORSTORE (MOSDAC) ---
mosdac_vectorstore = Chroma(persist_directory="mosdac_chroma", embedding_function=embedding_model)
mosdac_retriever = mosdac_vectorstore.as_retriever(search_kwargs={"k": 3})
mosdac_chain = RetrievalQA.from_chain_type(llm=llm, retriever=mosdac_retriever, chain_type="stuff")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file (optional)", type=["pdf", "txt"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if temp_path.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path)

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    file_vectorstore = Chroma.from_documents(split_docs, embedding_model)  # in-memory
    file_retriever = file_vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.custom_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=file_retriever, chain_type="stuff")
    st.success("✅ File processed. You can now ask questions about it below.")

# --- CHAT HISTORY UI ---
with st.container():
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user", avatar="🧑‍💻").write(user_msg)
        st.chat_message("assistant", avatar="🤖").write(bot_msg)

# --- CHAT INPUT ---
user_input = st.chat_input("💬 Enter your question")

if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        # Use uploaded file chain if available, else fallback to MOSDAC
        qa_chain = st.session_state.custom_qa_chain or mosdac_chain
        response = qa_chain.run(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)

    st.session_state.chat_history.append((user_input, response))
