import os
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import speech_recognition as sr

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- ENV + MODELS ---
load_dotenv()




# Specify model name or local path
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Wrap it properly
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"]
)

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="ISRO Bot 🚀", layout="wide")
st.title("🛰️ ISRO Help Bot")
st.markdown("Ask anything about **MOSDAC**, or upload your own **PDF/TXT** for Q&A!")

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

    file_vectorstore = Chroma.from_documents(split_docs, embedding_model)
    file_retriever = file_vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.custom_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=file_retriever, chain_type="stuff")
    st.success("✅ File processed. You can now ask questions about it below.")

# --- VOICE INPUT FUNCTION ---
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please speak your question.")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("❌ Could not understand audio.")
        return None
    except sr.RequestError:
        st.error("🔌 Speech Recognition API not available.")
        return None

# --- CHAT HISTORY DISPLAY ---
with st.container():
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user", avatar="🧑‍💻").write(user_msg)
        st.chat_message("assistant", avatar="🤖").write(bot_msg)

# --- INPUT METHOD CHOICE ---
input_mode = st.radio("Choose input method:", ["📝 Text", "🎤 Voice"], horizontal=True)

# --- USER INPUT HANDLING ---
user_input = None

if input_mode == "📝 Text":
    user_input = st.chat_input("💬 Enter your question here")
elif input_mode == "🎤 Voice":
    if st.button("🎤 Tap to Speak"):
        user_input = recognize_speech()
        if user_input:
            st.success(f"✅ You said: {user_input}")

# --- RESPONSE GENERATION ---
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

    with st.spinner("🤖 Generating response..."):
        qa_chain = st.session_state.custom_qa_chain or mosdac_chain
        response = qa_chain.run(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)

    st.session_state.chat_history.append((user_input, response))
