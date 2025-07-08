import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- LOAD VECTOR STORE ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="mosdac_chroma", embedding_function=embedding_model)

# --- LOAD RETRIEVER + LLM ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"]
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="ISRO Bot 🚀", layout="wide")
st.title("🛰️ ISRO Help Bot")
st.markdown("Ask anything about **MOSDAC** or **ISRO satellite data** 👇")

# --- SESSION STATE FOR CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- CHAT UI ---
with st.container():
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.chat_message("user", avatar="🧑‍💻").write(user_msg)
        st.chat_message("assistant", avatar="🤖").write(bot_msg)

# --- INPUT + RESPONSE HANDLER ---
user_input = st.chat_input("💬 Enter your question")

if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        response = qa_chain.run(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)

    # Save to chat history
    st.session_state.chat_history.append((user_input, response))
