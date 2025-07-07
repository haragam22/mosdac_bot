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

# --- STREAMLIT UI ---
st.set_page_config(page_title="ISRO Bot 🚀", layout="wide")
st.title("🛰️ ISRO Help Bot")
st.write("Ask anything about MOSDAC or ISRO satellite data...")

query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("Searching..."):
        response = qa_chain.run(query)
        st.success("Answer:")
        st.write(response)
