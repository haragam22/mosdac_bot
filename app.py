import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- SET ENVIRONMENT VARIABLES ---
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# --- LOAD VECTOR STORE ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="mosdac_chroma", embedding_function=embedding_model)

# --- LOAD RETRIEVER + LLM ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"]
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
