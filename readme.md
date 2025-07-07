# 🚀 ISRO HelpBot — Intelligent FAQ & Data Retrieval Assistant for MOSDAC

ISRO HelpBot is an AI-powered chatbot designed to provide instant, accurate, and structured responses related to the [MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre)](https://mosdac.gov.in/) website.

The bot scrapes relevant sections of the website, embeds the content using Sentence Transformers, stores it using ChromaDB, and answers user queries using an open-source LLM via OpenRouter.

---

## 📌 Features

* 🔍 **Intelligent Question Answering** over scraped content (text).
* 📁 **FAQ understanding** for user support queries (custom tuned).
* 🧠 **Vector store-based retrieval** using `ChromaDB` and `all-MiniLM-L6-v2`.
* 💬 **Natural Language Responses** via open-source LLMs (DeepSeek, Mistral etc.).
* 💻 **Streamlit Web Interface** for easy interaction.
* 🧵 **System Prompting** for structured, contextual, and professional answers.

---

## 📂 Project Structure

```bash
├── app.py                  # Streamlit UI code
├── scrape_and_store.py     # Scraping & preprocessing script
├── vector_db_setup.py      # Vector store creation code
├── chatbot_core.py         # Langchain LLM + QA chain
├── mosdac_chroma/          # Persisted Chroma vector DB
├── data/                   # Scraped and cleaned text files
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions (Run on any machine)

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/isro-helpbot.git
cd isro-helpbot
```

### 2. **Create & Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Set Your API Key (OpenRouter API)**

Create a `.env` file or set the environment variable:

```bash
export OPENAI_API_KEY="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"
```

Alternatively, replace `os.getenv("OPENAI_API_KEY")` in your code with your key directly if local only.

### 5. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## 🚪 Example Questions to Try

* “What is MOSDAC and what type of data does it provide?”
* “List ISRO's current meteorological satellite missions.”
* “How do I access oceanographic datasets?”
* “Explain the difference between INSAT-3D and SCATSAT-1.”
* “What is the process to download data from MOSDAC?”
* “Are there any live satellite services available?”
* “What are frequently asked questions on MOSDAC?”
* “How does ISRO handle real-time weather monitoring?”

---

## 🚧 Future Enhancements

* 🗏️ Read **Excel or CSV datasets** from MOSDAC
* 🖼️ Understand **satellite images** via captioning
* 📊 Create **visual dashboards** from live data
* 💡 Plug in **live LLM-based updaters** for real-time FAQs

---

## 🙌 Acknowledgements

* [Langchain](https://www.langchain.com/)
* [ChromaDB](https://www.trychroma.com/)
* [OpenRouter](https://openrouter.ai/)
* [HuggingFace Sentence Transformers](https://www.sbert.net/)
* [Streamlit](https://streamlit.io/)

---

## 📜 License

MIT License
