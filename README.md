# 🏔️ Banff Travel RAG: Multilingual AI Tour Guide

A Retrieval-Augmented Generation (RAG) system designed for Banff National Park tourism. This project demonstrates how to build a **context-aware**, **multilingual** AI assistant using localized embedding models and vector databases.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Pro-orange)

## 🚀 Key Features

* **Multilingual Retrieval:** Uses `paraphrase-multilingual-MiniLM-L12-v2` to enable seamless querying in Chinese/English against an English knowledge base.
* **Zero-Hallucination Design:** Strict prompt engineering ensures the AI admits ignorance when data is missing ("I don't know" instead of making things up).
* **Source Transparency:** The UI explicitly displays retrieved document chunks and metadata alongside generated answers for verification.
* **Cost-Effective Architecture:**
    * **Embedding:** Local HuggingFace model (Free & Private).
    * **Vector DB:** ChromaDB (Local & Persistent).
    * **LLM:** Google Gemini API (High free tier quota).

## 🏗️ Architecture

1.  **Ingestion Pipeline:** * Loads raw text/PDF data.
    * Cleans noise (Regex).
    * Splits text into semantic chunks (RecursiveCharacterTextSplitter).
    * Generates embeddings and stores in ChromaDB.
2.  **Retrieval Chain:**
    * Converts user query to vector.
    * Retrieves Top-K relevant chunks via Cosine Similarity.
3.  **Generation:**
    * Feeds context + query to Gemini 1.5 Flash.
    * Streamlit renders the response and citations.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Banff-Travel-RAG.git](https://github.com/YOUR_USERNAME/Banff-Travel-RAG.git)
    cd Banff-Travel-RAG
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

4.  **Initialize Knowledge Base**
    ```bash
    python src/ingestion.py
    ```

5.  **Run the App**
    ```bash
    streamlit run src/app_ui.py
    ```

## 📂 Project Structure

```text
Banff-RAG/
├── data/               # Raw knowledge source (Wikivoyage, Official Guides)
├── db/                 # ChromaDB persistence directory
├── src/
│   ├── ingestion.py    # ETL Pipeline: Cleaning -> Chunking -> Embedding
│   └── app_ui.py       # Streamlit UI & RAG Chain Logic
├── .env                # API Keys (Not uploaded)
├── .gitignore
└── requirements.txt


## 📝 Future Improvements
* Implement Hybrid Search (BM25 + Dense Vector).
* Add Multi-turn conversation memory.
* Deploy to Streamlit Cloud.
