import os
from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
DB_PATH = "./db/chroma_db"

def test_retrieval(query_text):
    print(f"--- 測試查詢: {query_text} ---")
    
    # 準備 Embedding Function
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 載入現有的資料庫
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 執行相似度搜尋 (Similarity Search)
    # k=3 表示回傳最相似的 3 筆資料
    results = vector_store.similarity_search_with_score(query_text, k=3)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n[結果 {i+1}] (距離分數: {score:.4f})") 
        # Chroma 的分數通常是 L2 Distance，越低代表越相似
        print(f"內容摘要: {doc.page_content[:150]}...") 
        print(f"來源: {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    # 測試一個具體的問題
    test_retrieval("班夫冬天有哪些推薦的活動？")