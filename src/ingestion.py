import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 設定資料路徑 (改成你的 wiki txt)
DATA_PATH = "./data/banff_wiki.txt" 
DB_PATH = "./db/chroma_db"

def clean_text(text):
    # 1. 移除 這種標籤
    text = re.sub(r'\\', '', text)
    
    # 2. 移除 OSM directions edit 這種導航雜訊
    text = re.sub(r'OSM directions(\s*edit)?', '', text)
    
    # 3. 移除 Wikidata 連結資訊
    text = re.sub(r'\(Q\d+\)\s*on\s*Wikidata', '', text)
    
    # 4. 移除多餘的空行 (把多個換行合併成兩個)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def ingest_data():
    print(f"--- 開始處理資料: {DATA_PATH} ---")
    
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    
    # [關鍵步驟] 清洗資料
    print("正在清洗文字雜訊...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        
    print(f"原始文件載入並清洗完成")

    # 文字切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文字切分完成，共產生 {len(chunks)} 個 chunks")

    # 向量化與儲存
    print("--- 載入本地 Embedding 模型 (第一次執行會下載模型檔，約 100MB) ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(DB_PATH):
        print("偵測到舊資料庫，正在清除以免重複入庫...")
        import shutil
        shutil.rmtree(DB_PATH) # 清除舊的向量資料庫資料夾

    print("--- 開始向量化並寫入資料庫 ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"--- 資料庫建置完成！儲存於: {DB_PATH} ---")

if __name__ == "__main__":
    ingest_data()
