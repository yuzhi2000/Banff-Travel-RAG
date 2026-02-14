import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from google import genai
# from google.genai import types

load_dotenv()

DB_PATH = "./db/chroma_db"

def format_docs(docs):
    """
    將檢索到的多個文件片段合併成一個字串
    """
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # 1. 準備 Embedding Function (必須跟 ingestion 時用的一模一樣)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 2. 載入向量資料庫
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # 3. 建立 Retriever (檢索器)
    # search_kwargs={"k": 3} 表示每次只找最相關的 3 筆資料
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4. 準備 LLM (使用 Gemini Pro)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", # 使用最新的 Flash 模型
        temperature=0,      # 設為 0 讓回答盡量根據事實，不要隨意發揮
    )

    # 5. 設計 Prompt Template (提示詞模板)
    # 這是 RAG 的靈魂，告訴 LLM 只能根據 Context 回答
    template = """你是一個專業的班夫國家公園旅遊嚮導。
    請根據以下的 Context 資訊回答使用者的問題。
    如果 Context 裡沒有答案，請直接說「抱歉，根據目前的資料我無法回答這個問題」，不要編造資訊。
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. 建立 RAG Chain (鏈)
    # 這是 LangChain 的 LCEL 語法 (LangChain Expression Language)
    # 流程: 
    #   1. 把 question 傳給 retriever 找 context
    #   2. 把 question 傳給 prompt
    #   3. 把 context 傳給 prompt
    #   4. 把 prompt 丟給 llm
    #   5. 把 llm 的輸出轉成字串
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. 互動迴圈
    print("--- 班夫旅遊助手已啟動 (輸入 'quit' 離開) ---")
    while True:
        query = input("\n請輸入你的問題: ")
        if query.lower() in ["quit", "exit"]:
            break
        
        print(f"正在思考中...")
        response = rag_chain.invoke(query)
        print(f"\n回答: {response}")

if __name__ == "__main__":
    main()
