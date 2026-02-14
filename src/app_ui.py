import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 設定網頁標題與配置
st.set_page_config(page_title="Banff 班夫智慧旅遊助手", layout="wide")
st.title("🏔️ Banff National Park AI Guide")
st.caption("基於 RAG 技術 (Gemini + ChromaDB + Multilingual Embedding)")

# 2. 載入環境變數與快取資源
load_dotenv()
DB_PATH = "./db/chroma_db"

# 使用 @st.cache_resource 避免每次網頁重整都重新載入模型 (這很重要！)
@st.cache_resource
def get_rag_chain():
    # A. 準備 Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # B. 載入向量資料庫
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    # k=5 確保能抓到補全的規則
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # C. 準備 LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

    # D. Prompt Template
    template = """你是一個專業的班夫國家公園旅遊嚮導。
    
    請遵循以下回答邏輯：
    1. **優先使用 Context**：如果下方的 Context 包含回答問題所需的資訊，請直接引用 Context 回答，並盡量詳細。
    2. **自有知識兜底**：如果 Context 裡 **完全沒有** 相關資訊，請使用你作為大型語言模型的自有知識來回答。
    
    ⚠️ **重要限制**：
    - 如果你使用了自有知識（非 Context 內容），請在回答的開頭加上標註：**「(注意：以下資訊來自 AI 資料庫，非本次檢索結果，僅供參考)」**。
    - 如果是關於具體數據（如票價、開放時間）且 Context 沒有，請誠實說不知道，不要瞎掰數字。

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # E. 構建 Chain (使用 RunnableParallel 來同時回傳答案與來源)
    # 這裡的技巧是：我們先並行取得 context 和 question，
    # 然後把這兩個丟給 prompt -> llm -> parser 產生 answer，
    # 最後我們會得到一個字典：{'context': [...], 'question': '...', 'answer': '...'}
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=prompt | llm | StrOutputParser())
    )
    
    return chain

# 初始化 Chain
rag_chain = get_rag_chain()

# 3. 處理 Chat History (Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是班夫旅遊助手，有什麼關於行程、交通或美食的問題都可以問我喔！"}]

# 顯示過去的對話紀錄
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. 處理使用者輸入
if user_query := st.chat_input("請輸入你的問題... (例如：夢蓮湖可以開車去嗎？)"):
    # 顯示使用者問題
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # AI 思考與生成
    with st.chat_message("assistant"):
        with st.spinner("正在檢索旅遊指南..."):
            # 呼叫 RAG Chain
            result = rag_chain.invoke(user_query)
            answer = result['answer']
            source_docs = result['context']

            # 顯示回答
            st.write(answer)
            
            # 更新對話紀錄
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- 關鍵功能：在側邊欄顯示引用來源 (Source Citation) ---
            with st.sidebar:
                st.header("📚 檢索到的參考資料")
                st.write(f"針對問題：**{user_query}**")
                st.divider()
                for i, doc in enumerate(source_docs):
                    with st.expander(f"來源片段 #{i+1}"):
                        st.markdown(f"**內容摘要:** {doc.page_content}")
                        st.caption(f"Metadata: {doc.metadata}")
