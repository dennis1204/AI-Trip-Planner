from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import traceback
from dotenv import load_dotenv

# 引入 Gemini 專用的 LangChain 模組
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. 初始化 FastAPI 應用程式
app = FastAPI(title="HK Travel Planner AI Backend (Gemini Version)")
load_dotenv()

# 1. 初始化 Gemini 模型 (推薦使用速度極快的 flash 模型)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    api_key=os.getenv("GEMINI_API_KEY")
)

# 2. 定義前端傳來的資料格式 (Data Model)
class ChatRequest(BaseModel):
    user_query: str
    # 在實際應用中，可以傳入 session_id 來管理不同使用者的對話歷史

# 全域變數暫存對話歷史 (僅為示範，實務上應存入資料庫或 Redis)
history = [
    {
        "role": "system",
        "content": "You are an expert HK travel planner...", # 放入你原本的 system prompt
    }
]

# 3. 建立 POST API 端點
@app.post("/api/chat")
async def chat_with_planner(request: ChatRequest):
    try:
        print(f"👉 收到請求: {request.user_query}")
        
        # 紀錄使用者的問題
        history.append(HumanMessage(content=request.user_query))

        print("👉 準備呼叫 Gemini API...")
        
        # 3. 將整個對話歷史直接丟給 Gemini 模型
        response = llm.invoke(history)
        
        # 取出 AI 的文字回覆
        ai_response = response.content
        print("👉 成功收到回應！")

        # 將 AI 的回答存入記憶中
        history.append(AIMessage(content=ai_response))

        return {
            "status": "success",
            "reply": ai_response
        }

    except Exception as e:
        print("❌ 發生錯誤！")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 執行指令: uvicorn main:app --reload