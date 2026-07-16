import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from rag_engine import sync_documents_from_folder, rag_graph_app, AgentState

app = FastAPI(title="Production Secure RAG Gateway")

class SyncRequest(BaseModel):
    folder_path: str

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_role: str

@app.post("/api/v1/sync")
async def sync_data(payload: SyncRequest):
    if not os.path.exists(payload.folder_path):
        raise HTTPException(status_code=400, detail="The target folder path does not exist.")
    try:
        sync_result = sync_documents_from_folder(payload.folder_path)
        return {"status": "success", "message": sync_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync execution failed: {str(e)}")

@app.post("/api/v1/chat")
async def chat_session(payload: ChatRequest):
    try:
        config = {"configurable": {"thread_id": payload.thread_id}}
        
        graph_input: AgentState = {
            "messages": [HumanMessage(content=payload.message)],
            "user_role": payload.user_role
        }
        
        output = rag_graph_app.invoke(graph_input, config=config)
        final_reply = output["messages"][-1].content
        
        return {
            "thread_id": payload.thread_id,
            "response": final_reply
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent runtime crash: {str(e)}")