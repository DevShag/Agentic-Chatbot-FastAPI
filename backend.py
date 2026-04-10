from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent
import os

# ----------------------------------
# Config (Environment Friendly)
# ----------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("POST", 9999))


class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

#API_URL = "http://0.0.0.0:9999/chat"

ALLOWED_MODEL_NAMES= ['llama3-70b-8192', 'mixtral-8x7b-32768', 'llama-3.3-70b-versatile', 'gpt-4o-mini']


#Step2: Setup AI Agent from FrontEnd Request
app = FastAPI(title="LangGraph AI Agent")

# Health Check (for Docker)
# ---------------------------
def health_check():
    return {'status': 'running'}


@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {'error' : 'Invalid model nmae. Kindly select a valid AI model'}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Create AI Agent and get response from it!
    response = get_response_from_ai_agent(
        llm_id = llm_id,
        query = query,
        allow_search = allow_search,
        system_prompt = system_prompt,
        provider = provider
    )
    return response


#Step3: Run app & Explore Swagger UI Docs
if __name__=="__main__":
    import uvicorn
   
    #uvicorn.run(app, host="127.0.0.1", port=9999)
    uvicorn.run(
        "backend:app",
        host=HOST,
        port=PORT,
        reload=True
    )