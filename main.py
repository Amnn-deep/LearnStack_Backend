from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import httpx
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pymongo
from bson import ObjectId
from fastapi import HTTPException
import jwt
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from fastapi import Body


app = FastAPI()

# Load environment variables from .env file (absolute path for reliability)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[str] = []

class ChatResponse(BaseModel):
    reply: str
    history: List[str]

class ChatHistoryOut(BaseModel):
    id: str
    history: list[str]
    last_message: str
    reply: str

    @classmethod
    def from_mongo(cls, doc):
        return cls(
            id=str(doc["_id"]),
            history=doc["history"],
            last_message=doc["last_message"],
            reply=doc["reply"]
        )

@app.get("/")
def root():
    import traceback
    print("[DEBUG] / root endpoint called")
    try:
        return {"message": "LearnStack AI FastAPI backend is running!"}
    except Exception as e:
        print("[ERROR] Exception in / root endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def get_gpt_response(message: str, history: list[str]) -> str:
    """Call Groq API for a response."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[Error: Groq API key not set in environment variable 'GROQ_API_KEY']"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = []
    # General system prompt for helpful AI assistant (like ChatGPT)
    messages.append({
        "role": "system",
        "content": (
            "You are an AI assistant. Answer user questions helpfully, accurately, and conversationally. If the user asks for study material, generate structured learning content. Otherwise, respond naturally."
        )
    })
    for i, msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": message})
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            # Remove all asterisks from the response
            response_text = response_text.replace('**', '').replace('*', '')
            return response_text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "[Error: Unauthorized. Your Groq API key is invalid or not permitted for this endpoint. Please check your API key and permissions.]"
            elif e.response.status_code == 429:
                return "[Error: Rate limit exceeded or quota reached for your Groq API key. Please check your Groq account, billing, or try again later.]"
            else:
                return f"[Error: Groq API returned status {e.response.status_code}: {e.response.text}]"
        except Exception as e:
            return f"[Error: {str(e)}]"

# MongoDB setup (sync for Vercel compatibility)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "learnstackai")
mongo_client = pymongo.MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60

security = HTTPBearer()

def create_jwt_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return decode_jwt_token(credentials.credentials)

# Example: Auth endpoint to get a token (for demo, no real user check)
@app.post("/auth/token")
async def login_demo(username: str):
    import traceback
    print(f"[DEBUG] /auth/token endpoint called with username={username}")
    try:
        token = create_jwt_token({"username": username})
        return {"access_token": token}
    except Exception as e:
        print("[ERROR] Exception in /auth/token endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Save chat to MongoDB after each chat
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_current_user)])
def chat_endpoint(chat: ChatRequest):
    import traceback
    print(f"[DEBUG] /chat endpoint called with message={chat.message}")
    try:
        reply = get_gpt_response(chat.message, chat.history)
        new_history = chat.history + [chat.message, reply]
        chat_doc = {
            "history": new_history,
            "last_message": chat.message,
            "reply": reply
        }
        result = db.chats.insert_one(chat_doc)
        print(f"[DEBUG] Chat inserted with id {result.inserted_id}")
        return ChatResponse(reply=reply, history=new_history)
    except Exception as e:
        print("[ERROR] Exception in /chat endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get all chat histories
@app.get("/chats", response_model=list[ChatHistoryOut], dependencies=[Depends(get_current_user)])
def get_all_chats():
    import traceback
    print("[DEBUG] /chats endpoint called")
    chats = []
    try:
        for doc in db.chats.find().sort("_id", -1):
            chats.append(ChatHistoryOut.from_mongo(doc))
        print(f"[DEBUG] Retrieved {len(chats)} chats from DB")
        return chats
    except Exception as e:
        print("[ERROR] Exception in /chats endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get a single chat by ID
@app.get("/chats/{chat_id}", response_model=ChatHistoryOut, dependencies=[Depends(get_current_user)])
def get_chat(chat_id: str):
    import traceback
    print(f"[DEBUG] /chats/{{chat_id}} endpoint called with chat_id={chat_id}")
    try:
        doc = db.chats.find_one({"_id": ObjectId(chat_id)})
        if not doc:
            print(f"[DEBUG] Chat with id {chat_id} not found in DB")
            raise HTTPException(status_code=404, detail="Chat not found")
        print(f"[DEBUG] Chat with id {chat_id} found, returning data")
        return ChatHistoryOut.from_mongo(doc)
    except Exception as e:
        print("[ERROR] Exception in /chats/{chat_id} endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Delete a chat by ID
@app.delete("/chats/{chat_id}", dependencies=[Depends(get_current_user)])
def delete_chat(chat_id: str):
    import traceback
    print(f"[DEBUG] /chats/{{chat_id}} DELETE endpoint called with chat_id={chat_id}")
    try:
        result = db.chats.delete_one({"_id": ObjectId(chat_id)})
        if result.deleted_count == 0:
            print(f"[DEBUG] Chat with id {chat_id} not found for deletion")
            raise HTTPException(status_code=404, detail="Chat not found")
        print(f"[DEBUG] Chat with id {chat_id} deleted successfully")
        return {"status": "deleted"}
    except Exception as e:
        print("[ERROR] Exception in /chats/{chat_id} DELETE endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Delete all chats
@app.delete("/chats", dependencies=[Depends(get_current_user)])
def delete_all_chats():
    import traceback
    print("[DEBUG] /chats DELETE endpoint called (delete all)")
    try:
        db.chats.delete_many({})
        print("[DEBUG] All chats deleted successfully")
        return {"status": "all deleted"}
    except Exception as e:
        print("[ERROR] Exception in /chats DELETE endpoint:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


