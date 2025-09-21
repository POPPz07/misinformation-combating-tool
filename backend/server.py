import os
import logging
import json
import base64
import io
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from PIL import Image
import google.generativeai as genai

# --- Initial Setup ---
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection ---
try:
    mongo_url = os.environ['MONGO_URL']
    db_name = os.environ['DB_NAME']
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
except KeyError as e:
    logging.error(f"FATAL: Environment variable {e} not found. Your .env file is incomplete.")
    exit()

# --- Gemini API Configuration ---
try:
    gemini_api_key = os.environ['GEMINI_API_KEY']
    genai.configure(api_key=gemini_api_key)
except KeyError:
    logging.error("FATAL: GEMINI_API_KEY not found in .env file.")
    gemini_api_key = None

# --- Pydantic Models (Data Structures) ---
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_token: str
    expires_at: datetime

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None

class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    content: str
    content_type: str
    credibility_score: int
    verdict: str
    confidence: float
    reasoning: str
    education_tips: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HistoryResponse(BaseModel):
    analyses: List[AnalysisResult]
    total_count: int

# --- Core Gemini Analysis Logic ---
import base64
import io
import json
import logging
from typing import Optional

import google.generativeai as genai
from fastapi import HTTPException
from PIL import Image

# Assume gemini_api_key is defined globally as before

async def analyze_content_with_gemini(content: str, image_base64: Optional[str]) -> dict:
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    # This prompt is for the first step: getting a detailed, text-based analysis.
    # It no longer needs to mention JSON.
    system_prompt = """You are TruthLens, an expert AI misinformation detection system. Your primary goal is to analyze content (text or images) and determine its credibility with high accuracy.

    **CRITICAL INSTRUCTIONS:**
    1.  **USE THE GOOGLE SEARCH TOOL:** Before providing any analysis, you MUST use the provided Google Search tool to find real-time information, verify claims, check sources, and look for context or debunking articles related to the content. Your reasoning must be grounded in the search results.
    2.  **DETAILED ANALYSIS:** Provide comprehensive reasoning based on your search findings. Explain any manipulative techniques, verify facts, assess source credibility, and give a clear verdict on the content's likely truthfulness.

    Analyze the user's content for indicators like emotional manipulation, unreliable sources, manipulated data, suspicious context, and logical fallacies. For images, also consider signs of visual manipulation."""

    try:
        # --- Step 1: Get grounded analysis using Google Search ---
        search_model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=system_prompt,
            # ✅ CORRECTED: Use a simple string in a list for the built-in search tool.
            tools=['google_search'],
            generation_config={"max_output_tokens": 2048}
        )

        # Prepare prompt for either image or text analysis
        prompt_parts = []
        if image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            mime_type = Image.MIME.get(img.format, "image/jpeg")
            prompt_parts.append({"mime_type": mime_type, "data": image_bytes})
            prompt_parts.append(f"Analyze this image. Accompanying text: {content[:500] if content else 'No text'}")
        else:
            prompt_parts.append(f"Analyze this text: {content[:4000]}")

        grounded_response = await search_model.generate_content_async(prompt_parts)
        grounded_text = grounded_response.text

        # --- Step 2: Structure the analysis into JSON ---
        json_model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-lite',
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 1024
            }
        )

        json_prompt = f"""
        Convert the following analysis into the exact JSON format specified.

        Source Analysis:
        ---
        {grounded_text}
        ---

        Required JSON format:
        {{
            "credibility_score": <number 0-100>,
            "verdict": "<string, one of: 'Likely True', 'Potentially Misleading', 'High Risk of Fake'>",
            "confidence": <number 0.0-1.0>,
            "reasoning": "<detailed explanation from source analysis>",
            "education_tips": ["<tip1>", "<tip2>", "<tip3>"]
        }}

        Ensure the output is ONLY the valid JSON object with no additional text or markdown.
        """

        json_response = await json_model.generate_content_async(json_prompt)

        try:
            return json.loads(json_response.text)
        except json.JSONDecodeError:
            logging.error("Failed to decode the final JSON response from the structuring model.")
            return {
                "credibility_score": 50,
                "verdict": "Analysis Inconclusive",
                "confidence": 0.3,
                "reasoning": "Unable to complete analysis due to a final content formatting error.",
                "education_tips": ["Double-check information from primary sources.", "Be wary of content that confirms strong biases."]
            }

    except Exception as e:
        logging.error(f"Error during the two-step Gemini analysis: {str(e)}")
        return {
            "credibility_score": 50,
            "verdict": "Analysis Inconclusive",
            "confidence": 0.3,
            "reasoning": "Unable to complete analysis due to a technical error during generation.",
            "education_tips": ["Always verify information with multiple reliable sources.", "Be skeptical of emotionally charged content."]
        }
# --- FastAPI App and Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("API is starting up.")
    yield
    logging.info("Closing MongoDB connection.")
    client.close()

app = FastAPI(title="TruthLens API", lifespan=lifespan)
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

# --- Authentication Helpers & Routes (Dummy Implementation) ---
async def get_current_user(request: Request) -> Optional[User]:
    session_token = request.cookies.get("session_token")
    if not session_token:
        return None
    
    session = await db.sessions.find_one({
        "session_token": session_token,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    if not session:
        return None
    
    user_data = await db.users.find_one({"id": session["user_id"]})
    return User(**user_data) if user_data else None

@api_router.post("/auth/session", summary="Create a dummy session for testing")
async def create_dummy_session(response: Response):
    dummy_email = "testuser@example.com"
    user_data = await db.users.find_one({"email": dummy_email})
    if not user_data:
        user = User(email=dummy_email, name="Test User")
        await db.users.insert_one(user.dict())
    else:
        user = User(**user_data)
    
    session_token = f"st_dummy_{uuid.uuid4()}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    session_obj = Session(user_id=user.id, session_token=session_token, expires_at=expires_at)
    await db.sessions.insert_one(session_obj.dict())
    
    response.set_cookie(
        "session_token", session_token, max_age=7*24*3600, httponly=True,
        secure=True, samesite="none", path="/"
    )
    return {"user": user.dict()}

@api_router.get("/auth/me", summary="Get current user info")
async def get_current_user_info(user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@api_router.post("/auth/logout", summary="Log out the user")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.sessions.delete_one({"session_token": session_token})
    response.delete_cookie("session_token", path="/")
    return {"message": "Logged out successfully"}

# --- Core API Routes ---
@api_router.post("/analyze", response_model=AnalysisResult, summary="Analyze content for misinformation")
async def analyze_content_endpoint(req: AnalysisRequest, user: Optional[User] = Depends(get_current_user)):
    content_text = req.text or "Image-only analysis"
    
    analysis_data = await analyze_content_with_gemini(
        content=content_text, image_base64=req.image_base64
    )
    
    content_type = "image" if req.image_base64 else "text"
    result = AnalysisResult(
        user_id=user.id if user else None,
        content=content_text[:250],
        content_type=content_type,
        **analysis_data
    )
    await db.analyses.insert_one(result.dict())
    return result

@api_router.get("/history", response_model=HistoryResponse, summary="Get user's analysis history")
async def get_analysis_history(user: User = Depends(get_current_user), limit: int = 20, skip: int = 0):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    analyses_cursor = db.analyses.find({"user_id": user.id}).sort("timestamp", -1).skip(skip).limit(limit)
    analyses = await analyses_cursor.to_list(length=limit)
    total_count = await db.analyses.count_documents({"user_id": user.id})
    
    return HistoryResponse(
        analyses=[AnalysisResult(**analysis) for analysis in analyses],
        total_count=total_count
    )

# --- Final App Configuration ---
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)