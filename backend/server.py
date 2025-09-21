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
from google import genai
from google.genai import types
from ddgs import DDGS
import asyncio




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
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    logging.error("FATAL: GEMINI_API_KEY not found in .env file.")


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


# --- DuckDuckGo Search Function ---
async def search_with_ddgs(query: str, max_results: int = 5) -> List[dict]:
    """
    Perform a search using DuckDuckGo and return results.
    Runs synchronous DDGS in executor to avoid blocking.
    """
    try:
        loop = asyncio.get_event_loop()
        # Run the synchronous DDGS search in an executor
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results))
        )
        return results
    except Exception as e:
        logging.error(f"Error performing DuckDuckGo search: {e}")
        return []


# --- Updated Gemini Analysis Function ---
async def analyze_content_with_gemini(content: str, image_base64: Optional[str]) -> dict:
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    client = genai.Client(api_key=gemini_api_key)
    
    # First, perform a DuckDuckGo search to gather context
    search_results = []
    if content and len(content.strip()) > 0:
        # Create a search query from the content (limit to reasonable length)
        search_query = content[:200] if len(content) > 200 else content
        search_results = await search_with_ddgs(search_query, max_results=5)
    
    # Format search results for Gemini
    search_context = ""
    if search_results:
        search_context = "\n\nRelevant search results from DuckDuckGo:\n"
        for idx, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            search_context += f"\n{idx}. Title: {title}\n   URL: {href}\n   Summary: {body}\n"
    
    # System prompt for Gemini
    system_prompt = f"""You are TruthLens, an expert AI misinformation detection system. Your primary goal is to analyze content (text or images) and determine its credibility with high accuracy.

    **CRITICAL INSTRUCTIONS:**
    1. **ANALYZE THE CONTENT:** Carefully analyze the provided content for credibility, checking for signs of misinformation, manipulation, or false claims.
    
    2. **USE SEARCH RESULTS:** I have provided DuckDuckGo search results below. Use these to:
       - Verify claims made in the content
       - Check if the information aligns with credible sources
       - Identify if the content contradicts established facts
       - Look for signs that the content might be from unreliable sources
    
    3. **JSON OUTPUT ONLY:** You MUST respond with only a valid JSON object matching this structure:
       - `credibility_score`: An integer from 0 (High Risk of Fake) to 100 (Likely True).
       - `verdict`: A string, one of: "Likely True", "Potentially Misleading", "High Risk of Fake".
       - `confidence`: A float from 0.0 to 1.0 representing your confidence in the assessment.
       - `reasoning`: A detailed string explaining your verdict, referencing the search results where relevant. Explain any manipulative techniques used (e.g., emotional language, missing sources, logical fallacies).
       - `education_tips`: A JSON array of 3-5 actionable string tips to help users identify similar misinformation in the future.

    Analyze for indicators like emotional manipulation, unreliable sources, manipulated data, suspicious context, and logical fallacies. For images, also consider signs of visual manipulation.
    
    {search_context}"""

    # Build contents: text-only or text + PIL image
    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            contents = [f"Analyze this image. Accompanying text: {content[:500] if content else 'No text'}", img]
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            contents = [f"Analyze this text (image processing failed): {content[:4000]}"]
    else:
        contents = [f"Analyze this text: {content[:4000]}"]

    # Configure Gemini without grounding (we're providing our own search results)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        max_output_tokens=1024,
        temperature=0.3  # Lower temperature for more consistent analysis
    )

    try:
        # Async call to Gemini
        resp = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",  # Using the experimental model without grounding
            contents=contents,
            config=config
        )

        # Parse the JSON response
        try:
            result = json.loads(resp.text)
            # Validate the response structure
            required_fields = ['credibility_score', 'verdict', 'confidence', 'reasoning', 'education_tips']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure types are correct
            result['credibility_score'] = int(result['credibility_score'])
            result['confidence'] = float(result['confidence'])
            if not isinstance(result['education_tips'], list):
                result['education_tips'] = [result['education_tips']]
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.error(f"Error parsing Gemini response: {e}, Response text: {resp.text[:500]}")
            return {
                "credibility_score": 50,
                "verdict": "Analysis Inconclusive",
                "confidence": 0.3,
                "reasoning": f"Unable to parse analysis result. Raw response: {resp.text[:500]}",
                "education_tips": ["Always verify information with multiple reliable sources.", "Be skeptical of content that cannot be properly analyzed."]
            }

    except Exception as e:
        logging.exception(f"Error during Gemini call: {e}")
        return {
            "credibility_score": 50,
            "verdict": "Analysis Inconclusive",
            "confidence": 0.3,
            "reasoning": f"Unable to complete analysis due to a technical error: {str(e)}",
            "education_tips": ["Always verify information with multiple reliable sources.", "Technical errors can occur - try again or use alternative fact-checking methods."]
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
    if not req.text and not req.image_base64:
        raise HTTPException(status_code=400, detail="Either text or image must be provided")
    
    content_text = req.text or "Image-only analysis"
    
    # Perform analysis with DuckDuckGo search + Gemini
    analysis_data = await analyze_content_with_gemini(
        content=content_text, 
        image_base64=req.image_base64
    )
    
    content_type = "image" if req.image_base64 else "text"
    result = AnalysisResult(
        user_id=user.id if user else None,
        content=content_text[:250],
        content_type=content_type,
        **analysis_data
    )
    
    # Save to database
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

# --- Health Check Endpoint ---
@api_router.get("/health", summary="Health check endpoint")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# --- Final App Configuration ---
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)