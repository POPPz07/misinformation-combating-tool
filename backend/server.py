from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import base64
import aiohttp
import json
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent, FileContentWithMimeType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer(auto_error=False)

# Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    content_type: str = "text"  # "text" or "image"

class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    content: str
    content_type: str
    credibility_score: int  # 0-100
    verdict: str  # "Likely True", "Potentially Misleading", "High Risk of Fake"
    confidence: float  # 0.0-1.0
    reasoning: str
    education_tips: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HistoryResponse(BaseModel):
    analyses: List[AnalysisResult]
    total_count: int

# Auth helper functions
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """Get current user from session token in cookie or Authorization header"""
    session_token = None
    
    # Try to get token from HttpOnly cookie first
    session_token = request.cookies.get("session_token")
    
    # Fallback to Authorization header
    if not session_token and credentials:
        session_token = credentials.credentials
    
    if not session_token:
        return None
    
    # Find session in database
    session = await db.sessions.find_one({
        "session_token": session_token,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not session:
        return None
    
    # Get user
    user = await db.users.find_one({"id": session["user_id"]})
    if not user:
        return None
    
    return User(**user)

# Analysis helper functions
async def analyze_content_with_gemini(content: str, content_type: str = "text", image_base64: Optional[str] = None) -> dict:
    """Analyze content using Gemini 2.5 Pro"""
    try:
        # Initialize Gemini chat
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"analysis_{uuid.uuid4()}",
            system_message="""You are TruthLens, an expert misinformation detection system. Your job is to analyze content and determine its credibility.

For each piece of content, provide:
1. Credibility Score (0-100): 0 = Definitely False, 100 = Definitely True
2. Verdict: "Likely True" (70-100), "Potentially Misleading" (30-69), "High Risk of Fake" (0-29)
3. Confidence Level (0.0-1.0): How confident you are in your assessment
4. Reasoning: Detailed explanation of why the content may be misleading
5. Education Tips: 3-5 actionable tips to help users identify similar misinformation

Look for these misinformation indicators:
- Emotional manipulation language
- Missing or unreliable sources
- Manipulated statistics or data
- Suspicious timing or context
- Visual manipulation (for images)
- Logical fallacies
- Appeals to fear or anger

Respond with a JSON object containing: credibility_score, verdict, confidence, reasoning, education_tips"""
        ).with_model("gemini", "gemini-2.0-flash")
        
        # Create message based on content type
        if content_type == "image" and image_base64:
            # Analyze image content
            image_content = ImageContent(image_base64=image_base64)
            message = UserMessage(
                text=f"Analyze this image for misinformation. Consider visual manipulation, context, and any text within the image. Original text context: {content[:500] if content else 'No additional text provided'}",
                file_contents=[image_content]
            )
        else:
            # Analyze text content
            message = UserMessage(text=f"Analyze this content for misinformation: {content}")
        
        # Send message and get response
        response = await chat.send_message(message)
        
        # Parse JSON response
        try:
            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError:
            # Fallback parsing if response isn't valid JSON
            return {
                "credibility_score": 50,
                "verdict": "Potentially Misleading",
                "confidence": 0.5,
                "reasoning": response[:500] + "..." if len(response) > 500 else response,
                "education_tips": [
                    "Always verify information with multiple reliable sources",
                    "Check the author's credentials and background",
                    "Look for peer-reviewed sources and official statements",
                    "Be skeptical of emotionally charged content",
                    "Cross-reference with fact-checking websites"
                ]
            }
            
    except Exception as e:
        logging.error(f"Error analyzing content with Gemini: {str(e)}")
        # Return a safe fallback result
        return {
            "credibility_score": 50,
            "verdict": "Potentially Misleading",
            "confidence": 0.3,
            "reasoning": "Unable to complete full analysis. Please verify this content manually with trusted sources.",
            "education_tips": [
                "Always verify information with multiple reliable sources",
                "Check the author's credentials and background",
                "Look for peer-reviewed sources and official statements",
                "Be skeptical of emotionally charged content",
                "Cross-reference with fact-checking websites"
            ]
        }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "TruthLens API is running"}

# Auth routes
@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    """Handle session creation after OAuth redirect"""
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")
    
    # Call Emergent auth API to get user data
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_id}
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=401, detail="Invalid session")
                
                user_data = await resp.json()
        except Exception as e:
            logging.error(f"Error fetching session data: {str(e)}")
            raise HTTPException(status_code=500, detail="Authentication error")
    
    # Create or get user
    existing_user = await db.users.find_one({"email": user_data["email"]})
    if existing_user:
        user = User(**existing_user)
    else:
        user = User(
            email=user_data["email"],
            name=user_data["name"],
            picture=user_data.get("picture")
        )
        await db.users.insert_one(user.dict())
    
    # Create session token
    session_token = f"st_{uuid.uuid4()}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    session_obj = Session(
        user_id=user.id,
        session_token=session_token,
        expires_at=expires_at
    )
    
    await db.sessions.insert_one(session_obj.dict())
    
    # Set HttpOnly cookie
    response.set_cookie(
        "session_token",
        session_token,
        max_age=7 * 24 * 3600,  # 7 days
        httponly=True,
        secure=True,
        samesite="none",
        path="/"
    )
    
    return {"user": user.dict(), "session_token": session_token}

@api_router.get("/auth/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current user information"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, user: User = Depends(get_current_user)):
    """Logout user"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get session token from cookie
    session_token = request.cookies.get("session_token")
    
    if session_token:
        # Remove session from database
        await db.sessions.delete_one({"session_token": session_token})
    
    # Clear cookie
    response.delete_cookie("session_token", path="/")
    
    return {"message": "Logged out successfully"}

# Analysis routes
@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_content(
    request: AnalysisRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Analyze content for misinformation"""
    if not request.text and not request.image_base64:
        raise HTTPException(status_code=400, detail="Either text or image must be provided")
    
    content = request.text or "Image analysis"
    
    # Analyze with Gemini
    analysis = await analyze_content_with_gemini(
        content=content,
        content_type=request.content_type,
        image_base64=request.image_base64
    )
    
    # Create result
    result = AnalysisResult(
        user_id=user.id if user else None,
        content=content[:200] + "..." if len(content) > 200 else content,
        content_type=request.content_type,
        credibility_score=analysis["credibility_score"],
        verdict=analysis["verdict"],
        confidence=analysis["confidence"],
        reasoning=analysis["reasoning"],
        education_tips=analysis["education_tips"]
    )
    
    # Save to database
    await db.analyses.insert_one(result.dict())
    
    return result

@api_router.get("/history", response_model=HistoryResponse)
async def get_analysis_history(
    user: User = Depends(get_current_user),
    limit: int = 20,
    skip: int = 0
):
    """Get user's analysis history"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Get user's analyses
    analyses = await db.analyses.find(
        {"user_id": user.id}
    ).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
    
    # Get total count
    total_count = await db.analyses.count_documents({"user_id": user.id})
    
    return HistoryResponse(
        analyses=[AnalysisResult(**analysis) for analysis in analyses],
        total_count=total_count
    )

# Public history route for non-authenticated users (browser storage)
@api_router.get("/public-history")
async def get_public_history():
    """Get recent public analyses (for demonstration)"""
    # Get recent analyses without user_id (public analyses)
    analyses = await db.analyses.find(
        {"user_id": None}
    ).sort("timestamp", -1).limit(10).to_list(10)
    
    return {"analyses": [AnalysisResult(**analysis) for analysis in analyses]}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()