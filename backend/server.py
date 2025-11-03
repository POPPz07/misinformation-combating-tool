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
from pydantic import BaseModel, Field
from PIL import Image
import google.generativeai as genai
from google.generativeai import types
from ddgs import DDGS
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import genai as vertex_genai
from google.genai import types as vertex_types
# Database Import (Firestore)
from google.cloud import firestore

# --- NEW: Firebase Admin Import ---
import firebase_admin
from firebase_admin import credentials, auth



# --- Initial Setup ---
ROOT_DIR = Path(__file__).parent
load_dotenv('.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ADD THESE TWO LINES ---
GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
LOCATION = os.environ['LOCATION']

# --- NEW: Firebase Admin SDK Setup ---
# This automatically uses GOOGLE_APPLICATION_CREDENTIALS
try:
    # Use Application Default Credentials (recommended for Cloud Run/Functions)
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': os.environ['GOOGLE_CLOUD_PROJECT'],
    })
    logging.info("Firebase Admin SDK initialized.")
except Exception as e:
    logging.error(f"FATAL: Could not initialize Firebase Admin SDK: {e}")
    # If GOOGLE_APPLICATION_CREDENTIALS env var is set, try that
    try:
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if cred_path:
            cred_obj = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred_obj, {
                'projectId': os.environ['GOOGLE_CLOUD_PROJECT'],
            })
            logging.info("Firebase Admin SDK initialized using GOOGLE_APPLICATION_CREDENTIALS file.")
        else:
            raise e
    except Exception as final_e:
        logging.error(f"FATAL: Could not initialize Firebase Admin SDK: {final_e}")
        exit()


# --- Firestore Connection ---
try:
    db = firestore.AsyncClient()
    logging.info("Successfully connected to Firestore.")
except Exception as e:
    logging.error(f"FATAL: Could not connect to Firestore: {e}")
    exit()


# --- Gemini API Configuration ---
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    logging.error("FATAL: GEMINI_API_KEY not found in .env file.")

vertex_client = vertex_genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=LOCATION,
    http_options=vertex_types.HttpOptions(timeout=60_000)
)




# --- Pydantic Models (Data Structures) ---
class User(BaseModel):
    id: str = Field(alias="uid") # Use Firebase 'uid' as the ID
    email: Optional[str] = None
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None



class SourceLink(BaseModel):
    title: str
    url: str



class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None # This will be the Firebase uid
    content: str
    content_type: str
    # Updated fields to match your refactored model
    credibilityScore: int
    statusLabel: str
    confidence: float
    analysisReasoning: str
    education_tips: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sourceLinks: List[SourceLink]  # <-- THIS IS THE FIX



class HistoryResponse(BaseModel):
    analyses: List[AnalysisResult]
    total_count: int


class FactCheckRequest(BaseModel):
    claim: str


class FactCheckResponse(BaseModel):
    claim: str
    analysis: str
    sources: List[SourceLink]


async def fact_check_with_flashlite(claim: str) -> dict:
    """
    Uses Gemini 2.5 Flash-Lite with Google Search grounding to verify a factual claim.
    Returns summarized text + list of sources.
    """
    try:
        google_search_tool = vertex_types.Tool(google_search=vertex_types.GoogleSearch())


        # Run in thread executor to avoid blocking
        loop = asyncio.get_event_loop()


        def run_vertex():
            response_text = ""
            last_chunk = None


            for chunk in vertex_client.models.generate_content_stream(
                model="gemini-2.5-flash-lite",
                contents=claim,
                config=vertex_types.GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0.0
                )
            ):
                if chunk.text:
                    response_text += chunk.text
                last_chunk = chunk


            sources = []
            if last_chunk and last_chunk.candidates and last_chunk.candidates[0].grounding_metadata:
                metadata = last_chunk.candidates[0].grounding_metadata
                if metadata.grounding_chunks:
                    for c in metadata.grounding_chunks:
                        if c.web:
                            sources.append({"title": c.web.title, "url": c.web.uri})
            return {"analysis": response_text.strip(), "sources": sources}


        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, run_vertex)
        return result


    except Exception as e:
        logging.exception(f"Error during Flash-Lite fact check: {e}")
        return {
            "analysis": f"Error: {str(e)}",
            "sources": []
        }




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
    """
    Analyzes text or image content using Google's Gemini API to detect misinformation,
    leveraging DuckDuckGo search results for contextual verification.
    Returns a structured JSON analysis.
    """
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")


    # Configure Gemini (replaces old genai.Client)
    genai.configure(api_key=gemini_api_key)


    # --- Perform contextual search via DuckDuckGo ---
    search_results = []
    if content and len(content.strip()) > 0:
        search_query = content[:200] if len(content) > 200 else content
        search_results = await search_with_ddgs(search_query, max_results=5)


    # --- Build search context for model ---
    search_context = ""
    if search_results:
        search_context = "\n\nRelevant search results from DuckDuckGo:\n"
        for idx, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            search_context += f"\n{idx}. Title: {title}\n   URL: {href}\n   Summary: {body}\n"


    # --- System prompt ---
    system_prompt = f"""You are TruthLens, an expert AI misinformation detection system. 
Your primary goal is to analyze content (text or images) and determine its credibility with high accuracy.


**CRITICAL INSTRUCTIONS:**
1. **ANALYZE THE CONTENT:** Carefully analyze the provided content for credibility, checking for signs of misinformation, manipulation, or false claims.
2. **USE SEARCH RESULTS:** Use DuckDuckGo search results to verify claims and identify inconsistencies.
3. **GENERATE HIGH-AUTHORITY SOURCE LINKS:** Provide 3-5 verifiable links from reputable sources (Wikipedia, Reuters, BBC, government sites, etc.).
4. **JSON OUTPUT ONLY:** You MUST respond with only a valid JSON object matching this structure:
   - `credibilityScore`: Integer 0–100 (0 = High Risk of Fake, 100 = Likely True)
   - `statusLabel`: One of "Likely True", "Potentially Misleading", "High Risk of Fake"
   - `confidence`: Float (0.0-1.0)
   - `analysisReasoning`: Detailed reasoning referencing the search results
   - `sourceLinks`: Array of 3–5 dicts each with "title" and "url"
   - `education_tips`: Array of 3–5 strings with user education advice


Analyze for emotional manipulation, unreliable sources, logical fallacies, or altered visuals.


{search_context}"""

    # <-- FIX: Logic moved here to correctly build the 'user_content_list'
    # This list will be passed to the sync function.
    
    # --- Build Gemini contents (text-only or image + text) ---
    pil_image = None
    user_content_list = [system_prompt] # Start with the system prompt

    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Add text and image to the content list
            user_content_list.append(f"Analyze this image. Accompanying text: {content[:500] if content else 'No text'}")
            user_content_list.append(pil_image)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            user_content_list.append(f"Analyze this text (image processing failed): {content[:4000]}")
    else:
        # Add text-only to the content list
        user_content_list.append(f"Analyze this text: {content[:4000]}")


    # --- Configure generation settings ---
    config = types.GenerationConfig(
        response_mime_type="application/json",
        max_output_tokens=1024,
        temperature=0.3
    )


    try:
        # Use synchronous Gemini call in executor to avoid blocking async event loop
        def run_gemini_sync(prepared_contents: List): # <-- FIX: Accept prepared list
            
            # <-- FIX: Wrong model name. 'gemini-1.5-flash' is the correct public model.
            model = genai.GenerativeModel("gemini-2.5-flash-lite")

            # <-- FIX: This function no longer builds its own contents.
            # It uses the list passed to it, which correctly includes the image.
            return model.generate_content(
                contents=prepared_contents,
                generation_config=config,
            )


        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            # <-- FIX: Pass the 'user_content_list' as an argument
            resp = await loop.run_in_executor(pool, run_gemini_sync, user_content_list)


        # --- Parse model response ---
        try:
            result = json.loads(resp.text)


            # Validate required fields
            required_fields = [
                'credibilityScore', 'statusLabel', 'confidence',
                'analysisReasoning', 'sourceLinks', 'education_tips'
            ]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")


            # Type normalization
            result['credibilityScore'] = int(result['credibilityScore'])
            result['confidence'] = float(result['confidence'])


            if not isinstance(result.get('education_tips'), list):
                result['education_tips'] = [result['education_tips']]
            if not isinstance(result.get('sourceLinks'), list):
                result['sourceLinks'] = []


            # Validate link structure
            validated_links = []
            for link in result['sourceLinks']:
                if isinstance(link, dict) and 'title' in link and 'url' in link:
                    validated_links.append({
                        'title': str(link['title']),
                        'url': str(link['url'])
                    })
            result['sourceLinks'] = validated_links


            return result


        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.error(f"Error parsing Gemini response: {e}, Response text: {resp.text[:500]}")
            return {
                "credibilityScore": 50,
                "statusLabel": "Analysis Inconclusive",
                "confidence": 0.3,
                "analysisReasoning": f"Unable to parse analysis result. Raw response: {resp.text[:500]}",
                "sourceLinks": [],
                "education_tips": [
                    "Always verify information with multiple reliable sources.",
                    "Be skeptical of content that cannot be properly analyzed."
                ]
            }


    except Exception as e:
        logging.exception(f"Error during Gemini call: {e}")
        return {
            "credibilityScore": 50,
            "statusLabel": "Analysis Inconclusive",
            "confidence": 0.3,
            "analysisReasoning": f"Unable to complete analysis due to a technical error: {str(e)}",
            "sourceLinks": [],
            "education_tips": [
                "Always verify information with multiple reliable sources.",
                "Technical errors can occur — try again or use alternative fact-checking methods."
            ]
        }



# --- FastAPI App and Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("API is starting up.")
    yield
    logging.info("Closing Firestore connection.")
    await db.close()



app = FastAPI(title="TruthLens API", lifespan=lifespan)
api_router = APIRouter(prefix="/api")
security = HTTPBearer()



# --- Authentication Helpers (NEW: FIREBASE TOKEN VERIFICATION) ---
async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Verifies the Firebase ID token and gets/creates a user in our Firestore.
    """
    if not creds or not creds.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = creds.credentials
    
    try:
        # Verify the token using Firebase Admin SDK
        # This is a synchronous call, but it's fast (mostly local validation + one network call for keys)
        # For very high throughput, you might run it in an executor, but this is usually fine.
        decoded_token = auth.verify_id_token(token)
        
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        
        # Get or create the user in *our* Firestore database
        user_ref = db.collection("users").document(uid)
        user_doc = await user_ref.get()
        
        if not user_doc.exists:
            # Create a new user entry
            user_data = {
                "uid": uid,
                "email": email,
                "name": name,
                "created_at": datetime.now(timezone.utc)
            }
            await user_ref.set(user_data)
            # Use Pydantic's alias to map 'uid' to 'id'
            return User(**user_data)
        else:
            # User already exists
            return User(**user_doc.to_dict())

    except auth.InvalidIdTokenError as e:
        logging.warning(f"Invalid ID token: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        logging.error(f"Auth error: {e}")
        raise HTTPException(status_code=500, detail="Error verifying authentication")





@api_router.get("/auth/me", summary="Get current user info")
async def get_current_user_info(user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user





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

    
    # Save to Firestore
    try:
        await db.collection("analyses").document(result.id).set(result.dict())
        return result
    except Exception as e:
        logging.error(f"Failed to save analysis to Firestore: {e}")
        raise HTTPException(status_code=500, detail="Failed to save analysis result")


@api_router.post("/factcheck", response_model=FactCheckResponse, summary="Verify a factual claim using Gemini 2.5 Flash-Lite")
async def fact_check_endpoint(req: FactCheckRequest, user: Optional[User] = Depends(get_current_user)):
    if not req.claim or len(req.claim.strip()) == 0:
        raise HTTPException(status_code=400, detail="Claim text must be provided.")


    logging.info(f"Fact-checking claim: {req.claim}")
    result = await fact_check_with_flashlite(req.claim)


    response = FactCheckResponse(
        claim=req.claim,
        analysis=result.get("analysis", ""),
        sources=[SourceLink(**src) for src in result.get("sources", [])]
    )


# Optionally store result in DB
    try:
        doc_data = {
            "user_id": user.id if user else None,
            "claim": req.claim,
            "analysis": response.analysis,
            "sources": [s.dict() for s in response.sources],
            "timestamp": datetime.now(timezone.utc)
        }
        await db.collection("factchecks").document().set(doc_data)
    except Exception as e:
        logging.error(f"Failed to save fact-check to Firestore: {e}")
        # Don't fail the whole request, just log the error


    return response




@api_router.get("/history", response_model=HistoryResponse, summary="Get user's analysis history")
async def get_analysis_history(user: User = Depends(get_current_user), limit: int = 20, skip: int = 0):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")


# 1. Create the query
    query_ref = db.collection("analyses").where(field_path="user_id", op_string="==", value=user.id).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).offset(skip)
    
    # 2. Get the documents
    analyses_docs_stream = query_ref.stream()
    analyses = [AnalysisResult(**doc.to_dict()) async for doc in analyses_docs_stream]
    
    # 3. Get the total count (This is a separate query in Firestore)
    count_query = db.collection("analyses").where(field_path="user_id", op_string="==", value=user.id)
    count_stream = count_query.stream()
    total_count = len([doc async for doc in count_stream])


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
