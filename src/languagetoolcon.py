"""
LanguageTool Middleware Server
Production-ready FastAPI middleware for LanguageTool response cleaning.
"""

import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import requests
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="LanguageTool Middleware", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LanguageTool backend configuration
LANGUAGETOOL_URL = "http://localhost:8081/v2/check"
MAX_TEXT_LENGTH = 50000  # Maximum text length to prevent abuse
REQUEST_TIMEOUT = (5, 30)  # Connect timeout, read timeout

# Session for connection pooling
session = requests.Session()
session.headers.update({"User-Agent": "LanguageTool-Middleware/1.0"})

# Server start time for uptime tracking
START_TIME = datetime.now()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses with timing."""
    start_time = time.time()
    
    # Log incoming request
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Incoming: {request.method} {request.url.path} from {client_host}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(
            f"Completed: {request.method} {request.url.path} "
            f"Status={response.status_code} Duration={duration:.3f}s"
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error: {request.method} {request.url.path} {str(e)} Duration={duration:.3f}s")
        raise


def clean_response(lt_response: dict) -> dict:
    """
    Transform LanguageTool's verbose JSON response into a clean, minimal format.
    
    Removes:
    - Software metadata
    - Warnings
    - Language detection details
    - Extended sentence ranges
    - Unnecessary nested structures
    
    Returns only the essential mistake information.
    """
    try:
        if not isinstance(lt_response, dict) or "matches" not in lt_response:
            logger.warning("Invalid LanguageTool response structure")
            return {"mistakes": [], "total_mistakes": 0, "summary": "No issues found"}
        
        mistakes = []
        matches = lt_response.get("matches", [])
        
        if not isinstance(matches, list):
            logger.warning("Matches field is not a list")
            return {"mistakes": [], "total_mistakes": 0, "summary": "No issues found"}
        
        for idx, match in enumerate(matches, start=1):
            if not isinstance(match, dict):
                logger.warning(f"Skipping invalid match at index {idx}")
                continue
                
            # Extract context text snippet
            context_info = match.get("context", {})
            context_text = context_info.get("text", "") if isinstance(context_info, dict) else ""
            
            # Get the actual error text from the original text using offset and length
            error_text = ""
            if "offset" in match and "length" in match:
                if context_text and isinstance(context_info, dict):
                    ctx_offset = context_info.get("offset", 0)
                    try:
                        error_text = context_text[ctx_offset:ctx_offset + match["length"]]
                    except (TypeError, IndexError):
                        error_text = ""
            
            # Extract replacement suggestions (limit to top 5 for readability)
            replacements_raw = match.get("replacements", [])
            replacements = []
            if isinstance(replacements_raw, list):
                replacements = [
                    r.get("value", "") 
                    for r in replacements_raw[:5]
                    if isinstance(r, dict) and "value" in r
                ]
            
            # Get rule information
            rule = match.get("rule", {})
            rule_id = "UNKNOWN"
            issue_type = ""
            if isinstance(rule, dict):
                rule_id = rule.get("id", "UNKNOWN")
                issue_type = rule.get("issueType", "")
            
            mistake = {
                "mistake_number": idx,
                "rule_id": rule_id,
                "message": match.get("message", ""),
                "short_message": match.get("shortMessage", ""),
                "context": context_text,
                "error_text": error_text,
                "sentence": match.get("sentence", ""),
                "replacements": replacements,
                "issue_type": issue_type,
            }
            
            mistakes.append(mistake)
        
        return {
            "mistakes": mistakes,
            "total_mistakes": len(mistakes),
            "summary": f"Found {len(mistakes)} potential issue(s)"
        }
    except Exception as e:
        logger.error(f"Error cleaning response: {str(e)}", exc_info=True)
        return {
            "mistakes": [],
            "total_mistakes": 0,
            "summary": "Error processing results"
        }


@app.post("/v2/check")
@limiter.limit("100/minute")
async def check_grammar(
    request: Request,
    text: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    """
    Grammar checking endpoint compatible with LanguageTool API.
    Accepts both form-data and JSON requests.
    Rate limit: 100 requests per minute per IP.
    """
    try:
        # Try to get data from form first (LanguageTool default)
        if text is None:
            # If not in form, try JSON body
            try:
                body = await request.json()
                text = body.get("text")
                language = body.get("language", language)
            except Exception as e:
                logger.warning(f"Failed to parse JSON body: {str(e)}")
                raise HTTPException(status_code=400, detail="No text provided")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text too long: {len(text)} characters")
            raise HTTPException(
                status_code=413,
                detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters"
            )
        
        # Default language if not specified
        if not language:
            language = "en-US"
        
        # Validate language code format (basic check)
        if not isinstance(language, str) or len(language) > 10:
            logger.warning(f"Invalid language code: {language}")
            language = "en-US"
        
        logger.info(f"Processing text: {len(text)} chars, language: {language}")
        
        # Forward request to LanguageTool
        lt_data = {
            "text": text,
            "language": language
        }
        
        response = session.post(
            LANGUAGETOOL_URL, 
            data=lt_data, 
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"LanguageTool returned {response.status_code}: {response.text[:200]}")
            raise HTTPException(
                status_code=response.status_code,
                detail="LanguageTool service error"
            )
        
        # Validate response is JSON
        try:
            lt_response = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON from LanguageTool: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail="Invalid response from LanguageTool"
            )
        
        # Clean and transform the response
        cleaned = clean_response(lt_response)
        
        logger.info(f"Processed successfully: {cleaned['total_mistakes']} mistakes found")
        
        return JSONResponse(content=cleaned)
    
    except HTTPException:
        raise
    except requests.Timeout:
        logger.error("LanguageTool request timeout")
        raise HTTPException(
            status_code=504,
            detail="LanguageTool service timeout"
        )
    except requests.ConnectionError as e:
        logger.error(f"Cannot connect to LanguageTool: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LanguageTool service"
        )
    except requests.RequestException as e:
        logger.error(f"LanguageTool request error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="LanguageTool service error"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Check if the middleware and LanguageTool backend are healthy."""
    try:
        response = session.get("http://localhost:8081", timeout=5)
        lt_status = "healthy" if response.status_code in [200, 404] else "unhealthy"
        lt_reachable = True
    except Exception as e:
        logger.warning(f"LanguageTool health check failed: {str(e)}")
        lt_status = "unreachable"
        lt_reachable = False
    
    uptime = datetime.now() - START_TIME
    
    return {
        "middleware": "healthy",
        "languagetool": lt_status,
        "languagetool_url": LANGUAGETOOL_URL,
        "uptime_seconds": int(uptime.total_seconds()),
        "version": "1.0.0",
        "max_text_length": MAX_TEXT_LENGTH
    }


@app.get("/")
async def root():
    """API information endpoint."""
    return {
        "service": "LanguageTool Middleware",
        "version": "1.0.0",
        "endpoints": {
            "/v2/check": "POST - Check grammar and spelling",
            "/health": "GET - Health check",
        },
        "backend": LANGUAGETOOL_URL
    }


if __name__ == "__main__":
    PORT = 7680  # Change to 8443 for testing without admin privileges
    
    logger.info(f"🚀 Starting LanguageTool Middleware on port {PORT}")
    logger.info(f"📡 Forwarding to LanguageTool at {LANGUAGETOOL_URL}")
    logger.info(f"⚙️  Max text length: {MAX_TEXT_LENGTH} characters")
    logger.info(f"🔒 Rate limit: 100 requests/minute per IP")
    
    if PORT == 443:
        logger.warning("⚠️  Port 443 requires administrator privileges on Windows")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )
