"""
GECToR Grammar Correction API Server
FastAPI server with rolling window correction for texts exceeding 512 tokens
Optimized for 2vCPU + 16GB RAM, ports: 80, 443, 8080
"""

"""
GECToR Grammar Correction API Server
"""

import os

# ===== CRITICAL: Set cache before imports =====
for cache_dir in ['/tmp/huggingface', '/tmp/huggingface/transformers', 
                  '/tmp/huggingface/datasets', '/tmp/huggingface/hub']:
    os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface/datasets'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface/hub'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Now safe to import transformers
import torch
torch.cuda.is_available = lambda: False

import torch
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import time
import re
import psutil
import traceback
from typing import List, Tuple, Optional
from datetime import datetime
import logging

# Configure structured logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Fix for PyTorch 2.9+ meta tensor error with GECToR
# Patch transformers' resize_token_embeddings to disable mean_resizing
from transformers import PreTrainedModel
_original_resize = PreTrainedModel.resize_token_embeddings

def _patched_resize(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=False):
    """Patched version that forces mean_resizing=False to avoid meta tensor errors"""
    return _original_resize(self, new_num_tokens, pad_to_multiple_of, mean_resizing=False)

PreTrainedModel.resize_token_embeddings = _patched_resize
logger.info("Applied PyTorch 2.9+ compatibility patch for GECToR")



# Global model variables
model = None
tokenizer = None
encode = None
decode = None

# Configuration
VERSION = "1.0.0"
MODEL_ID = 'gotutiyan/gector-deberta-large-5k'
MAX_TOKENS = 512
OVERLAP_RATIO = 0.15  # Optimized from 30% to 15% - reduces redundant processing
MIN_ERROR_PROB = 0.6
N_ITERATION = 5
MAX_TEXT_LENGTH = 50000  # Limit input to prevent memory issues
BATCH_SIZE = 4  # Process multiple chunks in parallel (optimal for 2vCPU)
RATE_LIMIT = "100/minute"  # Rate limit per IP

# Global state
startup_time: Optional[float] = None

# Custom Exceptions
class TextTooLargeError(Exception):
    """Raised when text exceeds maximum allowed length"""
    pass

class ModelProcessingError(Exception):
    """Raised when model processing fails"""
    pass

class ModelNotLoadedError(Exception):
    """Raised when model is not loaded"""
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer, encode, decode, startup_time
    
    # Track startup time for uptime monitoring
    startup_time = time.time()
    
    logger.info("Loading GECToR model from local cache...")
    
    try:
        if not os.path.exists('data/verb-form-vocab.txt'):
            raise RuntimeError("verb-form-vocab.txt not found in data/ directory!")
        
        # Load from local cache only
        model = GECToR.from_pretrained(MODEL_ID)
        model = model.to('cpu')
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        encode, decode = load_verb_dict('data/verb-form-vocab.txt')
        
        logger.info("Model loaded successfully from cache on CPU")
        logger.info(f"Version: {VERSION}")
        logger.info(f"Optimizations: Sentence-based chunking | {OVERLAP_RATIO*100}% overlap | Batch size: {BATCH_SIZE}")
        logger.info(f"Max tokens per chunk: {MAX_TOKENS}")
        logger.info(f"Rate limit: {RATE_LIMIT}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield  # Server runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down...")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="GECToR Grammar Correction API",
    version=VERSION,
    lifespan=lifespan,
    description="High-performance grammar correction API with rolling window support for long texts",
)

# Attach rate limiter to app state
app.state.limiter = limiter

# Add exception handler for rate limiting
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address(request)}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "limit": RATE_LIMIT
        }
    )

# Add exception handler for text too large
@app.exception_handler(TextTooLargeError)
async def text_too_large_handler(request: Request, exc: TextTooLargeError):
    logger.warning(f"Text too large: {str(exc)}")
    return JSONResponse(
        status_code=413,
        content={
            "error": "Payload too large",
            "message": f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters",
            "max_length": MAX_TEXT_LENGTH
        }
    )

# Add exception handler for model processing errors
@app.exception_handler(ModelProcessingError)
async def model_processing_error_handler(request: Request, exc: ModelProcessingError):
    logger.error(f"Model processing error: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Grammar correction failed",
            "message": "An error occurred while processing your text. Please try again."
        }
    )

# Add exception handler for model not loaded
@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    logger.error("Model not loaded when request received")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service unavailable",
            "message": "Grammar correction service is not ready. Please try again in a moment."
        }
    )

# Add generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CorrectionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This are wrong grammar and it need to be fix",
                "min_error_prob": 0.6,
                "n_iteration": 5
            }
        }
    )
    
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="Text to correct")
    min_error_prob: float = Field(0.6, ge=0.0, le=1.0, description="Minimum error probability threshold")
    n_iteration: int = Field(5, ge=1, le=10, description="Number of correction iterations")
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v

class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    num_chunks: int
    processing_time_seconds: float
    changes_made: bool

def split_into_complete_sentences(text: str) -> List[str]:
    """
    Split text into complete sentences with proper punctuation handling.
    
    Handles:
    - Sentence terminators: . ? ! (and standalone semicolons)
    - Abbreviations: Dr., Mr., Mrs., Ms., Ph.D., etc., i.e., e.g., vs., Inc., Ltd.
    - Semicolons in lists (doesn't split on these)
    - Preserves punctuation with each sentence
    
    Examples:
        "Dr. Smith likes apples; oranges; milk. He is happy!"
        → ["Dr. Smith likes apples; oranges; milk.", "He is happy!"]
        
        "I love it. They are great!"
        → ["I love it.", "They are great!"]
    """
    # Common abbreviations that end with period (not sentence boundaries)
    abbreviations = [
        'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr', 
        'Ph.D', 'M.D', 'B.A', 'M.A', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'etc', 'vs', 'Inc', 'Ltd', 'Co', 'Corp',
        'i.e', 'e.g', 'a.m', 'p.m', 'U.S', 'U.K',
        'i.e.', 'e.g.', 'a.m.', 'p.m.', 'U.S.', 'U.K.'
    ]
    
    # Split on sentence boundaries: [.!?] followed by space and capital letter
    # But check it's not an abbreviation first
    sentences = []
    current = []
    words = text.split()
    
    for i, word in enumerate(words):
        current.append(word)
        
        # Check if this word ends with a sentence terminator
        if word and word[-1] in '.!?':
            # Check if it's an abbreviation
            is_abbrev = any(word.rstrip('.!?').endswith(abbr.rstrip('.')) for abbr in abbreviations)
            
            # If not abbreviation and next word starts with capital, it's a sentence boundary
            if not is_abbrev and i + 1 < len(words) and words[i + 1][0].isupper():
                sentences.append(' '.join(current))
                current = []
    
    # Add remaining words
    if current:
        sentences.append(' '.join(current))
    
    # Handle semicolons that act as sentence separators (not in lists)
    # A semicolon is a separator if followed by space and capital letter
    result = []
    for sent in sentences:
        # Check if this sentence contains standalone semicolons (sentence separators)
        # Pattern: semicolon followed by space and capital letter, not in a list context
        # List context: has colons before semicolons (e.g., "buy: apples; oranges")
        if ';' in sent:
            # Check if it's a list (has colon before semicolons)
            colon_pos = sent.find(':')
            semicolon_positions = [i for i, c in enumerate(sent) if c == ';']
            
            # If semicolons come after a colon, it's likely a list - don't split
            is_list = colon_pos >= 0 and any(sp > colon_pos for sp in semicolon_positions)
            
            if not is_list:
                # Split on semicolons followed by space and capital
                subsents = re.split(r';\s+(?=[A-Z])', sent)
                # Add semicolon back to each part except last
                for i, ss in enumerate(subsents[:-1]):
                    result.append(ss + ';')
                if subsents[-1].strip():
                    result.append(subsents[-1])
            else:
                result.append(sent)
        else:
            result.append(sent)
    
    # Clean up and filter
    return [s.strip() for s in result if s.strip()]

def chunk_by_fixed_overlap(text: str, tokenizer, max_tokens: int = MAX_TOKENS, overlap_sentences: int = 2) -> Tuple[List[str], List[int]]:
    """
    Split text into chunks with FIXED sentence overlap (not percentage).
    
    Returns:
        chunks: List of text chunks
        overlap_counts: List indicating how many sentences in each chunk overlap from previous chunk
        
    Example with overlap_sentences=2:
        Sentences: ["A.", "B.", "C.", "D.", "E."]
        Chunks: 
        - Chunk 0: ["A.", "B.", "C."] → overlap_count=0
        - Chunk 1: ["B.", "C.", "D."] → overlap_count=2
        - Chunk 2: ["C.", "D.", "E."] → overlap_count=2
    """
    # Tokenize full text to count tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return ([text], [0])
    
    # Split into sentences first (better for GECToR context)
    sentences = split_into_complete_sentences(text)
    
    if len(sentences) == 0:
        return ([text], [0])
    
    chunks = []
    overlap_counts = []
    sentence_idx = 0
    
    while sentence_idx < len(sentences):
        current_chunk_sentences = []
        current_tokens = 0
        chunk_start_idx = sentence_idx
        
        # Determine overlap count for this chunk
        if len(chunks) == 0:
            # First chunk has no overlap
            overlap_count = 0
        else:
            # Subsequent chunks: go back by overlap_sentences
            overlap_count = min(overlap_sentences, sentence_idx)
            chunk_start_idx = sentence_idx - overlap_count
        
        # Build chunk starting from chunk_start_idx
        for i in range(chunk_start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_token_count > max_tokens and current_chunk_sentences:
                # Chunk is full, stop here
                break
            
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_token_count
            
            # If this is a new sentence (not overlap), advance the index
            if i >= sentence_idx:
                sentence_idx = i + 1
        
        # Save the chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
            overlap_counts.append(overlap_count)
        else:
            # Edge case: single sentence too long, split by words
            long_sentence = sentences[sentence_idx - 1 if sentence_idx > 0 else 0]
            words = long_sentence.split()
            word_chunk = []
            word_tokens = 0
            
            for word in words:
                word_token_count = len(tokenizer.encode(word, add_special_tokens=False))
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunks.append(' '.join(word_chunk))
                    overlap_counts.append(0)  # Word-level chunks don't have sentence overlap
                    word_chunk = []
                    word_tokens = 0
                
                word_chunk.append(word)
                word_tokens += word_token_count
            
            if word_chunk:
                chunks.append(' '.join(word_chunk))
                overlap_counts.append(0)
            
            sentence_idx += 1
    
    return (chunks, overlap_counts)

def merge_with_adaptive_overlap(
    corrected_chunks: List[str],
    overlap_counts: List[int],
    similarity_threshold: float = 0.9
) -> str:
    """
    Merge corrected chunks using adaptive overlap strategy.
    
    Compares overlapping sentences and:
    - If similarity >= threshold: Keep previous version (consistency)
    - If similarity < threshold: Use current version (better context)
    
    Args:
        corrected_chunks: List of corrected text chunks
        overlap_counts: Number of sentences in each chunk that overlap from previous
        similarity_threshold: Minimum similarity to keep previous version (default 0.9)
    
    Returns:
        Merged corrected text
        
    Example:
        Chunk 1: "I like apples. They are delicious."
        Chunk 2 (overlap=1): "They are delicious. My friend agrees."
        → If both have "They are delicious." (similarity=1.0), keep chunk 1's version
        → Add "My friend agrees."
    """
    from difflib import SequenceMatcher
    
    if len(corrected_chunks) == 0:
        return ""
    
    if len(corrected_chunks) == 1:
        return corrected_chunks[0]
    
    # Start with first chunk (no overlap)
    result_sentences = split_into_complete_sentences(corrected_chunks[0])
    
    for chunk_idx in range(1, len(corrected_chunks)):
        current_chunk = corrected_chunks[chunk_idx]
        overlap_count = overlap_counts[chunk_idx]
        
        # Split current chunk into sentences
        current_sentences = split_into_complete_sentences(current_chunk)
        
        if overlap_count == 0:
            # No overlap, just append all sentences
            result_sentences.extend(current_sentences)
            continue
        
        # Compare overlapping sentences
        # Get last N sentences from result (where N = overlap_count)
        comparison_count = min(overlap_count, len(result_sentences))
        previous_overlap_sents = result_sentences[-comparison_count:] if comparison_count > 0 else []
        
        # Get first N sentences from current chunk
        current_overlap_sents = current_sentences[:min(overlap_count, len(current_sentences))]
        
        # Compare each overlapping sentence
        sentences_to_replace = []
        for i in range(min(len(previous_overlap_sents), len(current_overlap_sents))):
            prev_sent = previous_overlap_sents[i]
            curr_sent = current_overlap_sents[i]
            
            # Calculate similarity
            similarity = SequenceMatcher(None, prev_sent.lower(), curr_sent.lower()).ratio()
            
            logger.debug(f"Comparing overlap {i+1}: similarity={similarity:.3f}")
            logger.debug(f"  Previous: {prev_sent[:80]}...")
            logger.debug(f"  Current:  {curr_sent[:80]}...")
            
            if similarity < similarity_threshold:
                # Current version is significantly different, use it (better context)
                sentences_to_replace.append((i, curr_sent))
                logger.debug(f"  → Using current version (better context)")
            else:
                # Keep previous version (consistency)
                logger.debug(f"  → Keeping previous version (consistent)")
        
        # Replace sentences if needed
        for idx, new_sent in sentences_to_replace:
            result_idx = len(result_sentences) - comparison_count + idx
            if 0 <= result_idx < len(result_sentences):
                result_sentences[result_idx] = new_sent
        
        # Add non-overlapping sentences from current chunk
        non_overlap_start = min(overlap_count, len(current_sentences))
        if non_overlap_start < len(current_sentences):
            result_sentences.extend(current_sentences[non_overlap_start:])
    
    return ' '.join(result_sentences)

def correct_text(text: str, min_error_prob: float = MIN_ERROR_PROB, n_iteration: int = N_ITERATION) -> Tuple[str, int]:
    """
    Correct text using GECToR with FIXED sentence overlap and adaptive merging.
    Returns: (corrected_text, num_chunks)
    
    Optimizations:
    - Sentence-based chunking with FIXED 2-sentence overlap
    - Adaptive merging: compares corrected overlaps, keeps better version
    - Batch processing (processes multiple chunks in parallel)
    
    Raises:
        ModelProcessingError: If correction fails
        ValueError: If input is invalid
    """
    # Validate input
    if not text or not text.strip():
        raise ValueError("Text cannot be empty or only whitespace")
    
    # Validate model is loaded
    if model is None or tokenizer is None or encode is None or decode is None:
        raise ModelNotLoadedError("Model components not fully loaded")
    
    try:
        # Split text into manageable chunks with FIXED sentence overlap
        chunks, overlap_counts = chunk_by_fixed_overlap(text, tokenizer, MAX_TOKENS, overlap_sentences=2)
        
        # Validate chunking results
        if not isinstance(chunks, list) or not isinstance(overlap_counts, list):
            raise ModelProcessingError("Invalid chunking result")
        
        if len(chunks) == 0:
            raise ValueError("Text could not be chunked properly")
        
        logger.info(f"Split text into {len(chunks)} chunks with fixed 2-sentence overlap")
        logger.info(f"Overlap counts: {overlap_counts}")
        
        # Process chunks in batches for better performance
        corrected_chunks = []
        batch_size = min(BATCH_SIZE, len(chunks))  # Don't exceed available chunks
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1} (chunks {batch_start+1}-{batch_end}/{len(chunks)})")
            
            try:
                # Process batch of chunks together
                corrected_batch = predict(
                    model, 
                    tokenizer, 
                    batch,  # Pass list of chunks for batch processing
                    encode, 
                    decode,
                    keep_confidence=0.0,
                    min_error_prob=min_error_prob,
                    n_iteration=n_iteration,
                    batch_size=batch_size,  # Use optimized batch size
                )
                
                # Validate prediction results
                if not isinstance(corrected_batch, list):
                    logger.warning(f"Prediction returned non-list: {type(corrected_batch)}")
                    corrected_chunks.extend(batch)  # Fallback to original
                else:
                    corrected_chunks.extend(corrected_batch)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_start//batch_size + 1}: {str(e)}")
                logger.error(traceback.format_exc())
                # Fallback to original chunks if correction fails
                corrected_chunks.extend(batch)
        
        # Validate we have all chunks
        if len(corrected_chunks) != len(chunks):
            logger.warning(f"Chunk count mismatch: {len(corrected_chunks)} vs {len(chunks)}")
            raise ModelProcessingError("Incomplete correction results")
        
        # Merge corrected chunks with adaptive overlap strategy
        final_corrected = merge_with_adaptive_overlap(corrected_chunks, overlap_counts, similarity_threshold=0.9)
        
        # Validate final result
        if not final_corrected or not isinstance(final_corrected, str):
            raise ModelProcessingError("Invalid correction result")
        
        return final_corrected, len(chunks)
        
    except (ModelProcessingError, ModelNotLoadedError, ValueError):
        # Re-raise expected exceptions
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        logger.error(f"Unexpected error in correct_text: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelProcessingError(f"Text correction failed: {type(e).__name__}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "status": "online",
        "service": "GECToR Grammar Correction API",
        "version": VERSION,
        "model": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "overlap_ratio": OVERLAP_RATIO,
        "max_text_length": MAX_TEXT_LENGTH,
        "rate_limit": RATE_LIMIT,
        "endpoints": {
            "correction": "POST /correct",
            "health": "GET /health",
        }
    }


@app.get("/health")
async def health():
    """Enhanced health check with uptime and system metrics"""
    
    # Calculate uptime
    uptime_seconds = time.time() - startup_time if startup_time else 0
    uptime_hours = uptime_seconds / 3600
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    
    # Determine status
    status = "healthy"
    if model is None:
        status = "unhealthy"
    elif cpu_percent > 90 or mem.percent > 90:
        status = "degraded"

    return {
        "status": status,
        "version": VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_hours": round(uptime_hours, 2),
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": "cpu",
        "max_tokens": MAX_TOKENS,
        "rate_limit": RATE_LIMIT,
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_available_gb": round(mem.available / (1024**3), 2),
            "memory_percent": mem.percent
        }
    }

@app.post("/correct", response_model=CorrectionResponse)
@limiter.limit(RATE_LIMIT)
async def correct_grammar(request: Request, data: CorrectionRequest):
    """
    Correct grammar in the provided text.
    Handles texts longer than 512 tokens using rolling window approach with adaptive merging.
    
    Rate limited to 100 requests per minute per IP address.
    """
    # Get client IP for logging
    client_ip = get_remote_address(request)
    request_start = time.time()
    
    # Log incoming request
    logger.info(f"[{client_ip}] Received correction request, text length: {len(data.text)} chars")
    
    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            raise ModelNotLoadedError("Grammar correction model is not loaded")
        
        # Explicit size check (redundant with Pydantic but good for clarity)
        if len(data.text) > MAX_TEXT_LENGTH:
            raise TextTooLargeError(f"Text length {len(data.text)} exceeds maximum {MAX_TEXT_LENGTH}")
        
        # Perform correction
        corrected, num_chunks = correct_text(
            data.text, 
            data.min_error_prob,
            data.n_iteration
        )
        
        processing_time = time.time() - request_start
        changes_made = corrected != data.text
        
        # Log successful completion
        logger.info(
            f"[{client_ip}] Correction completed: {processing_time:.2f}s, "
            f"{num_chunks} chunks, changes: {changes_made}"
        )
        
        return CorrectionResponse(
            original=data.text,
            corrected=corrected,
            num_chunks=num_chunks,
            processing_time_seconds=round(processing_time, 3),
            changes_made=changes_made
        )
    
    except (TextTooLargeError, ModelNotLoadedError, ModelProcessingError):
        # These are handled by exception handlers
        raise
    
    except ValueError as e:
        # Log and return 400 for validation errors
        logger.warning(f"[{client_ip}] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Log full stack trace, raise ModelProcessingError for generic message to client
        logger.error(f"[{client_ip}] Unexpected error during correction: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelProcessingError(f"Failed to process text: {type(e).__name__}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
    )
