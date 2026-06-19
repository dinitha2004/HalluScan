
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import os
from engine import engine
from claim_filter import claim_detector
import uvicorn
import contextlib

# Lifecycle manager for startup/shutdown
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        engine.load_probes()
        claim_detector.load_nli_model()
        success = engine.load_model()
        if not success:
            print("WARNING: Model failed to load on startup.")
    except Exception as e:
        print(f"Startup error: {e}")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS (Allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SentenceDetail(BaseModel):
    text: str
    confidence: float = 1.0
    entropy: float = 0.0
    accuracy_prob: float = 1.0
    is_claim: bool = True
    source: str = "sentencizer"

class QueryResponse(BaseModel):
    answer: str
    confidence: float = 1.0
    slt_confidence: float = 0.0
    sentence_details: list[SentenceDetail] = []
    error: str = None

@app.get("/status")
def get_status():
    return {
        "model_loaded": engine.model is not None,
        "probes_loaded": engine.probes is not None,
        "model_name": engine.model_name,
        "probe_name": engine.selected_probe['name'] if engine.selected_probe else "None",
        "claim_detector": "nli" if claim_detector.nli_available else "regex-only",
    }

class SetModelRequest(BaseModel):
    model_name: str

@app.post("/set_model")
def set_model(request: SetModelRequest):
    if engine.model_name == request.model_name and engine.model is not None:
         return {"status": "Already loaded"}
         
    # Unload existing
    engine.unload_model()
    
    # Load new
    success = engine.load_model(request.model_name)
    if not success:
         raise HTTPException(status_code=500, detail="Failed to load model")
         
    return {"status": "Model changed", "model_name": request.model_name}

@app.post("/infer", response_model=QueryResponse)
async def infer(request: QueryRequest):
    if not engine.model:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    # Run inference in threadpool to avoid blocking async loop (since model.generate is blocking)
    # Using asyncio.to_thread for PyTorch operations
    result = await asyncio.to_thread(engine.generate_response, request.question)
    
    if "error" in result:
         return QueryResponse(answer="", confidence=1.0, slt_confidence=0.0, error=result["error"])
         
    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        slt_confidence=result.get("slt_confidence", 0.0),
        sentence_details=result.get("sentence_details", [])
    )

# --- Serve built React frontend (production) ---
# This must come AFTER all API routes so they take priority
FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.exists(FRONTEND_BUILD):
    print(f"[Static] Serving frontend from {FRONTEND_BUILD}")
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_BUILD, "assets")), name="static-assets")
    
    # Catch-all: serve index.html for SPA routing
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Try to serve the exact file first (e.g. favicon, vite.svg)
        file_path = os.path.join(FRONTEND_BUILD, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html (SPA client-side routing)
        return FileResponse(os.path.join(FRONTEND_BUILD, "index.html"))
else:
    print(f"[Static] No frontend build found at {FRONTEND_BUILD} — API-only mode")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

