"""HallKing FastAPI backend — ONE Instruct model serves SEP + HalluShift + TSV + fusion.

Identical code path locally (run_local.py) and on Colab (notebook 7 + ngrok). When a built React
frontend exists at ../frontend/dist it is served too, so local = a single origin
(http://localhost:8000) with no CORS / backend-URL fiddling.

Run (in se_probes_env):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import json
import asyncio
import contextlib

# Local LLaMA cache only on Windows; on Colab/Linux leave HF_HOME to the environment.
if os.name == "nt":
    os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from pipeline import HallKingPipeline
from fusion import FusionModel
from claim_filter import claim_detector

DATASET = os.environ.get("HALLKING_DATASET", "triviaqa")
USE_CLAIM_FILTER = os.environ.get("HALLKING_CLAIM_FILTER", "1") not in ("0", "false", "False")

# The live demo serves the Option-B per-sentence heads (tools/train_claim_heads.py →
# artifacts/*/*_sentence_<tag>.*, models/fusion_claim_<tag>.pkl) in the sentence regime. Default tag "s1";
# override with HALLKING_SENTENCE_TAG. Option A (short-QA) is retired from serving — it over-flagged long
# answers — though its training code/artifacts remain for the notebooks.
SENTENCE_TAG = os.environ.get("HALLKING_SENTENCE_TAG", "s1")
# answer length: room for a full multi-sentence answer the demo defragments + scores per sentence.
MAX_NEW_TOKENS = int(os.environ.get("HALLKING_MAX_NEW_TOKENS", "256"))

# Diagnostic: log every query's per-sentence + aggregate scores. Terminal is always on; the file
# (one JSON object per query) is HALLKING_SCORE_LOG (default data/demo_scores.jsonl; set empty to disable).
SCORE_LOG_PATH = os.environ.get("HALLKING_SCORE_LOG", os.path.join(ROOT, "data", "demo_scores.jsonl"))

STATE = {"pipe": None, "error": None}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"[HallKing] loading pipeline (dataset={DATASET}, sentence regime tag={SENTENCE_TAG}, "
              f"single Instruct model) ...", flush=True)
        pipe = HallKingPipeline(dataset=DATASET, separate_tsv=False, sentence_tag=SENTENCE_TAG).load()
        fusion_path = os.path.join(ROOT, "models", f"fusion_claim_{SENTENCE_TAG}.pkl")
        if os.path.exists(fusion_path):
            pipe.fusion = FusionModel.load(fusion_path)
            print(f"[HallKing] fusion loaded: {os.path.relpath(fusion_path, ROOT)}", flush=True)
            # Apply the fusion's calibrated thresholds (the sentence fusion lives on a small score scale, so
            # the risk.py 0.50/0.74 defaults would never flag anything — they MUST come from this JSON).
            thr_path = os.path.join(ROOT, "models", f"fusion_claim_{SENTENCE_TAG}_thresholds.json")
            if os.path.exists(thr_path):
                with open(thr_path) as f:
                    thr = json.load(f)
                pipe.t_med, pipe.t_high = float(thr["t_med"]), float(thr["t_high"])
                print(f"[HallKing] thresholds loaded: t_med={pipe.t_med} t_high={pipe.t_high}", flush=True)
            else:
                print(f"[HallKing] WARNING: thresholds not found at {os.path.relpath(thr_path, ROOT)} — "
                      f"using risk.py defaults (t_med={pipe.t_med} t_high={pipe.t_high}); flags may be off",
                      flush=True)
        else:
            print(f"[HallKing] WARNING: fusion not found at {fusion_path} — /infer returns raw detectors only",
                  flush=True)
        if USE_CLAIM_FILTER:
            claim_detector.load_nli_model()   # DeBERTa NLI; falls back to regex-only on failure
        STATE["pipe"] = pipe
        print("[HallKing] ready.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE["error"] = str(e)
        print(f"[HallKing] STARTUP ERROR: {e}", flush=True)
    yield
    STATE["pipe"] = None


app = FastAPI(title="HallKing", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    max_new_tokens: int | None = None


@app.get("/status")
def status():
    pipe = STATE["pipe"]
    return {
        "model_loaded": pipe is not None,
        "fusion_loaded": pipe is not None and pipe.fusion is not None,
        "dataset": DATASET,
        "regime": "sentence",
        "sentence_tag": SENTENCE_TAG,
        "t_med": pipe.t_med if pipe is not None else None,
        "t_high": pipe.t_high if pipe is not None else None,
        "claim_detector": "nli" if claim_detector.nli_available else "regex-only",
        "model_name": pipe.engine.model_name if pipe is not None else None,
        "error": STATE["error"],
    }


def _log_scores(question: str, result: dict):
    """Diagnostic dump of one query's per-sentence + aggregate scores -> terminal (always) and
    SCORE_LOG_PATH jsonl (if set). Never raises into /infer."""
    try:
        agg = result.get("aggregate", {}) or {}
        sents = result.get("sentences", []) or []
        ans = result.get("answer", "")
        f = lambda v: "  None" if v is None else f"{float(v):+.3f}"
        print("\n" + "=" * 72, flush=True)
        print(f"[scores] Q: {question}", flush=True)
        print(f"[scores] A: {ans}", flush=True)
        print(f"[scores] aggregate: fused={f(agg.get('fused'))} tier={agg.get('tier')} "
              f"label={agg.get('label')!r} flagged={agg.get('n_flagged')}/{agg.get('n_sentences')} "
              f"| mean sep={f(agg.get('sep_entropy'))} hs={f(agg.get('hallushift'))} "
              f"tsv={f(agg.get('tsv_margin'))}", flush=True)
        for i, s in enumerate(sents):
            print(f"[scores]   s{i} [{str(s.get('tier')):7}] fused={f(s.get('fused'))} "
                  f"sep={f(s.get('sep_entropy'))} hs={f(s.get('hallushift'))} tsv={f(s.get('tsv_margin'))} "
                  f"claim={s.get('is_claim')} | {str(s.get('sentence',''))[:80]}", flush=True)
        print("=" * 72, flush=True)
        if SCORE_LOG_PATH:
            import time
            rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question,
                   "answer": ans, "aggregate": agg, "sentences": sents}
            with open(SCORE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[scores] logging failed (non-fatal): {e}", flush=True)


@app.post("/infer")
async def infer(req: QueryRequest):
    pipe = STATE["pipe"]
    if pipe is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Model not loaded yet")
    mnt = req.max_new_tokens or MAX_NEW_TOKENS
    # model.generate is blocking -> run off the event loop
    result = await asyncio.to_thread(pipe.score_with_sentences, req.question, mnt, USE_CLAIM_FILTER)
    _log_scores(req.question, result)
    return result


# --- serve the built React frontend (single-origin local + Colab). MUST come after the API routes. ---
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    print(f"[HallKing] serving frontend from {FRONTEND_DIST}", flush=True)
    _assets = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        fp = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(fp):
            return FileResponse(fp)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    print(f"[HallKing] no frontend build at {FRONTEND_DIST} — API-only mode", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
