"""HalluScan FastAPI backend — selectable models, each serving SEP + HalluShift + TSV + fusion.

A MODELS registry exposes one or more chat models (default: Llama-3.2-1B + Llama-3.1-8B), each with its own
per-sentence heads + fusion. By default only ONE model is resident in VRAM (the lazy default loads the 1B
fast); switching via the dropdown / `/select-model` offloads it and loads the other. `HALLKING_EAGER=1` keeps
all loaded (instant switch); `HALLKING_PREFETCH=1` downloads the others to the HF cache in the background so a
later swap is a quick disk->VRAM load (used on the Colab deploy).

Identical code path locally (run_local.py) and on Colab (notebook 7 + ngrok). When a built React frontend
exists at ../frontend/dist it is served too, so local = a single origin (http://localhost:8000) with no CORS.

Run (in se_probes_env):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import json
import asyncio
import contextlib
import threading

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
from engine import DEFAULT_MODEL_NAME

DATASET = os.environ.get("HALLKING_DATASET", "triviaqa")
USE_CLAIM_FILTER = os.environ.get("HALLKING_CLAIM_FILTER", "1") not in ("0", "false", "False")

# The live demo serves the Option-B per-sentence heads (tools/train_claim_heads.py →
# artifacts/*/*_sentence_<tag>.*, models/fusion_claim_<tag>.pkl) in the sentence regime. Default tag "s1";
# override with HALLKING_SENTENCE_TAG. Option A (short-QA) is retired from serving — it over-flagged long
# answers — though its training code/artifacts remain for the notebooks.
SENTENCE_TAG = os.environ.get("HALLKING_SENTENCE_TAG", "s1")
# Observe a retrained HalluShift head live WITHOUT changing the served (TSV-only) fusion: this loads ONLY
# the HalluShift head from hal_det_sentence_<HALLUSHIFT_TAG>_*, leaving SEP/TSV/fusion on SENTENCE_TAG. The
# per-sentence `hallushift` number in /infer then reflects the new head (display/diagnostic, not fused).
HALLUSHIFT_TAG = os.environ.get("HALLKING_HALLUSHIFT_TAG", SENTENCE_TAG)
# answer length: room for a full multi-sentence answer the demo defragments + scores per sentence.
MAX_NEW_TOKENS = int(os.environ.get("HALLKING_MAX_NEW_TOKENS", "256"))

# Multi-model registry: each key is a selectable model with its OWN per-sentence head tag + fusion. The 8B
# keeps the exact deployed config (SENTENCE_TAG / HALLUSHIFT_TAG / fusion_claim_<tag>); the 1B uses tag "l1b"
# (trained in notebooks/13). HALLKING_MODELS picks which to load (comma-sep, e.g. "8b" if VRAM is tight);
# HALLKING_DEFAULT_MODEL is used when /infer omits "model". Inference adapters auto-match each model's layers.
MODELS = {
    "8b": {"label": "Llama-3.1-8B", "model_name": DEFAULT_MODEL_NAME, "tag": SENTENCE_TAG, "hs_tag": HALLUSHIFT_TAG},
    "1b": {"label": "Llama-3.2-1B", "model_name": "meta-llama/Llama-3.2-1B-Instruct", "tag": "l1b", "hs_tag": "l1b"},
}
ENABLED = [k.strip() for k in os.environ.get("HALLKING_MODELS", "8b,1b").split(",") if k.strip() in MODELS]
DEFAULT_MODEL = os.environ.get("HALLKING_DEFAULT_MODEL", "8b")   # 8B is the demo default (the UI suggestion chips are tuned for it)
if DEFAULT_MODEL not in ENABLED:
    DEFAULT_MODEL = ENABLED[0] if ENABLED else "1b"

# Loading strategy (VRAM vs switch speed):
#   default (lazy)      -> only ONE model is resident; switching offloads it and loads the other.
#   HALLKING_EAGER=1    -> load ALL enabled into VRAM at startup (instant switch; only if VRAM is ample).
#   HALLKING_PREFETCH=1 -> lazy VRAM, but after the default is serving a background thread DOWNLOADS the other
#                          models to the HF cache, so a later switch is a quick disk->VRAM load (no network).
#                          (The Colab deploy sets this; locally it's a no-op since weights are already cached.)
def _truthy(v):
    return str(v).strip().lower() not in ("", "0", "false", "no")
EAGER = _truthy(os.environ.get("HALLKING_EAGER", "0"))
PREFETCH = _truthy(os.environ.get("HALLKING_PREFETCH", "0"))

# Diagnostic: log every query's per-sentence + aggregate scores. Terminal is always on; the file
# (one JSON object per query) is HALLKING_SCORE_LOG (default data/demo_scores.jsonl; set empty to disable).
SCORE_LOG_PATH = os.environ.get("HALLKING_SCORE_LOG", os.path.join(ROOT, "data", "demo_scores.jsonl"))

# pipes: key -> resident HallKingPipeline. current: which key is the active/served model. loading: a key
# mid-swap (for the UI to show "switching…"). prefetch: a key being downloaded in the background.
STATE = {"pipes": {}, "current": None, "loading": None, "prefetch": None, "error": None}
_swap_lock = threading.Lock()   # serialize model swaps so two switches can't race the VRAM


def _load_pipe(key: str, cfg: dict) -> HallKingPipeline:
    """Build one model's pipeline (model + SEP/HalluShift/TSV) and attach its fusion + calibrated thresholds.
    Mirrors the original single-model load, parameterised by the registry entry (model_name / tag / hs_tag)."""
    tag, hs_tag = cfg["tag"], cfg["hs_tag"]
    print(f"[HallKing] loading '{key}' ({cfg['model_name']}, sentence tag={tag}, hs_tag={hs_tag}) ...", flush=True)
    pipe = HallKingPipeline(model_name=cfg["model_name"], dataset=DATASET, separate_tsv=False,
                            sentence_tag=tag, hs_tag=hs_tag).load()
    fusion_path = os.path.join(ROOT, "models", f"fusion_claim_{tag}.pkl")
    if os.path.exists(fusion_path):
        pipe.fusion = FusionModel.load(fusion_path)
        print(f"[HallKing] '{key}' fusion loaded: {os.path.relpath(fusion_path, ROOT)}", flush=True)
        # Apply the fusion's calibrated thresholds (the sentence fusion lives on a small score scale, so
        # the risk.py 0.50/0.74 defaults would never flag anything — they MUST come from this JSON).
        thr_path = os.path.join(ROOT, "models", f"fusion_claim_{tag}_thresholds.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                thr = json.load(f)
            pipe.t_med, pipe.t_high = float(thr["t_med"]), float(thr["t_high"])
            print(f"[HallKing] '{key}' thresholds loaded: t_med={pipe.t_med} t_high={pipe.t_high}", flush=True)
        else:
            print(f"[HallKing] WARNING: '{key}' thresholds not found at {os.path.relpath(thr_path, ROOT)} — "
                  f"using risk.py defaults (t_med={pipe.t_med} t_high={pipe.t_high}); flags may be off",
                  flush=True)
    else:
        print(f"[HallKing] WARNING: '{key}' fusion not found at {fusion_path} — /infer returns raw detectors only",
              flush=True)
    return pipe


def _swap_to(key: str):
    """Make `key` the resident/active model. Lazy: unload the current pipe (free VRAM) then load `key`.
    Eager: every pipe stays resident, so this just flips the `current` pointer. Serialized by _swap_lock."""
    if key not in MODELS:
        raise ValueError(f"unknown model '{key}'")
    with _swap_lock:
        if STATE["current"] == key and STATE["pipes"].get(key) is not None:
            return
        STATE["loading"] = key
        try:
            if not EAGER:
                for k in list(STATE["pipes"]):
                    if k != key:
                        print(f"[HallKing] unloading '{k}' to free VRAM", flush=True)
                        try:
                            STATE["pipes"][k].engine.unload()   # frees the model + empties the CUDA cache
                        except Exception as e:
                            print(f"[HallKing] unload '{k}' failed (ignored): {e}", flush=True)
                        STATE["pipes"].pop(k, None)
            if STATE["pipes"].get(key) is None:
                STATE["pipes"][key] = _load_pipe(key, MODELS[key])
            STATE["current"] = key
            print(f"[HallKing] active model = '{key}'", flush=True)
        finally:
            STATE["loading"] = None


def _prefetch_models(keys):
    """Download (but do NOT VRAM-load) the given models' weights to the HF cache, so a later lazy swap is a
    quick disk->VRAM load instead of a multi-GB network download. Runs in a background thread on the deploy."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[HallKing] prefetch skipped (huggingface_hub unavailable): {e}", flush=True)
        return
    for k in keys:
        mn = MODELS[k]["model_name"]
        try:
            STATE["prefetch"] = k
            print(f"[HallKing] prefetching '{k}' ({mn}) to the HF cache ...", flush=True)
            snapshot_download(repo_id=mn)
            print(f"[HallKing] prefetched '{k}'", flush=True)
        except Exception as e:
            print(f"[HallKing] prefetch '{k}' failed (will download on first switch): {e}", flush=True)
    STATE["prefetch"] = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[HallKing] startup: enabled={ENABLED} default={DEFAULT_MODEL} eager={EAGER} prefetch={PREFETCH} "
          f"dataset={DATASET}", flush=True)
    try:
        if EAGER:
            for key in ENABLED:
                STATE["pipes"][key] = _load_pipe(key, MODELS[key])
            STATE["current"] = DEFAULT_MODEL if DEFAULT_MODEL in STATE["pipes"] else next(iter(STATE["pipes"]), None)
        else:
            _swap_to(DEFAULT_MODEL)   # load ONLY the default into VRAM (fast startup)
            if PREFETCH:
                others = [k for k in ENABLED if k != DEFAULT_MODEL]
                if others:
                    threading.Thread(target=_prefetch_models, args=(others,), daemon=True).start()
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE["error"] = str(e)
        print(f"[HallKing] STARTUP ERROR: {e}", flush=True)
    if USE_CLAIM_FILTER:
        try:
            claim_detector.load_nli_model()   # DeBERTa NLI (shared, never swapped); regex fallback on failure
        except Exception as e:
            print(f"[HallKing] NLI load failed (regex fallback): {e}", flush=True)
    print(f"[HallKing] ready. active={STATE['current']} resident={list(STATE['pipes'])}", flush=True)
    yield
    STATE["pipes"] = {}; STATE["current"] = None


app = FastAPI(title="HalluScan", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    max_new_tokens: int | None = None
    model: str | None = None   # registry key (e.g. "8b" / "1b"); omitted -> DEFAULT_MODEL


@app.get("/status")
def status():
    pipes = STATE["pipes"]
    cur = STATE["current"]
    cur_pipe = pipes.get(cur)
    available = [{"key": k, "label": MODELS[k]["label"], "loaded": k in pipes,
                  "fusion_loaded": (k in pipes and pipes[k].fusion is not None)} for k in ENABLED]
    return {
        # model_loaded/fusion_loaded reflect the ACTIVE model so the existing Navbar online dot keeps working.
        "model_loaded": cur_pipe is not None,
        "fusion_loaded": cur_pipe is not None and cur_pipe.fusion is not None,
        "dataset": DATASET,
        "regime": "sentence",
        "default_model": DEFAULT_MODEL,
        "current_model": cur,           # which model is resident/active right now
        "loading": STATE["loading"],    # a key mid-swap (UI shows "switching…"), else null
        "prefetch": STATE["prefetch"],  # a key being downloaded in the background, else null
        "available_models": available,
        "sentence_tag": SENTENCE_TAG,
        "hallushift_tag": HALLUSHIFT_TAG,
        "t_med": cur_pipe.t_med if cur_pipe is not None else None,
        "t_high": cur_pipe.t_high if cur_pipe is not None else None,
        "claim_detector": "nli" if claim_detector.nli_available else "regex-only",
        "model_name": cur_pipe.engine.model_name if cur_pipe is not None else None,
        "error": STATE["error"],
    }


class SelectRequest(BaseModel):
    model: str


@app.post("/select-model")
async def select_model(req: SelectRequest):
    """Pre-load a model so the dropdown switch is ready before the user sends. Returns immediately:
    'ready' if it's already resident, else 'loading' (a background swap starts; the UI polls /status)."""
    key = req.model
    if key not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model '{key}' (have {list(MODELS)})")
    if STATE["current"] == key and STATE["pipes"].get(key) is not None:
        return {"status": "ready", "current_model": key}
    if STATE["pipes"].get(key) is not None:           # already resident (eager) -> just flip the pointer
        await asyncio.to_thread(_swap_to, key)
        return {"status": "ready", "current_model": key}
    if STATE["loading"] is not None:
        return {"status": "loading", "loading": STATE["loading"]}
    STATE["loading"] = key                              # optimistic flag so /status reflects it immediately
    threading.Thread(target=_swap_to, args=(key,), daemon=True).start()
    return {"status": "loading", "loading": key}


def _log_scores(question: str, result: dict, model: str = ""):
    """Diagnostic dump of one query's per-sentence + aggregate scores -> terminal (always) and
    SCORE_LOG_PATH jsonl (if set). Never raises into /infer."""
    try:
        agg = result.get("aggregate", {}) or {}
        sents = result.get("sentences", []) or []
        ans = result.get("answer", "")
        f = lambda v: "  None" if v is None else f"{float(v):+.3f}"
        print("\n" + "=" * 72, flush=True)
        print(f"[scores] model={model} Q: {question}", flush=True)
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
            rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "question": question,
                   "answer": ans, "aggregate": agg, "sentences": sents}
            with open(SCORE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[scores] logging failed (non-fatal): {e}", flush=True)


@app.post("/infer")
async def infer(req: QueryRequest):
    key = req.model or STATE["current"] or DEFAULT_MODEL
    if key not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model '{key}' (have {list(MODELS)})")
    pipe = STATE["pipes"].get(key)
    if pipe is None:
        # requested model isn't resident -> lazy-swap to it (unless a DIFFERENT swap is already mid-flight)
        if STATE["loading"] is not None and STATE["loading"] != key:
            raise HTTPException(status_code=503,
                                detail=f"model '{STATE['loading']}' is loading; try again in a moment")
        await asyncio.to_thread(_swap_to, key)   # blocking load off the event loop
        pipe = STATE["pipes"].get(key)
    if pipe is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or f"model '{key}' not loaded yet")
    mnt = req.max_new_tokens or MAX_NEW_TOKENS
    # model.generate is blocking -> run off the event loop
    result = await asyncio.to_thread(pipe.score_with_sentences, req.question, mnt, USE_CLAIM_FILTER)
    result["model"] = key
    _log_scores(req.question, result, key)
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
