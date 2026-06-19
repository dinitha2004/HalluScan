#!/usr/bin/env python
"""One-command local launch for the HallKing demo.

    python run_local.py             # build the frontend, start the backend, open http://localhost:8000
    python run_local.py --dev       # Vite hot-reload (:5173) + backend (:8000)
    python run_local.py --no-build  # reuse the existing frontend/dist (faster restart)
    python run_local.py --sentence-tag s1   # serve a specific Option-B head set (default s1)
    python run_local.py --hallushift-tag s2 # serve s1 heads but show the s2 HalluShift score (un-fused, display-only)
    python run_local.py --port 8080 --python "C:/path/to/se_probes_env/Scripts/python.exe"

The backend needs the se_probes_env (torch / transformers / bitsandbytes); on Windows it is auto-detected.
The model load takes ~1-2 min on first start — the page shows "backend offline" until /status flips online.
"""
import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
BACKEND = os.path.join(ROOT, "backend")
DIST = os.path.join(FRONTEND, "dist")
DEFAULT_WIN_PY = r"D:/Github Repositories/semantic-entropy-probes/se_probes_env/Scripts/python.exe"


def backend_python(arg):
    if arg:
        return arg
    if os.environ.get("HALLKING_PY"):
        return os.environ["HALLKING_PY"]
    if os.name == "nt" and os.path.exists(DEFAULT_WIN_PY):
        return DEFAULT_WIN_PY
    return sys.executable


def _ensure_npm():
    if os.name != "nt" and not shutil.which("npm"):
        sys.exit("npm not found on PATH — install Node.js, or use --no-build with an existing frontend/dist.")


def run_npm(args):
    """Run an npm command in the frontend dir (shell on Windows so npm.cmd resolves)."""
    _ensure_npm()
    if os.name == "nt":
        subprocess.run("npm " + " ".join(args), cwd=FRONTEND, check=True, shell=True)
    else:
        subprocess.run(["npm", *args], cwd=FRONTEND, check=True)


def popen_npm(args):
    if os.name == "nt":
        return subprocess.Popen("npm " + " ".join(args), cwd=FRONTEND, shell=True)
    return subprocess.Popen(["npm", *args], cwd=FRONTEND)


def build_frontend():
    if not os.path.isdir(os.path.join(FRONTEND, "node_modules")):
        print("[run_local] installing frontend deps (npm install) ...", flush=True)
        run_npm(["install"])
    print("[run_local] building frontend (npm run build) ...", flush=True)
    run_npm(["run", "build"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="Vite hot-reload (:5173) + backend (:8000)")
    ap.add_argument("--no-build", action="store_true", help="reuse existing frontend/dist")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--python", default=None, help="python for the backend (default: se_probes_env on Windows)")
    ap.add_argument("--sentence-tag", default="s1",
                    help="Option-B per-sentence head tag the demo serves (default s1; sets HALLKING_SENTENCE_TAG)")
    ap.add_argument("--hallushift-tag", default=None,
                    help="load ONLY the HalluShift head from this tag (e.g. s2) while SEP/TSV/fusion stay on "
                         "--sentence-tag; the per-sentence hallushift number reflects it (display-only, not fused). "
                         "Default: same as --sentence-tag.")
    args = ap.parse_args()

    py = backend_python(args.python)
    print(f"[run_local] backend python: {py}", flush=True)
    procs = []
    try:
        if args.dev:
            print(f"[run_local] DEV: Vite (:5173) + backend (:{args.port})", flush=True)
            procs.append(popen_npm(["run", "dev"]))
            url = "http://localhost:5173"
        else:
            if args.no_build:
                if not os.path.isdir(DIST):
                    sys.exit("--no-build given but frontend/dist is missing; run once without it first.")
            else:
                build_frontend()
            url = f"http://localhost:{args.port}"

        hs_tag = args.hallushift_tag or args.sentence_tag
        env = {**os.environ, "PORT": str(args.port), "PYTHONUNBUFFERED": "1",
               "HALLKING_SENTENCE_TAG": args.sentence_tag, "HALLKING_HALLUSHIFT_TAG": hs_tag}
        print(f"[run_local] starting backend (uvicorn) on :{args.port} "
              f"(sentence_tag={args.sentence_tag}, hallushift_tag={hs_tag}) ...", flush=True)
        procs.append(subprocess.Popen([py, "-m", "uvicorn", "app:app", "--host", "0.0.0.0",
                                       "--port", str(args.port)], cwd=BACKEND, env=env))

        def open_browser():
            time.sleep(3)
            print(f"[run_local] opening {url}  (model load takes ~1-2 min on first start)", flush=True)
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

        procs[-1].wait()
    except KeyboardInterrupt:
        print("\n[run_local] shutting down ...", flush=True)
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate()


if __name__ == "__main__":
    main()
