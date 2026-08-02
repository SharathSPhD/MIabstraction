"""The Loom Studio GPU worker — the execution half of the web app.

Runs on the GB10 next to the compiler. Exposed to the deployed app through the ACD
tunnel chain (cloudflared quick tunnel behind a permanent Cloudflare Worker whose KV
holds the current tunnel URL). Three jobs:

  /explain   parse + plan + search-space for a program, no GPU spent — the editor's
             live feedback, including parse-time refusals as diagnostics
  /build     enqueue a build; the single worker loop executes builds one at a time
             with loom.app.build_open.build, writing stage events as they happen
  /health    who am I, what GPU, how deep is the queue

Programs are DATA. They are parsed by the Loom parser and never executed as code; a
program the parser refuses returns its refusal as a diagnostic, which is the language
working, not failing. Builds stream through Supabase when configured (build_events
rows) and always into local artifact directories. Passing builds are pushed to
Hugging Face when HF_PUSH=1.

Run:  LOOM_WORKER_KEY=... SUPABASE_URL=... SUPABASE_SERVICE_KEY=... \
      .venv/bin/uvicorn worker.server:app --port 8788
"""
from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

WORKER_KEY = os.environ.get("LOOM_WORKER_KEY", "")
ALLOWED_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
}
ALLOWED_CORPORA_PREFIX = "data/domains/"
MAX_QUEUE = 4

app = FastAPI(title="loom-studio-worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LOOM_ALLOWED_ORIGINS",
                                 "http://localhost:3000").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

_q: "queue.Queue[dict]" = queue.Queue()
_state: dict[str, dict] = {}          # build_id -> status record
_lock = threading.Lock()


def _sb():
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        return None
    from supabase import create_client
    return create_client(url, key)


def _auth(x_loom_key: str | None):
    if WORKER_KEY and x_loom_key != WORKER_KEY:
        raise HTTPException(401, "bad or missing X-Loom-Key")


class ExplainReq(BaseModel):
    source: str


class BuildReq(BaseModel):
    source: str
    target: str = "meta-llama/Llama-3.2-1B-Instruct"
    build_id: str | None = None       # supabase row id, if the app created one


def _validate(source: str, target: str) -> tuple:
    """Parse the program and enforce the worker's own safety rails: known target
    models only, corpora only from the repo's manifested domain data."""
    from loom.app.parse import AppSyntaxError, parse_program_text
    if target not in ALLOWED_MODELS:
        raise HTTPException(422, f"target must be one of {sorted(ALLOWED_MODELS)}")
    try:
        prog = parse_program_text(source)
    except AppSyntaxError as e:
        raise HTTPException(422, f"refused by the parser: {e}")
    app_ = next(iter(prog.apps.values()))
    from loom.app.capability import Kind
    for c in app_.of(Kind.KNOWLEDGE):
        pat = c.args.get("corpus", "")
        if not pat.startswith(ALLOWED_CORPORA_PREFIX):
            raise HTTPException(
                422, f"knows-from path {pat!r} is outside {ALLOWED_CORPORA_PREFIX}; "
                     "the studio builds only against the manifested domain corpora")
    return prog, app_


@app.get("/health")
def health():
    import torch
    return {"status": "ok", "service": "loom-studio-worker",
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "none"),
            "queue_depth": _q.qsize(),
            "running": next((b for b, s in _state.items()
                             if s.get("status") == "running"), None),
            "allowed_models": sorted(ALLOWED_MODELS)}


@app.post("/explain")
def explain(req: ExplainReq, x_loom_key: str | None = Header(default=None)):
    _auth(x_loom_key)
    from loom.app import cli as app_cli
    prog, app_ = _validate(req.source, next(iter(ALLOWED_MODELS)))
    buf = io.StringIO()
    # cmd_explain prints; capture it. It reads model config metadata only (no GPU).
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".loom", delete=False) as f:
        f.write(req.source)
        path = f.name
    try:
        with redirect_stdout(buf):
            rc = app_cli.cmd_explain(path)
    finally:
        os.unlink(path)
    return {"ok": rc == 0, "explain": buf.getvalue()}


@app.post("/build")
def submit_build(req: BuildReq, x_loom_key: str | None = Header(default=None)):
    _auth(x_loom_key)
    if _q.qsize() >= MAX_QUEUE:
        raise HTTPException(429, "build queue is full; try again later")
    _validate(req.source, req.target)
    build_id = req.build_id or str(uuid.uuid4())
    with _lock:
        _state[build_id] = {"status": "queued", "target": req.target,
                            "submitted": time.time()}
    _q.put({"build_id": build_id, "source": req.source, "target": req.target})
    return {"build_id": build_id, "status": "queued", "position": _q.qsize()}


@app.get("/build/{build_id}")
def build_status(build_id: str, x_loom_key: str | None = Header(default=None)):
    _auth(x_loom_key)
    s = _state.get(build_id)
    if not s:
        raise HTTPException(404, "unknown build id")
    return {k: v for k, v in s.items() if k != "report"} | {
        "report_ready": "report" in s}


@app.get("/build/{build_id}/report")
def build_report(build_id: str, x_loom_key: str | None = Header(default=None)):
    _auth(x_loom_key)
    s = _state.get(build_id)
    if not s or "report" not in s:
        raise HTTPException(404, "no report for this build id")
    return s["report"]


def _event(sb, build_id: str, seq: int, stage: str, payload: dict) -> int:
    rec = {"stage": stage, "payload": payload, "ts": time.time()}
    _state.setdefault(build_id, {}).setdefault("events", []).append(rec)
    if sb:
        try:
            sb.table("build_events").insert(
                {"build_id": build_id, "seq": seq, "stage": stage,
                 "payload": payload}).execute()
        except Exception:
            pass                       # supabase down must not kill a build
    return seq + 1


def _run_one(job: dict) -> None:
    build_id, target = job["build_id"], job["target"]
    sb = _sb()
    seq = 0
    with _lock:
        _state[build_id]["status"] = "running"
    if sb:
        try:
            sb.table("builds").update({"status": "running"}) \
              .eq("id", build_id).execute()
        except Exception:
            pass
    seq = _event(sb, build_id, seq, "start",
                 {"target": target, "note": "program validated; compiler starting"})
    out_dir = ROOT / "build" / f"studio-{build_id[:8]}"
    src_path = out_dir / "program.loom"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path.write_text(job["source"])
    try:
        from loom.app.build_open import build as loom_build
        report = loom_build(str(src_path), target, str(out_dir),
                            device="cuda", verify=True)
        seq = _event(sb, build_id, seq, "verified", {
            "passed": report.get("passed"),
            "expectations": report.get("expectations", []),
            "wall_clock_s": report.get("wall_clock_s")})
        hf_repo = None
        if report.get("passed") and os.environ.get("HF_PUSH") == "1":
            try:
                from worker.hf_publish import publish
                hf_repo = publish(out_dir, report)
                seq = _event(sb, build_id, seq, "published", {"hf_repo": hf_repo})
            except Exception as e:
                seq = _event(sb, build_id, seq, "publish_failed",
                             {"error": str(e)})
        status = "passed" if report.get("passed") else "failed"
        with _lock:
            _state[build_id].update({"status": status, "report": report,
                                     "hf_repo": hf_repo})
        if sb:
            try:
                sb.table("builds").update(
                    {"status": status, "report": report, "hf_repo": hf_repo}) \
                  .eq("id", build_id).execute()
            except Exception:
                pass
    except Exception as e:
        with _lock:
            _state[build_id].update({"status": "error", "error": str(e)})
        _event(sb, build_id, seq, "error",
               {"error": str(e), "traceback": traceback.format_exc()[-2000:]})
        if sb:
            try:
                sb.table("builds").update(
                    {"status": "error", "report": {"error": str(e)}}) \
                  .eq("id", build_id).execute()
            except Exception:
                pass


def _loop():
    while True:
        job = _q.get()
        try:
            _run_one(job)
        finally:
            _q.task_done()


threading.Thread(target=_loop, daemon=True).start()
