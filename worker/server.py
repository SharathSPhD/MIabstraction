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
# Every substrate the compiler may target here. Open-weight entries must be in the
# local HF cache; scratch entries carry no weights at all — the compiler makes them.
ALLOWED_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
    "google/gemma-2-9b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "nvidia/Nemotron-Mini-4B-Instruct",
    "gpt2",
}
SCRATCH_TARGETS = {"scratch(demo)", "scratch(flagship)"}
# A corpus must resolve to real files inside this repository. That is the whole
# rule: it stops a program reading /etc/passwd without deciding for the programmer
# which of the repo's corpora are respectable. The earlier prefix check refused
# examples/corpus/*.txt — a corpus that ships with the project — which is the
# compiler getting in the way of the thing it exists to do.
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


def _rpc(fn: str, payload: dict) -> None:
    """Fire-and-forget call to a Supabase definer function. The database trusts the
    worker via a shared secret checked inside the function, so only the public anon
    key ever lives in this process. A down database must never kill a build."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not (url and key):
        return
    import json as _json
    import urllib.request
    body = dict(payload)
    if fn.startswith("rpc_worker_"):
        body["p_secret"] = os.environ.get("LOOM_DB_SECRET", WORKER_KEY)
    req = urllib.request.Request(
        f"{url}/rest/v1/rpc/{fn}", method="POST",
        data=_json.dumps(body).encode(),
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def _auth(x_loom_key: str | None):
    if WORKER_KEY and x_loom_key != WORKER_KEY:
        raise HTTPException(401, "bad or missing X-Loom-Key")


class ExplainReq(BaseModel):
    source: str


class ChatReq(BaseModel):
    artifact: str                     # build uuid, or a named artifact directory
    message: str


class BuildReq(BaseModel):
    source: str
    target: str = "meta-llama/Llama-3.2-1B-Instruct"
    build_id: str | None = None       # supabase row id, if the app created one


def _validate(source: str, target: str) -> tuple:
    """Parse the program and enforce the worker's own safety rails: known target
    models only, corpora only from the repo's manifested domain data."""
    from loom.app.parse import AppSyntaxError, parse_program_text
    if target not in ALLOWED_MODELS | SCRATCH_TARGETS:
        raise HTTPException(
            422, f"target must be one of {sorted(ALLOWED_MODELS | SCRATCH_TARGETS)}")
    try:
        prog = parse_program_text(source)
    except AppSyntaxError as e:
        raise HTTPException(422, f"refused by the parser: {e}")
    app_ = next(iter(prog.apps.values()))
    from loom.app.capability import Kind
    for c in app_.of(Kind.KNOWLEDGE):
        pat = c.args.get("corpus", "")
        if not pat:
            continue
        if Path(pat).is_absolute() or ".." in Path(pat).parts:
            raise HTTPException(
                422, f"knows-from path {pat!r} must be a path inside the project")
        matches = [q for q in ROOT.glob(pat) if q.is_file()]
        if not matches:
            raise HTTPException(
                422, f"knows-from path {pat!r} matches no files in the project. "
                     f"Corpora that ship here: " +
                     ", ".join(sorted(
                         str(d.relative_to(ROOT))
                         for d in list(ROOT.glob("data/domains/*/corpus.txt"))
                         + list(ROOT.glob("examples/corpus/*.txt")))[:8]))
        if not all(str(q.resolve()).startswith(str(ROOT.resolve())) for q in matches):
            raise HTTPException(422, "knows-from must stay inside the project")
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
            "allowed_models": sorted(ALLOWED_MODELS),
            "scratch_targets": sorted(SCRATCH_TARGETS)}


@app.get("/isa")
def isa():
    """The instruction set, generated from the compiler's own tables."""
    from loom.isa import spec
    return spec()


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
    if req.build_id is not None:
        try:
            uuid.UUID(req.build_id)
        except ValueError:
            raise HTTPException(422, "build_id must be a UUID")
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


# ---- chat: talk to what was built ------------------------------------------------
_chat_cache: dict[str, object] = {}   # one loaded artifact at a time


def _artifact_dir(name: str) -> Path | None:
    """Resolve a chat target to an artifact directory that holds a PASSING report.
    A uuid means a studio build; anything else must match a named build directory.
    Never a path: the name is checked against the actual directory listing."""
    root = ROOT / "build"
    try:
        uuid.UUID(name)
        d = root / f"studio-{name[:8]}"
        cands = [d]
    except ValueError:
        cands = [d for d in root.iterdir() if d.is_dir() and d.name == name]
    for d in cands:
        rp = d / "report.json"
        if rp.exists():
            try:
                if json.loads(rp.read_text()).get("passed"):
                    return d
            except json.JSONDecodeError:
                pass
    return None


def _load_chat_model(d: Path):
    key = str(d)
    if key in _chat_cache:
        return _chat_cache[key]
    from loom.app.runtime import load_artifact
    for k in list(_chat_cache):
        m = _chat_cache.pop(k)
        try:
            m.detach()
            del m.module
        except Exception:
            pass
    import torch
    torch.cuda.empty_cache()
    lm = load_artifact(d, device="cuda")
    _chat_cache[key] = lm
    return lm


def _drop_chat_cache():
    for k in list(_chat_cache):
        m = _chat_cache.pop(k)
        try:
            m.detach()
            del m.module
        except Exception:
            pass
    import torch
    torch.cuda.empty_cache()


@app.get("/artifacts")
def artifacts(x_loom_key: str | None = Header(default=None)):
    """The artifacts a user can talk to: every build directory whose report passed."""
    _auth(x_loom_key)
    out = []
    root = ROOT / "build"
    for d in sorted(root.iterdir()):
        rp = d / "report.json"
        if not (d.is_dir() and rp.exists()):
            continue
        try:
            r = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("passed"):
            out.append({"name": d.name, "app": r.get("app"),
                        "base_model": r.get("base_model"),
                        "n_controls": r.get("n_controls_installed")})
    return {"artifacts": out}


@app.post("/chat")
def chat(req: ChatReq, x_loom_key: str | None = Header(default=None)):
    """One turn with a verified artifact — the model a user actually gets: base
    weights plus the adapters and calibrated controls the report verified."""
    _auth(x_loom_key)
    if len(req.message) > 2000:
        raise HTTPException(422, "message too long")
    d = _artifact_dir(req.artifact)
    if d is None:
        raise HTTPException(404, "no passing artifact by that name")
    with _chat_lock:
        lm = _load_chat_model(d)
        reply = lm.respond(req.message, max_new_tokens=200)
    return {"artifact": d.name, "reply": reply,
            "controls_active": len(getattr(lm, "controls", []))}


_chat_lock = threading.Lock()


def _event(sb, build_id: str, seq: int, stage: str, payload: dict) -> int:
    rec = {"stage": stage, "payload": payload, "ts": time.time()}
    _state.setdefault(build_id, {}).setdefault("events", []).append(rec)
    _rpc("rpc_worker_event", {"p_build": build_id, "p_seq": seq,
                              "p_stage": stage, "p_payload": payload})
    return seq + 1


def _run_one(job: dict) -> None:
    build_id, target = job["build_id"], job["target"]
    sb = None
    seq = 0
    with _lock:
        _state[build_id]["status"] = "running"
    _rpc("rpc_worker_update", {"p_build": build_id, "p_status": "running",
                               "p_report": None, "p_hf": None})
    _drop_chat_cache()                 # the GPU belongs to the build now
    seq = _event(sb, build_id, seq, "start",
                 {"target": target, "note": "program validated; compiler starting"})
    out_dir = ROOT / "build" / f"studio-{build_id[:8]}"
    src_path = out_dir / "program.loom"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path.write_text(job["source"])
    try:
        if target in SCRATCH_TARGETS:
            # No weights are downloaded on this path: the compiler chooses the
            # architecture, learns a tokenizer from the program's corpus, and
            # pretrains. Flagship effort is the long run.
            from loom.app.build_scratch import build as scratch_build
            effort = "flagship" if "flagship" in target else "demo"
            seq = _event(sb, build_id, seq, "substrate",
                         {"kind": "scratch", "effort": effort,
                          "note": "no downloaded weights; the model is made here"})
            report = scratch_build(str(src_path), str(out_dir), effort=effort,
                                   device="cuda")
        else:
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
        _rpc("rpc_worker_update", {"p_build": build_id, "p_status": status,
                                   "p_report": report, "p_hf": hf_repo})
    except Exception as e:
        with _lock:
            _state[build_id].update({"status": "error", "error": str(e)})
        _event(sb, build_id, seq, "error",
               {"error": str(e), "traceback": traceback.format_exc()[-2000:]})
        _rpc("rpc_worker_update", {"p_build": build_id, "p_status": "error",
                                   "p_report": {"error": str(e)}, "p_hf": None})


def _loop():
    while True:
        job = _q.get()
        try:
            _run_one(job)
        finally:
            _q.task_done()


threading.Thread(target=_loop, daemon=True).start()
