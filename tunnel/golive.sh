#!/usr/bin/env bash
# Bring the Loom Studio GPU worker online end to end (ACD pattern):
#   backend on :8788 -> cloudflared quick tunnel -> KV pointer -> permanent Worker URL.
#   ./tunnel/golive.sh          start everything and wire the chain
#   ./tunnel/golive.sh --stop   stop backend + tunnel and clear the pointer (OFFLINE)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=/tmp/loom-studio; mkdir -p "$RUN"
ENVF="$HOME/.loom-studio.env"
CF="$REPO/tunnel/cf.sh"

if [ "${1:-}" = "--stop" ]; then
  pkill -f "uvicorn worker.server:app" 2>/dev/null
  pkill -f "cloudflared tunnel --url http://localhost:8788" 2>/dev/null
  "$CF" clear || true
  echo "worker + tunnel stopped; gateway OFFLINE"
  exit 0
fi

if [ ! -f "$ENVF" ]; then
  echo "LOOM_WORKER_KEY=$(openssl rand -hex 24)" > "$ENVF"; chmod 600 "$ENVF"
  echo "generated $ENVF"
fi
# shellcheck disable=SC1090
source "$ENVF"
export LOOM_WORKER_KEY

cd "$REPO"
if ! pgrep -f "uvicorn worker.server:app" >/dev/null; then
  PYTHONPATH="$REPO" nohup "${VENV:-$HOME/projects/MIabstraction/.venv}/bin/uvicorn" worker.server:app \
    --port 8788 --host 127.0.0.1 > "$RUN/worker.log" 2>&1 &
  echo "worker starting (log: $RUN/worker.log)"
fi
until curl -s -m 2 localhost:8788/health >/dev/null; do sleep 2; done
echo "worker healthy"

if ! pgrep -f "cloudflared tunnel --url http://localhost:8788" >/dev/null; then
  nohup cloudflared tunnel --url http://localhost:8788 > "$RUN/tunnel.log" 2>&1 &
fi
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RUN/tunnel.log" | tail -1)
  [ -n "$URL" ] && break; sleep 2
done
[ -n "$URL" ] || { echo "no tunnel url appeared; see $RUN/tunnel.log" >&2; exit 1; }
echo "$URL" > "$RUN/tunnel_url.txt"
until curl -s -m 5 "$URL/health" >/dev/null; do sleep 2; done
"$CF" set "$URL"
echo "chain up: worker :8788 <- $URL <- $($CF url)"
