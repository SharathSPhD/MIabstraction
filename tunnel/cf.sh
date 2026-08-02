#!/usr/bin/env bash
# Cloudflare side of the live demo: the permanent Worker front door + the KV
# pointer it reads to find the current tunnel.
#
#   ./tunnel/cf.sh deploy        # (re)deploy the Worker from worker.js
#   ./tunnel/cf.sh set <url>     # point the gateway at a tunnel URL
#   ./tunnel/cf.sh get           # what the gateway currently points at
#   ./tunnel/cf.sh clear         # unset the pointer (demo goes OFFLINE cleanly)
#   ./tunnel/cf.sh status        # gateway + backend health
#
# Credentials come from ~/.cloudflare-creds.env (CF_EMAIL / CF_KEY / CF_ACCT),
# which is chmod 600 and deliberately NOT in the repo.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREDS="${CREDS:-$HOME/.cloudflare-creds.env}"
WORKER_NAME="${WORKER_NAME:-loom-studio-gw}"
KV_TITLE="${KV_TITLE:-loom_gateway}"
WORKER_URL="${WORKER_URL:-https://loom-studio-gw.sharath-sathish.workers.dev}"

[ -f "$CREDS" ] || { echo "missing $CREDS (needs CF_EMAIL, CF_KEY, CF_ACCT)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CREDS"
: "${CF_EMAIL:?}"; : "${CF_KEY:?}"; : "${CF_ACCT:?}"

API="https://api.cloudflare.com/client/v4"
auth=(-H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_KEY")

# Resolve the KV namespace id by title so this works on a fresh account too.
kv_id() {
  curl -s -m 30 "$API/accounts/$CF_ACCT/storage/kv/namespaces?per_page=100" "${auth[@]}" \
    | python3 -c "
import json,sys
t='$KV_TITLE'
for n in (json.load(sys.stdin).get('result') or []):
    if n['title']==t: print(n['id']); break
"
}

ensure_kv() {
  local id; id=$(kv_id)
  if [ -z "$id" ]; then
    id=$(curl -s -m 30 -X POST "$API/accounts/$CF_ACCT/storage/kv/namespaces" "${auth[@]}" \
          -H 'content-type: application/json' --data "{\"title\":\"$KV_TITLE\"}" \
        | python3 -c "import json,sys;print((json.load(sys.stdin).get('result') or {}).get('id',''))")
    [ -n "$id" ] || { echo "could not create KV namespace $KV_TITLE" >&2; exit 1; }
    echo "created KV namespace $KV_TITLE ($id)" >&2
  fi
  printf '%s' "$id"
}


# Print the current pointer, or "(unset)" -- a missing key comes back as a
# Cloudflare error envelope rather than an empty body.
gateway_url() {
  local kv="$1"
  curl -s -m 30 "$API/accounts/$CF_ACCT/storage/kv/namespaces/$kv/values/gateway_url" "${auth[@]}" \
    | python3 -c "
import sys
v=sys.stdin.read().strip()
print('(unset)' if (not v or v.startswith('{')) else v)
"
}

case "${1:-status}" in

  deploy)
    KV=$(ensure_kv)
    meta=$(mktemp); trap 'rm -f "$meta"' EXIT
    cat > "$meta" <<EOF
{"main_module":"worker.js","compatibility_date":"2026-01-01",
 "bindings":[{"type":"kv_namespace","name":"KV","namespace_id":"$KV"}]}
EOF
    echo "deploying Worker '$WORKER_NAME' (KV $KV)"
    curl -s -m 60 -X PUT "$API/accounts/$CF_ACCT/workers/scripts/$WORKER_NAME" "${auth[@]}" \
      -F "metadata=@$meta;type=application/json" \
      -F "worker.js=@$REPO/tunnel/worker.js;type=application/javascript+module" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('deploy success:', d['success'])
if not d['success']: print(d.get('errors')); sys.exit(1)
" || exit 1
    curl -s -m 30 -X POST "$API/accounts/$CF_ACCT/workers/scripts/$WORKER_NAME/subdomain" "${auth[@]}" \
      -H 'content-type: application/json' --data '{"enabled":true}' >/dev/null
    echo "live at $WORKER_URL"
    ;;

  set)
    URL="${2:-}"
    [ -n "$URL" ] || { echo "usage: cf.sh set <tunnel-url>" >&2; exit 1; }
    KV=$(ensure_kv)
    curl -s -m 30 -X PUT "$API/accounts/$CF_ACCT/storage/kv/namespaces/$KV/values/gateway_url" \
      "${auth[@]}" --data "$URL" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('gateway_url ->', '$URL' if d['success'] else 'FAILED '+str(d.get('errors')))
sys.exit(0 if d['success'] else 1)
"
    ;;

  get)
    KV=$(ensure_kv)
    echo "gateway_url = $(gateway_url "$KV")"
    ;;

  clear)
    # Drop the pointer so the Worker answers a clean 503 "offline" immediately,
    # instead of waiting to discover the tunnel is gone.
    KV=$(ensure_kv)
    curl -s -m 30 -X DELETE "$API/accounts/$CF_ACCT/storage/kv/namespaces/$KV/values/gateway_url" \
      "${auth[@]}" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('gateway_url cleared' if d['success'] else 'FAILED '+str(d.get('errors')))
sys.exit(0 if d['success'] else 1)
"
    ;;

  status)
    KV=$(ensure_kv)
    echo "worker    : $WORKER_URL"
    echo "kv        : $KV_TITLE ($KV)"
    echo "points at : $(gateway_url "$KV")"
    echo "health    : $(curl -s -m 30 "$WORKER_URL/health" | head -c 200)"
    ;;

  *) echo "usage: cf.sh {deploy|set <url>|get|clear|status}" >&2; exit 1;;
esac
