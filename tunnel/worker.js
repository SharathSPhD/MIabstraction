// Permanent front door for the ACD live demo.
//
//   Vercel (/api/dgx/*)  ->  https://acd-demo.sharath-sathish.workers.dev   [never changes]
//                              -> reads the CURRENT tunnel URL from KV
//                                 -> cloudflared quick tunnel -> DGX Spark :8787
//
// The point: a quick tunnel gets a new random hostname on every restart, and
// Vercel binds env vars at deploy time. Putting this Worker in front means the
// value Vercel holds is constant, and a tunnel restart is just a KV write --
// no redeploy, no downtime for the talk.
//
// NOTE vs the prabhasa worker: that one does `await resp.arrayBuffer()`, which
// buffers the whole body. That would destroy the SSE step-by-step stream this
// demo is built around, so here the upstream body is passed through untouched.

const ALLOWED = new Set(['health', 'prompts', 'graph', 'episode', 'steer']);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const seg = url.pathname.replace(/^\/+/, '').split('/')[0];

    if (seg === '' || seg === 'favicon.ico') {
      return json({ service: 'loom-studio gateway', backend: (await env.KV.get('gateway_url')) ? 'configured' : 'unset' }, 200);
    }
    if (!ALLOWED.has(seg)) return json({ error: 'not found' }, 404);

    const backend = await env.KV.get('gateway_url');
    if (!backend) return json({ error: 'offline', detail: 'gateway_url not set in KV' }, 503);

    const target = backend.replace(/\/$/, '') + url.pathname + url.search;

    // Forward only what the backend needs: its API key and the body type.
    const fwd = new Headers();
    const key = request.headers.get('x-loom-key');
    if (key) fwd.set('x-loom-key', key);
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      fwd.set('content-type', request.headers.get('content-type') || 'application/json');
    }

    let resp;
    try {
      resp = await fetch(target, {
        method: request.method,
        headers: fwd,
        body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
      });
    } catch (e) {
      // Tunnel is down / restarting. Surfaced as 503 so the Vercel proxy reports
      // OFFLINE and the demo page falls back to replay rather than erroring.
      return json({ error: 'offline', detail: String(e) }, 503);
    }

    // A dead/restarting quick tunnel does not throw -- Cloudflare's edge answers
    // with 530 (error 1033, "tunnel not found") as HTML. Translate that into the
    // same clean offline JSON as a thrown error, so the Vercel proxy and the demo
    // page see one consistent signal instead of an opaque HTML error page.
    if (resp.status === 530 || resp.status === 502 || resp.status === 504) {
      return json({ error: 'offline', detail: `tunnel unreachable (upstream ${resp.status})` }, 503);
    }

    const h = new Headers();
    const ct = resp.headers.get('content-type') || 'application/json';
    h.set('content-type', ct);
    if (ct.includes('text/event-stream')) {
      h.set('cache-control', 'no-cache');
      h.set('x-accel-buffering', 'no');
    }
    // Stream the body through as-is -- do NOT buffer, or SSE stops being live.
    return new Response(resp.body, { status: resp.status, headers: h });
  },
};

function json(o, status) {
  return new Response(JSON.stringify(o), { status, headers: { 'content-type': 'application/json' } });
}
