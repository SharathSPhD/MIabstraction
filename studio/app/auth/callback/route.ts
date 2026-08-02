// Supabase PKCE callback: the email confirmation / magic-link lands here with
// a `?code=` that must be exchanged for a session (cookies) server-side.
// Without this the user is stranded on `/?code=...` with no session.

import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { NextRequest, NextResponse } from "next/server";
import { SUPABASE_URL, SUPABASE_ANON_KEY, FORCE_LOCAL_MODE } from "@/lib/config";

export const dynamic = "force-dynamic";

type CookieToSet = { name: string; value: string; options?: CookieOptions };

function safeNext(raw: string | null): string {
  // Only allow same-origin relative paths to avoid open-redirects.
  if (raw && raw.startsWith("/") && !raw.startsWith("//")) return raw;
  return "/";
}

/** Origin honoring the reverse proxy (Vercel) forwarded host, when present. */
function resolveOrigin(req: NextRequest): string {
  const forwardedHost = req.headers.get("x-forwarded-host");
  if (forwardedHost) {
    const proto = req.headers.get("x-forwarded-proto") || "https";
    return `${proto}://${forwardedHost}`;
  }
  return req.nextUrl.origin;
}

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const origin = resolveOrigin(req);
  const code = searchParams.get("code");
  const next = safeNext(searchParams.get("next"));
  const authError =
    searchParams.get("error_description") || searchParams.get("error");

  const url = FORCE_LOCAL_MODE ? "" : SUPABASE_URL;
  const anonKey = FORCE_LOCAL_MODE ? "" : SUPABASE_ANON_KEY;

  if (code && url && anonKey) {
    // Bind the Supabase client's cookie writes to the exact response we return,
    // so the session cookies are guaranteed to be set on the redirect.
    let response = NextResponse.redirect(`${origin}${next}`);
    const supabase = createServerClient(url, anonKey, {
      cookies: {
        getAll() {
          return req.cookies.getAll();
        },
        setAll(cookiesToSet: CookieToSet[]) {
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options)
          );
        },
      },
    });

    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (user) return response;
    }
    return NextResponse.redirect(
      `${origin}/login?error=${encodeURIComponent(
        error?.message || "auth_callback_failed"
      )}`
    );
  }

  return NextResponse.redirect(
    `${origin}/login?error=${encodeURIComponent(
      authError || "auth_callback_failed"
    )}`
  );
}
