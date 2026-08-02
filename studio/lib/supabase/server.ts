import { createServerClient, type CookieOptions } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { SUPABASE_URL, SUPABASE_ANON_KEY, FORCE_LOCAL_MODE } from "../config";

type CookieToSet = { name: string; value: string; options?: CookieOptions };

const url = FORCE_LOCAL_MODE ? "" : SUPABASE_URL;
const anonKey = FORCE_LOCAL_MODE ? "" : SUPABASE_ANON_KEY;

export const supabaseConfigured = Boolean(url && anonKey);

/**
 * Cookie-based Supabase client for server components and route handlers.
 * Returns null when Supabase is not configured (local mode).
 */
export function createClient(): SupabaseClient | null {
  if (!url || !anonKey) return null;
  const cookieStore = cookies();
  return createServerClient(url, anonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet: CookieToSet[]) {
        try {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options)
          );
        } catch {
          // Called from a Server Component — safe to ignore when middleware
          // is refreshing sessions.
        }
      },
    },
  });
}
