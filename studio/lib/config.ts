// Public Supabase values — must come from env vars. No baked-in production
// credentials in source. When unset, the app runs in local mode.

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";

export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

// Escape hatch: set NEXT_PUBLIC_FORCE_LOCAL_MODE=1 to run without Supabase.
export const FORCE_LOCAL_MODE =
  process.env.NEXT_PUBLIC_FORCE_LOCAL_MODE === "1";
