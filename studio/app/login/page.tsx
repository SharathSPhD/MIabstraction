"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { AlertCircle, CheckCircle2, Loader } from "lucide-react";
import { createClient } from "@/lib/supabase/client";

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [mode, setMode] = useState<"password" | "magic">("password");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const supabase = createClient();
  const urlError = searchParams.get("error");

  useEffect(() => {
    if (urlError) {
      setError(decodeURIComponent(urlError));
    }
  }, [urlError]);

  const handleSignInPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    if (!supabase) {
      setError("Supabase not configured");
      setLoading(false);
      return;
    }

    const { error: err } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (err) {
      setError(err.message);
    } else {
      setSuccess("Signed in successfully!");
      // Full navigation, deliberately: router.push would reuse the client router
      // cache, which may hold the pre-login redirect for guarded routes.
      setTimeout(() => {
        const raw = searchParams.get("next") || "/";
        const next =
          raw.startsWith("/") && !raw.startsWith("//") && !raw.startsWith("/\\")
            ? raw
            : "/";
        window.location.assign(next);
      }, 400);
    }
    setLoading(false);
  };

  const handleSignInMagic = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    if (!supabase) {
      setError("Supabase not configured");
      setLoading(false);
      return;
    }

    const origin = typeof window !== "undefined" ? window.location.origin : "";
    const { error: err } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: `${origin}/auth/callback`,
      },
    });

    if (err) {
      setError(err.message);
    } else {
      setSuccess("Check your email for the magic link!");
    }
    setLoading(false);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    if (!supabase) {
      setError("Supabase not configured");
      setLoading(false);
      return;
    }

    const { error: err } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${typeof window !== "undefined" ? window.location.origin : ""}/auth/callback`,
      },
    });

    if (err) {
      setError(err.message);
    } else {
      setSuccess("Account created! Check your email to confirm.");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-paper px-6">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="font-serif text-3xl font-bold mb-2">Loom</h1>
          <p className="text-body">Sign in to your account</p>
        </div>

        {/* Mode toggle */}
        <div className="flex gap-2 mb-6 bg-panel p-1 rounded border border-hairline border-gray-300">
          <button
            onClick={() => {
              setMode("password");
              setError("");
              setSuccess("");
            }}
            className={`flex-1 px-4 py-2 rounded text-sm font-medium transition-colors ${
              mode === "password"
                ? "bg-accent text-white"
                : "text-ink hover:bg-gray-100"
            }`}
          >
            Password
          </button>
          <button
            onClick={() => {
              setMode("magic");
              setError("");
              setSuccess("");
            }}
            className={`flex-1 px-4 py-2 rounded text-sm font-medium transition-colors ${
              mode === "magic"
                ? "bg-accent text-white"
                : "text-ink hover:bg-gray-100"
            }`}
          >
            Magic Link
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border-l-2 border-red-400 rounded flex gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-red-900">{error}</p>
            </div>
          </div>
        )}

        {/* Success message */}
        {success && (
          <div className="mb-4 p-4 bg-green-50 border-l-2 border-green-400 rounded flex gap-3">
            <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-green-900">{success}</p>
            </div>
          </div>
        )}

        {mode === "password" ? (
          <form onSubmit={handleSignInPassword} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-ink mb-2">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-4 py-2 border border-hairline border-gray-300 rounded bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-ink mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-2 border border-hairline border-gray-300 rounded bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 px-4 bg-accent text-white rounded font-medium hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center justify-center gap-2"
            >
              {loading && <Loader className="w-4 h-4 animate-spin" />}
              Sign In
            </button>

            <div className="text-center text-sm">
              <p className="text-body">
                Don't have an account?{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode("password");
                    setError("");
                    setSuccess("Create account below");
                  }}
                  className="text-accent hover:underline font-medium"
                >
                  Create one
                </button>
              </p>
            </div>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-hairline border-gray-300" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-paper text-muted">Or create account</span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleSignUp}
              disabled={loading || !email || !password}
              className="w-full py-2 px-4 bg-panel text-ink rounded font-medium border border-hairline border-gray-300 hover:bg-gray-100 disabled:opacity-50 transition-colors"
            >
              Create Account
            </button>
          </form>
        ) : (
          <form onSubmit={handleSignInMagic} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-ink mb-2">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-4 py-2 border border-hairline border-gray-300 rounded bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 px-4 bg-accent text-white rounded font-medium hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center justify-center gap-2"
            >
              {loading && <Loader className="w-4 h-4 animate-spin" />}
              Send Magic Link
            </button>

            <p className="text-xs text-muted text-center">
              We'll send you a link to sign in. No password needed.
            </p>
          </form>
        )}

        <div className="mt-8 text-center">
          <Link href="/" className="text-sm text-accent hover:underline">
            ← Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-paper">
          <div className="card p-8 text-center">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
            <p className="text-body">Loading...</p>
          </div>
        </div>
      }
    >
      <LoginContent />
    </Suspense>
  );
}
