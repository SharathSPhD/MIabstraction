"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Loader } from "lucide-react";
import { Card, Button, Callout, SegmentedControl } from "@/components/ui";
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
      setTimeout(() => {
        const raw = searchParams.get("next") || "/";
        let next = "/";
        if (!/[\x00-\x1F\x7F]/.test(raw)) {
          try {
            const u = new URL(raw, window.location.origin);
            if (u.origin === window.location.origin)
              next = u.pathname + u.search + u.hash;
          } catch {}
        }
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
    <div className="min-h-screen flex items-center justify-center bg-paper px-6 py-12">
      <div className="w-full max-w-md animate-fade-up">
        <Card elevated className="p-8">
          <div className="text-center mb-8">
            <h1 className="font-serif text-4xl font-bold mb-2 text-ink">Loom</h1>
            <p className="text-body text-sm">Sign in to your account</p>
          </div>

          {error && (
            <Callout variant="error" className="mb-6">
              {error}
            </Callout>
          )}

          {success && (
            <Callout variant="success" className="mb-6">
              {success}
            </Callout>
          )}

          <SegmentedControl
            options={[
              { label: "Password", value: "password" },
              { label: "Magic Link", value: "magic" },
            ]}
            value={mode}
            onChange={(value) => {
              setMode(value as "password" | "magic");
              setError("");
              setSuccess("");
            }}
            className="mb-8 w-full"
          />

          {mode === "password" ? (
            <form onSubmit={handleSignInPassword} className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-ink mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full px-4 py-2 border border-hairline border-gray-300 rounded-lg bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1"
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
                  className="w-full px-4 py-2 border border-hairline border-gray-300 rounded-lg bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1"
                  required
                />
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full"
                size="lg"
              >
                {loading && <Loader className="w-4 h-4 animate-spin" />}
                Sign In
              </Button>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-hairline border-gray-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-white text-muted">Or create account</span>
                </div>
              </div>

              <Button
                type="button"
                variant="ghost"
                disabled={loading || !email || !password}
                onClick={handleSignUp}
                className="w-full"
                size="lg"
              >
                Create Account
              </Button>
            </form>
          ) : (
            <form onSubmit={handleSignInMagic} className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-ink mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full px-4 py-2 border border-hairline border-gray-300 rounded-lg bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1"
                  required
                />
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full"
                size="lg"
              >
                {loading && <Loader className="w-4 h-4 animate-spin" />}
                Send Magic Link
              </Button>

              <p className="text-xs text-muted text-center mt-4">
                We'll send you a link to sign in. No password needed.
              </p>
            </form>
          )}

          <div className="mt-8 text-center">
            <Link href="/" className="text-sm text-accent hover:underline font-medium">
              ← Back to Home
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-paper">
          <Card elevated className="p-8 text-center">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
            <p className="text-body">Loading...</p>
          </Card>
        </div>
      }
    >
      <LoginContent />
    </Suspense>
  );
}
