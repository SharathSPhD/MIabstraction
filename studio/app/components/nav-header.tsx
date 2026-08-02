"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { LogOut, Loader } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { GpuStatus } from "./gpu-status";

export function NavHeader() {
  const [user, setUser] = useState<{ email: string } | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }

    const checkSession = async () => {
      const {
        data: { user: authUser },
      } = await supabase.auth.getUser();

      if (authUser) {
        setUser({ email: authUser.email || "" });

        // Check if user is admin
        try {
          const { data, error } = await supabase
            .from("app_admins")
            .select("email")
            .eq("email", authUser.email);

          if (!error && data && data.length > 0) {
            setIsAdmin(true);
          }
        } catch {
          // Ignore errors — RLS will deny if not admin
        }
      }
      setLoading(false);
    };

    checkSession();
  }, [supabase]);

  const handleSignOut = async () => {
    if (!supabase) return;
    await supabase.auth.signOut();
    setUser(null);
    setIsAdmin(false);
    window.location.href = "/";
  };

  return (
    <nav className="border-b border-hairline border-gray-300">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="no-underline">
          <div className="font-serif text-xl font-bold text-ink hover:text-accent transition-colors">
            Loom
          </div>
        </Link>
        <div className="flex items-center gap-8">
          <div className="flex gap-6 text-sm">
            <Link href="/studio" className="text-ink hover:text-accent transition-colors">
              Studio
            </Link>
            <Link href="/builds" className="text-ink hover:text-accent transition-colors">
              Builds
            </Link>
            <Link href="/use" className="text-ink hover:text-accent transition-colors">
              Use
            </Link>
          </div>
          <div className="border-l border-hairline border-gray-300 pl-6">
            {loading ? (
              <Loader className="w-4 h-4 animate-spin text-muted" />
            ) : user ? (
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <p className="text-xs font-mono text-muted">{user.email}</p>
                  <div className="flex gap-2 items-center">
                    {isAdmin && (
                      <span className="badge-live text-xs px-2 py-0.5">Admin</span>
                    )}
                  </div>
                </div>
                <button
                  onClick={handleSignOut}
                  className="text-muted hover:text-ink transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <GpuStatus />
                <div className="mt-3">
                  <Link
                    href="/login"
                    className="text-xs text-accent hover:underline"
                  >
                    Sign in
                  </Link>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
