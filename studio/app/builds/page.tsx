"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Zap, CheckCircle2, XCircle, Clock } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { BuildReport } from "@/lib/types";

type LiveRow = {
  id: string;
  status: string;
  target_model: string;
  created_at: string;
  hf_repo: string | null;
};

const SB_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SB_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export default function BuildsPage() {
  const [showcase, setShowcase] = useState<BuildReport[]>([]);
  const [live, setLive] = useState<LiveRow[]>([]);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  useEffect(() => {
    if (!SB_URL || !SB_KEY) return;

    const pull = async () => {
      let authToken = `Bearer ${SB_KEY}`;
      if (supabase) {
        const { data } = await supabase.auth.getSession();
        if (data.session?.access_token) {
          authToken = `Bearer ${data.session.access_token}`;
        }
      }

      return fetch(
        `${SB_URL}/rest/v1/builds?select=id,status,target_model,created_at,hf_repo&order=created_at.desc&limit=20`,
        { headers: { apikey: SB_KEY, Authorization: authToken } }
      )
        .then((r) => (r.ok ? r.json() : []))
        .then((rows: LiveRow[]) => setLive(rows))
        .catch(() => {});
    };

    pull();
    const t = setInterval(pull, 10_000);
    return () => clearInterval(t);
  }, [supabase]);

  useEffect(() => {
    fetch("/api/showcase")
      .then((r) => r.json())
      .then((data) => {
        setShowcase(data || []);
        setLoading(false);
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Builds
          </h1>
          <p className="text-slate-400">Loading...</p>
        </div>
      </main>
    );
  }

  return (
    <main className="bg-night-950">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <h1 className="font-display text-5xl font-bold text-slate-100 mb-4">
          Builds
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl">
          Real builds that measure, search, and verify model behavior.
        </p>
      </div>

      <div className="border-t border-night-600/50" />

      {/* Live section */}
      {live.length > 0 && (
        <>
          <section className="max-w-7xl mx-auto px-6 py-16">
            <div className="mb-8">
              <h2 className="font-display text-3xl font-bold text-slate-100 mb-2">
                Live Builds
              </h2>
              <p className="text-slate-400">Real-time builds running now.</p>
            </div>
            <div className="space-y-3">
              {live.map((build) => (
                <Link key={build.id} href={`/builds/${build.id}`}>
                  <div className="card group cursor-pointer hover:bg-night-700/70 p-4 flex items-center justify-between">
                    <div className="flex-1">
                      <p className="font-mono text-sm text-slate-100 group-hover:text-gold-300 transition-colors">
                        {build.id}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        {build.target_model.split("/").pop()}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {build.status === "running" && (
                        <>
                          <Clock className="w-4 h-4 text-gold-400 animate-spin" />
                          <span className="text-xs text-gold-400 font-mono">Running</span>
                        </>
                      )}
                      {build.status === "completed" && (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span className="text-xs text-emerald-400 font-mono">Done</span>
                        </>
                      )}
                      {build.status === "failed" && (
                        <>
                          <XCircle className="w-4 h-4 text-rose-400" />
                          <span className="text-xs text-rose-400 font-mono">Failed</span>
                        </>
                      )}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>

          <div className="border-t border-night-600/50" />
        </>
      )}

      {/* Verified builds section */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="mb-8">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-2">
            Verified Builds
          </h2>
          <p className="text-slate-400">
            {showcase.length === 0
              ? "Builds are validated here."
              : `${showcase.length} prebuilt verified model${showcase.length !== 1 ? "s" : ""}.`}
          </p>
        </div>

        {showcase.length === 0 ? (
          <div className="card text-center py-12">
            <Zap className="w-8 h-8 text-slate-500 mx-auto mb-3" />
            <p className="text-slate-400">No verified builds yet.</p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {showcase.map((build, idx) => (
              <Link key={idx} href={`/builds/replay-${idx}`}>
                <div
                  className={`card group cursor-pointer hover:bg-night-700/70 p-6 h-full transition-colors border ${
                    build.passed
                      ? "border-emerald-500/30"
                      : "border-rose-500/30"
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-display text-lg font-bold text-slate-100 group-hover:text-gold-300 transition-colors">
                        {build.app}
                      </h3>
                      <p className="text-xs font-mono text-slate-400 mt-1">
                        {build.base_model.split("/").pop()}
                      </p>
                    </div>
                    {build.passed ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-5 h-5 text-rose-400 flex-shrink-0" />
                    )}
                  </div>

                  <div className="space-y-2 text-xs">
                    {build.wall_clock_s && (
                      <div className="flex justify-between text-slate-400">
                        <span>Time</span>
                        <span className="font-mono">
                          {(build.wall_clock_s / 60).toFixed(1)}m
                        </span>
                      </div>
                    )}
                    {(build.capabilities as any[])?.length > 0 && (
                      <div className="flex justify-between text-slate-400">
                        <span>Capabilities</span>
                        <span className="font-mono">
                          {(build.capabilities as any[]).length}
                        </span>
                      </div>
                    )}
                    {(build.expectations as any[])?.length > 0 && (
                      <div className="flex justify-between text-slate-400">
                        <span>Expectations</span>
                        <span className="font-mono">
                          {(build.expectations as any[]).length}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-4 border-t border-night-600/50">
                    {build.passed ? (
                      <span className="badge-emerald">Verified</span>
                    ) : (
                      <span className="badge-rose">Failed</span>
                    )}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>

      <div className="h-8" />
    </main>
  );
}
