"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
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

  useEffect(() => {
    if (!SB_URL || !SB_KEY) return;
    const pull = () =>
      fetch(
        `${SB_URL}/rest/v1/builds?select=id,status,target_model,created_at,hf_repo&order=created_at.desc&limit=20`,
        { headers: { apikey: SB_KEY, Authorization: `Bearer ${SB_KEY}` } }
      )
        .then((r) => (r.ok ? r.json() : []))
        .then((rows: LiveRow[]) => setLive(rows))
        .catch(() => {});
    pull();
    const t = setInterval(pull, 10_000);
    return () => clearInterval(t);
  }, []);

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
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Builds</h1>
        <p className="text-muted">Loading...</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <h1 className="font-serif text-4xl font-bold mb-4">Builds</h1>
      <p className="text-body mb-12">
        Real builds that measure, search, and verify model behavior.
      </p>

      {/* Live section */}
      <div className="mb-16">
        <h2 className="font-serif text-2xl font-bold mb-6 pb-4 border-b border-hairline border-gray-300">
          Live
        </h2>
        {live.length === 0 ? (
          <div className="card text-center py-12">
            <p className="text-muted text-sm">
              {SB_URL
                ? "No builds recorded yet — submit one from the Studio."
                : "Live build records need Supabase configured."}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {live.map((b) => (
              <Link key={b.id} href={`/builds/${b.id}`}>
                <div className="card-hover cursor-pointer flex items-center justify-between gap-4 py-3">
                  <div className="font-mono text-sm">{b.id.slice(0, 8)}</div>
                  <div className="text-sm text-muted flex-1">
                    {b.target_model.split("/").pop()}
                  </div>
                  <div className="text-xs text-muted font-mono">
                    {new Date(b.created_at).toLocaleString()}
                  </div>
                  {b.hf_repo ? (
                    <span className="text-xs font-mono text-verified">HF ↗</span>
                  ) : null}
                  <span
                    className={
                      b.status === "passed"
                        ? "badge-replay text-verified"
                        : b.status === "running"
                          ? "badge-live"
                          : b.status === "failed" || b.status === "error"
                            ? "badge-replay text-refusal"
                            : "badge-replay"
                    }
                  >
                    {b.status}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Verified showcase section */}
      <div>
        <h2 className="font-serif text-2xl font-bold mb-6 pb-4 border-b border-hairline border-gray-300">
          Verified Showcase
        </h2>

        {showcase.length === 0 ? (
          <div className="card text-center py-12">
            <p className="text-muted text-sm">
              No verified builds available yet.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {showcase.map((build, idx) => (
              <Link key={idx} href={`/builds/replay-${idx}`}>
                <div className="card-hover cursor-pointer">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h3 className="font-serif text-lg font-bold mb-2">
                        {build.app}
                      </h3>
                      <p className="text-xs font-mono text-muted mb-3 uppercase">
                        {build.base_model.split("/").pop()}
                      </p>
                      <div className="flex flex-wrap gap-4 text-sm text-body">
                        <div>
                          <span className="stat-label block">Wall Clock</span>
                          <span className="font-mono">
                            {build.wall_clock_s.toFixed(1)}s
                          </span>
                        </div>
                        <div>
                          <span className="stat-label block">
                            Expectations
                          </span>
                          <span className="font-mono">
                            {build.expectations_passed || 0}/{build.expectations.length}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      {build.passed ? (
                        <div className="badge-pass">PASS</div>
                      ) : (
                        <div className="badge-fail">FAIL</div>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
