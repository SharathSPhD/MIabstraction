"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Zap } from "lucide-react";
import { Card, Badge, Section, Divider, EmptyState, Stat } from "@/components/ui";
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
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Builds</h1>
        <p className="text-muted">Loading...</p>
      </div>
    );
  }

  return (
    <div className="bg-paper">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <h1 className="font-serif text-5xl font-bold mb-4">Builds</h1>
        <p className="text-lg text-body max-w-2xl">
          Real builds that measure, search, and verify model behavior.
        </p>
      </div>

      <Divider className="max-w-7xl mx-auto px-6" />

      {/* Live section */}
      <Section
        className="max-w-7xl mx-auto px-6"
        title="Live Builds"
        eyebrow="Real-time"
        description="Builds currently running or recently completed."
      >
        {live.length === 0 ? (
          <EmptyState
            icon={<Zap className="w-12 h-12 text-muted" />}
            title="No live builds"
            description={
              SB_URL
                ? "No builds recorded yet. Submit one from the Studio to get started."
                : "Live build records need Supabase configured."
            }
          />
        ) : (
          <div className="space-y-2">
            {live.map((b) => (
              <Link key={b.id} href={`/builds/${b.id}`} className="group">
                <Card interactive className="flex items-center justify-between py-4 px-6">
                  <div>
                    <div className="font-mono text-sm font-semibold text-ink">
                      {b.id.slice(0, 12)}...
                    </div>
                    <p className="text-xs text-muted mt-1">
                      {b.target_model.split("/").pop()}
                    </p>
                  </div>
                  <div className="text-xs text-muted font-mono">
                    {new Date(b.created_at).toLocaleString()}
                  </div>
                  {b.hf_repo && (
                    <Badge variant="default" className="ml-4">
                      HF Repo
                    </Badge>
                  )}
                  <Badge
                    variant={
                      b.status === "passed"
                        ? "pass"
                        : b.status === "running"
                          ? "live"
                          : b.status === "failed" || b.status === "error"
                            ? "fail"
                            : "pending"
                    }
                  >
                    {b.status}
                  </Badge>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </Section>

      <Divider className="max-w-7xl mx-auto px-6" />

      {/* Verified showcase section */}
      <Section
        className="max-w-7xl mx-auto px-6"
        title="Verified Showcase"
        eyebrow="Benchmark Results"
        description="Stable builds that have passed all verification criteria."
      >
        {showcase.length === 0 ? (
          <EmptyState
            title="No verified builds yet"
            description="Builds will appear here once they complete verification."
          />
        ) : (
          <div className="grid gap-6">
            {showcase.map((build, idx) => (
              <Link key={idx} href={`/builds/replay-${idx}`} className="group">
                <Card elevated interactive>
                  <div className="flex items-start justify-between gap-6">
                    <div className="flex-1">
                      <h3 className="font-serif text-xl font-bold mb-2 group-hover:text-accent transition-colors">
                        {build.app}
                      </h3>
                      <p className="text-xs font-mono uppercase text-accent tracking-wider mb-4">
                        {build.base_model.split("/").pop()}
                      </p>
                      <div className="grid grid-cols-3 gap-6">
                        <Stat
                          label="Wall Clock"
                          value={build.wall_clock_s.toFixed(1)}
                          unit="s"
                        />
                        <Stat
                          label="Expectations Passed"
                          value={build.expectations_passed || 0}
                          unit={`/ ${build.expectations.length}`}
                        />
                        <Stat
                          label="Status"
                          value={build.passed ? "PASS" : "FAIL"}
                          trend={build.passed ? "up" : "down"}
                        />
                      </div>
                    </div>
                    <div className="flex-shrink-0 mt-2">
                      <Badge variant={build.passed ? "pass" : "fail"}>
                        {build.passed ? "PASS" : "FAIL"}
                      </Badge>
                    </div>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </Section>
    </div>
  );
}
