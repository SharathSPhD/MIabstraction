"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ExternalLink } from "lucide-react";
import type { BuildReport } from "@/lib/types";

export default function BuildsPage() {
  const [builds, setBuilds] = useState<BuildReport[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/showcase")
      .then((r) => r.json())
      .then((data) => {
        setBuilds(data);
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
        <h2 className="font-serif text-3xl font-bold mb-8">Builds</h2>
        <p className="text-muted">Loading...</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="flex items-center justify-between mb-8">
        <h2 className="font-serif text-3xl font-bold">Builds</h2>
        <p className="text-sm text-muted">
          {builds.length === 0
            ? "No builds available"
            : `${builds.length} committed build${builds.length !== 1 ? "s" : ""}`}
        </p>
      </div>

      {builds.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-muted mb-4">
            When builds complete, they appear here.
          </p>
          <p className="text-sm">
            In development mode, shared build reports from the repo appear below
            as REPLAY mode.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {builds.map((build) => (
            <Link key={build.id || build.app} href={`/builds/${build.id || `replay-${builds.indexOf(build)}`}`}>
              <div className="card hover:shadow-md transition-shadow cursor-pointer">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-serif font-bold text-lg mb-2">
                      {build.app}
                    </h3>
                    <p className="text-sm text-body mb-3">
                      {build.base_model}
                    </p>
                    <div className="flex gap-4 text-xs text-muted">
                      <span>
                        {build.expectations_passed || 0} of{" "}
                        {build.expectations.length} expectations passed
                      </span>
                      <span>{build.wall_clock_s.toFixed(1)}s wall clock</span>
                    </div>
                  </div>
                  <div className="text-right">
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
  );
}
