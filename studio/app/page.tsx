"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { BuildReport } from "@/lib/types";

const PROGRAMS = [
  {
    name: "Clinic",
    tagline: "Medical reference",
    guardrail: "Refuses to diagnose",
  },
  {
    name: "Counsel",
    tagline: "Legal research",
    guardrail: "Never gives advice",
  },
  {
    name: "Desk",
    tagline: "Financial analysis",
    guardrail: "Refuses recommendations",
  },
  {
    name: "Foreman",
    tagline: "Safety protocols",
    guardrail: "Never authorizes work",
  },
  {
    name: "Stylist",
    tagline: "Editorial house style",
    guardrail: "Refuses living-author imitation",
  },
];

export default function Home() {
  const [showcase, setShowcase] = useState<BuildReport[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/showcase")
      .then((r) => r.json())
      .then((data) => {
        setShowcase(data || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  return (
    <div className="bg-paper">
      {/* Hero section */}
      <div className="max-w-4xl mx-auto px-6 py-16 md:py-24">
        <h1 className="font-serif text-5xl md:text-6xl font-bold tracking-tight mb-6">
          Write consequences for your LLM.
        </h1>
        <p className="text-lg text-body mb-8 max-w-2xl leading-relaxed">
          The compiler measures real capabilities, searches the parameter space, and verifies your expectations. Not simulation—actual behavioral proof.
        </p>
        <div className="flex flex-wrap gap-4">
          <Link href="/studio" className="button-primary">
            Open Editor
          </Link>
          <Link href="/builds" className="button-secondary">
            View Verified Builds
          </Link>
        </div>
      </div>

      <div className="divider max-w-7xl mx-auto px-6" />

      {/* Programs section */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="section-heading">Real Programs</div>
        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
          {PROGRAMS.map((prog) => (
            <Link key={prog.name} href={`/studio?example=${prog.name.toLowerCase()}`}>
              <div className="card-hover p-4 cursor-pointer h-full">
                <h3 className="font-serif text-lg font-bold mb-2">
                  {prog.name}
                </h3>
                <p className="text-xs text-muted mb-3 font-mono uppercase">
                  {prog.tagline}
                </p>
                <p className="text-sm text-body">
                  {prog.guardrail}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Verified builds section */}
      {!loading && showcase.length > 0 && (
        <>
          <div className="divider max-w-7xl mx-auto px-6" />
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="section-heading">Verified Builds</div>
            <div className="grid auto-cols-max gap-3 overflow-x-auto pb-4">
              {showcase.slice(0, 6).map((build, idx) => (
                <Link key={idx} href={`/builds/replay-${idx}`}>
                  <div className="card-hover px-4 py-3 cursor-pointer whitespace-nowrap">
                    <div className="font-serif font-bold mb-1">{build.app}</div>
                    <div className="flex gap-2 items-center">
                      <span className="text-xs font-mono text-muted">
                        {build.base_model.split("/").pop()}
                      </span>
                      {build.passed ? (
                        <span className="badge-pass">PASS</span>
                      ) : (
                        <span className="badge-fail">FAIL</span>
                      )}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
