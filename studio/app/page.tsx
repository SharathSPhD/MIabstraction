"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import type { BuildReport } from "@/lib/types";

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
    <main className="bg-night-950">
      {/* Hero section */}
      <div className="max-w-5xl mx-auto px-6 py-20 md:py-32">
        <div className="animate-fade-up">
          <h1 className="font-display text-6xl md:text-7xl font-bold tracking-tight mb-6 leading-tight text-slate-100">
            A programming language for language models.
          </h1>
          <p className="text-lg text-slate-300 mb-12 max-w-2xl leading-relaxed">
            Write what your model must know, what it must do, and what it must refuse. The compiler lowers your program through four proven layers, measures real internal behavior, searches the parameter space, and verifies every expectation.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link href="/studio">
              <button className="btn-gold gap-2">
                Open Editor
                <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
            <Link href="/builds">
              <button className="btn-ghost gap-2">
                View Verified Builds
                <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
          </div>
        </div>
      </div>

      <div className="border-t border-night-600/50" />

      {/* Four-layer pipeline */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <div className="mb-12">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-2">The Abstraction Stack</h2>
          <p className="text-slate-400">Four verified layers, each empirically validated.</p>
        </div>
        <div className="space-y-3">
          {[
            {
              level: "L3",
              name: "Program",
              desc: "Your source in Loom: knows/speaks/always/never/refuses clauses + tune/effort knobs.",
              link: "/language",
            },
            {
              level: "L2",
              name: "Capability Graph",
              desc: "Parsed capabilities with chosen strategies, measurement plans, and verification rules.",
              link: "/compiler",
            },
            {
              level: "L1",
              name: "Mech-Interp IR",
              desc: "The instruction set: read/amplify/suppress/install/monitor on measured internal objects.",
              link: "/compiler",
            },
            {
              level: "L0",
              name: "Substrate",
              desc: "Base model, adapters, controls installed, calibration results, the compiled artifact.",
              link: "/compiler",
            },
          ].map((layer, idx) => (
            <Link key={layer.level} href={layer.link}>
              <div className="card group cursor-pointer hover:bg-night-700/70">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 rounded-lg bg-gold-600/20 border border-gold-600/50 flex items-center justify-center font-display font-bold text-gold-300 text-lg">
                      {layer.level}
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="font-display text-lg font-bold text-slate-100 group-hover:text-gold-300 transition-colors mb-1">
                      {layer.name}
                    </h3>
                    <p className="text-slate-400 text-sm">{layer.desc}</p>
                  </div>
                  <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-gold-300 transition-colors flex-shrink-0" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <div className="border-t border-night-600/50" />

      {/* Substrate cards */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <div className="mb-12">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-2">Two Substrates</h2>
          <p className="text-slate-400">Choose where to build.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            {
              name: "From Scratch",
              desc: "No downloaded weights. The compiler chooses the architecture, learns a tokenizer, pretrains on your data.",
              icon: "🏗️",
            },
            {
              name: "Open-Weight Adapters",
              desc: "Start from a frozen base. The compiler installs steering controls and grafts verified circuits into weights.",
              icon: "🔌",
            },
          ].map((sub) => (
            <div key={sub.name} className="card p-6">
              <div className="text-3xl mb-3">{sub.icon}</div>
              <h3 className="font-display text-lg font-bold text-slate-100 mb-2">
                {sub.name}
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                {sub.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      <div className="border-t border-night-600/50" />

      {/* Showcase */}
      {!loading && showcase.length > 0 && (
        <section className="max-w-5xl mx-auto px-6 py-16">
          <div className="mb-12">
            <h2 className="font-display text-3xl font-bold text-slate-100 mb-2">Verified Builds</h2>
            <p className="text-slate-400">Prebuilt demo programs, fully measured.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-4 overflow-auto">
            {showcase.slice(0, 6).map((build, idx) => (
              <Link key={idx} href={`/builds/replay-${idx}`}>
                <div className="card group cursor-pointer hover:bg-night-700/70 h-full">
                  <div className="flex flex-col justify-between h-full">
                    <div>
                      <h4 className="font-display font-bold text-slate-100 text-lg group-hover:text-gold-300 transition-colors mb-1">
                        {build.app}
                      </h4>
                      <p className="text-xs text-slate-400 font-mono">
                        {build.base_model.split("/").pop()}
                      </p>
                    </div>
                    <div className="mt-4">
                      {build.passed ? (
                        <span className="badge-emerald">Verified</span>
                      ) : (
                        <span className="badge-rose">Failed</span>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      <div className="border-t border-night-600/50" />

      {/* Science summary */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <div className="card">
          <div className="flex items-start gap-4">
            <div className="text-2xl">📊</div>
            <div>
              <h3 className="font-display text-lg font-bold text-slate-100 mb-2">
                Five Preregistered Hypotheses
              </h3>
              <p className="text-slate-400 mb-4">
                The compiler's abstraction is empirically validated. Every hypothesis has a posterior probability and a published result artifact.
              </p>
              <Link href="/science">
                <button className="btn-ghost text-sm gap-1">
                  See all claims and controls
                  <ArrowRight className="w-3 h-3" />
                </button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      <div className="h-8" />
    </main>
  );
}
