"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import { Card, Button, Badge, Section, Divider } from "@/components/ui";
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

const LAYERS = [
  {
    level: "L3",
    name: "Program",
    description: "Your Loom source: expectations and capability specs that the model must satisfy.",
  },
  {
    level: "L2",
    name: "Capability Graph",
    description: "Parsed capabilities with chosen strategies and how each will be measured and verified.",
  },
  {
    level: "L1",
    name: "Mech-Interp IR",
    description: "Autotune search results: parameter gaps, trial data, behavioral gate margins, and steering controls.",
  },
  {
    level: "L0",
    name: "Substrate",
    description: "Base model, adapters, controls installed, calibration results, and the Hugging Face repo.",
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
      {/* Hero section - Magazine quality */}
      <div className="max-w-5xl mx-auto px-6 py-20 md:py-32">
        <div className="animate-fade-up">
          <h1 className="font-serif text-6xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
            Write consequences for your LLM.
          </h1>
          <p className="text-xl text-body mb-12 max-w-2xl leading-relaxed">
            The compiler measures real capabilities, searches the parameter space, and verifies your expectations. Not simulation—actual behavioral proof.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link href="/studio">
              <Button size="lg" className="gap-2">
                Open Editor
                <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link href="/builds">
              <Button size="lg" variant="ghost">
                View Verified Builds
              </Button>
            </Link>
          </div>
        </div>
      </div>

      <Divider className="max-w-7xl mx-auto px-6" />

      {/* Abstraction layers - Connected steps with arrows */}
      <Section
        className="max-w-7xl mx-auto px-6"
        title="The Abstraction Stack"
        eyebrow="Four Verified Layers"
        description="From specification through measured behavior: each layer builds on the previous, and each is empirically validated."
      >
        <div className="space-y-4">
          {LAYERS.map((layer, idx) => (
            <div key={layer.level} className="flex items-start gap-6">
              {/* Level marker */}
              <div className="flex-shrink-0">
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-lg bg-accent text-paper flex items-center justify-center font-serif font-bold text-lg">
                    {layer.level}
                  </div>
                  {idx < LAYERS.length - 1 && (
                    <div className="w-0.5 h-12 bg-accent/30 my-2" />
                  )}
                </div>
              </div>

              {/* Content */}
              <Card className="flex-1" elevated>
                <h3 className="font-serif text-lg font-bold mb-2">{layer.name}</h3>
                <p className="text-sm text-body leading-relaxed">{layer.description}</p>
              </Card>
            </div>
          ))}
        </div>
      </Section>

      <Divider className="max-w-7xl mx-auto px-6" />

      {/* Programs section - Program cards with guardrails */}
      <Section
        className="max-w-7xl mx-auto px-6"
        title="Real Programs"
        eyebrow="Verified Examples"
        description="Five guardrailed programs, each measured to refuse specific unsafe capabilities."
      >
        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
          {PROGRAMS.map((prog) => (
            <Link
              key={prog.name}
              href={`/studio?example=${prog.name.toLowerCase()}`}
              className="group"
            >
              <Card elevated interactive className="h-full flex flex-col">
                <h3 className="font-serif text-lg font-bold mb-3 group-hover:text-accent transition-colors">
                  {prog.name}
                </h3>
                <p className="text-xs font-mono uppercase text-accent tracking-wider mb-3">
                  {prog.tagline}
                </p>
                <div className="flex-1">
                  <p className="text-sm text-body mb-4">{prog.guardrail}</p>
                </div>
                <div className="flex items-center gap-2 text-xs font-mono text-accent font-semibold group-hover:gap-3 transition-all">
                  Open in Studio
                  <ArrowRight className="w-3 h-3" />
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </Section>

      {/* Verified builds section */}
      {!loading && showcase.length > 0 && (
        <>
          <Divider className="max-w-7xl mx-auto px-6" />
          <Section
            className="max-w-7xl mx-auto px-6"
            title="Verified Builds"
            eyebrow="Benchmark Results"
            description="Replayed builds from the showcase, each fully measured and verified."
          >
            <div className="flex gap-3 overflow-x-auto pb-4">
              {showcase.slice(0, 6).map((build, idx) => (
                <Link key={idx} href={`/builds/replay-${idx}`} className="flex-shrink-0">
                  <Card elevated interactive className="min-w-fit">
                    <div className="flex flex-col gap-3">
                      <div>
                        <h4 className="font-serif font-bold text-ink">{build.app}</h4>
                        <p className="text-xs text-muted font-mono">
                          {build.base_model.split("/").pop()}
                        </p>
                      </div>
                      <Badge variant={build.passed ? "pass" : "fail"}>
                        {build.passed ? "PASS" : "FAIL"}
                      </Badge>
                    </div>
                  </Card>
                </Link>
              ))}
            </div>
          </Section>
        </>
      )}
    </div>
  );
}
