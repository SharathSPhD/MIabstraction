"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Loader, Zap, ArrowRight } from "lucide-react";
import { explainProgram, buildProgram } from "@/lib/gpu";
import { createClient } from "@/lib/supabase/client";

function StudioContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [source, setSource] = useState("");
  const [selectedExample, setSelectedExample] = useState("clinic");
  const [examples, setExamples] = useState<Record<string, string>>({});
  const [explaining, setExplaining] = useState(false);
  const [building, setBuilding] = useState(false);
  const [explainText, setExplainText] = useState("");
  const [error, setError] = useState("");
  const [compilerRefusal, setCompilerRefusal] = useState("");
  const [targetModel, setTargetModel] = useState("");
  const [user, setUser] = useState<{ email: string } | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  useEffect(() => {
    const supabase = createClient();
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

        try {
          const { data } = await supabase
            .from("app_admins")
            .select("email")
            .eq("email", authUser.email);

          if (data && data.length > 0) {
            setIsAdmin(true);
          }
        } catch {
          // Ignore
        }
      }
      setLoading(false);
    };

    checkSession();
  }, []);

  useEffect(() => {
    // Fetch available models from GPU health
    fetch("/api/gpu/health")
      .then((r) => r.json())
      .then((data) => {
        if (data.allowed_models && data.allowed_models.length > 0) {
          setAvailableModels(data.allowed_models);
          setTargetModel(data.allowed_models[0]);
        }
      })
      .catch(() => {
        // Fallback models
        const defaults = [
          "meta-llama/Llama-3.2-1B-Instruct",
          "meta-llama/Llama-3.2-1B",
          "Qwen/Qwen2.5-0.5B-Instruct",
        ];
        setAvailableModels(defaults);
        setTargetModel(defaults[0]);
      });
  }, []);

  useEffect(() => {
    const example = searchParams.get("example");
    if (example) {
      setSelectedExample(example.toLowerCase());
    }

    fetch("/api/examples")
      .then((r) => r.json())
      .then((data) => {
        setExamples(data);
        const exampleToLoad = example?.toLowerCase() || "clinic";
        if (data[exampleToLoad]) {
          setSource(data[exampleToLoad]);
        } else if (data.clinic) {
          setSource(data.clinic);
        }
      })
      .catch(console.error);
  }, [searchParams]);

  const handleSelectExample = (name: string) => {
    setSelectedExample(name);
    if (examples[name]) {
      setSource(examples[name]);
    }
    setError("");
    setCompilerRefusal("");
    setExplainText("");
  };

  const handleExplain = async () => {
    setError("");
    setCompilerRefusal("");
    setExplainText("");
    setExplaining(true);

    const result = await explainProgram(source);

    if (!result.ok) {
      if (result.error?.includes("422")) {
        setCompilerRefusal(result.error);
      } else {
        setError(result.error || "Failed to explain program");
      }
    } else {
      setExplainText(result.text || "");
    }

    setExplaining(false);
  };

  const handleBuild = async () => {
    if (!isAdmin) {
      setError("Running a build spends real GPU time and is limited to operators.");
      return;
    }

    setError("");
    setCompilerRefusal("");
    setBuilding(true);

    let token: string | undefined;
    const supabase = createClient();
    if (supabase) {
      const { data } = await supabase.auth.getSession();
      if (data.session?.access_token) {
        token = data.session.access_token;
      }
    }

    const result = await buildProgram(source, targetModel, token);

    if (!result.ok) {
      if (result.error?.includes("422")) {
        setCompilerRefusal(result.error);
      } else if (result.error?.includes("429") || result.error?.includes("403")) {
        setError(
          "Running a build spends real GPU time and is limited to operators."
        );
      } else {
        setError(result.error || "Failed to start build");
      }
    } else if (result.id) {
      router.push(`/builds/${result.id}`);
      return;
    }

    setBuilding(false);
  };

  const groupedModels: Record<string, string[]> = {
    "From Scratch": availableModels.filter((m) => m.includes("scratch")),
    "Llama": availableModels.filter((m) => m.includes("Llama")),
    "Qwen": availableModels.filter((m) => m.includes("Qwen")),
    "Other": availableModels.filter(
      (m) =>
        !m.includes("scratch") &&
        !m.includes("Llama") &&
        !m.includes("Qwen")
    ),
  };

  return (
    <div className="bg-night-950 min-h-screen">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-12">
          <h1 className="font-display text-5xl font-bold text-slate-100 mb-3">
            Loom Editor
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl">
            Write declarative specifications. The compiler searches, measures, and verifies behavior.
          </p>
        </div>

        {/* Program selector */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2">
            {Object.keys(examples).map((name) => (
              <button
                key={name}
                onClick={() => handleSelectExample(name)}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  selectedExample === name
                    ? "bg-gold-600 text-night-950"
                    : "bg-night-800/50 border border-night-600 text-slate-300 hover:border-gold-600/60 hover:text-gold-300"
                }`}
              >
                {name.charAt(0).toUpperCase() + name.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Left pane: editor */}
          <div className="md:col-span-2 space-y-6">
            <div>
              <label className="label">Program Source</label>
              <textarea
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="input font-mono text-xs h-96 resize-none"
                placeholder="Write your Loom program here..."
              />
            </div>

            <div className="space-y-3">
              <label className="label">Target Model</label>
              <select
                value={targetModel}
                onChange={(e) => setTargetModel(e.target.value)}
                className="input"
              >
                {Object.entries(groupedModels).map(([group, models]) => {
                  const validModels = models.filter((m) => m);
                  if (validModels.length === 0) return null;
                  return (
                    <optgroup key={group} label={group}>
                      {validModels.map((model) => (
                        <option key={model} value={model}>
                          {model.split("/").pop()}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleExplain}
                disabled={explaining || !source.trim()}
                className={`btn-ghost flex-1 gap-2 ${explaining ? "opacity-50" : ""}`}
              >
                {explaining ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Explaining...
                  </>
                ) : (
                  <>
                    <ArrowRight className="w-4 h-4" />
                    Explain
                  </>
                )}
              </button>
              <button
                onClick={handleBuild}
                disabled={
                  building || !source.trim() || !isAdmin || loading
                }
                className={`btn-gold flex-1 gap-2 ${
                  building || !isAdmin ? "opacity-50" : ""
                }`}
              >
                {building ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Building...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Build
                  </>
                )}
              </button>
            </div>

            {/* Build restriction notice */}
            {!isAdmin && !loading && (
              <div className="card p-4 border-amber-500/50 bg-amber-500/5">
                <p className="text-amber-300 text-sm">
                  Running a build spends real GPU time and is limited to operators. Every
                  prebuilt demo is open to you — try Explain to see how the compiler reasons
                  about your program.
                </p>
              </div>
            )}

            {/* Error/refusal messages */}
            {error && (
              <div className="card p-4 border-rose-500/50 bg-rose-500/5">
                <p className="text-rose-300 text-sm">{error}</p>
              </div>
            )}

            {compilerRefusal && (
              <div className="card p-4 border-amber-500/50 bg-amber-500/5">
                <p className="text-amber-300 text-xs font-mono mb-2">Compiler Refusal (422)</p>
                <p className="text-slate-300 text-sm whitespace-pre-wrap">
                  {compilerRefusal}
                </p>
              </div>
            )}

            {explainText && (
              <div className="card p-4 border-gold-600/50 bg-gold-600/5">
                <p className="text-gold-300 text-xs font-mono uppercase tracking-wider mb-3">
                  Compiler Explanation
                </p>
                <pre className="text-xs text-slate-300 overflow-x-auto">
                  {explainText}
                </pre>
              </div>
            )}
          </div>

          {/* Right pane: info */}
          <div className="space-y-6">
            <div className="card p-6">
              <h3 className="font-display text-lg font-bold text-slate-100 mb-3">
                Getting Started
              </h3>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex gap-2">
                  <span className="text-gold-400 flex-shrink-0">1.</span>
                  <span>Select an example or write your own program</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-gold-400 flex-shrink-0">2.</span>
                  <span>Click Explain to see how the compiler reasons about it</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-gold-400 flex-shrink-0">3.</span>
                  <span>Operators can click Build to compile and run</span>
                </li>
              </ul>
            </div>

            <div className="card p-6">
              <h3 className="font-display text-lg font-bold text-slate-100 mb-3">
                Learn More
              </h3>
              <div className="space-y-2">
                <Link href="/language">
                  <div className="group cursor-pointer">
                    <p className="text-gold-300 text-sm font-medium group-hover:text-gold-200 transition-colors">
                      Language Reference
                    </p>
                    <p className="text-xs text-slate-500">Clauses and tune knobs</p>
                  </div>
                </Link>
                <Link href="/compiler">
                  <div className="group cursor-pointer">
                    <p className="text-gold-300 text-sm font-medium group-hover:text-gold-200 transition-colors">
                      Compiler Architecture
                    </p>
                    <p className="text-xs text-slate-500">ISA and lowering</p>
                  </div>
                </Link>
                <Link href="/science">
                  <div className="group cursor-pointer">
                    <p className="text-gold-300 text-sm font-medium group-hover:text-gold-200 transition-colors">
                      Science
                    </p>
                    <p className="text-xs text-slate-500">5 hypotheses, verified</p>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function StudioPage() {
  return (
    <Suspense fallback={<div className="bg-night-950 min-h-screen" />}>
      <StudioContent />
    </Suspense>
  );
}
