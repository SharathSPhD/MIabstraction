"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Loader, Zap } from "lucide-react";
import { Card, Button, Callout, RefusalCallout, Tabs, Tab, EmptyState } from "@/components/ui";
import { explainProgram, buildProgram } from "@/lib/gpu";
import { createClient } from "@/lib/supabase/client";

const MODELS = [
  "meta-llama/Llama-3.2-1B-Instruct",
  "meta-llama/Llama-2-7b-chat-hf",
  "mistralai/Mistral-7B-Instruct-v0.2",
];

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
  const [targetModel, setTargetModel] = useState(MODELS[0]);

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
      } else {
        setError(result.error || "Failed to start build");
      }
    } else if (result.id) {
      router.push(`/builds/${result.id}`);
      return;
    }

    setBuilding(false);
  };

  return (
    <div className="bg-paper min-h-screen">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-12">
          <h1 className="font-serif text-5xl font-bold mb-3">Loom Editor</h1>
          <p className="text-lg text-body max-w-2xl">
            Write declarative specifications. The compiler searches, measures, and verifies behavior across the parameter space.
          </p>
        </div>

        {/* Program selector tabs */}
        <Tabs
          defaultValue={selectedExample}
          className="mb-12"
        >
          {Object.keys(examples).map((name) => (
            <Tab
              key={name}
              label={name.charAt(0).toUpperCase() + name.slice(1)}
              value={name}
            >
              <div className="grid md:grid-cols-3 gap-8">
                {/* Left pane: editor */}
                <div className="md:col-span-2 space-y-6">
                  <div>
                    <label className="stat-label block mb-3">Program Source</label>
                    <textarea
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                      onClick={() => handleSelectExample(name)}
                      className="w-full h-96 px-4 py-3 border border-hairline border-gray-300 rounded-lg font-mono text-sm bg-white text-ink resize-none focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1"
                      placeholder="Write your Loom program here..."
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="stat-label">Target Model</label>
                    <select
                      value={targetModel}
                      onChange={(e) => setTargetModel(e.target.value)}
                      className="w-full px-4 py-2 border border-hairline border-gray-300 rounded-lg font-sans text-sm bg-white text-ink focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1"
                    >
                      {MODELS.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      onClick={handleExplain}
                      disabled={explaining || !source.trim()}
                      className="flex-1"
                      size="lg"
                    >
                      {explaining && <Loader className="w-4 h-4 animate-spin" />}
                      Explain
                    </Button>
                    <Button
                      onClick={handleBuild}
                      disabled={building || !source.trim()}
                      className="flex-1"
                      size="lg"
                    >
                      {building && <Loader className="w-4 h-4 animate-spin" />}
                      Build on GPU
                    </Button>
                  </div>
                </div>

                {/* Right pane: output */}
                <div className="space-y-4">
                  {compilerRefusal && (
                    <RefusalCallout message={compilerRefusal} />
                  )}

                  {error && (
                    <Callout variant="error">
                      {error}
                    </Callout>
                  )}

                  {explainText && (
                    <Card elevated>
                      <h3 className="font-serif font-bold mb-4">Search Plan</h3>
                      <pre className="text-xs leading-relaxed whitespace-pre-wrap font-mono text-body overflow-x-auto">
                        {explainText}
                      </pre>
                    </Card>
                  )}

                  {!explainText && !error && !compilerRefusal && (
                    <EmptyState
                      icon={<Zap className="w-12 h-12 text-muted" />}
                      title="Ready to compile"
                      description="Click Explain to see the compiler's search plan"
                    />
                  )}
                </div>
              </div>
            </Tab>
          ))}
        </Tabs>
      </div>
    </div>
  );
}

export default function StudioPage() {
  return (
    <Suspense fallback={<div className="max-w-7xl mx-auto px-6 py-12"><p>Loading...</p></div>}>
      <StudioContent />
    </Suspense>
  );
}
