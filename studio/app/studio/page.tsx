"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Loader, AlertCircle } from "lucide-react";
import { explainProgram, buildProgram } from "@/lib/gpu";

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

    const result = await buildProgram(source, targetModel);

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
    <div className="max-w-7xl mx-auto px-6 py-12">
      <h1 className="font-serif text-4xl font-bold mb-4">Loom Editor</h1>
      <p className="text-body mb-8">
        Write declarative specifications. The compiler searches, measures, and verifies.
      </p>

      {/* Program selector tabs */}
      <div className="flex gap-1 mb-8 border-b border-hairline border-gray-300">
        {Object.keys(examples).map((name) => (
          <button
            key={name}
            onClick={() => handleSelectExample(name)}
            className={`px-4 py-3 font-sans font-medium text-sm transition-colors duration-150 border-b-2 ${
              selectedExample === name
                ? "border-accent text-accent"
                : "border-transparent text-body hover:text-ink"
            }`}
          >
            {name.charAt(0).toUpperCase() + name.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        {/* Left pane: editor */}
        <div className="md:col-span-2 space-y-6">
          <div>
            <label className="block stat-label mb-3">Program Source</label>
            <textarea
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="w-full h-96 px-4 py-3 border border-hairline border-gray-300 font-mono text-sm bg-white text-ink resize-none"
              placeholder="Write your Loom program here..."
            />
          </div>

          <div className="space-y-3">
            <label className="block stat-label">Target Model</label>
            <select
              value={targetModel}
              onChange={(e) => setTargetModel(e.target.value)}
              className="w-full px-4 py-2 border border-hairline border-gray-300 font-sans text-sm bg-white text-ink"
            >
              {MODELS.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleExplain}
              disabled={explaining || !source.trim()}
              className="flex-1 button-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {explaining && <Loader className="w-4 h-4 animate-spin" />}
              Explain
            </button>
            <button
              onClick={handleBuild}
              disabled={building || !source.trim()}
              className="flex-1 button-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {building && <Loader className="w-4 h-4 animate-spin" />}
              Build on GPU
            </button>
          </div>
        </div>

        {/* Right pane: output */}
        <div className="space-y-4">
          {compilerRefusal && (
            <div className="diagnostic-box space-y-2">
              <div className="flex gap-2">
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <h4 className="font-semibold">The compiler refused</h4>
              </div>
              <p className="font-mono text-xs break-words text-yellow-900">
                {compilerRefusal}
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border-l-2 border-red-400 p-4 space-y-2">
              <p className="font-semibold text-red-900">Error</p>
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {explainText && (
            <div className="card space-y-3">
              <h3 className="font-serif font-bold">Search Plan</h3>
              <pre className="text-xs leading-relaxed whitespace-pre-wrap font-mono text-body">
                {explainText}
              </pre>
            </div>
          )}

          {!explainText && !error && !compilerRefusal && (
            <div className="card text-center py-12">
              <p className="text-muted text-sm">
                Click Explain to see the compiler's search plan
              </p>
            </div>
          )}
        </div>
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
