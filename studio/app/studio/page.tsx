"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader, AlertCircle } from "lucide-react";
import { explainProgram, buildProgram } from "@/lib/gpu";

export default function StudioPage() {
  const router = useRouter();
  const [source, setSource] = useState("");
  const [selectedExample, setSelectedExample] = useState("clinic");
  const [examples, setExamples] = useState<Record<string, string>>({});
  const [explaining, setExplaining] = useState(false);
  const [building, setBuilding] = useState(false);
  const [explainText, setExplainText] = useState("");
  const [error, setError] = useState("");
  const [compilerRefusal, setCompilerRefusal] = useState("");

  useEffect(() => {
    fetch("/api/examples")
      .then((r) => r.json())
      .then((data) => {
        setExamples(data);
        if (data.clinic) {
          setSource(data.clinic);
        }
      })
      .catch(console.error);
  }, []);

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

    const result = await buildProgram(source, "meta-llama/Llama-3.2-1B-Instruct");

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
      <h2 className="font-serif text-3xl font-bold mb-8">Loom Editor</h2>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Left pane: editor */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-semibold mb-2">
              Example Program
            </label>
            <select
              value={selectedExample}
              onChange={(e) => handleSelectExample(e.target.value)}
              className="w-full px-3 py-2 border border-panel rounded bg-white text-ink"
            >
              {Object.keys(examples).map((name) => (
                <option key={name} value={name}>
                  {name.charAt(0).toUpperCase() + name.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold mb-2">Source</label>
            <textarea
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="w-full h-96 px-3 py-2 border border-panel rounded font-mono text-sm bg-white text-ink"
              placeholder="Write your Loom program here..."
            />
          </div>

          <div className="flex gap-2">
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
              Build
            </button>
          </div>
        </div>

        {/* Right pane: output */}
        <div className="space-y-4">
          {compilerRefusal && (
            <div className="diagnostic-box flex gap-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold mb-1">
                  The compiler refused, and says why
                </p>
                <p className="font-mono text-xs break-words">{compilerRefusal}</p>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded text-sm text-red-900">
              <p className="font-semibold mb-1">Error</p>
              <p>{error}</p>
            </div>
          )}

          {explainText && (
            <div className="card">
              <h3 className="font-serif font-bold mb-4">Search Plan</h3>
              <pre className="text-xs leading-relaxed whitespace-pre-wrap">
                {explainText}
              </pre>
            </div>
          )}

          {!explainText && !error && !compilerRefusal && (
            <div className="card text-center text-muted py-12">
              <p>Click Explain to see the compiler's search plan</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
