"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { Loader, AlertCircle, Send, Download, GitCompare, ShieldCheck } from "lucide-react";

interface Artifact {
  name: string;
  app: string;
  base_model: string;
  n_controls: number;
}

interface Control {
  capability?: string;
  kind?: string;
  layer?: number;
  strength?: number;
}

interface BaseSide {
  available: boolean;
  base_model?: string | null;
  reply?: string;
  why?: string;
  note?: string;
}

interface PolicyDecision {
  allowed: boolean;
  reason: string;
  clause?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  answeredBy?: string;
  policy?: PolicyDecision;
  base?: BaseSide;
}

interface ChatResponse {
  reply: string;
  controls_active: number;
  controls?: Control[];
  adapters?: string[];
  answered_by?: string;
  policy?: PolicyDecision;
  base?: BaseSide;
  offline?: boolean;
  detail?: string;
}

// A transcript is only worth keeping if it is still there tomorrow. Keyed per artifact,
// because a conversation with one model is not a conversation with another.
const historyKey = (name: string) => `loom.use.history.${name}`;

export default function UsePage() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState("");
  const [controlsActive, setControlsActive] = useState(0);
  const [controls, setControls] = useState<Control[]>([]);
  const [adapters, setAdapters] = useState<string[]>([]);
  const [compare, setCompare] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const selectArtifact = (artifact: Artifact) => {
    setSelectedArtifact(artifact);
    setInput("");
    setControlsActive(0);
    setControls([]);
    setAdapters([]);
    setError("");
    let restored: Message[] = [];
    try {
      const raw = localStorage.getItem(historyKey(artifact.name));
      if (raw) restored = JSON.parse(raw) as Message[];
    } catch {
      restored = [];
    }
    setMessages(restored);
  };

  useEffect(() => {
    const fetchArtifacts = async () => {
      try {
        const res = await fetch("/api/gpu/artifacts");
        if (!res.ok) {
          const data = (await res.json()) as { offline?: boolean; detail?: string };
          throw new Error(data.detail || "Failed to load artifacts");
        }
        const data = (await res.json()) as { artifacts?: Artifact[] };
        const list = data.artifacts || [];
        setArtifacts(list);

        // A build page links here with the model it just made. Landing on a list and
        // being asked to find it again is the app forgetting what you were doing.
        const wanted = new URLSearchParams(window.location.search).get("artifact");
        const preselected = wanted ? list.find((a) => a.name === wanted) : undefined;
        if (preselected) {
          selectArtifact(preselected);
        } else {
          setSelectedArtifact((cur) =>
            cur && list.some((a) => a.name === cur.name) ? cur : null
          );
          if (wanted) {
            setError(
              `There is no model called "${wanted}" in the library any more. ` +
                `Pick one below — nothing else has changed.`
            );
          }
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    };

    fetchArtifacts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Persisted after every turn, including failed ones, so a refresh is never the thing
  // that loses a conversation.
  useEffect(() => {
    if (!selectedArtifact) return;
    try {
      localStorage.setItem(historyKey(selectedArtifact.name), JSON.stringify(messages));
    } catch {
      /* storage full or disabled: the chat still works, it just will not persist */
    }
  }, [messages, selectedArtifact]);

  const exportTranscript = () => {
    if (!selectedArtifact) return;
    const lines = [
      `# ${selectedArtifact.app}`,
      ``,
      `Model: \`${selectedArtifact.name}\``,
      `Substrate: \`${selectedArtifact.base_model}\``,
      controls.length
        ? `Controls: ${controls
            .map((c) => `${c.capability ?? c.kind} (layer ${c.layer}, dose ${c.strength})`)
            .join("; ")}`
        : `Controls: none installed`,
      adapters.length ? `Adapters: ${adapters.join(", ")}` : ``,
      ``,
      `---`,
      ``,
      ...messages.flatMap((m) => {
        const who = m.role === "user" ? "You" : selectedArtifact.app;
        const out = [`**${who}:** ${m.content}`];
        if (m.answeredBy) out.push(`  _answered by: ${m.answeredBy}_`);
        if (m.base?.available && m.base.reply)
          out.push(`  _base model (${m.base.base_model}) said:_ ${m.base.reply}`);
        else if (m.base && !m.base.available)
          out.push(`  _no base model to compare: ${m.base.why}_`);
        out.push(``);
        return out;
      }),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedArtifact.app.toLowerCase()}-transcript.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !selectedArtifact || chatLoading) return;

    const userMessage = input.trim();
    setChatLoading(true);
    setError("");

    const messagesBefore = messages;
    const updatedMessages: Message[] = [
      ...messages,
      { role: "user", content: userMessage },
    ];
    setMessages(updatedMessages);

    try {
      const res = await fetch("/api/gpu/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          artifact: selectedArtifact.name,
          message: userMessage,
          compare_to_base: compare,
        }),
      });

      const data = (await res.json()) as ChatResponse;

      if (!res.ok || data.offline) {
        setError(data.detail || "The model did not answer. Nothing was lost — your question is still in the box.");
        // Revert to the history as it was BEFORE this turn, and give the question
        // back. Reverting to a stale `messages` closure and having already cleared
        // the input is what made a failed turn erase what the person had typed.
        setMessages(messagesBefore);
        setInput(userMessage);
        return;
      }

      setInput("");
      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: data.reply,
          answeredBy: data.answered_by,
          policy: data.policy,
          base: data.base,
        },
      ]);
      setControlsActive(data.controls_active || 0);
      if (data.controls) setControls(data.controls);
      if (data.adapters) setAdapters(data.adapters);
    } catch (e) {
      setError("The model could not be reached. Your question is still in the box.");
      setMessages(messagesBefore);
      setInput(userMessage);
    } finally {
      setChatLoading(false);
    }
  };

  if (loading) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Use Your Model
          </h1>
          <div className="card text-center py-12">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4 text-gold-400" />
            <p className="text-slate-400">Loading available models...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error && artifacts.length === 0) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Use Your Model
          </h1>
          <div className="card border-rose-500/50 bg-rose-500/5 p-6">
            <div className="flex gap-3">
              <AlertCircle className="w-6 h-6 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-300 mb-1">Error</p>
                <p className="text-rose-200 text-sm">{error}</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (artifacts.length === 0) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Use Your Model
          </h1>
          <div className="card text-center py-12">
            <p className="text-slate-400 mb-4">No verified artifacts available yet.</p>
            <Link href="/builds" className="text-gold-300 hover:text-gold-200 text-sm">
              Build one first →
            </Link>
          </div>
        </div>
      </main>
    );
  }

  const madeHere = selectedArtifact?.base_model?.startsWith("scratch");

  return (
    <main className="bg-night-950 min-h-screen">
      <div className="max-w-7xl mx-auto px-6 py-12">
        {!selectedArtifact ? (
          <>
            <h1 className="font-display text-4xl font-bold text-slate-100 mb-4">
              Use Your Model
            </h1>
            <p className="text-slate-400 mb-8">
              Select a verified artifact to start chatting.
            </p>

            {error && (
              <div className="card border-amber-500/40 bg-amber-500/5 p-4 mb-6">
                <p className="text-sm text-amber-200">{error}</p>
              </div>
            )}

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {artifacts.map((artifact, idx) => (
                <button
                  key={idx}
                  onClick={() => selectArtifact(artifact)}
                  className="card group cursor-pointer text-left hover:bg-night-700/70 transition-colors h-full p-6"
                >
                  <h3 className="font-display text-lg font-bold text-slate-100 group-hover:text-gold-300 transition-colors mb-2">
                    {artifact.app}
                  </h3>
                  <p className="text-xs font-mono text-slate-400 uppercase mb-3">
                    {artifact.base_model.split("/").pop()}
                  </p>
                  <p className="text-sm text-slate-300 mb-4">{artifact.name}</p>
                  <div className="flex gap-4 text-xs text-slate-400">
                    <div>
                      <span className="block font-mono text-slate-100">
                        {artifact.n_controls}
                      </span>
                      <span className="text-slate-500">controls</span>
                    </div>
                    {artifact.base_model.startsWith("scratch") && (
                      <div>
                        <span className="block font-mono text-gold-300">made here</span>
                        <span className="text-slate-500">no base model</span>
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="grid md:grid-cols-4 gap-6">
              {/* What is actually answering you */}
              <div className="md:col-span-1">
                <div className="card p-4 sticky top-4">
                  <h3 className="font-display font-bold mb-3 text-slate-100">
                    Selected
                  </h3>
                  <div className="mb-4 pb-4 border-b border-night-600/50">
                    <p className="font-display font-bold text-sm text-slate-100">
                      {selectedArtifact.app}
                    </p>
                    <p className="text-xs font-mono text-slate-400 mt-1">
                      {selectedArtifact.base_model.split("/").pop()}
                    </p>
                    {madeHere && (
                      <p className="text-xs mt-2 text-gold-300">
                        Made here — architecture chosen, tokenizer learned and weights
                        trained by the compiler. There is no prior model underneath.
                      </p>
                    )}
                    {controlsActive > 0 && (
                      <p className="text-xs mt-2 text-gold-400 font-mono">
                        {controlsActive} control{controlsActive !== 1 ? "s" : ""} active
                      </p>
                    )}
                  </div>

                  {controls.length > 0 && (
                    <div className="mb-4 pb-4 border-b border-night-600/50">
                      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                        Controls in the loop
                      </p>
                      {controls.map((c, i) => (
                        <div key={i} className="mb-2">
                          <p className="text-xs text-slate-300">
                            {c.capability ?? c.kind}
                          </p>
                          <p className="text-[11px] font-mono text-slate-500">
                            layer {c.layer} · dose {c.strength}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {adapters.length > 0 && (
                    <div className="mb-4 pb-4 border-b border-night-600/50">
                      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                        Adapters reapplied
                      </p>
                      {adapters.map((a, i) => (
                        <p key={i} className="text-[11px] font-mono text-slate-400">
                          {a}
                        </p>
                      ))}
                    </div>
                  )}

                  <label className="flex items-start gap-2 mb-4 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={compare}
                      onChange={(e) => setCompare(e.target.checked)}
                      className="mt-0.5"
                    />
                    <span className="text-xs text-slate-300">
                      <span className="flex items-center gap-1 font-semibold">
                        <GitCompare className="w-3 h-3" /> Compare against the base
                      </span>
                      <span className="block text-slate-500 mt-1">
                        Put the same question to the untouched base model, so the
                        difference is the program&apos;s doing and not something the base
                        model already did. Slower — it answers twice.
                      </span>
                    </span>
                  </label>

                  <button
                    onClick={exportTranscript}
                    disabled={messages.length === 0}
                    className="btn-ghost w-full text-sm mb-2 gap-2 disabled:opacity-40"
                  >
                    <Download className="w-4 h-4" /> Export transcript
                  </button>
                  <button
                    onClick={() => setSelectedArtifact(null)}
                    className="btn-ghost w-full text-sm"
                  >
                    Change Artifact
                  </button>
                </div>
              </div>

              {/* Chat area */}
              <div className="md:col-span-3">
                <div
                  className="card flex flex-col h-full bg-night-800/50 border-night-600/50"
                  style={{ minHeight: "600px" }}
                >
                  <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.length === 0 ? (
                      <div className="flex items-center justify-center h-full text-center">
                        <div>
                          <p className="text-slate-400 mb-2">Start a conversation</p>
                          <p className="text-xs text-slate-500 max-w-md">
                            {madeHere
                              ? "This model was made from nothing by your program — every weight in it came from the corpus the program named."
                              : "Responses come from the verified artifact: base model + trained adapters + calibrated steering controls."}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <>
                        {messages.map((msg, idx) => (
                          <div key={idx}>
                            <div
                              className={`flex ${
                                msg.role === "user" ? "justify-end" : "justify-start"
                              }`}
                            >
                              <div
                                className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                                  msg.role === "user"
                                    ? "bg-gold-600/20 border border-gold-600/50 text-slate-100"
                                    : "bg-night-700/50 border border-night-600/50 text-slate-200"
                                }`}
                              >
                                <p className="text-sm whitespace-pre-wrap">
                                  {msg.content}
                                </p>
                              </div>
                            </div>

                            {/* Which layer answered. When the gate answers, the model was
                                never consulted — that is the point of keeping policy out
                                of the weights, so the app says it rather than implying
                                the model chose to decline. */}
                            {msg.role === "assistant" &&
                              msg.answeredBy?.startsWith("policy") && (
                                <div className="flex justify-start mt-1">
                                  <p className="text-[11px] text-amber-300/80 flex items-start gap-1 max-w-md">
                                    <ShieldCheck className="w-3 h-3 mt-0.5 flex-shrink-0" />
                                    <span>
                                      Answered by the policy gate — the model was not
                                      consulted.{" "}
                                      {msg.policy?.reason && (
                                        <span className="text-slate-500">
                                          {msg.policy.reason}
                                        </span>
                                      )}
                                    </span>
                                  </p>
                                </div>
                              )}

                            {/* The base model's answer to the same question. */}
                            {msg.role === "assistant" && msg.base && (
                              <div className="flex justify-start mt-2">
                                <div className="max-w-xs lg:max-w-md px-4 py-3 rounded-lg border border-dashed border-night-600/70 bg-night-900/40">
                                  <p className="text-[11px] uppercase tracking-wide text-slate-500 mb-1">
                                    {msg.base.available
                                      ? `Base model · ${msg.base.base_model
                                          ?.split("/")
                                          .pop()}`
                                      : "No base model"}
                                  </p>
                                  <p className="text-sm text-slate-400 whitespace-pre-wrap">
                                    {msg.base.available ? msg.base.reply : msg.base.why}
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                        {chatLoading && (
                          <div className="flex justify-start">
                            <div className="bg-night-700/50 border border-night-600/50 text-slate-200 px-4 py-3 rounded-lg">
                              <div className="flex gap-2 items-center">
                                <Loader className="w-4 h-4 animate-spin text-gold-400" />
                                <span className="text-sm text-slate-400">
                                  {compare
                                    ? "Answering twice — your model and the base model..."
                                    : "Model thinking (this may take 30-60s)..."}
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </>
                    )}
                  </div>

                  {error && (
                    <div className="px-6 py-3 bg-rose-500/10 border-t border-rose-500/50">
                      <div className="flex gap-2">
                        <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-rose-300">{error}</p>
                      </div>
                    </div>
                  )}

                  <form
                    onSubmit={handleSendMessage}
                    className="border-t border-night-600/50 p-4"
                  >
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Message the model..."
                        disabled={chatLoading}
                        className="input flex-1"
                      />
                      <button
                        type="submit"
                        disabled={chatLoading || !input.trim()}
                        className="btn-gold gap-2 disabled:opacity-50"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
