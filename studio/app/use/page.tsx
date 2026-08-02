"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { Loader, AlertCircle, Send } from "lucide-react";

interface Artifact {
  name: string;
  app: string;
  base_model: string;
  n_controls: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatResponse {
  reply: string;
  controls_active: number;
  offline?: boolean;
  detail?: string;
}

export default function UsePage() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState("");
  const [controlsActive, setControlsActive] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

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
        setSelectedArtifact((cur) =>
          cur && list.some((a) => a.name === cur.name) ? cur : null
        );
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    };

    fetchArtifacts();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSelectArtifact = (artifact: Artifact) => {
    setSelectedArtifact(artifact);
    setMessages([]);
    setInput("");
    setControlsActive(0);
    setError("");
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
        { role: "assistant", content: data.reply },
      ]);
      setControlsActive(data.controls_active || 0);
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

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {artifacts.map((artifact, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectArtifact(artifact)}
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
                  </div>
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="grid md:grid-cols-4 gap-6">
              {/* Artifact selector sidebar */}
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
                    {controlsActive > 0 && (
                      <p className="text-xs mt-2 text-gold-400 font-mono animate-pulse-gold">
                        {controlsActive} control{controlsActive !== 1 ? "s" : ""} active
                      </p>
                    )}
                  </div>
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
                  {/* Messages */}
                  <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.length === 0 ? (
                      <div className="flex items-center justify-center h-full text-center">
                        <div>
                          <p className="text-slate-400 mb-2">Start a conversation</p>
                          <p className="text-xs text-slate-500">
                            Responses come from the verified artifact: base model + trained
                            adapters + calibrated steering controls
                          </p>
                        </div>
                      </div>
                    ) : (
                      <>
                        {messages.map((msg, idx) => (
                          <div
                            key={idx}
                            className={`flex ${
                              msg.role === "user"
                                ? "justify-end"
                                : "justify-start"
                            }`}
                          >
                            <div
                              className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                                msg.role === "user"
                                  ? "bg-gold-600/20 border border-gold-600/50 text-slate-100"
                                  : "bg-night-700/50 border border-night-600/50 text-slate-200"
                              }`}
                            >
                              <p className="text-sm">{msg.content}</p>
                            </div>
                          </div>
                        ))}
                        {chatLoading && (
                          <div className="flex justify-start">
                            <div className="bg-night-700/50 border border-night-600/50 text-slate-200 px-4 py-3 rounded-lg">
                              <div className="flex gap-2 items-center">
                                <Loader className="w-4 h-4 animate-spin text-gold-400" />
                                <span className="text-sm text-slate-400">
                                  Model thinking (this may take 30-60s)...
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </>
                    )}
                  </div>

                  {/* Error message */}
                  {error && (
                    <div className="px-6 py-3 bg-rose-500/10 border-t border-rose-500/50">
                      <div className="flex gap-2">
                        <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-rose-300">{error}</p>
                      </div>
                    </div>
                  )}

                  {/* Input form */}
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
