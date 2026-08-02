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

  // Load available artifacts
  useEffect(() => {
    const fetchArtifacts = async () => {
      try {
        const res = await fetch("/api/gpu/artifacts");
        if (!res.ok) {
          const data = (await res.json()) as { offline?: boolean; detail?: string };
          throw new Error(data.detail || "Failed to load artifacts");
        }
        const data = (await res.json()) as { artifacts?: Artifact[] };
        setArtifacts(data.artifacts || []);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    };

    fetchArtifacts();
  }, []);

  // Auto-scroll to bottom of messages
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
    setInput("");
    setChatLoading(true);
    setError("");

    // Add user message to history
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
        setError(data.detail || "Failed to get response from model");
        // Remove the user message if the request failed
        setMessages(messages);
        return;
      }

      // Add assistant response
      setMessages([
        ...updatedMessages,
        { role: "assistant", content: data.reply },
      ]);
      setControlsActive(data.controls_active || 0);
    } catch (e) {
      setError(String(e));
      setMessages(messages);
    } finally {
      setChatLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Use Your Model</h1>
        <div className="card text-center py-12">
          <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-body">Loading available models...</p>
        </div>
      </div>
    );
  }

  if (error && artifacts.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Use Your Model</h1>
        <div className="card bg-red-50 border-l-2 border-red-400 py-6">
          <div className="flex gap-3">
            <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-red-900 mb-1">Error</p>
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (artifacts.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Use Your Model</h1>
        <div className="card text-center py-12">
          <p className="text-muted mb-4">No verified artifacts available yet.</p>
          <Link href="/builds" className="text-accent hover:underline text-sm">
            Build one first
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {!selectedArtifact ? (
        <>
          <h1 className="font-serif text-4xl font-bold mb-4">Use Your Model</h1>
          <p className="text-body mb-8">
            Select a verified artifact to start chatting.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
            {artifacts.map((artifact, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectArtifact(artifact)}
                className="card-hover p-6 cursor-pointer text-left h-full hover:shadow-md transition-all"
              >
                <h3 className="font-serif text-lg font-bold mb-2">{artifact.app}</h3>
                <p className="text-xs font-mono text-muted uppercase mb-3">
                  {artifact.base_model.split("/").pop()}
                </p>
                <p className="text-sm text-body mb-4">
                  {artifact.name}
                </p>
                <div className="flex gap-4 text-xs text-muted">
                  <div>
                    <span className="block font-mono text-ink">
                      {artifact.n_controls}
                    </span>
                    <span className="text-muted">controls</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          {/* Chat interface */}
          <div className="grid md:grid-cols-4 gap-6 mb-12">
            {/* Artifact selector sidebar */}
            <div className="md:col-span-1">
              <div className="card p-4 sticky top-4">
                <h3 className="font-serif font-bold mb-3">Selected</h3>
                <div className="mb-4 pb-4 border-b border-hairline border-gray-300">
                  <p className="font-serif font-bold text-sm">{selectedArtifact.app}</p>
                  <p className="text-xs font-mono text-muted mt-1">
                    {selectedArtifact.base_model.split("/").pop()}
                  </p>
                  {controlsActive > 0 && (
                    <p className="text-xs mt-2 text-verified font-mono">
                      {controlsActive} control{controlsActive !== 1 ? "s" : ""} active
                    </p>
                  )}
                </div>
                <button
                  onClick={() => setSelectedArtifact(null)}
                  className="w-full py-2 px-3 text-sm text-ink border border-hairline border-gray-300 rounded hover:bg-panel transition-colors"
                >
                  Change Artifact
                </button>
              </div>
            </div>

            {/* Chat area */}
            <div className="md:col-span-3">
              <div className="card flex flex-col h-full bg-white" style={{ minHeight: "600px" }}>
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                  {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-center">
                      <div>
                        <p className="text-muted mb-2">Start a conversation</p>
                        <p className="text-xs text-muted">
                          Responses come from the verified artifact:{" "}
                          <br />
                          base model + trained adapters + calibrated steering controls
                        </p>
                      </div>
                    </div>
                  ) : (
                    <>
                      {messages.map((msg, idx) => (
                        <div
                          key={idx}
                          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                          <div
                            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                              msg.role === "user"
                                ? "bg-accent text-white"
                                : "bg-panel text-ink"
                            }`}
                          >
                            <p className="text-sm">{msg.content}</p>
                          </div>
                        </div>
                      ))}
                      {chatLoading && (
                        <div className="flex justify-start">
                          <div className="bg-panel text-ink px-4 py-2 rounded-lg">
                            <div className="flex gap-2 items-center">
                              <Loader className="w-4 h-4 animate-spin" />
                              <span className="text-sm text-muted">
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
                  <div className="px-6 py-3 bg-red-50 border-t border-red-200">
                    <div className="flex gap-2">
                      <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-red-800">{error}</p>
                    </div>
                  </div>
                )}

                {/* Input form */}
                <form
                  onSubmit={handleSendMessage}
                  className="border-t border-hairline border-gray-300 p-4"
                >
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Message the model..."
                      disabled={chatLoading}
                      className="flex-1 px-4 py-2 border border-hairline border-gray-300 rounded bg-white text-ink placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent disabled:opacity-50"
                    />
                    <button
                      type="submit"
                      disabled={chatLoading || !input.trim()}
                      className="p-2 bg-accent text-white rounded hover:opacity-90 disabled:opacity-50 transition-opacity"
                      title="Send message"
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </div>
                  <p className="text-xs text-muted mt-2">
                    Each turn is answered fresh from your verified model.
                  </p>
                </form>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
