"use client";

import { useEffect, useState } from "react";
import { Zap } from "lucide-react";

export function GpuStatus() {
  const [status, setStatus] = useState<"online" | "offline" | "loading">("loading");
  const [queue, setQueue] = useState(0);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch("/api/gpu/health");
        if (response.ok) {
          const data = await response.json();
          setStatus("online");
          setQueue(data.queue_length || 0);
        } else {
          setStatus("offline");
        }
      } catch {
        setStatus("offline");
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          status === "online" ? "bg-emerald-400" : "status-dot-warn"
        } ${status === "online" ? "" : "animate-pulse"}`}
      />
      <div className="font-mono text-xs">
        {status === "loading" && <span className="text-slate-400">connecting...</span>}
        {status === "online" && (
          <>
            <span className="text-emerald-400 font-semibold">GB10 online</span>
            {queue > 0 && <span className="text-slate-400"> · queue {queue}</span>}
          </>
        )}
        {status === "offline" && (
          <span className="text-slate-500">offline — replay only</span>
        )}
      </div>
    </div>
  );
}
