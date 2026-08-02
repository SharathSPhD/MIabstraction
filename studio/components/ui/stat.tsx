import React from "react";

interface StatProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: "up" | "down" | "neutral";
  className?: string;
}

export function Stat({ label, value, unit, trend, className }: StatProps) {
  return (
    <div className={`flex flex-col gap-1 ${className || ""}`}>
      <label className="text-xs font-medium uppercase text-muted tracking-wider">
        {label}
      </label>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-lg font-bold tabular-nums text-ink">
          {value}
        </span>
        {unit && <span className="text-xs text-body font-mono">{unit}</span>}
        {trend && (
          <span
            className={`text-xs font-semibold ${
              trend === "up"
                ? "text-verified"
                : trend === "down"
                  ? "text-refusal"
                  : "text-muted"
            }`}
          >
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}
          </span>
        )}
      </div>
    </div>
  );
}

export function StatGrid({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6 p-6 bg-white border border-hairline border-gray-300 rounded-lg ${className || ""}`}
    >
      {children}
    </div>
  );
}
