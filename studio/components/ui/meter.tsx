import React from "react";

interface MeterProps {
  value: number;
  target?: number;
  min?: number;
  max?: number;
  label?: string;
  showLabel?: boolean;
}

export function Meter({
  value,
  target,
  min = 0,
  max = 1,
  label,
  showLabel = true,
}: MeterProps) {
  const range = max - min;
  const normalizedValue = ((value - min) / range) * 100;
  const normalizedTarget = target !== undefined ? ((target - min) / range) * 100 : undefined;

  return (
    <div className="flex flex-col gap-2">
      {showLabel && label && (
        <div className="flex justify-between items-baseline">
          <label className="text-xs font-medium uppercase text-muted tracking-wider">
            {label}
          </label>
          <span className="text-sm font-mono font-semibold text-ink">
            {value.toFixed(4)}
          </span>
        </div>
      )}
      <div className="relative h-6 bg-panel border border-hairline border-gray-300 rounded overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-accent to-accent/80 transition-all duration-300"
          style={{ width: `${Math.max(0, Math.min(100, normalizedValue))}%` }}
        />
        {normalizedTarget !== undefined && (
          <div
            className="absolute top-0 h-full w-0.5 bg-gray-500 opacity-60"
            style={{ left: `${normalizedTarget}%` }}
            title={`Target: ${target?.toFixed(4)}`}
          />
        )}
      </div>
    </div>
  );
}
