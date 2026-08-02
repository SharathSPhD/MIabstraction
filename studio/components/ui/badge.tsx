import React from "react";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "pass" | "fail" | "pending" | "live" | "replay" | "default";
  children: React.ReactNode;
}

export function Badge({ variant = "default", className, children, ...props }: BadgeProps) {
  const variantClasses = {
    pass: "inline-flex items-center rounded-full border border-hairline border-verified bg-green-50 px-3 py-1 text-xs font-semibold text-verified",
    fail: "inline-flex items-center rounded-full border border-hairline border-red-300 bg-red-50 px-3 py-1 text-xs font-semibold text-red-700",
    pending:
      "inline-flex items-center rounded-full border border-hairline border-blue-300 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700",
    live: "inline-flex items-center rounded-full border border-hairline border-verified bg-verified px-3 py-1 text-xs font-semibold text-white animate-pulse-subtle",
    replay:
      "inline-flex items-center rounded-full border border-hairline border-gray-400 bg-gray-400 px-3 py-1 text-xs font-semibold text-white",
    default:
      "inline-flex items-center rounded-full border border-hairline border-gray-300 bg-panel px-3 py-1 text-xs font-semibold text-ink",
  };

  return (
    <span className={`${variantClasses[variant]} ${className || ""}`} {...props}>
      {children}
    </span>
  );
}

export function Chip({
  children,
  className,
  onRemove,
}: {
  children: React.ReactNode;
  className?: string;
  onRemove?: () => void;
}) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border border-hairline border-gray-300 bg-panel px-3 py-1 text-xs text-ink ${className || ""}`}>
      {children}
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-1 text-muted hover:text-ink transition-colors"
          aria-label="Remove"
        >
          ×
        </button>
      )}
    </span>
  );
}
