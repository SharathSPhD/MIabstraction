import React from "react";
import { AlertCircle, CheckCircle2, AlertTriangle, Info } from "lucide-react";

interface CalloutProps {
  variant?: "info" | "warning" | "error" | "success";
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export function Callout({
  variant = "info",
  title,
  children,
  className,
}: CalloutProps) {
  const variantConfig = {
    info: {
      icon: Info,
      bg: "bg-blue-50",
      border: "border-blue-200",
      text: "text-blue-900",
      iconColor: "text-blue-600",
    },
    warning: {
      icon: AlertTriangle,
      bg: "bg-yellow-50",
      border: "border-yellow-200",
      text: "text-yellow-900",
      iconColor: "text-yellow-600",
    },
    error: {
      icon: AlertCircle,
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-900",
      iconColor: "text-red-600",
    },
    success: {
      icon: CheckCircle2,
      bg: "bg-green-50",
      border: "border-green-200",
      text: "text-green-900",
      iconColor: "text-green-600",
    },
  };

  const config = variantConfig[variant];
  const Icon = config.icon;

  return (
    <div
      className={`flex gap-4 p-4 rounded-lg border-l-4 ${config.bg} border ${config.border} ${className || ""}`}
    >
      <Icon className={`w-5 h-5 ${config.iconColor} flex-shrink-0 mt-0.5`} />
      <div className={config.text}>
        {title && <p className="font-semibold text-sm mb-1">{title}</p>}
        <div className="text-sm">{children}</div>
      </div>
    </div>
  );
}

export function RefusalCallout({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <Callout variant="error" title="Compiler Refusal" className={className}>
      <p className="font-mono text-xs whitespace-pre-wrap">{message}</p>
    </Callout>
  );
}
