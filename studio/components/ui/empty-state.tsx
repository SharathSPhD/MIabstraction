import React from "react";
import { AlertCircle } from "lucide-react";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  icon = <AlertCircle className="w-12 h-12 text-muted" />,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center py-16 px-6 bg-white border border-hairline border-gray-300 rounded-lg ${className || ""}`}
    >
      <div className="mb-4">{icon}</div>
      <h3 className="font-serif text-xl font-bold mb-2 text-ink">{title}</h3>
      {description && <p className="text-body text-center mb-6 max-w-sm">{description}</p>}
      {action && <div>{action}</div>}
    </div>
  );
}
