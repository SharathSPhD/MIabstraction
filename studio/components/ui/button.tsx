import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  children: React.ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  className,
  isLoading = false,
  disabled,
  children,
  ...props
}: ButtonProps) {
  const sizeClasses = {
    sm: "px-3 py-1 text-sm",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };

  const variantClasses = {
    primary:
      "inline-flex items-center justify-center gap-2 rounded-lg bg-accent text-paper font-medium hover:bg-ink transition-colors duration-150 border border-hairline border-accent focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed",
    ghost:
      "inline-flex items-center justify-center gap-2 rounded-lg border border-hairline border-gray-300 text-ink hover:bg-panel transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed",
    danger:
      "inline-flex items-center justify-center gap-2 rounded-lg bg-red-500 text-white font-medium hover:bg-red-600 transition-colors duration-150 border border-hairline border-red-500 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed",
  };

  return (
    <button
      disabled={disabled || isLoading}
      className={`${sizeClasses[size]} ${variantClasses[variant]} ${className || ""}`}
      {...props}
    >
      {children}
    </button>
  );
}
