import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  elevated?: boolean;
  interactive?: boolean;
}

export function Card({
  children,
  className,
  elevated = false,
  interactive = false,
  ...props
}: CardProps) {
  const baseClasses =
    "bg-white border border-hairline border-gray-300 p-6 rounded-lg transition-all duration-150";
  const elevatedClasses = elevated
    ? "shadow-sm hover:shadow-md hover:border-accent/30"
    : "";
  const interactiveClasses = interactive
    ? "hover:shadow-md hover:border-accent cursor-pointer hover:scale-[1.01]"
    : "";

  return (
    <div
      className={`${baseClasses} ${elevatedClasses} ${interactiveClasses} ${className || ""}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`mb-4 pb-4 border-b border-hairline border-gray-300 ${className || ""}`}>
      {children}
    </div>
  );
}

export function CardContent({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={className}>{children}</div>;
}

export function CardFooter({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`mt-4 pt-4 border-t border-hairline border-gray-300 flex gap-2 ${className || ""}`}>
      {children}
    </div>
  );
}
