import React from "react";

interface SectionProps {
  title: string;
  eyebrow?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
  headerAction?: React.ReactNode;
}

export function Section({
  title,
  eyebrow,
  description,
  children,
  className,
  headerAction,
}: SectionProps) {
  return (
    <section className={`py-12 ${className || ""}`}>
      <div className="mb-8">
        {eyebrow && (
          <p className="text-xs font-medium uppercase text-accent tracking-wider mb-2">
            {eyebrow}
          </p>
        )}
        <div className="flex items-start justify-between gap-6 pb-6 border-b border-hairline border-gray-300">
          <div className="flex-1">
            <h2 className="font-serif text-3xl font-bold tracking-tight mb-2">
              {title}
            </h2>
            {description && (
              <p className="text-body max-w-2xl leading-relaxed">{description}</p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      </div>
      <div className="mt-8">{children}</div>
    </section>
  );
}

export function Divider({ className }: { className?: string }) {
  return (
    <div className={`border-b border-hairline border-gray-300 my-12 ${className || ""}`} />
  );
}
