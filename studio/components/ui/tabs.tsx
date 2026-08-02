import React, { useState } from "react";

interface TabProps {
  label: string;
  value: string;
  children: React.ReactNode;
}

interface TabsProps {
  defaultValue?: string;
  className?: string;
  children: React.ReactElement<TabProps>[];
}

export function Tabs({ defaultValue, className, children }: TabsProps) {
  const tabs = React.Children.toArray(children) as React.ReactElement<TabProps>[];
  const [activeTab, setActiveTab] = useState(defaultValue || tabs[0]?.props.value || "");

  const activeContent = tabs.find((tab) => tab.props.value === activeTab)?.props.children;

  return (
    <div className={className}>
      <div className="flex gap-2 border-b border-hairline border-gray-300 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.props.value}
            onClick={() => setActiveTab(tab.props.value)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors duration-150 ${
              activeTab === tab.props.value
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-body hover:text-ink"
            }`}
          >
            {tab.props.label}
          </button>
        ))}
      </div>
      <div className="animate-fade-up">{activeContent}</div>
    </div>
  );
}

export function Tab({ children, ...props }: TabProps) {
  return <>{children}</>;
}

export function SegmentedControl({
  options,
  value,
  onChange,
  className,
}: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
}) {
  return (
    <div className={`flex gap-1 bg-panel p-1 rounded-lg border border-hairline border-gray-300 w-fit ${className || ""}`}>
      {options.map((option) => (
        <button
          key={option.value}
          onClick={() => onChange(option.value)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            value === option.value
              ? "bg-accent text-paper"
              : "text-ink hover:bg-gray-100"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
