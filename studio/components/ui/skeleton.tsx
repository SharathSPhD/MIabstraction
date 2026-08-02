import React from "react";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={`bg-panel rounded animate-pulse border border-hairline border-gray-300 ${className || "h-8 w-32"}`}
    />
  );
}

export function SkeletonCard() {
  return (
    <div className="bg-white border border-hairline border-gray-300 p-6 rounded-lg space-y-4">
      <Skeleton className="h-6 w-3/4" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-5/6" />
      <div className="flex gap-2 pt-4">
        <Skeleton className="h-8 w-20" />
        <Skeleton className="h-8 w-20" />
      </div>
    </div>
  );
}

export function SkeletonLine() {
  return <Skeleton className="h-4 w-full" />;
}

export function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <SkeletonCard />
      <SkeletonCard />
      <SkeletonCard />
    </div>
  );
}
