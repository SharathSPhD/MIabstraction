import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { GpuStatus } from "./components/gpu-status";

export const metadata: Metadata = {
  title: "Loom Studio",
  description: "Program your LLM with consequences",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-paper text-ink min-h-screen flex flex-col">
        <nav className="border-b border-hairline border-gray-300">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <Link href="/" className="no-underline">
              <div className="font-serif text-xl font-bold text-ink hover:text-accent transition-colors">
                Loom
              </div>
            </Link>
            <div className="flex items-center gap-8">
              <div className="flex gap-6 text-sm">
                <Link href="/studio" className="text-ink hover:text-accent transition-colors">
                  Studio
                </Link>
                <Link href="/builds" className="text-ink hover:text-accent transition-colors">
                  Builds
                </Link>
              </div>
              <div className="border-l border-hairline border-gray-300 pl-6">
                <GpuStatus />
              </div>
            </div>
          </div>
        </nav>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-hairline border-gray-300 mt-16 py-8">
          <div className="max-w-7xl mx-auto px-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-body">
                Loom: declarative programming for LLM consequences
              </p>
              <div className="flex gap-4 text-sm">
                <a href="https://github.com/qbz506/loom" target="_blank" rel="noopener noreferrer" className="text-ink hover:text-accent transition-colors">
                  GitHub
                </a>
                <a href="https://huggingface.co/qbz506" target="_blank" rel="noopener noreferrer" className="text-ink hover:text-accent transition-colors">
                  Hugging Face
                </a>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
