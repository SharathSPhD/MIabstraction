import type { Metadata } from "next";
import "./globals.css";
import { NavHeader } from "./components/nav-header";

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
      <body className="bg-night-950 text-slate-200 min-h-screen flex flex-col">
        <NavHeader />
        {children}
        <footer className="border-t border-night-600/50 bg-night-900/50 mt-16 py-8">
          <div className="max-w-7xl mx-auto px-6">
            <div className="flex items-center justify-between text-sm">
              <p className="text-slate-400">
                Loom: declarative programming for language model consequences
              </p>
              <div className="flex gap-4">
                <a href="https://github.com/qbz506/loom" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-gold-300 transition-colors">
                  GitHub
                </a>
                <a href="https://huggingface.co/qbz506" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-gold-300 transition-colors">
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
