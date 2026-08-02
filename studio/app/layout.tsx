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
      <body className="bg-paper text-ink min-h-screen flex flex-col">
        <NavHeader />
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
