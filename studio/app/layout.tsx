import type { Metadata } from "next";
import "./globals.css";

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
      <body className="bg-paper text-ink">
        <nav className="border-b border-panel">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <h1 className="font-serif text-2xl font-bold">Loom Studio</h1>
            <div className="space-x-6 text-sm">
              <a href="/" className="hover:text-body">
                Home
              </a>
              <a href="/studio" className="hover:text-body">
                Editor
              </a>
              <a href="/builds" className="hover:text-body">
                Builds
              </a>
            </div>
          </div>
        </nav>
        <main className="min-h-screen">{children}</main>
        <footer className="border-t border-panel mt-12 py-8">
          <div className="max-w-7xl mx-auto px-6 text-center text-sm text-muted">
            <p>Loom: declarative programming for LLM consequences</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
