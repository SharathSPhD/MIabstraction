import Link from "next/link";
import { Code, BookOpen, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <section className="mb-16 text-center">
        <h2 className="font-serif text-4xl font-bold mb-4">
          Program Your LLM
        </h2>
        <p className="text-lg text-body mb-8">
          Write consequences. The compiler measures, searches, and verifies them.
        </p>
        <div className="flex gap-4 justify-center">
          <Link href="/studio" className="button-primary">
            Open Editor
          </Link>
          <Link href="/builds" className="button-secondary">
            View Builds
          </Link>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Code className="w-6 h-6 text-ink" />
            <h3 className="font-serif text-xl font-bold">Write Programs</h3>
          </div>
          <p className="text-body">
            Define model behavior using Loom clauses: what it knows, how it speaks, and what it refuses.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-6 h-6 text-ink" />
            <h3 className="font-serif text-xl font-bold">Verify Behavior</h3>
          </div>
          <p className="text-body">
            The compiler measures real capabilities, searches the parameter space, and verifies your expectations.
          </p>
        </div>
      </section>

      <section>
        <h3 className="font-serif text-2xl font-bold mb-6">Example Programs</h3>
        <p className="text-body mb-8">
          Explore these built and verified Loom programs:
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            {
              name: "Clinic",
              description:
                "Medical reference assistant trained on held-out MedQuAD material",
            },
            {
              name: "Counsel",
              description:
                "Legal advisor that cites statutes and case law correctly",
            },
            {
              name: "Desk",
              description:
                "Reference desk staff that knows a library's collection",
            },
            {
              name: "Foreman",
              description:
                "Workplace safety overseer that enforces protocols",
            },
            {
              name: "Stylist",
              description:
                "Fashion advisor that maintains a consistent aesthetic",
            },
          ].map((prog) => (
            <div key={prog.name} className="card">
              <h4 className="font-serif font-bold mb-2">{prog.name}</h4>
              <p className="text-sm text-body">{prog.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
