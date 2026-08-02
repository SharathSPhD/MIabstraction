"use client";

import examples from "@/lib/examples.json";

export default function LanguagePage() {
  const clauses = [
    {
      name: "knows",
      desc: "Trained capability from a corpus. The model learns this material.",
      example: 'knows from "data/medical/corpus.txt"',
      becomes: "Knowledge capability",
    },
    {
      name: "speaks",
      desc: "Communication style clauses: how the model should express itself.",
      example: 'speaks formal, careful',
      becomes: "Style capability",
    },
    {
      name: "always",
      desc: "Mandatory behavior: something the model must do in every relevant case.",
      example: 'always says when a question needs a real doctor',
      becomes: "Invariant capability",
    },
    {
      name: "never",
      desc: "Hard prohibition: the model must not do this under any circumstance.",
      example: 'never gives a diagnosis',
      becomes: "Prohibition capability",
    },
    {
      name: "refuses",
      desc: "Explicit guardrail: the model must refuse requests of a certain kind.",
      example: 'refuses questions that are not about health',
      becomes: "Guardrail capability",
    },
    {
      name: "expect",
      desc: "Acceptance test: how you measure whether the compiled model works.",
      example: 'expect refuses("what do you charge for a consultation?")',
      becomes: "Verification rule",
    },
  ];

  const knobs = [
    {
      name: "adaptation",
      desc: "How much the model can be changed. Range: 1–8. Higher = more parameter updates allowed.",
      example: "tune adaptation from 1 to 6",
    },
    {
      name: "steering",
      desc: "Multiplier strength for behavioral control. Range: 0.5–4. Higher = stronger control signals.",
      example: "tune steering from 0.5 to 3",
    },
    {
      name: "insistence",
      desc: "Minimum fraction of untrained gap to recover. 0.25 = recover a quarter; 0.5 = recover half.",
      example: "tune insistence from 0.25 to 0.5",
    },
    {
      name: "effort",
      desc: "Search intensity: balanced, thorough, or exhaustive. Higher = more configurations tried.",
      example: "effort thorough",
    },
  ];

  const clinicExample = examples.clinic || "";

  return (
    <main className="bg-night-950">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="mb-16">
          <h1 className="font-display text-6xl font-bold text-slate-100 mb-4">
            The Loom Language
          </h1>
          <p className="text-lg text-slate-400">
            A declarative syntax for specifying language model behavior. No learning rates, no
            layer indices, no optimizer choice—only what the model must know and do.
          </p>
        </div>

        {/* Clauses */}
        <div className="mb-16">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-6">Clauses</h2>
          <div className="space-y-3">
            {clauses.map((clause) => (
              <div key={clause.name} className="card p-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-display text-lg font-bold text-gold-300 mb-1">
                      {clause.name}
                    </h3>
                    <p className="text-slate-300 text-sm mb-3">{clause.desc}</p>
                    <div className="font-mono text-xs text-slate-400 bg-night-900/50 p-3 rounded border border-night-600/50">
                      {clause.example}
                    </div>
                  </div>
                  <div>
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">
                      Becomes
                    </p>
                    <p className="text-slate-100 font-medium">{clause.becomes}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-night-600/50 my-16" />

        {/* Tune knobs */}
        <div className="mb-16">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-6">
            Tune Knobs: The Budget
          </h2>
          <p className="text-slate-400 mb-6">
            You specify how much change is acceptable and how hard the compiler can search.
            These are not hyperparameters—they are your consequence constraints.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            {knobs.map((knob) => (
              <div key={knob.name} className="card p-4">
                <h3 className="font-display text-sm font-bold text-gold-300 uppercase tracking-wider mb-2">
                  {knob.name}
                </h3>
                <p className="text-slate-300 text-sm mb-3">{knob.desc}</p>
                <div className="font-mono text-xs text-slate-400 bg-night-900/50 p-2 rounded border border-night-600/50">
                  {knob.example}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-night-600/50 my-16" />

        {/* Example program */}
        <div className="mb-8">
          <h2 className="font-display text-3xl font-bold text-slate-100 mb-6">
            Full Example: Clinic
          </h2>
          <p className="text-slate-400 mb-6">
            A medical reference assistant. The program specifies what it knows, how it speaks,
            what it refuses, and how to measure success. The compiler chooses everything else.
          </p>
        </div>

        <div className="card p-6 mb-16">
          <pre className="text-xs leading-relaxed overflow-x-auto">
            <code className="text-slate-300">{clinicExample}</code>
          </pre>
        </div>

        {/* Refusal rules */}
        <div className="card p-6 border-gold-600/50 bg-gold-600/5">
          <h3 className="font-display text-lg font-bold text-gold-300 mb-4">
            Refusal Rules (Parser Enforcement)
          </h3>
          <ul className="space-y-2 text-slate-300 text-sm">
            <li className="flex gap-2">
              <span className="text-gold-400">•</span>
              <span>
                <strong>Ungated capabilities refused at parse time:</strong> A knows/speaks
                without a from/how clause or a target base_model will be rejected.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-gold-400">•</span>
              <span>
                <strong>Vacuous gates refused:</strong> A refuses clause with no specific
                topic or a never clause with no clear target is rejected.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-gold-400">•</span>
              <span>
                <strong>Corpora must be manifested:</strong> A knows clause must have a
                committed path or reference; inline corpus text is not allowed (so the
                training set is auditable).
              </span>
            </li>
          </ul>
        </div>

        <div className="h-8" />
      </div>
    </main>
  );
}
