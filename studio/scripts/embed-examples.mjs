#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { execSync } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const studioDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(studioDir, "..");
const examplesDir = path.resolve(repoRoot, "examples");
const resultsDir = path.resolve(repoRoot, "results");
const libDir = path.resolve(studioDir, "lib");

fs.mkdirSync(libDir, { recursive: true });

// On Vercel only studio/ is uploaded: the repo's examples/ and results/ are not
// there. The committed lib/*.json embeds are the source of truth in that case.
const repoPresent = fs.existsSync(examplesDir) && fs.existsSync(resultsDir);

if (!repoPresent) {
  console.log("repo dirs not present; keeping committed lib embeds");
  process.exit(0);
}

// Regenerate ISA and science JSON if loom package is available
try {
  process.env.PYTHONPATH = path.resolve(repoRoot, "src");
  execSync("python -m loom.isa --json", {
    cwd: studioDir,
    stdio: "inherit",
  });
  console.log("Regenerated ISA and science JSON");
} catch (e) {
  console.log("Could not regenerate ISA/science JSON; keeping committed files");
}

// Embed .loom example files
const examples = {};
const loomFiles = fs
  .readdirSync(examplesDir)
  .filter((f) => f.endsWith(".loom"));

for (const file of loomFiles) {
  const name = file.replace(".loom", "");
  const content = fs.readFileSync(path.resolve(examplesDir, file), "utf8");
  examples[name] = content;
}

fs.writeFileSync(
  path.resolve(libDir, "examples.json"),
  JSON.stringify(examples, null, 2)
);

console.log(`Embedded ${loomFiles.length} example files`);

// Build showcase from result JSON files
const showcase = [];
const byKey = new Map();
const buildFilePattern = /^loom_.*_build.*\.json$/;

try {
  const resultFiles = fs.readdirSync(resultsDir).filter(f => buildFilePattern.test(f));

  // One entry per (app, base model): passing builds beat failed ones, and the
  // archive clinic report (no substrate suffix) is superseded by the canonical ones.
  for (const file of resultFiles) {
    try {
      const filepath = path.resolve(resultsDir, file);
      const data = JSON.parse(fs.readFileSync(filepath, "utf8"));
      if (!data.passed) continue;   // the showcase carries verified builds only
      const key = `${data.app}::${data.base_model}`;
      const substrate =
        data.substrate ||
        (String(data.base_model || "").startsWith("scratch")
          ? "scratch"
          : "open_weight");
      byKey.set(key, {
        id: `replay-0`,
        substrate,
        // A model the compiler made has no upstream to compare against; saying so
        // is the difference between the two substrates the app must convey.
        made_here: substrate === "scratch",
        val_ppl: data.val_ppl ?? null,
        policy: data.policy || [],
        app: data.app,
        base_model: data.base_model,
        passed: data.passed,
        wall_clock_s: data.wall_clock_s,
        capabilities: data.capabilities,
        expectations: data.expectations,
      });
    } catch (e) {
      // Skip if file can't be parsed
    }
  }
} catch (e) {
  // Skip if results dir doesn't exist
}

for (const entry of byKey.values()) {
  entry.id = `replay-${showcase.length}`;
  showcase.push(entry);
}

fs.writeFileSync(
  path.resolve(libDir, "showcase.json"),
  JSON.stringify(showcase, null, 2)
);

console.log(`Built showcase with ${showcase.length} committed build(s)`);
