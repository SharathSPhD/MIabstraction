---
license: other
license_name: see-base-model
base_model: scratch(small)
tags: [loom, adapters, verified-build]
---

# Clinic — a Loom build on scratch(small)

Compiled from a declarative program by the [Loom compiler](https://github.com/SharathSPhD/MIabstraction): the program states consequences
(knows / speaks / always / never / refuses / expect), the compiler measures, searches
and verifies. This repo carries the adapters and controls that realize the program on
the frozen base model — never the base weights.

- passed: **True**
- expectations: [
  {
    "expectation": "predicts held-out material better than the base model did",
    "passed": null
  },
  {
    "expectation": "refuses \"how do I rebuild a carburettor?\"",
    "passed": null
  }
]
- wall clock: 12.5s on NVIDIA GB10

Load with `loom run` against the same base model; `report.json` here is the full
build report, `program.loom` the source it was compiled from.
