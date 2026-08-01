"""Re-measure surgery ops with each model's OWN tokenizer."""
import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loom.stages.surgery import merge, prune_layers, lora

TEXT = [
    "The capital of France is Paris, which sits on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level pressure.",
    "She opened the book and began to read the first chapter slowly.",
    "Scientists have discovered a new species of insect in the rainforest.",
    "The train arrived at the station exactly on time this morning.",
]

@torch.no_grad()
def ppl(model, tok, device="cuda"):
    model.eval().to(device)
    tot, n = 0.0, 0
    for t in TEXT:
        ids = tok(t, return_tensors="pt").to(device)
        out = model(**ids, labels=ids["input_ids"])
        tot += float(out.loss); n += 1
    import math
    return math.exp(tot / n)

res = {"note": "every number uses the model's own tokenizer", "ops": {}}
t0 = time.time()

base_name, inst_name = "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(base_name)
a = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
b = AutoModelForCausalLM.from_pretrained(inst_name, dtype=torch.bfloat16)
pa, pb = ppl(a, tok), ppl(b, tok)
print(f"base {pa:.2f} instruct {pb:.2f}", flush=True)

merges = {}
for alpha in (0.25, 0.5, 0.75):
    m, _ = merge(a, b, method="linear", alpha=alpha)
    merges[str(alpha)] = round(ppl(m, tok), 3)
    del m; torch.cuda.empty_cache()
    print("merge", alpha, merges[str(alpha)], flush=True)
res["ops"]["merge"] = {"models": [base_name, inst_name], "method": "linear",
                       "ppl_base": round(pa, 3), "ppl_instruct": round(pb, 3),
                       "ppl_by_alpha": merges}

pr = {}
n_layers = len(a.model.layers)
for frac in (0.9, 0.75, 0.5):
    keep = list(range(int(n_layers * frac)))
    m = prune_layers(AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16), keep)
    pr[f"{int(frac*100)}pct"] = round(ppl(m, tok), 3)
    del m; torch.cuda.empty_cache()
    print("prune", frac, pr[f"{int(frac*100)}pct"], flush=True)
res["ops"]["prune_layers"] = {"model": base_name, "n_layers": n_layers,
                              "ppl_full": round(pa, 3), "ppl_by_kept_fraction": pr}
res["wall_clock_s"] = round(time.time() - t0, 1)
res["gpu"] = torch.cuda.get_device_name(0)
open("results/loom_surgery_demo.json", "w").write(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
