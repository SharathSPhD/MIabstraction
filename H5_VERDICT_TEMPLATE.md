# H5 Hypothesis Verdict: Weight Sparsity → Smaller Faithful Circuits

## Hypothesis (H5 from SPEC.md)

> Training with weight sparsity yields a **smaller, more faithful** minimal circuit 
> for an algorithmic task than a matched dense model, at similar task performance.

## Verdict Logic

**Supports H5 if and only if:**
1. `circuit_size_sparse < circuit_size_dense` (sparse circuit is smaller)
   AND
2. `faithfulness_sparse >= faithfulness_dense` (sparse circuit is at least as faithful)

**Falsifies H5 if:**
- Sparse circuit is NOT smaller, OR
- Sparse circuit IS smaller but significantly less faithful

---

## Definitions

### Circuit Size
- Count of nodes: (layer, head) for attention + (layer, -1) for MLPs
- Example: 3-layer model with 2 heads each = max 3×(2+1) = 9 nodes

### Faithfulness  
- Definition: `accuracy_with_circuit / accuracy_full`
- Calculation: (1) Full model accuracy on validation set, (2) Replace all non-circuit nodes with batch-mean, (3) Measure accuracy on ablated model
- Range: [0, 1], where 1.0 = perfect circuit (no information loss)
- Interpretation: Faithfulness >= 0.95 means circuit retains 95%+ of full model's capability

---

## Expected Results (if training reaches >90% accuracy)

### Scenario 1: Sparse Smaller & Faithful (H5 SUPPORTED)
```
Dense:  circuit_size=8, accuracy=0.95, faithfulness=0.85
Sparse: circuit_size=5, accuracy=0.92, faithfulness=0.87
Verdict: SUPPORTS H5 ✓
Reasoning: Sparse smaller (5<8) and more faithful (0.87>0.85)
```

### Scenario 2: Sparse Smaller but Unfaithful (H5 FALSIFIED)
```
Dense:  circuit_size=8, accuracy=0.95, faithfulness=0.85
Sparse: circuit_size=4, accuracy=0.90, faithfulness=0.65
Verdict: FALSIFIES H5 ✗
Reasoning: Sparse smaller but less faithful (0.65<0.85)
          → sparsity forced removal of important neurons
```

### Scenario 3: Sparse Same Size (H5 FALSIFIED)
```
Dense:  circuit_size=8, accuracy=0.95, faithfulness=0.85
Sparse: circuit_size=8, accuracy=0.93, faithfulness=0.88
Verdict: FALSIFIES H5 ✗
Reasoning: Sparse not smaller (8=8), so H5 requirement fails
          → weight sparsity didn't reduce circuit complexity
```

---

## Interpretation & Leak Budget

### If H5 Supported:
- Weight sparsity is a valid mechanism for discovering minimal circuits
- Sparser weights → clearer feature roles → smaller sufficient circuits
- Leak Budget: < 0.15 (faithfulness >= 0.85 means < 15% information loss)

### If H5 Falsified:
- Weight sparsity alone doesn't guarantee circuit minimality
- Possible: (1) sparsity removes important nodes, (2) circuits naturally similar for task
- Leak Budget: >= 0.25 (faithfulness < 0.75 suggests unreliable circuit)

---

## Next Steps (Upon Completion)

1. **Extract Dense Circuit:**
   - Run greedy mean-ablation with 95% threshold
   - Record circuit size and faithfulness

2. **Extract Sparse Circuits (best q):**
   - Identify best sparse model (lowest loss or highest accuracy)
   - Extract circuit same way
   - Record size and faithfulness

3. **Compare:**
   - Size: sparse vs dense
   - Faithfulness: sparse vs dense
   - Apply verdict logic

4. **Update results/e5_sparsity/result.json:**
   - `supports: true/false`
   - `circuit_size_dense`, `circuit_size_sparse`
   - `faithfulness_dense`, `faithfulness_sparse`
   - `accuracy_dense`, `accuracy_sparse`
   - `leak_budget` = 1.0 - faithfulness_sparse

5. **Commit on exp/e5-sparsity branch:**
   - Include result.json and pareto.png
   - Include final H5 verdict in commit message
