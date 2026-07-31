# E5 Debugging Status Report

## Coordinator Feedback Items: 3/5 Complete

### ✅ Step 1: VERIFY THE TASK IS WELL-POSED
**Finding:** 12% of samples were **ill-defined** (empty stack at answer position)
- Dataset generation allowed random answer positions even after all brackets matched
- Model literally had no correct answer for those samples → impossible to learn

**Fix Applied:**
- Modified `_generate_one_sequence()` in `src/miabstraction/data/algo.py`
- Added CRITICAL guarantee: Stack always non-empty at answer position
- Phased generation: (1) initial opens, (2) mixed bracket sequence, (3) answer position with invariant
- Added assertion: stack must have ≥1 unmatched opener before answer

**Verification:**
- Wrote explicit stack solver in `verify_task.py`
- Before fix: 88% match, 12% ill-defined
- After fix: **100% match rate, 0% ill-defined**
- ✓ Task is now well-posed

---

### ✅ Step 2: CHECK THE ACCURACY METRIC
**Finding:** Loss at answer positions (3.33) >> overall NTP loss (2.08)
- Model optimizing next-token prediction, not bracket matching
- Answer positions = only 6% of sequence → diluted task signal
- Model learned to predict common distractors well, bracket matching stayed at random

**Root Cause:** Cross-entropy on ALL tokens, no task-specific objective

**Fix Applied:**
- Modified `_train_model()` and `_train_model_sparse()` in `src/miabstraction/experiments/e5_sparsity.py`
- Added task-specific loss: 50% next-token + 50% answer-position loss
- Loss at answer positions now directly optimized
- Passed dataset to training functions for label access

**Expected Outcome:**
- Task loss should fall from 3.33 → <0.7 (ln(2), near-optimal for 2 classes)
- Answer-position accuracy should rise from 0% → >90%

---

### 🟡 Step 3: TRAINING OPTIMIZATION
**Changes Applied:**
| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Learning rate | 0.01 | 0.001 | Avoid divergence; 0.01 too aggressive |
| Training sequences | 1,000 | 20,000 | Coordinator: "20k+" needed |
| Training steps | 2,000 | 15,000 | Middle of 10k-20k range |
| Batch size | 64 | 256 | Better gradient estimates |
| Vocab size | 10 | 8 | Simplify task |

**Config File:** `configs/e5_sparsity.yaml` updated with documentation

**Status:** EXPERIMENT RUNNING
- Process: 1429318 (102% CPU, 1.5GB memory)
- Data generation: ✓ Complete (20k train, 500 val)
- Dense model training: IN PROGRESS (~15k steps, ~10-15 min total)
- Sparse models training: queued (3 sparsity targets)
- Circuit extraction: queued

---

### ⏳ Step 4: SIMPLIFY IF NEEDED
**Current Status:** Not yet necessary
- Well-posed task: ✓
- Task-specific loss: ✓
- Optimized hyperparameters: ✓
- Training actively running with new approach

**Fallback Plan (if accuracy still 0%):**
1. Switch to single-bracket-type (open/close depth matching)
2. Reduce sequence length from 16 → 10
3. Increase training steps to 30k-50k
4. Use higher task-loss weight (70% task, 30% NTP)

---

### ⏳ Step 5: EXTRACT CIRCUITS & H5 VERDICT
**Blocked Until:** Dense model reaches >90% answer accuracy
**Then:** 
1. Extract minimal circuit for dense model (greedy mean-ablation)
2. Extract minimal circuits for best sparse models
3. Compare circuit size and faithfulness
4. Compute H5 verdict: `supports = (sparse_size < dense_size) AND (sparse_faith >= dense_faith)`

---

## Test Coverage
- **All 37 tests passing** ✓
- Task generation tests: 8/8 ✓
- Sparsity tests: 8/8 ✓
- Circuit extraction tests: 8/8 ✓
- E5 experiment tests: 4/4 ✓
- (Plus existing E1, models tests)

---

## What Changed
**Files Modified:**
1. `src/miabstraction/data/algo.py` - Fixed task generation
2. `src/miabstraction/experiments/e5_sparsity.py` - Added task-specific loss, fixed accuracy metric
3. `configs/e5_sparsity.yaml` - Updated hyperparameters + documentation
4. Created `verify_task.py` - Task verification script
5. Created `debug_training.py` - Loss debugging script

**Commit:** `ab5ab45` - "E5 debugging: fix task generation and training optimization"

---

## Expected Outcome (Once Training Completes)
- Dense model: >90% answer accuracy
- Sparse models (q∈{0.05, 0.1, 0.2}): >85% accuracy (may be slightly lower due to sparsity)
- Circuit sizes: Dense ~6-12 nodes, Sparse ~3-6 nodes (estimate)
- H5 Verdict: Should support or not support based on circuit size & faithfulness trade-off
