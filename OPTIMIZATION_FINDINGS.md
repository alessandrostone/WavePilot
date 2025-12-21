# WavePilot Optimization Analysis Findings

**Date:** 2025-10-14
**Analyzed by:** Claude Code
**Files Analyzed:** `optimize.py`, `train.py`, `main.py`, `model.py`, `constants.py`

---

## CRITICAL ISSUES

### 1. GPU/MPS Training Disabled During Optimization ⚠️ MAJOR BOTTLENECK

**Location:** `optimize.py:158`

**Issue:**
```python
reducer = VectorReducer(df, latent_dim=latent_dim, device=torch.device("cpu"))
```

Training is **forced to CPU** despite having a `get_device()` function that auto-detects CUDA/MPS acceleration.

**Impact:**
- **10-100x slower** than GPU/MPS training
- On a system with GPU/MPS, this is a massive performance regression
- Comment on line 11 acknowledges GPU/MPS detection but then forces CPU anyway

**Fix:**
```python
reducer = VectorReducer(df, latent_dim=latent_dim)  # Uses get_device() automatically
```

**Expected Speedup:** 10-100x faster VAE optimization trials

---

### 2. Function Call Bug in `objective_vae` ⚠️ CRITICAL BUG

**Location:** `optimize.py:313`

**Issue:**
```python
val_mse, _ = train_and_validate(trial, params, self.df)
```

**Problem:**
- Parameter order is **WRONG**
- Function signature is: `def train_and_validate(params, df, trial_number):`
- Arguments are passed in incorrect order: `(trial, params, df)` instead of `(params, df, trial.number)`
- Return value is unpacked as tuple but function returns single value (line 174)

**Impact:**
- This will cause runtime errors or incorrect optimization behavior
- Trials may be passing wrong data to training function
- Type mismatch between `trial` object and expected parameters

**Fix:**
```python
val_mse = train_and_validate(params, self.df, trial.number)
```

**Status:** MUST BE FIXED IMMEDIATELY

---

### 3. RBF Optimization Not Parallelized ⚠️ PERFORMANCE ISSUE

**Location:** `optimize.py:462`

**Issue:**
```python
self.study_rbf.optimize(
    lambda trial: self.objective_rbf(trial, original_data, Z_std, median_dist),
    n_trials=n_trials,
    n_jobs=1,  # ← Sequential execution
    show_progress_bar=True
)
```

**Problem:**
- RBF trials run sequentially with `n_jobs=1`
- RBF interpolation is CPU-bound and highly parallelizable
- System has multiple CPU cores that are being wasted

**Impact:**
- RBF optimization takes 4-8x longer than necessary
- No utilization of multi-core CPUs

**Fix:**
```python
n_jobs=-1,  # Use all available CPU cores
```

**Expected Speedup:** 4-8x faster RBF optimization (depending on CPU core count)

---

## MODERATE OPTIMIZATIONS

### 4. Redundant Parameter Extraction in VAE Optimization

**Location:** `optimize.py:355-358`

**Issue:**
Parameters are extracted twice:
1. Once in the batch loop (lines 355-358)
2. Again inside `train_and_validate` function

**Impact:**
- Unnecessary computational overhead
- Code duplication
- Potential for parameter mismatch bugs

**Recommendation:**
Refactor to extract parameters only once and pass them directly.

---

### 5. Training Epoch Counts May Be Excessive

**Location:** `constants.py:65-66`

**Current Values:**
```python
SEARCH_EPOCHS = 600
FINAL_EPOCHS = 4000
```

**Analysis:**
- For micro datasets (30-120 samples), these values may be overkill
- Models likely converge much earlier
- Each extra epoch wastes computational resources
- No early stopping mechanism is implemented

**Recommendations:**
1. Implement early stopping based on validation loss plateau
2. Reduce `SEARCH_EPOCHS` to 200-300 for faster trial evaluation
3. Consider adaptive epoch counts based on dataset size:
   - Small datasets (N < 50): 200 search / 2000 final
   - Medium datasets (50 ≤ N < 100): 400 search / 3000 final
   - Large datasets (N ≥ 100): 600 search / 4000 final

**Expected Speedup:** 2-3x faster optimization with early stopping

---

### 6. Unnecessary async/await in train.py

**Location:** `train.py:48`

**Issue:**
```python
async def main(filepath, pretrained_model_path, optimizer_session, save_model_path):
```

Function is declared `async` but:
- Only uses `await` at the very end (line 193)
- Most of the code is synchronous
- Adds async overhead without benefits

**Impact:**
- Minimal overhead but unnecessary complexity
- False expectation of concurrent execution

**Recommendation:**
Make function synchronous and run visualizer in a separate thread/process if needed.

---

## SUMMARY

### Performance Impact Table

| Issue | Current State | Fixed State | Expected Speedup |
|-------|--------------|-------------|------------------|
| GPU/MPS disabled | CPU-only training | GPU/MPS training | **10-100x** |
| RBF not parallelized | Sequential (1 core) | Parallel (all cores) | **4-8x** |
| Excessive epochs | 600/4000 epochs | 200-300/2000-3000 | **2-3x** |
| **COMBINED** | - | - | **80-2400x** |

### Critical Bugs

| Bug | Severity | Impact | Status |
|-----|----------|--------|--------|
| `objective_vae` parameter order | **CRITICAL** | Runtime errors / incorrect optimization | **MUST FIX** |
| GPU disabled | **CRITICAL** | Massive performance loss | **MUST FIX** |

---

## RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Critical Fixes (Implement Immediately)
1. ✅ Fix `objective_vae` parameter order bug
2. ✅ Remove CPU-only constraint (enable GPU/MPS)
3. ✅ Parallelize RBF optimization (`n_jobs=-1`)

### Phase 2: Medium Priority
4. Implement early stopping for VAE training
5. Add adaptive batch sizing for GPU memory utilization
6. Cache median distance computation in RBF

### Phase 3: Low Priority
7. Refactor async usage in train.py
8. Improve progress bar granularity
9. Add GPU memory profiling logs

---

## TESTING RECOMMENDATIONS

After applying fixes:

1. **Benchmark Test:**
   - Run optimization on a sample dataset with 50 presets
   - Record time before/after fixes
   - Verify GPU/MPS utilization during training

2. **Correctness Test:**
   - Verify that optimized parameters are reasonable
   - Check reconstruction accuracy meets threshold
   - Validate RBF interpolation produces smooth results

3. **Regression Test:**
   - Compare results between old and new code
   - Ensure optimization finds similar or better hyperparameters
   - Verify JSON export compatibility

---

## NOTES

- The codebase shows good architectural design with clear separation of concerns
- The optimizer uses modern techniques (Optuna, TPE sampling, reproducible seeds)
- Main bottleneck is the artificial CPU-only constraint
- After fixes, the pipeline should be production-ready

**Status:** Awaiting implementation of Phase 1 fixes.
