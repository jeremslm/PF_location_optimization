# Convergence Window Analysis

## Problem

Bayesian and L-BFGS runs were collected with a fixed convergence window of 50
(stored in `convergence_03_25/01`). The convergence window controls when each
method stops:

- **L-BFGS**: stops after `window` consecutive completed starts show no relative
  improvement greater than 0.1%.
- **Bayesian phase 1**: stops after `window` consecutive GP-guided evals show no
  relative improvement greater than 0.1%.

The question is: what would the cost and eval-count outcomes look like if a
smaller window (25, 10) had been used? Smaller window = stops earlier = fewer
evals, potentially worse cost.


## What was done

### Retroactive replay (`notebooks/conv_window_analysis.ipynb`)

For L-BFGS and Bayesian phase 1, the per-step history needed to replay the
stopping criterion is saved in each `results.json`:

- `start_costs` / `start_boundaries` — best cost and cumulative eval count after
  each completed L-BFGS start.
- `bayesian_convergence_history` — running-best cost after each Bayesian phase-1
  eval (length = `n_bayesian_evals`).

The notebook walks these histories forward and fires the stopping criterion at
the first point where it would have triggered for a given window. This gives
exact eval counts and costs at stopping for both methods.

**Usage**: change `CONV_WINDOW` in the first cell (try 50, 25, 10) and re-run.
All tables and visuals update automatically.

### Limitation: Bayesian refinement cannot be replayed

The current notebook shows **phase-1 cost only** for Bayesian. The reason is
that phase 2 (L-BFGS refinement on acquisition candidates) depends on the GP
model state at the time phase 1 stops. If phase 1 stopped earlier (smaller
window), the GP would have had fewer observations, the acquisition function
would suggest different candidates, and the refinement results would be
different. The actual refinement costs in the JSON cannot be reused because they
were computed from the window-50 GP.


## Proposed fix

Add `x_history` (parameter vectors, phase 1 only) to the data saved by
`opt_comp_convergence.py`. This is manageable in size: for 6 coils, phase 1 is
~434 evals × 12 params = ~5K floats per run.

With `x_history` available, full replay for any window W becomes possible:

1. Truncate phase-1 cost + x history at the window-W stopping eval.
2. Sort candidates by cost, deduplicate in real (R, Z) space using the existing
   `_deduplicate_candidates` logic.
3. Run L-BFGS refinement on the top-N unique candidates.
4. The result is a genuine cost for window W, not an approximation.

This requires one rerun of all configs after the code change, but then any
window value is fully replayable without touching the optimizer.

### Code change needed

In `opt_comp_convergence.py`, inside `run_bayesian`, add to the saved result
dict (after phase 1, before phase 2):

```python
'x_history_phase1': [x.tolist() for x in self._x_history[:bayesian_evals]],
```

Then in `conv_window_analysis.ipynb`, the replay cell can:

1. Load `x_history_phase1` and `cost_history[:n_bayesian_evals]`.
2. Find the stopping eval for window W.
3. Take the lowest-cost unique candidates from `x_history_phase1[:stop_eval]`.
4. Run `scipy.optimize.minimize` (L-BFGS-B) from each candidate using the
   stored objective (requires the optimizer to be reconstructable, or the
   notebook to call the actual objective — this part needs further design).

### Outstanding design question

The refinement step calls the real objective function (TokaMaker). The notebook
currently only does post-processing on saved JSON data. To replay refinement,
the notebook would need access to the live objective, which means it would need
to run in the same environment as `opt_comp_convergence.py`. This is feasible
but adds complexity — worth deciding whether this belongs in the notebook or in
a separate script.