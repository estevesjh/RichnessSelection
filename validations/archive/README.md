# Archived validation scripts

Frozen-in-time exploratory scripts, superseded or one-shot. Kept for
the record; not maintained, not on any import path (`de.py` was moved
here from `src/richness_selection/`, so `de_ring_i1.py` will not run
as-is). Their `validations/cache/*.csv` artifacts remain in place for
notebook 03.

| script | why archived |
|---|---|
| `frozen_kernels.py` | Prototype of the frozen-physics algorithm (kernels, assembly, quad truth, pedagogical figs). Superseded by `src/richness_selection/frozen_bsel.py` (`FrozenSelBias`), `validations/frozen_bsel_validation.py`, and `docs/make_frozen_algorithm_figs.py`. |
| `frozen_fast.py` | Vectorized/cached variant of the prototype; its numerics were folded into `FrozenSelBias`. |
| `de.py` + `de_ring_i1.py` | Takahasi–Mori double-exponential quadrature experiment for the ring's twin peaks. Dead end: the frozen reformulation removes the peaks instead of integrating them harder. |
| `nz_bias_convergence.py` | One-time convergence study that pinned `Nz_bias=48` (already merged into `config.py`). |
| `frozen_mor_epsilon_stress.py` | MOR redshift-evolution-slope stress test for a section of the old frozen-note draft (now commented out of the tex). |
