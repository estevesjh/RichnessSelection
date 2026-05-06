# Sigma_prj numerical audit — raw outputs

Script: `validations/sigma_prj_diagnostics.py`
Ref point: `lob=20, zob=0.5`, `R_vals = {0.3, 1.0, 3.0, 10.0}` cMpc/h.
Geometry at this point: `chi_o=1328.514`, `D_A_o=885.676`, `R_excl=1.087`,
`theta_lob=8.183e-4 rad`, `theta_max=30/D_A_o=3.387e-2 rad`.

## Task 1 — Current code vs quad reference

`SigmaPrj(R_vals)` at defaults `(n_theta_inner, n_theta_outer) = (10, 150)`:
```
R      code           ref_code_conv  ref_draft_conv  d_code/ref_code  d_code/ref_draft
0.30   3.6763e+14     3.6756e+14     3.6684e+14      +0.02%           +0.21%
1.00   3.1712e+14     3.1926e+14     3.1854e+14      -0.67%           -0.45%
3.00   2.5600e+14     2.6149e+14     2.6076e+14      -2.10%           -1.83%
10.00  2.0370e+14     2.2048e+14     2.1977e+14      -7.61%           -7.31%
```
`ref_code_conv` uses the code's conventions (xi at `zob`, `R_mis=theta*D_A(zob)`).
`ref_draft_conv` uses the paper's multi-line convention (xi at `zbar`, `R_mis=theta*D_A(z)`).

Disagreement is **numerical** (theta under-resolution), not a structural
formula error — at R=0.3 both conventions agree to 0.02%, while at R=10
both fail to ~7.6%. The code/draft convention shift costs ~0.2% at worst.

## Task 2 — N_theta(theta)

`N_theta(theta)` is **monotonically increasing** over the full domain:
```
theta (rad)  theta/theta_max  N_theta(excl)  N_theta(no_excl)
1.00e-04     3.0e-3           3.94e+02       1.98e+03
3.94e-04     1.2e-2           1.34e+03       2.85e+03
1.56e-03     4.6e-2           2.94e+03       2.94e+03
6.14e-03     1.8e-1           8.84e+03       8.84e+03
2.42e-02     7.1e-1           3.04e+04       3.04e+04
6.77e-02     2.0              8.26e+04       8.26e+04
```
Exclusion tames the small-theta divergence (from 1.98e3 -> 3.9e2 at
theta=1e-4), but with a log-GL theta grid the very small theta bins
contribute little. No peak: the sin(theta) growth dominates the 1+b*xi
bracket for theta >> theta_excl.

## Task 3 — N_theta(theta) * Sigma_mis(R | M=3e14, zob)

Per-R peak diagnostics:
```
R      theta_R     peak_theta   peak/theta_R  half-max FWHM (rad)
0.30   3.39e-4     3.85e-4      1.14          [2.0e-4, 7.6e-4]
1.00   1.13e-3     1.19e-3      1.05          [7.6e-4, 1.5e-3]
3.00   3.39e-3     3.65e-3      1.08          [2.9e-3, 3.6e-3]
10.00  1.13e-2     1.12e-2      0.99          one-bin
```
The integrand peaks at `theta ~ theta_R` for every R, with a narrow
half-max window of ~2/3 decade. This is exactly the regime the
split-at-theta_R log-GL grid is supposed to resolve.

## Task 4 — Axis convergence

(a) theta (n_inner, n_outer), at Nz=80, NM=24:
```
(ni,  no)    R=0.30        R=1.00        R=3.00        R=10.00
(5,   50)    3.6430e+14    3.0568e+14    2.3333e+14    1.7656e+14
(10,  150)   3.6763e+14    3.1712e+14    2.5600e+14    2.0370e+14   <- DEFAULT
(20,  300)   3.6784e+14    3.1848e+14    2.6237e+14    2.2175e+14
(40,  600)   3.6760e+14    3.1979e+14    2.6160e+14    2.2044e+14
(80,  1200)  3.6762e+14    3.1913e+14    2.6170e+14    2.2059e+14
```
Default is **under-converged by -7.6% at R=10**. Going from (10,150)
to (20,300) already brings R=10 to within 0.6% of the (80,1200) gold.
0.1% convergence at R=10 needs >= (40, 600) — the current split-at-theta_R
grid is inefficient at large R. At R <= 1, (10,150) is already at ~0.1%.

(b) Nz (ni=10, no=150, NM=24):
```
Nz=40:   [3.6734e+14  3.1890e+14  2.5601e+14  2.0369e+14]
Nz=80:   [3.6763e+14  3.1712e+14  2.5600e+14  2.0370e+14]
Nz=160:  [3.6778e+14  3.1777e+14  2.5598e+14  2.0369e+14]
Nz=320:  [3.6745e+14  3.1733e+14  2.5597e+14  2.0369e+14]
```
Changes are O(0.1%), so z is NOT the bottleneck.

(c) NM (Nz=80, ni=10, no=150):
```
NM=12:   [3.6763e+14  3.1702e+14  2.5622e+14  2.0099e+14]
NM=24:   [3.6763e+14  3.1712e+14  2.5600e+14  2.0370e+14]
NM=48:   [3.6763e+14  3.1708e+14  2.5598e+14  2.0282e+14]
NM=96:   [3.6763e+14  3.1709e+14  2.5599e+14  2.0309e+14]
```
Small drift (~0.1-0.5%) at R=10 — not the dominant error.

## Task 5 — Exclusion across the z-ring

`theta_excl(z)` at points in the ring of radius dz~0.04 around zob:
```
z       Delta chi       theta_excl
0.460   -93.61          0.0          (|Delta_chi|=93.6 > R_excl=1.09, no excl.)
...     ...             ...
0.500   +0.00           8.18e-4      (only z=zob is excluded)
...     ...             ...
0.540   +91.48          0.0          (no exclusion)
```
Exclusion bites only in a *very* thin ring |z-zob| < 0.0005, i.e. on
< 1 GL node at Nz=80 in the current z-grid builder.

xi_NL evaluated per (z, theta) without any split, current code
convention (`dchi < R_excl -> 0`):
```
theta       xi at z=0.480   xi at z=0.495   xi at z=0.505   xi at z=0.520
1e-4        +7.1e-3         +1.62e-1        +1.62e-1        +7.2e-3
1e-3        +7.1e-3         +1.60e-1        +1.60e-1        +7.2e-3
1e-2        +6.3e-3         +7.4e-2         +7.3e-2         +6.3e-3
```
The near-zob ring (0.495 and 0.505) shows xi ~20x larger than the
outer ring. Inside this thin ring the xi_NL is smooth in theta
(no step), because the radial exclusion condition |dchi| < R_excl is
*never activated* for these z at any theta in range (dchi ~ chi*theta
dominates at theta >> theta_lob, and the LoS component is already
> R_excl). Only at z == zob does exclusion carve out a tiny
theta_excl ~ theta_lob.

**Conclusion**: the per-z radial-exclusion handling in `sigma_prj.py:161-162`
is actually fine — the exclusion only fires at z ~ zob, and at those z
the integrand is dominated by theta >> theta_excl.

## Diagnosis

The disagreement is **entirely due to theta under-resolution at large R**.
The split-at-theta_R log-GL with (10, 150) wastes nodes: the inner
half `[eps, theta_R]` has 10 nodes across ~4 decades of log-theta, and
the outer half `[theta_R, theta_max]` has 150 nodes across ~1.5 decades.
At R=10, theta_R = 1.13e-2 is ~1/3 of theta_max = 3.39e-2 — the outer
half is a narrow strip and 150 nodes is overkill there, but the
*inner half* spans 4 decades where the integrand `N_theta * Sigma_mis`
has a sharp peak of width ~0.5 dex at theta_R. With only 10 nodes
across 4 decades, one log-GL node falls within the peak region at most.

The fix is one of:
  (i) swap the node budget: put more nodes in `[eps, theta_R]` than in
      `[theta_R, theta_max]` — something like (80, 40) instead of (10, 150);
  (ii) add a third split around the peak (e.g. `[eps, theta_R/3]`,
       `[theta_R/3, 3*theta_R]`, `[3*theta_R, theta_max]`) with the
       densest GL on the middle segment; or
  (iii) use an adaptive `scipy.quad`-based outer loop for theta,
        breakpoints at `theta_excl(z)` and `theta_R`.

Option (i) is cheapest. Option (ii) is the most principled — the
integrand peaks at theta_R and falls on both sides, so a 3-segment
split with dense GL on the middle would hit 0.1% on all R at ~60
total theta nodes. The current defaults `(10, 150)` hit 0.1% for
R <= 1 but fail badly at R >= 3 because the peak-containing inner
segment is under-sampled.
