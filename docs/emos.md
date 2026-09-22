# EMOS — Ensemble Model Output Statistics

EMOS (also called *nonhomogeneous Gaussian regression*, NGR) is the parametric
marginal calibrator in AutoUQ. This note describes what it is, the model it fits,
the parameter constraints we impose, how the `per=` grouping changes the shape of
the fit, and how our choices relate to standard practice in the post-processing
literature.

Implementation: `src/autouq/calibrators/emos.py` (class `EMOS`) and the shared
axis-grouping helpers in `src/autouq/calibrators/grouping.py`.

---

## 1. What it is

EMOS takes a *raw ensemble forecast* and returns a *calibrated Gaussian
predictive* at each site. It is pure post-processing — it never touches the
underlying forecast model.

- **Input.** `calibrate(true, pred)` where `pred` is the raw ensemble of
  shape `(B, T, *S, C, M)` (batch, time/lead, spatial, channel, members) and
  `true` is the matching observations `(B, T, *S, C)`. `predict` / `sample`
  then take a fresh `pred` of the same forecast shape.
- **What it does.** It collapses the member axis `M` into two per-site summaries
  — the ensemble **mean** `x̄` and **sample variance** `s²` — and fits an affine
  map from those to a Gaussian predictive `N(μ, σ²)`.
- **Role.** `EMOS` is a `SamplableCalibrator`: `predict` returns central
  intervals, `sample` draws an ensemble from the calibrated marginals. It is a
  **marginal** calibrator — it models each site independently and does **not**
  capture cross-site or cross-lead dependence. To restore joint structure, compose
  it with ECC (the ready-made `EMOSECC` pipeline).
- **Assumption.** The predictive is forced Gaussian, so EMOS fits continuous,
  roughly symmetric quantities (e.g. temperature) well, but not bounded/skewed
  ones (e.g. precipitation) without a transform.

---

## 2. The model and parameter constraints

For each fitted group (see §3), EMOS fits

```
μ_cal   = β₀ + β₁ · x̄              (mean: affine in the ensemble mean)
σ²_cal  = γ₀ + γ₁ · s²             (variance: affine in the ensemble variance)
```

subject to `γ₀, γ₁ ≥ 0`, by minimising the mean **closed-form Gaussian CRPS**
over the calibration set. Every group is solved to its own optimum (see "Fitting"
below). It subsumes pure variance inflation as the constrained special case
`β₀=0, β₁=1, γ₀=0` (we do **not** currently expose that as a fitting mode — see §6).

### Non-negativity on each coefficient

We enforce `γ₀, γ₁ ≥ 0` as bounds on **each coefficient**:

```
σ²_cal = γ₀ + γ₁ · s²,   γ₀ ≥ 0, γ₁ ≥ 0
```

This is deliberately **not** a single link wrapped around the whole expression,
such as `softplus(γ₀ + γ₁·s²)`. Constraining each coefficient is better for this
model class for three reasons:

1. **It stays canonical EMOS.** The textbook model (Gneiting et al. 2005) makes
   the variance *affine* in `s²`. Our form is exactly that; the wrap-the-sum form
   is a nonlinear, saturating transform of an affine argument — a different model.
2. **It enforces the correct spread→variance relationship.** With the wrap-the-sum
   form, the slope is `dσ²/ds² = sigmoid(γ₀+γ₁s²)·γ₁`, whose sign equals the sign
   of `γ₁`. Nothing stops the optimizer choosing `γ₁ < 0`, i.e. predictive
   variance *decreasing* as the ensemble spreads out — physically backwards. The
   bound `γ₁ ≥ 0` keeps the slope non-negative.
3. **It is interpretable.** `γ₀` is a variance floor (uncertainty even when the
   ensemble collapses); `γ₁` is the spread multiplier. They stay separate.

Earlier versions imposed the same constraints through a per-coefficient softplus,
`softplus(γ₀) + softplus(γ₁)·s²`. Its gradient vanishes as a coefficient
approaches zero (`softplus′(r) ≈ eʳ`), so a coefficient pushed towards zero during
the fit could not come back even when its optimum was well above zero. With bounds,
a coefficient can go to zero where zero is optimal (a common case for `γ₀`) without
freezing on the way.

The wrap-the-sum / single-link idiom *is* the right choice in **neural**
distributional regression (e.g. Rasp & Lerch 2018 use `exp(·)`), where the
standard deviation is the output of a flexible network and there are no separate
`γ₀, γ₁` to constrain. That is a different model class from the affine EMOS here.

### Fitting: every group to its own optimum

The groups are independent problems of four coefficients each, and the objective is
their sum. `calibrate` solves them all at once but each to its own convergence
(`autouq.calibrators.optimize.minimize_per_group`). Every group gets its gradient
and 4×4 Hessian in closed form (the Gaussian CRPS has closed-form derivatives,
`mathutils.gaussian_crps_derivatives`), then takes a Newton step on a
positive-definite model of that Hessian (scaled by its diagonal, eigenvalues
replaced by their absolute values), in which no variance coefficient moves more than
90% of the way to zero, and backtracks until its CRPS decreases enough. Iterates
therefore stay inside the bounds, and a coefficient whose optimum is zero approaches
it geometrically. A group stops when the predicted decrease of its step falls below
`tol` (default `1e-10`) times its mean CRPS. `max_iter` (default 100) caps the
iterations; groups that miss `tol` within it trigger a `RuntimeWarning` naming how
many.

Earlier versions ran one L-BFGS over all groups' coefficients together, on the mean
CRPS pooled over groups. That couples the groups' stopping rules, iteration budget
and curvature model: the pooled loss change drops below the fixed tolerance while
individual groups are still far from their optimum, and groups that differ from
each other slow one another down. On 100 lead times with widely different
coefficients, about half the leads ended more than 1% above their optimum and far
from nominal coverage. Solving each group separately to a relative tolerance brings
every lead to its optimum (the regression test in `tests/test_emos.py` checks this).

### Other numerical details

- The fit runs in float64 whatever the input dtype; the fitted coefficients are
  cast back to the forecast's dtype and device at `predict`/`sample` time.
- Inputs are standardised per group, and the ensemble variance is additionally
  scaled to mean one per group, so every coefficient is of order one during the
  fit. (After standardising by the truth's spread alone, the ensemble variance of a
  sharp forecast can be around 1e-4, leaving `γ₁` badly scaled.)
- A small floor `_VAR_FLOOR = 1e-12` is added inside the `sqrt` for strict
  positivity / numerical safety (it is *not* a learned floor).
- Initialisation: `β₀=0, β₁=1`, with the predictive variance matched to the raw
  ensemble mean's mean squared error and split evenly between `γ₀` and `γ₁·s²`.
- **Divergence guard.** If the fit produces non-finite coefficients it raises
  `RuntimeError` rather than silently storing `NaN`s.

---

## 3. Per-group fitting and the shape change

### How `per=` works

`per=` names the axis *roles* that each get their **own** coefficients; every
other axis is **pooled** (shares one fit). Roles are an `AxisRole` `StrEnum`
(`TIME`, `SPACE`, `CHANNEL`) — strings like `"time"` and `AxisRole.TIME` are
interchangeable. `SPACE` expands to *every* spatial axis. **Batch (axis 0) is
always pooled.** Default is `per=("time",)` — one fit per lead time, the classic
per-lead EMOS.

- **Number of affine "sets"** = product of the sizes of the grouped axes. Each set
  is 4 scalars (`β₀, β₁, γ₀, γ₁`).
- **Samples per group** = product of the sizes of the *pooled* axes, **including
  batch `B`**. This is the data each set is fit from — the number that governs
  overfitting.

### Per-group standardisation

Before fitting, inputs are standardised per group: the location (mean) and scale
(floored std) are estimated from `true` over the pooled axes
(`group_location_scale`), the fit runs on the standardised `O(1)` quantities, and
the result is de-standardised on output. This is algebraically equivalent to
fitting on raw values — it only conditions the optimisation so it behaves well at
any data scale (the ensemble variance is also scaled to mean one per group; see
§2). Each group needs **at least 2 pooled samples** (so `B ≥ 2`
for the full cross) or `group_location_scale` raises `ValueError`.

### Worked example — `64×64` grid, 3 channels, 100 lead times

Working layout `(B, 100, 64, 64, 3)`. Verified counts:

| `per=` | affine sets | samples / group | meaning |
|---|---:|---:|---|
| `("time",)` | 100 | `12,288·B` | per lead (channels pooled) |
| `("time","channel")` | 300 | `4,096·B` | per lead, per variable |
| `("space",)` | 4,096 | `300·B` | per pixel (channels pooled) |
| `("space","channel")` | 12,288 | `100·B` | per pixel, per variable |
| `("time","space")` | 409,600 | `3·B` | per (lead, pixel) |
| `("time","space","channel")` | 1,228,800 | `B` | per (lead, pixel, variable) — pools only `B` |

### Are the sets fit separately?

Yes, statistically and numerically, in one batched computation. The four
coefficient tensors carry one entry per group. Because the loss is a sum of
per-element terms and each element uses only its own group's coefficients, a
group's gradient and Hessian depend **only on that group's data**, so each group
takes its own Newton step, its own line search and its own stopping test (§2).
So:

- There is **no sharing of statistical strength** across groups (no smoothing).
- Groups that differ from each other do not hold each other back; the result for
  a group is the same as fitting that group alone (a regression test checks this).
- Each Newton iteration costs one full-tensor pass for the CRPS and its closed-form
  derivatives, plus the line-search passes. No autograd graph is kept, so memory
  stays a small multiple of the float64 working tensors.

---

## 4. Standard practice (literature)

There is no "temporal vs spatial correction" split in the EMOS literature. The
standard recipe is a single **local** model fit per **(location, lead time,
variable)**, pooling only over a rolling window of past forecast cases:

- **Local vs global.** EMOS-loc fits separate parameters per location; EMOS-gl
  pools all locations into one parameter set. Global "cannot correct
  location-specific deficiencies"; local **improves considerably** on global (one
  temperature study: CRPS 1.42 local vs 1.79 global). Local is the gold standard
  when data permits.
- **Per lead time, separately.** Even the neural successor (Rasp & Lerch 2018)
  fits "a single model per lead time".
- **Per variable, separately.** Different physical variables get their own models.
- **Pooled axis = training archive.** Coefficients are estimated over a rolling
  window of past dates/cases. Windows of ~20–40 days are common; up to ~720 days
  for temperature. So the field routinely fits 4 parameters from anywhere between
  ~20 and a few hundred samples per group.

**Mapping to our axes.** `B` (batch) plays the role of the training archive, so
standard local EMOS = `per=("time","space","channel")`, pooling only over `B`.
That is the `1,228,800`-set row above — the literature's default, *provided `B` is
a sufficient archive*.

**Middle grounds for small samples / dense grids.** Because per-(location, lead)
fitting is data-hungry, the field developed regularised compromises we do **not**
yet implement:

- *Semi-local* estimation — pool "similar" locations (Lerch & Baran 2017).
- *Lead-time-continuous* postprocessing — smooth coefficients across lead times
  instead of fitting each independently (Wessel et al. 2024).
- *Smooth EMOS (SEMOS)* and explicit L2 penalties on coefficients (Jobst 2024).

Our `per=` only offers the discrete pool / don't-pool lever, not these smooth
in-between options.

---

## 5. Practical guidance

The decision is governed by `B` (independent training cases) and how dense the
grid is. A `64×64` grid is 4,096 "locations" — far denser than the tens-to-
hundreds in station studies — so the favourable archive/location ratio is harder
to hit.

- **Channel almost always goes in `per=`.** Distinct variables have distinct
  scales/biases; pooling them blends unrelated quantities in the standardisation.
  Only drop `channel` if the channels are genuinely the same quantity.
- **Put an axis in `per=` only where bias or spread genuinely varies; pool the
  rest.** Then sanity-check `samples/group = (product of pooled axis sizes) × B`
  stays comfortably above a few dozen.

| Situation | Suggested `per=` | Rationale |
|---|---|---|
| `B` large (hundreds+) | `("time","space","channel")` | Standard local EMOS — best skill |
| `B` moderate | `("space","channel")` or `("time","channel")` | Pool the axis with weaker structure |
| `B` small (tens) | `("time","channel")` (+ L2 once available) | Global variant; regularise |

The honest one-liner: **standard practice is *local* — `per=("time","space",
"channel")` when data allows — and pooling an axis is a deliberate, data-driven
compromise**, not the default.

---

## 6. What we have vs. what is future work

**Implemented**

- Configurable per-axis fitting (`per=`) over time / space / channel, any
  combination, via the `AxisRole` enum.
- Per-group standardise → fit → de-standardise (scale-invariant conditioning).
- Per-coefficient non-negativity bounds on the variance coefficients, with every
  group solved to its own convergence and a warning for any group that is not.
- Reproducible sampling via `sample(generator=)`.
- Guards: alpha validation, non-finite coefficients, degenerate per-group scale,
  device-portable fitted parameters.
- `EMOSECC` — the EMOS-marginals + ECC-dependence pipeline.

**Not implemented (candidate future work, roughly in priority order)**

1. **L2 / ridge penalty** on coefficients — cheapest, generic regularizer; the
   most useful guard for fine-grained `per=`. (This is what SEMOS/tsEMOS do.)
2. **Constrained-fit mode** — freeze `β₀=0, β₁=1, γ₀=0` to fit only a spread
   multiplier; a robust tiny-sample fallback.
3. **Spatial smoothing / semi-local pooling** — borrow strength across nearby or
   similar locations rather than independent per-pixel fits.
4. **Lead-time-continuous coefficients** — smooth across lead times.
5. **Non-Gaussian predictives** — for bounded/skewed variables.

---

## References

- Gneiting, Raftery, Westveld, Goldman (2005). *Calibrated Probabilistic
  Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.*
  Monthly Weather Review.
- Rasp, Lerch (2018). *Neural Networks for Postprocessing Ensemble Weather
  Forecasts.* Monthly Weather Review.
- Lerch, Baran (2017). *Similarity-based semi-local estimation of EMOS models.*
- Wessel et al. (2024). *Lead-time-continuous statistical postprocessing of
  ensemble weather forecasts.* QJRMS.
- Jobst et al. (2024). *Time-series-based Ensemble Model Output Statistics for
  temperature forecasts postprocessing.* QJRMS.
- Vannitsem et al. (2021). *Statistical Postprocessing for Weather Forecasts —
  Review, Challenges and Avenues in a Big Data World.*
