# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A working directory for symbolic regression with [PySR](https://github.com/MilesCranmer/PySR)
on NCAR HPC (Casper/Derecho). Not a package and not a git repo.

**Reference material** (read-only inputs, not things to run):

- `pysrtips_bety260326.py` — the main body of accumulated PySR tuning lore, in its docstring.
  Read it before changing PySR hyperparameters anywhere. **Not runnable as-is**: it imports
  `data_loader_helper` and reads a `.nc` file, neither of which exists anywhere on this system.
- `pysr_demo.ipynb` — self-contained teaching notebook. Rediscovers `LH = C·U·(qs − qa)` from
  noisy synthetic data with two deliberate distractor features. Runs serial and fast.
- `Julio_Information.txt` — pointer to Julio Bacmeister's GW analysis at
  `/glade/work/juliob/HiRes_ana_dev/Drivers/Analysis` (external).
- Grundner et al. 2024 (JAMES) PDF — the methodological precedent: symbolic regression on data
  with sequential feature selection, and the finding that the discovered equation *transferred*
  better than the NN it competed with.

**The GW source-function work** (the active project) replaces the hand-built form in Julio's
`BasicAna4GWP-fitting.ipynb`, `τ = y0 + A·D^a/(1+D/D0) + B·D^c·P^b`, which scores r = 0.5784:

| file | role |
|---|---|
| `prep_gwsrc_data.py` | builds `gwsrc_data.npz` from Julio's netCDF + topo. Run in `npl-2026a` (the PySR env has no netCDF4). |
| `run_pysr_gwsrc.py` | the search. Modes `dp` / `tilt` / `full` select feature sets. |
| `submit_pysr.pbs` | PBS wrapper: `qsub -v MODE=tilt,NITER=2000 submit_pysr.pbs` |
| `scaling_experiment.py` | ratio vs z-score vs log input scaling, per the tips' instruction to test it |
| `nn_gwsrc.py` | MLP reference models per feature set — a skill-ceiling instrument, not a deliverable. Needs `credit-casper-modern` (torch + GPU). |
| `pysr_gwsrc_fit.ipynb` | the narrative: baseline, ceiling, screening, transfer test, results |

**The event-composite branch** compares Julio's MLP (`EventCompositeAna_NN-sweep.ipynb`,
`predictors.py`, `mlp_utils.py` in his Analysis dir) against the gridded work:

| file | role |
|---|---|
| `julio_event_vendor.py` | verbatim ports of his `dynamic_steer_launch`, `build_target`, `vorticity_centroid_levels`, `vtime_avg_events`. Vendored, not imported — see below. |
| `prep_event_data.py` | 90 GB of event pickles → the 13 MB `event_data.npz`. Run in `npl-2026a`. |
| `nn_event_gwsrc.py` | fits **his** `mlp_utils.fit_mlp_general`, scores it in log space. Needs `credit-casper-modern`. |
| `event_ceiling.py` | converts an event-framework r into the pointwise r it implies |
| `event_vs_grid.py` | the joint table, both frameworks on one axis |
| `smoothing_ladder.py` | **skill vs coarse-graining scale** — the one that settles what score to aim for |

Three traps in reaching his code. His `utils.py` uses PEP 701 nested-quote f-strings, so the
`predictors → analysis_utils → event_utils → file_utils → utils` chain needs Python ≥3.12 and
cannot be imported in the 3.11 torch env — hence the vendoring. The pickles are numpy-2 and
record `event_utils.AttrDict`, so unpickling needs numpy ≥2 (`npl-2026a`) plus a stub module
(`install_event_utils_stub`). And `mlp_utils.apply_mlp` moves its input to the device you pass,
while `fit_mlp_general` puts the model on cuda when it can — his notebook hard-codes `'cpu'` and
only works because it runs CPU-only. Pass `next(model.parameters()).device`.

### Two findings that should govern any further work here

1. **The functional form was never the bottleneck — at σ=2. At σ=0 it partly is.** Any `f(D, P)`
   is capped by within-cell variance (`distro_fitting.cell_spread`), and the cap moves with the
   scale (`baseline_s0.py`):

   | | var(log τ) | Julio's coeffs | same form, refit here | ceiling on any `f(D,P)` |
   |---|---|---|---|---|
   | σ=2, SH | 1.24 | 0.5784 | 0.5816 | 0.5843 |
   | σ=2, NH | 1.02 | 0.6309 | 0.6991 | 0.6992 |
   | **σ=0, SH** | 2.03 | **0.3258** | 0.3704 | **0.3952** |
   | **σ=0, NH** | 1.78 | **0.3189** | 0.4295 | 0.4389 |

   At σ=2 the hand-built form sits within 0.006 of a bound that holds for every possible `f(D,P)`
   — confirmed four ways (binned variance decomposition across bin resolutions, a gradient-boosted
   tree, an MLP, and PySR stalling at 0.58). **On a bare f09 column it scores 0.326 against a
   0.395 bound**, so ~0.07 is recoverable, and most of that is just refitting coefficients that
   were tuned at the wrong scale (0.326 → 0.370 from refitting alone). But the bound itself is
   low: `D` and `P` cannot carry a per-column source function. The six-predictor set reaches 0.514
   there (`smoothing_ladder.py`), so **at σ=0 the payoff is in predictors, not in the form.**

2. **Fit globally, or the feature-selection answer comes out backwards.** Trained on the SH
   alone, wind and height predictors (`spd_zs`, `z_steer`, …) buy ~0.19 of SH skill and drive
   NH R² *negative* — they look actively harmful, and only the tilt family transfers. Trained on
   both hemispheres (`PYSR_TRAIN=both`), those same predictors become the best available: NH r
   goes 0.652 → 0.761, calibration slope ~0.92. Same data, same architecture, opposite
   conclusion. Always report the NH column, and always record *what the model was fitted to* —
   `run_pysr_gwsrc.py` writes `train_on` into every front JSON because under `sh` the NH column
   is a transfer test and under `both` it is only a held-out-time test.

3. **Every r in this project is a statement about a coarse-graining scale, not about a model.**
   `smoothing_ladder.py` fits the same form-free tree on the same six predictors at a ladder of
   averaging widths on the f09 grid, and skill rises monotonically with the width:

   | target | var(log τ) | SH-test r | NH r |
   |---|---|---|---|
   | **bare f09 column** — what a CAM parameterization owes | 2.02 | **0.514** | 0.468 |
   | σ=1 | 1.56 | 0.608 | 0.645 |
   | σ=2 — the smoothing this whole directory inherits | 1.24 | 0.652 | 0.694 |
   | σ=3 | 1.00 | 0.671 | 0.682 |
   | σ=5 | 0.66 | 0.681 | 0.627 |
   | 11×11 box — Julio's event target | 1.01 | 0.666 | 0.697 |

   Roughly **+0.15 of r is available for free by averaging harder**, and none of the headline
   numbers — 0.578, 0.70, 0.761 — is measured at the scale CAM needs. Julio's event windows sit on
   this same f09 grid (dlon 1.25°, dlat 0.9424°), so an 11×11 window is ~10°×14°, about 121 CAM
   columns; and Julio's `sigma=(0,2,2)` smoothing, inherited from `BasicAna4GWP-fitting.ipynb`
   into `prep_gwsrc_data.py`, means the gridded work is not fitting a bare column either. Two
   independent routes agree his 0.761 is worth ~0.59–0.62 on a bare column: the variance ratio in
   `event_ceiling.py` gives 0.617, and rescaling by the tree's own 0.514/0.666 gives 0.59.

   **Decision (2026-08-17): σ=2 stays the target.** The curve flattens right at it — σ=2 → 11×11
   is +0.013 SH / +0.003 NH for 2.4× the averaging, inside the noise floor once synoptic
   autocorrelation is accounted for (SE(r) ≈ 0.013 at N_eff ~2000). And the NH *peaks* at σ=2
   and falls away beyond it (0.694 → 0.682 → 0.627 at σ=3, 5), so more smoothing starts costing
   transfer skill. Averaging harder past σ=2 buys nothing and gives up the interpretation.

   `GWSRC_SIGMA` defaults to 2; caches are always named `gwsrc_data_s{σ}.npz` and every front
   JSON records `sigma`, so a score can never be read without its scale. The σ=0 machinery and
   `gwsrc_data_s0.npz` stay put — σ=0 is the reference for what a strictly per-column
   parameterization could achieve (0.514 with six predictors, 0.395 for any `f(D,P)`).

   **Where the real headroom is, at fixed scale.** At the 11×11 scale a form-free tree on the
   gridded predictors gets 0.666, while Julio's event NN gets 0.761 — a **~0.10 gap at the same
   coarse-graining**, which no amount of extra smoothing addresses. Candidate causes, in the
   order worth testing: his predictors are time-averaged over 3 internal steps while ours are
   instantaneous; his target sits at a per-event *dynamic* launch level from a vorticity centroid
   rather than a fixed per-column `k_launch`; and he samples tilt at four level-ranges (ZS–ZL
   average, at ZL, at ZS, surface–ZS) where the gridded set collapses that vertical structure.

## Environment

Everything lives in one conda env carrying PySR 1.5.10 (Julia 1.12.6 + SymbolicRegression.jl 1.11.3
under `$CONDA_PREFIX/julia_env`):

```bash
source /glade/u/apps/opt/miniforge/default/etc/profile.d/conda.sh
conda activate /glade/work/wchapman/conda-envs/L96M2lines
```

In JupyterHub, select the **`PySR 1.5.10 (L96M2lines)`** kernel (kernelspec `pysr-latest`, wrapper
at `~/.local/share/jupyter/kernels/pysr-latest/launch.sh`).

**Always `conda activate` — never invoke the env's python by absolute path.** `juliapkg` resolves
its Julia project from `CONDA_PREFIX`; an unactivated launch inherits JupyterHub's prefix and dies
with `PermissionError` on the read-only `/glade/u/apps/jupyterhub/jh-23.11/julia_env`. The kernel
wrapper works around this by activating and pinning `PYTHON_JULIAPKG_PROJECT`.

Two related traps encoded in that wrapper: do **not** put `$CONDA_PREFIX/lib` on
`LD_LIBRARY_PATH` (the conda libstdc++ segfaults the bundled Julia), and keep `PYTHONNOUSERSITE=1`
so `~/.local/lib/python3.9/site-packages` doesn't shadow the env's numpy/scipy/sklearn.

First `from pysr import PySRRegressor` boots Julia: ~10 s warm, minutes if it must precompile.

`nn_gwsrc.py` needs a different env — torch 2.4.1 + CUDA:

```bash
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
```

and `prep_gwsrc_data.py` a third, for netCDF4: `conda activate npl-2026a`.

## Running

**Never size a run from `nproc` or `NCPUS`.** PBS sets `ompthreads=1`, which puts
`OMP_NUM_THREADS=1` in the environment, and GNU `nproc` honours that — it reports `1` inside a
32-core job, and a JupyterHub session shows `NCPUS=1` while actually holding 16 cores.
`len(os.sched_getaffinity(0))` is the truth; `run_pysr_gwsrc.py` uses it.

Related: `procs=` is **ignored** under `parallelism="multithreading"` (PySR warns and moves on).
Thread count comes from `PYTHON_JULIACALL_THREADS`, default `auto`, which resolves against the
affinity mask. `submit_pysr.pbs` pins it explicitly.

**The search budget is `niterations × populations`, not `niterations`.** At ~120 iterations/min on
32 cores, `niterations=2000` with `populations=96` needs ~27 h. Rather than shrink `niterations`
(the tips want it large), bound the wall clock with `timeout_in_seconds` / `PYSR_TIMEOUT` — PySR
returns its hall of fame on timeout and the scoring pass still runs. Keep it under the PBS
walltime.

`resources_used.cpupercent` is a lifetime average and reads low (~100) for the first few minutes
while Julia precompiles single-threaded. Check it again later before concluding threading is
broken; a healthy 32-core run shows ~2400–2600.

Set `output_directory=` and a distinct `run_id`/`RUN_TAG` per run, or concurrent searches of the
same mode overwrite each other's checkpoints.

## PySR tuning conventions used here

These are hard-won and specific; deviating from them will usually make results worse.

- **Scale inputs to O(1)–O(10) before fitting.** This is the single highest-leverage knob. The demo
  uses humidity in g/kg rather than kg/kg precisely because the kg/kg framing (constant ~3900 times
  a near-cancellation of two ~0.02 numbers) is what separates R²=0.87 from exact recovery.
  Scaling choices change the *form* of the recovered equation, not just its fit.
- **Prune the operator set to what the physics can contain.** Every unused operator enlarges the
  search space for nothing.
- **Experiment at large `populations`/`niterations` (O(1000)) early.** Small runs do not extrapolate
  — new regimes emerge at scale, including different variables becoming important.
- **Nonlinear operators need nested constraints**, e.g. `nested_constraints={"exp": {"exp": 0, ...}}`;
  keys must be unique. `constraints={"^": (-1, 1)}` keeps power arguments from ballooning. Prefer
  these over `complexity_of_operators`, which tends to eliminate operators wholesale.
- **The ratio `complexity_of_variables`/`complexity_of_constants` matters**, not each alone; the
  production script uses 2/1 for 8 inputs. Setting `complexity_of_variables=1` invites baroque
  variable combinations early.
- **`select_k_features` does not work well here.** To reduce variable count, raise
  `complexity_of_variables` and run long.
- **Batching (`batching=True`, `batch_size≈1000`) is required above ~10⁴ rows.** Longer runs with
  small batches beat short runs over all the data.
- Reading the result means reading the **Pareto front** (`model.equations_`) and finding the elbow —
  not just taking `model.get_best()`.
- **Fit the target the metric is computed on.** For the GW work the score is a log-space
  correlation, so the search fits `log(τ/Y_REF)` rather than `τ`. That optimises the metric
  directly and conditions a 4.5-decade target into O(1) at the same time.
- **Ratio scaling, not z-scoring**, for strictly-positive inputs — measured, not assumed
  (`scaling_experiment.py`): ratio 0.703, z-score 0.692, log 0.691. The z-scored run returned an
  expression with *no power law anywhere*, because sending a positive variable negative removes
  `x^a` from the reachable space. This is the tips' warning case, reproduced.
- **Floor variables with runaway dynamic range.** `precl` reaches 1e-19 of its reference; one such
  point inside a batch detonates any `log(P)` or `1/P` subexpression. Flooring at 1e-4 costs ~1.2%
  of the sample and still lets `P^b` vanish where it physically should.

Upstream references: https://astroautomata.com/PySR/tuning/ (some of it outdated) and
https://ai.damtp.cam.ac.uk/pysr/api/ for the full `PySRRegressor` signature.

## Should we distil a neural net?

`form_vs_ceiling.py` answers this by handing a form-free tree exactly the features,
scaling, split and target PySR got. A positive gap means the *equation* wins.

| mode | fit | feats | tree SH / NH | PySR SH / NH | gap SH / NH |
|---|---|---|---|---|---|
| dp | sh | 2 | 0.575 / 0.590 | 0.574 / 0.570 | −0.001 / −0.020 |
| dp | both | 2 | 0.607 / 0.653 | 0.569 / 0.652 | −0.038 / −0.001 |
| **tilt** | **sh** | 5 | 0.637 / 0.661 | 0.638 / **0.707** | +0.001 / **+0.047** |
| tilt | both | 5 | 0.663 / 0.708 | 0.635 / 0.703 | −0.028 / −0.005 |
| **full** | **both** | 8 | **0.730 / 0.735** | 0.695 / 0.698 | **−0.036 / −0.037** |

Two findings, and they pull in opposite directions:

- **With 8 features the symbolic search is ~0.036 short of form-free.** That is the *upper bound*
  on what distilling any teacher could buy on this problem, and it is real but modest.
- **In the one case that is a true transfer test** (`tilt`, SH-fit, so NH is genuinely unseen),
  the equation *beats* the tree by +0.047 in the NH. This reproduces Grundner et al. 2024: the
  flexible model's extra in-domain skill did not survive the transfer. Some of that 0.036 is
  probably SH weather, not physics.

**So: do not distil Julio's NN.** It is defined on a different population (19,888 detected events
with 11×11 windows, not all columns), it is trained on features CAM would have to compute anyway —
and if you can compute them you can hand them to PySR directly — and we have already shown that
particular net is suboptimal (`loss_power=5` costs it 0.03; SH-only fitting costs it 0.075).
Distilling it inherits all three.

**Do distil the gridded teachers** — this is the scoped version of the same idea, and
`distill_teachers.py` has cleared the one objection that would have killed it. A boosted tree is
piecewise constant, so its 0.730 could have been staircase memorisation that no smooth expression
could track. It is not: an MLP on the identical 8 features reaches **0.7333 / 0.7337**, matching
the tree's 0.7297 / 0.7340, and the two teachers agree with each other at r = 0.978. So a *smooth*
function of `D, Dl, Dg, P, Us, Ul, Zs, Zt` scoring ~0.733 demonstrably exists, and the direct-fit
front's 0.695 is genuinely ~0.038 short of it.

`run_pysr_distill.py {mlp|tree|data}` fits PySR to a teacher's predictions instead of the data;
`data` is the control, same script, same budget, same split, fitting the noisy target. Every
equation is scored against **real** data on held-out times and on the NH; `r_teacher` is a
diagnostic only. Submit with `submit_distill.pbs`.

**Result: it does not work.** Three completed 5 h runs, compared front-to-front at matched
complexity (`compare_distill.py`, n=31):

| run | best SH | best NH | mean SH cx≥25 | mean NH cx≥25 |
|---|---|---|---|---|
| data (control) | 0.7040 | 0.7094 | 0.6976 | 0.6898 |
| tree-distilled | 0.7047 | 0.7083 | 0.6984 | 0.6886 |
| mlp-distilled | 0.7036 | 0.7076 | 0.6945 | 0.6916 |

Paired mean differences against the control: tree ΔSH +0.0013±0.0011, ΔNH −0.0023±0.0025; mlp
ΔSH −0.0004±0.0011, ΔNH +0.0023±0.0013 — and those SEs are optimistic, since equations at
different complexities are far from independent. At matched complexity the three fronts score
*identically to four decimals* through cx=17, despite only 1–2 of 31 being the same string, so
they are finding algebraically equivalent expressions. Denoising the target changed nothing: the
~0.038 gap to the teachers is a **representational** limit of this operator set at maxsize=40, not
a fitness-signal limit.

**The calibration to remember: run-to-run variance of the PySR search is ~±0.01 here** — the
control scored 0.7040 where an earlier identical direct fit scored 0.6949. That is 5× any
distillation effect. Never read a 0.01 difference between two single runs as real; the only reason
this comparison could resolve anything is that it was paired across 31 complexities.

## Evaluating a candidate parameterization

Three metrics, and the second and third are the ones that decide:

- `r` in log space — comparable to Julio's `source_fit_refvals.score` and to the MLP sweep.
- `R²` alongside it. `r` is blind to a multiplicative offset; CAM needs the magnitude. The gap
  between `r²` and `R²` is the calibration error, and it is large in every model here including
  the incumbent. Julio's MLP sweep reports test r = 0.704 against test R² = 0.398 — that ~0.1 gap
  is recoverable skill, and its `|error|^5` loss was the suspected cause. **Confirmed and
  measured** (`nn_event_gwsrc.py`): switching `loss_power` 5 → 2 on his best predictor set takes
  SH-test r 0.761 → 0.795 and NH R² 0.418 → 0.491, with the NH calibration slope going 0.74 →
  0.88. Training on both hemispheres as well gives 0.790 SH / 0.785 NH at slopes 1.01 / 0.89 and
  essentially zero bias. Both are one-line changes to `MyHyperparameters`.
- **SH → NH transfer.** Split on contiguous *time blocks* (adjacent timesteps are strongly
  autocorrelated; a random split leaks), then score on the other hemisphere. Per
  `source_fit_refvals.compare_models`, a form that fits SH better and transfers worse has
  absorbed SH-specific collinearity rather than physics.

Two numerical traps that produced wrong answers here:

- **`np.polyfit` on ~10⁶ float32 values is badly conditioned** and silently returns a wrong slope
  (0.50 where the answer is 0.95), which then yields an `R²_cal` below `R²`. Cast to float64 and
  prefer the closed-form fit. `R²` after a best-affine correction must equal `r²` — assert it.
- **All generalization claims here rest on one month.** 247 three-hourly steps of DYAMOND August
  2016; the test block is the last 9.4 days, and the SH and NH test blocks are simultaneous. With
  σ=2 smoothing and synoptic correlation the effective sample size is far below 4.26M, so
  differences of ~0.01 in r are not meaningful.

## Figures

The demo notebook defines a small shared matplotlib style (recessive grid, neutral ink) so its
figures read as one system; `pysr_gwsrc_fit.ipynb` extends it to a four-hue categorical palette
(`#4269D0` blue, `#C77B26` amber, `#1A9E7F` teal, `#9B4FBF` purple) checked with the `dataviz`
validator for CVD separation and contrast in both light and dark. Match it when adding plots.
