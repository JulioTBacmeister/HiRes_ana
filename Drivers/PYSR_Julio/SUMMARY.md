# Gravity-wave source function: symbolic regression results

W. Chapman, 2026-08-26. Data: CAM cam77_dyamond1_prod1, DYAMOND August 2016,
247 three-hourly steps on the f09 grid (dlon 1.25 deg, dlat 0.9424 deg).
Target `epwp_zl`, ocean columns with smoothed MXDIS < 200 m.
SH band 80S-20S is the fitting hemisphere, NH 0-80N the transfer test.

## Definitions

Terms used throughout this document, in the order they first matter.

**Symbolic regression.** A search over algebraic expressions, rather than over
the coefficients of a fixed formula. PySR (v1.5.10, SymbolicRegression.jl
1.11.3) evolves a population of candidate expressions built from a chosen set
of operators, here `+ - * / ^` and `exp log sqrt square`. The output is not one
equation but a set of them, one per complexity level.

**Complexity (cx).** The size of an expression, counted in nodes of its
expression tree. Each variable costs 2, each constant 1, each operator 1. The
search was capped at cx = 40. For scale, the recommended equation below is
cx = 17 and Julio's hand-built form has 7 free parameters.

**Pareto front.** The set of equations that are the most accurate found at each
complexity. Reading a symbolic regression result means reading this front and
locating the elbow, the complexity past which further terms buy little
accuracy, not taking the single best-scoring equation.

**Gridded framework.** The setup used for the equations in this document. One
sample is one f09 column at one timestep: about 4.3 million samples per
hemisphere. This is the setting a CAM parameterization runs in, since CAM must
supply a source in every column.

**Event framework.** Julio's alternative setup, used by his MLP. Detected
gravity-wave events are extracted from the same simulation, and one sample is
one event: predictors and target are averaged over an 11 by 11 window of f09
columns (roughly 1000 km) and over three internal timesteps, at a launch level
diagnosed per event from a vorticity centroid. There are 19,888 SH and 7,594 NH
events. Scores from the two frameworks are not directly comparable because the
targets differ.

**Coarse-graining, sigma.** The width, in f09 grid cells, of a Gaussian blur
applied to every field, target and predictors alike, before fitting. Sigma = 0
leaves bare columns. Sigma = 2, inherited from
`BasicAna4GWP-fitting.ipynb`, replaces each value with a weighted mean of about
50 neighbouring columns, roughly 180 by 210 km at 50 deg latitude. The number of
samples is unchanged; only the smoothness of the field changes. Skill depends
strongly on this choice, so every score here is quoted with its sigma.

**Log-space r.** The Pearson correlation between log(predicted) and
log(observed). Momentum flux spans about 4.5 decades, so all fitting and
scoring is done on the logarithm. This is the same statistic as `r_full` in
Julio's `source_fit_refvals.score`.

**dex.** A factor of ten in log10. A bias of +0.21 dex means predictions are
high by a factor of 10^0.21 = 1.6.

**Ceiling.** An upper bound on achievable r, obtained by variance
decomposition rather than by fitting. Binning the data by the predictors and
comparing the spread of log(tau) between bins to the spread within them gives
r_max = sqrt(var_between / (var_between + var_within)). Scatter within a bin
cannot be predicted by any function of those predictors, so no model of them
can exceed this value. It is a property of the data and the predictor set, not
of any model.

**Transfer test.** Fitting in the Southern Hemisphere and scoring in the
Northern. A form that fits the SH well and transfers poorly has absorbed
SH-specific structure rather than physics. Models fitted on both hemispheres
are not transfer-tested; for those the NH column is a held-out-time score only.

**Time-blocked split.** Train and test sets separated by contiguous blocks of
time rather than at random. Adjacent three-hourly steps are strongly
correlated, so a random split leaks the test set into training and inflates
scores.

**Teacher, student, distillation.** Distillation fits a simple model (the
student, here a PySR equation) to the predictions of a flexible model (the
teacher), instead of to the observed data. The rationale is that the teacher's
output is a smooth, noise-free function, whereas the real target at sigma = 2
is roughly half irreducible scatter, which weakens the signal distinguishing
one candidate expression from another.

**Gridded teachers.** The two flexible models trained here for that purpose, on
the same 8 predictors, scaling, split and target given to PySR: a gradient
boosted regression tree (`HistGradientBoostingRegressor`, 500 iterations, 127
leaves) and a multilayer perceptron (256-256-128, GELU). Both reach r = 0.73.
They are distinct from Julio's event MLP, which is trained in the event
framework on different data. "Distilling the gridded teachers does not work"
means that PySR equations fitted to these two models' predictions scored no
better against real data than equations fitted to the noisy data directly.

**Control run.** The third of the three distillation runs, fitting the noisy
observed target through the identical script, budget and split. Without it, the
apparent gain of the distilled runs over an older direct fit would have been
misread as a distillation effect; it was ordinary run-to-run variance of a
stochastic search.

**Paired comparison at matched complexity.** Comparing two fronts equation by
equation at equal complexity, rather than comparing their single best scores. A
front's best score is one order statistic out of about 30 and is the noisiest
possible basis for comparison.

**Steering level, launch level.** Two heights diagnosed per column in Julio's
pipeline, at which predictors are sampled. The steering level sits lower
(median 6.5 km) than the launch level (median 10.0 km), where the target flux
is evaluated.

## Variables and units

| symbol | field | physical units | scaling used in the fit |
|---|---|---|---|
| tau | `epwp_zl` = rho * sqrt(u'w'^2 + v'w'^2) at launch level | Pa | `tau/tau_ref`, tau_ref = 4.68e-3 Pa |
| D | `tilt_zs` = \|zeta dV/dz\| at steering level | s^-2 | D_ref = 3e-7 s^-2 |
| Dl | `tilt_zl`, same at launch level | s^-2 | D_ref |
| Dg | `tilt_zg_zs`, mass-weighted mean over surface to steering level | s^-2 | D_ref |
| P | `precl`, large-scale precipitation rate | m s^-1 | P_ref = 3e-8 m/s (2.6 mm/day), floored at 1e-4 P_ref |
| Us | \|V\| at steering level | m s^-1 | U_ref = 10 m/s |
| Zs | steering-level height | m | Z_ref = 5000 m |
| Zt | \|zeta\| at steering level | s^-1 | zeta_ref = 3e-5 s^-1 |

All scaled variables are dimensionless and O(1)-O(10). Every score reported is
the log-space Pearson r, corr(log pred, log obs), the same statistic as
`source_fit_refvals.score` r_full. Linear-space r runs 0.10 to 0.15 lower
because exp() restores a heavy right tail (p99/p50 = 16.6).

## Best equations

Fitted on both hemispheres, time-blocked 70/30 split, sigma = 2 smoothing.

**Recommended, complexity 17.** In scaled variables:

    tau/tau_ref = sqrt( ( P^2 + (Us*Dg)^2 + Dl ) / 2.357 )

SH-test r = 0.678, NH r = 0.677.

The summands depend on disjoint sets of variables, so the equation separates
into three source terms in the manner of Grundner et al. (2024, their Equation
10). The separation is exact and is in tau^2:

    tau^2 = T_conv(precl) + T_dyn(spd_zs, tilt_zg_zs) + T_bg(tilt_zl)

    T_conv = c_conv * precl^2                    convective
    T_dyn  = c_dyn  * (spd_zs * tilt_zg_zs)^2    dynamical, moving-mountain
    T_bg   = c_bg   * tilt_zl                    background

    c_conv = 1.033e10 Pa^2 (m/s)^-2
    c_dyn  = 1.033e6  Pa^2 (m s^-3)^-2
    c_bg   = 30.98    Pa^2 s^2

Each coefficient is tau_ref^2/2.357 divided by the reference constants of its
own term, raised to the power at which that term enters:
c_conv = tau_ref^2/(2.357 P_ref^2), c_dyn = tau_ref^2/(2.357 (U_ref D_ref)^2),
and c_bg = tau_ref^2/(2.357 D_ref), the last linear because T_bg is linear in
tilt_zl. The three coefficients therefore absorb the reference values, and the
decomposed form can be quoted without them.

Grundner et al. separate cloud cover additively in the predicted quantity
itself. Here the separation appears in the square of the predicted quantity.
Momentum flux is quadratic in wave amplitude, so sources that are mutually
uncorrelated contribute additively to tau^2 rather than to tau. The recovered
structure is consistent with three uncorrelated sources, though the fit alone
does not establish that they are physically independent.

### Term contributions

Mean fractional contribution to tau^2, and the fraction of columns in which
each term is the largest:

| term | SH share | SH dominant | NH share | NH dominant |
|---|---|---|---|---|
| T_conv | 18.3% | 17.1% | 36.8% | 38.1% |
| T_dyn | 51.3% | 56.8% | 15.4% | 10.9% |
| T_bg | 30.4% | 26.1% | 47.8% | 51.1% |

Partitioned by precipitation, mean share of tau^2:

| regime | T_conv | T_dyn | T_bg |
|---|---|---|---|
| SH, wettest 10% | 61.2% | 34.6% | 4.2% |
| SH, driest 50% | 2.1% | 55.9% | 42.0% |
| NH, wettest 10% | 94.8% | 2.9% | 2.2% |
| NH, driest 50% | 5.3% | 21.0% | 73.7% |

The terms activate in separate regimes. T_conv dominates in precipitating
columns in both hemispheres. T_dyn carries most of the flux in the dry Southern
Ocean, where the storm track supplies strong steering-level winds acting on
tilted shear. T_bg dominates the dry Northern Hemisphere. The hemispheric
asymmetry in T_dyn, 51% against 15%, is the largest single difference between
the two basins and is the term most at risk of encoding Southern Ocean
conditions rather than general physics.

**Highest scoring, complexity 34:**

    tau/tau_ref = sqrt( ( P^2 + Dl*(Dg + (Us*Dg)^2) )
                        / ( Dg + (Zs + Zt/D)/1.484 ) ) + 0.202

SH-test r = 0.704, NH r = 0.690.

The cx=17 form is preferred. It costs 0.026 of r, transfers with no hemispheric
penalty, and four independent searches with different fit targets recovered it
with the same coefficient to four significant figures. The cx=34 denominator
has no clear physical reading.

### Units of the equation

Both sides are dimensionless. The left side is `tau/tau_ref`, so tau is
recovered in Pa by multiplying by tau_ref = 4.68e-3 Pa. On the right, each of
the three summands is separately dimensionless by construction: `P` divides
m/s by m/s, `Dl` and `Dg` divide s^-2 by s^-2, and `Us` divides m/s by m/s. The
sum is therefore legal and the constant 2.357 is a pure number.

The three-term form above is already dimensional: tau^2 in Pa^2, with c_conv,
c_dyn and c_bg carrying the units that convert each predictor group. Written as
a single expression, with precl in m/s, tilt in s^-2 and speed in m/s:

    tau [Pa] = 4.68e-3 * sqrt( (  (precl/3e-8)^2
                                + (spd_zs*tilt_zg_zs / (10 * 3e-7))^2
                                + tilt_zl/3e-7 ) / 2.357 )

The equation is not dimensionally homogeneous in the usual sense, because the
three terms enter at different powers of their scaled variables: quadratic in P
and in Us*Dg, linear in Dl. The reference constants are therefore part of the
model rather than cosmetic normalisation. Halving P_ref and rescaling P to
describe the same physical column changes the relative weight of the terms by a
factor of 1.7, and the fitted constant would no longer apply. Quoting the
equation in the c_conv, c_dyn, c_bg form avoids this trap, since those three
coefficients absorb the reference values.

For a typical Southern Ocean column (precl = 3e-8 m/s = 2.6 mm/day, tilt_zg_zs
= tilt_zl = 3e-7 s^-2, spd_zs = 15 m/s) the form gives tau = 6.3 mPa.

The dynamical term is an effective obstacle amplitude (tilt) multiplied by the
advecting wind, which is the moving-mountain source. Nothing in the operator
set or the loss imposed the separation into disjoint variable groups, and no
constraint required the sum to appear inside a square root.

## Scores

sigma = 2, log-space r, held-out times:

| model | inputs | SH | NH |
|---|---|---|---|
| Julio's form, published coefficients | D, P | 0.578 | 0.631 |
| Julio's form, refit at this scale | D, P | 0.582 | 0.699 |
| ceiling for any f(D, P) | D, P | 0.584 | 0.699 |
| PySR, same two inputs | D, P | 0.574 | 0.570 |
| PySR cx=17 | 5 vars | 0.678 | 0.677 |
| PySR cx=34 | 8 vars | 0.704 | 0.690 |
| MLP teacher | 8 vars | 0.733 | 0.734 |
| boosted-tree teacher | 8 vars | 0.730 | 0.734 |

Calibration: R^2 in log space is 0.469 at cx=34 against r^2 = 0.496, so
roughly 47 percent of log-variance is explained.

## Comment on Julio's equation

    tau = y0 + A D^a/(1 + D/D0) + B P^b D^c        r = 0.5784 (SH)

The hand-built form is not the limiting factor. Binned variance decomposition
caps any function of (D, P) at r = 0.584, and the published coefficients reach
0.578. A full-length symbolic search restricted to the same two inputs reached
0.584 in-sample and 0.574 held out. Three further checks agree: a boosted tree
on (D, P) gives 0.574, an MLP gives 0.576, and PySR stalls at 0.58. The form is
within 0.006 of a bound that holds for every possible f(D, P). The gain from
0.578 to 0.678 came from adding predictors, not from changing the functional
form.

Two properties of the incumbent are worth revisiting. It over-predicts by
+0.21 dex (a factor of 1.6) in the SH, consistent with the pred/obs median of
1.593 from Julio's own scorer. Its coefficients were fitted at sigma = 2; on
bare f09 columns they score 0.326, and refitting at that scale alone recovers
0.370 against a ceiling of 0.395.

## Was distilling the neural network necessary?

No, on two separate grounds.

Julio's event-composite MLP should not be distilled. It is defined on a
different population (19,888 SH detected events with 11x11 windows, not all
columns), its inputs are features CAM would have to compute anyway, and two of
its settings are measurably suboptimal: switching `loss_power` from 5 to 2
raises SH-test r from 0.761 to 0.795, and fitting both hemispheres raises NH r
from 0.710 to 0.785. Its 0.761 is also measured on an 11x11 footprint mean
(about 1000 km), which corresponds to r = 0.62 pointwise.

Distilling the gridded teachers does not work either. Three matched 5 h runs
fit PySR to the tree's predictions, the MLP's predictions, and the noisy data
as a control, then scored every equation against real data. Paired across 31
matched complexities:

| run | best SH | best NH | mean SH cx>=25 | mean NH cx>=25 |
|---|---|---|---|---|
| data (control) | 0.7040 | 0.7094 | 0.6976 | 0.6898 |
| tree-distilled | 0.7047 | 0.7083 | 0.6984 | 0.6886 |
| mlp-distilled | 0.7036 | 0.7076 | 0.6945 | 0.6916 |

Paired mean differences against the control: tree dSH = +0.0013 +/- 0.0011,
dNH = -0.0023 +/- 0.0025; mlp dSH = -0.0004 +/- 0.0011, dNH = +0.0023 +/-
0.0013. These standard errors are optimistic because equations at neighbouring
complexities are not independent. Through cx = 17 the three fronts score
identically to four decimals while only 1 to 2 of 31 equations are the same
string, so the searches are recovering algebraically equivalent expressions
from a noisy target, a step function, and a smooth network alike.

Removing target noise therefore changed nothing. The 0.03 gap between the
symbolic front and the teachers is a limit of this operator set at maxsize 40,
not a limit of the fitness signal.

## Calibration to carry forward

Run-to-run variance of the PySR search is about +/- 0.01 in r. The control run
scored 0.7040 where an earlier direct fit with identical settings scored
0.6949. That spread is five times any distillation effect measured here. Single
runs differing by 0.01 cannot be distinguished; the paired comparison above
resolved the question only because it was matched across 31 complexities.

## Limitations

All results rest on one month of one simulation. The test block is the final
9.4 days, and the SH and NH test blocks are simultaneous. With sigma = 2
smoothing and synoptic autocorrelation the effective sample size is far below
the nominal 4.26 million, so differences of about 0.01 in r are not meaningful.
Models fitted on both hemispheres are validated on unseen times, not unseen
space, so they do not establish that the wind terms carry physics rather than
geography. Holding out a latitude band would test this directly.

Every score is tied to a coarse-graining scale. Skill rises monotonically with
averaging width, from r = 0.514 on bare f09 columns to 0.652 at sigma = 2 and
0.666 for an 11x11 box mean, using the same model and predictors. The sigma = 2
choice is inherited from `BasicAna4GWP-fitting.ipynb`. It is retained because
the curve flattens there: an 11x11 mean adds only 0.013 in the SH, and NH skill
peaks at sigma = 2 and declines beyond it.
