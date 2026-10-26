# Code and data for an IEEE Access paper (2026)

## A Data-Driven Bounded-Disturbance Bounded-Error (BDBE) Lyapunov Certificate for Online EKF-Based Neural Identification of a Batch Distillation Column

This repository contains the code and data behind the paper. Every number
and figure in it is reproducible with four Python scripts: three on
the public ITB distillation dataset (one of them also runs the synthetic
benchmarks) and one on the public Cascaded Tanks benchmark. No MATLAB is used.

Archived version cited in the paper (v2.0.0):
<https://doi.org/10.5281/zenodo.23116303>

---

## Reproduce every number and figure

```bash
python -m venv venv_audit
# Linux/macOS:  source venv_audit/bin/activate
# Windows:      venv_audit\Scripts\Activate.ps1
pip install -r requirements.txt
python audit_final.py              # nominal audit -> results.json
python mc_ordering_extension.py    # revision 1 experiments -> mc_results.json
python revision2_experiments.py    # revision 2 experiments -> revision2_results.json
python benchmark_cts.py            # Cascaded Tanks benchmark -> cts_results.json (needs cts.csv)
```

`audit_final.py` is the reference implementation of Algorithm 1. It runs the
twelve weight-space identifier runs and the SGD step-size sweep, prints the
values reported in Tables 3, 4 and 9, and regenerates `fig1_data.png` ..
`fig8_zoom.png`. Runtime is about one to two minutes, dominated by the
identifier runs and the sweep.

`mc_ordering_extension.py` (added in revision 1) imports `audit_final` and
writes `mc_results.json`:

- the 30-seed Monte Carlo over randomized initial weights
  (Table 7, `w_i(0) ~ U[0, 0.10]`, fixed seed 12345);
- the Q-ordering / q sensitivity study
  (Table 8: both orderings, `q in {1e-7, 1e-6, 1e-5}`, all six EKF cases);
- the covariance-spectrum / Tier-2 ball anatomy of Section IV-D
  (`M_bar`, `omega_bar`, `r_bar`, the ball of Eq. (32), the 31-of-60 spectrum
  count, the directional factor of Eq. (35), and the post-transient re-evaluation);
- the runtime and peak-memory figures quoted in Section III-D.

`revision2_experiments.py` (added in revision 2) also imports `audit_final`
and writes `revision2_results.json`:

- E1: theta-RLS (11-dim, `Q = 0`) at `R in {2, 1, 0.05}` (Table 6): numerical
  check of Theorem 1 and Eq. (17), the `Q = 1e-6` gap, and the theta-SGD sweep
  around `alpha*`;
- E2-E4: right-hand side of the Tier-2 innovation bound, Eq. (33), for the
  weight-space runs, certified floor vs `R`, and the decomposition of the bound
  gap (NARX, `R = 1`);
- E5: forgetting factor, covariance cap and covariance resetting (Table 10);
- E6: sensitivity of the convergence criterion to its thresholds (Table 5);
- E7: model order `m = n in {2, 3, 5, 8}`;
- E8: synthetic persistently exciting benchmarks with known true parameters
  (12 nonlinear seeds and one linear ARX plant);
- E9: candidate explanations of the earlier MATLAB/Python difference at
  `R = 0.05` (Section IV-L).

Note: the printed labels of `revision2_experiments.py` use the equation numbers
of an earlier draft; its `RHS(31)` is Eq. (33) of the paper.

`benchmark_cts.py` (added in revision 2) is standalone. It runs the same
identifiers, hyperparameters and audit on the Cascaded Tanks benchmark
(Schoukens et al., 2016; <https://www.nonlinearbenchmark.org>), normalized with
the estimation-set statistics, and writes `cts_results.json` and
`fig12_cts.png` (Table 11, Fig. 12).

The ITB scripts read the public dataset `asli1_datatraining_31.xlsx`.
`benchmark_cts.py` instead reads a local `cts.csv`: one header row, then the
columns `uEst, uVal, yEst, yVal` (1024 samples each) in that order.

> **Windows note.** The peak-memory figure uses `psutil`
> (`pip install psutil`, already in `requirements.txt`). Without it every other
> number is still produced; only the memory line reports "unavailable".

### Figures

The figures are not stored in the repository; each script writes its PNG files
when it runs. The file names keep the numbering of the first draft, so they do
not all match the figure numbers of the paper:

| paper | file | script |
|---|---|---|
| Figs. 1-3, 5, 7-9 | `fig1_data.png` .. `fig8_zoom.png` (except the Fig. 4 file) | `audit_final.py` |
| Fig. 4 | `fig4_innov.png` | `revision2_experiments.py` |
| Fig. 6 | `fig9_floor_vs_R.png` | `revision2_experiments.py` |
| Figs. 10-11 | `fig10_synth.png`, `fig11_synth_linear.png` | `revision2_experiments.py` |
| Fig. 12 | `fig12_cts.png` | `benchmark_cts.py` |

---

## Key files

| file | contents |
|---|---|
| `audit_final.py` | Algorithm 1, full Python implementation |
| `results.json` | outputs of the nominal audit |
| `mc_ordering_extension.py` | revision 1 experiments: Monte Carlo, ordering sensitivity, Sec. IV-D spectrum, runtime |
| `mc_results.json` | outputs of the revision 1 experiments |
| `revision2_experiments.py` | revision 2 experiments E1-E9 |
| `revision2_results.json` | outputs of the revision 2 experiments |
| `benchmark_cts.py` | Algorithm 1 on the Cascaded Tanks benchmark |
| `cts.csv` | Cascaded Tanks data (estimation + validation) |
| `cts_results.json` | outputs of the Cascaded Tanks benchmark |
| `asli1_datatraining_31.xlsx` | public dataset (columns `u`, `ys`), 3000 samples |
| `requirements.txt` | minimum dependency versions |
| `requirements_lock.txt` | exact versions used to produce the reported numbers |

Dataset and Arduino acquisition sketch:
<https://github.com/alifsyah222/Tesis-Distilasi-Public>

Audit scripts:
<https://github.com/alifsyah222/IEEE-Access_A-Data-Driven-BDBE-Lyapunov-Certificate>

---

## Two conventions worth knowing before reading the numbers

**EKF ordering.** The EKF is run with an explicit prediction step,
`P^-(k) = P(k-1) + Q`, followed by the measurement update without a trailing
`+Q`: the ordering of Eq. (23), which is what the proof of Theorem 3 uses. It
is also the ordering of the earlier studies (conference paper, journal paper,
thesis) and of their MATLAB code. The compact form, Eq. (8), adds `Q` after
the measurement update instead. The two agree to `O(q)` and coincide as
`q -> 0`; for the linear structure they differ by under 1.5% in MSE, but for
the nonlinear structure the difference is amplified along the trajectory
(Table 8). See Remark 4.

**Convergence.** The transient-convergence instance of Definition 1 is the
first `k0` at which `|e(k)| <= 0.01` and stays below for 8 consecutive
instances, searched within `k <= 120`. A run whose full-run MSE exceeds
`1.5e-4`, or whose weights blow up, is not counted as converged
("degraded" / "diverged"). This definition is used because the output keeps
varying to the end of the record, so the best local parameters are
effectively time-varying and the weights never settle permanently; only
transient convergence is well defined. Absolute counts therefore differ from
the earlier studies, whose counts were read visually from the output-response
plots. The sensitivity to the thresholds is reported in Table 5 (experiment E6
of `revision2_experiments.py`).
