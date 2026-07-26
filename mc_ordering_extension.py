#!/usr/bin/env python3
# =============================================================================
#  Monte Carlo + Q-ordering sensitivity extension of Algorithm 1
#  Companion to audit_final.py  (Nasution & Mahayana, IEEE Access, 2026)
#
#  Produces the numbers of the revised paper (main.tex):
#    - Table V  (tab:mc)       : Monte Carlo over 30 randomized initial weights
#    - Table VI (tab:ordering) : sensitivity to the Q-ordering and to q, reported
#                                as per-architecture RANGES of the paired relative
#                                differences (prediction-first vs trailing-+Q),
#                                over 3 R values x 3 q values (36 EKF runs)
#    - Runtime / memory report (Section III-E): EKF cost without eigenvalue
#      monitoring (ms/instance), fully-monitored 3000-instance run (s), and
#      peak resident memory (MB).
#
#  Cross-platform: runs on Windows, Linux and macOS.
#    On Windows the peak-memory figure requires psutil:  pip install psutil
#
#  Run:  python mc_ordering_extension.py        (~3 min)
# =============================================================================
import sys, time, json
import numpy as np
import audit_final as af

O, N, n = af.O, af.N, af.n
Q_NOM, P0, W0 = af.Q_, af.P0, af.W0
E_TOL = af.E_TOL


# --------------------------------------------------- cross-platform peak memory
def peak_rss_mb():
    """Peak resident set size in MB (true peak on Windows/Linux/macOS)."""
    # Unix: resource.ru_maxrss is the true lifetime peak (KB on Linux, bytes on macOS)
    if sys.platform != 'win32':
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return ru / (1024 * 1024) if sys.platform == 'darwin' else ru / 1024
        except Exception:
            pass
    # Windows (or fallback): psutil peak working set
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, 'peak_wset'):          # Windows: true peak working set
            return mi.peak_wset / (1024 * 1024)
        return mi.rss / (1024 * 1024)         # last-resort fallback: current RSS
    except Exception:
        return float('nan')                   # psutil not installed on Windows


CASES = [('NARX', 'SGD', 0.5, None), ('NARX', 'EKF', None, 2.0),
         ('NARX', 'SGD', 1.0, None), ('NARX', 'EKF', None, 1.0),
         ('NARX', 'SGD', 0.75, None), ('NARX', 'EKF', None, 0.05),
         ('ARMA', 'SGD', 0.5, None), ('ARMA', 'EKF', None, 2.0),
         ('ARMA', 'SGD', 1.0, None), ('ARMA', 'EKF', None, 1.0),
         ('ARMA', 'SGD', 0.75, None), ('ARMA', 'EKF', None, 0.05)]


def run_ext(alg, nl, alpha=None, R=None, w0=None, q=Q_NOM,
            ordering='pred_first', eig_every=25):
    """run() of audit_final with configurable init, q, and Q-ordering."""
    w = np.full(O, W0) if w0 is None else w0.copy()
    P = P0 * np.eye(O) if alg == 'EKF' else None
    e2, ea, supJ2 = [], [], 0.0
    pmin, pmax = np.inf, 0.0
    for k in range(n, N):
        x = af.xvec(k)
        ynet, J, z = af.forward(w, x, nl)
        e = af.y[k] - ynet
        e2.append(e * e); ea.append(abs(e))
        supJ2 = max(supJ2, float(J @ J))
        if alg == 'SGD':
            w = w + alpha * e * J
        else:
            if ordering == 'pred_first':          # Eq. (22): P^- = P + Q, no trailing +Q
                P = P + q * np.eye(O)
                S = J @ P @ J + R; K = (P @ J) / S
                w = w + K * e
                P = P - np.outer(K, J @ P)
            else:                                  # thesis form: trailing +Q
                S = J @ P @ J + R; K = (P @ J) / S
                w = w + K * e
                P = P - np.outer(K, J @ P) + q * np.eye(O)
            if (k - n) % eig_every == 0:
                ev = np.linalg.eigvalsh(P)
                pmin, pmax = min(pmin, ev[0]), max(pmax, ev[-1])
        if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
            return dict(status='diverged')
    e2 = np.array(e2)
    return dict(status='ok', MSE=float(e2.mean()),
                conv=af.conv_transient(alg, nl, e2, ea),
                supJac2=float(supJ2), pmin=float(pmin), pmax=float(pmax))


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(n_seeds=30):
    """Table V: 30 randomized inits w_i(0) ~ U[0, 0.10] per case (fixed seed)."""
    rng = np.random.default_rng(12345)
    out = {}
    for arch, alg, al, Rv in CASES:
        key = f"{arch}-{alg}-{al if al is not None else Rv}"
        mses, convs, sj2, ndiv = [], [], [], 0
        for s in range(n_seeds):
            w0 = rng.uniform(0.0, 0.10, size=O)   # randomized init, mean 0.05
            r = run_ext(alg, arch == 'NARX', alpha=al, R=Rv, w0=w0)
            if r['status'] != 'ok':
                ndiv += 1; continue
            mses.append(r['MSE']); sj2.append(r['supJac2'])
            convs.append(r['conv'] if isinstance(r['conv'], int) else np.nan)
        mses = np.array(mses); convs = np.array(convs, float)
        ok = np.isfinite(convs)
        out[key] = dict(
            n=n_seeds, diverged=ndiv,
            mse_mean=float(mses.mean()), mse_std=float(mses.std(ddof=1)),
            conv_median=float(np.nanmedian(convs)) if ok.any() else None,
            conv_min=float(np.nanmin(convs)) if ok.any() else None,
            conv_max=float(np.nanmax(convs)) if ok.any() else None,
            frac_transient_conv=float(ok.mean()),
            supJ2_min=float(np.min(sj2)), supJ2_max=float(np.max(sj2)))
        c = out[key]
        print(f"{key:18s} MSE {c['mse_mean']:.3e} ± {c['mse_std']:.1e}"
              f"  conv med {c['conv_median']} [{c['conv_min']},{c['conv_max']}]"
              f"  frac {c['frac_transient_conv']:.2f}  div {ndiv}"
              f"  supJ2 [{c['supJ2_min']:.2f},{c['supJ2_max']:.2f}]")
    total_div = sum(v['diverged'] for v in out.values())
    print(f"-> total trials {12 * n_seeds}, diverged {total_div} "
          f"(paper claim: no run diverged)")
    return out


# ------------------------------------------------- Q-ordering / q sensitivity
def ordering_sensitivity():
    """Table VI: both Q-orderings for every EKF case at q in {1e-7,1e-6,1e-5}."""
    out = {}
    for arch, alg, al, Rv in CASES:
        if alg != 'EKF':
            continue
        for q in (1e-7, 1e-6, 1e-5):
            for ordn in ('pred_first', 'trailing'):
                key = f"{arch}-R{Rv}-q{q:.0e}-{ordn}"
                r = run_ext('EKF', arch == 'NARX', R=Rv, q=q, ordering=ordn)
                out[key] = dict(MSE=r['MSE'], supJac2=r['supJac2'], conv=r['conv'])
                print(f"{key:34s} MSE {r['MSE']:.4e}  supJ2 {r['supJac2']:.3f}"
                      f"  conv {r['conv']}")

    # paired relative differences per (arch, R, q): prediction-first vs trailing
    rel = {}
    conv_all_unchanged = True
    for arch in ('ARMA', 'NARX'):
        for Rv in (2.0, 1.0, 0.05):
            for q in (1e-7, 1e-6, 1e-5):
                a = out[f"{arch}-R{Rv}-q{q:.0e}-pred_first"]
                b = out[f"{arch}-R{Rv}-q{q:.0e}-trailing"]
                rel[f"{arch}-R{Rv}-q{q:.0e}"] = dict(
                    dMSE_pct=100 * abs(a['MSE'] - b['MSE']) / a['MSE'],
                    dJ2_pct=100 * abs(a['supJac2'] - b['supJac2']) / a['supJac2'])
                if a['conv'] != b['conv']:
                    conv_all_unchanged = False
    for k, v in rel.items():
        print(f"rel-diff {k:22s} dMSE {v['dMSE_pct']:6.2f}%  dsupJ2 {v['dJ2_pct']:6.2f}%")

    # per-architecture RANGES -> exactly the entries of Table VI (tab:ordering)
    summary = {}
    for arch in ('ARMA', 'NARX'):
        dmse = [rel[k]['dMSE_pct'] for k in rel if k.startswith(arch + '-')]
        dj2 = [rel[k]['dJ2_pct'] for k in rel if k.startswith(arch + '-')]
        summary[arch] = dict(dMSE_min=min(dmse), dMSE_max=max(dmse),
                             dJ2_min=min(dj2), dJ2_max=max(dj2))
    print("-" * 70)
    print("TABLE VI (paired relative differences, range over 3 R x 3 q = 9 pairs):")
    for arch in ('ARMA', 'NARX'):
        s = summary[arch]
        print(f"  {arch}-FNN: |dMSE|/MSE {s['dMSE_min']:.1f} to {s['dMSE_max']:.1f}%"
              f"   |dsupJ2|/supJ2 {s['dJ2_min']:.1f} to {s['dJ2_max']:.1f}%")
    print(f"  Transient-convergence instance identical in all 36 runs: "
          f"{conv_all_unchanged}")
    summary['conv_all_unchanged'] = conv_all_unchanged
    return out, rel, summary


# ----------------------------------------------------------- runtime / memory
def runtime_report():
    """Section III-E: per-instance EKF cost, fully-monitored run, peak memory."""
    inst = N - n

    # (a) EKF WITHOUT eigenvalue monitoring -> paper's "~0.04 ms per instance"
    t0 = time.perf_counter()
    run_ext('EKF', True, R=1.0, eig_every=10**9)
    t_unmon = time.perf_counter() - t0
    ms_unmon = 1000 * t_unmon / inst

    # (b) EKF FULLY MONITORED (eigendecomposition every instance) -> paper's "~0.6 s"
    t0 = time.perf_counter()
    run_ext('EKF', True, R=1.0, eig_every=1)
    t_mon = time.perf_counter() - t0
    ms_mon = 1000 * t_mon / inst

    # (c) one SGD full run (context; not a headline number in the paper)
    t0 = time.perf_counter()
    run_ext('SGD', True, alpha=0.5)
    t_sgd = time.perf_counter() - t0

    peak_mb = peak_rss_mb()
    rep = dict(n_w=O, N=N, instances=inst,
               t_ekf_unmonitored_s=t_unmon, ms_per_instance_unmonitored=ms_unmon,
               t_ekf_monitored_s=t_mon, ms_per_instance_monitored=ms_mon,
               t_sgd_run_s=t_sgd, peak_rss_mb=peak_mb)
    print(f"EKF without eig. monitoring: {t_unmon:.3f} s  ->  "
          f"{ms_unmon:.3f} ms/instance (n_w={O})")
    print(f"EKF fully monitored (eig. every step): {t_mon:.2f} s  "
          f"({ms_mon:.3f} ms/instance)")
    print(f"SGD full run: {t_sgd:.3f} s")
    if np.isnan(peak_mb):
        print("peak resident memory: unavailable "
              "(install psutil on Windows: pip install psutil)")
    else:
        print(f"peak resident memory: {peak_mb:.0f} MB")
    return rep


if __name__ == "__main__":
    print("=" * 70); print("MONTE CARLO (30 randomized initializations per case)"); print("=" * 70)
    t0 = time.perf_counter()
    mc = monte_carlo(30)
    print("=" * 70); print("Q-ORDERING AND q SENSITIVITY (EKF cases)"); print("=" * 70)
    osens, rel, osummary = ordering_sensitivity()
    print("=" * 70); print("RUNTIME / MEMORY"); print("=" * 70)
    rt = runtime_report()
    json.dump(dict(monte_carlo=mc, ordering=osens, ordering_rel=rel,
                   ordering_summary=osummary, runtime=rt),
              open("mc_results.json", "w"), indent=1)
    print(f"\nmc_results.json saved. Total extension time "
          f"{time.perf_counter()-t0:.0f} s.")
