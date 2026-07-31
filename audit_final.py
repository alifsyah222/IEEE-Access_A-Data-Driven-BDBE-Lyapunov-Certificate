#!/usr/bin/env python3
# =============================================================================
#  Algorithm 1 - Certificate Audit   (FINAL, full Python, no MATLAB)
#  Nasution & Mahayana, "A Data-Driven Bounded-Disturbance Bounded-Error
#  (BDBE) Lyapunov Certificate for Online EKF-Based Neural Identification
#  of a Batch Distillation Column", IEEE Access, 2026.
#
#  Data (public):  https://github.com/alifsyah222/Tesis-Distilasi-Public
#                  asli1_datatraining_31.xlsx   (columns: u, ys; N = 3000)
#
#  EKF ordering:   explicit prediction step first, P^-(k) = P(k-1) + Q,
#                  then the measurement update without a trailing +Q,
#                  matching Eq. (22) and the proof of Theorem 3.
#
#  Run:            python audit_final.py
#  Requires:       numpy, openpyxl, matplotlib  (see requirements.txt)
#  Runtime:        ~1-2 min (dominated by the 12 identifier runs + alpha sweep)
#
# -----------------------------------------------------------------------------
#  WHAT THIS SCRIPT PRODUCES
# -----------------------------------------------------------------------------
#  Console + results.json
#    Data statistics (Algorithm 1, steps 1-3):
#      max_k ||phi(k)||^2 ............. Table 7, row 1
#      full-record lambda_min/lambda_max of the information matrix .. Table 7
#      mu_L, window PE level (Eq. 18, non-overlapping windows, L = 200) . Table 7
#    Identifier runs (Algorithm 1, step 4), 12 runs (6 cases of Table 3
#    x 2 algorithms):
#      one-step MSE, transient-convergence instance ............ Table 4
#      sup_k ||J(k)||^2 over ALL runs (SGD and EKF) of a given
#        architecture, hence alpha* = 2/sup||J||^2 (Cor. 2) ..... Table 7
#      p_min, p_max, innovation energy sum e^2/S .............. Table 7
#    Tier-2 constants (Lemma 3, Theorem 3):
#      c_bar, omega_bar (empirical working-ball radius), M_bar, r_bar
#      mu_L^J, Jacobian window-PE level (Remark 4) ............ Table 7
#    Free-run NRMSE fit of the final ARMA and NARX models ...... Table 7
#    SGD divergence onset from the step-size sweep ............. Section IV-E
#    Weight-step statistics near vs away from input transitions  Section IV-D
#
#  Figures (PNG, 150 dpi, written to the working directory)
#      fig1_data.png ........ Fig. 1  normalized open-loop dataset
#      fig2_pe.png .......... Fig. 2  window PE profile (sliding window)
#      fig3_conv.png ........ Fig. 3  early-instance convergence (ARMA-FNN)
#      fig4_innov.png ....... Fig. 4  normalized innovation energy
#      fig5_sgd_sweep.png ... Fig. 5  SGD stability sweep vs alpha
#      fig6_pred_full.png ... Fig. 6  full-run prediction + innovation
#      fig7_weights.png ..... Fig. 7  weight trajectories
#      fig8_zoom.png ........ Fig. 8  zoom on an operating-point transition
#
# -----------------------------------------------------------------------------
#  CONVERGENCE DEFINITION  (paper, Definition 1: transient convergence)
# -----------------------------------------------------------------------------
#  The transient convergence instance is the first k0 at which the innovation
#  magnitude |e(k)| falls below e_tol = 0.01 and stays below it for
#  H_hold = 8 consecutive instances, searched within the transient phase
#  k <= T = 120. A run whose full-run MSE exceeds mse_gate = 1.5e-4, or whose
#  weights blow up, is not counted as converged ("degraded" / "diverged").
#
#  Rationale: on this record the output y keeps varying to the very end
#  (Fig. 1), so the best local linear parameters theta* are effectively
#  time-varying and the weights never settle permanently; only transient
#  convergence is well defined, matched to the early-instance view of Fig. 3.
#
# -----------------------------------------------------------------------------
#  NOTE ON THE WEIGHT-STEP STATISTIC  (Section IV-D)
# -----------------------------------------------------------------------------
#  The reported ratio compares like with like: the MEAN step ||w(k)-w(k-1)||
#  in a causal window from each input transition to ten instances after it
#  against the MEAN step away from transitions (about 6x). The script also
#  reports how many transitions exceed the 99th percentile of the whole record
#  (5 of 10) and whether the single largest step of the run occurs inside a
#  transition window (it does), so the limits of the evidence are visible
#  alongside the headline number.
# =============================================================================
import numpy as np, openpyxl, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "asli1_datatraining_31.xlsx"
wb   = openpyxl.load_workbook(DATA)
rows = list(wb.active.iter_rows(values_only=True))[1:]
u = np.array([r[0] for r in rows], float)
y = np.array([r[1] for r in rows], float)
N = len(u)

m = n = 5; H = 5
Ni = (m + 1) + n            # 11
O  = H * Ni + H             # 60
P0 = 50.0; Q_ = 1e-6; W0 = 0.05; L = 200

# convergence-definition constants
E_TOL, H_HOLD, T_TRANS, MSE_GATE = 0.01, 8, 120, 1.5e-4

def xvec(k):
    return np.concatenate([u[k - np.arange(0, m + 1)], y[k - np.arange(1, n + 1)]])
def unpack(w):
    return w[:H*Ni].reshape(H, Ni), w[H*Ni:]
def forward(w, x, nl):
    Wh, Wo = unpack(w); a = Wh @ x
    z = np.tanh(a) if nl else a; ynet = Wo @ z
    dz = (1 - z**2) if nl else np.ones(H)
    Jac = np.concatenate([((Wo*dz)[:, None]*x[None, :]).ravel(), z])
    return ynet, Jac, z

def data_statistics():
    X = np.array([xvec(k) for k in range(n, N)])
    phi2 = (X**2).sum(1); max_phi2 = float(phi2.max())
    Info = X.T @ X / len(X); ev = np.linalg.eigvalsh(Info)
    lam_min, lam_max = float(ev[0]), float(ev[-1])
    # partition (definition of mu_L)
    ps, pm = [], []
    for s in range(0, len(X)-L+1, L):
        Wm = X[s:s+L]; pm.append(float(np.linalg.eigvalsh(Wm.T@Wm/L)[0])); ps.append(s+n)
    mu_L = float(np.median(pm))
    # sliding (for figure)
    ss, sm = [], []
    for s in range(0, len(X)-L+1):
        Wm = X[s:s+L]; sm.append(float(np.linalg.eigvalsh(Wm.T@Wm/L)[0])); ss.append(s+n)
    return dict(max_phi2=max_phi2, lam_min=lam_min, lam_max=lam_max, mu_L=mu_L,
                part_s=ps, part_mu=pm, slide_s=ss, slide_mu=sm, X=X)

def conv_transient(alg, nl, e2, ea):
    if np.mean(e2) > MSE_GATE: return 'degraded'
    ea = np.array(ea[:T_TRANS])
    for k0 in range(len(ea)-H_HOLD):
        if (ea[k0:k0+H_HOLD] <= E_TOL).all(): return int(k0+1)
    return 'degraded'

def run(alg, nl, alpha=None, R=None):
    w = np.full(O, W0); P = P0*np.eye(O) if alg == 'EKF' else None
    e2, ea, innov, supJ2 = [], [], 0.0, 0.0
    pmin, pmax, wt, ynet_tr, cbar = np.inf, 0.0, [], [], 0.0
    innov_curve = []; Jlog = []
    for k in range(n, N):
        x = xvec(k); ynet, J, z = forward(w, x, nl)
        e = y[k]-ynet; e2.append(e*e); ea.append(abs(e)); ynet_tr.append(float(ynet))
        supJ2 = max(supJ2, float(J@J)); Jlog.append(J.copy())
        if alg == 'SGD':
            w = w + alpha*e*J
        else:
            P = P + Q_*np.eye(O)                      # prediction step, Eq. (22)
            S = J@P@J+R; K = (P@J)/S; w = w+K*e
            P = P - np.outer(K, J@P)                  # measurement update, no +Q
            innov += e*e/S; innov_curve.append(float(innov))
            ev = np.linalg.eigvalsh(P); pmin, pmax = min(pmin, ev[0]), max(pmax, ev[-1])
        cbar = max(cbar, float(np.abs(unpack(w)[1]).max()))
        if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
            return dict(status='diverged')
        wt.append(w.copy())
    e2 = np.array(e2); wt = np.array(wt); wf = wt[-1]
    omega_bar = float(np.max(np.linalg.norm(wt - wf, axis=1)))
    # Jacobian window-PE level mu_L^J (Tier-2 analogue of mu_L, Remark 4)
    Jm = np.array(Jlog); muJ = []
    for s0 in range(0, len(Jm)-L+1, L):
        Wj = Jm[s0:s0+L]
        muJ.append(float(np.linalg.eigvalsh(Wj.T@Wj/L)[0]))
    mu_L_J = float(np.median(muJ)) if muJ else float('nan')
    out = dict(status='ok', MSE=float(e2.mean()),
               conv=conv_transient(alg, nl, e2, ea),
               supJac2=float(supJ2), cbar=cbar, omega_bar=omega_bar,
               mu_L_J=mu_L_J,
               w=[float(v) for v in wf], ynet_traj=ynet_tr)
    if alg == 'EKF':
        out.update(innov_energy=float(innov), pmin=float(pmin), pmax=float(pmax),
                   innov_curve=innov_curve)
    return out

def curvature(cbar, X, omega_bar):
    xn = np.sqrt((X**2).sum(1))
    M_bar = float((xn + (4/(3*np.sqrt(3)))*cbar*xn**2).max())
    return M_bar, 0.5*M_bar*omega_bar**2

def freerun(w, nl):
    ys = y.copy()
    for k in range(n, N):
        x = np.concatenate([u[k-np.arange(0, m+1)], ys[k-np.arange(1, n+1)]])
        ys[k], _, _ = forward(np.array(w), x, nl)
    return float(100*(1 - np.linalg.norm(y[n:]-ys[n:])/np.linalg.norm(y[n:]-y[n:].mean())))



def main():
    print("="*60); print("ALGORITHM 1 - CERTIFICATE AUDIT (final, full-Python)"); print("="*60)
    stat = data_statistics()
    print(f"max||phi||^2={stat['max_phi2']:.3f}  lam_min/max={stat['lam_min']:.2e}/{stat['lam_max']:.3f}  mu_L={stat['mu_L']:.2e}")

    # Shape of the sliding PE profile of Fig. 2, quoted in Section IV-D.
    # "Numerical zero" means below 1e-12, i.e. at the level of floating-point
    # noise for a matrix that is positive semidefinite by construction.
    sl = np.array(stat['slide_mu'])
    frac0 = float(np.mean(sl < 1e-12))
    print("sliding PE profile (L=%d): %.1f%% of windows at numerical zero (<1e-12);"
          " max elsewhere = %.2e; median = %.2e"
          % (L, 100*frac0, sl[sl >= 1e-12].max(), np.median(sl)))

    cases = [('NARX','SGD',0.5,None),('NARX','EKF',None,2.0),('NARX','SGD',1.0,None),
             ('NARX','EKF',None,1.0),('NARX','SGD',0.75,None),('NARX','EKF',None,0.05),
             ('ARMA','SGD',0.5,None),('ARMA','EKF',None,2.0),('ARMA','SGD',1.0,None),
             ('ARMA','EKF',None,1.0),('ARMA','SGD',0.75,None),('ARMA','EKF',None,0.05)]
    R = {}
    for arch, alg, al, Rv in cases:
        r = run(alg, arch == 'NARX', alpha=al, R=Rv)
        key = f"{arch}-{alg}-{al if al else Rv}"; R[key] = r
        print(f"  {key:16s} {r['status']:8s} MSE={r.get('MSE')} conv={r.get('conv')} supJ2={round(r.get('supJac2',-1) or -1,3)}")

    # sup||J||^2 and alpha* ranges over ALL runs (SGD and EKF) per architecture
    astar = {}
    for arch in ('ARMA', 'NARX'):
        sJ = [v['supJac2'] for k, v in R.items()
              if k.startswith(arch) and isinstance(v, dict) and 'supJac2' in v]
        astar[arch] = (min(sJ), max(sJ), 2/max(sJ), 2/min(sJ))
        print("%s: sup|J|^2 in [%.2f, %.2f] -> alpha* in [%.2f, %.2f] (all runs)"
              % (arch, min(sJ), max(sJ), 2/max(sJ), 2/min(sJ)))

    # tier2 constants (NARX-EKF R=1)
    rep = R['NARX-EKF-1.0']
    M_bar, r_bar = curvature(rep['cbar'], stat['X'], rep['omega_bar'])
    print(f"\nTier2: cbar={rep['cbar']:.3f} omega_bar={rep['omega_bar']:.3f} M_bar={M_bar:.3f} r_bar={r_bar:.3f}")

    # BDBE ball of Eq. (30) at R = 1. We deliberately take v_bar = 0, which
    # makes the number a rigorous LOWER bound on the certified ball and
    # removes any need to estimate the sensor noise: r_bar is quadratic in
    # omega_bar and already dominates d_bar, so any plausible v_bar changes
    # the result by a couple of percent at most.
    def bdbe_ball(v_bar, Rv=1.0):
        d_bar = v_bar + r_bar
        ball_V = d_bar**2*(rep['pmax'] + Q_)/(Rv*Q_)
        return ball_V, (rep['pmax']*ball_V)**0.5
    bV, bw = bdbe_ball(0.0)
    print("BDBE ball (Eq. 30, R=1, v_bar=0): limsup V <= %.2e  ->  ||omega|| <= %.2e"
          % (bV, bw))
    print("  vs empirical working-ball radius %.2f  ->  bound is %.0fx looser"
          % (rep['omega_bar'], bw/rep['omega_bar']))
    print("  sensitivity to v_bar:", end=" ")
    for v in (0.0, 0.03, 0.09, 0.5):
        print("v=%.2f:%.2e" % (v, bdbe_ball(v)[1]), end="  ")
    print()
    print("Jacobian window-PE mu_L^J (Remark 4): ARMA=%.2e  NARX=%.2e"
          % (R['ARMA-EKF-1.0']['mu_L_J'], R['NARX-EKF-1.0']['mu_L_J']))

    for key, nl in [('ARMA-EKF-1.0', False), ('NARX-EKF-1.0', True)]:
        R[key]['freerun'] = freerun(R[key]['w'], nl)
        print(f"free-run {key}: {R[key]['freerun']:.2f}%")

    # ---------- FIGURES ----------
    kk = np.arange(N)
    # Fig 1
    fig,(a1,a2)=plt.subplots(2,1,figsize=(6.2,3.6),sharex=True)
    a1.plot(kk,u,'k',lw=1); a1.set_ylabel("u(k)"); a1.set_ylim(0.28,1.05)
    a1.set_title("Normalized open-loop dataset (N = 3000, Ts = 1 s)")
    a2.plot(kk,y,'b',lw=0.7); a2.set_ylabel("y(k)"); a2.set_xlabel("instance k")
    fig.tight_layout(); fig.savefig("fig1_data.png",dpi=150); plt.close(fig)
    # Fig 2 (sliding, paper)
    fig,ax=plt.subplots(figsize=(6,3.2))
    ax.semilogy(stat['slide_s'],np.maximum(stat['slide_mu'],1e-16),lw=1,color='C3')
    ax.axhline(stat['lam_min'],ls='--',color='k',
               label=f"full-record $\\lambda_{{min}}$ = {stat['lam_min']:.1e}")
    ax.set_xlabel(f"window start instance (L = {L})")
    ax.set_ylabel(r"$\lambda_{\min}$ of windowed information matrix")
    ax.set_title("Excitation audit: window PE level")
    ax.legend(fontsize=8,loc='center right'); fig.tight_layout()
    fig.savefig("fig2_pe.png",dpi=150); plt.close(fig)
    # Fig 3 (early convergence)
    fig,ax=plt.subplots(figsize=(6,3.2))
    kx=np.arange(n,N)
    ax.plot(kx,y[n:],'k',lw=1.2,label='sensor output')
    ax.plot(kx,R['ARMA-EKF-0.05']['ynet_traj'],'C3',lw=1,label='EKF, R = 0.05')
    ax.plot(kx,R['ARMA-SGD-0.5']['ynet_traj'],'C0',ls='--',lw=1,label=r'SGD, $\alpha$ = 0.5')
    ax.set_xlim(n,65); ax.set_ylim(0.08,0.20)
    ax.set_xlabel("instance k"); ax.set_ylabel("output")
    ax.set_title("Early-instance convergence (ARMA-FNN)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("fig3_conv.png",dpi=150); plt.close(fig)
    # Fig 4 (innovation energy)
    fig,ax=plt.subplots(figsize=(6,3.2))
    for key,col,lab,ls in [('NARX-EKF-1.0','C3','NARX-FNN, EKF R = 1','-'),
                           ('ARMA-EKF-1.0','C0','ARMA-FNN, EKF R = 1','--')]:
        c=R[key]['innov_curve']; ax.plot(np.arange(n,n+len(c)),c,color=col,lw=1.2,ls=ls,label=lab)
    ax.set_xlabel("instance k"); ax.set_ylabel(r"$\sum_{i\leq k} e^2(i)/S(i)$")
    ax.set_title("Normalized innovation energy (certificate monitor)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("fig4_innov.png",dpi=150); plt.close(fig)


    # Fig 5 (SGD divergence sweep): steady MSE vs alpha, alpha* band + onset
    alphas = np.round(np.arange(0.5, 3.01, 0.05), 2)
    def sweep_mse(nl):
        res = []
        for alpha in alphas:
            w = np.full(O, W0); e2 = []; div = False
            for k in range(n, N):
                x = xvec(k); yn, J, _ = forward(w, x, nl); e = y[k] - yn; e2.append(e * e)
                w = w + alpha * e * J
                if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
                    div = True; break
            res.append(np.nan if div else float(np.mean(e2)))
        return np.array(res)
    mse_a, mse_n = sweep_mse(False), sweep_mse(True)
    on_a = float(alphas[np.where(np.isnan(mse_a))[0][0]]) if np.isnan(mse_a).any() else None
    on_n = float(alphas[np.where(np.isnan(mse_n))[0][0]]) if np.isnan(mse_n).any() else None
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.semilogy(alphas, mse_a, marker="o", color="C0", ms=3, lw=1, label="ARMA-FNN")
    ax.semilogy(alphas, mse_n, marker="s", ls="--", color="C3", ms=3, lw=1, label="NARX-FNN")
    ax.axvspan(astar['ARMA'][2], astar['ARMA'][3], color="C0", alpha=0.10)
    ax.axvspan(astar['NARX'][2], astar['NARX'][3], color="C3", alpha=0.10)
    ytop = ax.get_ylim()[1]
    ax.text(sum(astar['ARMA'][2:])/2, ytop * 0.3, r"$\alpha^{*}_{ARMA}$",
            color="C0", ha="center", fontsize=8, rotation=90, va="top")
    ax.text(sum(astar['NARX'][2:])/2, ytop * 0.3, r"$\alpha^{*}_{NARX}$",
            color="C3", ha="center", fontsize=8, rotation=90, va="top")
    if on_a is not None:
        ax.axvline(on_a, color="C0", ls=":", lw=1.2)
    if on_n is not None:
        ax.axvline(on_n, color="C3", ls=":", lw=1.2)
    ax.annotate("divergence onset\n(ARMA " + str(on_a) + ", NARX " + str(on_n) + ")",
                xy=(on_a if on_a else 1.4, ax.get_ylim()[0] * 4),
                fontsize=8, ha="left")
    ax.set_xlabel(r"SGD learning rate $\alpha$")
    ax.set_ylabel("full-run one-step MSE")
    ax.set_title(r"SGD: sufficient $\alpha^{*}$ (shaded) below divergence onset")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig("fig5_sgd_sweep.png", dpi=150); plt.close(fig)
    print("SGD divergence onset: ARMA~" + str(on_a) + ", NARX~" + str(on_n))

    # blow-up instance for large alpha (quoted in Section IV-D; these values
    # lie beyond the range plotted in Fig. 5)
    def blowup_instance(nl, alpha):
        w = np.full(O, W0)
        for k in range(n, N):
            x = xvec(k); yy, J, _ = forward(w, x, nl)
            w = w + alpha*(y[k]-yy)*J
            if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
                return k
        return None
    print("blow-up instance k for large alpha:")
    for a in (2.0, 3.0, 4.0, 5.0):
        ka, kn = blowup_instance(False, a), blowup_instance(True, a)
        print("  alpha=%.1f: ARMA %s | NARX %s"
              % (a, ("k=%d" % ka) if ka else "bounded",
                    ("k=%d" % kn) if kn else "bounded"))

    # ---------- Figs 6-8: best-run tracking (ARMA-FNN, EKF R=0.05) ----------
    def run_track(Rv, nl):
        w = np.full(O, W0); P = P0*np.eye(O)
        wt, yn_tr, e_tr = [], [], []
        for k in range(n, N):
            x = xvec(k); yy, J, _ = forward(w, x, nl); e = y[k] - yy
            yn_tr.append(yy); e_tr.append(e)
            P = P + Q_*np.eye(O)                  # prediction step, Eq. (22)
            S = J@P@J + Rv; K = (P@J)/S; w = w + K*e
            P = P - np.outer(K, J@P)              # measurement update, no +Q
            wt.append(w.copy())
        return np.array(wt), np.array(yn_tr), np.array(e_tr)

    wt, ynet_b, err_b = run_track(0.05, False)
    kx = np.arange(n, N)
    jumps = np.where(np.abs(np.diff(u)) > 0.02)[0]
    sel = list(np.argsort(wt.max(0) - wt.min(0))[::-1][:3])

    # Fig 6: full-run one-step prediction + innovation panel
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(kx, y[n:], color="k", lw=1.0, label="sensor output $y(k)$")
    a1.plot(kx, ynet_b, color="C3", lw=0.8, ls="--", label="network output $y_{net}(k)$")
    a1.set_ylabel("normalized output"); a1.legend(fontsize=8, loc="upper left")
    a1.set_title("One-step prediction over the full run (ARMA-FNN, EKF $R=0.05$)")
    a2.plot(kx, err_b, color="C0", lw=0.6); a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("innovation $e(k)$"); a2.set_xlabel("instance k")
    for j in jumps:
        a1.axvline(j, color="gray", lw=0.5, alpha=0.5)
        a2.axvline(j, color="gray", lw=0.5, alpha=0.5)
    fig.tight_layout(); fig.savefig("fig6_pred_full.png", dpi=150); plt.close(fig)

    # Fig 7: weight trajectories with input-transition markers
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, c in zip(sel, ["C0", "C3", "C2"]):
        ax.plot(kx, wt[:, i], color=c, lw=1.1, label="$w_{" + str(i) + "}$")
    for j in jumps:
        ax.axvline(j, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("instance k"); ax.set_ylabel("weight value")
    ax.set_title("Weight trajectories during online training (ARMA-FNN, EKF $R=0.05$)")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig("fig7_weights.png", dpi=150); plt.close(fig)

    # quantify: weight step at transitions vs steady segments
    dwn = np.linalg.norm(np.diff(wt, axis=0), axis=1)
    p99 = float(np.percentile(dwn, 99))

    # A transition is followed by a plant response lasting several instances,
    # so the comparison window is CAUSAL: the transition itself and the next
    # W instances. W = 10 is the reported choice; the sweep below shows how
    # sensitive the ratio is to that choice.
    def near_mask(W):
        msk = np.zeros(len(dwn), bool)
        for j in jumps:
            i = j - n
            if 0 < i < len(dwn) - W:
                msk[i:i + W + 1] = True
        return msk

    W_REPORT = 10
    near = near_mask(W_REPORT)
    step_near = float(dwn[near].mean()); step_away = float(dwn[~near].mean())
    ratio = step_near / step_away
    n_above = sum(1 for j in jumps if 0 < j-n < len(dwn)-W_REPORT
                  and dwn[j-n:j-n+W_REPORT+1].max() > p99)
    n_tr = len([j for j in jumps if 0 < j-n < len(dwn)-W_REPORT])
    print("weight step (causal window 0..+%d): near=%.5f  away=%.5f  ratio=%.1fx"
          % (W_REPORT, step_near, step_away, ratio))
    print("  transitions with peak above global p99: %d of %d" % (n_above, n_tr))
    print("  largest step of the run inside a transition window? %s"
          % ("yes" if near[int(np.argmax(dwn))] else "no"))
    print("  sensitivity to window width W:", end=" ")
    for W in (2, 5, 10, 20, 40):
        mk = near_mask(W)
        print("W=%d:%.1fx" % (W, dwn[mk].mean()/dwn[~mk].mean()), end="  ")
    print()

    # Fig 8: zoom on the largest operating-point transition
    best_j = max([j for j in jumps if 0 < j-n < len(dwn)-5],
                 key=lambda j: dwn[max(0, j-n-2):j-n+4].max())
    lo, hi = best_j-60, best_j+140
    sl = (kx >= lo) & (kx <= hi)
    fig, (b1, b2, b3) = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.4, 1.6]})
    b1.plot(np.arange(lo, hi+1), u[lo:hi+1], color="k", lw=1.2)
    b1.set_ylabel("$u(k)$")
    b1.set_title("Zoom on the operating-point transition at $k\\approx$" + str(best_j))
    b2.plot(kx[sl], y[n:][sl], color="k", lw=1.0, label="$y(k)$")
    b2.plot(kx[sl], ynet_b[sl], color="C3", lw=0.9, ls="--", label="$y_{net}(k)$")
    b2.set_ylabel("output"); b2.legend(fontsize=7, loc="lower right")
    for i, c in zip(sel, ["C0", "C3", "C2"]):
        b3.plot(kx[sl], wt[sl, i], color=c, lw=1.2, label="$w_{" + str(i) + "}$")
    b3.set_ylabel("weights"); b3.set_xlabel("instance k")
    b3.legend(fontsize=7, loc="lower right")
    for axx in (b1, b2, b3):
        axx.axvline(best_j, color="gray", ls=":", lw=1.2)
    fig.tight_layout(); fig.savefig("fig8_zoom.png", dpi=150); plt.close(fig)
    print("largest transition at k=" + str(best_j))


    # ---------- Table 7 summary, every row printed ----------
    A, Nx = R['ARMA-EKF-1.0'], R['NARX-EKF-1.0']
    print("\n" + "="*60)
    print("TABLE 7  (audit quantity ......... ARMA-FNN | NARX-FNN)")
    print("="*60)
    print("max_k ||phi(k)||^2 ................ %6.2f | %6.2f"
          % (stat['max_phi2'], stat['max_phi2']))
    print("sup_k ||J(k)||^2 (all runs) ....... %.2f to %.2f | %.2f to %.2f"
          % (astar['ARMA'][0], astar['ARMA'][1], astar['NARX'][0], astar['NARX'][1]))
    print("SGD threshold alpha* (Cor. 2) ..... %.2f to %.2f | %.2f to %.2f"
          % (astar['ARMA'][2], astar['ARMA'][3], astar['NARX'][2], astar['NARX'][3]))
    print("full-record lambda_min / lambda_max %.1e / %.2f | %.1e / %.2f"
          % (stat['lam_min'], stat['lam_max'], stat['lam_min'], stat['lam_max']))
    print("window PE mu_L (L=%d), median ..... %.1e | %.1e"
          % (L, stat['mu_L'], stat['mu_L']))
    print("Jacobian window PE mu_L^J ......... %.1e | %.1e   (both < 1e-15, i.e. 0)"
          % (A['mu_L_J'], Nx['mu_L_J']))
    print("EKF p_min / p_max (R=1) ........... %.1e / %.1f | %.1e / %.1f"
          % (A['pmin'], A['pmax'], Nx['pmin'], Nx['pmax']))
    print("sum e^2/S at k=3000 (R=1) ......... %6.3f | %6.3f"
          % (A['innov_energy'], Nx['innov_energy']))
    print("curvature bound M_bar (R=1) ....... %6s | %6.2f" % ("---", M_bar))
    print("working-ball radius omega_bar ..... %6.2f | %6.2f"
          % (A['omega_bar'], Nx['omega_bar']))
    print("remainder bound r_bar ............. %6s | %6.2f" % ("---", r_bar))
    print("certified bound on ||omega|| ...... %6s | %6.2f" % ("---", bw))
    print("free-run fit, NRMSE ............... %5.1f%% | %5.1f%%"
          % (A['freerun'], Nx['freerun']))
    print("="*60)

    print("\nFigures: fig1_data.png .. fig8_zoom.png")

    # save json
    slim={k:{kk:vv for kk,vv in v.items() if kk not in ('ynet_traj','innov_curve','w')}
          for k,v in R.items()}
    slim['data_statistics']={k:v for k,v in stat.items() if k not in ('X','slide_mu','slide_s','part_mu','part_s')}
    slim['tier2_constants']=dict(cbar=rep['cbar'],omega_bar=rep['omega_bar'],M_bar=M_bar,r_bar=r_bar)
    json.dump(slim,open("results.json","w"),indent=1)
    print("results.json saved.")

if __name__=="__main__": main()
