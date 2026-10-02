#!/usr/bin/env python3
# =============================================================================
#  Revision-2 experiments (resubmission of Access-2026-39064)
#
#  E1  Tier-1 on the parameter it is proved for: theta-RLS (11-dim, J = phi,
#      Q = 0) at R in {2, 1, 0.05}.  Machine-precision check of Theorem 1,
#      right-hand side of the innovation certificate (17), theta-SGD sweep
#      showing that alpha* = 0.286 is sharp, and the O(q) gap at Q = 1e-6.
#  E2  Right-hand sides next to left-hand sides: (17) for theta-RLS,
#      (31) for the executed w-space runs, with a consistent v_bar.
#  E3  Certified floor  d_bar^2 (p_max+q)/(R q)  versus R, next to the
#      empirical one-step MSE (Remark 3 / R2 #3).
#  E4  Decomposition of the five orders of magnitude (R2 #6).
#  E5  Forgetting factor / covariance resetting on the ITB record (R2 #21).
#  E6  Sensitivity of the transient-convergence criterion (R1, R2 #9, #15).
#  E7  Model-order sensitivity (R3).
#  E8  Synthetic persistently-exciting benchmark with known w*: Tier-2 ball
#      evaluated segment-wise, self-consistency (Prop. 1), true ||omega(k)||.
#  E9  Candidate explanations of the MATLAB/Python R = 0.05 discrepancy.
#
#  Everything is written to revision2_results.json and printed.
# =============================================================================
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import audit_final as A          # dataset, model, nominal identifiers

u, y, N = A.u, A.y, A.N
m, n, H, Ni, O = A.m, A.n, A.H, A.Ni, A.O
P0, Q_, W0, L = A.P0, A.Q_, A.W0, A.L
xvec, forward, unpack = A.xvec, A.forward, A.unpack
OUT = {}

def kx(): return np.arange(n, N)

# ----------------------------------------------------------------------------
#  Generic identifiers
# ----------------------------------------------------------------------------
def theta_rls(R, q=0.0, p0=P0, th0=None, lam=1.0, Pcap=None, ydata=None, udata=None):
    """RLS / Kalman filter on theta in R^11 with J = phi = x(k)."""
    uu = u if udata is None else udata; yy = y if ydata is None else ydata
    NN = len(uu)
    def xv(k): return np.concatenate([uu[k-np.arange(0, m+1)], yy[k-np.arange(1, n+1)]])
    th = np.zeros(Ni) if th0 is None else th0.copy(); P = p0*np.eye(Ni)
    e_hist, S_hist, th_hist, P_hist = [], [], [], []
    pmin, pmax = np.inf, 0.0
    for k in range(n, NN):
        phi = xv(k); e = yy[k] - phi@th
        P = P + q*np.eye(Ni)
        if lam < 1.0: P = P/lam
        if Pcap is not None:            # constant-trace style cap: P <= Pcap*I
            ev, U = np.linalg.eigh(P); P = U@np.diag(np.minimum(ev, Pcap))@U.T
        S = phi@P@phi + R; K = P@phi/S; th = th + K*e
        P = P - np.outer(K, phi@P)
        ev = np.linalg.eigvalsh(P); pmin, pmax = min(pmin, ev[0]), max(pmax, ev[-1])
        e_hist.append(e); S_hist.append(S); th_hist.append(th.copy()); P_hist.append(P.copy())
    return dict(e=np.array(e_hist), S=np.array(S_hist), th=np.array(th_hist),
                P=P_hist, pmin=float(pmin), pmax=float(pmax))

def theta_sgd(alpha, th0=None, ydata=None, udata=None):
    uu = u if udata is None else udata; yy = y if ydata is None else ydata
    NN = len(uu)
    def xv(k): return np.concatenate([uu[k-np.arange(0, m+1)], yy[k-np.arange(1, n+1)]])
    th = np.zeros(Ni) if th0 is None else th0.copy(); e_hist = []; th_hist = []
    for k in range(n, NN):
        phi = xv(k); e = yy[k] - phi@th; e_hist.append(e)
        th = th + alpha*e*phi; th_hist.append(th.copy())
        if not np.isfinite(th).all() or np.abs(th).max() > 1e6:
            return dict(status='diverged', k=k, e=np.array(e_hist))
    return dict(status='ok', e=np.array(e_hist), th=np.array(th_hist))

def w_ekf(nl, R, q=Q_, w0=None, lam=1.0, Pcap=None, reset_every=None,
          ydata=None, udata=None, keep_P=False):
    """Network-weight EKF (60-dim), prediction-first ordering; optional
    forgetting factor lam < 1, covariance cap, or periodic resetting."""
    uu = u if udata is None else udata; yy = y if ydata is None else ydata
    NN = len(uu)
    def xv(k): return np.concatenate([uu[k-np.arange(0, m+1)], yy[k-np.arange(1, n+1)]])
    w = np.full(O, W0) if w0 is None else w0.copy(); P = P0*np.eye(O)
    e_h, S_h, w_h, J_h, pmin_h, pmax_h, cmax = [], [], [], [], [], [], 0.0
    supJ2 = 0.0; Pfinal = None
    for i, k in enumerate(range(n, NN)):
        x = xv(k); ynet, J, z = forward(w, x, nl); e = yy[k]-ynet
        supJ2 = max(supJ2, float(J@J))
        P = P + q*np.eye(O)
        if lam < 1.0: P = P/lam
        if Pcap is not None:
            ev, U = np.linalg.eigh(P); P = U@np.diag(np.minimum(ev, Pcap))@U.T
        if reset_every and i > 0 and i % reset_every == 0: P = P0*np.eye(O)
        S = J@P@J + R; K = P@J/S; w = w + K*e
        P = P - np.outer(K, J@P)
        ev = np.linalg.eigvalsh(P); pmin_h.append(ev[0]); pmax_h.append(ev[-1])
        cmax = max(cmax, float(np.abs(unpack(w)[1]).max()))
        e_h.append(e); S_h.append(S); w_h.append(w.copy()); J_h.append(J.copy())
        if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
            return dict(status='diverged', k=k)
    return dict(status='ok', e=np.array(e_h), S=np.array(S_h), w=np.array(w_h),
                J=np.array(J_h), pmin=np.array(pmin_h), pmax=np.array(pmax_h),
                cbar=cmax, supJ2=supJ2, Pfinal=P)

def freerun(wf, nl, uu=None, yy=None):
    uu = u if uu is None else uu; yy = y if yy is None else yy
    ys = yy.copy()
    for k in range(n, len(uu)):
        x = np.concatenate([uu[k-np.arange(0, m+1)], ys[k-np.arange(1, n+1)]])
        ys[k], _, _ = forward(wf, x, nl)
    err = yy[n:]-ys[n:]; ref = yy[n:]-yy[n:].mean()
    fit = 100*(1-np.linalg.norm(err)/np.linalg.norm(ref))
    nrmse = 100*np.linalg.norm(err)/np.linalg.norm(ref)
    return float(fit), float(nrmse)

def freerun_theta(th, uu=None, yy=None):
    uu = u if uu is None else uu; yy = y if yy is None else yy
    ys = yy.copy()
    for k in range(n, len(uu)):
        phi = np.concatenate([uu[k-np.arange(0, m+1)], ys[k-np.arange(1, n+1)]])
        ys[k] = phi@th
        if not np.isfinite(ys[k]) or abs(ys[k]) > 1e6: return float('-inf'), float('inf')
    err = yy[n:]-ys[n:]; ref = yy[n:]-yy[n:].mean()
    return float(100*(1-np.linalg.norm(err)/np.linalg.norm(ref))), float(100*np.linalg.norm(err)/np.linalg.norm(ref))

def conv_instance(e, e_tol=0.01, hold=8, T=120, gate=1.5e-4):
    if np.mean(e**2) > gate: return 'degraded'
    ea = np.abs(e[:T])
    for k0 in range(len(ea)-hold):
        if (ea[k0:k0+hold] <= e_tol).all(): return int(k0+1)
    return 'degraded'

# ----------------------------------------------------------------------------
#  Reference (declared) quantities for the ITB record
# ----------------------------------------------------------------------------
X = np.array([xvec(k) for k in range(n, N)]); Y = y[n:]
theta_ls, *_ = np.linalg.lstsq(X, Y, rcond=None)         # best fixed linear model
v_ls = Y - X@theta_ls
vbar_rms = float(np.sqrt(np.mean(v_ls**2))); vbar_max = float(np.abs(v_ls).max())
max_phi2 = float((X**2).sum(1).max()); alpha_star = 2/max_phi2
print("=== reference quantities ===")
print("best fixed linear model (LS over full record): RMS residual = %.4e, max |v| = %.4e" % (vbar_rms, vbar_max))
print("max||phi||^2 = %.3f -> alpha* = %.4f" % (max_phi2, alpha_star))
fit_ls, nrmse_ls = freerun_theta(theta_ls)
print("free-run fit of the best fixed linear model (LS): %.1f%% (NRMSE %.0f%%)" % (fit_ls, nrmse_ls))
OUT['reference'] = dict(freerun_fit_ls=fit_ls, theta_ls=theta_ls.tolist(), vbar_rms=vbar_rms, vbar_max=vbar_max,
                        max_phi2=max_phi2, alpha_star=alpha_star)

# ----------------------------------------------------------------------------
#  E1  theta-RLS, Q = 0, three R values
# ----------------------------------------------------------------------------
print("\n=== E1: theta-RLS (Tier 1 as proved: 11-dim, J = phi, Q = 0) ===")
E1 = {}
th0 = np.zeros(Ni)   # theta(0) = 0 <- w(0)=0.05*1 gives theta(0) = 5*0.05*0.05 = 0.0125 per entry
th0 = np.full(Ni, H*W0*W0)   # image of the nominal w(0) under the map theta = sum_j c_j w_in,j
for R in (2.0, 1.0, 0.05):
    r = theta_rls(R, q=0.0, th0=th0)
    e, S, th = r['e'], r['S'], r['th']
    # Theorem 1 identity with the declared reference theta_ls
    om = theta_ls[None, :] - np.vstack([th0[None, :], th[:-1]])       # omega(k-1)
    omk = theta_ls[None, :] - th                                        # omega(k)
    Pinv_prev = [np.linalg.inv(P0*np.eye(Ni))] + [np.linalg.inv(P) for P in r['P'][:-1]]
    Vprev = np.array([o@Pi@o for o, Pi in zip(om, Pinv_prev)])
    Vk = np.array([o@np.linalg.inv(P)@o for o, P in zip(omk, r['P'])])
    v = v_ls
    resid = Vk - (Vprev + v**2/R - e**2/S)
    innov = np.cumsum(e**2/S)
    V0 = float(om[0]@om[0]/P0)
    tau = len(e)
    rhs_rms = V0 + tau*vbar_rms**2/R
    rhs_exact = V0 + np.sum(v**2)/R
    rhs_max = V0 + tau*vbar_max**2/R
    rhs_curve = V0 + np.cumsum(v**2)/R
    holds = bool((innov <= rhs_curve).all())
    margin = float((rhs_curve - innov).min())
    fit, nrmse = freerun_theta(th[-1])
    E1[R] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), innov_final=float(innov[-1]),
                 V0=V0, rhs_rms=float(rhs_rms), rhs_exact=float(rhs_exact), rhs_max=float(rhs_max),
                 identity_max_abs_residual=float(np.abs(resid).max()),
                 bound_holds_every_k=holds, min_margin=margin,
                 pmin=r['pmin'], pmax=r['pmax'], freerun_fit=fit, freerun_nrmse=nrmse,
                 innov_curve=innov.tolist(), V_curve=Vk.tolist(), Vfinal=float(Vk[-1]),
                 omega_final=float(np.linalg.norm(omk[-1])), omega0=float(np.linalg.norm(om[0])))    
    print("R=%5.2f MSE=%.4e conv=%s  sum e^2/S=%.4f  RHS(17): exact=%.4f, tau*vbar_rms^2/R+V0=%.4f, max-v=%.2f  "
          "| identity residual %.1e | V(N)=%.3e ||omega(N)||=%.3f (||omega(0)||=%.3f) | pmin=%.1e "
          "| holds=%s (min margin %.1e) | free-run fit %.1f%%"
          % (R, E1[R]['MSE'], E1[R]['conv'], innov[-1], rhs_exact, rhs_rms, rhs_max,
             E1[R]['identity_max_abs_residual'], Vk[-1], E1[R]['omega_final'], E1[R]['omega0'],
             E1[R]['pmin'], holds, margin, fit))
OUT['E1_theta_rls'] = {str(k): {kk: vv for kk, vv in v.items() if kk not in ('innov_curve', 'V_curve')} for k, v in E1.items()}
np.save("theta_rls_curves.npy", {str(k): (v['innov_curve'], v['V_curve']) for k, v in E1.items()}, allow_pickle=True)

# O(q) gap at Q = 1e-6 (theta, J = phi): Theorem 3 with r = 0 applies
print("\n--- theta-RLS with Q = 1e-6 (Theorem 3 with J = phi, r = 0) ---")
E1q = {}
for R in (2.0, 1.0, 0.05):
    r = theta_rls(R, q=Q_, th0=th0); e, S = r['e'], r['S']
    innov = float(np.sum(e**2/S))
    E1q[R] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), innov_final=innov, pmin=r['pmin'], pmax=r['pmax'],
                  dMSE_rel=float((np.mean(e**2)-E1[R]['MSE'])/E1[R]['MSE']),
                  dinnov_rel=float((innov-E1[R]['innov_final'])/E1[R]['innov_final']))
    print("R=%5.2f MSE=%.4e (%.2f%% vs Q=0) sum e^2/S=%.4f (%.2f%% vs Q=0) pmin=%.2e pmax=%.1f"
          % (R, E1q[R]['MSE'], 100*E1q[R]['dMSE_rel'], innov, 100*E1q[R]['dinnov_rel'], r['pmin'], r['pmax']))
OUT['E1_theta_rls_q1e-6'] = {str(k): v for k, v in E1q.items()}

# theta-SGD sweep: sharpness of alpha* = 2/max||phi||^2
print("\n--- theta-SGD sweep (sharpness of the a-priori certificate) ---")
alphas = np.round(np.arange(0.10, 0.60, 0.01), 3)
sweep = {}
for a in alphas:
    r = theta_sgd(a, th0=th0)
    sweep[float(a)] = 'div' if r['status'] == 'diverged' else float(np.mean(r['e']**2))
onset = next((a for a in alphas if sweep[float(a)] == 'div'), None)
print("nominal init: first divergence at alpha = %s (alpha* = %.3f)" % (onset, alpha_star))
rng = np.random.default_rng(12345); seeds_onset = []
for s in range(30):
    t0 = rng.uniform(0, 0.10, O); Wh, Wo = unpack(t0); th_s = Wo@Wh   # theta image of random w(0)
    on = None
    for a in np.round(np.arange(0.20, 0.60, 0.01), 3):
        if theta_sgd(a, th0=th_s)['status'] == 'diverged': on = float(a); break
    seeds_onset.append(on)
print("30 random initializations: divergence onset in [%.2f, %.2f]; runs at alpha=0.5 diverge in %d/30"
      % (min(seeds_onset), max(seeds_onset), sum(1 for o in seeds_onset if o is not None and o <= 0.5)))
OUT['E1_theta_sgd_sweep'] = dict(sweep={str(k): v for k, v in sweep.items()}, onset_nominal=onset,
                                 onset_seeds_min=min(seeds_onset), onset_seeds_max=max(seeds_onset))

# ----------------------------------------------------------------------------
#  E2/E3/E4  executed w-space runs: RHS of (31), floor vs R, decomposition
# ----------------------------------------------------------------------------
print("\n=== E2/E3: executed w-space EKF runs, RHS of (31), certified floor vs R ===")
E23 = {}
for arch, nl in (('ARMA', False), ('NARX', True)):
    for R in (2.0, 1.0, 0.05):
        r = w_ekf(nl, R); e, S = r['e'], r['S']
        wf = r['w'][-1]; om_bar = float(np.max(np.linalg.norm(r['w']-wf, axis=1)))
        pmax = float(r['pmax'].max()); pmin = float(r['pmin'].min())
        M_bar, r_bar = A.curvature(r['cbar'], X, om_bar)
        # segment bound on c: c_bar_seg = c_bar_run + omega_bar (Lemma 3, segment version)
        M_seg, r_seg = A.curvature(r['cbar']+om_bar, X, om_bar)
        vb = vbar_rms                     # consistent declared residual level
        d_bar = vb + r_seg
        rho = pmax/(pmax+Q_)
        floor_V = d_bar**2/(R*(1-rho)); floor_w = float(np.sqrt(pmax*floor_V))
        innov = float(np.sum(e**2/S))
        V0 = float((om_bar**2)/P0)          # ||omega(0)|| declared <= omega_bar
        rhs31 = V0 + len(e)*d_bar**2/R
        rhs31_vonly = V0 + len(e)*vb**2/R     # what remains if r were zero
        fit, nrmse = freerun(wf, nl)
        E23[f"{arch}-{R}"] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), innov=innov,
                                 rhs31=float(rhs31), rhs31_vonly=float(rhs31_vonly),
                                 omega_bar=om_bar, cbar=float(r['cbar']), M_bar=float(M_bar), r_bar=float(r_bar),
                                 M_seg=float(M_seg), r_seg=float(r_seg), d_bar=float(d_bar),
                                 pmin=pmin, pmax=pmax, rho=float(rho), floor_V=float(floor_V), floor_w=floor_w,
                                 supJ2=float(r['supJ2']), freerun_fit=fit, freerun_nrmse=nrmse)
        print("%s R=%5.2f MSE=%.4e conv=%s | sum e^2/S=%.3f vs RHS(31)=%.3e (v-only %.3f) | omega_bar=%.2f c_bar=%.2f "
              "M_seg=%.2f r_seg=%.2f d_bar=%.2f | pmax=%.1f pmin=%.1e | floor ||omega||<=%.2e | free-run fit %.1f%% (NRMSE %.0f%%)"
              % (arch, R, E23[f"{arch}-{R}"]['MSE'], E23[f"{arch}-{R}"]['conv'], innov, rhs31, rhs31_vonly, om_bar,
                 r['cbar'], M_seg, r_seg, d_bar, pmax, pmin, floor_w, fit, nrmse))
OUT['E23_wspace'] = E23

# E4 decomposition at NARX R=1
nx = E23['NARX-1.0']; d = nx['d_bar']; pmax, pmin = nx['pmax'], nx['pmin']
ball = d*np.sqrt(pmax*(pmax+Q_)/(1.0*Q_)); ball_pmin = d*np.sqrt(pmin*(pmin+Q_)/(1.0*Q_))
print("\n=== E4: decomposition (NARX, R=1) === ball=%.2e ; with p_max->p_min: %.2e (factor %.1e); d_bar/omega_bar=%.1f; "
      "total ball/omega_bar=%.1e; sqrt(pmax/pmin)=%.0f" % (ball, ball_pmin, ball/ball_pmin, d/nx['omega_bar'], ball/nx['omega_bar'], np.sqrt(pmax/pmin)))
OUT['E4'] = dict(ball=float(ball), ball_pmin=float(ball_pmin), factor_pmax=float(ball/ball_pmin),
                 factor_remainder=float(d/nx['omega_bar']), total=float(ball/nx['omega_bar']), directional=float(np.sqrt(pmax/pmin)))

# ----------------------------------------------------------------------------
#  E5  forgetting factor / resetting on the ITB record
# ----------------------------------------------------------------------------
print("\n=== E5: forgetting factor / covariance cap / resetting (NARX EKF R=1, and theta-RLS R=1) ===")
E5 = {}
def spectrum_count(P, thr=45.0): return int((np.linalg.eigvalsh(P) > thr).sum())
for label, kw in [('nominal', {}), ('lambda=0.999', dict(lam=0.999)), ('lambda=0.99', dict(lam=0.99)),
                  ('lambda=0.99 cap p0', dict(lam=0.99, Pcap=P0)), ('reset every 500', dict(reset_every=500))]:
    r = w_ekf(True, 1.0, **kw)
    if r['status'] != 'ok':
        print("%-20s diverged at k=%d" % (label, r['k'])); E5[label] = dict(status='diverged', k=r['k']); continue
    e = r['e']; wf = r['w'][-1]; om_bar = float(np.max(np.linalg.norm(r['w']-wf, axis=1)))
    pmax = float(r['pmax'].max()); pmax_end = float(r['pmax'][-1]); pmin = float(r['pmin'].min())
    M_seg, r_seg = A.curvature(r['cbar']+om_bar, X, om_bar); d_bar = vbar_rms + r_seg
    ball = d_bar*np.sqrt(pmax*(pmax+Q_)/Q_)
    fit, nrmse = freerun(wf, True)
    E5[label] = dict(status='ok', MSE=float(np.mean(e**2)), conv=conv_instance(e), pmax_run=pmax, pmax_end=pmax_end, pmin=pmin,
                     n_dirs_above_45=spectrum_count(r['Pfinal']), omega_bar=om_bar, ball=float(ball), freerun_fit=fit)
    print("%-20s MSE=%.3e conv=%s p_max(run)=%.3g p_max(N)=%.3g p_min=%.1e dirs>45: %2d omega_bar=%.2f ball=%.2e free-run fit %.1f%%"
          % (label, E5[label]['MSE'], E5[label]['conv'], pmax, pmax_end, pmin, E5[label]['n_dirs_above_45'], om_bar, ball, fit))
for label, kw in [('theta nominal', {}), ('theta lambda=0.99', dict(lam=0.99)), ('theta lambda=0.99 cap', dict(lam=0.99, Pcap=P0))]:
    r = theta_rls(1.0, q=0.0, th0=th0, **kw); e = r['e']
    fit, nrmse = freerun_theta(r['th'][-1])
    E5[label] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), pmax=r['pmax'], pmin=r['pmin'], freerun_fit=fit,
                     pmax_end=float(np.linalg.eigvalsh(r['P'][-1])[-1]))
    print("%-20s MSE=%.3e conv=%s p_max(run)=%.3g p_max(N)=%.3g p_min=%.1e free-run fit %.1f%%"
          % (label, E5[label]['MSE'], E5[label]['conv'], r['pmax'], E5[label]['pmax_end'], r['pmin'], fit))
OUT['E5_forgetting'] = E5

# ----------------------------------------------------------------------------
#  E6  criterion sensitivity
# ----------------------------------------------------------------------------
print("\n=== E6: sensitivity of the transient-convergence criterion ===")
runs = {}
for arch, nl in (('NARX', True), ('ARMA', False)):
    for R in (2.0, 1.0, 0.05): runs[f"{arch}-EKF-{R}"] = w_ekf(nl, R)['e']
    for al in (0.5, 1.0, 0.75):
        rr = A.run('SGD', nl, alpha=al); runs[f"{arch}-SGD-{al}"] = None
        # recompute SGD innovation sequence
        w = np.full(O, W0); eh = []
        for k in range(n, N):
            x = xvec(k); yn, J, _ = forward(w, x, nl); e = y[k]-yn; eh.append(e); w = w + al*e*J
        runs[f"{arch}-SGD-{al}"] = np.array(eh)
for k_ in ('theta-RLS-2.0', 'theta-RLS-1.0', 'theta-RLS-0.05'):
    runs[k_] = theta_rls(float(k_.split('-')[-1]), q=0.0, th0=th0)['e']
E6 = {}
grid = [(0.005, 8, 120, 1.5e-4), (0.01, 8, 120, 1.5e-4), (0.02, 8, 120, 1.5e-4),
        (0.01, 4, 120, 1.5e-4), (0.01, 16, 120, 1.5e-4), (0.01, 8, 60, 1.5e-4), (0.01, 8, 240, 1.5e-4),
        (0.01, 8, 120, 1.0e-4), (0.01, 8, 120, 3.0e-4), (0.01, 8, 120, 5.0e-4)]
hdr = "%-18s" % "run" + "".join("%-25s" % ("t%.3g/h%d/T%d/g%.1e" % g) for g in grid)
print(hdr)
for name, e in runs.items():
    row = [str(conv_instance(e, *g)) for g in grid]; E6[name] = dict(zip([str(g) for g in grid], row))
    print("%-18s" % name + "".join("%-25s" % c for c in row))
mses = {k: float(np.mean(e**2)) for k, e in runs.items()}
print("MSE gap: max converged-class MSE = %.2e, min degraded-class MSE = %.2e"
      % (max(v for k, v in mses.items() if conv_instance(runs[k]) != 'degraded'),
         min(v for k, v in mses.items() if conv_instance(runs[k]) == 'degraded')))
OUT['E6_criterion'] = dict(table=E6, mses=mses)

# ----------------------------------------------------------------------------
#  E7  model-order sensitivity
# ----------------------------------------------------------------------------
print("\n=== E7: model-order sensitivity (theta-RLS Q=0 and NARX EKF, R=1) ===")
E7 = {}
for order in (2, 3, 5, 8):
    A.m = A.n = order; A.Ni = 2*order+1; A.O = H*A.Ni + H
    m_, n_, Ni_, O_ = A.m, A.n, A.Ni, A.O
    Xo = np.array([np.concatenate([u[k-np.arange(0, m_+1)], y[k-np.arange(1, n_+1)]]) for k in range(n_, N)]); Yo = y[n_:]
    # theta RLS
    th = np.full(Ni_, H*W0*W0); P = P0*np.eye(Ni_); eh = []
    for i, k in enumerate(range(n_, N)):
        phi = Xo[i]; e = Yo[i]-phi@th; S = phi@P@phi+1.0; K = P@phi/S; th = th+K*e; P = P-np.outer(K, phi@P); eh.append(e)
    ys = y.copy()
    for k in range(n_, N):
        phi = np.concatenate([u[k-np.arange(0, m_+1)], ys[k-np.arange(1, n_+1)]]); ys[k] = phi@th
    fit_t = 100*(1-np.linalg.norm(y[n_:]-ys[n_:])/np.linalg.norm(y[n_:]-y[n_:].mean())) if np.isfinite(ys).all() else float('-inf')
    Info = Xo.T@Xo/len(Xo); lam = np.linalg.eigvalsh(Info)
    mu = np.median([np.linalg.eigvalsh(Xo[s:s+L].T@Xo[s:s+L]/L)[0] for s in range(0, len(Xo)-L+1, L)])
    # NARX EKF in w-space of this order
    def fwd(w, x):
        Wh = w[:H*Ni_].reshape(H, Ni_); Wo = w[H*Ni_:]; a = Wh@x; z = np.tanh(a); yn = Wo@z
        Jac = np.concatenate([((Wo*(1-z**2))[:, None]*x[None, :]).ravel(), z]); return yn, Jac
    w = np.full(O_, W0); P = P0*np.eye(O_); en = []
    for i, k in enumerate(range(n_, N)):
        x = Xo[i]; yn, J = fwd(w, x); e = Yo[i]-yn; en.append(e); P = P+Q_*np.eye(O_)
        S = J@P@J+1.0; K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)
    ys = y.copy()
    for k in range(n_, N):
        x = np.concatenate([u[k-np.arange(0, m_+1)], ys[k-np.arange(1, n_+1)]]); ys[k], _ = fwd(w, x)
        if not np.isfinite(ys[k]) or abs(ys[k]) > 1e6: ys[:] = np.nan; break
    fit_n = 100*(1-np.linalg.norm(y[n_:]-ys[n_:])/np.linalg.norm(y[n_:]-y[n_:].mean())) if np.isfinite(ys).all() else float('-inf')
    E7[order] = dict(theta_MSE=float(np.mean(np.array(eh)**2)), theta_freerun_fit=float(fit_t),
                     narx_MSE=float(np.mean(np.array(en)**2)), narx_freerun_fit=float(fit_n),
                     lam_min=float(lam[0]), lam_max=float(lam[-1]), mu_L_median=float(mu), max_phi2=float((Xo**2).sum(1).max()))
    print("m=n=%d: theta-RLS MSE=%.3e fit=%6.1f%% | NARX-EKF MSE=%.3e fit=%6.1f%% | lam_min=%.1e mu_L(med)=%.1e max||phi||^2=%.2f"
          % (order, E7[order]['theta_MSE'], fit_t, E7[order]['narx_MSE'], fit_n, lam[0], mu, E7[order]['max_phi2']))
A.m = A.n = m; A.Ni = Ni; A.O = O
OUT['E7_order'] = {str(k): v for k, v in E7.items()}

# ----------------------------------------------------------------------------
#  E8  synthetic persistently exciting benchmark, known w* (12 plant/input seeds)
# ----------------------------------------------------------------------------
print("\n=== E8: synthetic PE benchmark with known w* (H=2 tanh plant, 8-level PRBS, q=1e-3, R=0.1) ===")
import itertools
Hs = 2; Os = Hs*Ni + Hs; qs, Rs_ = 1e-3, 0.1; Ns = 3000; vb_syn = 0.01
def fwd2(w, x):
    Wh = w[:Hs*Ni].reshape(Hs, Ni); Wo = w[Hs*Ni:]; a = Wh@x; z = np.tanh(a); yn = Wo@z
    return yn, np.concatenate([((Wo*(1-z**2))[:, None]*x[None, :]).ravel(), z])
def synth_case(seed):
    rng = np.random.default_rng(seed)
    us = np.empty(Ns); k = 0
    while k < Ns:
        h = int(rng.integers(3, 11)); us[k:k+h] = rng.choice(np.linspace(0.0, 1.0, 8)); k += h
    Wh = rng.normal(0, 1.0, (Hs, Ni)); Wh[:, m+1:] *= 0.4; Wo = rng.normal(0, 0.8, Hs)
    w_true = np.concatenate([Wh.ravel(), Wo]); yt = np.zeros(Ns)
    for k in range(n, Ns):
        x = np.concatenate([us[k-np.arange(0, m+1)], yt[k-np.arange(1, n+1)]]); yt[k] = fwd2(w_true, x)[0]
    ym = yt + rng.uniform(-vb_syn, vb_syn, Ns); ym[:n] = yt[:n]
    Xs = np.array([np.concatenate([us[k-np.arange(0, m+1)], ym[k-np.arange(1, n+1)]]) for k in range(n, Ns)])
    w = rng.uniform(0, 0.1, Os); P = P0*np.eye(Os); wh, pmx, pmn, eh, Sh, Jh = [], [], [], [], [], []
    for i, k in enumerate(range(n, Ns)):
        x = Xs[i]; yn, J = fwd2(w, x); e = ym[k]-yn; P = P + qs*np.eye(Os); S = J@P@J + Rs_; K = P@J/S; w = w + K*e
        P = P - np.outer(K, J@P); ev = np.linalg.eigvalsh(P); pmx.append(ev[-1]); pmn.append(ev[0]); wh.append(w.copy()); eh.append(e); Sh.append(S)
        Jh.append(J.copy())
    wh, pmx, pmn, eh, Sh, Jh = map(np.array, (wh, pmx, pmn, eh, Sh, Jh))
    best = np.inf
    for perm in itertools.permutations(range(Hs)):
        for signs in itertools.product([1, -1], repeat=Hs):
            Wh2 = Wh[list(perm)]*np.array(signs)[:, None]; Wo2 = Wo[list(perm)]*np.array(signs)
            wt2 = np.concatenate([Wh2.ravel(), Wo2]); dd = np.linalg.norm(wt2-wh[-1])
            if dd < best: best = dd; wstar = wt2
    om = np.linalg.norm(wstar[None, :]-wh, axis=1)
    hs = np.array([fwd2(wstar, x)[0] for x in Xs]); v = ym[n:]-hs; vmax = float(np.abs(v).max())
    vrms = float(np.sqrt(np.mean(v**2)))
    mu = np.array([np.linalg.eigvalsh(Xs[s0:s0+L].T@Xs[s0:s0+L]/L)[0] for s0 in range(0, len(Xs)-L+1, L)])
    muJ = np.array([np.linalg.eigvalsh(Jh[s0:s0+L].T@Jh[s0:s0+L]/L)[0] for s0 in range(0, len(Jh)-L+1, L)])
    muJ_post = np.array([np.linalg.eigvalsh(Jh[s0:s0+L].T@Jh[s0:s0+L]/L)[0]
                         for s0 in range(1500, len(Jh)-L+1, L)])
    # free run of the final model
    ys = ym.copy()
    for k in range(n, Ns):
        x = np.concatenate([us[k-np.arange(0, m+1)], ys[k-np.arange(1, n+1)]]); ys[k] = fwd2(wh[-1], x)[0]
    fit = float(100*(1-np.linalg.norm(ym[n:]-ys[n:])/np.linalg.norm(ym[n:]-ym[n:].mean())))
    seg = {}
    for k0 in (0, 1500):
        pmax = float(pmx[k0:].max()); pmin = float(pmn[k0:].min()); ombar = float(om[k0:].max()); cb = float(np.abs(wh[k0:, Hs*Ni:]).max())
        M, r_ = A.curvature(cb+ombar, Xs, ombar); d = vmax + r_; rho = pmax/(pmax+qs); fV = d**2/(Rs_*(1-rho)); ball = float(np.sqrt(pmax*fV))
        dirball = float(np.sqrt(pmin*fV))
        seg[k0] = dict(pmax=pmax, pmin=pmin, omega_bar=ombar, r=float(r_), d=float(d), ball=ball, directional_ball=dirball,
                       vrms=vrms, r_over_vrms=float(r_/vrms),
                       amp_mu=float(1/np.sqrt(mu.min())),
                       amp_muJ=(float(1/np.sqrt(muJ_post.min())) if (k0 and muJ_post.size and muJ_post.min() > 0)
                                else (float(1/np.sqrt(muJ.min())) if muJ.min() > 0 else float('inf'))),
                       self_consistent=bool(pmax*fV <= ombar**2), MSE=float(np.mean(eh[k0:]**2)), innov=float(np.sum((eh**2/Sh)[k0:])))
    return dict(om_final=float(om[-1]), vmax=vmax, vrms=vrms,
                muJ_min=float(muJ.min()), muJ_med=float(np.median(muJ)),
                muJ_post_min=float(muJ_post.min()) if muJ_post.size else float('nan'),
                mu_min=float(mu.min()), mu_med=float(np.median(mu)), fit=fit, seg=seg,
                om_curve=om, pmax_curve=pmx, MSE=float(np.mean(eh**2)))
E8 = {}
best_seed, best_ball = None, np.inf
for seed in range(1, 13):
    r = synth_case(seed); E8[seed] = {k: v for k, v in r.items() if not k.endswith('curve')}
    s1 = r['seg'][1500]; s0 = r['seg'][0]
    print("seed %2d: mu_L(min)=%.1e | ||omega(N)||=%.3f (nearest symmetric copy) one-step MSE=%.1e free-run fit=%5.1f%% | full-horizon ball=%.1e | "
          "segment k>=1500: p_max=%.2f omega_bar=%.3f r_bar=%.3f d_bar=%.3f v_rms=%.4f r/v_rms=%.1f mu_L=%.1e muJ_L=%.1e 1/sqrt(muJ)=%.1f ball=%.1e (dir. %.1e) self-consistent=%s"
          % (seed, r['mu_min'], r['om_final'], r['MSE'], r['fit'], s0['ball'], s1['pmax'], s1['omega_bar'], s1['r'], s1['d'],
             s1['vrms'], s1['r_over_vrms'], r['mu_min'], r['muJ_post_min'], s1['amp_muJ'], s1['ball'], s1['directional_ball'], s1['self_consistent']))
    if s1['ball'] < best_ball: best_ball, best_seed, best_run = s1['ball'], seed, r
print("best seed %d" % best_seed)
_amp = [E8[s_]['seg'][1500]['amp_muJ'] for s_ in E8]
_rv  = [E8[s_]['seg'][1500]['r_over_vrms'] for s_ in E8]
print("  amplification 1/sqrt(muJ_L) over the 12 seeds: %.1f to %.1f" % (min(_amp), max(_amp)))
_mj = [E8[s_]['muJ_post_min'] for s_ in E8]
print("  Jacobian window PE muJ_L (k>=1500):           %.2e to %.2e" % (min(_mj), max(_mj)))
# which excitation level separates the seeds whose weights converge?
def _rank(a):
    a = np.asarray(a, float); r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
def _spearman(a, b):
    ra, rb = _rank(a), _rank(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra@rb)/np.sqrt((ra@ra)*(rb@rb)))
_seeds = sorted(E8)
_om  = [E8[s_]['om_final'] for s_ in _seeds]
_muJ = [E8[s_]['muJ_post_min'] for s_ in _seeds]
_muR = [E8[s_]['mu_min'] for s_ in _seeds]
rho_J = _spearman(np.log10(_muJ), _om); rho_R = _spearman(np.log10(_muR), _om)
_conv = sorted(s_ for s_ in _seeds if E8[s_]['om_final'] < 0.9)          # converged seeds
_topJ = sorted(sorted(_seeds, key=lambda s_: -E8[s_]['muJ_post_min'])[:len(_conv)])
print("  Spearman rank correlation with ||omega(N)||: muJ_L %.2f, mu_L %.2f" % (rho_J, rho_R))
print("  converged seeds (||omega(N)||<0.9): %s ; seeds with the largest muJ_L: %s ; identical: %s"
      % (_conv, _topJ, _conv == _topJ))
print("  muJ_L on converged seeds: %.1e to %.1e ; on the others: %.1e to %.1e"
      % (min(E8[s_]['muJ_post_min'] for s_ in _conv), max(E8[s_]['muJ_post_min'] for s_ in _conv),
         min(E8[s_]['muJ_post_min'] for s_ in _seeds if s_ not in _conv),
         max(E8[s_]['muJ_post_min'] for s_ in _seeds if s_ not in _conv)))
E8_sep = dict(spearman_muJ=rho_J, spearman_mu=rho_R, converged=_conv, top_muJ=_topJ,
              identical=bool(_conv == _topJ))
print("  remainder/residual r_bar/v_rms:               %.1f to %.1f" % (min(_rv), max(_rv)))
print("  seed %d (best): r_bar=%.3f, v_rms=%.4f, ratio=%.1f"
      % (best_seed, E8[best_seed]['seg'][1500]['r'], E8[best_seed]['seg'][1500]['vrms'],
         E8[best_seed]['seg'][1500]['r_over_vrms']))
OUT['E8_synthetic_nonlinear'] = {str(k): {kk: (vv if kk != 'seg' else {str(a): b for a, b in vv.items()}) for kk, vv in v.items()} for k, v in E8.items()}
OUT['E8_synthetic_nonlinear']['config'] = dict(H=Hs, q=qs, R=Rs_, N=Ns, vbar=vb_syn, seeds=12)
OUT['E8_synthetic_nonlinear']['separation'] = E8_sep

# Tier-1 on a synthetic linear ARMA plant with a PRBS input: Theorem 2 bound vs true error
rng = np.random.default_rng(2026); us = np.empty(Ns); k = 0
while k < Ns:
    h = int(rng.integers(3, 11)); us[k:k+h] = rng.choice(np.linspace(0.0, 1.0, 8)); k += h
theta_true = np.array([0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.9, -0.3, 0.15, -0.08, 0.05])   # full-order ARX plant, dominant pole ~0.9
yl = np.zeros(Ns)
for k in range(n, Ns):
    phi = np.concatenate([us[k-np.arange(0, m+1)], yl[k-np.arange(1, n+1)]]); yl[k] = phi@theta_true
yl_meas = yl + rng.uniform(-vb_syn, vb_syn, Ns); yl_meas[:n] = yl[:n]
Xl = np.array([np.concatenate([us[k-np.arange(0, m+1)], yl_meas[k-np.arange(1, n+1)]]) for k in range(n, Ns)])
theta_ref_l, *_ = np.linalg.lstsq(Xl, yl_meas[n:], rcond=None)   # best fixed linear predictor on the measured regressor (Remark 1(ii))
eiv_bias = float(np.linalg.norm(theta_ref_l-theta_true))
v_l = yl_meas[n:]-Xl@theta_ref_l; vbar_l = float(np.abs(v_l).max())
mu_l = np.array([np.linalg.eigvalsh(Xl[s0:s0+L].T@Xl[s0:s0+L]/L)[0] for s0 in range(0, len(Xl)-L+1, L)])
r = theta_rls(1.0, q=0.0, th0=np.zeros(Ni), udata=us, ydata=yl_meas)
om_l = np.linalg.norm(theta_ref_l[None, :]-r['th'], axis=1)
tau = np.arange(1, len(om_l)+1); om0 = np.linalg.norm(theta_ref_l)
bound = (om0**2/P0 + tau*vbar_l**2/1.0)/(1/P0 + (tau//L)*L*mu_l.min()/1.0)
bound_rms = (om0**2/P0 + np.cumsum(v_l**2)/1.0)/(1/P0 + (tau//L)*L*mu_l.min()/1.0)
# directional Tier-1 bound (Cauchy-Schwarz in the P^{-1} inner product) along the best-excited direction of P(N)
evN, UN = np.linalg.eigh(r['P'][-1]); zbest = UN[:, 0]
Vk_l = np.array([(theta_ref_l-t)@np.linalg.inv(P)@(theta_ref_l-t) for t, P in zip(r['th'], r['P'])])
dir_true = np.array([(zbest@(theta_ref_l-t))**2 for t in r['th']])
dir_bound = np.array([(zbest@P@zbest) for P in r['P']])*(om0**2/P0 + np.cumsum(v_l**2)/1.0)
print("  directional: best direction of P(N): true (z^T omega)^2 = %.2e, bound (z^T P z) V-bound = %.2e (ratio %.1f)" % (dir_true[-1], dir_bound[-1], dir_bound[-1]/dir_true[-1]))
E8lin_dir = dict(true=dir_true, bound=dir_bound)
print("synthetic linear plant: EIV bias ||theta_LS-theta_true||=%.3f; mu_L(min)=%.3e (median %.2e), v_bar=%.4f, ||omega(N)||^2=%.2e vs Theorem-2 bound %.2e (ratio %.1f; with exact sum v^2: %.2e); "
      "bound holds at every k: %s" % (eiv_bias, mu_l.min(), np.median(mu_l), vbar_l, om_l[-1]**2, bound[-1], bound[-1]/om_l[-1]**2, bound_rms[-1], bool((om_l**2 <= bound).all())))
OUT['E8_synthetic_linear'] = dict(dir_true_final=float(dir_true[-1]), dir_bound_final=float(dir_bound[-1]), eiv_bias=eiv_bias, mu_L_min=float(mu_l.min()), mu_L_median=float(np.median(mu_l)), vbar=vbar_l, omega_final_sq=float(om_l[-1]**2),
                                  bound_final=float(bound[-1]), bound_final_exactsum=float(bound_rms[-1]), holds=bool((om_l**2 <= bound).all()))
E8lin = dict(om=om_l, bound=bound, bound_rms=bound_rms, dir_true=dir_true, dir_bound=dir_bound)

# ----------------------------------------------------------------------------
#  E9  candidate explanations for the R = 0.05 MATLAB/Python discrepancy
# ----------------------------------------------------------------------------
print("\n=== E9: candidates for the earlier 2.3587e-4 at R = 0.05 (ARMA, linear-activation network) ===")
E9 = {}
def variant(name, **kw):
    r = w_ekf(False, 0.05, **kw); mse = float(np.mean(r['e']**2)) if r['status'] == 'ok' else float('nan')
    E9[name] = mse; print("  %-40s MSE=%.4e" % (name, mse))
variant('nominal (prediction-first, q=1e-6)')
variant('q=1e-5', q=1e-5); variant('q=1e-4', q=1e-4); variant('q=0', q=0.0)
# a-posteriori error (after the update) instead of innovation
r = w_ekf(False, 0.05); ea = []
for i, k in enumerate(range(n, N)):
    ea.append(y[k]-forward(r['w'][i], xvec(k), False)[0])
E9['a-posteriori error'] = float(np.mean(np.array(ea)**2)); print("  %-40s MSE=%.4e" % ('a-posteriori (post-update) error', E9['a-posteriori error']))
# trailing +Q ordering
w = np.full(O, W0); P = P0*np.eye(O); eh = []
for k in range(n, N):
    x = xvec(k); yn, J, _ = forward(w, x, False); e = y[k]-yn; eh.append(e)
    S = J@P@J+0.05; K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)+Q_*np.eye(O)
E9['trailing +Q'] = float(np.mean(np.array(eh)**2)); print("  %-40s MSE=%.4e" % ('trailing +Q ordering', E9['trailing +Q']))
# single precision
w = np.full(O, W0, dtype=np.float32); P = (P0*np.eye(O)).astype(np.float32); eh = []
for k in range(n, N):
    x = xvec(k).astype(np.float32); yn, J, _ = forward(w, x, False); J = J.astype(np.float32); e = np.float32(y[k])-yn; eh.append(float(e))
    S = J@P@J+np.float32(0.05); K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)+np.float32(Q_)*np.eye(O, dtype=np.float32)
E9['single precision'] = float(np.mean(np.array(eh)**2)); print("  %-40s MSE=%.4e" % ('single precision, trailing +Q', E9['single precision']))
# tanh-derivative Jacobian used with a linear forward pass (a plausible legacy-code slip)
w = np.full(O, W0); P = P0*np.eye(O); eh = []
for k in range(n, N):
    x = xvec(k); Wh, Wo = unpack(w); a = Wh@x; yn = Wo@a; e = y[k]-yn; eh.append(e)
    dz = 1-np.tanh(a)**2; J = np.concatenate([((Wo*dz)[:, None]*x[None, :]).ravel(), a])
    P = P+Q_*np.eye(O); S = J@P@J+0.05; K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)
E9['tanh-derivative Jacobian'] = float(np.mean(np.array(eh)**2)); print("  %-40s MSE=%.4e" % ('tanh-derivative Jacobian, linear output', E9['tanh-derivative Jacobian']))
# MSE over the first 1000 / 500 samples only
r = w_ekf(False, 0.05)
for T in (100, 500, 1000):
    E9[f'MSE over first {T}'] = float(np.mean(r['e'][:T]**2)); print("  %-40s MSE=%.4e" % (f'MSE over first {T} samples', E9[f'MSE over first {T}']))

# --- candidates found by inspecting the released MATLAB livescript ----------
# The legacy code (a) starts its loop at k = 1 with a zero-padded regressor,
# (b) scales the hidden layer by der1 = 0.5, and (c) assigns Rekf = 1/alpha
# after Rekf = 0.05, so the executed R may have been 1.
def legacy_run(R, der=1.0, from_k1=False, qq=Q_):
    """ARMA-FNN EKF in weight space with optional MATLAB conventions."""
    w = np.full(O, W0); P = P0*np.eye(O); eh = []
    k0 = 0 if from_k1 else n
    for k in range(k0, N):
        if from_k1:                                   # zero-padded regressor
            uu = np.array([u[k-j] if k-j >= 0 else 0.0 for j in range(0, m+1)])
            yy = np.array([y[k-j] if k-j >= 1 else 0.0 for j in range(1, n+1)])
            x = np.concatenate([uu, yy])
        else:
            x = xvec(k)
        Wh, Wo = unpack(w); a = der*(Wh@x); yn = Wo@a
        e = y[k]-yn; eh.append(e)
        J = np.concatenate([(der*Wo[:, None]*x[None, :]).ravel(), a])
        P = P+qq*np.eye(O); S = J@P@J+R; K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)
        if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
            return float('nan')
    return float(np.mean(np.array(eh)**2))

for nm, kw in [('loop from k=1, zero-padded regressor', dict(R=0.05, from_k1=True)),
               ('hidden-layer scale der1 = 0.5',        dict(R=0.05, der=0.5)),
               ('both (k=1 and der1 = 0.5)',            dict(R=0.05, der=0.5, from_k1=True)),
               ('R = 1 (Rekf = 1/alpha overwrites)',    dict(R=1.0)),
               ('R = 1, k=1, der1 = 0.5',               dict(R=1.0, der=0.5, from_k1=True))]:
    E9[nm] = legacy_run(**kw); print("  %-40s MSE=%.4e" % (nm, E9[nm]))

_tgt = 2.3587e-4
_fin = {kk: vv for kk, vv in E9.items() if np.isfinite(vv)}
_best = min(_fin, key=lambda kk: abs(np.log10(_fin[kk]/_tgt)))
print("  -> closest to the earlier %.4e: %s (ratio %.2f)" % (_tgt, _best, _fin[_best]/_tgt))
OUT['E9_matlab_discrepancy'] = E9

# ----------------------------------------------------------------------------
#  Figures
# ----------------------------------------------------------------------------
# Fig 4 (new): innovation energy with the right-hand sides drawn
curves = np.load("theta_rls_curves.npy", allow_pickle=True).item()
fig, ax = plt.subplots(figsize=(6, 3.4))
inn, _ = curves['1.0']; tau = np.arange(1, len(inn)+1)
ax.plot(np.arange(n, n+len(inn)), inn, 'C0', lw=1.3, label=r'$\theta$-RLS, $R=1$, $Q=0$: $\sum e^2/S$')
ax.plot(np.arange(n, n+len(inn)), E1[1.0]['V0'] + tau*vbar_rms**2/1.0, 'C0', ls='--', lw=1.0,
        label=r'RHS of (17), $\bar v=$ RMS residual (%.4f)' % vbar_rms)
ax.plot(np.arange(n, n+len(inn)), E1[1.0]['V0'] + np.cumsum(v_ls**2)/1.0, 'C0', ls=':', lw=1.0,
        label=r'RHS of (17), exact $\sum v^2/R$')
rN = w_ekf(True, 1.0); innN = np.cumsum(rN['e']**2/rN['S'])
ax.plot(np.arange(n, n+len(innN)), innN, 'C3', lw=1.3, label=r'NARX-FNN EKF, $R=1$, $q=10^{-6}$: $\sum e^2/S$ (RHS of (33) with $\bar d$: %.1e, off scale)' % nx['rhs31'])
ax.set_xlabel('instance k'); ax.set_ylabel(r'$\sum_{i\leq k} e^2(i)/S(i)$')
ax.set_title('Innovation energy against the certified right-hand side')
ax.legend(fontsize=6.5, loc='upper left'); fig.tight_layout(); fig.savefig("fig4_innov.png", dpi=150); plt.close(fig)

# Fig 9 (new): certified floor vs R and empirical MSE
fig, ax1 = plt.subplots(figsize=(6, 3.2))
Rs = [2.0, 1.0, 0.05]
for arch, c in (('ARMA', 'C0'), ('NARX', 'C3')):
    ax1.loglog(Rs, [E23[f"{arch}-{R}"]['floor_w'] for R in Rs], marker='o', color=c, label=f'{arch}: certified radius $\\Omega$ from (32)')
ax1.set_xlabel('$R$'); ax1.set_ylabel('certified floor (log)')
ax2 = ax1.twinx()
for arch, c in (('ARMA', 'C0'), ('NARX', 'C3')):
    ax2.semilogx(Rs, [E23[f"{arch}-{R}"]['MSE'] for R in Rs], marker='s', ls='--', color=c, label=f'{arch}: one-step MSE')
ax2.set_ylabel('one-step MSE'); ax2.set_ylim(0, 1.1e-4)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, fontsize=7, loc='upper center'); ax1.set_title('What (32) trades off in $R$, and what the record shows')
fig.tight_layout(); fig.savefig("fig9_floor_vs_R.png", dpi=150); plt.close(fig)

# Fig 10 (new): synthetic nonlinear benchmark, best seed: true ||omega(k)||, ball, p_max(k)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True)
kk = np.arange(n, n+len(best_run['om_curve']))
a1.semilogy(kk, best_run['om_curve'], 'C3', lw=1.1, label=r'true $\|\omega(k)\|$ (nearest symmetric copy of $w^*$)')
a1.axhline(best_run['seg'][0]['ball'], color='k', ls='--', lw=1, label='ball (32), full horizon: %.1e' % best_run['seg'][0]['ball'])
a1.axhline(best_run['seg'][1500]['ball'], color='C2', ls='-.', lw=1, label=r'ball (32), segment $k\geq1500$: %.1f' % best_run['seg'][1500]['ball'])
a1.axhline(best_run['seg'][1500]['directional_ball'], color='C0', ls=':', lw=1, label=r'directional ball (35), best direction: %.2f' % best_run['seg'][1500]['directional_ball'])
a1.set_ylabel(r'$\|\omega(k)\|$'); a1.legend(fontsize=6.5, loc='upper right')
a1.set_title('Synthetic PE benchmark, seed %d (tanh plant $H=2$, EKF $R=0.1$, $q=10^{-3}$)' % best_seed, fontsize=9)
a2.semilogy(kk, best_run['pmax_curve'], 'C0', lw=1.1, label=r'$\lambda_{\max}(P(k))$: contracts in every direction on PE data')
a2.set_ylabel(r'$p_{\max}(k)$'); a2.set_xlabel('instance k'); a2.legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig10_synth.png", dpi=150); plt.close(fig)

# Fig 11 (new): synthetic linear plant: Theorem 2 bound vs true squared error
fig, ax = plt.subplots(figsize=(6, 3.0))
ax.semilogy(np.arange(n, n+len(E8lin['om'])), E8lin['om']**2, 'C0', lw=1.1, label=r'true $\|\omega(\tau)\|^2$ ($\theta$-RLS, $R=1$, $Q=0$)')
ax.semilogy(np.arange(n, n+len(E8lin['om'])), E8lin['bound'], 'k--', lw=1, label=r'bound (19), $\bar v=\max|v|$, measured $\mu_L$')
ax.semilogy(np.arange(n, n+len(E8lin['om'])), E8lin['bound_rms'], 'k:', lw=1, label=r'bound (19) with exact $\sum v^2/R$')
ax.semilogy(np.arange(n, n+len(E8lin['om'])), E8lin['dir_true'], 'C3', lw=1.0, label=r'true $(z^\top\omega)^2$, best-excited direction $z$')
ax.semilogy(np.arange(n, n+len(E8lin['om'])), E8lin['dir_bound'], 'C3', ls='--', lw=1.0, label=r'directional bound (20), same $z$')
ax.set_xlabel(r'instance $\tau$'); ax.set_ylabel('squared parameter error'); ax.legend(fontsize=7)
ax.set_title('Theorem 2 on a persistently exciting synthetic linear record', fontsize=9); fig.tight_layout(); fig.savefig("fig11_synth_linear.png", dpi=150); plt.close(fig)

json.dump(OUT, open("revision2_results.json", "w"), indent=1, default=float)
print("\nrevision2_results.json written; figures fig4_innov.png, fig9_floor_vs_R.png, fig10_synth.png, fig11_synth_linear.png")
