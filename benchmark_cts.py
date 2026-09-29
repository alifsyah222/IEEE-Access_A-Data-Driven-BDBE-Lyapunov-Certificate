#!/usr/bin/env python3
# Algorithm 1 on the public Cascaded Tanks benchmark (Schoukens, Mattson,
# Wigren, Noel, 2016; nonlinearbenchmark.org). 1024 estimation + 1024
# validation samples, Ts = 4 s, multisine input, lower-tank level output.
# Same identifiers, same hyperparameters, same audit as on the ITB record.
import numpy as np, json, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = [r for r in list(csv.reader(open("cts.csv")))[1:] if len(r) >= 4 and r[0].strip()]
uE = np.array([float(r[0]) for r in rows]); uV = np.array([float(r[1]) for r in rows])
yE = np.array([float(r[2]) for r in rows]); yV = np.array([float(r[3]) for r in rows])
# normalize to [0,1] with the ESTIMATION set statistics (as the ITB record is normalized)
umin, umax = uE.min(), uE.max(); ymin, ymax = yE.min(), yE.max()
nu = lambda x: (x-umin)/(umax-umin); ny = lambda x: (x-ymin)/(ymax-ymin)
u, y = nu(uE), ny(yE); uv, yv = nu(uV), ny(yV)
N = len(u); m = n = 5; H = 5; Ni = 2*m+1; O = H*Ni+H; P0 = 50.0; W0 = 0.05; L = 200
OUT = dict(N=N, Ts=4, y_scale=float(ymax-ymin))

def xv(uu, yy, k): return np.concatenate([uu[k-np.arange(0, m+1)], yy[k-np.arange(1, n+1)]])
def forward(w, x, nl):
    Wh = w[:H*Ni].reshape(H, Ni); Wo = w[H*Ni:]; a = Wh@x; z = np.tanh(a) if nl else a
    dz = (1-z**2) if nl else np.ones(H)
    return Wo@z, np.concatenate([((Wo*dz)[:, None]*x[None, :]).ravel(), z])
def conv_instance(e, e_tol=0.01, hold=8, T=120, gate=1.5e-4):
    if np.mean(e**2) > gate: return 'degraded'
    ea = np.abs(e[:T])
    for k0 in range(len(ea)-hold):
        if (ea[k0:k0+hold] <= e_tol).all(): return int(k0+1)
    return 'degraded'
def curvature(cbar, X, om):
    xn = np.sqrt((X**2).sum(1)); M = float((xn+(4/(3*np.sqrt(3)))*cbar*xn**2).max()); return M, 0.5*M*om**2
def fit(yy, ys): return float(100*(1-np.linalg.norm(yy-ys)/np.linalg.norm(yy-yy.mean())))

# ---------------- data statistics (Algorithm 1, steps 1-3)
X = np.array([xv(u, y, k) for k in range(n, N)]); Y = y[n:]
max_phi2 = float((X**2).sum(1).max()); alpha_star = 2/max_phi2
Info = X.T@X/len(X); ev = np.linalg.eigvalsh(Info)
mu_win = np.array([np.linalg.eigvalsh(X[s:s+L].T@X[s:s+L]/L)[0] for s in range(0, len(X)-L+1, L)])
mu_slide = np.array([np.linalg.eigvalsh(X[s:s+L].T@X[s:s+L]/L)[0] for s in range(0, len(X)-L+1)])
theta_ls, *_ = np.linalg.lstsq(X, Y, rcond=None); v_ls = Y-X@theta_ls
vbar_rms, vbar_max = float(np.sqrt(np.mean(v_ls**2))), float(np.abs(v_ls).max())
print("=== Cascaded Tanks: data statistics ===")
print("N=%d max||phi||^2=%.3f alpha*=%.4f lam_min/max=%.2e/%.3f mu_L(min)=%.2e median=%.2e (windows: %d) frac sliding<1e-12: %.1f%%"
      % (N, max_phi2, alpha_star, ev[0], ev[-1], mu_win.min(), np.median(mu_win), len(mu_win), 100*np.mean(mu_slide < 1e-12)))
print("best fixed linear model: RMS residual %.4e, max|v| %.4e" % (vbar_rms, vbar_max))
OUT['stats'] = dict(max_phi2=max_phi2, alpha_star=alpha_star, lam_min=float(ev[0]), lam_max=float(ev[-1]),
                    mu_L_min=float(mu_win.min()), mu_L_median=float(np.median(mu_win)), n_windows=int(len(mu_win)),
                    vbar_rms=vbar_rms, vbar_max=vbar_max)

# ---------------- Tier 1: theta-RLS, Q = 0
def theta_rls(R, th0):
    th = th0.copy(); P = P0*np.eye(Ni); e_h, S_h, th_h, P_h = [], [], [], []
    for i in range(len(X)):
        phi = X[i]; e = Y[i]-phi@th; S = phi@P@phi+R; K = P@phi/S; th = th+K*e; P = P-np.outer(K, phi@P)
        e_h.append(e); S_h.append(S); th_h.append(th.copy()); P_h.append(P.copy())
    return np.array(e_h), np.array(S_h), np.array(th_h), P_h
def freerun_theta(th, uu, yy):
    ys = yy.copy()
    for k in range(n, len(uu)):
        ys[k] = xv(uu, ys, k)@th
        if not np.isfinite(ys[k]) or abs(ys[k]) > 1e6: return float('-inf')
    return fit(yy[n:], ys[n:])
th0 = np.full(Ni, H*W0*W0)
# The LS reference is ill-conditioned here (lambda_min*N/R << 1/p0), so the
# declared reference is the prior-regularized limit of the R=1 recursion
# (the ridge solution with prior p0 I around theta(0)); Remark 1 allows any
# admissible pair. It is used for ALL three R runs, so the R=2 and R=0.05
# checks are not tautological.
print("||theta_LS|| = %.1f (ill-conditioned: lambda_min*N/R = %.1e vs prior 1/p0 = %.2e)" % (np.linalg.norm(theta_ls), ev[0]*len(X), 1/P0))
_, _, th_ref_run, _ = theta_rls(1.0, th0); theta_ref = th_ref_run[-1]; v_ref = Y-X@theta_ref
vref_rms, vref_max = float(np.sqrt(np.mean(v_ref**2))), float(np.abs(v_ref).max())
print("declared reference = ridge limit of the R=1 recursion: ||theta*|| = %.2f, RMS residual %.4e, max|v| %.4e; free-run fit val %.1f%%" % (np.linalg.norm(theta_ref), vref_rms, vref_max, freerun_theta(theta_ref, uv, yv)))
OUT['stats'].update(theta_ls_norm=float(np.linalg.norm(theta_ls)), theta_ref_norm=float(np.linalg.norm(theta_ref)), vref_rms=vref_rms, vref_max=vref_max)
theta_ls, v_ls, vbar_rms, vbar_max = theta_ref, v_ref, vref_rms, vref_max
print("\n=== Tier 1: theta-RLS (Q=0), reference = ridge limit at R=1 ===")
T1 = {}
for R in (2.0, 1.0, 0.05):
    e, S, th, Ph = theta_rls(R, th0)
    om_prev = theta_ls[None, :]-np.vstack([th0[None, :], th[:-1]]); om_k = theta_ls[None, :]-th
    Pinv_prev = [np.eye(Ni)/P0]+[np.linalg.inv(P) for P in Ph[:-1]]
    Vprev = np.array([o@Pi@o for o, Pi in zip(om_prev, Pinv_prev)]); Vk = np.array([o@np.linalg.inv(P)@o for o, P in zip(om_k, Ph)])
    resid = np.abs(Vk-(Vprev+v_ls**2/R-e**2/S)).max()
    innov = float(np.sum(e**2/S)); V0 = float(om_prev[0]@om_prev[0]/P0); rhs = V0+np.sum(v_ls**2)/R; rhs_max = V0+len(e)*vbar_max**2/R
    # Theorem 2 bound with theta_LS reference and mu_L(min); directional in best direction
    tau = np.arange(1, len(e)+1); om0 = np.linalg.norm(om_prev[0])
    bound2 = (om0**2/P0+np.cumsum(v_ls**2)/R)/(1/P0+(tau//L)*L*mu_win.min()/R)
    evN, UN = np.linalg.eigh(Ph[-1]); z = UN[:, 0]
    dir_true = float((z@om_k[-1])**2); dir_bound = float((z@Ph[-1]@z)*(om0**2/P0+np.sum(v_ls**2)/R))
    holds = bool((np.linalg.norm(om_k, axis=1)**2 <= bound2).all())
    T1[R] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), conv_nogate=conv_instance(e, gate=1.0), innov=innov, rhs=float(rhs), rhs_max=float(rhs_max), identity_residual=float(resid),
                 VN=float(Vk[-1]), omega_N_sq=float(np.linalg.norm(om_k[-1])**2), bound19_N=float(bound2[-1]), bound19_holds=holds,
                 dir_true=dir_true, dir_bound=dir_bound, freerun_est=freerun_theta(th[-1], u, y), freerun_val=freerun_theta(th[-1], uv, yv),
                 pmin=float(np.linalg.eigvalsh(Ph[-1])[0]), pmax_N=float(np.linalg.eigvalsh(Ph[-1])[-1]))
    r = T1[R]
    print("R=%5.2f MSE=%.3e conv=%s (no gate: %s) | sum e^2/S=%.4f vs RHS(17)=%.4f (max|v|: %.2f) | identity residual %.1e | ||omega(N)||^2=%.2e vs (19)=%.2e holds=%s | dir: %.1e vs %.1e | free-run fit est %.1f%% val %.1f%%"
          % (R, r['MSE'], r['conv'], r['conv_nogate'], innov, rhs, rhs_max, resid, r['omega_N_sq'], r['bound19_N'], holds, dir_true, dir_bound, r['freerun_est'], r['freerun_val']))
    if R == 2.0: T1curves = dict(innov=np.cumsum(e**2/S), rhs=V0+np.cumsum(v_ls**2)/R, om2=np.linalg.norm(om_k, axis=1)**2, b19=bound2)
OUT['tier1'] = {str(k): v for k, v in T1.items()}
OUT['tier1']['freerun_val_LS'] = freerun_theta(theta_ls, uv, yv)
print("free-run fit of theta_LS on validation: %.1f%%" % OUT['tier1']['freerun_val_LS'])

# theta-SGD sweep (sharpness of alpha*)
def theta_sgd(alpha, th0):
    th = th0.copy()
    for i in range(len(X)):
        phi = X[i]; e = Y[i]-phi@th; th = th+alpha*e*phi
        if not np.isfinite(th).all() or np.abs(th).max() > 1e6: return 'div'
    return 'ok'
onset = next((a for a in np.round(np.arange(0.05, 2.0, 0.01), 3) if theta_sgd(a, th0) == 'div'), None)
rng = np.random.default_rng(12345); onsets = []
for s_ in range(30):
    w0 = rng.uniform(0, 0.10, O); Wh = w0[:H*Ni].reshape(H, Ni); Wo = w0[H*Ni:]
    onsets.append(next((a for a in np.round(np.arange(0.05, 2.0, 0.01), 3) if theta_sgd(a, Wo@Wh) == 'div'), None))
print("theta-SGD divergence onset: nominal %s, 30 seeds in [%s, %s]; alpha* = %.3f" % (onset, min(onsets), max(onsets), alpha_star))
OUT['sgd'] = dict(onset_nominal=onset, onset_min=min(onsets), onset_max=max(onsets))

# ---------------- Tier 2: NARX-FNN EKF in weight space
def w_ekf(nl, R, q=1e-6, w0=None):
    w = np.full(O, W0) if w0 is None else w0.copy(); P = P0*np.eye(O)
    e_h, S_h, w_h, pmx, pmn, cmax, supJ2 = [], [], [], [], [], 0.0, 0.0
    for i, k in enumerate(range(n, N)):
        x = X[i]; yn, J = forward(w, x, nl); e = Y[i]-yn; supJ2 = max(supJ2, float(J@J))
        P = P+q*np.eye(O); S = J@P@J+R; K = P@J/S; w = w+K*e; P = P-np.outer(K, J@P)
        ev = np.linalg.eigvalsh(P); pmx.append(ev[-1]); pmn.append(ev[0]); cmax = max(cmax, float(np.abs(w[H*Ni:]).max()))
        e_h.append(e); S_h.append(S); w_h.append(w.copy())
    return dict(e=np.array(e_h), S=np.array(S_h), w=np.array(w_h), pmax=np.array(pmx), pmin=np.array(pmn), cbar=cmax, supJ2=supJ2, P=P)
def freerun_w(w, nl, uu, yy):
    ys = yy.copy()
    for k in range(n, len(uu)):
        ys[k], _ = forward(w, xv(uu, ys, k), nl)
        if not np.isfinite(ys[k]) or abs(ys[k]) > 1e6: return float('-inf')
    return fit(yy[n:], ys[n:])
print("\n=== Tier 2: weight-space EKF runs (H=5, q=1e-6, w(0)=0.05) ===")
T2 = {}
for arch, nl in (('ARMA', False), ('NARX', True)):
    for R in (2.0, 1.0, 0.05):
        r = w_ekf(nl, R); e, S = r['e'], r['S']
        for k0, tag in ((0, 'full'), (300, 'seg')):
            wf = r['w'][-1]; om_bar = float(np.max(np.linalg.norm(r['w'][k0:]-wf, axis=1)))
            pmax = float(r['pmax'][k0:].max()); pmin = float(r['pmin'][k0:].min())
            M, rb = curvature(r['cbar']+om_bar, X, om_bar); d = vbar_rms+rb; rho = pmax/(pmax+1e-6)
            Omega = float(np.sqrt(pmax*d**2/(R*(1-rho))))
            T2[f"{arch}-{R}-{tag}"] = dict(MSE=float(np.mean(e**2)), conv=conv_instance(e), conv_nogate=conv_instance(e, gate=1.0), innov=float(np.sum(e**2/S)), rhs33=float(om_bar**2/P0+len(e)*d**2/R),
                                          omega_bar=om_bar, pmax=pmax, pmin=pmin, dirs_above_45=int((np.linalg.eigvalsh(r['P']) > 45).sum()),
                                          M=float(M), r_bar=float(rb), d_bar=float(d), Omega=Omega, self_consistent=bool(Omega <= om_bar),
                                          supJ2=float(r['supJ2']), freerun_est=freerun_w(wf, nl, u, y), freerun_val=freerun_w(wf, nl, uv, yv))
        t = T2[f"{arch}-{R}-full"]; ts = T2[f"{arch}-{R}-seg"]
        print("%s R=%5.2f MSE=%.3e conv=%s (no gate %s) supJ2=%.2f | innov %.3f vs (33) %.2e | p_max=%.2f (seg %.2f) p_min=%.1e dirs>45: %d | omega_bar=%.2f d_bar=%.2f Omega=%.2e (seg %.2e) sc=%s | free-run fit est %.1f%% val %.1f%%"
              % (arch, R, t['MSE'], t['conv'], t['conv_nogate'], t['supJ2'], t['innov'], t['rhs33'], t['pmax'], ts['pmax'], t['pmin'], t['dirs_above_45'], t['omega_bar'], t['d_bar'], t['Omega'], ts['Omega'], t['self_consistent'], t['freerun_est'], t['freerun_val']))
        if arch == 'NARX' and R == 1.0: T2curves = dict(pmax=r['pmax'], innov=np.cumsum(e**2/S))
OUT['tier2'] = T2
# Monte Carlo over initializations for NARX R=1 (free-run on validation, p_max)
rng = np.random.default_rng(12345); mc = []
for s_ in range(30):
    r = w_ekf(True, 1.0, w0=rng.uniform(0, 0.10, O))
    mc.append((float(np.mean(r['e']**2)), float(r['pmax'].max()), float(r['pmax'][-1]), freerun_w(r['w'][-1], True, uv, yv), float(r['supJ2'])))
mc = np.array(mc)
print("NARX R=1, 30 random inits: MSE %.2e±%.1e | p_max(run) %.1f-%.1f | p_max(N) %.2f-%.2f | free-run val fit %.1f to %.1f%% (median %.1f) | supJ2 %.2f-%.2f"
      % (mc[:, 0].mean(), mc[:, 0].std(), mc[:, 1].min(), mc[:, 1].max(), mc[:, 2].min(), mc[:, 2].max(), mc[:, 3].min(), mc[:, 3].max(), np.median(mc[:, 3]), mc[:, 4].min(), mc[:, 4].max()))
OUT['mc'] = dict(MSE_mean=float(mc[:, 0].mean()), MSE_std=float(mc[:, 0].std()), pmax_run=[float(mc[:, 1].min()), float(mc[:, 1].max())],
                 pmax_N=[float(mc[:, 2].min()), float(mc[:, 2].max())], fit_val=[float(mc[:, 3].min()), float(np.median(mc[:, 3])), float(mc[:, 3].max())],
                 supJ2=[float(mc[:, 4].min()), float(mc[:, 4].max())])
# validation RMSE in original units for the best NARX model (comparability with the literature)
def rmse_val(w, nl):
    ys = yv.copy()
    for k in range(n, len(uv)): ys[k], _ = forward(w, xv(uv, ys, k), nl)
    return float(np.sqrt(np.mean((yV[n:]-(ys[n:]*(ymax-ymin)+ymin))**2)))
r = w_ekf(True, 1.0); OUT['rmse_val_narx_nominal'] = rmse_val(r['w'][-1], True)
print("NARX R=1 nominal: validation free-run RMSE in original units: %.3f (level range %.2f)" % (OUT['rmse_val_narx_nominal'], ymax-ymin))

json.dump(OUT, open("cts_results.json", "w"), indent=1, default=float)

# ---------------- figure
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
ax[0, 0].plot(u, 'k', lw=0.7); ax[0, 0].plot(y, 'b', lw=0.7); ax[0, 0].set_title('Cascaded tanks, estimation record (normalized)', fontsize=8); ax[0, 0].set_xlabel('k'); ax[0, 0].legend(['u', 'y'], fontsize=7)
ax[0, 1].semilogy(np.arange(n, n+len(mu_slide)), np.maximum(mu_slide, 1e-16), 'C3', lw=1); ax[0, 1].axhline(ev[0], ls='--', color='k', lw=1)
ax[0, 1].set_title(r'window PE level $\mu_L$ (sliding, $L=200$)', fontsize=8); ax[0, 1].set_xlabel('window start')
ax[1, 0].plot(np.arange(n, N), T1curves['innov'], 'C0', lw=1.2, label=r'$\theta$-RLS $R=2$: $\sum e^2/S$'); ax[1, 0].plot(np.arange(n, N), T1curves['rhs'], 'C0--', lw=1, label='RHS of (17)')
ax[1, 0].plot(np.arange(n, N), T2curves['innov'], 'C3', lw=1.2, label='NARX EKF $R=1$'); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel('k'); ax[1, 0].set_title('innovation energy vs bound', fontsize=8)
ax[1, 1].semilogy(np.arange(n, N), T1curves['om2'], 'C0', lw=1.2, label=r'$\|\omega(\tau)\|^2$, $\theta$-RLS'); ax[1, 1].semilogy(np.arange(n, N), T1curves['b19'], 'k--', lw=1, label='bound (19)')
ax[1, 1].semilogy(np.arange(n, N), T2curves['pmax'], 'C2', lw=1, label=r'$p_{\max}(k)$, NARX EKF'); ax[1, 1].legend(fontsize=7); ax[1, 1].set_xlabel('k'); ax[1, 1].set_title('parameter bound and covariance', fontsize=8)
fig.tight_layout(); fig.savefig('fig12_cts.png', dpi=150)
print("fig12_cts.png written")
