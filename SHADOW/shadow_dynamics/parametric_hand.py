"""Parametric (grey-box) Shadow Hand model with identifiable physical parameters (experiment 36).

Every actuated joint j follows (Isaac hand: gravity off, PhysX implicit position drive):

    M_j(q) qdd_j = K_j (a_j - q_j) - D_j qd_j - Fc_j tanh(qd_j / 0.02) + sum_f kappa_{f,j} delta_f(q) + tau_ff_j [+ tau_obs_j]
    M_j(q)   = composite subtree inertia about joint j's axis at the window start (FK + parallel-axis, physics_prior.NominalHand) + armature_j
    delta_f  = clip(c_f q_J1 + b_f, J0 limits) - (c_f q_J1 + b_f)   fixed-tendon stretch of finger f (nonzero only with J0 at a limit)
    q_J0,f   = clip(c_f q_J1 + b_f, J0 limits),  qd_J0,f = c_f qd_J1 inside the limits                      passive distal joints

Parameters (stored raw; `values()` gives physical units): tau_j = D_j / K_j [s], K_j [N m/rad], armature_j [kg m^2], Fc_j [N m] for the 20
actuated joints, c_f [-] and b_f [rad] for FF/MF/RF/LF, kappa_{f,j} [N m/rad] on that finger's flexion / metacarpal joints. Implicit Euler
at 120 Hz (6 substeps per 20 Hz action), joint limits clamp q and zero qd. tau_ff is a known feed-forward torque (torque-probe data, else 0);
tau_obs is the optional load observer of the nominal prior (an ablation, not physics).

Experiment 37a, cfg['residual_observer'] = {gain, max_nm}: a residual disturbance observer. From the window's last 3 transitions it fits one
constant torque per actuated joint, d_j = gain * sum_i g_ij e_ij / sum_i g_ij^2, where e_ij is the joint-angle error of the model's own one-step
prediction and g_ij the one-step angle response to a unit torque (diagonal, by finite difference). d is clipped to +-max_nm and held over the
horizon. Unlike tau_obs it only estimates what the identified model misses, so it does not double-count the tendon terms.
Experiment 44: cfg['residual_observer']['exclude'] = [joint names] zeroes d on those joints.

Units at the interface match physics_prior.NominalHand.rollout: deg, deg/s in and out; radians internally.
"""
from __future__ import annotations
import copy, json, math
from pathlib import Path
import numpy as np
import torch
from .physics_prior import NominalHand, FINGERS

GROUPS = (("WRJ1", "WRJ0"), ("FFJ3", "FFJ2", "FFJ1"), ("MFJ3", "MFJ2", "MFJ1"), ("RFJ3", "RFJ2", "RFJ1"), ("LFJ4", "LFJ3", "LFJ2", "LFJ1"),
          ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"))   # torque-transmission blocks (within one finger / the wrist)
KAPPA_JOINTS = {"FF": ("FFJ2", "FFJ1"), "MF": ("MFJ2", "MFJ1"), "RF": ("RFJ2", "RFJ1"), "LF": ("LFJ4", "LFJ2", "LFJ1")}
V_EPS = 0.02   # rad/s, Coulomb smoothing


class ParametricHand(torch.nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = copy.deepcopy(cfg); base_cfg = copy.deepcopy(cfg); base_cfg.pop("identified", None)
        self.base = NominalHand(base_cfg, device); b = self.base; dev = torch.device(device)
        self.names = b.joint_names; self.act = b.act; n = len(self.names)
        am = torch.zeros(n, dtype=torch.bool, device=dev); am[self.act] = True; self.register_buffer("act_mask", am)
        self.register_buffer("lo", b.lo.clone()); self.register_buffer("hi", b.hi.clone())
        self.mid, self.dist = b.mid, b.dist; self.substeps, self.h = b.substeps, b.h
        km = torch.zeros(4, n, device=dev)
        for fi, f in enumerate(FINGERS):
            for j in KAPPA_JOINTS[f]: km[fi, self.names.index(f"robot0_{j}")] = 1
        self.register_buffer("kappa_mask", km)
        an = [self.names[i][7:] for i in self.act]; lm = torch.zeros(20, 20, device=dev)
        for g in GROUPS:
            for a in g:
                for b_ in g:
                    if a != b_: lm[an.index(b_), an.index(a)] = 1   # lam[i, j]: torque appearing at joint i per unit known torque applied at joint j
        self.register_buffer("lam_mask", lm); self.act_names = an
        K0 = b.K.clone(); K0[~am] = 1.0; D0 = b.D.clone(); D0[~am] = 0.1
        self.register_buffer("K_nom", K0); self.register_buffer("tau_nom", D0 / K0); self.register_buffer("arm_nom", b.arm.clone())
        # raw parameters, initialised at the nominal values
        self.log_tau = torch.nn.Parameter(torch.log(D0 / K0)); self.log_K = torch.nn.Parameter(torch.log(K0))
        self.log_arm = torch.nn.Parameter(torch.log(b.arm.clamp_min(1e-6))); self.raw_fc = torch.nn.Parameter(torch.full((n,), -8.0, device=dev))
        self.c = torch.nn.Parameter(b.c.clone()); self.b = torch.nn.Parameter(b.b.clone()); self.kappa = torch.nn.Parameter(torch.zeros(4, n, device=dev))
        self.lam = torch.nn.Parameter(torch.zeros(20, 20, device=dev))   # known-torque transmission (probe data only; Li et al.'s C^T analogue)
        self.observer = bool(cfg.get("disturbance", False))
        if "identified" in cfg: self.load_values(cfg["identified"]["values"])
        self._M_sub = None

    # ---- physical values
    def K(self): return self.log_K.exp()
    def D(self): return self.log_tau.exp() * self.log_K.exp()
    def arm(self): return self.log_arm.exp()
    def Fc(self): return torch.nn.functional.softplus(self.raw_fc) * 0.01

    def values(self):
        A = lambda x: x.detach().cpu().numpy()
        act = self.act; nm = [self.names[i] for i in act]
        kap = {f: {j: float(self.kappa[fi, self.names.index(f"robot0_{j}")]) for j in KAPPA_JOINTS[f]} for fi, f in enumerate(FINGERS)}
        return dict(joints=nm, tau_s=A(self.log_tau.exp())[act].tolist(), K_Nm_rad=A(self.K())[act].tolist(), D_Nms_rad=A(self.D())[act].tolist(),
                    armature_kgm2=A(self.arm())[act].tolist(), Fc_Nm=A(self.Fc())[act].tolist(),
                    tendon_gain=dict(zip(FINGERS, A(self.c).tolist())), tendon_offset_rad=dict(zip(FINGERS, A(self.b).tolist())), kappa_Nm_rad=kap,
                    transmission={f"{self.act_names[i]}<-{self.act_names[j]}": float(self.lam[i, j]) for i in range(20) for j in range(20) if self.lam_mask[i, j] > 0})

    @torch.no_grad()
    def load_values(self, v):
        idx = [self.names.index(j) for j in v["joints"]]; t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.log_K.device)
        self.log_tau[idx] = t(v["tau_s"]).log(); self.log_K[idx] = t(v["K_Nm_rad"]).log(); self.log_arm[idx] = t(v["armature_kgm2"]).clamp_min(1e-9).log()
        fc = t(v["Fc_Nm"]).clamp_min(1e-7) / 0.01; self.raw_fc[idx] = fc + torch.log(-torch.expm1(-fc))   # inverse softplus
        self.c.copy_(t([v["tendon_gain"][f] for f in FINGERS])); self.b.copy_(t([v["tendon_offset_rad"][f] for f in FINGERS]))
        self.kappa.zero_()
        for fi, f in enumerate(FINGERS):
            for j, k in v["kappa_Nm_rad"][f].items(): self.kappa[fi, self.names.index(f"robot0_{j}")] = k
        self.lam.zero_()
        for key, x in v.get("transmission", {}).items():
            a, b_ = key.split("<-"); self.lam[self.act_names.index(a), self.act_names.index(b_)] = x

    def to_prior_config(self, extra=None):
        cfg = copy.deepcopy(self.cfg); cfg["identified"] = dict(values=self.values(), **(extra or {})); cfg["disturbance"] = self.observer; return cfg

    # ---- dynamics
    def _passive(self, q):
        q1 = q[:, self.mid]; free = self.c * q1 + self.b
        q0 = torch.minimum(torch.maximum(free, self.lo[self.dist]), self.hi[self.dist]); return q0, q0 - free

    def subtree_inertia(self, q):
        return self.base.effective_inertia(q) - self.base.arm[None]

    def rollout(self, s, fa, pa=None, tau_ff=None, grad=False, tau_ff_direct=None):
        """s (N, >=1, 48) deg/deg/s (last state used), fa (N, H, 20) future applied targets deg, pa (N, >=1, 20) past applied targets (observer only),
        tau_ff (N, H, 20) N m feed-forward torques or None -> (N, H, 48) deg, deg/s."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            x = s[:, -1].float(); q = torch.deg2rad(x[:, :24]); v = torch.deg2rad(x[:, 24:]); A = torch.deg2rad(fa.float())
            with torch.no_grad(): Msub = self.subtree_inertia(q)
            M = Msub + self.arm()[None]; K, D, Fc = self.K()[None], self.D()[None], self.Fc()[None]; h = self.h; act = self.act; am = self.act_mask
            kap = self.kappa * self.kappa_mask
            def loads(q, v):
                _, dlt = self._passive(q); return -Fc * torch.tanh(v / V_EPS) + dlt @ kap
            tau_c = torch.zeros_like(q)
            ro = self.cfg.get("residual_observer")
            if ro:
                assert pa is not None and s.shape[1] >= 2, "the residual observer needs past states and actions"
                tau_c = self.residual_load(s, pa, float(ro.get("gain", 1.0)), float(ro.get("max_nm", 0.3)))
                if ro.get("exclude"):   # experiment 44: joints left to the identified equation (e.g. the tendon-coupled J1)
                    keep = torch.ones(len(self.names), device=tau_c.device); keep[[self.names.index(f"robot0_{j}") for j in ro["exclude"]]] = 0; tau_c = tau_c * keep
            if self.observer:
                assert pa is not None, "the observer needs the last applied target"
                tgt0 = q.clone(); tgt0[:, act] = torch.deg2rad(pa[:, -1].float())
                tau_c = (D * v - K * (tgt0 - q) - loads(q, v)) * am
            out = []
            for k in range(A.shape[1]):
                tgt = q.detach().clone(); tgt[:, act] = A[:, k]
                ff = torch.zeros_like(q)
                if tau_ff is not None or tau_ff_direct is not None:
                    if tau_ff is not None: u = tau_ff[:, k].float(); u = u + u @ (self.lam * self.lam_mask).T
                    else: u = tau_ff_direct[:, k].float()   # residual observer sensitivity: no transmission
                    ff = ff.index_put((torch.arange(len(q), device=q.device)[:, None], torch.as_tensor(act, device=q.device)[None]), u)
                for _ in range(self.substeps):
                    tau_o = loads(q, v) + ff + tau_c
                    vn = (M * v + h * (K * (tgt - q) + tau_o)) / (M + h * D + h * h * K)
                    qn = q + h * vn
                    at = (qn < self.lo) | (qn > self.hi); qn = torch.minimum(torch.maximum(qn, self.lo), self.hi); vn = torch.where(at, torch.zeros_like(vn), vn)
                    q = torch.where(am, qn, q); v = torch.where(am, vn, v)
                    q0, _ = self._passive(q); inside = (q0 > self.lo[self.dist]) & (q0 < self.hi[self.dist])
                    q = q.index_copy(1, torch.as_tensor(self.dist, device=q.device), q0)
                    v = v.index_copy(1, torch.as_tensor(self.dist, device=q.device), torch.where(inside, self.c * v[:, self.mid], torch.zeros_like(q0)))
                out.append(torch.cat([torch.rad2deg(q), torch.rad2deg(v)], -1))
            return torch.stack(out, 1)

    @torch.no_grad()
    def residual_load(self, s, pa, gain=1.0, max_nm=0.3, eps=1e-2):
        """(N, 24) N m constant torque per actuated joint that best explains the model's one-step angle errors over the past window.
        s (N, W+1, 48) deg past states, pa (N, >=W, 20) deg past applied targets; pa[:, i] moves s[:, i] -> s[:, i+1]."""
        W = min(s.shape[1] - 1, pa.shape[1]); N = s.shape[0]
        x0 = s[:, -W - 1:-1].reshape(N * W, 1, -1); x1 = torch.deg2rad(s[:, -W:, :24].reshape(N * W, 24).float()); A = pa[:, -W:].reshape(N * W, 1, 20)
        cfg_ro = self.cfg.pop("residual_observer", None); obs = self.observer; self.observer = False
        try:
            base = torch.deg2rad(self.rollout(x0, A)[:, 0, :24])
            unit = torch.zeros(N * W, 1, 20, device=s.device) + eps
            pert = torch.deg2rad(self.rollout(x0, A, tau_ff_direct=unit)[:, 0, :24])
        finally:
            self.observer = obs
            if cfg_ro is not None: self.cfg["residual_observer"] = cfg_ro
        g = ((pert - base) / eps).reshape(N, W, 24); e = (x1 - base).reshape(N, W, 24)
        d = gain * (g * e).sum(1) / (g.square().sum(1) + 1e-12)
        return (d.clamp(-max_nm, max_nm) * self.act_mask).float()

    # ---- parameter vector for Fisher information (identifiable-or-not analysis)
    def param_spec(self, with_lam=False):
        """List of (name, raw tensor, flat index, physical label) for the parameters that act in the model."""
        spec = []
        for i in self.act:
            j = self.names[i][7:]
            spec += [(f"log_tau[{j}]", self.log_tau, i), (f"log_K[{j}]", self.log_K, i), (f"log_arm[{j}]", self.log_arm, i), (f"raw_fc[{j}]", self.raw_fc, i)]
        for fi, f in enumerate(FINGERS):
            spec += [(f"c[{f}]", self.c, fi), (f"b[{f}]", self.b, fi)]
            for j in KAPPA_JOINTS[f]: spec.append((f"kappa[{f},{j}]", self.kappa, fi * len(self.names) + self.names.index(f"robot0_{j}")))
        if with_lam:
            for i in range(20):
                for j in range(20):
                    if self.lam_mask[i, j] > 0: spec.append((f"lam[{self.act_names[i]}<-{self.act_names[j]}]", self.lam, i * 20 + j))
        return spec


def load_parametric(path_or_cfg, device="cpu"):
    cfg = json.loads(Path(path_or_cfg).read_text()) if isinstance(path_or_cfg, (str, Path)) else path_or_cfg
    return ParametricHand(cfg, device).eval()
