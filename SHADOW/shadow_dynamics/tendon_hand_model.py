"""Tendon-structured hand model of experiment 49, in the form of Li, Chen, Ringwald, Pozo Fortunic, Ganguly, Haddadin (arXiv 2408.13044), eqs. (1)-(3):

    M(q) qdd + N(q, qd) + G(q) + h(q, qd) = C^T (f_t - f_f(f_t, ldot)) [+ J^T f_ext],     ldot = C qd
    h = tau_k(q) + tau_d(qd) + tau_f(qd)

  M, N, G   rigid-body terms from the kinematics only (kinematics_only.Kinematics) and 10 inertial parameters per body, physically consistent by
            construction: each body's 4x4 pseudo-inertia J = [[Sigma, m c], [m c^T, m]] = U U^T with U upper triangular (positive diagonal).
            Plus a per-joint armature on the diagonal of M.
  h         per joint: tau_k = k0 + k1 q + k2 q^2 (elasticity), tau_d = d qd (damping), tau_f = Fc tanh(qd / 0.02) (Coulomb).
  tendons   a configured list (shadow_dynamics/data/tendons_sim.json): each tendon crosses some joints with constant moment arms (a row of C, learned)
            and is either 'measured' (force f_t from a sensor or the actuator model) or 'passive' (elastic force from its length, PhysX fixed-tendon
            form: f = -k (L - L0) - k_lim * (L - clip(L, lo, hi)) - d Ldot). Tendon friction on measured tendons:
            f_f = (mu0 + mu1 |f_t|) tanh(ldot / 0.02).
  actuator  command -> measured-tendon force. 'pd_joint' (simulation): f_t = clip(K (a - l) - D ldot, +-u_max) per actuator tendon, l its length.
            The real hand gets its own form; the plant does not change.

Integration (as PhysX): implicit Euler at 120 Hz, 6 substeps per 20 Hz command. The actuator's PD terms (when not saturated) and the passive
tendons' spring and damping are taken implicitly through their linearisation; the rigid-body bias, h and f_f explicitly. Joint limits clamp q
and zero qd. In 'measured' mode the recorded f_t replaces the actuator (applied explicitly, held over the substep).

Interface units: deg and deg/s in and out (state = 24 angles + 24 velocities); SI inside.
"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
import torch
from .kinematics_only import Kinematics, load_kinematics

V_EPS = 0.02   # rad/s, tanh smoothing of Coulomb terms


def _inv_softplus(y): return y + torch.log(-torch.expm1(-y))


def pseudo_from_physical(m, c, I_com):
    """m (B,), c (B, 3) COM in the body frame, I_com (B, 3, 3) about the COM in the body frame -> pseudo-inertia (B, 4, 4)."""
    Ib = I_com + m[:, None, None] * ((c * c).sum(-1)[:, None, None] * torch.eye(3, dtype=m.dtype, device=m.device) - c[:, :, None] * c[:, None, :])   # about the body origin
    S = 0.5 * Ib.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * torch.eye(3, dtype=m.dtype, device=m.device) - Ib
    J = torch.zeros(len(m), 4, 4, dtype=m.dtype, device=m.device); J[:, :3, :3] = S; J[:, :3, 3] = m[:, None] * c; J[:, 3, :3] = m[:, None] * c; J[:, 3, 3] = m
    return J


def physical_from_pseudo(J):
    m = J[:, 3, 3]; hvec = J[:, :3, 3]; S = J[:, :3, :3]; c = hvec / m[:, None]
    Ib = S.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * torch.eye(3, dtype=J.dtype, device=J.device) - S
    I_com = Ib - m[:, None, None] * ((c * c).sum(-1)[:, None, None] * torch.eye(3, dtype=J.dtype, device=J.device) - c[:, :, None] * c[:, None, :])
    return m, c, I_com


class TendonHand(torch.nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = copy.deepcopy(cfg); dev = torch.device(device)
        self.kin = Kinematics(cfg["kinematics"] if isinstance(cfg["kinematics"], dict) else load_kinematics(cfg["kinematics"]), dev)
        k = self.kin; self.names = k.joint_names; nj = self.nj = len(self.names); nb = self.nb = len(k.body_names)
        self.act = k.active; self.pas = k.passive; self.substeps = cfg["physics_hz"] // cfg["control_hz"]; self.h = 1.0 / cfg["physics_hz"]
        self.register_buffer("g_root", torch.tensor(cfg.get("gravity_root", [0.0, 0.0, 0.0]), dtype=torch.float32, device=dev))
        moving = k.sub.amax(0) > 0; self.register_buffer("moving", moving)   # bodies that move with some joint (the rest never enter the dynamics)
        # inertial parameters: upper-triangular U per body, J = U U^T; diagonal through log, from a generic small body unless initialised
        iu = torch.triu_indices(4, 4, 1); self.register_buffer("iu", iu)
        self.U_logdiag = torch.nn.Parameter(torch.zeros(nb, 4, device=dev)); self.U_off = torch.nn.Parameter(torch.zeros(nb, iu.shape[1], device=dev))
        self.set_inertial(*self._generic_bodies(nb, dev))
        self.raw_arm = torch.nn.Parameter(torch.full((nj,), -9.0, device=dev))                 # armature = 1e-3 softplus(raw) kg m^2
        self.k_el = torch.nn.Parameter(torch.zeros(nj, 3, device=dev))                          # tau_k coefficients (N m, N m/rad, N m/rad^2)
        self.d_j = torch.nn.Parameter(torch.zeros(nj, device=dev))                              # N m s/rad
        self.raw_fc = torch.nn.Parameter(torch.full((nj,), -9.0, device=dev))                  # Fc = 0.01 softplus(raw) N m
        T = cfg["tendons"]; self.tendons = T["tendons"]; nt = self.nt = len(self.tendons)
        Cm = torch.zeros(nt, nj, device=dev); Cmask = torch.zeros(nt, nj, dtype=torch.bool, device=dev); Fm = torch.zeros(nt, nj, device=dev)
        for i, t in enumerate(self.tendons):
            for jn, r, fc in zip(t["joints"], t["moment_arms"], t.get("force_arms", t["moment_arms"])):
                Cm[i, self.names.index(jn)] = r; Cmask[i, self.names.index(jn)] = True; Fm[i, self.names.index(jn)] = fc
        self.C_raw = torch.nn.Parameter(Cm); self.register_buffer("C_mask", Cmask)
        # force transmission: Li et al. (and any real, lossless tendon) have tau = C^T f, i.e. force arms = moment arms. PhysX fixed tendons
        # distribute the force with separate 'forceCoefficient's, so a tendon may carry its own force arms (learned; equal to C when absent).
        self.split_force = [i for i, t in enumerate(self.tendons) if "force_arms" in t]
        self.Cf_raw = torch.nn.Parameter(Fm)
        self.meas = [i for i, t in enumerate(self.tendons) if t["kind"] == "measured"]; self.passive_t = [i for i, t in enumerate(self.tendons) if t["kind"] == "passive"]
        npv = len(self.passive_t); self.n_meas = len(self.meas)
        self.pt_logk = torch.nn.Parameter(torch.full((npv,), -9.0, device=dev)); self.pt_logkl = torch.nn.Parameter(torch.full((npv,), 0.0, device=dev))
        self.pt_logd = torch.nn.Parameter(torch.full((npv,), -9.0, device=dev)); self.pt_L0 = torch.nn.Parameter(torch.zeros(npv, device=dev))
        self.pt_lo = torch.nn.Parameter(torch.full((npv,), -0.05, device=dev)); self.pt_hi = torch.nn.Parameter(torch.full((npv,), 0.05, device=dev))
        # simulator-only pinned term (experiment 49 gate 0): while the passive tendon's 'pin_joint' (J0) sits at a joint limit, PhysX adds
        # tau = -k_pin (L - L_pin) on 'pin_target' (J1). Tendons without 'pin_joint' (the real-hand config) have no such term.
        self.pin_t = [i for i in self.passive_t if "pin_joint" in self.tendons[i]]
        self.pin_j0 = [self.names.index(self.tendons[i]["pin_joint"]) for i in self.pin_t]; self.pin_j1 = [self.names.index(self.tendons[i]["pin_target"]) for i in self.pin_t]
        self.pin_logk = torch.nn.Parameter(torch.full((len(self.pin_t),), -9.0, device=dev)); self.pin_L0 = torch.nn.Parameter(torch.zeros(len(self.pin_t), device=dev))
        self.mu = torch.nn.Parameter(torch.zeros(self.n_meas, 2, device=dev))                  # tendon friction mu0 (N or N m), mu1 (-)
        # actuator ('pd_joint'): one measured tendon per commanded joint, in action order
        A = cfg["actuator"]; self.actuator_kind = A["kind"]; assert self.actuator_kind == "pd_joint"
        self.act_tendon = [self.tendons[i]["name"] for i in self.meas]; assert len(self.meas) == len(self.act)
        self.log_K = torch.nn.Parameter(torch.full((self.n_meas,), float(np.log(A.get("K0", 1.0))), device=dev))
        self.log_D = torch.nn.Parameter(torch.full((self.n_meas,), float(np.log(A.get("D0", 0.1))), device=dev))
        self.log_umax = torch.nn.Parameter(torch.full((self.n_meas,), float(np.log(A.get("umax0", 1.0))), device=dev))
        if "identified" in cfg: self.load_values(cfg["identified"])

    # ---------- parameters in physical units
    @staticmethod
    def _generic_bodies(nb, dev):   # a 10 g sphere-like body at its frame origin: the from-scratch starting point
        return torch.full((nb,), 0.01, device=dev), torch.zeros(nb, 3, device=dev), torch.eye(3, device=dev).expand(nb, 3, 3) * 1e-6

    def U(self):   # off-diagonals relative to their row's diagonal, so every raw parameter is O(1) whatever the body's size
        d = self.U_logdiag.exp(); U = torch.diag_embed(d).clone(); U[:, self.iu[0], self.iu[1]] = self.U_off * d[:, self.iu[0]]; return U

    def pseudo(self): U = self.U(); return U @ U.transpose(-1, -2)
    def inertial(self): return physical_from_pseudo(self.pseudo())

    @torch.no_grad()
    def set_inertial(self, m, c, I_com):
        J = pseudo_from_physical(m.double(), c.double(), I_com.double()); P = torch.flip(torch.eye(4, dtype=torch.float64, device=J.device), [0])
        L = torch.linalg.cholesky(P @ J @ P); U = (P @ L @ P).float()   # J = U U^T, U upper triangular
        d = U.diagonal(dim1=-2, dim2=-1); self.U_logdiag.copy_(torch.log(d)); self.U_off.copy_(U[:, self.iu[0], self.iu[1]] / d[:, self.iu[0]])

    def arm(self): return 1e-3 * torch.nn.functional.softplus(self.raw_arm)
    def Fc(self): return 0.01 * torch.nn.functional.softplus(self.raw_fc)
    def C(self): return self.C_raw * self.C_mask

    def Cf(self):
        """Force-transmission matrix: C for every tendon without separate force arms (the Li et al. form), its own arms otherwise."""
        if not self.split_force: return self.C()
        own = torch.zeros(self.nt, 1, dtype=torch.bool, device=self.C_raw.device); own[self.split_force] = True
        return torch.where(own, self.Cf_raw * self.C_mask, self.C())
    def K(self): return self.log_K.exp()
    def D(self): return self.log_D.exp()
    def umax(self): return self.log_umax.exp()

    # ---------- rigid-body terms
    def _bodies(self, q):
        T, ax, org = self.kin(q); R = T[..., :3, :3]; p = T[..., :3, 3]
        m, c, Ic = self.inertial(); cw = (R @ c[None, :, :, None])[..., 0] + p; Iw = R @ Ic[None] @ R.transpose(-1, -2)
        return R, p, ax, org, m, cw, Iw

    def mass_matrix(self, q, bodies=None):
        """(N, nj, nj): sum_b m Jv^T Jv + Jw^T Iw Jw over bodies, plus the armature."""
        R, p, ax, org, m, cw, Iw = bodies if bodies is not None else self._bodies(q); sub = self.kin.sub   # (nj, B)
        Jw = ax[:, :, None, :] * sub[None, :, :, None]                                                     # (N, nj, B, 3)
        Jv = torch.cross(ax[:, :, None, :].expand(-1, -1, cw.shape[1], -1), cw[:, None] - org[:, :, None], dim=-1) * sub[None, :, :, None]
        M = torch.einsum("njbi,b,nkbi->njk", Jv, m, Jv) + torch.einsum("njbi,nbil,nkbl->njk", Jw, Iw, Jw)
        return M + torch.diag_embed(self.arm())[None]

    def rnea(self, q, qd, qdd, gravity=True, bodies=None):
        """Generalised force (N, nj) that produces qdd at (q, qd): M qdd + N(q, qd) [+ G(q)] (no armature, no h, no tendons)."""
        R, p, ax, org, m, cw, Iw = bodies if bodies is not None else self._bodies(q); N_ = len(q); dev = q.device; nb = self.nb
        w = [None] * nb; al = [None] * nb; ap = [None] * nb; z3 = torch.zeros(N_, 3, device=dev)
        w[self.kin.root] = z3; al[self.kin.root] = z3; ap[self.kin.root] = z3
        for pa, c, ji in self.kin.order:
            if ji >= 0:
                z = ax[:, ji]; wd = z * qd[:, ji:ji + 1]; o = org[:, ji]
                w[c] = w[pa] + wd; al[c] = al[pa] + z * qdd[:, ji:ji + 1] + torch.cross(w[pa], wd, dim=-1)
            else:
                o = p[:, c]; w[c] = w[pa]; al[c] = al[pa]
            r = o - p[:, pa]; ao = ap[pa] + torch.cross(al[pa], r, dim=-1) + torch.cross(w[pa], torch.cross(w[pa], r, dim=-1), dim=-1)
            r2 = p[:, c] - o; ap[c] = ao + torch.cross(al[c], r2, dim=-1) + torch.cross(w[c], torch.cross(w[c], r2, dim=-1), dim=-1)
        W = torch.stack(w, 1); AL = torch.stack(al, 1); AP = torch.stack(ap, 1); rc = cw - p
        ac = AP + torch.cross(AL, rc, dim=-1) + torch.cross(W, torch.cross(W, rc, dim=-1), dim=-1)
        g = self.g_root if gravity else torch.zeros(3, device=dev)
        f = m[None, :, None] * (ac - g)                                                                    # (N, B, 3) force on each body
        IwW = (Iw @ W[..., None])[..., 0]; n = (Iw @ AL[..., None])[..., 0] + torch.cross(W, IwW, dim=-1)   # moment about each COM
        arm_ = cw[:, None] - org[:, :, None]                                                               # (N, nj, B, 3)
        mom = n[:, None] + torch.cross(arm_, f[:, None].expand_as(arm_), dim=-1)
        return ((mom * ax[:, :, None]).sum(-1) * self.kin.sub[None]).sum(-1)

    def motion(self, q, qd, qdd):
        """Forward recursion only: body origins p, rotations R (N, B, 3[,3]), angular velocity / acceleration and origin acceleration (N, B, 3), joint axes / origins."""
        T, ax, org = self.kin(q); R = T[..., :3, :3]; p = T[..., :3, 3]; N_ = len(q); dev = q.device; nb = self.nb
        w = [None] * nb; al = [None] * nb; ap = [None] * nb; z3 = torch.zeros(N_, 3, device=dev)
        w[self.kin.root] = z3; al[self.kin.root] = z3; ap[self.kin.root] = z3
        for pa, c, ji in self.kin.order:
            if ji >= 0:
                z = ax[:, ji]; wd = z * qd[:, ji:ji + 1]; o = org[:, ji]
                w[c] = w[pa] + wd; al[c] = al[pa] + z * qdd[:, ji:ji + 1] + torch.cross(w[pa], wd, dim=-1)
            else:
                o = p[:, c]; w[c] = w[pa]; al[c] = al[pa]
            r = o - p[:, pa]; ao = ap[pa] + torch.cross(al[pa], r, dim=-1) + torch.cross(w[pa], torch.cross(w[pa], r, dim=-1), dim=-1)
            r2 = p[:, c] - o; ap[c] = ao + torch.cross(al[c], r2, dim=-1) + torch.cross(w[c], torch.cross(w[c], r2, dim=-1), dim=-1)
        return p, R, torch.stack(w, 1), torch.stack(al, 1), torch.stack(ap, 1), ax, org

    SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def inertial_regressor(self, q, qd, qdd, gravity=True):
        """Rigid-body regressor Y (N, nj, B, 10): tau_rb = sum_b Y[..., b, :] @ pi_b with the standard parameters per body in its own frame,
        pi_b = (m, m c_x, m c_y, m c_z, Ibar_xx, Ibar_yy, Ibar_zz, Ibar_xy, Ibar_xz, Ibar_yz) (Ibar about the body origin). Linear in pi."""
        p, R, W, AL, AP, ax, org = self.motion(q, qd, qdd)
        g = self.g_root if gravity else torch.zeros(3, device=q.device); a_g = AP - g                          # origin acceleration minus gravity
        arm_ = p[:, None] - org[:, :, None]                                                                   # (N, nj, B, 3)
        axb = ax[:, :, None, :]; sub = self.kin.sub[None, :, :, None]
        cols = [ (axb * torch.cross(arm_, a_g[:, None].expand_as(arm_), dim=-1)).sum(-1) ]                     # m
        for k in range(3):
            rk = R[..., k]; f = torch.cross(AL, rk, dim=-1) + torch.cross(W, torch.cross(W, rk, dim=-1), dim=-1); n = torch.cross(rk, a_g, dim=-1)
            cols.append((axb * (n[:, None] + torch.cross(arm_, f[:, None].expand_as(arm_), dim=-1))).sum(-1))
        for i, j in self.SYM6:
            E = torch.zeros(3, 3, device=q.device); E[i, j] = 1; E[j, i] = 1
            Iw = R @ E @ R.transpose(-1, -2); n = (Iw @ AL[..., None])[..., 0] + torch.cross(W, (Iw @ W[..., None])[..., 0], dim=-1)
            cols.append((axb * n[:, None]).sum(-1))
        return torch.stack(cols, -1) * sub

    def standard_params(self):
        """(B, 10) standard inertial parameters of the current model (same order as inertial_regressor)."""
        J = self.pseudo(); m = J[:, 3, 3]; hv = J[:, :3, 3]; S = J[:, :3, :3]; Ib = S.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * torch.eye(3, device=J.device) - S
        return torch.cat([m[:, None], hv, torch.stack([Ib[:, i, j] for i, j in self.SYM6], -1)], -1)

    def gravity_torque(self, q): return self.rnea(q, torch.zeros_like(q), torch.zeros_like(q), gravity=True)
    def coriolis(self, q, qd): return self.rnea(q, qd, torch.zeros_like(q), gravity=False)

    # ---------- joint and tendon terms
    def h_terms(self, q, qd):
        kq = self.k_el; return kq[:, 0] + kq[:, 1] * q + kq[:, 2] * q * q + self.d_j * qd + self.Fc() * torch.tanh(qd / V_EPS)

    def passive_force(self, q, qd):
        """Passive tendon forces (N, npv) and their linearisation (stiffness, damping) for the implicit step."""
        C = self.C()[self.passive_t]; L = q @ C.T; Ld = qd @ C.T
        k, kl, d = self.pt_logk.exp(), self.pt_logkl.exp(), self.pt_logd.exp(); lo, hi = self.pt_lo, torch.maximum(self.pt_hi, self.pt_lo + 1e-6)
        viol = L - torch.minimum(torch.maximum(L, lo), hi); out = (viol != 0).float()
        f = -k * (L - self.pt_L0) - kl * viol - d * Ld
        return f, k + kl * out, d.expand_as(f), C, self.Cf()[self.passive_t]

    PIN_EPS = 0.005   # rad: J0 counts as pinned within this distance of a limit

    def pinned_torque(self, q):
        """Simulator-only: (N, nj) torque of the pinned-J0 lock (zero for configs without 'pin_joint')."""
        tau = torch.zeros_like(q)
        if not self.pin_t: return tau
        C = self.C()[self.pin_t]; L = q @ C.T; j0 = self.pin_j0
        pinned = ((q[:, j0] < self.kin.lo[j0] + self.PIN_EPS) | (q[:, j0] > self.kin.hi[j0] - self.PIN_EPS)).float()
        return tau.index_add(1, torch.tensor(self.pin_j1, device=q.device), -self.pin_logk.exp() * (L - self.pin_L0) * pinned)

    def tendon_friction(self, f_meas, qd):
        ld = qd @ self.C()[self.meas].T; return (self.mu[:, 0] + self.mu[:, 1] * f_meas.abs()) * torch.tanh(ld / V_EPS)

    def actuator_force(self, a, q, qd):
        """'pd_joint': f = clip(K (a - l) - D ldot, +-u_max) on the actuator tendons; a (N, 20) rad in action order."""
        C = self.C()[self.meas]; l = q @ C.T; ld = qd @ C.T; u = self.K() * (a - l) - self.D() * ld
        return u.clamp(-self.umax(), self.umax()), (u.abs() > self.umax())

    # ---------- dynamics
    def step(self, q, v, a=None, f_meas=None):
        """One implicit substep. q, v (N, nj) rad, rad/s; a (N, 20) rad command, or f_meas (N, n_meas) measured actuator-tendon force."""
        bodies = self._bodies(q); M = self.mass_matrix(q, bodies); bias = self.rnea(q, v, torch.zeros_like(q), True, bodies)
        h = self.h; nj = self.nj; dev = q.device
        Kq = torch.zeros(len(q), nj, nj, device=dev); Dv = torch.zeros_like(Kq)
        tau = -bias - self.h_terms(q, v) + self.pinned_torque(q)
        fp, kp, dp, Cp, Cpf = self.passive_force(q, v)
        tau = tau + fp @ Cpf
        Kq = Kq + torch.einsum("ti,nt,tj->nij", Cpf, kp, Cp); Dv = Dv + torch.einsum("ti,nt,tj->nij", Cpf, dp, Cp)
        Ca = self.C()[self.meas]
        if f_meas is None:
            fa, sat = self.actuator_force(a, q, v); lin = (~sat).float()
            Kq = Kq + torch.einsum("ti,nt,tj->nij", Ca, self.K() * lin, Ca); Dv = Dv + torch.einsum("ti,nt,tj->nij", Ca, self.D() * lin, Ca)
            fa_used = fa
        else:
            fa_used = f_meas
        tau = tau + (fa_used - self.tendon_friction(fa_used, v)) @ self.Cf()[self.meas]
        # implicit Euler on the linear parts: (M + h Dv + h^2 Kq) v+ = M v + h (tau + Dv v)  [tau already holds -Kq-part at q and -Dv v]
        lhs = M + h * Dv + h * h * Kq; rhs = (M @ v[..., None])[..., 0] + h * (tau + (Dv @ v[..., None])[..., 0])
        vn = torch.linalg.solve(lhs, rhs[..., None])[..., 0]; qn = q + h * vn
        at = (qn < self.kin.lo) | (qn > self.kin.hi); qn = torch.minimum(torch.maximum(qn, self.kin.lo), self.kin.hi); vn = torch.where(at, torch.zeros_like(vn), vn)
        return qn, vn

    def rollout(self, s, fa=None, f_meas=None, substep_states=False):
        """s (N, 48) deg/deg/s start state; fa (N, H, 20) deg commands, or f_meas (N, H*substeps, n_meas) per-substep measured force.
        Returns (N, H, 48) deg/deg/s at the control rate (or every substep)."""
        q = torch.deg2rad(s[:, :self.nj].float()); v = torch.deg2rad(s[:, self.nj:].float()); out = []
        Hs = fa.shape[1] if fa is not None else f_meas.shape[1] // self.substeps
        for t in range(Hs):
            a = torch.deg2rad(fa[:, t].float()) if fa is not None else None
            for k in range(self.substeps):
                q, v = self.step(q, v, a, None if f_meas is None else f_meas[:, t * self.substeps + k])
                if substep_states: out.append(torch.cat([torch.rad2deg(q), torch.rad2deg(v)], -1))
            if not substep_states: out.append(torch.cat([torch.rad2deg(q), torch.rad2deg(v)], -1))
        return torch.stack(out, 1)

    # ---------- values
    def values(self):
        A = lambda x: x.detach().cpu().numpy().tolist(); m, c, I = self.inertial()
        return dict(joints=list(self.names), bodies=list(self.kin.body_names), mass=A(m), com=A(c), inertia_com=A(I), armature=A(self.arm()), k_el=A(self.k_el),
                    d_joint=A(self.d_j), Fc=A(self.Fc()), C=A(self.C()), Cf=A(self.Cf()), tendons=[t["name"] for t in self.tendons], passive_k=A(self.pt_logk.exp()),
                    passive_k_lim=A(self.pt_logkl.exp()), passive_d=A(self.pt_logd.exp()), passive_L0=A(self.pt_L0), passive_lo=A(self.pt_lo), passive_hi=A(self.pt_hi),
                    tendon_mu=A(self.mu), K=A(self.K()), D=A(self.D()), umax=A(self.umax()), pin_k=A(self.pin_logk.exp()), pin_L0=A(self.pin_L0))

    @torch.no_grad()
    def load_values(self, v):
        dev = self.U_logdiag.device; t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
        self.set_inertial(t(v["mass"]), t(v["com"]), t(v["inertia_com"]))
        self.raw_arm.copy_(_inv_softplus((t(v["armature"]) / 1e-3).clamp_min(1e-6))); self.k_el.copy_(t(v["k_el"])); self.d_j.copy_(t(v["d_joint"]))
        self.raw_fc.copy_(_inv_softplus((t(v["Fc"]) / 0.01).clamp_min(1e-6))); self.C_raw.copy_(t(v["C"])); self.Cf_raw.copy_(t(v.get("Cf", v["C"])))
        self.pt_logk.copy_(torch.log(t(v["passive_k"]).clamp_min(1e-6))); self.pt_logkl.copy_(torch.log(t(v["passive_k_lim"]).clamp_min(1e-6)))
        self.pt_logd.copy_(torch.log(t(v["passive_d"]).clamp_min(1e-6))); self.pt_L0.copy_(t(v["passive_L0"])); self.pt_lo.copy_(t(v["passive_lo"])); self.pt_hi.copy_(t(v["passive_hi"]))
        if "pin_k" in v and self.pin_t: self.pin_logk.copy_(torch.log(t(v["pin_k"]).clamp_min(1e-6))); self.pin_L0.copy_(t(v["pin_L0"]))
        self.mu.copy_(t(v["tendon_mu"])); self.log_K.copy_(torch.log(t(v["K"]))); self.log_D.copy_(torch.log(t(v["D"]))); self.log_umax.copy_(torch.log(t(v["umax"])))


def sim_tendon_config(kin, coupling=None, force_arms=None, pinned=False):
    """Simulation tendon set: one measured actuator tendon per commanded joint (moment arm 1 on its own joint) and one passive coupling tendon
    per finger over J1 and J0 (moment arms from `coupling` = {finger: (arm_J1, arm_J0)}, else a generic (1, -1) start). The coupling tendons carry
    separate force arms (PhysX forceCoefficient, default (1, 1)): PhysX fixed tendons are not reciprocal; and a pinned-J0 lock term on J1
    (pin_joint / pin_target; gate 0: J1 torque -570 L while J0 sits at a limit). Both are simulator-only: the real-hand config omits them."""
    names = kin["joint_names"]; T = []
    for i in kin["active"]:
        n = names[i]; T.append(dict(name=f"act_{n[7:]}", kind="measured", joints=[n], moment_arms=[1.0]))
    for f in ("FF", "MF", "RF", "LF"):
        r = (coupling or {}).get(f, (1.0, -1.0)); fa = (force_arms or {}).get(f, (1.0, 1.0))
        t = dict(name=f"cpl_{f}", kind="passive", joints=[f"robot0_{f}J1", f"robot0_{f}J0"], moment_arms=list(r), force_arms=list(fa))
        if pinned: t.update(pin_joint=f"robot0_{f}J0", pin_target=f"robot0_{f}J1")   # off by default: neither LS nor rollout fits could learn it (exp 49)
        T.append(t)
    return dict(tendons=T, note="simulation: 20 measured actuator tendons (Isaac drive torque) + 4 passive J1-J0 coupling tendons (PhysX fixed tendons)")
