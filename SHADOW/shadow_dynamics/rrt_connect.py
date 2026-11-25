"""Experiment 50: RRT-Connect (Kuffner & LaValle 2000) in the hand's 20-D command space, batched in lockstep over many planning problems.

Each iteration every unsolved problem extends one tree toward a random sample (step `step` deg) and then greedily connects the other tree
toward the new node; all edge checks of the iteration go to the GPU in one batch (HandCSpace.valid on points every `res` deg). Points within
`exempt` deg of a problem's start or goal are not checked, and each problem tolerates the penetration its own start or goal already has
(+0.5 mm; curled-finger goals touch). A straight start-goal line is tried first.
Paths are smoothed by random shortcutting. Returns per problem an (n, 20) waypoint array (deg) or None if no path was found.
"""
from __future__ import annotations
import time
import numpy as np


class BatchRRTConnect:
    def __init__(self, cspace, low, high, step=10.0, res=2.0, exempt=5.0, goal_bias=0.1, seed=0):
        self.cs = cspace; self.low = np.asarray(low, float); self.high = np.asarray(high, float)
        self.step, self.res, self.exempt, self.bias = step, res, exempt, goal_bias; self.rng = np.random.default_rng(seed)

    # ---------- batched segment checks
    def _points(self, segs):
        """segs: list of (problem i, p0, p1) -> flat points, owner segment index, fraction along the segment."""
        P, own, frac = [], [], []
        for k, (i, p0, p1) in enumerate(segs):
            n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / self.res))); f = np.arange(1, n + 1) / n
            P.append(p0[None] + f[:, None] * (p1 - p0)[None]); own.append(np.full(n, k)); frac.append(f)
        if not P: return np.zeros((0, 20)), np.zeros(0, int), np.zeros(0)
        return np.concatenate(P), np.concatenate(own), np.concatenate(frac)

    def _valid_prefix(self, segs):
        """For each segment, the largest fraction f in [0, 1] such that every checked point up to f is valid."""
        P, own, frac = self._points(segs)
        ok = np.ones(len(P), bool)
        if len(P):
            ex = np.zeros(len(P), bool)
            for k, (i, p0, p1) in enumerate(segs):
                m = own == k; ex[m] = (np.linalg.norm(P[m] - self.start[i], axis=1) < self.exempt) | (np.linalg.norm(P[m] - self.goal[i], axis=1) < self.exempt)
            chk = np.where(~ex)[0]
            if len(chk): ok[chk] = self.cs.penetration(P[chk].astype(np.float32)) <= self.thr[np.array([segs[k][0] for k in own[chk]])]
        out = np.ones(len(segs))
        for k in range(len(segs)):
            m = np.where(own == k)[0]; bad = m[~ok[m]]
            if len(bad): first = bad.min(); prev = m[m < first]; out[k] = frac[prev].max() if len(prev) else 0.0
        return out

    def plan(self, starts, goals, max_iter=600, shortcut_rounds=60):
        t0 = time.time(); E = len(starts); self.start = np.asarray(starts, float); self.goal = np.asarray(goals, float)
        pen = lambda X: self.cs.penetration(np.asarray(X, np.float32))
        self.thr = np.maximum(self.cs.threshold, np.maximum(pen(self.start), pen(self.goal)) + 0.5e-3)   # contact the start or goal already has is allowed
        paths = [None] * E; info = dict(straight=np.zeros(E, bool), iterations=np.zeros(E, int), solved=np.zeros(E, bool))
        f = self._valid_prefix([(i, self.start[i], self.goal[i]) for i in range(E)])
        for i in range(E):
            if f[i] >= 1.0: paths[i] = np.stack([self.start[i], self.goal[i]]); info["straight"][i] = True; info["solved"][i] = True
        # trees: nodes (list of arrays) and parent indices; tree 0 grows from the start, tree 1 from the goal
        T = {i: [[[self.start[i]], [-1]], [[self.goal[i]], [-1]]] for i in range(E) if paths[i] is None}
        side = {i: 0 for i in T}
        for it in range(max_iter):
            act = [i for i in T if paths[i] is None]
            if not act: break
            new = {}
            segs = []
            for i in act:
                A = T[i][side[i]]; other = self.goal[i] if side[i] == 0 else self.start[i]
                q = other if self.rng.random() < self.bias else self.rng.uniform(self.low, self.high)
                nodes = np.asarray(A[0]); j = int(np.argmin(np.linalg.norm(nodes - q, axis=1))); d = q - nodes[j]; n = np.linalg.norm(d)
                qn = nodes[j] + d * min(1.0, self.step / max(n, 1e-9)); new[i] = (j, qn); segs.append((i, nodes[j], qn))
            fe = self._valid_prefix(segs)
            conn = []
            for (i, p0, qn), fk in zip(segs, fe):
                if fk < 1.0: continue
                A = T[i][side[i]]; A[0].append(qn); A[1].append(new[i][0])
                B = T[i][1 - side[i]]; nb = np.asarray(B[0]); jb = int(np.argmin(np.linalg.norm(nb - qn, axis=1))); conn.append((i, nb[jb], qn, jb))
            fc = self._valid_prefix([(i, p0, p1) for i, p0, p1, _ in conn]) if conn else []
            for (i, p0, p1, jb), fk in zip(conn, fc):
                B = T[i][1 - side[i]]; L = np.linalg.norm(p1 - p0); reach = fk * L; par = jb
                for s in np.arange(self.step, reach + 1e-9, self.step):   # chain of nodes along the valid prefix
                    B[0].append(p0 + (p1 - p0) * s / max(L, 1e-9)); B[1].append(par); par = len(B[0]) - 1
                if fk >= 1.0:
                    if np.linalg.norm(B[0][par] - p1) > 1e-6: B[0].append(p1.copy()); B[1].append(par); par = len(B[0]) - 1
                    A = T[i][side[i]]
                    def trace(tr, k):
                        out = []
                        while k >= 0: out.append(tr[0][k]); k = tr[1][k]
                        return out
                    pa = trace(A, len(A[0]) - 1); pb = trace(B, par)   # each from the joint node back to its root
                    path = (pa[::-1] + pb[1:]) if side[i] == 0 else (pb[::-1] + pa[1:])
                    paths[i] = np.stack(path); info["solved"][i] = True; info["iterations"][i] = it + 1
            for i in act: side[i] = 1 - side[i]
        # shortcut smoothing
        for _ in range(shortcut_rounds):
            segs, idx = [], []
            for i, p in enumerate(paths):
                if p is None or len(p) <= 2: continue
                a, b = sorted(self.rng.choice(len(p), 2, replace=False))
                if b - a < 2: continue
                segs.append((i, p[a], p[b])); idx.append((i, a, b))
            if not segs: break
            fs = self._valid_prefix(segs)
            for (i, a, b), fk in zip(idx, fs):
                if fk >= 1.0: paths[i] = np.concatenate([paths[i][:a + 1], paths[i][b:]])
        info["length_deg"] = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() if p is not None else np.nan for p in paths])
        info["waypoints"] = np.array([len(p) if p is not None else 0 for p in paths]); info["seconds"] = time.time() - t0
        return paths, info


class PathFollower:
    """Reference on a waypoint path: the point `lookahead` deg (path length) ahead of the projection of the current configuration,
    never moving backward along the path; the final point is the goal."""

    def __init__(self, paths, goals, lookahead=20.0):
        self.paths = [np.asarray(p if p is not None else np.stack([g, g]), float) for p, g in zip(paths, goals)]
        self.cum = [np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]) for p in self.paths]
        self.s = np.zeros(len(self.paths)); self.look = lookahead

    def _point(self, i, s):
        c = self.cum[i]; p = self.paths[i]; s = min(max(s, 0.0), c[-1]); k = int(np.searchsorted(c, s, side="right") - 1); k = min(k, len(p) - 2)
        seg = c[k + 1] - c[k]; w = 0.0 if seg <= 0 else (s - c[k]) / seg; return p[k] + w * (p[k + 1] - p[k])

    def _project(self, i, x):
        p = self.paths[i]; c = self.cum[i]; best, bs = np.inf, 0.0
        for k in range(len(p) - 1):
            d = p[k + 1] - p[k]; L2 = float(d @ d); w = 0.0 if L2 <= 0 else float(np.clip((x - p[k]) @ d / L2, 0, 1)); e = np.linalg.norm(p[k] + w * d - x)
            if e < best: best, bs = e, c[k] + w * (c[k + 1] - c[k])
        return bs

    def reference(self, x):
        """x (E, 20) current configuration (deg) -> (E, 20) reference."""
        out = np.zeros_like(np.asarray(x, float))
        for i in range(len(self.paths)):
            self.s[i] = max(self.s[i], self._project(i, x[i])); out[i] = self._point(i, self.s[i] + self.look)
        return out

    def set_path(self, i, path, goal):
        """Replace problem i's path (replanning); progress restarts on the new path."""
        p = np.asarray(path if path is not None else np.stack([goal, goal]), float); self.paths[i] = p
        self.cum[i] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]); self.s[i] = 0.0

    def progress(self):
        return np.array([s / max(c[-1], 1e-9) for s, c in zip(self.s, self.cum)])
