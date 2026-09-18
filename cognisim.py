#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cognisim.py - reference prototype for the AAAI-27 demonstration

    CogniSim: A Simulator Trainer That Reads Cognitive Load
    and Keeps It Private

Run it with no arguments:

    python3 cognisim.py

It writes every number reported in the paper and the video to CSV files under
./out/ and prints a short summary. One dependency: numpy. Runtime ~30 s on a
laptop CPU. Every stage is seeded, so two runs on the same machine produce
byte-identical CSVs.

WHAT IS REAL AND WHAT IS SIMULATED
----------------------------------
No human data is used or required. The physiological streams come from a
generative sensor model (Stage 1) whose parameter ranges are taken from the
published literature cited in the paper; they are *plausible synthetic
signals*, not recordings. Everything downstream of the sensors - feature
extraction, the load classifier, knowledge tracing, the RL policy, federated
averaging, the DP mechanism and the gradient-inversion attack - is a real
implementation operating on those signals. Numbers this script prints are
therefore properties of the pipeline under a simulator, and are reported as
such. They are not, and are never presented as, results from human trainees.

STAGES
------
  1  Scenario + multimodal sensor simulator (V1-cut, 60 Hz base rate)
  2  Feature extraction and cognitive-load classification (softmax regression)
  3  Bayesian knowledge tracing -> mastery posterior
  4  Tabular Q-learning adaptation policy vs. a fixed-syllabus baseline
  5  Closed-loop runtime: the per-timestep loop of Algorithm 1
  6  Federated learning with two-tier differential privacy (epsilon sweep)
  7  Gradient-inversion attack, DP off vs. on
  8  CBTA metric suite + summary

OUTPUTS (./out/)
----------------
  sensor_trace.csv        one row per 60 Hz sample of the demonstrated run
  adaptation_trace.csv    one row per decision step of the closed loop
  load_classifier.csv     per-class precision/recall/F1 on held-out episodes
  knowledge_tracing.csv   per-opportunity mastery posterior and prediction
  rl_vs_baseline.csv      per-trainee outcomes, adaptive policy vs. baseline
  dp_utility_sweep.csv    global-model accuracy across the epsilon sweep
  inversion_attack.csv    reconstruction quality with DP off and on
  cbta_metrics.csv        the six competency metrics for the demonstrated run
  summary.csv             every headline number in one file

Licence: MIT.  Contact: atik.mahabub@inrs.ca
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy is required:  pip install numpy")

SEED = 20270101
OUTDIR = "out"

# --------------------------------------------------------------------------
# Scenario constants. Aviation values follow the worked case study in the
# paper; physiological ranges follow the sources cited there.
# --------------------------------------------------------------------------
FS = 60.0                  # base sampling rate of the fused stream [Hz]
T_END = 60.0               # simulated seconds per episode
T_PRE = 8.0                # quiet pre-taxi period used as baseline [s]
T_V1 = 18.2                # V1 crossed [s]
T_FAIL = 18.7              # engine 2 compressor stall [s]
PUPIL_BASE = 4.2           # [mm]
PUPIL_PEAK = 6.8           # [mm] at saturation
LFHF_BASE = 1.2            # HRV low/high frequency ratio
LFHF_PEAK = 4.1
THETA_RISE_DB = 1.8        # frontal theta power rise at saturation [dB]
DETECT_BUDGET_S = 0.70     # stimulus -> classified load state
TUNNEL_DWELL = 0.62        # dwell fraction on engine instruments that counts
                           # as attentional tunneling

# Load regimes the classifier must separate.
REGIMES = ["nominal", "high_intrinsic", "high_extraneous"]

# Adaptation actions available to the policy.
ACTIONS = ["maintain", "reduce_difficulty", "increase_difficulty",
           "strip_modality", "diegetic_cue"]


# ==========================================================================
# Stage 1 - scenario and multimodal sensor simulator
# ==========================================================================
@dataclass
class Trainee:
    """A synthetic trainee. `skill` drives both load and performance."""
    tid: int
    skill: float               # [0,1] competency on the V1-cut unit
    capacity: float            # working-memory capacity multiplier
    reactivity: float          # how strongly physiology tracks load
    pupil_offset: float = 0.0  # individual resting pupil diameter [mm]
    lfhf_offset: float = 0.0   # individual resting LF/HF
    dwell_bias: float = 0.0    # individual scan-pattern bias
    site: int = 0

    @staticmethod
    def sample(rng: np.random.Generator, tid: int, site: int = 0,
               skill_mu: float = 0.45) -> "Trainee":
        return Trainee(
            tid=tid,
            skill=float(np.clip(rng.normal(skill_mu, 0.18), 0.02, 0.98)),
            capacity=float(np.clip(rng.normal(1.0, 0.13), 0.6, 1.5)),
            reactivity=float(np.clip(rng.normal(1.0, 0.18), 0.5, 1.6)),
            pupil_offset=float(rng.normal(0.0, 0.55)),
            lfhf_offset=float(rng.normal(0.0, 0.42)),
            dwell_bias=float(rng.normal(0.0, 0.07)),
            site=site,
        )


def load_profile(t: np.ndarray, tr: Trainee, difficulty: float,
                 clutter: float, stripped_at: float | None) -> dict:
    """Ground-truth cognitive load over time (the thing we try to infer).

    Intrinsic load follows task difficulty and the trainee's skill deficit.
    Extraneous load is driven by display clutter and spikes at the failure,
    decaying with the trainee's ability to reorganise attention. Germane load
    is whatever capacity is left over - the quantity training wants to
    maximise.
    """
    deficit = 1.0 - tr.skill
    task = 1.0 / (1.0 + np.exp(-(t - T_PRE) / 0.6))      # rest -> task ramp
    intrinsic = (0.06 + (0.16 + 0.55 * difficulty * (0.35 + 0.65 * deficit)) * task)

    spike = np.zeros_like(t)
    after = t >= T_FAIL
    tau = 2.4 + 3.4 * deficit                       # slower recovery if weak
    spike[after] = np.exp(-(t[after] - T_FAIL) / tau)
    extraneous = 0.06 + (0.04 + 0.34 * clutter) * task + (0.34 + 0.46 * deficit) * spike

    if stripped_at is not None:                     # modality stripping
        m = t >= stripped_at
        extraneous[m] *= 0.46                       # clutter removed
        extraneous[m] += 0.02

    intrinsic = intrinsic / tr.capacity
    extraneous = extraneous / tr.capacity
    germane = np.clip(1.15 - intrinsic - extraneous, 0.0, None)
    return {"intrinsic": intrinsic, "extraneous": extraneous, "germane": germane}


def simulate_episode(rng: np.random.Generator, tr: Trainee,
                     difficulty: float = 0.72, clutter: float = 0.8,
                     stripped_at: float | None = None) -> dict:
    """Generate one episode of fused multimodal sensor data at FS."""
    t = np.arange(0.0, T_END, 1.0 / FS)
    L = load_profile(t, tr, difficulty, clutter, stripped_at)
    total = L["intrinsic"] + L["extraneous"]
    k = tr.reactivity

    pupil = (PUPIL_BASE + tr.pupil_offset + (PUPIL_PEAK - PUPIL_BASE) * k * np.clip(total - 0.35, 0, 1.4)
             + rng.normal(0, 0.11, t.size))
    theta = (THETA_RISE_DB * k * np.clip(total - 0.40, 0, 1.3)
             + rng.normal(0, 0.16, t.size))
    lfhf = (LFHF_BASE + tr.lfhf_offset + (LFHF_PEAK - LFHF_BASE) * k * np.clip(total - 0.45, 0, 1.3)
            + rng.normal(0, 0.14, t.size))

    # Gaze: dwell fraction on the engine instruments and instrument scan rate.
    dwell = np.clip(0.18 + tr.dwell_bias + 0.72 * np.clip(L["extraneous"] - 0.22, 0, 1.0)
                    + rng.normal(0, 0.035, t.size), 0, 1)
    scan = np.clip(2.6 - 2.0 * dwell + rng.normal(0, 0.16, t.size), 0.05, None)

    # Ground-truth regime label per sample.
    y = np.zeros(t.size, dtype=int)
    y[L["intrinsic"] > 0.55] = 1
    y[L["extraneous"] > 0.50] = 2                   # extraneous dominates
    return {"t": t, "pupil": pupil, "theta": theta, "lfhf": lfhf,
            "dwell": dwell, "scan": scan, "y": y, "load": L, "trainee": tr}


# ==========================================================================
# Stage 2 - feature extraction and cognitive-load classification
# ==========================================================================
WIN = int(0.6 * FS)         # 600 ms analysis window
HOP = int(0.2 * FS)         # 200 ms hop -> decision every 200 ms
BASE_S = T_PRE - 1.0        # quiet period used to calibrate each trainee
FEATURES = ["pupil_dev", "pupil_d", "theta_dev", "theta_d",
            "lfhf_dev", "dwell_dev", "scan_dev", "scan_d"]


def window_features(ep: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a 600 ms window over an episode -> (X, y, window end time).

    Absolute pupil diameter and LF/HF differ more between people than they do
    between load states, so each channel is expressed as a deviation from that
    trainee's own resting baseline, measured over the first BASE_S seconds of
    the session. Without this step the classifier spends its capacity learning
    who the trainee is rather than what state they are in.
    """
    nb = int(BASE_S * FS)
    base = {k: float(ep[k][:nb].mean()) for k in ("pupil", "theta", "lfhf", "dwell", "scan")}
    X, Y, T = [], [], []
    for a in range(0, ep["t"].size - WIN, HOP):
        b = a + WIN
        p, th, lf = ep["pupil"][a:b], ep["theta"][a:b], ep["lfhf"][a:b]
        dw, sc = ep["dwell"][a:b], ep["scan"][a:b]
        X.append([p.mean() - base["pupil"], p[-1] - p[0],
                  th.mean() - base["theta"], th[-1] - th[0],
                  lf.mean() - base["lfhf"], dw.mean() - base["dwell"],
                  sc.mean() - base["scan"], sc[-1] - sc[0]])
        Y.append(np.bincount(ep["y"][a:b], minlength=3).argmax())
        T.append(ep["t"][b - 1])
    return np.asarray(X), np.asarray(Y), np.asarray(T)


class SoftmaxClassifier:
    """Multinomial logistic regression - small enough to invert by hand,
    which is exactly why it is used for the gradient-leakage demo."""

    def __init__(self, n_feat: int, n_cls: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_feat, n_cls))
        self.b = np.zeros(n_cls)
        self.mu = np.zeros(n_feat)
        self.sd = np.ones(n_feat)

    def fit_scaler(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def _z(self, X):
        return (X - self.mu) / self.sd

    def probs(self, X):
        z = self._z(X) @ self.W + self.b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def grads(self, X, y, w=None):
        """Gradients of (optionally class-weighted) mean cross-entropy."""
        n = X.shape[0]
        P = self.probs(X)
        P[np.arange(n), y] -= 1.0
        if w is not None:
            P *= w[y][:, None]
        Z = self._z(X)
        return Z.T @ P / n, P.sum(0) / n

    def step(self, gW, gb, lr):
        self.W -= lr * gW
        self.b -= lr * gb

    def fit(self, X, y, epochs=220, lr=0.55, batch=256, seed=0, balanced=True):
        self.fit_scaler(X)
        rng = np.random.default_rng(seed)
        n, k = X.shape[0], self.W.shape[1]
        w = None
        if balanced:
            cnt = np.bincount(y, minlength=k).astype(float)
            w = (n / (k * np.maximum(cnt, 1.0)))
        for _ in range(epochs):
            idx = rng.permutation(n)
            for a in range(0, n, batch):
                j = idx[a:a + batch]
                gW, gb = self.grads(X[j], y[j], w)
                self.step(gW, gb, lr)
        return self

    def predict(self, X):
        return self.probs(X).argmax(1)


def prf(y_true, y_pred, n_cls=3):
    """Per-class precision / recall / F1 plus accuracy and macro-F1."""
    rows, f1s = [], []
    for c in range(n_cls):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((REGIMES[c], tp, fp, fn, p, r, f))
        f1s.append(f)
    return rows, float((y_true == y_pred).mean()), float(np.mean(f1s))


# ==========================================================================
# Stage 3 - Bayesian knowledge tracing
# ==========================================================================
@dataclass
class BKT:
    """Two-state knowledge tracing with slip/guess, one per competency.

    `p_learn` is not a constant here: the runtime sets it from the germane
    load estimate, so the learning rate rises and falls with the capacity
    actually left for schema construction. That is Cognitive Load Theory
    written as an update rule.
    """
    p_init: float = 0.25
    p_learn: float = 0.09
    p_slip: float = 0.10
    p_guess: float = 0.22
    p: float = field(init=False)

    def __post_init__(self):
        self.p = self.p_init

    def predict(self) -> float:
        """P(correct on the next opportunity)."""
        return self.p * (1 - self.p_slip) + (1 - self.p) * self.p_guess

    def update(self, correct: bool) -> float:
        if correct:
            num = self.p * (1 - self.p_slip)
            den = num + (1 - self.p) * self.p_guess
        else:
            num = self.p * self.p_slip
            den = num + (1 - self.p) * (1 - self.p_guess)
        post = num / max(den, 1e-12)
        self.p = post + (1 - post) * self.p_learn
        return self.p


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (ties averaged)."""
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ==========================================================================
# Stage 4 - adaptation policy (tabular Q-learning) vs. fixed syllabus
# ==========================================================================
N_MASTERY_BINS, N_LOAD_BINS = 5, 3


def discretise(mastery: float, load_regime: int) -> int:
    m = min(int(mastery * N_MASTERY_BINS), N_MASTERY_BINS - 1)
    return m * N_LOAD_BINS + load_regime


class TrainingEnv:
    """One maneuver = one step. The policy picks how the next maneuver is
    staged; the trainee's competency and the load they experience follow.

    Reward trades three things off against each other: learning gained,
    critical errors committed, and wall-clock training time spent.
    """

    def __init__(self, rng, trainee: Trainee, max_steps: int = 40):
        self.rng, self.tr, self.max_steps = rng, trainee, max_steps

    def reset(self, scripted: bool = False):
        self.scripted = scripted
        self.skill = self.tr.skill
        self.bkt = BKT()
        self.difficulty = 0.55
        self.clutter = 0.85
        self.stripped = False
        self.step_i = 0
        self.errors = 0
        self.minutes = 0.0
        return self._observe(stripped=False)

    def _observe(self, stripped: bool | None = None):
        if stripped is None:
            stripped = getattr(self, "stripped", False)
        deficit = 1.0 - self.skill
        intrinsic = (0.22 + 0.55 * self.difficulty * (0.35 + 0.65 * deficit)) / self.tr.capacity
        extraneous = (0.10 + 0.34 * self.clutter + 0.30 * deficit) / self.tr.capacity
        if stripped:
            extraneous *= 0.46
        regime = 0
        if intrinsic > 0.62:
            regime = 1
        if extraneous > 0.55:
            regime = 2
        self.last = (intrinsic, extraneous, regime)
        return discretise(self.bkt.p, regime)

    def step(self, action: int):
        if self.scripted:
            # Hours-based syllabus: difficulty ramps on the clock, identical
            # for every trainee, and the display is never decluttered.
            self.difficulty = float(np.clip(0.45 + 0.055 * self.step_i, 0.45, 1.0))
            self.stripped = False
            action = 0
        name = ACTIONS[action]
        cue = name == "diegetic_cue"
        if name == "strip_modality":
            self.stripped = True          # the decluttered display persists
        stripped = self.stripped
        if name == "reduce_difficulty":
            self.difficulty = max(0.25, self.difficulty - 0.12)
        elif name == "increase_difficulty":
            self.difficulty = min(1.0, self.difficulty + 0.12)
            self.stripped = False         # full display restored as load drops

        intrinsic, extraneous, regime = self.last
        if stripped:
            extraneous *= 0.46

        # Overload suppresses both performance and learning; a well matched
        # difficulty (just above current mastery) maximises learning.
        overload = max(0.0, intrinsic + extraneous - 0.95)
        match = 1.0 - abs(self.difficulty - (0.35 + 0.55 * self.skill))
        gain = 0.055 * max(0.0, match) * (1.0 - 0.85 * overload)
        if cue:
            gain *= 1.22                       # in-world coaching helps
        self.skill = float(np.clip(self.skill + gain, 0, 1))
        germane = max(0.0, 1.15 - intrinsic - extraneous)
        self.bkt.p_learn = float(np.clip(0.02 + 0.20 * germane, 0.01, 0.30))

        p_correct = float(np.clip(0.12 + 0.86 * self.skill - 0.55 * overload
                                  - 0.25 * max(0.0, self.difficulty - self.skill), 0.02, 0.985))
        correct = self.rng.random() < p_correct
        self.bkt.update(correct)
        if not correct and (overload > 0.06 or self.rng.random() < 0.28):
            self.errors += 1

        self.minutes += 6.5 + (1.6 if cue else 0.0) + (0.8 if name == "strip_modality" else 0.0)
        self.step_i += 1
        reward = 26.0 * gain - 1.25 * (0 if correct else 1) - 0.22 * (self.minutes / 6.5)
        done = self.bkt.p >= 0.92 or self.step_i >= self.max_steps
        return self._observe(stripped), reward, done, {"correct": correct,
                                                       "overload": overload,
                                                       "regime": regime}


def train_policy(seed: int, n_episodes: int = 2600):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_MASTERY_BINS * N_LOAD_BINS, len(ACTIONS)))
    alpha, gamma = 0.22, 0.93
    for ep in range(n_episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * n_episodes))
        env = TrainingEnv(rng, Trainee.sample(rng, ep))
        s = env.reset()
        done = False
        while not done:
            a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * (0 if done else Q[s2].max()) - Q[s, a])
            s = s2
    return Q


def evaluate(Q, seed: int, n_trainees: int = 220):
    """Adaptive policy vs. a fixed syllabus, same trainees, same seeds."""
    rows = []
    for i in range(n_trainees):
        tr = Trainee.sample(np.random.default_rng(seed + i), i)
        out = {}
        for mode in ("baseline", "adaptive"):
            env = TrainingEnv(np.random.default_rng(seed * 7 + i), tr)
            s = env.reset(scripted=(mode == "baseline"))
            done = False
            while not done:
                a = 0 if mode == "baseline" else int(Q[s].argmax())
                s, _, done, _ = env.step(a)
            out[mode] = (env.step_i, env.minutes, env.errors, env.bkt.p, env.skill)
        rows.append((i, tr.skill, *out["baseline"], *out["adaptive"]))
    return rows


# ==========================================================================
# Stage 5 - the closed loop of Algorithm 1, on one demonstrated run
# ==========================================================================
def closed_loop(rng, clf: SoftmaxClassifier, tr: Trainee, seed: int = 0,
                _stripped_at: float | None = None):
    """Run the demonstrated V1-cut with the loop live, and log every step.

    Two passes: the first decides when modality stripping fires, the second
    re-renders the same episode (same trainee, same seed) with that display
    state in effect, so the logged sensor trace and the CBTA metrics describe
    a session in which the intervention actually happened.
    """
    ep = simulate_episode(np.random.default_rng(seed), tr, difficulty=0.78,
                          clutter=0.85, stripped_at=_stripped_at)
    X, y, tw = window_features(ep)
    P = clf.probs(X)
    pred = P.argmax(1)

    bkt = BKT(p_init=float(np.clip(tr.skill, 0.05, 0.95)))
    trace, stripped_at, cue_at, detect_at = [], None, None, None
    hot = 0
    for i, t in enumerate(tw):
        regime, conf = int(pred[i]), float(P[i].max())
        correct = rng.random() < (0.35 + 0.6 * tr.skill - 0.3 * (regime == 2))
        bkt.update(correct)
        action = "maintain"
        if regime == 2 and conf > 0.75:
            hot += 1
            if hot >= 2 and stripped_at is None:       # sustained ~400 ms
                stripped_at, action = t, "strip_modality"
                detect_at = t - T_FAIL
            elif stripped_at is not None and cue_at is None:
                cue_at, action = t, "diegetic_cue"
        else:
            hot = 0
            if regime == 1 and bkt.p < 0.5:
                action = "reduce_difficulty"
        trace.append(dict(t=round(float(t), 3), regime=REGIMES[regime],
                          confidence=round(conf, 4),
                          mastery=round(bkt.p, 4), action=action,
                          d_pupil_mm=round(float(X[i, 0]), 3),
                          d_theta_db=round(float(X[i, 2]), 3),
                          d_lfhf=round(float(X[i, 4]), 3),
                          d_dwell=round(float(X[i, 5]), 3)))
    if _stripped_at is None and stripped_at is not None:
        # second pass, now with the decluttered display in effect
        return closed_loop(rng, clf, tr, seed=seed, _stripped_at=stripped_at)
    return ep, trace, stripped_at, cue_at, detect_at


def cbta_metrics(ep, trace, stripped_at, cue_at) -> dict:
    """The six metrics the paper writes to the competency ledger."""
    t = ep["t"]
    dwell, scan = ep["dwell"], ep["scan"]
    post = t >= T_FAIL
    gaze_latency = (stripped_at - T_FAIL) if stripped_at else float("nan")
    if stripped_at:
        hold = int(0.5 * FS)                       # dwell must stay down 500 ms
        ok = (dwell < TUNNEL_DWELL).astype(int)
        run = np.convolve(ok, np.ones(hold, dtype=int), mode="valid")
        # measured from the failure event to sustained recovery of the scan
        idx = np.where((t[:run.size] >= T_FAIL) & (run == hold))[0]
        delta_corr = float(t[idx[0]] - T_FAIL) if idx.size else float("nan")
    else:
        delta_corr = float("nan")
    stability = float(1.0 / (1.0 + np.std(ep["lfhf"][post])))
    optimal_scan = 2.2
    gather = float(np.clip(1.0 - np.mean(np.abs(scan[post] - optimal_scan)) / optimal_scan, 0, 1))
    vocal = float(np.clip(0.12 + 0.55 * np.mean(ep["load"]["extraneous"][post]), 0, 1))
    adhere = float(np.mean([r["action"] != "maintain" or r["regime"] != "high_extraneous"
                            for r in trace]))
    return {"gaze_to_action_latency_s": round(gaze_latency, 3),
            "delta_of_correction_s": round(delta_corr, 3),
            "cognitive_stability_index": round(stability, 4),
            "information_gathering_efficiency": round(gather, 4),
            "vocal_stress_variation_index": round(vocal, 4),
            "procedure_adherence": round(adhere, 4)}


# ==========================================================================
# Stage 6 - federated learning with two-tier differential privacy
# ==========================================================================
def gaussian_sigma(clip: float, eps: float, delta: float = 1e-5) -> float:
    """Gaussian mechanism noise scale for L2 sensitivity `clip`."""
    if not math.isfinite(eps):
        return 0.0
    return clip * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def pca_fit(X: np.ndarray, var_keep: float = 0.95):
    """PCA on standardised features, keeping `var_keep` of the variance.

    The paper applies this before the DP mechanism: fewer parameters means
    less injected noise for the same epsilon, which is where most of the
    utility at a tight privacy budget comes from.
    """
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    k = int(np.searchsorted(np.cumsum(w) / w.sum(), var_keep) + 1)
    return {"mu": mu, "sd": sd, "V": V[:, :k], "k": k,
            "explained": float((np.cumsum(w) / w.sum())[k - 1])}


def pca_apply(P, X):
    return ((X - P["mu"]) / P["sd"]) @ P["V"]


def federated_train(sites, Xte, yte, eps_client, eps_central, seed,
                    rounds=16, local_steps=12, lr=0.5, delta=1e-5,
                    clip_percentile=95.0, pca=None):
    """FedAvg. Each client clips its update, adds client-side DP noise; the
    aggregator averages and adds a second, central noise tier."""
    rng = np.random.default_rng(seed)
    if pca is not None:
        sites = [(pca_apply(pca, Xs), ys) for Xs, ys in sites]
        Xte = pca_apply(pca, Xte)
    n_feat = sites[0][0].shape[1]
    glob = SoftmaxClassifier(n_feat, len(REGIMES), seed=seed)
    allX = np.vstack([s[0] for s in sites])
    glob.fit_scaler(allX)

    n_par = glob.W.size + glob.b.size
    for _ in range(rounds):
        deltas, norms = [], []
        for Xs, ys in sites:
            loc = SoftmaxClassifier(n_feat, len(REGIMES), seed=seed)
            loc.W, loc.b = glob.W.copy(), glob.b.copy()
            loc.mu, loc.sd = glob.mu, glob.sd
            for _ in range(local_steps):
                j = rng.integers(0, Xs.shape[0], min(128, Xs.shape[0]))
                gW, gb = loc.grads(Xs[j], ys[j])
                loc.step(gW, gb, lr)
            d = np.concatenate([(loc.W - glob.W).ravel(), loc.b - glob.b])
            deltas.append(d)
            norms.append(np.linalg.norm(d))
        clip = float(np.percentile(norms, clip_percentile)) + 1e-12
        sig_c = gaussian_sigma(clip, eps_client, delta)
        upd = []
        for d, nr in zip(deltas, norms):
            d = d * min(1.0, clip / (nr + 1e-12))                 # clip
            if sig_c > 0:
                d = d + rng.normal(0, sig_c / math.sqrt(n_par), n_par)
            upd.append(d)
        agg = np.mean(upd, axis=0)
        sig_s = gaussian_sigma(clip / len(sites), eps_central, delta)
        if sig_s > 0:
            agg = agg + rng.normal(0, sig_s / math.sqrt(n_par), n_par)
        glob.W += agg[:glob.W.size].reshape(glob.W.shape)
        glob.b += agg[glob.W.size:]
    acc = float((glob.predict(Xte) == yte).mean())
    return glob, acc


# ==========================================================================
# Stage 7 - gradient-inversion attack
# ==========================================================================
def inversion_attack(clf: SoftmaxClassifier, x: np.ndarray, y: int,
                     eps: float | None, rng, clip: float = 1.0,
                     delta: float = 1e-5):
    """For a softmax layer the per-example gradient is dL/dW = z (p - onehot)^T,
    so every column of dL/dW is a scalar multiple of the standardised input z.
    Recovering z (and therefore the trainee's features) needs no optimisation
    at all - which is precisely the leak differential privacy has to close.
    """
    gW, _ = clf.grads(x[None, :], np.array([y]))
    if eps is not None:
        nr = np.linalg.norm(gW)
        gW = gW * min(1.0, clip / (nr + 1e-12))
        gW = gW + rng.normal(0, gaussian_sigma(clip, eps, delta) / math.sqrt(gW.size), gW.shape)
    col = int(np.argmax(np.abs(gW).sum(0)))
    z_hat = gW[:, col]
    z_true = (x - clf.mu) / clf.sd
    denom = (np.linalg.norm(z_hat) * np.linalg.norm(z_true)) + 1e-12
    cos = float(abs(z_hat @ z_true) / denom)
    scale = (z_true @ z_hat) / (z_hat @ z_hat + 1e-12)
    x_hat = clf.mu + clf.sd * (scale * z_hat)
    nrmse = float(np.linalg.norm(x_hat - x) / (np.linalg.norm(x) + 1e-12))
    return cos, nrmse, x_hat


# ==========================================================================
# CSV helpers
# ==========================================================================
def write_csv(name: str, header, rows):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    global OUTDIR
    ap = argparse.ArgumentParser(
        description="CogniSim reference prototype (AAAI-27 demonstration).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--episodes", type=int, default=90,
                    help="episodes used to train/test the load classifier")
    ap.add_argument("--trainees", type=int, default=220,
                    help="trainees in the policy evaluation")
    ap.add_argument("--sites", type=int, default=12,
                    help="federated sites")
    ap.add_argument("--quick", action="store_true",
                    help="smaller run for a smoke test (~5 s)")
    args = ap.parse_args(argv)

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    if args.quick:
        args.episodes, args.trainees, args.sites = 24, 40, 6
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    S = {}                                   # summary values

    # ---------------------------------------------------------- stage 1 + 2
    print("[1/7] simulating episodes and extracting features ...")
    Xtr, ytr, Xte, yte = [], [], [], []
    n_test = max(6, args.episodes // 4)
    for i in range(args.episodes):
        tr = Trainee.sample(rng, i)
        strip = None if i % 3 else float(rng.uniform(19.2, 21.0))
        ep = simulate_episode(rng, tr, difficulty=float(rng.uniform(0.45, 0.95)),
                              clutter=float(rng.uniform(0.5, 1.0)), stripped_at=strip)
        X, y, _ = window_features(ep)
        (Xte if i >= args.episodes - n_test else Xtr).append(X)
        (yte if i >= args.episodes - n_test else ytr).append(y)
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    Xte, yte = np.vstack(Xte), np.concatenate(yte)

    print("[2/7] training the cognitive-load classifier ...")
    clf = SoftmaxClassifier(Xtr.shape[1], len(REGIMES), seed=args.seed).fit(
        Xtr, ytr, seed=args.seed)
    rows, acc, mf1 = prf(yte, clf.predict(Xte))
    write_csv("load_classifier.csv",
              ["regime", "tp", "fp", "fn", "precision", "recall", "f1"],
              [[r[0], r[1], r[2], r[3], round(r[4], 4), round(r[5], 4), round(r[6], 4)]
               for r in rows])
    S["load_accuracy"] = round(acc, 4)
    S["load_macro_f1"] = round(mf1, 4)
    S["load_train_windows"] = int(Xtr.shape[0])
    S["load_test_windows"] = int(Xte.shape[0])
    S["decision_period_ms"] = int(1000 * HOP / FS)
    S["window_ms"] = int(1000 * WIN / FS)

    # ------------------------------------------------------------- stage 3
    print("[3/7] knowledge tracing ...")
    kt_rows, y_true, p_bkt, p_static = [], [], [], []
    for i in range(80 if not args.quick else 20):
        tr = Trainee.sample(rng, 10_000 + i)
        bkt = BKT()
        for op in range(30):
            p = bkt.predict()
            correct = rng.random() < np.clip(0.1 + 0.85 * min(1.0, tr.skill + 0.02 * op), 0, 0.97)
            y_true.append(int(correct)); p_bkt.append(p); p_static.append(0.5)
            kt_rows.append([i, op, round(p, 4), int(correct), round(bkt.update(correct), 4)])
    write_csv("knowledge_tracing.csv",
              ["trainee", "opportunity", "p_correct_pred", "observed_correct", "mastery_post"],
              kt_rows)
    S["kt_auc_bkt"] = round(auc(np.array(y_true), np.array(p_bkt)), 4)
    S["kt_auc_static"] = round(auc(np.array(y_true), np.array(p_static)), 4)

    # ------------------------------------------------------------- stage 4
    print("[4/7] training the adaptation policy (Q-learning) ...")
    Q = train_policy(args.seed, n_episodes=600 if args.quick else 2600)
    ev = evaluate(Q, args.seed, n_trainees=args.trainees)
    write_csv("rl_vs_baseline.csv",
              ["trainee", "initial_skill",
               "base_maneuvers", "base_minutes", "base_errors", "base_mastery", "base_skill",
               "adapt_maneuvers", "adapt_minutes", "adapt_errors", "adapt_mastery", "adapt_skill"],
              [[r[0], round(r[1], 4), r[2], round(r[3], 1), r[4], round(r[5], 4), round(r[6], 4),
                r[7], round(r[8], 1), r[9], round(r[10], 4), round(r[11], 4)] for r in ev])
    a = np.array([[r[3], r[4], r[6], r[8], r[9], r[11]] for r in ev], dtype=float)
    base_min, base_err, base_skill = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    ad_min, ad_err, ad_skill = a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean()
    S["time_to_competency_baseline_min"] = round(base_min, 2)
    S["time_to_competency_adaptive_min"] = round(ad_min, 2)
    S["time_to_competency_change_pct"] = round(100 * (ad_min - base_min) / base_min, 2)
    S["critical_errors_baseline"] = round(base_err, 3)
    S["critical_errors_adaptive"] = round(ad_err, 3)
    S["critical_errors_change_pct"] = round(100 * (ad_err - base_err) / max(base_err, 1e-9), 2)
    S["final_skill_baseline"] = round(base_skill, 4)
    S["final_skill_adaptive"] = round(ad_skill, 4)

    # ------------------------------------------------------------- stage 5
    print("[5/7] closed-loop run of the demonstrated V1-cut ...")
    demo = Trainee.sample(np.random.default_rng(args.seed + 777), 0, skill_mu=0.42)
    ep, trace, strip_t, cue_t, detect = closed_loop(
        np.random.default_rng(args.seed + 99), clf, demo, seed=args.seed + 99)
    step = max(1, int(FS // 20))                     # log at 20 Hz
    write_csv("sensor_trace.csv",
              ["t_s", "pupil_mm", "theta_db", "hrv_lf_hf", "dwell_frac",
               "scan_hz", "intrinsic", "extraneous", "germane", "regime"],
              [[round(float(ep["t"][i]), 3), round(float(ep["pupil"][i]), 4),
                round(float(ep["theta"][i]), 4), round(float(ep["lfhf"][i]), 4),
                round(float(ep["dwell"][i]), 4), round(float(ep["scan"][i]), 4),
                round(float(ep["load"]["intrinsic"][i]), 4),
                round(float(ep["load"]["extraneous"][i]), 4),
                round(float(ep["load"]["germane"][i]), 4),
                REGIMES[int(ep["y"][i])]] for i in range(0, ep["t"].size, step)])
    write_csv("adaptation_trace.csv", list(trace[0].keys()),
              [list(r.values()) for r in trace])
    S["detection_latency_s"] = round(detect, 3) if detect else float("nan")
    S["detection_within_budget"] = bool(detect is not None and detect <= DETECT_BUDGET_S)
    S["modality_stripped_at_s"] = round(strip_t, 3) if strip_t else float("nan")
    S["diegetic_cue_at_s"] = round(cue_t, 3) if cue_t else float("nan")

    m = cbta_metrics(ep, trace, strip_t, cue_t)
    write_csv("cbta_metrics.csv", ["metric", "value"], list(m.items()))
    S.update(m)

    # ------------------------------------------------------------- stage 6
    print("[6/7] federated learning with two-tier differential privacy ...")
    per = max(1, Xtr.shape[0] // args.sites)
    sites = [(Xtr[i * per:(i + 1) * per], ytr[i * per:(i + 1) * per])
             for i in range(args.sites)]
    pca = pca_fit(Xtr, 0.95)
    S["pca_components"] = pca["k"]
    S["pca_explained_variance"] = round(pca["explained"], 4)
    sweep, rounds = [], 10 if args.quick else 16
    for ec, es, label in [(float("inf"), float("inf"), "no DP"),
                          (8.0, 4.0, "eps_total=12"),
                          (4.0, 2.0, "eps_total=6"),
                          (2.0, 1.0, "eps_total=3  (paper)"),
                          (1.0, 0.5, "eps_total=1.5"),
                          (0.5, 0.25, "eps_total=0.75")]:
        _, acc_fl = federated_train(sites, Xte, yte, ec, es, args.seed,
                                    rounds=rounds, pca=pca)
        tot = "inf" if math.isinf(ec) else f"{ec + es:.2f}"
        sweep.append([label, "inf" if math.isinf(ec) else ec,
                      "inf" if math.isinf(es) else es, tot, round(acc_fl, 4)])
        print(f"        {label:22s} accuracy {acc_fl:.4f}")
    write_csv("dp_utility_sweep.csv",
              ["setting", "epsilon_client", "epsilon_central", "epsilon_total", "test_accuracy"],
              sweep)
    S["fl_accuracy_no_dp"] = sweep[0][4]
    S["fl_accuracy_eps3"] = sweep[3][4]
    S["fl_utility_cost_eps3"] = round(sweep[0][4] - sweep[3][4], 4)
    S["fl_sites"] = args.sites
    S["fl_rounds"] = rounds

    # ------------------------------------------------------------- stage 7
    print("[7/7] gradient-inversion attack, DP off vs. on ...")
    att_rng = np.random.default_rng(args.seed + 5)
    rows_att = []
    for trial in range(40 if not args.quick else 10):
        k = int(att_rng.integers(0, Xte.shape[0]))
        for eps, tag in [(None, "DP off"), (2.0, "DP on (client eps=2)"),
                         (1.0, "DP on (client eps=1)")]:
            cos, nrmse, _ = inversion_attack(clf, Xte[k], int(yte[k]), eps, att_rng)
            rows_att.append([trial, tag, round(cos, 4), round(nrmse, 4)])
    write_csv("inversion_attack.csv",
              ["trial", "setting", "cosine_similarity", "nrmse"], rows_att)
    arr = np.array([[r[2], r[3]] for r in rows_att], dtype=float).reshape(-1, 3, 2)
    S["inversion_cosine_dp_off"] = round(float(arr[:, 0, 0].mean()), 4)
    S["inversion_cosine_dp_eps2"] = round(float(arr[:, 1, 0].mean()), 4)
    S["inversion_cosine_dp_eps1"] = round(float(arr[:, 2, 0].mean()), 4)

    # ---------------------------------------------------------------- done
    S["seed"] = args.seed
    write_csv("summary.csv", ["key", "value"], [[k, v] for k, v in S.items()])
    S["runtime_s"] = round(time.time() - t0, 1)   # printed only: keeps CSVs identical

    print("\n" + "=" * 68)
    print("CogniSim prototype - summary   (all figures are simulator outputs)")
    print("=" * 68)
    print(f"  load classifier         accuracy {S['load_accuracy']:.3f}   "
          f"macro-F1 {S['load_macro_f1']:.3f}")
    print(f"  knowledge tracing       AUC {S['kt_auc_bkt']:.3f} "
          f"(chance {S['kt_auc_static']:.3f})")
    print(f"  time to competency      {S['time_to_competency_baseline_min']:.1f} -> "
          f"{S['time_to_competency_adaptive_min']:.1f} min "
          f"({S['time_to_competency_change_pct']:+.1f} %)")
    print(f"  critical errors         {S['critical_errors_baseline']:.2f} -> "
          f"{S['critical_errors_adaptive']:.2f} per trainee "
          f"({S['critical_errors_change_pct']:+.1f} %)")
    lat = S["detection_latency_s"]
    print(f"  detection latency       {lat:.2f} s "
          f"({'within' if S['detection_within_budget'] else 'OVER'} the "
          f"{DETECT_BUDGET_S:.2f} s budget)")
    print(f"  federated accuracy      {S['fl_accuracy_no_dp']:.3f} without DP -> "
          f"{S['fl_accuracy_eps3']:.3f} at eps_total=3 "
          f"(cost {S['fl_utility_cost_eps3']:.3f})")
    print(f"  gradient inversion      cos {S['inversion_cosine_dp_off']:.3f} with DP off -> "
          f"{S['inversion_cosine_dp_eps2']:.3f} at client eps=2")
    print(f"  wrote {len(os.listdir(OUTDIR))} CSV files to ./{OUTDIR}/   "
          f"({S['runtime_s']:.1f} s, seed {S['seed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
