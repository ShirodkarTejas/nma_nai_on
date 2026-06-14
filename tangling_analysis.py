#!/usr/bin/env python3
"""
Trajectory Tangling Analysis for NMAP Ablation Study
=====================================================

Scientific question
-------------------
Does the neuromodulatory broadcast (environment_type signal → frequency + amplitude
scaling) physically SEPARATE the locomotor attractors for water/land/transition,
reducing trajectory tangling — or does it only create temporal (speed) separation
while sharing the same spatial attractor?

Experimental design
-------------------
We run all 5 ablation checkpoints under three viscosity conditions:
    Water      : env_type = [1.0, 0.0, 0.10]  (fast, high-amp)
    Transition : env_type = [0.5, 0.5, 0.40]
    Land       : env_type = [0.0, 1.0, 0.80]  (slow, low-amp)

Hidden state h(t) ∈ R^25 per step:
    osc_d, osc_v  (2)                  ← relaxation oscillator output
    bneuron_d_0..4, bneuron_v_0..4     ← B-neuron activations (10)
    muscle_d_0..4, muscle_v_0..4       ← muscle activations (10)
    env_mod, freq_scale, amp_scale      ← neuromodulatory state (3)

Three complementary metrics
---------------------------
1. PCA 3D trajectory plots — visual attractor geometry
2. Cross-condition centroid distance matrix — spatial attractor separation
3. Trajectory tangling Q(τ) — Russo et al. 2018 (Nature Neuroscience)
      Q(τ) = max_{τ'≠τ, |τ-τ'|>Δ} ||ẋ(τ) - ẋ(τ')||² / (||x(τ)-x(τ')||² + ε)
   Applied WITHIN each condition to measure self-tangling (attractor quality).
   Applied ACROSS conditions (merged trajectory) to measure cross-attractor tangling.
   LOW self-Q = smooth cyclical attractor.  LOW cross-Q = well-separated attractors.

Interpretation key
------------------
If neuromodulation creates TRUE spatial separation:
  * Cross-condition Q should be LOW  (large denominator from distant states)
  * Self-Q should be LOW             (clean oscillator cycles)
  * Centroid distance W–L should be HIGH

If neuromodulation only creates temporal (speed) separation:
  * Cross-condition Q will be HIGH   (trajectories cross at different velocities)
  * Centroid distance modest
  * Self-Q patterns differ across conditions
"""

try:
    import gym_bridge  # noqa: F401
except ModuleNotFoundError:
    pass

import os
import sys
import warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCAP = os.path.dirname(_HERE)
_TONIC = os.path.join(_HERE, "tonic")
for _p in (_HERE, _NCAP, _TONIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from swimmer.models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ABLATIONS = {
    "01\nBaseline": "results/01_ncap_baseline/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_6links_oscillator_period60_training_modecurriculum_checkpoint_step_1000000.pt",
    "02\nSparse Init": "results/02_ncap_sparse_init/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_6links_oscillator_period60_training_modecurriculum_checkpoint_step_1000000.pt",
    "03\nReg Only": "results/03_ncap_reg_only/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_6links_oscillator_period60_training_modecurriculum_checkpoint_step_1000000.pt",
    "04\nFull Priors": "results/04_ncap_full_priors/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_6links_oscillator_period60_training_modecurriculum_checkpoint_step_1000000.pt",
    "05\nFull NMAP": "results/05_ncap_full_nmap/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_6links_oscillator_period60_training_modecurriculum_checkpoint_step_1000000.pt",
}

# Long labels for text summaries
ABLATION_LONG = [
    "01-Baseline (shared, no priors)",
    "02-Sparse Init (per-joint)",
    "03-Reg Only (shared+reg)",
    "04-Full Priors (init+reg)",
    "05-Full NMAP (init+reg+neuromod)",
]

CONDITIONS = {
    "Water":      np.array([1.0, 0.0, 0.10], dtype=np.float32),
    "Transition": np.array([0.5, 0.5, 0.40], dtype=np.float32),
    "Land":       np.array([0.0, 1.0, 0.80], dtype=np.float32),
}
COND_COLORS = {"Water": "#2166ac", "Transition": "#74add1", "Land": "#d73027"}
COND_LIST   = list(CONDITIONS.keys())

N_JOINTS = 5
T_STEPS  = 600        # 10 oscillator cycles at base period 60
T_DELTA  = 10         # tangling: min temporal gap
EPS_TNG  = 1e-3       # tangling denominator regulariser
OUT_DIR  = "results/tangling_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Dark theme
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_CLR = "#21262d"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _is_per_joint(sd: dict) -> bool:
    return any("bneuron_d_osc_0" in k for k in sd)


def load_model(ckpt_path: str) -> EnhancedBiologicalNCAPSwimmer:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    model = EnhancedBiologicalNCAPSwimmer(
        n_joints=N_JOINTS,
        oscillator_period=60,
        use_weight_sharing=not _is_per_joint(sd),
        include_environment_adaptation=True,
        include_goal_direction=False,
        locomotion_only_mode=True,
        action_scaling_factor=1.8,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Hidden-state extraction via instrumented forward pass
# ---------------------------------------------------------------------------

def _extract_hidden(
    model: EnhancedBiologicalNCAPSwimmer,
    env_type: np.ndarray,
    T: int = T_STEPS,
) -> np.ndarray:
    """
    Return shape (T, 25) hidden state trajectories.

    Dynamics: simple spring-damper integrator (τ_out → joint_pos) so the
    model drives itself in a closed loop, capturing its intrinsic attractor.
    """
    model.reset()
    joint_pos = np.zeros(N_JOINTS, dtype=np.float32)
    k_damp = 0.92
    k_spring = 0.04

    env_arg = torch.tensor(env_type, dtype=torch.float32)
    _cap = {}
    original_fwd = model.__class__.forward

    def instrumented_forward(self, joint_pos_in, environment_type=None, **kwargs):
        self._constrain_parameters()

        jp = joint_pos_in if isinstance(joint_pos_in, torch.Tensor) else torch.tensor(joint_pos_in, dtype=torch.float32)
        jp = jp.to(self._device)
        ts = torch.tensor([float(self.timestep)], dtype=torch.float32, device=self._device)

        if jp.dim() == 1:
            jp = jp.unsqueeze(0)

        # Environment adaptation
        amp_scale = freq_scale = 1.0
        env_mod = 0.0
        if environment_type is not None and self.include_environment_adaptation:
            try:
                ev = environment_type.detach().cpu().numpy().flatten().tolist() \
                     if isinstance(environment_type, torch.Tensor) else list(environment_type)
                land_flag = float(ev[1])
                visc_norm = float(ev[2]) if len(ev) > 2 else 0.1
                if land_flag > 0.5:
                    freq_scale = self.land_frequency_scale.item()
                    amp_scale  = self.land_amplitude_scale.item() * (1.0 + 0.5 * visc_norm)
                    env_mod    = -0.1
                else:
                    freq_scale = self.water_frequency_scale.item()
                    amp_scale  = self.water_amplitude_scale.item() * (1.0 + 0.5 * visc_norm)
                    env_mod    = 0.1
            except Exception:
                pass

        # Oscillator
        if self.include_head_oscillators:
            osc_d, osc_v = self.relaxation_oscillator(ts.item(), goal_bias=0.0, environment_factor=freq_scale)
            osc_d = osc_d.to(self._device)
            osc_v = osc_v.to(self._device)
        else:
            osc_d = osc_v = torch.tensor(0.0, device=self._device)

        exc = self.exc;  inh = self.inh;  ws = self.ws
        from swimmer.models.biological_ncap import graded

        jl = 2 * np.pi / (self.n_joints + 1)
        jp_norm = torch.clamp(jp / jl, -1, 1)
        jp_d = jp_norm.clamp(min=0, max=1)
        jp_v = jp_norm.clamp(min=-1, max=0).neg()

        bn_d_vals = [];  bn_v_vals = [];  mu_d_vals = [];  mu_v_vals = [];  tq_list = []
        for i in range(self.n_joints):
            bn_d = bn_v = torch.zeros_like(jp_norm[..., 0, None])

            if self.include_proprioception and i > 0:
                ps = 1.0 + env_mod
                bn_d = bn_d + jp_d[..., i-1, None] * exc(self.params[ws(f"bneuron_d_prop_{i}", "bneuron_prop")]) * ps
                bn_v = bn_v + jp_v[..., i-1, None] * exc(self.params[ws(f"bneuron_v_prop_{i}", "bneuron_prop")]) * ps

            if self.include_head_oscillators:
                if i == 0:
                    od, ov, sc = osc_d, osc_v, 1.0
                else:
                    dts = max(0, int(ts.item()) - i * 15)
                    od, ov = self.relaxation_oscillator(dts, goal_bias=0.0, environment_factor=freq_scale)
                    od = od.to(self._device);  ov = ov.to(self._device)
                    sc = 0.8
                ps = (1.0 + env_mod) * sc
                bn_d = bn_d + od * exc(self.params[ws(f"bneuron_d_osc_{i}", "bneuron_osc")]) * ps
                bn_v = bn_v + ov * exc(self.params[ws(f"bneuron_v_osc_{i}", "bneuron_osc")]) * ps

            bn_d = graded(bn_d);  bn_v = graded(bn_v)
            ps = 1.0 + env_mod
            mu_d = graded(bn_d * exc(self.params[ws(f"muscle_d_d_{i}", "muscle_ipsi")]) * ps +
                          bn_v * inh(self.params[ws(f"muscle_d_v_{i}", "muscle_contra")]) * ps)
            mu_v = graded(bn_v * exc(self.params[ws(f"muscle_v_v_{i}", "muscle_ipsi")]) * ps +
                          bn_d * inh(self.params[ws(f"muscle_v_d_{i}", "muscle_contra")]) * ps)

            bn_d_vals.append(bn_d.squeeze().item());  bn_v_vals.append(bn_v.squeeze().item())
            mu_d_vals.append(mu_d.squeeze().item());  mu_v_vals.append(mu_v.squeeze().item())
            tq_list.append(mu_d - mu_v)

        _cap["osc_d"] = osc_d.item();  _cap["osc_v"] = osc_v.item()
        _cap["bn_d"]  = bn_d_vals;     _cap["bn_v"]  = bn_v_vals
        _cap["mu_d"]  = mu_d_vals;     _cap["mu_v"]  = mu_v_vals
        _cap["env_mod"]   = env_mod
        _cap["freq_scale"] = freq_scale
        _cap["amp_scale"]  = amp_scale

        torques = torch.cat(tq_list, -1)
        out = torch.clamp(torques * amp_scale, -1.0, 1.0) * self.action_scaling_factor
        for i in range(2, self.n_joints):
            out[..., i] = out[..., i] * 0.8
        self.timestep += 1
        return out.squeeze(0)

    model.__class__.forward = instrumented_forward
    hidden_list = []
    try:
        with torch.no_grad():
            for _ in range(T):
                jp_t = torch.tensor(joint_pos, dtype=torch.float32)
                tq   = model.forward(jp_t, environment_type=env_arg)
                tq_np = tq.detach().cpu().numpy()
                joint_pos = np.clip(k_damp * (joint_pos + tq_np) - k_spring * joint_pos, -3.0, 3.0)
                h = np.array(
                    [_cap["osc_d"], _cap["osc_v"]]
                    + _cap["bn_d"] + _cap["bn_v"]
                    + _cap["mu_d"] + _cap["mu_v"]
                    + [_cap["env_mod"], _cap["freq_scale"], _cap["amp_scale"]],
                    dtype=np.float32,
                )
                hidden_list.append(h)
    finally:
        model.__class__.forward = original_fwd
    return np.stack(hidden_list)


# ---------------------------------------------------------------------------
# Trajectory tangling Q(τ) — Russo et al. 2018
# ---------------------------------------------------------------------------

def compute_tangling(X: np.ndarray, delta: int = T_DELTA, eps: float = EPS_TNG) -> np.ndarray:
    """
    X: (T, D)
    Returns Q: (T,)
    """
    T = X.shape[0]
    dX = np.diff(X, axis=0, prepend=X[:1])   # velocity; shape (T, D)
    Q  = np.zeros(T)
    mask_template = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]) > delta  # (T, T)
    # num[i,j] = ||dX[i] - dX[j]||²
    num = np.sum((dX[:, None, :] - dX[None, :, :]) ** 2, axis=-1)   # (T, T)
    den = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1) + eps  # (T, T)
    ratio = num / den   # (T, T)
    ratio = np.where(mask_template, ratio, 0.0)
    Q = ratio.max(axis=1)
    return Q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis():
    print("=" * 72)
    print("NMAP Hidden-State Trajectory Tangling Analysis")
    print("=" * 72)

    # Stage 1: collect hidden trajectories --------------------------------
    all_hidden = {}   # ablation_name → {cond: ndarray (T, D)}
    abl_keys = list(ABLATIONS.keys())

    for abl_name, ckpt_rel in ABLATIONS.items():
        clean = abl_name.replace("\n", " ")
        print(f"\nLoading {clean} ...")
        model = load_model(os.path.join(_HERE, ckpt_rel))
        cond_dict = {}
        for cond, env_type in CONDITIONS.items():
            cond_dict[cond] = _extract_hidden(model, env_type, T=T_STEPS)
            print(f"  {cond}: {cond_dict[cond].shape}")
        all_hidden[abl_name] = cond_dict

    # Stage 2: PCA (fit on merged per ablation) ---------------------------
    print("\nFitting PCA ...")
    pca_data = {}   # abl → {pcs, labels, expl}
    for abl_name, cond_dict in all_hidden.items():
        merged = np.vstack([cond_dict[c] for c in COND_LIST])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(merged)
        pca    = PCA(n_components=3)
        pcs    = pca.fit_transform(scaled)
        labels = []
        for c in COND_LIST:
            labels.extend([c] * T_STEPS)
        pca_data[abl_name] = {"pcs": pcs, "labels": labels, "expl": pca.explained_variance_ratio_,
                               "scaler": scaler, "pca": pca}

    # Stage 3: Tangling metrics -------------------------------------------
    print("Computing Q(t) ...")

    # 3a: self-tangling WITHIN each condition
    self_Q = {}   # abl → {cond: Q_array}
    for abl_name, cond_dict in all_hidden.items():
        self_Q[abl_name] = {}
        for cond in COND_LIST:
            self_Q[abl_name][cond] = compute_tangling(cond_dict[cond])

    # 3b: cross-condition tangling on MERGED trajectory
    cross_Q = {}  # abl → Q_array
    for abl_name, cond_dict in all_hidden.items():
        merged = np.vstack([cond_dict[c] for c in COND_LIST])
        cross_Q[abl_name] = compute_tangling(merged)

    # 3c: centroid separation matrix
    sep_mat = {}  # abl → (3, 3)
    for abl_name, cond_dict in all_hidden.items():
        scaler = pca_data[abl_name]["scaler"]
        centroids = {c: scaler.transform(cond_dict[c]).mean(0) for c in COND_LIST}
        mat = np.zeros((3, 3))
        for i, ci in enumerate(COND_LIST):
            for j, cj in enumerate(COND_LIST):
                mat[i, j] = np.linalg.norm(centroids[ci] - centroids[cj])
        sep_mat[abl_name] = mat

    # Precompute boundary Q for use in figures
    boundary_Q_vals = []
    for a in abl_keys:
        Qcr = cross_Q[a]
        bv = np.mean(np.concatenate([Qcr[T_STEPS-10:T_STEPS+10],
                                     Qcr[2*T_STEPS-10:2*T_STEPS+10]]))
        boundary_Q_vals.append(bv)

    # Load neuromodulatory params from checkpoints
    water_freqs, land_freqs, water_amps, land_amps = [], [], [], []
    for ckpt_rel in ABLATIONS.values():
        ck = torch.load(os.path.join(_HERE, ckpt_rel), map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        water_freqs.append(float(sd["water_frequency_scale"]))
        land_freqs.append(float(sd["land_frequency_scale"]))
        water_amps.append(float(sd["water_amplitude_scale"]))
        land_amps.append(float(sd["land_amplitude_scale"]))

    # Print summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("QUANTITATIVE SUMMARY")
    print("=" * 72)
    header = f"{'Ablation':<32} {'Q_water':>9} {'Q_trans':>9} {'Q_land':>9} {'Q_cross':>9} {'Sep(W-L)':>9}"
    print(header)
    print("-" * 72)
    for abl_name, long_name in zip(abl_keys, ABLATION_LONG):
        qw  = self_Q[abl_name]["Water"].mean()
        qt  = self_Q[abl_name]["Transition"].mean()
        ql  = self_Q[abl_name]["Land"].mean()
        qcr = cross_Q[abl_name][T_STEPS:2*T_STEPS].mean()
        sep = sep_mat[abl_name][0, 2]
        print(f"  {long_name:<30} {qw:>9.3f} {qt:>9.3f} {ql:>9.3f} {qcr:>9.3f} {sep:>9.4f}")

    # ------------------------------------------------------------------
    # Shared style helpers
    # ------------------------------------------------------------------
    n_abl = len(abl_keys)
    SHORT  = [k.replace("\n", "\n") for k in abl_keys]   # tick labels

    # Axis labels with a unit-like note inside brackets
    def _style_ax(ax, xlabel="", ylabel="", title="", title_color="white",
                  fontsize_title=11, fontsize_label=10):
        ax.set_facecolor(PANEL_BG)
        if xlabel:
            ax.set_xlabel(xlabel, color="#9ca3af", fontsize=fontsize_label, labelpad=6)
        if ylabel:
            ax.set_ylabel(ylabel, color="#9ca3af", fontsize=fontsize_label, labelpad=6)
        if title:
            ax.set_title(title, color=title_color, fontsize=fontsize_title,
                         fontweight="semibold", pad=8)
        ax.tick_params(colors="#6b7280", labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(GRID_CLR)
            ax.spines[sp].set_linewidth(0.8)

    def _panel_letter(ax, letter, x=-0.10, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                color="white", fontsize=13, fontweight="bold", va="top")

    # Confidence ellipse helper (1-sigma)
    def _confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        cov = np.cov(x, y)
        pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-9)
        rx = np.sqrt(1 + pearson) * n_std
        ry = np.sqrt(1 - pearson) * n_std
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)
        ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
        transform = (transforms.Affine2D()
                     .rotate_deg(45)
                     .scale(scale_x, scale_y)
                     .translate(mean_x, mean_y)
                     + ax.transData)
        ellipse.set_transform(transform)
        ax.add_patch(ellipse)

    # =========================================================
    # FIGURE 1: 3D PCA — Hidden-State Attractor Geometry
    # Layout: 2 rows × 3 cols (last cell used for legend)
    # =========================================================
    print("\nPlotting Figure 1: 3D PCA trajectories ...")
    fig1 = plt.figure(figsize=(20, 11), facecolor=DARK_BG)
    fig1.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.10,
                         wspace=0.05, hspace=0.30)

    subplot_idx = [1, 2, 3, 4, 5]   # positions in a 2×3 grid (6th cell = legend)
    for idx, (abl_name, pd_item) in enumerate(pca_data.items()):
        row, col = divmod(idx, 3)
        ax = fig1.add_subplot(2, 3, idx + 1, projection="3d")
        ax.set_facecolor(DARK_BG)

        pcs, labels, expl = pd_item["pcs"], pd_item["labels"], pd_item["expl"]
        for cond in COND_LIST:
            mask = np.array([l == cond for l in labels])
            x, y, z = pcs[mask, 0], pcs[mask, 1], pcs[mask, 2]
            c = COND_COLORS[cond]
            ax.plot(x, y, z, color=c, linewidth=1.6, alpha=0.85)
            ax.scatter(x[0],  y[0],  z[0],  color=c, s=50, marker="o", depthshade=False, zorder=6)
            ax.scatter(x[-1], y[-1], z[-1], color=c, s=50, marker="^", depthshade=False, zorder=6)

        ev_pct = "/".join(f"{100*e:.0f}" for e in expl)
        abl_clean = abl_name.replace("\n", " ")
        ax.set_title(f"{abl_clean}\n({ev_pct}% variance)", color="white",
                     fontsize=9.5, fontweight="semibold", pad=4)
        ax.set_xlabel("PC 1", color="#6b7280", fontsize=7, labelpad=0)
        ax.set_ylabel("PC 2", color="#6b7280", fontsize=7, labelpad=0)
        ax.set_zlabel("PC 3", color="#6b7280", fontsize=7, labelpad=0)
        ax.tick_params(colors="#6b7280", labelsize=5.5, pad=0)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.4)
        ax.view_init(elev=22, azim=-55)

    # Legend in 6th subplot slot
    ax_leg = fig1.add_subplot(2, 3, 6)
    ax_leg.set_facecolor(DARK_BG)
    ax_leg.axis("off")
    legend_handles = (
        [plt.Line2D([0], [0], color=COND_COLORS[c], lw=2.5, label=c) for c in COND_LIST]
        + [plt.scatter([], [], color="white", s=55, marker="o", label="Trajectory start"),
           plt.scatter([], [], color="white", s=55, marker="^", label="Trajectory end")]
    )
    ax_leg.legend(handles=legend_handles, loc="center", fontsize=11,
                  facecolor=PANEL_BG, edgecolor="#374151", labelcolor="white",
                  framealpha=0.9, borderpad=1.2, labelspacing=1.0)
    ax_leg.text(0.5, 0.88, "Condition", color="white", fontsize=10,
                fontweight="bold", ha="center", transform=ax_leg.transAxes)

    fig1.text(0.5, 0.93,
              "Neural State-Space Geometry: Low-Level Controller Hidden-State Trajectories",
              color="white", fontsize=14, fontweight="bold", ha="center", va="bottom")
    fig1.text(0.5, 0.905,
              "Principal Component Analysis of interneuron (B-neuron) and muscle activations "
              "across 10 oscillator cycles per condition  |  h(t) ∈ ℝ²⁵",
              color="#9ca3af", fontsize=10, ha="center", va="bottom")

    out1 = os.path.join(OUT_DIR, "01_pca_3d_state_space.png")
    fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig1)
    print(f"  Saved {out1}")

    # =========================================================
    # FIGURE 2: 2D Attractor Geometry — trajectories + 1-sigma ellipses
    # =========================================================
    print("Plotting Figure 2: 2D attractor ellipses ...")
    fig2, axes2 = plt.subplots(1, n_abl, figsize=(5.2 * n_abl, 5.5), facecolor=DARK_BG)
    fig2.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.18, wspace=0.28)

    for ax, (abl_name, pd_item) in zip(axes2, pca_data.items()):
        pcs, labels, expl = pd_item["pcs"], pd_item["labels"], pd_item["expl"]
        for cond in COND_LIST:
            mask = np.array([l == cond for l in labels])
            x, y  = pcs[mask, 0], pcs[mask, 1]
            c = COND_COLORS[cond]
            ax.plot(x, y, color=c, lw=1.5, alpha=0.75)
            ax.scatter(x[0],  y[0],  color=c, s=70, marker="o", zorder=6, linewidths=0)
            ax.scatter(x[-1], y[-1], color=c, s=70, marker="^", zorder=6, linewidths=0)
            _confidence_ellipse(x, y, ax, n_std=1.5,
                                facecolor=c, alpha=0.12, edgecolor=c,
                                linewidth=1.2, linestyle="--", zorder=2)
            cx, cy = np.mean(x), np.mean(y)
            ax.scatter(cx, cy, color=c, s=120, marker="x", linewidths=2.5, zorder=7)

        _style_ax(ax,
                  xlabel=f"PC 1  ({100*expl[0]:.0f}% var.)",
                  ylabel=f"PC 2  ({100*expl[1]:.0f}% var.)",
                  title=abl_name.replace("\n", " "))

    handles2 = (
        [plt.Line2D([0], [0], color=COND_COLORS[c], lw=2, label=c) for c in COND_LIST]
        + [plt.Line2D([0], [0], color="white", lw=0, marker="x",
                      markersize=9, markeredgewidth=2, label="Centroid")]
    )
    fig2.legend(handles=handles2, loc="lower center", ncol=4, fontsize=10,
                facecolor=PANEL_BG, edgecolor="#374151", labelcolor="white",
                framealpha=0.9, bbox_to_anchor=(0.5, 0.00))

    fig2.text(0.5, 0.90,
              "Locomotor Attractor Geometry in the Principal Plane (PC1–PC2)",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig2.text(0.5, 0.865,
              "Shaded ellipses: 1.5σ covariance envelope per condition  |  "
              "✕ marks attractor centroid  |  "
              "Overlap = shared attractor; separation = neuromod creates distinct states",
              color="#9ca3af", fontsize=9.5, ha="center")

    out2 = os.path.join(OUT_DIR, "02_pca_2d_attractor_ellipses.png")
    fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig2)
    print(f"  Saved {out2}")

    # =========================================================
    # FIGURE 3: Pairwise Centroid Distance Matrix
    # =========================================================
    print("Plotting Figure 3: centroid separation heatmaps ...")
    fig3, axes3 = plt.subplots(1, n_abl, figsize=(4.8 * n_abl, 4.8), facecolor=DARK_BG)
    fig3.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.16, wspace=0.40)
    vmax = max(m.max() for m in sep_mat.values())

    for ax, (abl_name, mat) in zip(axes3, sep_mat.items()):
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(COND_LIST, color="#9ca3af", fontsize=9, rotation=35, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(COND_LIST, color="#9ca3af", fontsize=9)
        ax.set_title(abl_name.replace("\n", " "), color="white", fontsize=9.5,
                     fontweight="semibold", pad=8)
        for i in range(3):
            for j in range(3):
                txt_color = "white" if mat[i, j] < 0.65 * vmax else "#111111"
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        color=txt_color, fontsize=10, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#6b7280", labelsize=7)
        cb.set_label("Euclidean dist. [std. units]", color="#9ca3af", fontsize=8)

    fig3.text(0.5, 0.91,
              "Pairwise Euclidean Distance Between Locomotor Attractor Centroids",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig3.text(0.5, 0.875,
              "Computed in standardized hidden-state space  |  "
              "Off-diagonal entries measure spatial separation between environmental attractors",
              color="#9ca3af", fontsize=9.5, ha="center")

    out3 = os.path.join(OUT_DIR, "03_attractor_centroid_distances.png")
    fig3.savefig(out3, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig3)
    print(f"  Saved {out3}")

    # =========================================================
    # FIGURE 4: Trajectory Tangling Q(τ) — self & cross combined
    # Layout: top rows = self-Q timeseries per ablation (one col per condition),
    #         bottom = cross-condition Q(τ) per ablation
    # =========================================================
    print("Plotting Figure 4: trajectory tangling Q(t) combined ...")
    t_axis  = np.arange(T_STEPS)
    t_cross = np.arange(3 * T_STEPS)
    smW, smC = 25, 40

    fig4 = plt.figure(figsize=(22, 3.0 * n_abl + 3.5), facecolor=DARK_BG)
    gs4  = GridSpec(n_abl + 1, 4, figure=fig4,
                    hspace=0.55, wspace=0.32,
                    left=0.07, right=0.97, top=0.91, bottom=0.06)

    # Top section: self-Q per ablation (3 conditions side by side)
    for row, (abl_name, cond_Qs) in enumerate(self_Q.items()):
        for col, cond in enumerate(COND_LIST):
            ax = fig4.add_subplot(gs4[row, col])
            Q  = cond_Qs[cond]
            ax.fill_between(t_axis, Q, alpha=0.18, color=COND_COLORS[cond])
            ax.plot(t_axis, np.convolve(Q, np.ones(smW)/smW, "same"),
                    color=COND_COLORS[cond], lw=1.8)
            mean_Q = Q.mean()
            ax.axhline(mean_Q, color=COND_COLORS[cond], lw=1.0, ls="--", alpha=0.55)
            ax.text(0.97, 0.93, f"μ = {mean_Q:.1f}", ha="right", va="top",
                    transform=ax.transAxes, color="white", fontsize=8.5)
            _style_ax(ax,
                      ylabel="Q(τ)" if col == 0 else "",
                      xlabel="Time step" if row == n_abl - 1 else "")
            if row == 0:
                ax.set_title(f"Self-tangling — {cond}",
                             color=COND_COLORS[cond], fontsize=10, fontweight="semibold")
            if col == 0:
                ax.text(-0.28, 0.50, abl_name.replace("\n", " "),
                        transform=ax.transAxes, color="white", fontsize=8.5,
                        va="center", ha="left", rotation=0)

        # 4th column: mean self-Q bar chart for this ablation
        ax_bar = fig4.add_subplot(gs4[row, 3])
        bar_vals = [cond_Qs[c].mean() for c in COND_LIST]
        bars = ax_bar.barh(COND_LIST, bar_vals,
                           color=[COND_COLORS[c] for c in COND_LIST],
                           alpha=0.85, height=0.55)
        for b, v in zip(bars, bar_vals):
            ax_bar.text(v + 0.5, b.get_y() + b.get_height()/2,
                        f"{v:.0f}", va="center", color="white", fontsize=8)
        _style_ax(ax_bar, xlabel="Mean Q(τ)")
        if row == 0:
            ax_bar.set_title("Mean per condition", color="white",
                             fontsize=10, fontweight="semibold")
        ax_bar.tick_params(colors="#9ca3af", labelsize=8.5)

    # Bottom row: cross-condition Q per ablation (overlaid lines)
    ax_cross = fig4.add_subplot(gs4[n_abl, :])
    ax_cross.set_facecolor(PANEL_BG)
    cross_colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_abl))
    for i, (abl_name, Qcr) in enumerate(cross_Q.items()):
        ax_cross.plot(t_cross,
                      np.convolve(Qcr, np.ones(smC)/smC, "same"),
                      color=cross_colors[i], lw=1.8, alpha=0.9,
                      label=abl_name.replace("\n", " "))
    for ci, cond in enumerate(COND_LIST):
        t0, t1 = ci * T_STEPS, (ci + 1) * T_STEPS
        ax_cross.axvspan(t0, t1, alpha=0.06, color=COND_COLORS[cond])
        ax_cross.text((t0+t1)/2, ax_cross.get_ylim()[0] if False else 0,
                      cond, color=COND_COLORS[cond], fontsize=9, ha="center", va="bottom")
    for sep in [T_STEPS, 2 * T_STEPS]:
        ax_cross.axvline(sep, color="#6b7280", lw=1.0, ls="--", alpha=0.6)
    _style_ax(ax_cross,
              xlabel="Time step  (Water → Transition → Land, concatenated)",
              ylabel="Q(τ)",
              title="Cross-Condition Trajectory Tangling Q(τ)  "
                    "—  all ablations overlaid  |  "
                    "Spikes near boundaries indicate overlapping attractors")
    ax_cross.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#374151",
                    labelcolor="white", loc="upper right", framealpha=0.9)

    fig4.text(0.5, 0.94,
              "Trajectory Tangling Q(τ): Attractor Self-Organization and Cross-Condition Interference",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig4.text(0.5, 0.916,
              "Q(τ) = maxᵯ'  ||ẋ(τ) − ẋ(τ')||"
              "² / (||x(τ) − x(τ')||² + ε)   "
              "[Russo et al., Nat. Neurosci. 2018]  "
              "—  Low Q = clean attractor; High Q = state-velocity ambiguity",
              color="#9ca3af", fontsize=9.5, ha="center")

    out4 = os.path.join(OUT_DIR, "04_trajectory_tangling_Q.png")
    fig4.savefig(out4, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig4)
    print(f"  Saved {out4}")

    # =========================================================
    # FIGURE 5: Oscillator Limit Cycles (phase portraits)
    # =========================================================
    print("Plotting Figure 5: oscillator phase portraits ...")
    fig5, axes5 = plt.subplots(n_abl, 3, figsize=(14, 3.0 * n_abl), facecolor=DARK_BG)
    fig5.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.06,
                         hspace=0.45, wspace=0.35)

    for row, (abl_name, cond_dict) in enumerate(all_hidden.items()):
        for col, cond in enumerate(COND_LIST):
            ax = axes5[row, col]
            h = cond_dict[cond]
            osc_d, osc_v = h[:, 0], h[:, 1]
            t_frac = np.linspace(0, 1, len(osc_d))
            # Use LineCollection for smooth gradient coloring
            from matplotlib.collections import LineCollection
            segs = np.array([[[osc_d[k], osc_v[k]], [osc_d[k+1], osc_v[k+1]]]
                              for k in range(len(osc_d)-1)])
            lc = LineCollection(segs, cmap="plasma",
                                norm=plt.Normalize(0, 1), linewidth=1.5, alpha=0.8)
            lc.set_array(t_frac[:-1])
            ax.add_collection(lc)
            ax.autoscale()
            ax.scatter(osc_d[0], osc_v[0], color="white", s=55, marker="o", zorder=6)
            ax.scatter(osc_d[-1], osc_v[-1], color="white", s=55, marker="^", zorder=6)

            _style_ax(ax,
                      xlabel="Dorsal oscillator activity" if row == n_abl-1 else "",
                      ylabel="Ventral oscillator activity" if col == 0 else "")
            if row == 0:
                ax.set_title(f"{cond}", color=COND_COLORS[cond],
                             fontsize=11, fontweight="semibold")
            if col == 0:
                ax.set_ylabel(f"{abl_name.replace(chr(10), ' ')}\n\nVentral activity",
                              color="white", fontsize=8.5)

            # Annotate: cycle count readable from how many times orbit closes
            freq_ratio = (water_freqs[row] if cond == "Water" else
                          (land_freqs[row] if cond == "Land" else
                           0.5*(water_freqs[row]+land_freqs[row])))
            ax.text(0.97, 0.05, f"f×{freq_ratio:.1f}",
                    transform=ax.transAxes, color="#9ca3af",
                    fontsize=8, ha="right", va="bottom")

    fig5.text(0.5, 0.93,
              "Relaxation Oscillator Limit Cycles Under Environmental Neuromodulation",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig5.text(0.5, 0.905,
              "Phase portrait of head oscillator (dorsal vs. ventral activity)  |  "
              "Color: time (early = purple, late = yellow)  |  "
              "f×: learned frequency scaling factor  |  "
              "Distinct limit cycles per row = neuromod reshapes oscillator geometry",
              color="#9ca3af", fontsize=9.5, ha="center")

    out5 = os.path.join(OUT_DIR, "05_oscillator_limit_cycles.png")
    fig5.savefig(out5, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig5)
    print(f"  Saved {out5}")

    # =========================================================
    # FIGURE 6: B-Neuron Joint Activation Heatmaps  [NEW]
    # For the best ablation (most separation): show bneuron_d and bneuron_v
    # across all joints and all conditions as a time × joint heatmap.
    # =========================================================
    print("Plotting Figure 6: B-neuron joint activation heatmaps ...")
    best_abl = abl_keys[np.argmax([sep_mat[a][0, 2] for a in abl_keys])]

    fig6 = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig6.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.10,
                         hspace=0.50, wspace=0.35)
    gs6 = GridSpec(2, 3, figure=fig6)

    joint_labels = [f"Joint {i}" for i in range(N_JOINTS)]
    t_show = np.arange(T_STEPS)

    for ci, cond in enumerate(COND_LIST):
        h = all_hidden[best_abl][cond]
        bn_d = h[:, 2:7].T    # (5 joints, T)
        bn_v = h[:, 7:12].T

        # Dorsal row
        ax_d = fig6.add_subplot(gs6[0, ci])
        im_d = ax_d.imshow(bn_d, aspect="auto", cmap="Blues",
                           extent=[0, T_STEPS, N_JOINTS - 0.5, -0.5],
                           vmin=0, vmax=1)
        _style_ax(ax_d,
                  xlabel="Time step" if True else "",
                  title=f"{cond}  —  Dorsal B-Neurons",
                  title_color=COND_COLORS[cond])
        ax_d.set_yticks(range(N_JOINTS))
        ax_d.set_yticklabels(joint_labels, color="#9ca3af", fontsize=8.5)
        cb_d = plt.colorbar(im_d, ax=ax_d, fraction=0.04, pad=0.03)
        cb_d.ax.tick_params(colors="#6b7280", labelsize=7)
        cb_d.set_label("Activation", color="#9ca3af", fontsize=8)
        if ci == 0:
            _panel_letter(ax_d, "A")

        # Ventral row
        ax_v = fig6.add_subplot(gs6[1, ci])
        im_v = ax_v.imshow(bn_v, aspect="auto", cmap="Reds",
                           extent=[0, T_STEPS, N_JOINTS - 0.5, -0.5],
                           vmin=0, vmax=1)
        _style_ax(ax_v,
                  xlabel="Time step",
                  title=f"{cond}  —  Ventral B-Neurons",
                  title_color=COND_COLORS[cond])
        ax_v.set_yticks(range(N_JOINTS))
        ax_v.set_yticklabels(joint_labels, color="#9ca3af", fontsize=8.5)
        cb_v = plt.colorbar(im_v, ax=ax_v, fraction=0.04, pad=0.03)
        cb_v.ax.tick_params(colors="#6b7280", labelsize=7)
        cb_v.set_label("Activation", color="#9ca3af", fontsize=8)
        if ci == 0:
            _panel_letter(ax_v, "B", y=1.08)

    abl_clean_best = best_abl.replace("\n", " ")
    fig6.text(0.5, 0.92,
              "Joint-Wise Interneuron (B-Neuron) Population Activity Across Viscosity Regimes",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig6.text(0.5, 0.895,
              f"Model: {abl_clean_best}  |  "
              "Each row = one body joint (anterior → posterior)  |  "
              "Column = time  |  "
              "Color intensity = activation strength  |  "
              "Distinct spatiotemporal patterns = environment-specific motor programs",
              color="#9ca3af", fontsize=9.5, ha="center")

    out6 = os.path.join(OUT_DIR, "06_bneuron_population_activity.png")
    fig6.savefig(out6, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig6)
    print(f"  Saved {out6}")

    # =========================================================
    # FIGURE 7: Multi-Metric Ablation Summary Dashboard
    # Panels: A self-Q bars, B Sep(W-L), C boundary cross-Q, D neuromod params
    # + Radar chart inset comparing normalized metrics
    # =========================================================
    print("Plotting Figure 7: ablation summary dashboard ...")
    fig7 = plt.figure(figsize=(22, 10), facecolor=DARK_BG)
    gs7  = GridSpec(2, 3, figure=fig7, hspace=0.52, wspace=0.38,
                    left=0.06, right=0.97, top=0.88, bottom=0.10)

    ax7a = fig7.add_subplot(gs7[0, 0])   # A: self-Q grouped bars
    ax7b = fig7.add_subplot(gs7[0, 1])   # B: W-L centroid distance
    ax7c = fig7.add_subplot(gs7[1, 0])   # C: cross-Q at boundaries
    ax7d = fig7.add_subplot(gs7[1, 1])   # D: learned neuromod parameters
    ax7e = fig7.add_subplot(gs7[:, 2], polar=True)   # E: radar chart

    x_pos = np.arange(n_abl)
    bar_w = 0.25

    # Panel A: self-Q grouped bars
    for ci, cond in enumerate(COND_LIST):
        vals = [self_Q[a][cond].mean() for a in abl_keys]
        ax7a.bar(x_pos + (ci - 1) * bar_w, vals, bar_w,
                 label=cond, color=COND_COLORS[cond], alpha=0.85, linewidth=0)
    _style_ax(ax7a, ylabel="Mean Q(τ)  [lower ↓ = better]",
              title="Intra-Condition Self-Tangling per Ablation")
    ax7a.set_xticks(x_pos);  ax7a.set_xticklabels(SHORT, color="white", fontsize=8.5)
    ax7a.legend(fontsize=9, facecolor=DARK_BG, edgecolor=GRID_CLR, labelcolor="white")
    _panel_letter(ax7a, "A")

    # Panel B: W-L centroid distance
    wl_dists  = [sep_mat[a][0, 2] for a in abl_keys]
    bar_clrs  = ["#e8c468" if d == max(wl_dists) else "#4a9eff" for d in wl_dists]
    bars7b    = ax7b.bar(x_pos, wl_dists, color=bar_clrs, alpha=0.85, linewidth=0)
    for b, v in zip(bars7b, wl_dists):
        ax7b.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                  ha="center", color="white", fontsize=9, fontweight="bold")
    _style_ax(ax7b,
              ylabel="Centroid distance  [std. units, higher ↑ = better]",
              title="Water–Land Attractor Spatial Separation")
    ax7b.set_xticks(x_pos);  ax7b.set_xticklabels(SHORT, color="white", fontsize=8.5)
    _panel_letter(ax7b, "B")

    # Panel C: cross-Q at condition boundaries
    bclrs = ["#cc6677" if v == max(boundary_Q_vals) else "#44aa99"
             for v in boundary_Q_vals]
    bars7c = ax7c.bar(x_pos, boundary_Q_vals, color=bclrs, alpha=0.85, linewidth=0)
    for b, v in zip(bars7c, boundary_Q_vals):
        ax7c.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                  ha="center", color="white", fontsize=9, fontweight="bold")
    _style_ax(ax7c,
              ylabel="Q(τ) at condition boundary  [lower ↓ = better]",
              title="Cross-Condition Tangling at Attractor Boundaries")
    ax7c.set_xticks(x_pos);  ax7c.set_xticklabels(SHORT, color="white", fontsize=8.5)
    _panel_letter(ax7c, "C")

    # Panel D: learned neuromodulatory parameters
    ax7d.plot(x_pos, water_freqs, "o-",  color="#2166ac", lw=2.0, ms=9, label="Water freq. scale")
    ax7d.plot(x_pos, land_freqs,  "s-",  color="#d73027", lw=2.0, ms=9, label="Land freq. scale")
    ax7d.plot(x_pos, water_amps,  "^--", color="#74add1", lw=1.6, ms=8, label="Water amp. scale")
    ax7d.plot(x_pos, land_amps,   "v--", color="#f4a261", lw=1.6, ms=8, label="Land amp. scale")
    ax7d.axhline(1.0, color="#6b7280", ls=":", lw=1.0, alpha=0.7)
    ax7d.text(n_abl - 1 + 0.05, 1.03, "baseline = 1.0",
              color="#6b7280", fontsize=8, va="bottom")
    _style_ax(ax7d,
              ylabel="Learned scale factor",
              title="Neuromodulatory Broadcast Strength\n(Learned freq./amp. parameters per ablation)")
    ax7d.set_xticks(x_pos);  ax7d.set_xticklabels(SHORT, color="white", fontsize=8.5)
    ax7d.legend(fontsize=8.5, facecolor=DARK_BG, edgecolor=GRID_CLR,
                labelcolor="white", ncol=2)
    _panel_letter(ax7d, "D")

    # Panel E: Radar chart — normalized multi-metric comparison
    metrics = ["Q Water\n(inv.)", "Q Trans.\n(inv.)", "Q Land\n(inv.)",
               "Sep.\n(W-L)", "Cross-Q\n(inv.)"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize all metrics to [0, 1]; invert Q metrics (lower = better)
    raw = {
        "Q Water": np.array([self_Q[a]["Water"].mean() for a in abl_keys]),
        "Q Trans": np.array([self_Q[a]["Transition"].mean() for a in abl_keys]),
        "Q Land":  np.array([self_Q[a]["Land"].mean() for a in abl_keys]),
        "Sep":     np.array([sep_mat[a][0, 2] for a in abl_keys]),
        "Cross-Q": np.array(boundary_Q_vals),
    }
    def _norm_inv(v): return 1.0 - (v - v.min()) / (v.max() - v.min() + 1e-9)
    def _norm(v):     return      (v - v.min()) / (v.max() - v.min() + 1e-9)
    normalized = np.column_stack([
        _norm_inv(raw["Q Water"]),
        _norm_inv(raw["Q Trans"]),
        _norm_inv(raw["Q Land"]),
        _norm(raw["Sep"]),
        _norm_inv(raw["Cross-Q"]),
    ])

    radar_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_abl))
    ax7e.set_facecolor(DARK_BG)
    ax7e.set_theta_offset(np.pi / 2)
    ax7e.set_theta_direction(-1)
    ax7e.set_thetagrids(np.degrees(angles[:-1]), metrics,
                        color="#9ca3af", fontsize=9)
    ax7e.set_rlabel_position(30)
    ax7e.tick_params(colors="#6b7280", labelsize=7)
    ax7e.set_ylim(0, 1)
    ax7e.spines["polar"].set_color(GRID_CLR)
    ax7e.grid(color=GRID_CLR, linewidth=0.7)
    ax7e.set_facecolor(PANEL_BG)

    for i, (abl_name, vals) in enumerate(zip(abl_keys, normalized)):
        v = vals.tolist() + vals[:1].tolist()
        ax7e.plot(angles, v, color=radar_colors[i], lw=2.0, alpha=0.9,
                  label=abl_name.replace("\n", " "))
        ax7e.fill(angles, v, color=radar_colors[i], alpha=0.10)

    ax7e.legend(fontsize=8.5, facecolor=PANEL_BG, edgecolor="#374151",
                labelcolor="white", loc="upper left",
                bbox_to_anchor=(-0.30, 1.18))
    ax7e.set_title("Multi-Metric\nAblation Profile\n(normalized, outer = better)",
                   color="white", fontsize=10, fontweight="semibold", pad=18)
    _panel_letter(ax7e, "E", x=-0.18, y=1.12)

    fig7.text(0.5, 0.93,
              "NMAP Ablation Study: Attractor Quality and Neuromodulatory Separation — Summary",
              color="white", fontsize=14, fontweight="bold", ha="center")
    fig7.text(0.5, 0.905,
              "Does the neuromodulatory broadcast from the high-level controller reduce "
              "trajectory tangling in the low-level locomotor circuit?",
              color="#9ca3af", fontsize=10, ha="center")

    out7 = os.path.join(OUT_DIR, "07_ablation_summary_dashboard.png")
    fig7.savefig(out7, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig7)
    print(f"  Saved {out7}")

    # Final summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print(f"Output: {os.path.abspath(OUT_DIR)}")
    print("=" * 72)
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")

    best_sep_idx   = np.argmax([sep_mat[a][0, 2] for a in abl_keys])
    best_self_water = np.argmin([self_Q[a]["Water"].mean() for a in abl_keys])
    print(f"\nKey findings:")
    print(f"  Best water-land attractor separation: {ABLATION_LONG[best_sep_idx]}")
    print(f"  Cleanest water attractor (min self-Q): {ABLATION_LONG[best_self_water]}")


if __name__ == "__main__":
    run_analysis()
