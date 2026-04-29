"""Publication-grade narrative figures for experiments/ results."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from visualization.data_sources import (
    ROOT,
    load_ablation,
    load_adversarial,
    load_feature_forgery,
    load_feature_sensitivity,
    load_quantum_protection,
    load_temporal_generalization,
    load_topological_invariance,
)
from visualization.mpl_setup import (
    apply_default_style,
    get_plot_modules,
    journal_palette,
    save_figure,
    set_axis_style,
)

plt = get_plot_modules(ROOT)
apply_default_style(plt)
COL = journal_palette()
THEME = {
    "primary": COL["qgad"],          # Hybrid / Q-GAD main color
    "secondary": COL["gnn"],         # Auxiliary contrast
    "tertiary": COL["accent"],       # Accent
    "neutral": "#5F6875",            # Baseline / text-support
    "soft_blue": "#E6EDF6",          # Soft fill
    "soft_warm": "#F3EEE4",          # Soft warm band
    "positive": "#2E7D32",           # Positive delta
    "sensitivity": "#4A6F86",        # Sensitivity-specific (cool steel)
    "quantum": "#8273A8",            # Quantum-margin specific
    "topology": "#C39280",           # Topology-specific (warm clay)
}


def _eps_from_key(key: str) -> float:
    return float(key.split("_")[-1])


def _epsilon_keys(payload: dict) -> list[str]:
    return sorted([k for k in payload.keys() if k.startswith("epsilon_")], key=_eps_from_key)


def _clear_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()


def _panel_badge(ax, label: str) -> None:
    # Panel letters are embedded into subplot titles for robust, overlap-free layout.
    return


def _polish_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#D6D6D6", alpha=0.17, linewidth=0.6)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3E3E3E")
    set_axis_style(ax)


def plot_story_attack_pipeline(out: Path) -> None:
    """Figure 1: threat-facing robustness evidence (advanced layout)."""
    forgery = load_feature_forgery()
    adv = load_adversarial()["results"]["qgad"]
    sens = load_feature_sensitivity()["results"]
    protect = load_quantum_protection()["results"]

    q_clean = float(forgery["qgad"]["baseline_recall"])
    q_forged = float(forgery["qgad"]["forged_recall"])
    x_clean = float(forgery["xgboost"]["baseline_recall"])
    x_forged = float(forgery["xgboost"]["forged_recall"])
    q_drop = q_clean - q_forged
    x_drop = x_clean - x_forged
    robustness_gap = x_drop - q_drop

    adv_keys = _epsilon_keys(adv)
    eps = [0.0] + [_eps_from_key(k) for k in adv_keys]
    f1 = [float(adv["clean_f1"]) * 100.0] + [float(adv[k]["f1"]) * 100.0 for k in adv_keys]
    auc = [float(adv["clean_auc"]) * 100.0] + [float(adv[k]["auc"]) * 100.0 for k in adv_keys]
    adv_retention = f1[-1] / max(f1[0], 1e-9)

    sens_eps = [0.01, 0.05, 0.1]
    sens_clean_f1 = float(sens["clean"]["f1"])
    sens_q = [float(sens[f"Quantum_Features_epsilon_{e}"]["f1"]) / max(sens_clean_f1, 1e-9) * 100.0 for e in sens_eps]
    sens_c = [float(sens[f"Classical_Features_epsilon_{e}"]["f1"]) / max(sens_clean_f1, 1e-9) * 100.0 for e in sens_eps]
    sens_b = [float(sens[f"Both_Features_epsilon_{e}"]["f1"]) / max(sens_clean_f1, 1e-9) * 100.0 for e in sens_eps]
    sens_qc_gap_01 = float(sens["Quantum_Features_epsilon_0.1"]["f1"]) - float(sens["Classical_Features_epsilon_0.1"]["f1"])

    full = protect["Q-GAD (Full)"]
    zero = protect["Q-GAD (ZeroQuantum)"]
    protect_eps = [_eps_from_key(k) for k in _epsilon_keys(full)]
    full_ret = [float(full[f"epsilon_{e}"]["retained_percent"]) for e in protect_eps]
    zero_ret = [float(zero[f"epsilon_{e}"]["retained_percent"]) for e in protect_eps]
    delta_ret = [f - z for f, z in zip(full_ret, zero_ret)]

    fig = plt.figure(figsize=(16.3, 9.7), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (A) Forgery drop bars (lower is better).
    models = ["Q-GAD", "XGBoost"]
    drop_pp = [q_drop * 100.0, x_drop * 100.0]
    bars = ax_a.bar(models, drop_pp, color=[THEME["primary"], THEME["neutral"]], alpha=0.86, edgecolor=COL["text"], linewidth=0.8, width=0.56)
    for b in bars:
        ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.4, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10.0)
    ax_a.set_ylabel("Recall drop (percentage points, lower is better)")
    ax_a.set_ylim(0, max(drop_pp) + 6.0)
    ax_a.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax_a.set_title("A. Feature Forgery Robustness", pad=10)
    ax_a.text(
        0.07,
        0.5,
        f"Q-GAD reduces drop by {robustness_gap * 100:.1f} pts",
        transform=ax_a.transAxes,
        fontsize=9.8,
        color=COL["text"],
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#AFAFAF", "linewidth": 0.78},
    )
    _polish_axis(ax_a)

    # (B) Adversarial trajectory with direct end labels.
    ax_b.plot(eps, f1, marker="o", color=THEME["primary"], linewidth=2.4)
    ax_b.plot(eps, auc, marker="s", color=THEME["secondary"], linewidth=2.4)
    ax_b.fill_between(eps, f1, auc, color=THEME["soft_blue"], alpha=0.35, zorder=0)
    ax_b.axvspan(0.05, 0.10, color=THEME["soft_warm"], alpha=0.58, zorder=0)
    ax_b.text(0.074, max(max(f1), max(auc)) + 2.2, "high-perturbation regime", fontsize=9.2, color=COL["neutral"], ha="center")
    ax_b.set_xlabel("Perturbation budget ε")
    ax_b.set_ylabel("Score (%)")
    ax_b.set_xlim(-0.003, 0.106)
    ax_b.set_ylim(min(min(f1), min(auc)) - 2.0, max(max(f1), max(auc)) + 4.2)
    ax_b.set_title("B. Adversarial Performance Retention", pad=10)
    ax_b.legend(["F1", "AUC"], frameon=False, loc="upper right")
    ax_b.text(
        0.02,
        0.05,
        f"F1 retention at ε=0.10: {adv_retention * 100:.1f}%",
        transform=ax_b.transAxes,
        fontsize=10.0,
        color=THEME["primary"],
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.72},
    )
    _polish_axis(ax_b)

    # (C) Feature sensitivity with terminal labels (no legend overlap).
    ax_c.plot(sens_eps, sens_q, marker="o", color=THEME["primary"], linewidth=2.2)
    ax_c.plot(sens_eps, sens_c, marker="s", color=THEME["secondary"], linewidth=2.2)
    ax_c.plot(sens_eps, sens_b, marker="^", color=THEME["tertiary"], linewidth=2.2)
    ax_c.axhline(100.0, color="#B0B0B0", linewidth=0.9, linestyle="--")
    ax_c.set_xlabel("Perturbation budget ε")
    ax_c.set_ylabel("F1 retention (%)")
    ax_c.set_xlim(0.008, 0.115)
    ax_c.set_ylim(min(min(sens_q), min(sens_c), min(sens_b)) - 1.2, max(max(sens_q), max(sens_c), max(sens_b)) + 1.8)
    ax_c.set_title("C. Feature Sensitivity Decomposition", pad=10)
    ax_c.legend(["Quantum perturbed", "Classical perturbed", "Both perturbed"], frameon=False, loc="upper right")
    ax_c.text(
        0.03,
        0.95,
        f"Q-C F1 gap at ε=0.10: {sens_qc_gap_01 * 100:.1f} pts",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        color=THEME["primary"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.72},
    )
    _polish_axis(ax_c)

    # (D) Grouped bars + positive deltas across eps.
    x = np.arange(len(protect_eps))
    width = 0.32
    b_full = ax_d.bar(x - width / 2, full_ret, width, color=THEME["primary"], alpha=0.86, edgecolor=COL["text"], linewidth=0.75, label="Full hybrid")
    b_zero = ax_d.bar(x + width / 2, zero_ret, width, color=THEME["neutral"], alpha=0.82, edgecolor=COL["text"], linewidth=0.75, label="Zero-quantum")
    for bf, bz, d in zip(b_full, b_zero, delta_ret):
        y_top = max(bf.get_height(), bz.get_height()) + 1.0
        x0 = bf.get_x() + bf.get_width() / 2
        x1 = bz.get_x() + bz.get_width() / 2
        ax_d.plot([x0, x1], [y_top, y_top], color="#6E6E6E", linewidth=0.8)
        ax_d.text((x0 + x1) / 2, y_top + 0.35, f"+{d:.2f}", ha="center", va="bottom", fontsize=9.2, color=THEME["primary"])
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{e:.2f}" for e in protect_eps])
    ax_d.set_xlabel("Perturbation budget ε")
    ax_d.set_ylabel("Retained F1 (%)")
    ax_d.set_ylim(min(min(full_ret), min(zero_ret)) - 3.0, max(max(full_ret), max(zero_ret)) + 4.5)
    ax_d.set_title("D. Quantum Branch Contribution", pad=10)
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9.1)
    ax_d.text(0.98, 0.94, "Positive deltas at all ε", transform=ax_d.transAxes, ha="right", va="top", fontsize=9.6, color=THEME["positive"])
    _polish_axis(ax_d)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.88, wspace=0.18, hspace=0.32)
    fig.suptitle("Threat-Facing Robustness Evaluation", fontsize=17.4, y=0.975)
    fig.text(
        0.5,
        0.946,
        "Forgery resistance, adversarial retention, feature sensitivity, and quantum contribution consistently indicate security robustness.",
        ha="center",
        va="top",
        fontsize=10.5,
        color=COL["neutral"],
    )
    save_figure(fig, out / "experiments_story_01_attack_pipeline", dpi=650)
    plt.close(fig)


def plot_story_generalization_mechanism(out: Path) -> None:
    """Figure 2: mechanism-facing evidence (3-panel cleaner layout)."""
    temporal = load_temporal_generalization()["results"]
    topo = load_topological_invariance()
    ablation = load_ablation(prefer_current=True)

    gaps = [int(r["gap_weeks"]) for r in temporal]
    train_f1 = [float(r["train_f1"]) * 100.0 for r in temporal]
    test_f1 = [float(r["test_f1"]) * 100.0 for r in temporal]
    retention = [float(r["retention_rate"]) for r in temporal]

    iso = np.asarray(topo["isomorphic_pairs"]["similarities"], dtype=float)
    non = np.asarray(topo["non_isomorphic_pairs"]["similarities"], dtype=float)
    sep = float(topo["verification"]["separation"])

    rows = {str(r["model"]): r for _, r in ablation.iterrows()}
    metrics = ["precision", "recall", "f1", "auc"]
    metric_names = ["Precision", "Recall", "F1", "AUC"]
    best_single = [max(float(rows["Classical"][m]), float(rows["Quantum"][m])) * 100.0 for m in metrics]
    hybrid = [float(rows["Hybrid"][m]) * 100.0 for m in metrics]
    uplift = [h - b for h, b in zip(hybrid, best_single)]
    hybrid_gain = uplift[2]
    improved_count = int(sum(1 for u in uplift if u > 0.0))
    near_parity_count = int(sum(1 for u in uplift if abs(u) <= 1.0))

    fig = plt.figure(figsize=(16.3, 8.9), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.17, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (A) Temporal transfer stability.
    ax_a2 = ax_a.twinx()
    ax_a2.fill_between(gaps, retention, [min(retention)] * len(gaps), color=THEME["soft_warm"], alpha=0.55, zorder=0)
    ax_a.plot(gaps, train_f1, marker="o", color=THEME["neutral"], linewidth=2.3)
    ax_a.plot(gaps, test_f1, marker="s", color=THEME["primary"], linewidth=2.3)
    ax_a2.plot(gaps, retention, marker="^", color=THEME["tertiary"], linewidth=2.1, linestyle="--")
    ax_a.set_xlabel("Temporal gap (weeks)")
    ax_a.set_ylabel("F1 (%)")
    ax_a2.set_ylabel("Retention (%)")
    ax_a.set_xlim(min(gaps) - 0.55, max(gaps) + 1.0)
    ax_a.set_ylim(min(test_f1) - 4.2, max(train_f1) + 3.5)
    ax_a2.set_ylim(min(retention) - 6.0, 100.0)
    ax_a2.grid(False)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.tick_params(axis="y", labelsize=9.3, colors=THEME["neutral"])
    ax_a.set_title("A. Temporal Generalization Stability", pad=10)
    l1, = ax_a.plot([], [], marker="o", color=THEME["neutral"], linewidth=2.3, label="Train")
    l2, = ax_a.plot([], [], marker="s", color=THEME["primary"], linewidth=2.3, label="Test")
    l3, = ax_a2.plot([], [], marker="^", color=THEME["tertiary"], linewidth=2.1, linestyle="--", label="Retention")
    ax_a.legend(handles=[l1, l2, l3], frameon=False, loc="upper right", ncol=3, fontsize=8.7)
    ax_a.text(
        0.05,
        0.65,
        f"Average retention: {np.mean(retention):.1f}%",
        transform=ax_a.transAxes,
        fontsize=9.8,
        color=THEME["primary"],
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.72},
    )
    _polish_axis(ax_a)

    # (B) Topological separation.
    vp = ax_b.violinplot([iso, non], positions=[1, 2], widths=0.5, showmeans=True, showextrema=False, showmedians=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor([THEME["primary"], THEME["secondary"]][i])
        body.set_edgecolor(COL["text"])
        body.set_alpha(0.38)
    vp["cmeans"].set_color("#2F2F2F")
    vp["cmeans"].set_linewidth(1.1)
    jitter_iso = np.linspace(-0.08, 0.08, len(iso))
    jitter_non = np.linspace(-0.08, 0.08, len(non))
    ax_b.scatter(1 + jitter_iso, iso, s=15, color=THEME["primary"], alpha=0.55, zorder=3)
    ax_b.scatter(2 + jitter_non, non, s=15, color=THEME["secondary"], alpha=0.55, zorder=3)
    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(["Isomorphic", "Non-isomorphic"])
    ax_b.set_ylabel("Cosine similarity")
    ax_b.set_ylim(0.93, 1.0025)
    ax_b.set_title("B. Topological Similarity Separation", pad=10)
    ax_b.text(0.03, 0.08, f"Separation score: {sep:.4f}", transform=ax_b.transAxes, fontsize=9.8, color=THEME["primary"])
    _polish_axis(ax_b)

    # (C) Hybrid uplift across metrics.
    y = np.arange(len(metrics))
    for i in range(len(metrics)):
        ax_c.plot([best_single[i], hybrid[i]], [y[i], y[i]], color="#B8B8B8", linewidth=2.2, zorder=1)
    ax_c.scatter(best_single, y, s=72, color=THEME["neutral"], edgecolors=COL["text"], linewidths=0.85, zorder=3, label="Best single branch")
    ax_c.scatter(hybrid, y, s=78, color=THEME["primary"], edgecolors=COL["text"], linewidths=0.85, zorder=3, label="Hybrid")
    for i, up in enumerate(uplift):
        sign = "+" if up >= 0 else ""
        label = f"{sign}{up:.1f}" if abs(up) >= 1.0 else f"{sign}{up:.1f} (near parity)"
        color = THEME["positive"] if up > 0 else THEME["neutral"]
        x_text = hybrid[i] + 1.2 if up >= 0 else hybrid[i] - 12.5
        ax_c.text(x_text, y[i], label, va="center", fontsize=9.3, color=color)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(metric_names)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Score (%)")
    ax_c.set_xlim(min(best_single) - 6.0, max(hybrid) + 14.0)
    ax_c.set_title("C. Hybrid Uplift Across Metrics", pad=10)
    ax_c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.18, 1.01), ncol=2)
    ax_c.text(
        0.02,
        0.08,
        f"F1 uplift vs best single: +{hybrid_gain:.1f} pts; "
        f"{improved_count}/4 metrics improved ({near_parity_count}/4 near parity)",
        transform=ax_c.transAxes,
        fontsize=9.8,
        color=THEME["primary"],
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.72},
    )
    _polish_axis(ax_c)

    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.10, top=0.88, wspace=0.17, hspace=0.31)
    fig.suptitle("Generalization and Structural Mechanism Analysis", fontsize=17.3, y=0.975)
    fig.text(
        0.5,
        0.946,
        "Temporal transfer, topology-aware representation, and metric-level hybrid uplift provide coherent mechanism evidence.",
        ha="center",
        va="top",
        fontsize=10.5,
        color=COL["neutral"],
    )
    save_figure(fig, out / "experiments_story_02_generalization_mechanism", dpi=650)
    plt.close(fig)


def plot_story_evidence_constellation(out: Path) -> None:
    """Figure 3: cross-experiment synthesis map (7 experiments, low-overlap layout)."""
    forgery = load_feature_forgery()
    adv = load_adversarial()["results"]["qgad"]
    sens = load_feature_sensitivity()["results"]
    protect = load_quantum_protection()["results"]
    temporal = load_temporal_generalization()["results"]
    topo = load_topological_invariance()
    ablation = load_ablation(prefer_current=True)

    q_clean = float(forgery["qgad"]["baseline_recall"])
    q_forged = float(forgery["qgad"]["forged_recall"])
    q_retained = q_forged / max(q_clean, 1e-9)
    forgery_gap = float(forgery["advantage"]["drop_gap_xgb_minus_qgad"])

    adv_f1_ret = float(adv["epsilon_0.1"]["f1"]) / max(float(adv["clean_f1"]), 1e-9)
    adv_auc_ret = float(adv["epsilon_0.1"]["auc"]) / max(float(adv["clean_auc"]), 1e-9)

    sens_clean_f1 = float(sens["clean"]["f1"])
    sens_clean_auc = float(sens["clean"]["auc"])
    sens_q_f1 = float(sens["Quantum_Features_epsilon_0.1"]["f1"])
    sens_q_auc = float(sens["Quantum_Features_epsilon_0.1"]["auc"])
    sens_c_f1 = float(sens["Classical_Features_epsilon_0.1"]["f1"])
    sens_adv = sens_q_f1 - sens_c_f1
    sens_f1_ret = sens_q_f1 / max(sens_clean_f1, 1e-9)
    sens_auc_ret = sens_q_auc / max(sens_clean_auc, 1e-9)
    sens_impact = np.clip(0.60 * sens_f1_ret + 0.40 * (0.5 + 5.0 * sens_adv), 0.0, 1.0)

    full_01 = float(protect["Q-GAD (Full)"]["epsilon_0.1"]["retained_percent"])
    zero_01 = float(protect["Q-GAD (ZeroQuantum)"]["epsilon_0.1"]["retained_percent"])
    quantum_delta = full_01 - zero_01

    temporal_ret = float(np.mean([float(r["retention_rate"]) for r in temporal])) / 100.0
    temporal_consistency = 1.0 - float(np.std([float(r["test_f1"]) for r in temporal])) / max(float(np.mean([float(r["test_f1"]) for r in temporal])), 1e-9)

    sep = float(topo["verification"]["separation"])
    topo_stability = float(topo["isomorphic_pairs"]["mean_similarity"])

    rows = {str(r["model"]): r for _, r in ablation.iterrows()}
    f1_cls = float(rows["Classical"]["f1"])
    f1_q = float(rows["Quantum"]["f1"])
    f1_h = float(rows["Hybrid"]["f1"])
    hybrid_gain = f1_h - max(f1_cls, f1_q)
    hybrid_auc = float(rows["Hybrid"]["auc"])

    # Dedicated colors from a single theme map to keep palette consistent.
    color_map = {
        "Forgery Gap": THEME["primary"],
        "Adversarial Retention": THEME["secondary"],
        "Feature Sensitivity": THEME["sensitivity"],
        "Quantum Margin": THEME["quantum"],
        "Temporal Stability": THEME["tertiary"],
        "Topology Invariance": THEME["topology"],
        "Hybrid Gain": THEME["positive"],
    }

    points = [
        {
            "name": "Forgery Gap",
            "short": "Forgery",
            "x": np.clip(forgery_gap, 0.0, 1.0),
            "y": np.clip(q_retained, 0.0, 1.0),
            "size": 1120,
            "color": color_map["Forgery Gap"],
            "note": f"drop-gap +{forgery_gap * 100:.1f} pts",
        },
        {
            "name": "Adversarial Retention",
            "short": "Adversarial",
            "x": np.clip(adv_f1_ret, 0.0, 1.0),
            "y": np.clip(adv_auc_ret, 0.0, 1.0),
            "size": 1020,
            "color": color_map["Adversarial Retention"],
            "note": f"F1 keep {adv_f1_ret * 100:.1f}%",
        },
        {
            "name": "Feature Sensitivity",
            "short": "Sensitivity",
            "x": np.clip(sens_impact, 0.0, 1.0),
            "y": np.clip(sens_auc_ret, 0.0, 1.0),
            "size": 860,
            "color": color_map["Feature Sensitivity"],
            "note": f"Q-C @eps0.1 = {sens_adv * 100:.1f} pts",
        },
        {
            "name": "Quantum Margin",
            "short": "Quantum",
            "x": np.clip(quantum_delta / 5.0, 0.0, 1.0),
            "y": np.clip(full_01 / 100.0, 0.0, 1.0),
            "size": 780,
            "color": color_map["Quantum Margin"],
            "note": f"+{quantum_delta:.2f} pts @eps0.1",
        },
        {
            "name": "Temporal Stability",
            "short": "Temporal",
            "x": np.clip(temporal_ret, 0.0, 1.0),
            "y": np.clip(temporal_consistency, 0.0, 1.0),
            "size": 900,
            "color": color_map["Temporal Stability"],
            "note": f"retention {temporal_ret * 100:.1f}%",
        },
        {
            "name": "Topology Invariance",
            "short": "Topology",
            "x": np.clip(sep * 120.0, 0.0, 1.0),
            "y": np.clip(topo_stability, 0.0, 1.0),
            "size": 760,
            "color": color_map["Topology Invariance"],
            "note": f"separation {sep:.4f}",
        },
        {
            "name": "Hybrid Gain",
            "short": "Hybrid",
            "x": np.clip(hybrid_gain * 3.5, 0.0, 1.0),
            "y": np.clip(hybrid_auc / 0.90, 0.0, 1.0),
            "size": 1080,
            "color": color_map["Hybrid Gain"],
            "note": f"F1 uplift +{hybrid_gain * 100:.1f} pts",
        },
    ]

    for p in points:
        p["integrated"] = 0.58 * float(p["x"]) + 0.42 * float(p["y"])

    fig = plt.figure(figsize=(15.8, 8.8), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("white")
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.56, 1.08)
    ax.set_title("A. Impact-Stability Constellation", pad=8)
    ax.set_xlabel("Security impact score (normalized)")
    ax.set_ylabel("Stability confidence score (normalized)")
    ax.grid(True, alpha=0.16)
    ax.axvline(0.80, linestyle="--", linewidth=0.85, color="#B7B7B7", alpha=0.80, zorder=0)
    ax.axhline(0.80, linestyle="--", linewidth=0.85, color="#B7B7B7", alpha=0.80, zorder=0)

    story_order = [
        "Quantum Margin",
        "Adversarial Retention",
        "Feature Sensitivity",
        "Forgery Gap",
        "Temporal Stability",
        "Topology Invariance",
        "Hybrid Gain",
    ]
    lookup = {p["name"]: p for p in points}
    for i in range(len(story_order) - 1):
        p0 = lookup[story_order[i]]
        p1 = lookup[story_order[i + 1]]
        ax.annotate(
            "",
            xy=(p1["x"], p1["y"]),
            xytext=(p0["x"], p0["y"]),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "linestyle": "--", "color": "#8A8A8A", "alpha": 0.75},
            zorder=1,
        )

    # Slight visual offsets for tightly clustered points (readability only).
    display_offset = {
        "Adversarial Retention": (-0.004, 0.002),
        "Feature Sensitivity": (0.005, 0.010),
        "Forgery Gap": (0.001, -0.003),
    }

    # Numbered bubbles avoid dense text overlaps in the left panel.
    index_map = {name: idx + 1 for idx, name in enumerate(story_order)}
    for p in points:
        dx, dy = display_offset.get(p["name"], (0.0, 0.0))
        x_plot = float(p["x"]) + dx
        y_plot = float(p["y"]) + dy
        ax.scatter(
            x_plot,
            y_plot,
            s=p["size"],
            c=p["color"],
            alpha=0.86,
            edgecolors=COL["text"],
            linewidths=1.0,
            zorder=3,
        )
        ax.text(
            x_plot,
            y_plot,
            str(index_map[p["name"]]),
            ha="center",
            va="center",
            fontsize=11.0,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    ax.text(
        0.558,
        0.566,
        "Numbered bubbles encode seven experiments; arrows denote the evidence narrative path.",
        fontsize=9.1,
        color=COL["neutral"],
        ha="left",
    )
    _polish_axis(ax)

    ranked = sorted(points, key=lambda d: float(d["integrated"]), reverse=True)
    yy = np.arange(len(ranked))
    int_scores = [100.0 * float(p["integrated"]) for p in ranked]
    impact_scores = [100.0 * float(p["x"]) for p in ranked]
    stability_scores = [100.0 * float(p["y"]) for p in ranked]
    colors = [p["color"] for p in ranked]

    ax_r.barh(yy, int_scores, color=colors, alpha=0.86, edgecolor=COL["text"], linewidth=0.7, height=0.54, label="Integrated score")
    ax_r.scatter(impact_scores, yy - 0.11, s=40, marker="D", color="#3F3F3F", zorder=3, label="Impact component")
    ax_r.scatter(stability_scores, yy + 0.11, s=40, marker="o", facecolors="white", edgecolors="#3F3F3F", linewidths=0.9, zorder=3, label="Stability component")
    score_col_x = 105.9
    for i, p in enumerate(ranked):
        ax_r.text(56.1, i, p["note"], va="center", ha="left", fontsize=8.15, color=COL["text"])
        ax_r.text(
            score_col_x,
            i,
            f"{int_scores[i]:.1f}",
            va="center",
            ha="right",
            fontsize=9.0,
            color=COL["text"],
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#B6B6B6", "linewidth": 0.6},
        )
    ax_r.set_yticks(yy)
    ax_r.set_yticklabels([f"{index_map[p['name']]}. {p['short']}" for p in ranked])
    ax_r.invert_yaxis()
    ax_r.set_xlim(55.6, 106.3)
    ax_r.set_xlabel("Integrated evidence score (%)")
    ax_r.set_title("B. Evidence Ranking and Metrics", pad=8)
    ax_r.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.77, -0.005))
    _polish_axis(ax_r)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.855, wspace=0.18)
    fig.suptitle("Integrated Cross-Experiment Evidence Map", fontsize=16.9, y=0.975)
    fig.text(
        0.5,
        0.943,
        "Seven experiment threads are unified into a single impact-stability narrative with explicit quantitative ranking.",
        ha="center",
        va="top",
        fontsize=10.3,
        color=COL["neutral"],
    )
    fig.text(
        0.5,
        0.919,
        "Integrated score = 0.58 x Impact + 0.42 x Stability; "
        "scales: Δquantum/5, separation x120, gain x3.5, "
        "sensitivity impact = 0.6*F1ret + 0.4*(0.5 + 5*(Q-C)).",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COL["neutral"],
    )
    save_figure(fig, out / "experiments_story_03_evidence_constellation", dpi=650)
    plt.close(fig)


def main() -> None:
    out = ROOT / "outputs" / "visualizations" / "experiments"
    _clear_output_dir(out)
    plot_story_attack_pipeline(out)
    plot_story_generalization_mechanism(out)
    plot_story_evidence_constellation(out)
    print(f"[OK] Experiment narrative figures saved to {out}")


if __name__ == "__main__":
    main()
