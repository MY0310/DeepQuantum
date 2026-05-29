"""Single-story publication figure for Q-GAD vs GNN baselines."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from visualization.data_sources import ROOT, load_model_comparison
    from visualization.mpl_setup import (
        apply_default_style,
        get_plot_modules,
        journal_palette,
        save_figure,
        set_axis_style,
    )
else:
    from .data_sources import ROOT, load_model_comparison
    from .mpl_setup import (
        apply_default_style,
        get_plot_modules,
        journal_palette,
        save_figure,
        set_axis_style,
    )

plt = get_plot_modules(ROOT)
apply_default_style(plt)
COL = journal_palette()


def _safe_pct(numer: float, denom: float) -> float:
    return float(numer / denom * 100.0) if abs(denom) > 1e-12 else 0.0


def _to_percent(value: float, metric: str) -> float:
    if metric == "accuracy":
        return float(value)
    return float(value) * 100.0


def generate_model_comparison() -> Path:
    out = ROOT / "outputs" / "visualizations" / "model_comparison"
    out.mkdir(parents=True, exist_ok=True)

    df = load_model_comparison()
    df.to_csv(out / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    q = df[df["method"] == "Q-GAD"].iloc[0]
    g = df[df["method"] != "Q-GAD"].copy()
    gnn_mean = g[["accuracy", "precision", "recall", "f1", "auc", "ap", "params"]].mean()

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("auc", "AUC"),
        ("ap", "AP"),
    ]

    gains = np.array(
        [
            _safe_pct(float(q[key]) - float(gnn_mean[key]), float(gnn_mean[key]))
            for key, _ in metric_specs
        ],
        dtype=float,
    )
    wins_vs_mean = int(np.sum(gains > 0.0))
    wins_vs_best = int(sum(float(q[k]) >= float(g[k].max()) - 1e-12 for k, _ in metric_specs))
    param_reduction = _safe_pct(float(gnn_mean["params"]) - float(q["params"]), float(gnn_mean["params"]))
    top1_metrics = 0
    for col, bigger_better in [
        ("accuracy", True),
        ("precision", True),
        ("recall", True),
        ("f1", True),
        ("auc", True),
        ("ap", True),
        ("params", False),
    ]:
        series = df[col].to_numpy(dtype=float)
        qv = float(q[col])
        if bigger_better and qv >= float(series.max()) - 1e-12:
            top1_metrics += 1
        if (not bigger_better) and qv <= float(series.min()) + 1e-12:
            top1_metrics += 1

    methods = df["method"].tolist()
    params_k = df["params"].to_numpy(dtype=float) / 1000.0
    f1_pct = df["f1"].to_numpy(dtype=float) * 100.0
    auc_vals = df["auc"].to_numpy(dtype=float)
    bubble_sizes = np.interp(auc_vals, (auc_vals.min(), auc_vals.max()), (230.0, 600.0))

    fig = plt.figure(figsize=(15.4, 8.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.05], height_ratios=[1.0, 1.0])
    ax_dom = fig.add_subplot(gs[:, :2])
    ax_eff = fig.add_subplot(gs[0, 2])
    ax_note = fig.add_subplot(gs[1, 2])

    y = np.arange(len(metric_specs))
    legend_added = False
    x_min_candidates = []
    x_max_candidates = []
    for i, (key, label) in enumerate(metric_specs):
        vals = g[key].to_numpy(dtype=float)
        vals_pct = np.array([_to_percent(v, key) for v in vals], dtype=float)
        q_pct = _to_percent(float(q[key]), key)
        mean_pct = float(vals_pct.mean())
        lo = float(vals_pct.min())
        hi = float(vals_pct.max())
        gain = _safe_pct(float(q[key]) - float(g[key].mean()), float(g[key].mean()))

        x_min_candidates.extend([lo, mean_pct, q_pct])
        x_max_candidates.extend([lo, hi, mean_pct, q_pct])

        # Use a simple line segment instead of hlines collection to avoid
        # intermittent Win+Matplotlib native crashes in transform handling.
        ax_dom.plot(
            [lo, hi],
            [i, i],
            color=COL["gnn_light"],
            linewidth=10,
            alpha=0.30,
            zorder=1,
            label="GNN range (min-max)" if not legend_added else None,
        )
        ax_dom.plot(
            mean_pct,
            i,
            marker="D",
            markersize=6.2,
            color=COL["gnn"],
            linestyle="None",
            zorder=3,
            label="GNN mean" if not legend_added else None,
        )
        ax_dom.plot(
            q_pct,
            i,
            marker="o",
            markersize=8.2,
            color=COL["qgad"],
            markeredgecolor=COL["text"],
            markeredgewidth=0.8,
            linestyle="None",
            zorder=4,
            label="Q-GAD" if not legend_added else None,
        )
        ax_dom.annotate(
            "",
            xy=(q_pct, i),
            xytext=(mean_pct, i),
            arrowprops={"arrowstyle": "->", "color": COL["qgad"], "lw": 1.2, "alpha": 0.95},
            zorder=2,
        )
        ax_dom.text(
            max(q_pct, hi) + 0.42,
            i,
            f"{gain:+.2f}%",
            va="center",
            ha="left",
            fontsize=13.6,
            color=COL["text"],
        )
        legend_added = True

    ax_dom.set_yticks(y)
    ax_dom.set_yticklabels([label for _, label in metric_specs])
    ax_dom.invert_yaxis()
    ax_dom.set_xlabel("Score (%)")
    ax_dom.set_title("Dominance Map: Q-GAD against GNN baseline distribution", pad=8)
    x_left = float(min(x_min_candidates) - 1.4)
    x_right = float(max(x_max_candidates) + 5.2)
    ax_dom.set_xlim(x_left, x_right)
    ax_dom.legend(loc="lower right", frameon=False, ncol=3, handletextpad=0.5, columnspacing=1.4)
    ax_dom.text(
        x_left + 0.2,
        -0.75,
        "Right-side labels show relative gain over GNN mean",
        fontsize=9,
        color=COL["neutral"],
    )
    set_axis_style(ax_dom)

    colors = [COL["qgad"] if m == "Q-GAD" else COL["gnn_light"] for m in methods]
    markers = ["o" if m != "Q-GAD" else "*" for m in methods]
    for i, method in enumerate(methods):
        ax_eff.scatter(
            params_k[i],
            f1_pct[i],
            s=bubble_sizes[i],
            c=colors[i],
            marker=markers[i],
            edgecolors=COL["text"],
            linewidths=1.4 if method == "Q-GAD" else 0.8,
            alpha=0.95,
            zorder=3 if method == "Q-GAD" else 2,
        )
        ax_eff.annotate(
            method,
            (params_k[i], f1_pct[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
        )
    ax_eff.annotate(
        "Better zone",
        xy=(params_k.min() + 1.2, f1_pct.max() - 0.1),
        xytext=(params_k.min() + 6.3, f1_pct.max() - 0.95),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COL["text"]},
        fontsize=9,
    )
    ax_eff.set_xlabel("Parameters (thousand)")
    ax_eff.set_ylabel("F1 (%)")
    ax_eff.set_title("Efficiency-Quality Frontier")
    set_axis_style(ax_eff)

    ax_note.axis("off")
    cards = [
        ("Predictive Lead", f"{wins_vs_mean}/6 metrics beat GNN mean"),
        ("SOTA Coverage", f"{wins_vs_best}/6 metrics beat best single GNN"),
        ("Model Compactness", f"{param_reduction:.2f}% fewer params vs GNN mean"),
        ("Global Rank", f"Top-1 on {top1_metrics}/7 tracked indicators"),
    ]
    y0 = 0.95
    for title, text in cards:
        ax_note.text(
            0.03,
            y0,
            title,
            fontsize=12,
            fontweight="bold",
            color=COL["text"],
            ha="left",
            va="top",
        )
        ax_note.text(
            0.03,
            y0 - 0.09,
            text,
            fontsize=11,
            color=COL["text"],
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#F6F3EE",
                "edgecolor": "#C4BBAF",
                "linewidth": 0.8,
            },
        )
        y0 -= 0.24

    fig.suptitle("Q-GAD vs GNN Baselines: Single-Figure Narrative Comparison", fontsize=15, y=1.01)
    fig.text(
        0.5,
        0.98,
        "Metric dominance, efficiency frontier, and concise scorecard",
        ha="center",
        va="top",
        fontsize=10.5,
        color=COL["neutral"],
    )
    fig.text(
        0.706,
        0.016,
        "Baselines: GCN, GAT, GraphSAGE, GIN   |   Dataset: Elliptic++ test periods 35-49",
        fontsize=9.0,
        color=COL["neutral"],
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.07, top=0.93, wspace=0.28, hspace=0.36)

    save_figure(fig, out / "model_comparison_story", dpi=650)
    plt.close(fig)

    print(f"[OK] Model comparison story figure saved to {out}")
    return out


def main() -> None:
    generate_model_comparison()


if __name__ == "__main__":
    main()
