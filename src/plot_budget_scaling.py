#!/usr/bin/env python3
"""Plot budget scaling curves from v2 rebuttal experiments."""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib", file=sys.stderr)
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent.parent / "justx" / "paper" / "rebuttal" / "icml26" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = {
    "google/gemini-2.0-flash-001": "Gemini Flash",
    "openai/gpt-4o": "GPT-4o",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
}

# Method styling
STYLE = {
    "PLeak":        {"color": "#d62728", "ls": "--", "marker": "v"},
    "Raccoon":      {"color": "#ff7f0e", "ls": "--", "marker": "s"},
    "Zhang-et-al":  {"color": "#9467bd", "ls": "--", "marker": "D"},
    "Bare-Agent":   {"color": "#7f7f7f", "ls": ":",  "marker": "o"},
    "L14-Only":     {"color": "#bcbd22", "ls": ":",  "marker": "^"},
    "Random-UCB":   {"color": "#17becf", "ls": ":",  "marker": "<"},
    "JustAsk-Full": {"color": "#2ca02c", "ls": "-",  "marker": "*", "lw": 2.5, "ms": 12},
}

LABEL = {
    "PLeak": "PLeak",
    "Raccoon": "Raccoon",
    "Zhang-et-al": "Zhang et al.",
    "Bare-Agent": "Bare-Agent",
    "L14-Only": "L14-Only",
    "Random-UCB": "Random-UCB",
    "JustAsk-Full": "JustAsk (Ours)",
}


def load_v2():
    results = {}
    for f in sorted(DATA_DIR.glob("rebuttal_v2_*.json")):
        data = json.load(open(f))
        short = MODEL_SHORT.get(data["model"], data["model"])
        results[short] = data
    return results


def plot_model(ax, model_name, data):
    budgets = data["budgets"]
    results = data["results"]
    methods = data["methods"]

    for method in methods:
        st = STYLE.get(method, {})
        vals = []
        for b in budgets:
            bk = str(b)
            if bk in results and method in results[bk]:
                vals.append(results[bk][method]["summary"]["best_sim"])
            else:
                vals.append(None)
        valid_b = [b for b, v in zip(budgets, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        if not valid_v:
            continue
        ax.plot(
            valid_b, valid_v,
            label=LABEL.get(method, method),
            color=st.get("color", "gray"),
            linestyle=st.get("ls", "-"),
            marker=st.get("marker", "o"),
            linewidth=st.get("lw", 1.5),
            markersize=st.get("ms", 7),
        )

    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Budget (B)")
    ax.set_ylabel("Best Sim-GT")
    ax.set_xticks(budgets)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)


def main():
    v2 = load_v2()
    if not v2:
        print("No v2 data found yet.")
        return

    # Plot hard models side by side
    hard_models = ["Claude Sonnet 4", "GPT-4o"]
    available_hard = [m for m in hard_models if m in v2]

    if available_hard:
        fig, axes = plt.subplots(1, len(available_hard), figsize=(6 * len(available_hard), 5))
        if len(available_hard) == 1:
            axes = [axes]
        for ax, model in zip(axes, available_hard):
            plot_model(ax, model, v2[model])
        axes[-1].legend(loc="lower right", fontsize=8)
        fig.suptitle("Budget Scaling on Defended Models (Best Sim-GT)", fontsize=14, y=1.02)
        fig.tight_layout()
        path = OUT_DIR / "budget_scaling_hard.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
        plt.close()

    # Plot all 4 models in a 2x2 grid
    all_models = ["DeepSeek V3", "Gemini Flash", "Claude Sonnet 4", "GPT-4o"]
    available_all = [m for m in all_models if m in v2]
    if len(available_all) >= 2:
        ncols = 2
        nrows = (len(available_all) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, model in enumerate(available_all):
            plot_model(axes[i], model, v2[model])
        for j in range(len(available_all), len(axes)):
            axes[j].set_visible(False)
        axes[-1].legend(loc="lower right", fontsize=8)
        fig.suptitle("Budget Scaling Across All Models (Best Sim-GT)", fontsize=14, y=1.02)
        fig.tight_layout()
        path = OUT_DIR / "budget_scaling_all.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
        plt.close()


if __name__ == "__main__":
    main()
