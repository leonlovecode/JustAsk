#!/usr/bin/env python3
"""Generate rebuttal tables and figures from experiment results."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TABLE_DIR = Path(__file__).parent.parent.parent / "justx" / "paper" / "rebuttal" / "icml26" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_v1_results():
    """Load v1 results from JSON files."""
    models = {}
    for f in sorted(DATA_DIR.glob("rebuttal_results_*.json")):
        data = json.load(open(f))
        model = data["model"]
        # Shorten model names
        short = {
            "google/gemini-2.0-flash-001": "Gemini Flash",
            "openai/gpt-4o": "GPT-4o",
            "anthropic/claude-sonnet-4": "Claude S4",
            "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
        }.get(model, model)
        models[short] = {r["name"]: r for r in data["results"]}
    return models


MODEL_SHORT = {
    "google/gemini-2.0-flash-001": "Gemini Flash",
    "openai/gpt-4o": "GPT-4o",
    "anthropic/claude-sonnet-4": "Claude S4",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
}


def load_v2_results():
    """Load v2 budget-scaling results (all models)."""
    results = {}
    for f in sorted(DATA_DIR.glob("rebuttal_v2_*.json")):
        data = json.load(open(f))
        model = data["model"]
        short = MODEL_SHORT.get(model, model)
        results[short] = data
    return results


def v2_at_budget(v2_data, budget):
    """Extract a flat {model: {method: summary}} dict from v2 data at a given budget."""
    out = {}
    bk = str(budget)
    for model, data in v2_data.items():
        results = data.get("results", {})
        if bk in results:
            out[model] = {m: r["summary"] for m, r in results[bk].items()}
    return out


def table_r1_baseline_comparison(models):
    """Table R1: Baselines vs JustAsk across models."""
    # Order models by difficulty (easy -> hard)
    model_order = ["DeepSeek V3", "Gemini Flash", "Claude S4", "GPT-4o"]
    methods = ["PLeak", "Raccoon", "Zhang-et-al", "JustAsk-Full"]
    method_labels = {
        "PLeak": "PLeak [1]",
        "Raccoon": "Raccoon [2]",
        "Zhang-et-al": "Zhang et al. [3]",
        "JustAsk-Full": "JustAsk (Ours)",
    }

    lines = []
    lines.append("## Table R1: Baseline Comparison (Best Sim-GT, B=3)")
    lines.append("")

    # Header
    header = "| Method              |"
    sep = "|--------------------|"
    for m in model_order:
        header += f" {m:>12} |"
        sep += "-------------|"
    header += "     Mean |"
    sep += "----------|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        label = method_labels[method]
        row = f"| {label:<19} |"
        vals = []
        for model in model_order:
            if model in models and method in models[model]:
                v = models[model][method]["best_sim"]
                vals.append(v)
                row += f"     {v:>6.3f} |"
            else:
                row += "        N/A |"
        mean = sum(vals) / len(vals) if vals else 0
        row += f"  {mean:.3f} |"
        lines.append(row)

    lines.append("")

    # Add drop analysis
    lines.append("### Degradation from Easiest to Hardest Model")
    lines.append("")
    lines.append("| Method              | DeepSeek V3 | GPT-4o  | Drop   |")
    lines.append("|--------------------|-------------|---------|--------|")
    for method in methods:
        label = method_labels[method]
        easy = models.get("DeepSeek V3", {}).get(method, {}).get("best_sim", 0)
        hard = models.get("GPT-4o", {}).get(method, {}).get("best_sim", 0)
        drop = ((easy - hard) / easy * 100) if easy > 0 else 0
        lines.append(f"| {label:<19} | {easy:>11.3f} | {hard:>7.3f} | {drop:>5.1f}% |")

    return "\n".join(lines)


def table_r2_ablation(models):
    """Table R2: Framework ablation across models."""
    model_order = ["DeepSeek V3", "Gemini Flash", "Claude S4", "GPT-4o"]
    variants = ["Bare-Agent", "L14-Only", "Random-UCB", "JustAsk-Full"]
    descriptions = {
        "Bare-Agent": "Plain LLM (no framework)",
        "L14-Only": "+ Best single skill (L14)",
        "Random-UCB": "+ Full skill taxonomy (random)",
        "JustAsk-Full": "+ UCB-guided + multi-turn",
    }

    lines = []
    lines.append("## Table R2: Framework Ablation (Best Sim-GT, B=3)")
    lines.append("")

    header = "| Variant             |"
    sep = "|--------------------|"
    for m in model_order:
        header += f" {m:>12} |"
        sep += "-------------|"
    header += "     Mean |"
    sep += "----------|"
    lines.append(header)
    lines.append(sep)

    for var in variants:
        desc = descriptions[var]
        row = f"| {desc:<19} |"
        vals = []
        for model in model_order:
            if model in models and var in models[model]:
                v = models[model][var]["best_sim"]
                vals.append(v)
                row += f"     {v:>6.3f} |"
            else:
                row += "        N/A |"
        mean = sum(vals) / len(vals) if vals else 0
        row += f"  {mean:.3f} |"
        lines.append(row)

    lines.append("")
    lines.append("**Reading guide**: Each row adds one framework component.")
    lines.append("The improvement from Bare-Agent to JustAsk-Full shows the framework's contribution.")

    return "\n".join(lines)


def table_v2_budget_scaling(v2_data):
    """Generate budget scaling table from v2 results — per model."""
    if not v2_data:
        return "## Budget Scaling: [pending v2 results]"

    lines = []
    for model, data in v2_data.items():
        budgets = data["budgets"]
        methods = data["methods"]
        results = data["results"]

        lines.append(f"## Budget Scaling: {model}")
        lines.append("")

        header = "| Method              |"
        sep = "|--------------------|"
        for b in budgets:
            header += f" B={b:<2} Best | B={b:<2} Avg |"
            sep += "----------|----------|"
        lines.append(header)
        lines.append(sep)

        for method in methods:
            row = f"| {method:<19} |"
            for b in budgets:
                bk = str(b)
                if bk in results and method in results[bk]:
                    s = results[bk][method]["summary"]
                    row += f"  {s['best_sim']:>7.3f} | {s['avg_sim']:>8.3f} |"
                else:
                    row += "      N/A |      N/A |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def table_v2_cross_model(v2_data, budget):
    """Cross-model comparison at a specific budget from v2 data."""
    if not v2_data:
        return f"## Cross-Model at B={budget}: [pending v2 results]"

    model_order = ["DeepSeek V3", "Gemini Flash", "Claude S4", "GPT-4o"]
    methods = ["PLeak", "Raccoon", "Zhang-et-al", "Bare-Agent", "L14-Only", "Random-UCB", "JustAsk-Full"]
    method_labels = {
        "PLeak": "PLeak [1]",
        "Raccoon": "Raccoon [2]",
        "Zhang-et-al": "Zhang et al. [3]",
        "Bare-Agent": "Bare-Agent",
        "L14-Only": "L14-Only",
        "Random-UCB": "Random-UCB",
        "JustAsk-Full": "JustAsk (Ours)",
    }

    bk = str(budget)
    lines = []
    lines.append(f"## Cross-Model Comparison (Best Sim-GT, B={budget})")
    lines.append("")

    header = "| Method              |"
    sep = "|--------------------|"
    for m in model_order:
        header += f" {m:>12} |"
        sep += "-------------|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        label = method_labels.get(method, method)
        row = f"| {label:<19} |"
        for model in model_order:
            if model in v2_data:
                results = v2_data[model].get("results", {})
                if bk in results and method in results[bk]:
                    v = results[bk][method]["summary"]["best_sim"]
                    row += f"     {v:>6.3f} |"
                else:
                    row += "        N/A |"
            else:
                row += "        N/A |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def table_ucb_skill_discovery(v2_data, budget):
    """Show which skill UCB selected as best per model."""
    if not v2_data:
        return f"## UCB Skill Discovery at B={budget}: [pending]"

    bk = str(budget)
    lines = []
    lines.append(f"## UCB Skill Discovery (B={budget})")
    lines.append("")
    lines.append("| Model        | Best Skill | Sim-GT | Description                       |")
    lines.append("|--------------|------------|--------|-----------------------------------|")

    skill_desc = {
        "L1": "Developer debug mode",
        "L2": "JSON format request",
        "L4": "Translate-then-back",
        "L6": "API documentation",
        "L7": "Research authority",
        "L10": "Social proof",
        "L12": "Reciprocity",
        "L14": "Introspection",
        "H8": "FITD escalation (4 turns)",
        "H4+L14": "Distraction + introspection (3 turns)",
        "H1+L2": "Pretext + JSON (2 turns)",
    }

    for model in ["DeepSeek V3", "Gemini Flash", "Claude S4", "GPT-4o"]:
        if model not in v2_data:
            continue
        results = v2_data[model].get("results", {})
        if bk not in results or "JustAsk-Full" not in results[bk]:
            continue
        rounds = results[bk]["JustAsk-Full"]["rounds"]
        best_round = max(rounds, key=lambda r: r["sim"])
        sk = best_round.get("skill", "?")
        sim = best_round["sim"]
        desc = skill_desc.get(sk, sk)
        lines.append(f"| {model:<12} | {sk:<10} | {sim:.3f}  | {desc:<33} |")

    lines.append("")
    return "\n".join(lines)


def main():
    # Load data
    v1 = load_v1_results()
    v2 = load_v2_results()

    print(f"Loaded v1 data for models: {list(v1.keys())}")
    print(f"Loaded v2 data for models: {list(v2.keys())}")

    # Generate tables
    sections = []

    # v1 tables (original experiments)
    sections.append(table_r1_baseline_comparison(v1))
    sections.append(table_r2_ablation(v1))

    # v2 tables (budget scaling)
    for b in [3, 5, 10]:
        sections.append(table_v2_cross_model(v2, b))

    sections.append(table_v2_budget_scaling(v2))

    for b in [3, 5, 10]:
        sections.append(table_ucb_skill_discovery(v2, b))

    # Combine
    output = "# ICML 2026 Rebuttal Tables\nGenerated: auto\n\n"
    output += "\n---\n\n".join(sections)

    out_path = TABLE_DIR / "generated_tables.md"
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nTables written to: {out_path}")

    # Also print to stdout
    print("\n" + "=" * 70)
    print(output)


if __name__ == "__main__":
    main()
