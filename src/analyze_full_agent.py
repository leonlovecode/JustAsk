#!/usr/bin/env python3
"""Analyze full-agent evaluation results and compare to template-based results."""

import json
import sys
from pathlib import Path
from statistics import mean, stdev


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_single_file(path: Path):
    data = load_results(path)
    print(f"\n{'='*70}")
    print(f"File: {path.name}")
    print(f"{'='*70}")

    for model, budgets in data.items():
        print(f"\n## {model}")
        for budget, seeds in budgets.items():
            sims = [s["best_sim"] for s in seeds]
            avg_sims = [s["avg_sim"] for s in seeds]
            skills = [s["best_skill"] for s in seeds]

            print(f"\n  B={budget} ({len(seeds)} seeds):")
            for i, s in enumerate(seeds):
                print(f"    seed {s['seed']}: best={s['best_sim']:.3f} ({s['best_skill']}), "
                      f"avg={s['avg_sim']:.3f}, rounds={s['rounds_used']}")
            if len(sims) > 1:
                print(f"    --- Mean best: {mean(sims):.3f} +/- {stdev(sims):.3f}")
                print(f"    --- Mean avg:  {mean(avg_sims):.3f} +/- {stdev(avg_sims):.3f}")
            else:
                print(f"    --- Best: {sims[0]:.3f}")

            # Skill frequency
            from collections import Counter
            skill_counts = Counter(skills)
            print(f"    --- Best skills: {dict(skill_counts)}")

            # Per-round analysis across seeds
            if len(seeds) > 0 and seeds[0].get("rounds"):
                round_sims = {}
                for s in seeds:
                    for r in s["rounds"]:
                        rn = r["round"]
                        if rn not in round_sims:
                            round_sims[rn] = []
                        round_sims[rn].append(r["sim"])
                print(f"    --- Per-round avg sim:")
                for rn in sorted(round_sims.keys()):
                    vals = round_sims[rn]
                    print(f"        R{rn}: {mean(vals):.3f} ({len(vals)} seeds)")


def compare_to_template(full_agent_path: Path, template_dir: Path):
    """Compare full-agent results to template-based v2 results."""
    full_data = load_results(full_agent_path)

    # Load template results for matching models
    template_files = list(template_dir.glob("rebuttal_v2_*.json"))

    print(f"\n{'='*70}")
    print("COMPARISON: Full-Agent vs Template-Based (v2)")
    print(f"{'='*70}")

    for model, budgets in full_data.items():
        # Find matching template file
        model_short = model.split("/")[-1]
        matching = [f for f in template_files if model_short in f.name]

        if not matching:
            print(f"\n## {model}: No template-based comparison available")
            for budget, seeds in budgets.items():
                sims = [s["best_sim"] for s in seeds]
                print(f"  B={budget}: Full-Agent = {mean(sims):.3f} "
                      f"(+/- {stdev(sims):.3f})" if len(sims) > 1 else
                      f"  B={budget}: Full-Agent = {sims[0]:.3f}")
            continue

        template_data = load_results(matching[0])
        print(f"\n## {model}")

        for budget, seeds in budgets.items():
            b = int(budget)
            full_sims = [s["best_sim"] for s in seeds]
            full_mean = mean(full_sims)

            # Get template results at same budget
            template_results = template_data.get("results", {}).get(str(b), {})
            if not template_results:
                template_results = template_data.get("results", {}).get(b, {})

            print(f"\n  B={b}:")
            print(f"    Full-Agent (DeepSeek agent): {full_mean:.3f} "
                  f"(+/- {stdev(full_sims):.3f})" if len(full_sims) > 1 else
                  f"    Full-Agent (DeepSeek agent): {full_sims[0]:.3f}")

            if template_results:
                for method, mdata in template_results.items():
                    if isinstance(mdata, dict) and "summary" in mdata:
                        t_sim = mdata["summary"]["best_sim"]
                        print(f"    {method}: {t_sim:.3f}")


def main():
    data_dir = Path(__file__).parent.parent / "data"

    # Find all full-agent result files
    files = sorted(data_dir.glob("full_agent_eval_*.json"))
    if not files:
        print("No full-agent evaluation results found.")
        sys.exit(1)

    # Filter out empty/failed results
    valid_files = []
    for f in files:
        data = load_results(f)
        has_results = False
        for model, budgets in data.items():
            for budget, seeds in budgets.items():
                if any(s["best_sim"] > 0 for s in seeds):
                    has_results = True
        if has_results:
            valid_files.append(f)

    print(f"Found {len(valid_files)} valid result files (out of {len(files)} total)")

    for f in valid_files:
        analyze_single_file(f)

    # Compare latest results to template-based
    if valid_files:
        latest = valid_files[-1]
        compare_to_template(latest, data_dir)


if __name__ == "__main__":
    main()
