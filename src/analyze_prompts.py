"""
Quick analysis of all extracted system prompts.

Extracts basic statistics without API calls.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def analyze_prompt(content: str, model_id: str) -> dict:
    """Analyze a single prompt for key patterns."""
    content_lower = content.lower()

    # Identity patterns
    identity_patterns = {
        "openai": ["openai", "chatgpt", "gpt-4", "gpt-5"],
        "anthropic": ["anthropic", "claude"],
        "google": ["google", "gemini", "deepmind"],
        "xai": ["xai", "grok", "elon musk"],
        "meta": ["meta", "llama", "facebook"],
        "deepseek": ["deepseek"],
        "mistral": ["mistral"],
        "cohere": ["cohere", "command"],
        "baidu": ["baidu", "ernie"],
        "alibaba": ["alibaba", "qwen", "tongyi"],
        "tencent": ["tencent", "hunyuan"],
        "bytedance": ["bytedance", "seed"],
    }

    claimed_identity = None
    for provider, patterns in identity_patterns.items():
        for p in patterns:
            if p in content_lower:
                claimed_identity = provider
                break
        if claimed_identity:
            break

    # Actual identity from model_id
    actual_provider = model_id.split("/")[0] if "/" in model_id else model_id.split("-")[0]

    # Safety constraint keywords
    hard_constraints = []
    constraint_keywords = {
        "violence": ["violence", "physical harm", "hurt people", "injure"],
        "self_harm": ["self-harm", "self harm", "suicide", "self-mutilation"],
        "csam": ["csam", "child sexual", "minors", "children sexual"],
        "illegal": ["illegal", "criminal", "unlawful", "crime"],
        "privacy": ["privacy", "doxxing", "personal information", "pii"],
        "malware": ["malware", "virus", "exploit", "hack"],
        "fraud": ["fraud", "deception", "impersonation", "scam"],
    }

    for constraint, keywords in constraint_keywords.items():
        for kw in keywords:
            if kw in content_lower:
                hard_constraints.append(constraint)
                break

    # Core principles
    principles = []
    principle_keywords = {
        "helpful": ["helpful", "assist", "help users"],
        "honest": ["honest", "truthful", "accurate", "factual"],
        "harmless": ["harmless", "safe", "avoid harm"],
        "transparent": ["transparent", "transparency"],
    }

    for principle, keywords in principle_keywords.items():
        for kw in keywords:
            if kw in content_lower:
                principles.append(principle)
                break

    # Check for verbatim markers
    has_verbatim = bool(re.search(r"<policy>|<system>|```\s*\n.*?(you are|your role)", content, re.IGNORECASE | re.DOTALL))

    # Check for priority hierarchy
    has_hierarchy = bool(re.search(r"priority|hierarchy|precedence|safety\s*>|first.*then", content_lower))

    # Check for refusal templates
    has_refusal_templates = bool(re.search(r"refus|can't help|cannot assist|sorry.*but", content_lower))

    return {
        "model_id": model_id,
        "char_count": len(content),
        "line_count": content.count("\n") + 1,
        "claimed_identity": claimed_identity,
        "actual_provider": actual_provider,
        "identity_match": claimed_identity == actual_provider if claimed_identity else None,
        "hard_constraints": hard_constraints,
        "num_hard_constraints": len(hard_constraints),
        "core_principles": principles,
        "num_principles": len(principles),
        "has_verbatim": has_verbatim,
        "has_priority_hierarchy": has_hierarchy,
        "has_refusal_templates": has_refusal_templates,
    }


def load_and_analyze_all(base_path: str) -> list[dict]:
    """Load and analyze all prompts."""
    results = []
    base = Path(base_path)

    # T0
    t0_path = base / "data" / "T0"
    if t0_path.exists():
        for agent_dir in t0_path.iterdir():
            if agent_dir.is_dir():
                prompt_file = agent_dir / "system_prompt.md"
                if prompt_file.exists():
                    content = prompt_file.read_text()
                    model_id = f"claude-code/{agent_dir.name}"
                    analysis = analyze_prompt(content, model_id)
                    analysis["phase"] = "T0"
                    results.append(analysis)

    # T1
    t1_path = base / "data" / "T1"
    if t1_path.exists():
        for model_dir in t1_path.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                prompt_file = model_dir / "system_prompt.md"
                if prompt_file.exists():
                    content = prompt_file.read_text()
                    model_id = model_dir.name.replace("_", "/", 1)
                    analysis = analyze_prompt(content, model_id)
                    analysis["phase"] = "T1"
                    results.append(analysis)

    return results


def print_summary(results: list[dict]):
    """Print summary statistics."""
    t0 = [r for r in results if r["phase"] == "T0"]
    t1 = [r for r in results if r["phase"] == "T1"]

    print("=" * 60)
    print("SYSTEM PROMPT ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal prompts: {len(results)}")
    print(f"  T0 (Claude Code): {len(t0)}")
    print(f"  T1 (Black-box): {len(t1)}")

    print(f"\nAverage length:")
    print(f"  T0: {sum(r['char_count'] for r in t0) / len(t0):.0f} chars, {sum(r['line_count'] for r in t0) / len(t0):.0f} lines")
    print(f"  T1: {sum(r['char_count'] for r in t1) / len(t1):.0f} chars, {sum(r['line_count'] for r in t1) / len(t1):.0f} lines")

    # Identity confusion
    t1_with_identity = [r for r in t1 if r["identity_match"] is not None]
    confused = [r for r in t1_with_identity if r["identity_match"] == False]
    print(f"\nIdentity Analysis (T1):")
    print(f"  Models with detected identity: {len(t1_with_identity)}")
    print(f"  Identity matches: {len([r for r in t1_with_identity if r['identity_match']])}")
    print(f"  Identity confused: {len(confused)}")
    if confused:
        print(f"  Confused models:")
        for r in confused:
            print(f"    - {r['model_id']}: claims {r['claimed_identity']}, actually {r['actual_provider']}")

    # Hard constraints
    print(f"\nHard Constraint Coverage:")
    constraint_counts = defaultdict(int)
    for r in results:
        for c in r["hard_constraints"]:
            constraint_counts[c] += 1
    for c, count in sorted(constraint_counts.items(), key=lambda x: -x[1]):
        print(f"  {c}: {count} ({count / len(results) * 100:.0f}%)")

    # Core principles
    print(f"\nCore Principle Coverage:")
    principle_counts = defaultdict(int)
    for r in results:
        for p in r["core_principles"]:
            principle_counts[p] += 1
    for p, count in sorted(principle_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count} ({count / len(results) * 100:.0f}%)")

    # Verbatim extraction
    verbatim = [r for r in results if r["has_verbatim"]]
    print(f"\nVerbatim Content: {len(verbatim)} ({len(verbatim) / len(results) * 100:.0f}%)")
    for r in verbatim:
        print(f"  - {r['model_id']}")

    # Priority hierarchy
    hierarchy = [r for r in results if r["has_priority_hierarchy"]]
    print(f"\nPriority Hierarchy: {len(hierarchy)} ({len(hierarchy) / len(results) * 100:.0f}%)")

    # Refusal templates
    refusal = [r for r in results if r["has_refusal_templates"]]
    print(f"\nRefusal Templates: {len(refusal)} ({len(refusal) / len(results) * 100:.0f}%)")


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--output-csv", default="tables/prompt_analysis.csv")
    parser.add_argument("--output-json", default="tables/prompt_analysis.json")
    args = parser.parse_args()

    results = load_and_analyze_all(args.base_path)
    print_summary(results)

    # Save CSV
    csv_path = Path(args.base_path) / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "phase", "model_id", "char_count", "line_count",
            "claimed_identity", "actual_provider", "identity_match",
            "num_hard_constraints", "hard_constraints",
            "num_principles", "core_principles",
            "has_verbatim", "has_priority_hierarchy", "has_refusal_templates"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames if k in r}
            row["hard_constraints"] = "|".join(r["hard_constraints"])
            row["core_principles"] = "|".join(r["core_principles"])
            writer.writerow(row)

    print(f"\nSaved CSV to: {csv_path}")

    # Save JSON
    json_path = Path(args.base_path) / args.output_json
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()
