#!/usr/bin/env python3
"""
Score production eval responses using local sentence-transformers embeddings.

Uses BAAI/bge-large-en-v1.5 for consistent cross-method comparison
when OpenRouter embedding API is unavailable.

Usage:
    uv run --project ~/.claude/cc-python python3 src/score_local_embeddings.py --all
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"

GT_FILES = {
    "Perplexity Sonar": GROUND_TRUTH_DIR / "perplexity_prompt.txt",
    "Perplexity Sonar Pro": GROUND_TRUTH_DIR / "perplexity_prompt.txt",
    "Cursor Agent CLI": GROUND_TRUTH_DIR / "cursor_cli_prompt.txt",
    "OpenAI Codex CLI": GROUND_TRUTH_DIR / "codex_cli_prompt.txt",
    "Google Gemini CLI": GROUND_TRUTH_DIR / "gemini_cli_prompt.txt",
    "GitHub Copilot CLI": GROUND_TRUTH_DIR / "copilot_cli_prompt.txt",
}

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded ({model.get_sentence_embedding_dimension()} dims)")
    return model


def compute_similarity(text, gt_text, model, gt_cache):
    if gt_text not in gt_cache:
        gt_cache[gt_text] = model.encode(gt_text[:8000], normalize_embeddings=True)
    emb_gt = gt_cache[gt_text]
    emb_resp = model.encode(text[:8000], normalize_embeddings=True)
    sim = float(np.dot(emb_gt, emb_resp))
    return max(0, round(sim, 4))


def score_file(filepath, model):
    data = json.loads(filepath.read_text())
    gt_cache = {}

    for key, entry in data.items():
        display_name = entry["config"]["display_name"]
        method = entry["config"].get("method", "JustAsk")
        gt_file = GT_FILES.get(display_name)
        if gt_file is None or not gt_file.exists():
            print(f"SKIP {key}: no ground truth for {display_name}")
            continue

        gt_text = gt_file.read_text()
        print(f"\nScoring [{method}] {display_name}...")

        for seed_data in entry["seeds"]:
            for result in seed_data["results"]:
                if not result["success"]:
                    continue
                content = result.get("response_content", "")
                if not content:
                    continue

                sim = compute_similarity(content, gt_text, model, gt_cache)
                result["sim"] = sim
                print(f"  [{result['skill']}]: sim={sim:.3f} ({len(content)} chars)")

            # Recompute seed-level aggregates
            successful = [r for r in seed_data["results"] if r["success"]]
            if successful:
                best = max(successful, key=lambda r: r["sim"])
                seed_data["best_sim"] = best["sim"]
                seed_data["best_skill"] = best["skill"]
                seed_data["avg_sim"] = round(mean([r["sim"] for r in successful]), 4)

        # Recompute entry-level aggregates
        best_sims = [s["best_sim"] for s in entry["seeds"]]
        entry["aggregate"]["best_sim_mean"] = round(mean(best_sims), 4)
        entry["aggregate"]["best_sim_sd"] = round(
            stdev(best_sims) if len(best_sims) > 1 else 0.0, 4
        )
        entry["aggregate"]["per_seed_bests"] = [round(s, 4) for s in best_sims]
        print(
            f"  -> {method} on {display_name}: "
            f"best={entry['aggregate']['best_sim_mean']:.3f}"
        )

    out = filepath.with_name(filepath.stem + "_scored_local.json")
    out.write_text(json.dumps(data, indent=2))
    print(f"\nScored results saved to: {out}")
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON result files to score")
    parser.add_argument("--all", action="store_true", help="Score all baseline files")
    args = parser.parse_args()

    model = load_model()

    if args.all:
        files = sorted(
            PROJECT_ROOT.glob(
                "data/production_eval_justask_pleak_raccoon_zhang_*.json"
            )
        )
        files = [f for f in files if "_scored" not in f.name]
    else:
        files = [Path(f) for f in args.files]

    # Collect all results for summary
    all_results = {}
    for f in files:
        print(f"\n{'='*60}")
        print(f"Processing: {f.name}")
        print(f"{'='*60}")
        data = score_file(f, model)
        all_results.update(data)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Method Comparison on Production Systems")
    print(f"{'='*60}")

    # Group by target
    targets = {}
    for key, entry in all_results.items():
        method, target = key.split("/", 1)
        if target not in targets:
            targets[target] = {}
        targets[target][method] = entry["aggregate"]["best_sim_mean"]

    # Print table
    methods = ["JustAsk", "PLeak", "Raccoon", "Zhang"]
    header = f"{'Target':<20}" + "".join(f"{m:<10}" for m in methods)
    print(header)
    print("-" * len(header))
    for target in sorted(targets):
        row = f"{target:<20}"
        for m in methods:
            val = targets[target].get(m, None)
            row += f"{val:<10.3f}" if val is not None else f"{'N/A':<10}"
        print(row)


if __name__ == "__main__":
    main()
