#!/usr/bin/env python3
"""
Batch score production eval responses with embeddings.

Reads JSON result files from production_eval runs (with --skip-embedding),
computes cosine similarity to ground truth using text-embedding-3-large,
and writes updated result files with sim scores.

Usage:
    uv run --project ~/.claude/cc-python python3 src/score_production_responses.py \
        data/production_eval_justask_pleak_raccoon_zhang_20260328_2215.json

    # Or score all unscored files:
    uv run --project ~/.claude/cc-python python3 src/score_production_responses.py --all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"

GT_FILES = {
    "Perplexity Sonar": GROUND_TRUTH_DIR / "perplexity_prompt.txt",
    "Perplexity Sonar Pro": GROUND_TRUTH_DIR / "perplexity_prompt.txt",
    "Cursor Agent CLI": GROUND_TRUTH_DIR / "cursor_cli_prompt.txt",
    "OpenAI Codex CLI": GROUND_TRUTH_DIR / "codex_cli_prompt.txt",
    "Google Gemini CLI": GROUND_TRUTH_DIR / "gemini_cli_prompt.txt",
    "GitHub Copilot CLI": GROUND_TRUTH_DIR / "copilot_cli_prompt.txt",
}


def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            model="openai/text-embedding-3-large", input=text[:8000]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  [embedding error: {e}]")
        return None


def compute_similarity(text, gt_text, client, gt_cache):
    if gt_text not in gt_cache:
        gt_cache[gt_text] = get_embedding(gt_text, client)
    emb_gt = gt_cache[gt_text]
    emb_resp = get_embedding(text, client)
    if emb_gt is None or emb_resp is None:
        return None
    sim = float(
        np.dot(emb_gt, emb_resp)
        / (np.linalg.norm(emb_gt) * np.linalg.norm(emb_resp))
    )
    return max(0, round(sim, 4))


def score_file(filepath, client):
    data = json.loads(filepath.read_text())
    gt_cache = {}
    updated = False

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
                if result["sim"] != -1.0 and result["sim"] != 0.0:
                    continue  # already scored

                content = result.get("response_content", "")
                if not content:
                    continue

                sim = compute_similarity(content, gt_text, client, gt_cache)
                if sim is not None:
                    result["sim"] = sim
                    updated = True
                    print(f"  [{result['skill']}]: sim={sim:.3f}")
                else:
                    print(f"  [{result['skill']}]: FAILED")
                time.sleep(0.5)

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
        entry["aggregate"]["best_sim_sd"] = round(stdev(best_sims) if len(best_sims) > 1 else 0.0, 4)
        entry["aggregate"]["per_seed_bests"] = [round(s, 4) for s in best_sims]

        print(f"  -> {method} on {display_name}: {entry['aggregate']['best_sim_mean']:.3f} +/- {entry['aggregate']['best_sim_sd']:.3f}")

    if updated:
        out = filepath.with_name(filepath.stem + "_scored.json")
        out.write_text(json.dumps(data, indent=2))
        print(f"\nScored results saved to: {out}")
    else:
        print("\nNo updates needed.")


def main():
    import os
    from openai import OpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON result files to score")
    parser.add_argument("--all", action="store_true", help="Score all unscored files")
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Quick API test
    try:
        test = client.embeddings.create(model="openai/text-embedding-3-large", input="test")
        print(f"Embedding API OK ({len(test.data[0].embedding)} dims)")
    except Exception as e:
        print(f"Embedding API FAILED: {e}")
        print("Wait for OpenRouter embedding API to recover, then retry.")
        sys.exit(1)

    if args.all:
        files = sorted(PROJECT_ROOT.glob("data/production_eval_*_2026*.json"))
        files = [f for f in files if "_scored" not in f.name]
    else:
        files = [Path(f) for f in args.files]

    for f in files:
        print(f"\n{'='*60}")
        print(f"Processing: {f.name}")
        print(f"{'='*60}")
        score_file(f, client)


if __name__ == "__main__":
    main()
