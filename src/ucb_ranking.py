#!/usr/bin/env python3
"""UCB Ranking Tool - Provides ranked skill combinations for exploration/exploitation."""

import argparse
import json
import math
from pathlib import Path

# Constants
EXPLORATION_CONSTANT = math.sqrt(2)  # c = √2 ≈ 1.414
DATA_DIR = Path(__file__).parent.parent / "data"
KNOWLEDGE_FILE = DATA_DIR / "extraction_knowledge.json"


def calculate_ucb(visits: int, successes: int, total_visits: int) -> float:
    """
    Calculate UCB score for a combination.

    UCB = successes/visits + c × √(ln(N)/visits)

    Args:
        visits: Number of times this combination was tried
        successes: Number of successful extractions
        total_visits: Total visits across all combinations in section

    Returns:
        UCB score
    """
    if visits <= 0:
        visits = 1  # Safety, but should not happen with pre-initialization

    success_rate = successes / visits
    exploration_bonus = EXPLORATION_CONSTANT * math.sqrt(math.log(total_visits) / visits)

    return success_rate + exploration_bonus


def assign_ranks(stats: list[dict]) -> list[dict]:
    """
    Assign tied ranks to combinations.

    Same visits + successes = same rank.
    After tied ranks, next rank skips appropriately.

    Args:
        stats: List of dicts with combination, visits, successes, ucb

    Returns:
        List with rank field added
    """
    # Sort by UCB descending
    sorted_stats = sorted(stats, key=lambda x: x["ucb"], reverse=True)

    if not sorted_stats:
        return []

    # Assign ranks with ties
    ranked = []
    current_rank = 1

    for i, entry in enumerate(sorted_stats):
        if i == 0:
            entry["rank"] = current_rank
        else:
            prev = sorted_stats[i - 1]
            # Tied if same visits AND same successes
            if entry["visits"] == prev["visits"] and entry["successes"] == prev["successes"]:
                entry["rank"] = prev["rank"]
            else:
                entry["rank"] = i + 1  # Skip to position (1-indexed)

        ranked.append(entry)

    return ranked


def load_knowledge() -> dict:
    """Load extraction knowledge from JSON file."""
    with open(KNOWLEDGE_FILE) as f:
        return json.load(f)


def is_multi_turn(combo: str) -> bool:
    """Check if a skill combo is multi-turn (H-prefixed)."""
    return combo.startswith("H")


def get_rankings(top_n: int = 30) -> dict:
    """
    Get UCB rankings for all skill combinations in a unified table.

    All skills (single-turn and multi-turn) are treated as states in the same
    bandit space, enabling direct comparison and unified exploration.

    Args:
        top_n: Maximum number of entries to return

    Returns:
        Dict with unified rankings and metadata
    """
    knowledge = load_knowledge()
    skill_stats = knowledge.get("skill_stats", {})

    # Collect all skills (single-turn and multi-turn together)
    all_entries = []

    for combo, stats in skill_stats.items():
        # Skip metadata keys
        if combo.startswith("_"):
            continue
        all_entries.append({
            "combination": combo,
            "type": "multi" if is_multi_turn(combo) else "single",
            "visits": stats.get("visits", 1),
            "successes": stats.get("successes", 0),
            "partials": stats.get("partials", 0),
        })

    # Calculate total visits across ALL skills
    total_visits = sum(e["visits"] for e in all_entries)
    total_visits = max(total_visits, 1)  # Ensure minimum of 1 for log calculation

    # Calculate UCB for each skill
    for entry in all_entries:
        entry["ucb"] = round(
            calculate_ucb(entry["visits"], entry["successes"], total_visits), 2
        )

    # Assign ranks
    ranked = assign_ranks(all_entries)

    # Apply top_n limit
    ranked = ranked[:top_n]

    # Count by type for summary
    single_count = sum(1 for e in all_entries if e["type"] == "single")
    multi_count = sum(1 for e in all_entries if e["type"] == "multi")

    return {
        "rankings": ranked,
        "total_combinations": len(all_entries),
        "single_turn_count": single_count,
        "multi_turn_count": multi_count,
        "total_visits": total_visits,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UCB Ranking Tool - Get ranked skill combinations"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum number of entries per section (default: 20)",
    )
    args = parser.parse_args()

    result = get_rankings(top_n=args.top)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
