"""Cross-skill extraction validation.

Reuses metrics from skill_testing.py:
- compute_semantic_similarity()

Verification protocol (from experiment-plan.md):
1. Two successful skills required (cross-skill agreement)
2. Each skill run twice (self-consistency check)
"""

from skill_testing import (
    compute_semantic_similarity,
)

# Unified threshold for both cross-verify and self-consistency
# As per START.md design: semantic_sim > 0.7
SEMANTIC_THRESHOLD = 0.7
SELF_CONSISTENCY_THRESHOLD = 0.7


def cross_validate(
    extraction1: str,
    extraction2: str,
    use_mock: bool = False,
) -> dict:
    """
    Cross-validate two extractions from different skills.

    Per START.md design:
    - Require >=2 successful extractions with DIFFERENT skills
    - semantic_sim > 0.7 -> verified (unified threshold)
    - Only mark model as "extracted" when cross-validated

    Args:
        extraction1: First extraction result
        extraction2: Second extraction result
        use_mock: If True, use word overlap for semantic similarity (testing)

    Returns:
        Dict with:
            - verified: bool
            - confidence: "high" | "medium" | "low"
            - semantic_sim: float
    """
    # Handle empty inputs early
    if not extraction1 or not extraction2:
        return {
            "verified": False,
            "confidence": "low",
            "semantic_sim": 0.0,
        }

    # Semantic similarity (with mock option for testing)
    if use_mock:
        # Simple word overlap for testing
        words1 = set(extraction1.lower().split())
        words2 = set(extraction2.lower().split())
        if not words1 or not words2:
            semantic = 0.0
        else:
            overlap = len(words1 & words2)
            semantic = overlap / max(len(words1), len(words2))
    else:
        semantic = compute_semantic_similarity(extraction1, extraction2)

    # Determine verification status (semantic_sim > 0.7)
    verified = semantic > SEMANTIC_THRESHOLD

    # Determine confidence level based on semantic similarity
    if semantic > 0.7:
        confidence = "high"
    elif semantic > 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "verified": verified,
        "confidence": confidence,
        "semantic_sim": round(semantic, 3),
    }


def check_self_consistency(
    run1: str,
    run2: str,
    use_mock: bool = False,
) -> dict:
    """
    Check self-consistency: same skill run twice should produce similar output.

    Per START.md design:
    - Each successful skill must be run TWICE
    - Verify: semantic_sim(run1, run2) > 0.7 (unified threshold)

    Args:
        run1: First run output
        run2: Second run output (same skill)
        use_mock: If True, use word overlap for semantic similarity (testing)

    Returns:
        Dict with:
            - consistent: bool (semantic_sim > 0.7)
            - semantic_sim: float
    """
    # Handle empty inputs
    if not run1 or not run2:
        return {
            "consistent": False,
            "semantic_sim": 0.0,
        }

    # Semantic similarity (with mock option for testing)
    if use_mock:
        words1 = set(run1.lower().split())
        words2 = set(run2.lower().split())
        if not words1 or not words2:
            semantic = 0.0
        else:
            overlap = len(words1 & words2)
            semantic = overlap / max(len(words1), len(words2))
    else:
        semantic = compute_semantic_similarity(run1, run2)

    consistent = semantic > SELF_CONSISTENCY_THRESHOLD

    return {
        "consistent": consistent,
        "semantic_sim": round(semantic, 3),
    }
