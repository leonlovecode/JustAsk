# JustAsk: Curious Code Agents Reveal System Prompts in Frontier LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2601.21233-b31b1b.svg)](https://arxiv.org/abs/2601.21233)

## Abstract

Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present **JustAsk**, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, JustAsk requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound–based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on **41** black-box commercial models across multiple providers, JustAsk consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems.

## Overview

JustAsk investigates system prompt extraction from frontier LLMs through a curiosity-driven skill evolution framework. The core idea: **Coding-Agent-as-Hacker** — an autonomous agent discovers and refines extraction techniques by treating each model interaction as a learning opportunity.

**Key Insight:** Unlike traditional approaches that rely on static prompt templates, JustAsk continuously learns from every interaction. Each API call reveals something about the target model's behavior, defenses, and vulnerabilities. The agent evolves its skill set organically through experience — not model fine-tuning.

## Theoretical Formulation

**Skill Set Definition:**

```
Skill Set = Skills (fixed) + Rules (evolving) + Stats (evolving)
```

| Component  | Role                                      | Evolves? |
| ---------- | ----------------------------------------- | -------- |
| **Skills** | Fixed vocabulary (L1-L14, H1-H15)         | No       |
| **Rules**  | Exploitation knowledge (long-term memory) | Yes      |
| **Stats**  | Exploration guidance (UCB)                | Yes      |

**Evolution Loop:**

```
K₀ = {skills, rules=∅, stats=∅}         // Initial knowledge

For each target m:
    τ ← THINK(K, m)                     // Check rules for this model family
    C ← SELECT(τ, stats)                // Select skill combination (UCB)
    O ← ∅                               // Initial observation

    while not extracted or budget not exhausted:
        prompt ← CONSTRUCT(C)           // Build prompt from skill combination
        O ← ACT(prompt, m)              // Call API and observe result
        stats ← UPDATE_STATS(C, O)      // Always update stats
        τ ← THINK(O, K)                 // What clues? What patterns?
        C ← ADAPT(C, τ, stats)          // Adjust combination based on observations

    K ← EVOLVE_RULES(K, O)              // Add/delete/merge/refine rules
```

**Skill Selection (UCB — Upper Confidence Bound):**

```
UCB(Cᵢ) = success_rate(Cᵢ) + c · √(ln(N) / nᵢ)
          ─────────────────   ─────────────────
           exploitation        exploration bonus
                               (curiosity)
```

## Structure

```
.
├── README.md                          # This file
├── START.md                           # Detailed agent instructions
├── config/
│   └── exp_config.yaml                # Experiment configuration
├── docs/
│   ├── PAP.md                         # Persuasion templates & skill mappings
│   └── PAP_taxonomy.jsonl             # 40 real-world persuasion patterns
└── src/
    ├── skill_evolving.py              # Main extraction via OpenRouter
    ├── skill_testing.py               # Controlled evaluation script
    ├── skill_testing_v3.py            # Controlled evaluation (v3)
    ├── ucb_ranking.py                 # UCB skill selection algorithm
    ├── knowledge.py                   # Knowledge persistence operations
    ├── util.py                        # Shared utilities (API calls)
    ├── validation.py                  # Cross-verify & self-consistency
    ├── analyze_prompts.py             # Prompt analysis tools
    ├── taxonomy_extractor.py          # Taxonomy extraction
    ├── run_controlled_exp_v2.py       # Controlled experiment runner
    ├── run_consistency_experiments.py  # Consistency experiment runner
    ├── consistency_convergence.py     # Convergence analysis
    └── phase2_hypothesis_tests.py     # Hypothesis testing (Phase 2)
```

## Controlled Evaluation

Test extraction skills against an oracle system prompt with injected secrets:

```bash
python src/skill_testing.py --model openai/gpt-5.2
python src/skill_testing.py --model openai/gpt-5.2 --target gpt
```

**Metrics:**

| Metric          | Description                        |
| --------------- | ---------------------------------- |
| **Semantic Sim** | Embedding cosine similarity        |
| **Secret Leak**  | Fraction of injected secrets found |

**Difficulty Levels:** easy, medium, hard, extreme

**Success Thresholds:**
- `SUCCESS`: Semantic similarity > 0.7
- `PARTIAL`: 0.6 < Semantic similarity <= 0.7
- `FAIL`: Semantic similarity <= 0.6

## Setup

### Python Environment

```bash
conda create -n justask python=3.11
conda activate justask
pip install python-dotenv requests numpy
```

### Configuration

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Experiment Workflow

1. Start fresh session with empty skill set
2. Extract target models sequentially (skills evolve)
3. After each extraction, run controlled evaluation
4. Analyze skill evolution and extraction patterns

See `START.md` for detailed agent instructions.

## Citation

```bibtex
@article{zheng2026justask,
  title={Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs},
  author={Zheng, Xiang and Wu, Yutao and Huang, Hanxun and Li, Yige and Ma, Xingjun and Li, Bo and Jiang, Yu-Gang and Wang, Cong},
  journal={arXiv preprint arXiv:2601.21233},
  year={2026}
}
```

## License

MIT
