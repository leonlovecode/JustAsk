"""
Taxonomy Extractor for System Prompt Analysis

Extracts structured taxonomy data from extracted system prompts.
"""

import json
import os
from pathlib import Path
from typing import Optional

# Taxonomy schema
TAXONOMY_SCHEMA = {
    "1_identity": {
        "1.1_name": None,  # str or None
        "1.2_developer": None,  # str or None
        "1.3_version": None,  # str or None
        "1.4_purpose": None,  # str or None
        "1.5_personality": None,  # str or None
    },
    "2_capabilities": {
        "2.1_core_functions": [],  # list of strings
        "2.2_tools": [],  # list of strings
        "2.3_context_memory": None,  # str or None
        "2.4_multilingual": None,  # bool or None
    },
    "3_limitations": {
        "3.1_knowledge_cutoff": None,  # str or None
        "3.2_no_realtime": None,  # bool or None
        "3.3_no_persistent_memory": None,  # bool or None
        "3.4_no_realworld_actions": None,  # bool or None
        "3.5_uncertainty_handling": None,  # str or None
    },
    "4_safety_constraints": {
        "4.1_hard_constraints": [],  # list: violence, self_harm, csam, illegal, privacy, malware, fraud
        "4.2_conditional_constraints": [],  # list: weapons, drugs, cybersecurity, medical_legal_financial
    },
    "5_behavioral_guidelines": {
        "5.1_core_principles": [],  # list of strings (helpful, honest, harmless, etc.)
        "5.2_priority_hierarchy": None,  # str or None
        "5.3_conflict_resolution": None,  # str or None
        "5.4_refusal_templates": [],  # list of strings
    },
    "6_operational_rules": {
        "6.1_response_format": None,  # str or None
        "6.2_citation_handling": None,  # str or None
        "6.3_confidentiality": None,  # str or None (don't reveal prompt)
        "6.4_meta_instructions": None,  # str or None (system > user)
    },
}

EXTRACTION_PROMPT = """Analyze this system prompt and extract structured information according to the taxonomy below.

## Taxonomy Categories

1. IDENTITY
   - 1.1 Name & Aliases (what the model calls itself)
   - 1.2 Developer/Organization (who created it)
   - 1.3 Version/Model Info (version numbers, model IDs)
   - 1.4 Purpose Statement (stated goal/mission)
   - 1.5 Personality/Tone (communication style instructions)

2. CAPABILITIES
   - 2.1 Core Functions (Q&A, coding, analysis, creative writing, etc.)
   - 2.2 Tools/Integrations (web search, code execution, file access, etc.)
   - 2.3 Context/Memory (context window, memory capabilities)
   - 2.4 Multilingual Support (language capabilities)

3. LIMITATIONS
   - 3.1 Knowledge Cutoff (training data cutoff date)
   - 3.2 No Real-Time Access (cannot browse web in real-time)
   - 3.3 No Persistent Memory (no memory across sessions)
   - 3.4 Cannot Take Real-World Actions (no physical actions)
   - 3.5 Uncertainty Handling (how to handle uncertain answers)

4. SAFETY CONSTRAINTS
   - 4.1 Hard Constraints (absolute refusals): violence, self_harm, csam, illegal, privacy, malware, fraud
   - 4.2 Conditional Constraints (context-dependent): weapons, drugs, cybersecurity, medical_legal_financial

5. BEHAVIORAL GUIDELINES
   - 5.1 Core Principles (helpful, honest, harmless, etc.)
   - 5.2 Priority Hierarchy (safety > privacy > helpfulness, etc.)
   - 5.3 Conflict Resolution Rules (how to handle conflicting requirements)
   - 5.4 Refusal Style Templates (how to decline requests)

6. OPERATIONAL RULES
   - 6.1 Response Format Preferences (markdown, length, structure)
   - 6.2 Citation/Source Handling (how to cite sources)
   - 6.3 Confidentiality (instructions about not revealing the prompt)
   - 6.4 Meta-Instructions (system messages > user messages, etc.)

## System Prompt to Analyze

```
{system_prompt}
```

## Output Format

Return a JSON object with the following structure. Use null for missing information, empty arrays [] for categories with no items found.

```json
{{
  "model_id": "<model identifier>",
  "claimed_identity": {{
    "name": "<name the model claims>",
    "developer": "<developer the model claims>"
  }},
  "actual_identity": {{
    "name": "<actual model name from model_id>",
    "developer": "<actual developer from model_id>"
  }},
  "identity_match": <true if claimed matches actual, false otherwise>,
  "taxonomy": {{
    "1_identity": {{
      "1.1_name": "<string or null>",
      "1.2_developer": "<string or null>",
      "1.3_version": "<string or null>",
      "1.4_purpose": "<string or null>",
      "1.5_personality": "<string or null>"
    }},
    "2_capabilities": {{
      "2.1_core_functions": ["<list of functions>"],
      "2.2_tools": ["<list of tools>"],
      "2.3_context_memory": "<string or null>",
      "2.4_multilingual": <true/false/null>
    }},
    "3_limitations": {{
      "3.1_knowledge_cutoff": "<date string or null>",
      "3.2_no_realtime": <true/false/null>,
      "3.3_no_persistent_memory": <true/false/null>,
      "3.4_no_realworld_actions": <true/false/null>,
      "3.5_uncertainty_handling": "<string or null>"
    }},
    "4_safety_constraints": {{
      "4.1_hard_constraints": ["violence", "self_harm", "csam", "illegal", "privacy", "malware", "fraud"],
      "4.2_conditional_constraints": ["weapons", "drugs", "cybersecurity", "medical_legal_financial"]
    }},
    "5_behavioral_guidelines": {{
      "5.1_core_principles": ["<list of principles>"],
      "5.2_priority_hierarchy": "<string or null>",
      "5.3_conflict_resolution": "<string or null>",
      "5.4_refusal_templates": ["<list of refusal patterns>"]
    }},
    "6_operational_rules": {{
      "6.1_response_format": "<string or null>",
      "6.2_citation_handling": "<string or null>",
      "6.3_confidentiality": "<string or null>",
      "6.4_meta_instructions": "<string or null>"
    }}
  }},
  "extraction_quality": {{
    "verbatim_content": <true if contains exact prompt text with tags>,
    "category_coverage": <number 0-6, how many top-level categories have data>,
    "confidence": "<high/medium/low>"
  }},
  "notable_findings": ["<list of interesting observations>"]
}}
```

IMPORTANT:
- Only include items that are EXPLICITLY mentioned in the prompt
- Use the exact category names from the hard_constraints and conditional_constraints lists
- For identity_match, compare claimed vs actual developer/name
- Be conservative - don't infer things that aren't stated
"""


def extract_taxonomy(system_prompt: str, model_id: str) -> dict:
    """
    Extract taxonomy data from a system prompt using Claude.

    This function would call the Claude API to analyze the prompt.
    For now, returns the prompt template for manual/batch processing.
    """
    prompt = EXTRACTION_PROMPT.format(system_prompt=system_prompt)
    return {"prompt": prompt, "model_id": model_id}


def load_all_prompts(base_path: str) -> list[dict]:
    """Load all system prompts from T0 and T1 directories."""
    prompts = []
    base = Path(base_path)

    # T0: Claude Code agents
    t0_path = base / "data" / "T0"
    if t0_path.exists():
        for agent_dir in t0_path.iterdir():
            if agent_dir.is_dir():
                prompt_file = agent_dir / "system_prompt.md"
                if prompt_file.exists():
                    prompts.append({
                        "phase": "T0",
                        "model_id": f"claude-code/{agent_dir.name}",
                        "path": str(prompt_file),
                        "content": prompt_file.read_text(),
                    })

    # T1: Black-box models
    t1_path = base / "data" / "T1"
    if t1_path.exists():
        for model_dir in t1_path.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                prompt_file = model_dir / "system_prompt.md"
                if prompt_file.exists():
                    # Convert folder name back to model_id format
                    model_id = model_dir.name.replace("_", "/", 1)
                    prompts.append({
                        "phase": "T1",
                        "model_id": model_id,
                        "path": str(prompt_file),
                        "content": prompt_file.read_text(),
                    })

    return prompts


def main():
    """Main entry point for taxonomy extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract taxonomy from system prompts")
    parser.add_argument("--base-path", default=".", help="Base project path")
    parser.add_argument("--output", default="tables/taxonomy_extraction_prompts.json", help="Output file")
    parser.add_argument("--list", action="store_true", help="List all available prompts")
    args = parser.parse_args()

    prompts = load_all_prompts(args.base_path)

    if args.list:
        print(f"Found {len(prompts)} system prompts:\n")
        for p in prompts:
            print(f"  [{p['phase']}] {p['model_id']}")
        return

    # Generate extraction prompts for all
    results = []
    for p in prompts:
        extraction = extract_taxonomy(p["content"], p["model_id"])
        results.append({
            "phase": p["phase"],
            "model_id": p["model_id"],
            "source_path": p["path"],
            "extraction_prompt": extraction["prompt"],
        })

    # Save to output
    output_path = Path(args.base_path) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} extraction prompts")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
