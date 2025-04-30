# synthetic_transcript_generator.py
"""Generate realistic adviser–client transcripts **with digressions** and the
corresponding structured‑truth JSON.

Changes in this version (v0.2):
================================
*   Adds a *digression* action that kicks‑in every `DIGRESSION_INTERVAL` turns
    (or when chosen at random) where the adviser explains a topic drawn from
    the partially‑built JSON.
*   Keeps a `digressed_keys` set so each JSON path is discussed at most once.
*   `build_prompt()` now delegates to either `_build_section_prompt()` or
    `_build_digression_prompt()` depending on that action.
*   Conversation questions are shuffled across **and within** sections to avoid
    rigid, form‑like ordering.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from textwrap import dedent
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Conversation schema definition (same hierarchy you supplied)
# ---------------------------------------------------------------------------
SCHEMA = {
    "personal_details": [
        "title", "first_name", "middle_names", "last_name", "known_as",
        "pronouns", "date_of_birth", "place_of_birth", "nationality",
        "gender", "legal_sex", "marital_status", "home_phone",
        "mobile_phone", "email_address",
    ],
    "current_address": [
        "ownership_status", "postcode", "house_name_or_number",
        "street_name", "address_line3", "address_line4", "town_city",
        "county", "country", "move_in_date", "previous_addresses",
    ],
    "employment": [
        "country_domiciled", "resident_for_tax", "national_insurance_number",
        "employment_status", "desired_retirement_age", "occupation",
        "employer", "employment_started", "highest_rate_of_tax_paid",
    ],
    "health_details": [
        "current_state_of_health", "state_of_health_explanation", "smoker",
        "cigarettes_per_day", "smoker_since", "long_term_care_needed",
        "long_term_care_explanation", "will", "will_information",
        "power_of_attorney", "attorney_details",
    ],
    # … add other sections (pensions, loans, etc.) here if needed
}

SECTION_ORDER = list(SCHEMA.keys())  # we’ll still start with personal_details
DIGRESSION_INTERVAL = 4               # every N turns we *try* a digression
RNG = random.Random()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _remaining_subkeys(section: str, truth: Dict[str, Dict[str, str]]) -> List[str]:
    """Return sub‑keys in *section* that are still empty in *truth*."""
    current = truth.get(section, {})
    return [k for k in SCHEMA[section] if k not in current or current[k] == ""]


def _choose_next_section(truth: Dict[str, Dict[str, str]]) -> Tuple[str, List[str]]:
    """Pick a section that still has missing sub‑keys and return (section, subkeys)."""
    outstanding = [s for s in SECTION_ORDER if _remaining_subkeys(s, truth)]
    if not outstanding:
        return "", []
    section = outstanding[0] if truth == {} else RNG.choice(outstanding)
    subkeys = _remaining_subkeys(section, truth)
    RNG.shuffle(subkeys)
    # ask only a slice (2‑4) at a time for natural pacing
    ask_now = subkeys[: RNG.randint(2, min(4, len(subkeys)))]
    return section, ask_now

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_section_prompt(
    last_lines: str,
    section: str,
    subkeys: List[str],
) -> str:
    """Prompt asking the model to collect *subkeys* from *section*."""
    keys_formatted = "\n".join(f"- {k}" for k in subkeys)
    return dedent(
        f"""
        You are a friendly CERTIFIED FINANCIAL PLANNER having a discovery call.
        Continue the conversation naturally based on the last lines:
        ---
        {last_lines.strip()}
        ---
        **Goal right now**: obtain the following details (in *this* order):
        {keys_formatted}

        **Instructions**
        1. Ask one question at a time; react to answers.
        2. Do *not* ask for any other data yet.
        3. End with a short acknowledgement (no summary).

        **Output JSON wrapper** (after the dialogue):
        {{
          "conversation_chunk": "<the dialogue you just wrote>",
          "structured_data": {{
            "{section}": {{ "{subkeys[0]}": "…", "{subkeys[1] if len(subkeys)>1 else ''}": "…" }}
          }}
        }}
        """
    )


def _build_digression_prompt(
    last_lines: str,
    truth: Dict[str, Dict[str, str]],
    explained_keys: List[str],
) -> str:
    """Build a prompt where the adviser gives an explanatory monologue.

    *explained_keys* keeps track of JSON paths already used for digressions so we
    don’t repeat them.
    """
    # Flatten truth into paths like "personal_details.nationality"
    flat: List[str] = []
    for sec, kv in truth.items():
        for k, v in kv.items():
            if v:  # only filled values are meaningful
                flat.append(f"{sec}.{k}")
    remaining_topics = [p for p in flat if p not in explained_keys]
    if not remaining_topics:
        return ""  # nothing to digress about
    topic = RNG.choice(remaining_topics)
    explained_keys.append(topic)

    section, subkey = topic.split(".")
    value = truth[section][subkey]

    return dedent(
        f"""
        You are still the financial adviser.  Take a pause from questioning and
        deliver a **concise but instructive digression** (~120–200 words)
        connected to the client’s previously shared information.

        • Context lines:
        {last_lines.strip()}
        • Relevant fact: `{subkey}` = "{value}" (from section "{section}")

        **Instructions**
        1. Speak *only the adviser* this turn (no client replies).
        2. Explain why this fact matters (e.g., tax residency, retirement
           planning, health, risk tolerance, etc.).  Use lay language.
        3. Do *not* re‑ask for that fact; do *not* discuss any topic you’ve
           already digressed on.
        4. End with an inviting hand‑off question that sets up the next section.

        **Output JSON wrapper** (after the monologue):
        {{
          "conversation_chunk": "<your paragraph here>",
          "digression_on": "{topic}"
        }}
        """
    )


def build_prompt(
    last_lines: str,
    truth: Dict[str, Dict[str, str]],
    turn_index: int,
    explained_keys: List[str],
) -> str:
    """High‑level prompt selector."""
    # Decide whether we digress
    wants_digress = turn_index > 0 and turn_index % DIGRESSION_INTERVAL == 0
    if wants_digress:
        prompt = _build_digression_prompt(last_lines, truth, explained_keys)
        if prompt:  # we still may fall back if nothing to say
            return prompt
    # Otherwise collect next chunk of data
    section, subkeys = _choose_next_section(truth)
    if not section:
        return "DONE"  # all sections complete
    return _build_section_prompt(last_lines, section, subkeys)

# ---------------------------------------------------------------------------
# Driver loop (simplified — integrate into your existing orchestrator)
# ---------------------------------------------------------------------------

def run_driver_loop(model, temperature=0.7, seed: int | None = None):
    """Pseudocode skeleton; integrate with your OpenAI / LLM call stack."""
    if seed is not None:
        RNG.seed(seed)

    truth: Dict[str, Dict[str, str]] = {}
    transcript: List[str] = []
    explained_keys: List[str] = []

    for turn in range(60):  # hard cap
        last_lines = "\n".join(transcript[-6:])  # a few previous lines
        prompt = build_prompt(last_lines, truth, turn, explained_keys)
        if prompt == "DONE":
            break

        # --- CALL MODEL -----------------------------------------------------
        # response = llm_call(model, prompt, temperature)
        # --------------------------------------------------------------------
        response = fake_llm_call(prompt)   # placeholder for unit tests

        # Parse JSON wrapper
        try:
            payload = json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Model did not return JSON on turn {turn}\n{response}") from exc

        transcript.append(payload["conversation_chunk"].strip())

        # Merge structured data (if any)
        if "structured_data" in payload:
            for sec, kv in payload["structured_data"].items():
                truth.setdefault(sec, {}).update(kv)

        # Exit once every sub‑key has a non‑empty value
        done = all(not _remaining_subkeys(sec, truth) for sec in SCHEMA)
        if done:
            break

    return "\n\n".join(transcript), deepcopy(truth)

# ---------------------------------------------------------------------------
# Stand‑in for an actual LLM call (for unit testing only)
# ---------------------------------------------------------------------------

def fake_llm_call(prompt: str) -> str:  # noqa: D401 – dummy helper
    """Return a deterministic fake JSON so unit tests don’t call OpenAI."""
    stub_dialogue = "Financial Adviser: <stub>\nClient 1: <stub>"
    return json.dumps({"conversation_chunk": stub_dialogue, "structured_data": {}})


if __name__ == "__main__":
    txt, js = run_driver_loop(model="gpt‑stub", seed=42, temperature=0.0)
    print("---TRANSCRIPT---\n", txt)
    print("---TRUTH JSON---\n", json.dumps(js, indent=2))
