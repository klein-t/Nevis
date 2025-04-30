"""synthetic_data_generator.py

Generate a synthetic adviser-client conversation **and** the aligned ground-truth JSON
information (CIF-style) in one shot.  The script follows the iterative prompting
scheme outlined in the accompanying design note:
   • always starts with the Personal Details / Client-1 block
   • at each turn chooses another still-empty section, shuffles its sub-keys and
     asks the LLM to reveal them in that order
   • stores the conversation chunk plus the extracted values
   • loops until every scalar field in the template JSON is filled (or until the
     maximum turns is reached)

The result is written to disk as:
   ├─ <out_dir>/transcript.md
   └─ <out_dir>/ground_truth.json

-------------
Quick start
-------------
$ export OPENAI_API_KEY=sk-...
$ python synthetic_data_generator.py  \
          --out-dir synthetic_case_01  \
          --model gpt-4o-mini          \
          --temperature 0.8            \
          --seed 123

Dependencies: openai>=1.3.5, rich (optional for colourful logging)
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import textwrap
import logging
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import openai  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("[ERROR] The 'openai' package is required.  pip install openai>=1.3.5") from exc

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional colourful prints ----------------------------------------------------
try:
    from rich import print
except ModuleNotFoundError:  # pragma: no cover
    print = lambda x: None  # noqa: E731: dummy silent logger if rich missing

# -----------------------------------------------------------------------------
# 1.  Canonical empty template --------------------------------------------------
# -----------------------------------------------------------------------------

def cif_template() -> Dict:
    """Deep‑copyable template JSON with every scalar value set to *None* (or empty)."""
    return {
        "personal_details": {
            "client_1": {
                "title": None,
                "first_name": None,
                "middle_names": None,
                "last_name": None,
                "known_as": None,
                "pronouns": None,
                "date_of_birth": None,
                "place_of_birth": None,
                "nationality": None,
                "gender": None,
                "legal_sex": None,
                "marital_status": None,
                "home_phone": None,
                "mobile_phone": None,
                "email_address": None,
            },
            "current_address": {
                "ownership_status": None,
                "postcode": None,
                "house_name_or_number": None,
                "street_name": None,
                "address_line3": None,
                "address_line4": None,
                "town_city": None,
                "county": None,
                "country": None,
                "move_in_date": None,
                "previous_addresses": [],  # list of dicts
            },
            "dependants_children": [],
        },
        "employment": {
            "client_1": {
                "country_domiciled": None,
                "resident_for_tax": None,
                "national_insurance_number": None,
                "employment_status": None,
                "desired_retirement_age": None,
                "occupation": None,
                "employer": None,
                "employment_started": None,
                "highest_rate_of_tax_paid": None,
                "notes": None,
            },
        },
        "incomes": [],
        "expenses": {
            "loan_repayments": [],
            "housing_expenses": [],
            "motoring_expenses": [],
            "personal_expenses": [],
            "professional_expenses": [],
            "miscellaneous_expenses": [],
            "notes": None,
        },
        "pensions": [],
        "savings_investments": [],
        "other_assets": [],
        "loans_mortgages": [],
        "health_details": {
            "client_1": {
                "current_state_of_health": None,
                "state_of_health_explanation": None,
                "smoker": None,
                "cigarettes_per_day": None,
                "smoker_since": None,
                "long_term_care_needed": None,
                "long_term_care_explanation": None,
                "will": None,
                "information_about_will": None,
                "power_of_attorney": None,
                "attorney_details": None,
            },
        },
        "protection_policies": [],
        "objectives": None,
    }

# -----------------------------------------------------------------------------
# 2.  Prompt builder -----------------------------------------------------------
# -----------------------------------------------------------------------------

def build_prompt(last_lines: str, section_path: str, subkeys: List[str]) -> str:
    """Assemble the few‑shot instructions + section request for this turn."""
    logger.debug(f"Building prompt for section: {section_path}, with subkeys: {subkeys}")
    
    if not subkeys:
        logger.error(f"Empty subkeys list for section {section_path}")
        raise ValueError(f"Empty subkeys list for section {section_path}")
        
    shuffled = random.sample(subkeys, k=len(subkeys))
    logger.debug(f"Shuffled subkeys: {shuffled}")
    
    bullets = "\n".join(f"- {k}" for k in shuffled)
    
    # Handle template based on number of items
    json_template = ""
    if len(shuffled) == 1:
        json_template = f'"{shuffled[0]}": "…"'
        logger.warning(f"Only one subkey for section {section_path}: {shuffled[0]}")
    elif len(shuffled) >= 2:
        json_template = f'"{shuffled[0]}": "…", "{shuffled[1]}": "…" /* etc. */'
    else:
        # This should never happen due to the check above
        logger.critical(f"Unexpected empty shuffled list for {section_path}")
        
    return textwrap.dedent(
        f"""
        You are generating the **next** 1‑3 minutes of a discovery call between a
        financial adviser and the prospective client(s).  Continue naturally from
        the lines below (do *not* repeat them verbatim):
        ---
        {last_lines or '<start of call>'}
        ---

        This turn must focus on **{section_path}**.  Make sure the dialogue uncovers
        *all* the following details **in exactly this order** – keep the order even
        if it feels unusual:
        {bullets}

        After the dialogue, return a JSON object *only* with the shape:
        {{
          "conversation_chunk": "<the dialogue you wrote>",
          "structured_data": {{
            "{section_path}": {{
              {json_template}
            }}
          }}
        }}

        Keep the response from the client casual and from time to time verbose. Occasionally, add superfluous details when applicable.
        Do not wrap the JSON in markdown fences; do not add comments; make sure
        it parses with `json.loads`.
        """
    )

# -----------------------------------------------------------------------------
# 3.  Helper: walk & merge ------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_subdict(root: Dict, path: str) -> Dict:
    """Return the mutable sub‑dict at dotted *path*.  Creates intermediate levels."""
    keys = path.split(".")
    node = root
    for k in keys:
        if k not in node or node[k] is None:
            node[k] = {}
        node = node[k]
    return node


def merge_into(master: Dict, section_path: str, payload: Dict) -> None:
    """Overwrite only *null* values in *master* with *payload* (non‑destructive)."""
    target = _get_subdict(master, section_path)
    for k, v in payload.items():
        if (isinstance(target.get(k), (type(None), list)) and not target[k]) or target[k] is None:
            target[k] = v


# -----------------------------------------------------------------------------
# 4.  The main driver loop ------------------------------------------------------
# -----------------------------------------------------------------------------

def run_driver_loop(model: str = "gpt-4o-mini", *, temperature: float = 0.8, seed: int | None = None,
                    max_turns: int = 50) -> Tuple[str, Dict]:
    """Iterate until every scalar in the JSON is filled or *max_turns* exceeded.

    Returns
    -------
    final_transcript : str
        Full conversation, chunks separated by blank lines.
    final_json : Dict
        Populated CIF‑style object.
    """
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info("No random seed provided")

    master_json = cif_template()

    # Ordered deque of section paths we want to cover (edit / extend as needed)
    sections = collections.deque([
        "personal_details.client_1",
        "personal_details.current_address",
        "employment.client_1",
        "health_details.client_1",
    ])
    logger.info(f"Initial sections queue: {list(sections)}")
    
    # Keep first deterministic, shuffle the rest
    remaining_sections = list(sections)[1:]
    random.shuffle(remaining_sections)
    sections = collections.deque([sections[0]] + remaining_sections)
    logger.info(f"Shuffled sections queue: {list(sections)}")

    transcript_chunks: List[str] = []
    last_lines = ""

    openai_client = openai.Client()

    turn = 0
    while sections and turn < max_turns:
        turn += 1
        section_path = sections.popleft()
        logger.info(f"Turn {turn}: Processing section {section_path}")
        
        # find the list of subkeys still missing (or return early if none)
        subdict = _get_subdict(master_json, section_path)
        subkeys_missing = [k for k, v in subdict.items() if v in (None, [], {})]
        logger.info(f"Subkeys missing for {section_path}: {subkeys_missing}")
        
        if not subkeys_missing:
            logger.info(f"Section {section_path} already filled - skipping")
            continue  # already filled – skip

        try:
            prompt = build_prompt(last_lines, section_path, subkeys_missing)
        except ValueError as exc:
            logger.error(f"Failed to build prompt: {exc}")
            logger.error(f"Current state of section {section_path}: {subdict}")
            # Re-append this section at the end to try again later
            sections.append(section_path)
            continue

        try:
            logger.debug(f"Calling OpenAI API with model {model}")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"OpenAI API call failed at turn {turn}: {exc}")
            raise RuntimeError(f"OpenAI API call failed at turn {turn}: {exc}") from exc

        content = response.choices[0].message.content.strip()
        try:
            payload = json.loads(content)
            logger.debug(f"Parsed JSON response successfully")
        except json.JSONDecodeError as exc:  # pragma: no cover
            logger.error(f"Model returned invalid JSON on turn {turn}:\n{content}")
            raise ValueError(f"Model returned invalid JSON on turn {turn}:\n{content}") from exc

        chunk = payload.get("conversation_chunk", "").strip()
        data = payload.get("structured_data", {}).get(section_path, {})
        logger.debug(f"Extracted data for section {section_path}: {data}")

        if not isinstance(data, dict):
            logger.error(f"Expected dict for section {section_path}, got: {data}")
            raise ValueError(f"Expected dict for section {section_path}, got: {data}")

        # 1) accumulate dialogue
        transcript_chunks.append(chunk)
        last_lines = "\n".join(chunk.splitlines()[-3:])  # last 3 lines

        # 2) merge extracted values
        merge_into(master_json, section_path, data)
        logger.debug(f"Updated values for {section_path}")

        # 3) Re‑append section if still incomplete
        section_after_update = _get_subdict(master_json, section_path)
        incomplete = [k for k, v in section_after_update.items() if v in (None, [], {})]
        if incomplete:
            logger.info(f"Section {section_path} still has incomplete fields: {incomplete}")
            sections.append(section_path)

    logger.info(f"Driver loop completed after {turn} turns")
    return "\n\n".join(transcript_chunks), master_json


# -----------------------------------------------------------------------------
# 5.  CLI wrapper ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_case(out_dir: Path, *, model: str, temperature: float, seed: int | None) -> None:
    """Wrapper that runs the loop and writes artefacts to *out_dir*."""
    logger.info(f"Generating case in {out_dir} with model={model}, temp={temperature}, seed={seed}")
    
    transcript, truth = run_driver_loop(model=model, temperature=temperature, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.md").write_text(transcript, encoding="utf-8")
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    logger.info(f"Saved synthetic case to {out_dir.resolve()}")
    print(f"[bold green]✓[/] Saved synthetic case to {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# 6.  Main guard ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic adviser–client fact‑find data")
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_output"), help="folder for artefacts")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--num-cases", type=int, default=10, help="number of synthetic cases to generate")
    parser.add_argument("--log-file", type=str, default=None, help="file to save logs to")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    
    # Configure file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    logger.info("Starting synthetic data generation")
    logger.info(f"Arguments: {vars(args)}")

    if os.getenv("OPENAI_API_KEY") is None:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    base_dir = args.out_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created base directory: {base_dir}")
    
    for i in range(1, args.num_cases + 1):
        logger.info(f"Starting case {i} of {args.num_cases}")
        case_dir = base_dir / f"case_{i:03d}"
        # If seed is provided, increment it for each case to ensure variation
        case_seed = None if args.seed is None else args.seed + i - 1
        
        try:
            generate_case(case_dir, model=args.model, temperature=args.temperature, seed=case_seed)
            logger.info(f"Successfully generated case {i}")
        except Exception as exc:
            logger.error(f"Failed to generate case {i}: {exc}", exc_info=True)
            print(f"[bold red]✗[/] Failed to generate case {i}: {exc}")
            continue
        
    logger.info(f"Completed generation of synthetic cases")
    print(f"[bold green]✓[/] Generated {args.num_cases} synthetic cases in {base_dir.resolve()}")


if __name__ == "__main__":
    main()
