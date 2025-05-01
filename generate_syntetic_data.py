"""synthetic_data_generator.py

Generate a synthetic adviser-client conversation 
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

from gfft import gfft_template

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
        # This should never happen due to the check above, but just in case
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
        *all* the following details, batch them together if possible. Keep the conversation casual, some details can be asked in follow up questions:
        {bullets}

        IMPORTANT: Always clearly indicate who is speaking by prefixing each turn with "Adviser:" or "Client:" 
        labels. This diarization is critical for proper transcription formatting.

        IMPORTANT: When extracting data for the structured_data JSON, standardize the values to be concise and consistent.
        Follow these standardization guidelines:
        
        1. For boolean or yes/no fields:
           - Use only "yes", "no", or null values
           - Example: Convert "I haven't set up a will yet" to "no"
           - Example: Convert "Yes, I do have one" to "yes"
        
        2. For status fields (health, conditions, etc.):
           - Use standardized terms: "excellent", "good", "fair", "poor", or more specific medical terms when appropriate
           - Example: Convert "Pretty good overall, usual wear and tear" to "good"
           
        3. For explanatory fields:
           - Keep them brief but informative, 10-15 words maximum
           - When something doesn't exist, use null rather than explanations like "Hasn't been set up yet"
           
        4. For numerical fields:
           - Use plain values without units or explanatory text
           - Example: Convert "About 10 cigarettes per day" to "10"
           
        5. For date fields:
           - Use ISO format (YYYY-MM-DD) where possible, or consistent formats like "January 2020"
        
        6. When information is absent or unknown:
           - Use null values instead of phrases like "not provided" or "unknown"

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


def build_digression_prompt(last_lines: str, partial_json: Dict, already_explained: set) -> str:
    """Create a prompt for adviser to explain topics already collected but not yet explained."""
    logger.debug("Building digression prompt")
    
    # Serialize the partial JSON for the model to see what's already known
    json_dump = json.dumps(partial_json, indent=2)
    
    # Identify all filled scalar fields that haven't been explained yet
    def collect_fields(obj, prefix=""):
        """Recursively collect all non-empty scalar fields with their dotted paths."""
        fields = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    fields.extend(collect_fields(v, path))
                elif isinstance(v, list) and v and all(isinstance(item, dict) for item in v):
                    # Skip array fields for simplicity...
                    pass
                elif v not in (None, [], {}): 
                    fields.append(path)
        return fields
    
    filled_fields = set(collect_fields(partial_json))
    new_fields = filled_fields - already_explained
    logger.debug(f"Fields to explain: {new_fields}")
    
    if not new_fields:
        logger.warning("No new fields to explain in digression")
        # Return a minimal set to explain if nothing new
        new_fields = set(random.sample(list(filled_fields), min(3, len(filled_fields))))
    
    # Take 3-5 fields to explain (or fewer if less available)
    fields_to_explain = random.sample(list(new_fields), min(random.randint(3, 5), len(new_fields)))
    bullets = "\n".join(f"- {k}" for k in fields_to_explain)
    
    return textwrap.dedent(
        f"""
        Continue naturally from the last lines below (do *not* repeat them verbatim):
        ---
        {last_lines or '<start of call>'}
        ---
        
        You are a professional financial adviser conducting a fact-finding assessment with a client. 
        This is an important initial consultation where you need to build trust while demonstrating expertise.
        
        Do not ask any new personal questions in this turn.
        
        You (the adviser) now take a short pause to explain some financial concepts based on what you've learned.
        
        IMPORTANT: Always clearly indicate who is speaking by prefixing each turn with "Adviser:" or "Client:" 
        labels. This diarization is essential even in explanatory sections where the adviser speaks more.
        
        Here is what you already know in machine form:
        {json_dump}
        
        Explain, in friendly but thorough layman terms, the following points (in order):
        {bullets}
        
        Be somewhat verbose and detailed in your explanations. Use relatable examples and metaphors where appropriate.
        Draw connections between the client's specific situation and these financial concepts.
        
        Your explanation should:
        - Be conversational and include brief client acknowledgements
        - Take about 20-30 seconds of speaking time (approximately 150-200 words)
        - Use plain English while still introducing a few key financial terms (with explanations)
        - Show genuine expertise while remaining accessible to non-financial professionals
        - Occasionally reference how these concepts might affect the client's long-term financial planning
        - Always clearly label each speaker turn with "Adviser:" or "Client:" prefixes
        
        Return only JSON with the shape:
        {{
          "conversation_chunk": "<your thorough explanation plus brief client acknowledgements>"
        }}
        
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
        Populated GFFT‑style object.
    """
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info("No random seed provided")

    master_json = gfft_template()

    # Ordered deque of section paths we want to cover (edit / extend as needed)
    sections = collections.deque([
        "personal_details.client",
        "personal_details.current_address",
        "employment.client",
        "health_details.client",
        "incomes",
        "expenses.loan_repayments",
        "expenses.housing_expenses",
        "expenses.motoring_expenses",
        "expenses.personal_expenses",
        "expenses.professional_expenses",
        "expenses.miscellaneous_expenses",
        "pensions",
        "savings_investments",
        "loans_mortgages",
        "protection_policies",
    ])
    logger.info(f"Initial sections queue: {list(sections)}")
    
    # Keep first deterministic, shuffle the rest
    initial_section = sections[0] # Keep personal_details.client first
    remaining_sections = list(sections)[1:]
    random.shuffle(remaining_sections)
    sections = collections.deque([initial_section] + remaining_sections)
    logger.info(f"Shuffled sections queue: {list(sections)}")

    transcript_chunks: List[str] = []
    last_lines = ""
    
    # Track explained fields to avoid repetition
    explained_fields = set()
    
    # Counter to trigger digressions
    turns_since_last_digression = 0

    openai_client = openai.Client()

    turn = 0
    while sections and turn < max_turns:
        turn += 1
        section_path = sections.popleft()
        logger.info(f"Turn {turn}: Processing section {section_path}")
        
        # Check if this is a digression turn
        if section_path == "__DIGRESSION__":
            logger.info("Processing digression turn")
            try:
                prompt = build_digression_prompt(last_lines, master_json, explained_fields)
                
                logger.debug(f"Calling OpenAI API for digression with model {model}")
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                payload = json.loads(content)
                
                chunk = payload.get("conversation_chunk", "").strip()
                
                # Accumulate dialogue
                transcript_chunks.append(chunk)
                last_lines = "\n".join(chunk.splitlines()[-3:])  # last 3 lines
                
                # Extract fields that were just explained from the prompt
                for line in prompt.splitlines():
                    if line.strip().startswith("- "):
                        field = line.strip()[2:].strip()
                        explained_fields.add(field)
                
                # Reset digression counter
                turns_since_last_digression = 0
                logger.info("Digression turn completed")
                continue
                
            except Exception as exc:
                logger.error(f"Digression turn failed: {exc}")
                # If digression fails, just continue with normal flow
                turns_since_last_digression += 1
                continue
        
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
        
        # Extract structured data only for non-digression turns
        if section_path != "__DIGRESSION__":
            data = payload.get("structured_data", {}).get(section_path, {})
            logger.debug(f"Extracted data for section {section_path}: {data}")

            if not isinstance(data, dict):
                logger.error(f"Expected dict for section {section_path}, got: {data}")
                raise ValueError(f"Expected dict for section {section_path}, got: {data}")
                
            # Merge extracted values
            merge_into(master_json, section_path, data)
            logger.debug(f"Updated values for {section_path}")

        # 1) accumulate dialogue
        transcript_chunks.append(chunk)
        last_lines = "\n".join(chunk.splitlines()[-3:])  # last 3 lines

        # 3) Re‑append section if still incomplete
        section_after_update = _get_subdict(master_json, section_path)
        incomplete = [k for k, v in section_after_update.items() if v in (None, [], {})]
        if incomplete:
            logger.info(f"Section {section_path} still has incomplete fields: {incomplete}")
            sections.append(section_path)
            
        # Increment counter for regular turns
        turns_since_last_digression += 1
        
        # Schedule a digression after every 2 regular turns if we have new data
        if turns_since_last_digression >= 2:
            # Check if we have fields that have not been explained yet
            def collect_filled_fields(obj, prefix=""):
                """Collect all filled scalar fields with their dotted paths."""
                fields = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        path = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            fields.extend(collect_filled_fields(v, path))
                        elif v not in (None, [], {}):  # Field has a value
                            fields.append(path)
                return fields
            
            filled_fields = set(collect_filled_fields(master_json))
            if filled_fields - explained_fields:
                logger.info("Scheduling digression turn")
                sections.appendleft("__DIGRESSION__")

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


def main() -> None:  
    args = parse_args()
    
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
