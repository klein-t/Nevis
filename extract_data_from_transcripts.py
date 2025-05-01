import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict
from gfft import gfft_template

try:
    import openai
except ImportError: 
    raise SystemExit("[ERROR] The 'openai' package is required. pip install openai>=1.3.5")


# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def build_extraction_messages(transcript: str, target_schema_json: str) -> list[dict]:
    """Builds the system and user messages for the LLM call."""
    system_message_content = f"""
You are an expert data extraction AI assistant.
Your task is to carefully read the provided financial advisory conversation transcript and extract all relevant information into a structured JSON format.

Adhere strictly to the following JSON schema. Only include fields explicitly mentioned in the schema. If information for a field is not present in the transcript, use a 'null' value for that field.
Do not include any explanatory text outside the final JSON object.

Target JSON Schema:
```json
{target_schema_json}
```

Follow these standardization guidelines precisely:
1. Boolean/Yes/No fields: Use only "yes", "no", or null.
2. Status fields (health, etc.): Use standardized terms like "excellent", "good", "fair", "poor".
3. Explanatory fields: Keep concise (10-15 words max). Use null if the item doesn't exist.
4. Numerical fields: Use plain numbers without units.
5. Date fields: Use ISO format (YYYY-MM-DD) or consistent formats like "Month YYYY".
6. Absent/Unknown info: Use null.

Output *only* the final JSON object.
"""

    user_message_content = f"""
Transcript to process:
---
{transcript}
---
"""

    return [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_message_content}
    ]

def process_case(case_dir: Path, client: openai.Client, model: str, output_filename: str) -> None:
    """Processes a single case directory."""
    transcript_file = case_dir / "transcript.md"
    output_file = case_dir / output_filename

    if not transcript_file.is_file():
        logger.warning(f"Transcript file not found in {case_dir}, skipping.")
        return

    try:
        transcript_content = transcript_file.read_text(encoding="utf-8")
        logger.info(f"Read transcript from {transcript_file}")

        # Get the target schema as a JSON string
        target_schema = gfft_template()
        target_schema_json = json.dumps(target_schema, indent=2)

        messages = build_extraction_messages(transcript_content, target_schema_json)

        logger.info(f"Calling OpenAI API for {case_dir.name}...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # Pass the list of messages
            response_format={"type": "json_object"}
            # Add temperature=0 if deterministic extraction is desired
        )

        extracted_data_json = response.choices[0].message.content.strip()
        logger.info(f"Received response from OpenAI for {case_dir.name}")

        # Validate and save the JSON
        try:
            extracted_data = json.loads(extracted_data_json)
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully extracted and saved data to {output_file}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for {case_dir.name}. Response content:")
            logger.error(extracted_data_json) # Log the raw JSON separately
            # Optionally save the raw response for debugging
            error_file = case_dir / f"{output_filename}.error.txt"
            error_file.write_text(extracted_data_json, encoding="utf-8")
            logger.error(f"Raw response saved to {error_file}")

    except openai.APIError as e:
        logger.error(f"OpenAI API error processing {case_dir.name}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred processing {case_dir.name}: {e}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured data from transcripts using OpenAI.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("synthetic_output"),
        help="Directory containing the case_xxx subfolders."
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="extracted_data.json",
        help="Filename for the extracted JSON output within each case folder."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI chat model to use for extraction."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File to save logs to."
    )
    args = parser.parse_args()


    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.info("Starting data extraction process.")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output filename: {args.output_filename}")
    logger.info(f"Model: {args.model}")

    if os.getenv("OPENAI_API_KEY") is None:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    client = openai.Client()

    if not args.input_dir.is_dir():
        logger.error(f"Input directory not found: {args.input_dir}")
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    case_count = 0
    processed_count = 0
    for item in args.input_dir.iterdir():
        if item.is_dir() and item.name.startswith("case_"):
            case_count += 1
            logger.info(f"--- Processing {item.name} ---")
            try:
                process_case(item, client, args.model, args.output_filename)
                processed_count +=1
            except Exception as e: # Catch potential errors within process_case if not handled
                 logger.error(f"Critical error processing {item.name}: {e}", exc_info=True)


    logger.info(f"--- Extraction complete ---")
    logger.info(f"Found {case_count} case directories.")
    logger.info(f"Successfully processed {processed_count} cases.")

if __name__ == "__main__":
    main() 