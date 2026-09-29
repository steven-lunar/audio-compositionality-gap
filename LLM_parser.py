#!/usr/bin/env python3
"""Recover intended answers from saved model responses with an LLM parser.

The parser is optional and never changes the input JSONL. Credentials are
read by the OpenAI client from ``OPENAI_API_KEY``; no key is stored here.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


TASK_PROMPTS = {
    "math": (
        "Extract the final numeric answer intended by the model. Do not "
        "recompute or correct it. Ignore surrounding reasoning and malformed "
        "tags. If no numeric answer is identifiable, mark extraction failed."
    ),
    "trivia": (
        "Extract the final short answer intended by the model. Do not judge "
        "correctness. Ignore surrounding reasoning and malformed tags. If no "
        "clear answer is identifiable, mark extraction failed."
    ),
    "asr": (
        "Extract only the transcript produced by the model. Remove wrapper "
        "commentary but do not correct or rephrase the transcript. If no "
        "transcript is identifiable, mark extraction failed."
    ),
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "extracted_answer": {"type": "string"},
                    "extraction_successful": {"type": "boolean"},
                },
                "required": [
                    "segment_index",
                    "extracted_answer",
                    "extraction_successful",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


def failed_result(num_segments: int) -> dict:
    return {
        "segments": [
            {
                "segment_index": index,
                "extracted_answer": "",
                "extraction_successful": False,
            }
            for index in range(1, num_segments + 1)
        ]
    }


class LLMParser:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    def extract(self, prediction_raw: str, task_type: str, num_segments: int) -> dict:
        if task_type not in TASK_PROMPTS:
            raise ValueError(f"Unsupported task type: {task_type}")
        if not isinstance(prediction_raw, str) or not prediction_raw.strip():
            return failed_result(num_segments)

        segment_instruction = (
            f"Return exactly one entry for each of {num_segments} segment(s), "
            f"indexed 1 through {num_segments}. A segment answer may contain "
            "multiple entities; preserve the complete answer for that segment."
        )
        system_prompt = f"{TASK_PROMPTS[task_type]} {segment_instruction}"
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f'Raw model response:\n"""\n{prediction_raw}\n"""',
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction_result",
                            "schema": EXTRACTION_SCHEMA,
                            "strict": True,
                        },
                    },
                )
                return json.loads(response.choices[0].message.content)
            except Exception as exc:  # API errors vary by client version.
                last_exception = exc
                error = str(exc).lower()
                transient = any(
                    marker in error
                    for marker in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "connection",
                    )
                )
                if not transient or attempt == self.max_retries - 1:
                    break
                delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"[WARN] parser attempt {attempt + 1} failed; retrying in {delay:.1f}s")
                time.sleep(delay)
        print(f"[ERROR] parser failed: {last_exception}")
        return failed_result(num_segments)


def extract_file(
    input_file: Path,
    output_file: Path,
    task_type: str,
    num_segments: int,
    model: str,
) -> None:
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output must differ; raw predictions are immutable.")
    parser = LLMParser(model=model)
    processed_ids = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    processed_ids.add(json.loads(line)["concat_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    for record in tqdm(records, desc="Parsing saved responses"):
        if record.get("concat_id") in processed_ids:
            continue
        prediction_raw = record.get("prediction_raw")
        extraction = parser.extract(prediction_raw, task_type, num_segments)
        by_index = {row["segment_index"]: row for row in extraction["segments"]}
        record["llm_extracted_answers"] = [
            by_index.get(index, {}).get("extracted_answer", "")
            for index in range(1, num_segments + 1)
        ]
        record["llm_extraction_successful"] = [
            bool(by_index.get(index, {}).get("extraction_successful", False))
            for index in range(1, num_segments + 1)
        ]
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", required=True, choices=TASK_PROMPTS)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--num-segments", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    output_file = args.output_file or args.input_file.with_name(
        f"{args.input_file.stem}_parsed.jsonl"
    )
    extract_file(
        args.input_file,
        output_file,
        args.task_type,
        args.num_segments,
        args.model,
    )


if __name__ == "__main__":
    main()
