"""
bgnoise_trivia_qa_eval.py

Adaptation of env_trivia_qa_eval.py's two functions
(evaluate_conditional_env_trivia / evaluate_conditional_env_trivia_cot) for
bg_noise_builder.py's output -- same adaptation already applied to the math
side in bgnoise_math_eval.py (evaluate_conditional_bgnoise_math /
_cot). See that file's docstring for the full rationale; summarized here:

SCHEMA DIFFERENCES FROM env_trivia_qa_eval.py, and how each is handled:
  1. TARGET SELECTION: the original used random.Random(concat_id).choice at
     EVALUATION time. bg_noise_builder.py's records already carry a
     pre-assigned, balanced target_index/target_bg_category -- read
     directly instead, with a fallback to the old random-choice convention
     for any metadata that somehow lacks target_index.
  2. distractors are now selected by INDEX (`i != target_index`) rather
     than by `!=` object comparison.
  3. bg_category, question, and expected_answers field names are already
     identical between schemas -- no renaming needed for those.
  4. EXTRA INFORMATIONAL FIELDS: bg_noise_builder.py's segments also carry
     gender/emotion/snr_db (left over from the emotion-TTS-then-noise-
     mixing pipeline). Never mentioned in the prompt, never affect
     grading -- the cue is still background-noise category only -- but
     surfaced in the output record (target_gender, target_emotion,
     snr_db, and gender/emotion per distractor) for later secondary
     analysis, matching bgnoise_math_eval.py.
  5. Output field renamed target_bg -> target_bg_category and
     matched_distractor_bg -> matched_distractor_bg_category, matching
     bg_noise_builder.py's own top-level field naming and
     bgnoise_math_eval.py's convention.

normalize_answer, the whisper normalizer setup, and extract_trivia_answer
usage are all copied UNCHANGED -- purely lexical answer matching, no
schema dependency. Prompt wording is otherwise unchanged in both the
base and CoT variant from env_trivia_qa_eval.py.

Written as new, separately-named functions (evaluate_conditional_bgnoise_trivia /
_cot) rather than edited in place, matching how every other adaptation in
this project has been handled.
"""

import os
import json
import random
import re
import string
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
from src.evaluators.extractor import *

# Initialize normalizer globally
whisper_normalizer = EnglishTextNormalizer()


def normalize_answer(s):
    """
    Hybrid Normalization for Audio-LLM QA. Copied unchanged from
    env_trivia_qa_eval.py -- purely lexical, no schema dependency.
    """
    if not isinstance(s, str):
        return ""

    s = whisper_normalizer(s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_conditional_env_trivia(model, dataset_dir, metadata_path, output_path):
    print("\n--- Starting Concatenated Audio (Background-Noise-Conditioned, bg_noise_builder.py data) Trivia-QA Evaluation ---")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]

    # ==========================================
    # Checkpoint and resume logic
    # ==========================================
    processed_ids = set()
    correct_count = 0
    total_tested = 0

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])

                    total_tested += 1
                    if data.get("is_correct"):
                        correct_count += 1
                except json.JSONDecodeError:
                    continue

    if processed_ids:
        print(f"Detected historical progress! {len(processed_ids)} tests completed.")
        print(f"Current historical accuracy: {correct_count/total_tested:.2%} ({correct_count}/{total_tested})")
        print(f"Automatically skipping completed items...\n")
    # ==========================================

    for record in tqdm(records, desc="Evaluating Conditional Trivia-QA (bg_noise)"):
        concat_id = record["concat_id"]

        if concat_id in processed_ids:
            continue

        audio_path = os.path.join(dataset_dir, record["audio_path"])

        # Target is pre-assigned and balanced by bg_noise_builder.py --
        # read it directly instead of a per-item random choice. Fall back
        # to the old random.Random(concat_id) convention for any metadata
        # that somehow lacks target_index.
        if "target_index" in record:
            target_index = record["target_index"]
            target_segment = record["sequence"][target_index]
        else:
            rng = random.Random(concat_id)
            target_segment = rng.choice(record["sequence"])
            target_index = record["sequence"].index(target_segment)

        target_bg = target_segment["bg_category"]

        # Extract the list of ground truth answers and normalize them
        gt_answers_raw = target_segment.get("expected_answers", [])
        gt_answers_norm = [normalize_answer(str(ans)) for ans in gt_answers_raw]

        # Distractors selected by index rather than `!=` object comparison.
        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]

        # Prompt UNCHANGED from evaluate_conditional_env_trivia -- cue is
        # background noise only; gender/emotion (also present in this
        # richer dataset's segments) are deliberately never mentioned.
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct trivia questions played sequentially, "
            f"each mixed with a different background noise.\n"
            f"Find the trivia question that is accompanied by the '{target_bg}' sound and answer it.\n"
            f"CRITICAL INSTRUCTION: Provide the absolute shortest possible answer. Output ONLY the core entity, name, or word.\n"
            f"Output the concise answer inside <ans> and </ans> tags. "
            f"Do not output any reasoning or other text."
        )

        prediction_raw = model.query(audio_path, prompt)
        prediction_extracted = extract_trivia_answer(prediction_raw)
        prediction_norm = normalize_answer(prediction_extracted)

        # Exact match using the normalized sets
        is_correct = prediction_norm in gt_answers_norm if prediction_norm else False

        is_distracted = False
        matched_distractor_bg = None

        # Check if the model was tricked by a distractor problem
        if not is_correct and prediction_norm:
            for dist in distractors:
                dist_answers_raw = dist.get("expected_answers", [])
                dist_answers_norm = [normalize_answer(str(ans)) for ans in dist_answers_raw]

                if prediction_norm in dist_answers_norm:
                    is_distracted = True
                    matched_distractor_bg = dist["bg_category"]
                    break

        if is_correct:
            error_type = "Correct"
        elif is_distracted:
            error_type = "Distracted"
        else:
            error_type = "Hallucination"

        if is_correct:
            correct_count += 1
        total_tested += 1

        result_dict = {
            "concat_id": concat_id,
            "target_index": target_index,
            "target_bg_category": target_bg,
            "target_question": target_segment.get("question", ""),
            "ground_truth_raw": gt_answers_raw,
            "ground_truth_norm": gt_answers_norm,
            "prediction_raw": prediction_raw,
            "prediction_extracted": prediction_extracted,
            "prediction_norm": prediction_norm,
            "is_correct": is_correct,

            "error_type": error_type,
            "is_distracted": is_distracted,
            "matched_distractor_bg_category": matched_distractor_bg,

            "all_distractors_info": [
                {
                    "bg_category": d["bg_category"],
                    "expected_answers": d.get("expected_answers", []),
                    # Informational only -- see module docstring.
                    "gender": d.get("gender"),
                    "emotion": d.get("emotion"),
                } for d in distractors
            ],
            # Informational only, not used for grading -- see module docstring.
            "target_gender": target_segment.get("gender"),
            "target_emotion": target_segment.get("emotion"),
            "snr_db": target_segment.get("snr_db"),
        }

        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        # Select first ground truth to print for terminal readability
        display_gt = gt_answers_raw[0] if gt_answers_raw else ""
        print(f"\n[Concat ID: {concat_id} | Target: {target_bg}] GT: {display_gt} | Pred: {prediction_extracted} | Correct: {is_correct}")


def evaluate_conditional_env_trivia_cot(model, dataset_dir, metadata_path, output_path):
    """
    CoT variant, mirroring evaluate_conditional_env_trivia_cot exactly
    (prompt only differs by the added step-by-step reasoning instruction)
    -- same target/distractor/output-field adaptation as
    evaluate_conditional_bgnoise_trivia above.
    """
    print("\n--- Starting Concatenated Audio (Background-Noise-Conditioned, CoT, bg_noise_builder.py data) Trivia-QA Evaluation ---")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]

    # ==========================================
    # Checkpoint and resume logic
    # ==========================================
    processed_ids = set()
    correct_count = 0
    total_tested = 0

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])

                    total_tested += 1
                    if data.get("is_correct"):
                        correct_count += 1
                except json.JSONDecodeError:
                    continue

    if processed_ids:
        print(f"Detected historical progress! {len(processed_ids)} tests completed.")
        print(f"Current historical accuracy: {correct_count/total_tested:.2%} ({correct_count}/{total_tested})")
        print(f"Automatically skipping completed items...\n")
    # ==========================================

    for record in tqdm(records, desc="Evaluating Conditional Trivia-QA (bg_noise, CoT)"):
        concat_id = record["concat_id"]

        if concat_id in processed_ids:
            continue

        audio_path = os.path.join(dataset_dir, record["audio_path"])

        if "target_index" in record:
            target_index = record["target_index"]
            target_segment = record["sequence"][target_index]
        else:
            rng = random.Random(concat_id)
            target_segment = rng.choice(record["sequence"])
            target_index = record["sequence"].index(target_segment)

        target_bg = target_segment["bg_category"]

        gt_answers_raw = target_segment.get("expected_answers", [])
        gt_answers_norm = [normalize_answer(str(ans)) for ans in gt_answers_raw]

        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]

        # CoT prompt: identical structure to evaluate_conditional_env_trivia_cot.
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct trivia questions played sequentially, "
            f"each mixed with a different background noise.\n"
            f"Think through this step by step:\n"
            f"1. Identify which segment is accompanied by the '{target_bg}' background sound.\n"
            f"2. State the trivia question asked in that segment.\n"
            f"3. Recall and determine the answer to that trivia question.\n"
            f"After your reasoning, provide the absolute shortest possible answer: output ONLY the core entity, name, or word "
            f"inside <ans> and </ans> tags.\n"
            f"The <ans> tag must contain only the concise final answer -- no reasoning inside the tag itself."
        )

        prediction_raw = model.query(audio_path, prompt)
        prediction_extracted = extract_trivia_answer(prediction_raw)
        prediction_norm = normalize_answer(prediction_extracted)

        is_correct = prediction_norm in gt_answers_norm if prediction_norm else False

        is_distracted = False
        matched_distractor_bg = None

        if not is_correct and prediction_norm:
            for dist in distractors:
                dist_answers_raw = dist.get("expected_answers", [])
                dist_answers_norm = [normalize_answer(str(ans)) for ans in dist_answers_raw]

                if prediction_norm in dist_answers_norm:
                    is_distracted = True
                    matched_distractor_bg = dist["bg_category"]
                    break

        if is_correct:
            error_type = "Correct"
        elif is_distracted:
            error_type = "Distracted"
        else:
            error_type = "Hallucination"

        if is_correct:
            correct_count += 1
        total_tested += 1

        result_dict = {
            "concat_id": concat_id,
            "target_index": target_index,
            "target_bg_category": target_bg,
            "target_question": target_segment.get("question", ""),
            "ground_truth_raw": gt_answers_raw,
            "ground_truth_norm": gt_answers_norm,
            "prediction_raw": prediction_raw,
            "prediction_extracted": prediction_extracted,
            "prediction_norm": prediction_norm,
            "is_correct": is_correct,

            "error_type": error_type,
            "is_distracted": is_distracted,
            "matched_distractor_bg_category": matched_distractor_bg,

            "all_distractors_info": [
                {
                    "bg_category": d["bg_category"],
                    "expected_answers": d.get("expected_answers", []),
                    "gender": d.get("gender"),
                    "emotion": d.get("emotion"),
                } for d in distractors
            ],
            "target_gender": target_segment.get("gender"),
            "target_emotion": target_segment.get("emotion"),
            "snr_db": target_segment.get("snr_db"),
            "prompt_variant": "cot",
        }

        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        # Select first ground truth to print for terminal readability
        display_gt = gt_answers_raw[0] if gt_answers_raw else ""
        print(f"\n[Concat ID: {concat_id} | Target: {target_bg}] GT: {display_gt} | Pred: {prediction_extracted} | Correct: {is_correct}")


if __name__ == "__main__":
    pass