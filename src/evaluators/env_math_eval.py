"""
env_math_eval.py

Adaptation of env_math_eval.py's CONDITIONAL/COMPOSITIONAL functions
(evaluate_conditional_env_math / evaluate_conditional_env_math_cot) for
bg_noise_builder.py's output. evaluate_single_env_math (the atomic,
single-segment baseline) is untouched/out of scope, per your request --
only the two-segment, background-noise-conditioned select task is
adapted here. Written as new, separately-named functions rather than
edited in place, matching how every other cue-type adaptation in this
project has been handled (gender_eval.py -> emotion_eval.py,
evaluate_dual_env_snr -> evaluate_dual_bgnoise_snr, etc.) -- the originals
stay usable for whatever dataset they were built for.

SCHEMA DIFFERENCES FROM env_math_eval.py, and how each is handled:
  1. TARGET SELECTION: the original used random.Random(concat_id).choice
     at EVALUATION time. bg_noise_builder.py's records already carry a
     pre-assigned, fully balanced target_index/target_bg_category (see
     that script's target + distractor balancing) -- read directly
     instead, with a fallback to the old random-choice convention for any
     metadata that somehow lacks target_index (same defensive pattern
     already used in gender_eval.py/emotion_eval.py).
  2. get_expected_answer is copied UNCHANGED -- its existing defensive
     fallback chain (expected_answer / expected_answers / answer.text)
     already covers bg_noise_builder.py's "expected_answer" field with no
     modification needed.
  3. bg_category and question field names are already identical between
     schemas -- no renaming needed for those two.
  4. distractors are now selected by INDEX (`if i != target_index`)
     rather than by `!=` object comparison -- a small robustness
     improvement already standard elsewhere in this project, avoiding any
     edge case with two structurally-identical segment dicts.
  5. EXTRA INFORMATIONAL FIELDS: bg_noise_builder.py's segments also
     carry gender/emotion/snr_db (left over from the emotion-TTS-then-
     noise-mixing pipeline that produced this audio). These are NEVER
     mentioned in the prompt and never affect grading -- the cue is still
     background-noise category ONLY, same "isolate one axis" principle
     used throughout this project -- but they're surfaced in the output
     record (target_gender, target_emotion, snr_db, and gender/emotion
     per distractor) purely so later secondary analysis has them without
     needing to re-join against the metadata file.
  6. Output field renamed target_bg -> target_bg_category (and
     matched_distractor_bg -> matched_distractor_bg_category) to match
     bg_noise_builder.py's own top-level field naming convention
     (target_bg_category/distractor_bg_category), rather than keeping the
     shorter name the original function happened to use.

Prompt wording is otherwise UNCHANGED from env_math_eval.py -- the cue,
framing, and tag contract are identical; only the data plumbing feeding
into it changed.
"""

import os
import json
import random
from tqdm import tqdm
from src.evaluators.extractor import *


def get_expected_answer(segment):
    """
    Dynamically extracts the ground truth answer string from various schema
    formats. Copied unchanged from env_math_eval.py -- already compatible
    with bg_noise_builder.py's "expected_answer" field.
    Supports:
    1. segment["expected_answer"] (string or list)
    2. segment["answer"]["text"] (list)
    """
    ans = segment.get("expected_answer") or segment.get("expected_answers")

    if ans is None and "answer" in segment and isinstance(segment["answer"], dict):
        ans = segment["answer"].get("text")

    if isinstance(ans, list):
        return str(ans[0]).strip() if ans else ""

    return str(ans).strip() if ans is not None else ""


def evaluate_conditional_env_math(model, dataset_dir, metadata_path, output_path):
    print(f"\n--- Starting Concatenated Audio (Background-Noise-Conditioned, bg_noise_builder.py data) Evaluation ---")

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

    for record in tqdm(records, desc="Evaluating Conditional Math (bg_noise)"):
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
        ground_truth_ans = get_expected_answer(target_segment)

        # Distractors selected by index rather than `!=` object comparison.
        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]

        # Prompt UNCHANGED from evaluate_conditional_env_math -- cue is
        # background noise only; gender/emotion (also present in this
        # richer dataset's segments) are deliberately never mentioned.
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct math problems played sequentially, "
            f"each mixed with a different background noise.\n"
            f"Find the math problem that is accompanied by the '{target_bg}' sound and calculate its answer.\n"
            f"Output ONLY the final number inside <ans> and </ans> tags. "
            f"Do not output any reasoning, equations, or other text."
        )

        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_clean = extract_math_answer(prediction_raw)

        is_correct = (ground_truth_ans == prediction_clean)

        is_distracted = False
        matched_distractor_bg = None

        if not is_correct:
            for dist in distractors:
                dist_ans = get_expected_answer(dist)
                if prediction_clean == dist_ans:
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
            "target_question": target_segment["question"],
            "ground_truth": ground_truth_ans,
            "prediction_raw": prediction_raw,
            "prediction_clean": prediction_clean,
            "is_correct": is_correct,
            "error_type": error_type,
            "is_distracted": is_distracted,
            "matched_distractor_bg_category": matched_distractor_bg,
            "all_distractors_info": [
                {
                    "bg_category": d["bg_category"],
                    "answer": get_expected_answer(d),
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

        print(f"\n[Concat ID: {concat_id} | Target: {target_bg}] GT: {ground_truth_ans} | Pred: {prediction_clean} | Correct: {is_correct}")


def evaluate_conditional_env_math_cot(model, dataset_dir, metadata_path, output_path):
    """
    CoT variant, mirroring evaluate_conditional_env_math_cot exactly
    (prompt only differs by the added step-by-step reasoning instruction)
    -- same target/distractor/output-field adaptation as
    evaluate_conditional_bgnoise_math above.
    """
    print(f"\n--- Starting Concatenated Audio (Background-Noise-Conditioned, CoT, bg_noise_builder.py data) Evaluation ---")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]

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

    for record in tqdm(records, desc="Evaluating Conditional Math (bg_noise, CoT)"):
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
        ground_truth_ans = get_expected_answer(target_segment)

        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]

        # CoT prompt: identical structure to evaluate_conditional_env_math_cot.
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct math problems played sequentially, "
            f"each mixed with a different background noise.\n"
            f"Think through this step by step:\n"
            f"1. Identify which segment is accompanied by the '{target_bg}' background sound.\n"
            f"2. State the math problem spoken in that segment.\n"
            f"3. Compute the answer to that math problem.\n"
            f"After your reasoning, output ONLY the final number inside <ans> and </ans> tags.\n"
            f"The <ans> tag must contain only the final numeric answer -- no reasoning inside the tag itself."
        )

        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_clean = extract_math_answer(prediction_raw)

        is_correct = (ground_truth_ans == prediction_clean)

        is_distracted = False
        matched_distractor_bg = None

        if not is_correct:
            for dist in distractors:
                dist_ans = get_expected_answer(dist)
                if prediction_clean == dist_ans:
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
            "target_question": target_segment["question"],
            "ground_truth": ground_truth_ans,
            "prediction_raw": prediction_raw,
            "prediction_clean": prediction_clean,
            "is_correct": is_correct,
            "error_type": error_type,
            "is_distracted": is_distracted,
            "matched_distractor_bg_category": matched_distractor_bg,
            "all_distractors_info": [
                {
                    "bg_category": d["bg_category"],
                    "answer": get_expected_answer(d),
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

        print(f"\n[Concat ID: {concat_id} | Target: {target_bg}] GT: {ground_truth_ans} | Pred: {prediction_clean} | Correct: {is_correct}")


if __name__ == "__main__":
    pass