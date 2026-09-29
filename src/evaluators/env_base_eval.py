"""
evaluate_dual_bgnoise_snr.py

Adaptation of evaluate_dual_env_snr for the bg_noise_builder.py output
(gender + emotion TTS synthesis, THEN background-noise mixed in at a given
SNR -- concat_id prefix "bgnoise_math_"/"bgnoise_trivia_", cue_type
"bg_noise"). Same atomic task -- classify the background sound in each of
the two segments, in order, tags-only, no reasoning -- just adapted to this
dataset's metadata shape.

SCHEMA DIFFERENCES FROM evaluate_dual_env_snr, and how each is handled:
  1. NO "sequence_index" field on segments. bg_noise_builder.py (like
     every other builder in this project -- gender_builder.py,
     emotion_builder.py) writes record["sequence"] already in actual
     playback order (index 0 = first-spoken segment, index 1 =
     second-spoken segment) -- there is nothing to sort by. Ground truth
     is read directly from record["sequence"] in list order, instead of
     sorted(record["sequence"], key=lambda s: s["sequence_index"]).
  2. Segments also carry "gender" and "emotion" (left over from the
     emotion-TTS step that produced the speech before noise was mixed
     in). These are IRRELEVANT to this classification task and are never
     mentioned in the prompt -- the model is only ever asked to identify
     the background sound, exactly as before. They ARE included in the
     output record as informational-only fields (per_segment_gender /
     per_segment_emotion, plus the top-level target_bg_category /
     distractor_bg_category already on the record), purely so a later
     secondary analysis (e.g. "does background-sound classification
     accuracy vary with the speaker's emotion") has the data available
     without needing to re-join against the metadata file -- consistent
     with how the rest of this project always keeps secondary axes
     visible rather than silently discarding them. None of this affects
     grading.
  3. TAGS: kept as <event_1>/<event_2> (not <ans_1>/<ans_2>) -- see the
     chat answer for why -- so extract_dual_environment_events and the
     rest of the "atomic environmental classification" family stay
     unchanged and directly comparable across dataset variants.

Everything else (TARGET_CATEGORIES-constrained prompt, checkpoint/resume
on concat_id, extract_dual_environment_events usage, is_correct_1/2 +
correct_count + is_exact_match/is_partial_match grading, append-mode
output) is unchanged from evaluate_dual_env_snr.
"""

import json
import os

from tqdm import tqdm
from src.evaluators.extractor import *

TARGET_CATEGORIES = [
    "rain", "sea_waves", "crickets", "chirping_birds", "wind" 
]


def evaluate_dual_bgnoise_snr(model, dataset_dir, metadata_path, output_path):
    """
    Atomic environmental-sound classification on bg_noise_builder.py's
    output (gender+emotion-synthesized speech, background noise mixed in
    at a given SNR). Ground truth comes directly from each segment's
    bg_category field in record["sequence"], already in playback order
    (no sequence_index field exists in this schema -- see module
    docstring point 1).
    """
    print("Starting Dual Environmental Sound Classification (bg_noise_builder.py data)")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]

    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                except json.JSONDecodeError:
                    continue

    categories_str = ", ".join(TARGET_CATEGORIES)

    for record in tqdm(records, desc="Evaluating Dual Env (bg_noise)"):
        concat_id = record["concat_id"]

        if concat_id in processed_ids:
            continue

        audio_path = os.path.join(dataset_dir, record["audio_path"])

        # Ground truth from each segment's bg_category, in the order
        # they already appear in record["sequence"] -- this schema has
        # no sequence_index field to sort by (see module docstring).
        ordered_segments = record["sequence"]
        ground_truth = [seg["bg_category"].lower() for seg in ordered_segments]

        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct trivia or math questions "
            f"played sequentially, each mixed with a different background sound.\n"
            f"Identify the background sound present in each segment, in the order they appear.\n"
            f"CRITICAL INSTRUCTION: You MUST choose your answers ONLY from the following list of categories:\n"
            f"[{categories_str}]\n\n"
            f"You MUST wrap your answer for the first segment's background sound inside <event_1> and </event_1> tags.\n"
            f"You MUST wrap your answer for the second segment's background sound inside <event_2> and </event_2> tags.\n"
            f"Do not output anything else."
        )

        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_list = extract_dual_environment_events(prediction_raw)

        is_correct_1 = (prediction_list[0] == ground_truth[0])
        is_correct_2 = (prediction_list[1] == ground_truth[1])
        correct_count = int(is_correct_1) + int(is_correct_2)
        is_exact_match = (correct_count == 2)
        is_partial_match = (correct_count == 1)

        result_dict = {
            "concat_id": concat_id,
            "ground_truth": ground_truth,
            "prediction_raw": prediction_raw,
            "extracted_answers": prediction_list,
            "is_correct_1": is_correct_1,
            "is_correct_2": is_correct_2,
            "correct_count": correct_count,
            "is_exact_match": is_exact_match,
            "is_partial_match": is_partial_match,
            # Informational only, not used for grading -- see module
            # docstring point 2.
            "target_bg_category": record.get("target_bg_category"),
            "distractor_bg_category": record.get("distractor_bg_category"),
            "per_segment_gender": [seg.get("gender") for seg in ordered_segments],
            "per_segment_emotion": [seg.get("emotion") for seg in ordered_segments],
        }

        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")