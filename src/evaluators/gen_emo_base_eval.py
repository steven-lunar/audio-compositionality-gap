"""
evaluate_dual_gender_emotion.py

Atomic speaker-attribute identification -- gender and emotion -- on
emotion_builder.py's output (the same dataset already consumed by
emotion_eval.py / evaluate_conditional_asr_emotion.py / seg_idx_eval.py's
cue_mode="emotion"). Same structure as evaluate_dual_bgnoise_snr.py: the
model hears the same two-segment audio used by the compositional
gender/emotion-cue tasks and must classify BOTH segments' attribute
values, in order, tags-only, no reasoning -- giving the atomic "can it
identify gender / can it identify emotion" data point needed for the same
kind of isolated-atomic vs. compositional-task comparison as the
background-noise classification task.

SCHEMA: no "sequence_index" field (same as bg_noise_builder.py's output)
-- record["sequence"] is already in playback order (index 0 = first-
spoken segment), so ground truth is read directly in list order, no
sorting needed.

TWO ATTRIBUTES, ONE SHARED IMPLEMENTATION: gender identification and
emotion recognition are structurally identical tasks (classify segment 1,
classify segment 2, same grading), differing only in which field is
ground truth, which fixed category list is valid, and which tag prefix is
used. _evaluate_dual_attribute holds the one real implementation;
evaluate_dual_gender and evaluate_dual_emotion are thin wrappers, per the
same "graders cannot drift apart" convention used by every other
task family in this project (gender_eval.py, emotion_eval.py, etc.). This
also means each attribute already gets its OWN separate model query and
OWN prompt (parameterized from the same template) -- gender and emotion
are never asked about in the same call here, unlike
evaluate_dual_env_gender_emotion.py's combined 3-attribute task.

TAGS: <gender_1>/<gender_2> and <emotion_1>/<emotion_2> -- NOT
<event_1>/<event_2> (that's specifically the environmental-sound-category
tag, semantically wrong here) and NOT <ans_1>/<ans_2> (this is
classification, not a QA/math answer). extract_dual_tagged_values is a
generic two-tag extractor parameterized by tag prefix, so gender and
emotion share one extraction function instead of needing two near-
identical hardcoded ones.

OUT-OF-VOCABULARY CONTENT FIX (added after real models were observed
outputting "neutral" for the emotion attribute despite it never being one
of EMOTION_CATEGORIES): a DIFFERENT failure mode from the tag-name
confusion fix below -- the tag is well-formed, but the word INSIDE it
isn't one of the categories the prompt actually listed (models appear to
fall back to their own training-time emotion vocabulary, which commonly
includes "neutral", instead of respecting the constrained list). Same
two-pronged fix already applied to evaluate_dual_env_gender_emotion.py:
  1. PROMPT: _evaluate_dual_attribute now takes an optional
     out_of_vocab_examples list -- concrete words known to be a real
     problem for THIS attribute -- and, when given, names them directly as
     a negative example (a generic "stay in the list" reminder alone was
     apparently not enough to stop this exact real failure). Gender has no
     known instance of this, so evaluate_dual_gender leaves it unset (the
     prompt still gets a general "don't invent a category" reminder, just
     without a named example); evaluate_dual_emotion passes
     ["neutral", "calm"] -- the actually-observed offender plus one likely
     sibling.
  2. TRACKING: in_vocab_1/2 -- whether each extracted answer is actually
     one of this attribute's allowed categories, kept SEPARATE from
     is_correct_1/2. An out-of-vocabulary answer is still simply graded as
     "wrong" against ground truth (no crash, no silent recovery, no
     special-casing in the is_correct_* computation) -- these fields exist
     purely so "the model ignored the category list" can be told apart
     from "the model picked a valid category but the wrong one" during
     analysis, same separate-axes convention as strict_extract's format
     axis and validate_emotion_synthesis.py's answer_set_mass_share.

Each task's OTHER attribute (emotion when evaluating gender, gender when
evaluating emotion) is included in the output as an informational-only
field (per_segment_<other attribute>), same rationale as
evaluate_dual_bgnoise_snr.py's per_segment_gender/emotion fields -- never
used for grading, just kept visible for later secondary analysis (e.g.
"does gender-identification accuracy vary with the speaker's emotion?").
bg_category is deliberately NEVER read as a ground-truth source for
either task -- in this schema it's only a backward-compat alias of gender
(see emotion_builder.py), not an independent attribute.

Everything else (TARGET_CATEGORIES-style category-constrained prompt,
checkpoint/resume on concat_id, is_correct_1/2 + correct_count +
is_exact_match/is_partial_match grading, append-mode output, no
aggregate summary printing) matches evaluate_dual_bgnoise_snr.py's
structure exactly.
"""

import json
import os
import re

from tqdm import tqdm


GENDER_CATEGORIES = ["male", "female"]
# Same 5 emotions and order used throughout this project (EMOTION_ADJ in
# emotion_eval.py / evaluate_conditional_asr_emotion.py / seg_idx_eval.py).
EMOTION_CATEGORIES = ["angry", "disgust", "fear", "happy", "sad"]
# EMOTION_CATEGORIES = ["angry", "happy", "sad"]


def extract_dual_tagged_values(raw, tag_prefix):
    """
    Generic two-tag extractor for <{tag_prefix}_1>/<{tag_prefix}_2>,
    parameterized so gender and emotion classification can share one
    function instead of two hardcoded-tag-name copies. Mirrors
    extract_dual_environment_events's behavior: lowercased, stripped
    content; None if a tag is missing or malformed.
    """
    results = []
    for i in (1, 2):
        m = re.search(
            rf"<{tag_prefix}_{i}>(.*?)</{tag_prefix}_{i}>",
            raw, re.DOTALL | re.IGNORECASE,
        )
        results.append(m.group(1).strip().lower() if m else None)
    return results


def _in_vocab_pair(extracted, categories):
    """
    Whether each of the two extracted answers is actually one of the
    allowed categories -- separate from correctness (see module docstring,
    "OUT-OF-VOCABULARY CONTENT FIX"). None (missing/malformed tag) is
    always False here, same as it's always incorrect in the is_correct_*
    computation -- this is a diagnostic split of "wrong", not an alternate
    way to be right.
    """
    cat_set = {c.lower() for c in categories}
    return [(e in cat_set) if e is not None else False for e in extracted]


def _evaluate_dual_attribute(
    model, dataset_dir, metadata_path, output_path,
    attribute, other_attribute, categories, tag_prefix,
    attribute_noun, tone_clause, task_desc,
    out_of_vocab_examples=None,
):
    """
    Shared implementation for both evaluate_dual_gender and
    evaluate_dual_emotion. See module docstring.

    out_of_vocab_examples: optional list of concrete words known to be a
    real out-of-vocabulary failure for this attribute (e.g. "neutral" for
    emotion) -- when given, the prompt names them directly as a negative
    example instead of relying on a generic reminder alone.
    """
    print(f"Starting Dual {task_desc} (emotion_builder.py data)")

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

    categories_str = ", ".join(categories)
    # Used to build a concrete correct-format example and a concrete
    # negative example below -- the first two categories in the fixed
    # list are enough for both.
    example_cat_a, example_cat_b = categories[0], categories[1]

    if out_of_vocab_examples:
        examples_str = "', '".join(out_of_vocab_examples)
        plural = "those are" if len(out_of_vocab_examples) > 1 else "that is"
        out_of_vocab_note = (
            f"Even if a different word feels like a more natural description, you must still pick "
            f"the closest match from the list given -- do NOT invent, add, or substitute a category "
            f"that is not explicitly listed above. For example, do NOT answer '{examples_str}' -- "
            f"{plural} NOT in the list above; you must pick whichever of [{categories_str}] is the "
            f"closest match instead.\n\n"
        )
    else:
        out_of_vocab_note = (
            f"Even if a different word feels like a more natural description, you must still pick "
            f"the closest match from the list given -- do NOT invent, add, or substitute a category "
            f"that is not explicitly listed above.\n\n"
        )

    for record in tqdm(records, desc=f"Evaluating Dual {task_desc}"):
        concat_id = record["concat_id"]

        if concat_id in processed_ids:
            continue

        audio_path = os.path.join(dataset_dir, record["audio_path"])

        # Ground truth from each segment's attribute field, in the order
        # they already appear in record["sequence"] -- no sequence_index
        # field in this schema (see module docstring).
        ordered_segments = record["sequence"]
        ground_truth = [seg[attribute].lower() for seg in ordered_segments]

        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct trivia or math questions "
            f"played sequentially, each {tone_clause}.\n"
            f"Identify the {attribute_noun} in each segment, in the order they appear.\n"
            f"CRITICAL INSTRUCTION: You MUST choose your answers ONLY from the following list of categories:\n"
            f"[{categories_str}]\n\n"
            f"{out_of_vocab_note}"
            f"You MUST wrap your answer for the first segment's {attribute_noun} inside <{tag_prefix}_1> and </{tag_prefix}_1> tags.\n"
            f"You MUST wrap your answer for the second segment's {attribute_noun} inside <{tag_prefix}_2> and </{tag_prefix}_2> tags.\n"
            f"The tag names must be exactly '{tag_prefix}_1' and '{tag_prefix}_2' in every case -- do NOT use the "
            f"category word itself as the tag name. For example, do NOT write <{example_cat_a}> or <{example_cat_b}>; "
            f"the category word goes INSIDE the tag as plain text, not as the tag name.\n"
            f"Format Example: <{tag_prefix}_1>[answer_for_segment_1]</{tag_prefix}_1><{tag_prefix}_2>[answer_for_segment_2]</{tag_prefix}_2>\n"
            f"Do not output anything else."
        )

        # prompt = (
        #     f"Listen to the audio carefully. It contains exactly TWO distinct trivia or math questions "
        #     f"played sequentially, each {tone_clause}.\n"
        #     f"Identify the {attribute_noun} in each segment, in the order they appear.\n"
        #     f"CRITICAL INSTRUCTION: You MUST choose your answers ONLY from the following list of categories:\n"
        #     f"[{categories_str}]\n\n"
        #     f"{out_of_vocab_note}"
        #     f"You MUST wrap your answer for the first segment's {attribute_noun} inside <{tag_prefix}_1> and </{tag_prefix}_1> tags.\n"
        #     f"You MUST wrap your answer for the second segment's {attribute_noun} inside <{tag_prefix}_2> and </{tag_prefix}_2> tags.\n"
        #     f"The tag names must be exactly '{tag_prefix}_1' and '{tag_prefix}_2' in every case -- do NOT use the "
        #     f"category word itself as the tag name. For example, do NOT write <{example_cat_a}> or <{example_cat_b}>; "
        #     f"the category word goes INSIDE the tag as plain text, not as the tag name.\n"
        #     f"Correct example format: <{tag_prefix}_1>{example_cat_a}</{tag_prefix}_1><{tag_prefix}_2>{example_cat_b}</{tag_prefix}_2>\n"
        #     f"IMPORTANT: '{example_cat_a}' and '{example_cat_b}' above are only a FORMAT example, not the "
        #     f"answer -- they may or may not be correct for this audio. You must listen to THIS audio and "
        #     f"determine the actual {attribute_noun} for each segment yourself. Do not copy the example "
        #     f"words verbatim, and do not repeat any of these instructions in your response.\n"
        #     f"Do not output anything else."
        # )

        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_list = extract_dual_tagged_values(prediction_raw, tag_prefix)

        is_correct_1 = (prediction_list[0] == ground_truth[0])
        is_correct_2 = (prediction_list[1] == ground_truth[1])
        correct_count = int(is_correct_1) + int(is_correct_2)
        is_exact_match = (correct_count == 2)
        is_partial_match = (correct_count == 1)

        # Out-of-vocabulary tracking -- separate diagnostic axis, never
        # used for grading. See module docstring.
        in_vocab_1, in_vocab_2 = _in_vocab_pair(prediction_list, categories)

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
            "in_vocab_1": in_vocab_1,
            "in_vocab_2": in_vocab_2,
            f"target_{attribute}": record.get(f"target_{attribute}"),
            f"distractor_{attribute}": record.get(f"distractor_{attribute}"),
            # Informational only, not used for grading -- see module docstring.
            f"per_segment_{other_attribute}": [seg.get(other_attribute) for seg in ordered_segments],
        }

        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")


def evaluate_dual_gender(model, dataset_dir, metadata_path, output_path):
    """Atomic speaker-gender identification on both segments."""
    return _evaluate_dual_attribute(
        model, dataset_dir, metadata_path, output_path,
        attribute="gender",
        other_attribute="emotion",
        categories=GENDER_CATEGORIES,
        tag_prefix="gender",
        attribute_noun="speaker's gender",
        tone_clause="spoken by a speaker of a different gender",
        task_desc="Speaker Gender Identification",
        # No known real out-of-vocabulary failure for gender -- prompt
        # still gets the general "don't invent a category" reminder, just
        # without a named negative example.
        out_of_vocab_examples=None,
    )


def evaluate_dual_emotion(model, dataset_dir, metadata_path, output_path):
    """Atomic speaker-emotion recognition on both segments."""
    return _evaluate_dual_attribute(
        model, dataset_dir, metadata_path, output_path,
        attribute="emotion",
        other_attribute="gender",
        categories=EMOTION_CATEGORIES,
        tag_prefix="emotion",
        attribute_noun="speaker's emotional tone",
        tone_clause="spoken with a different emotional tone of voice",
        task_desc="Speaker Emotion Recognition",
        # The actually-observed real failure ("neutral"), plus one likely
        # sibling word -- see module docstring, "OUT-OF-VOCABULARY CONTENT FIX".
        out_of_vocab_examples=["neutral", "calm"],
    )