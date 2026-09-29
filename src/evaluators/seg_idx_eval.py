"""
Segment index prediction: atomic probe of target-segment SELECTION.

The model hears the same two-segment audio used in the compositional baseline
and must output ONLY the index (1 or 2) of the segment accompanied by the
cued background sound. No transcription, no reasoning -- selection isolated.

Design rules (parity is load-bearing):
  * The prompt mirrors the compositional baseline verbatim except for the
    output demand. Target bg label ONLY -- no vocabulary, no distractor label
    (extra labels would allow solving by elimination, which the compositional
    task never permits, and would break the ladder inference).
  * Target selection uses random.Random(concat_id), identical to the baseline,
    so items are PAIRED with compositional QA / conditional-ASR runs
    (enables item-level joint analysis and the selection-verified subset).
  * flip_target=True re-runs the same items with the OTHER segment cued:
    (a) counterfactual cue-flip consistency -- predictions should flip;
    (b) main + flipped runs together are exactly position-balanced.
  * prompt_variant in {0,1,2}: information-equivalent paraphrases for the
    prompt-sensitivity robustness check.
  * cue_mode="position_concept" is a CONTROL with no acoustic-cue demand
    ("which segment is played first/second?"). Near-ceiling performance there
    shows the positional-report concept is not the bottleneck.

2AFC structure -> chance = 50%. The summary prints position-conditioned
rates, balanced accuracy, and choice bias (the d'/bias decomposition inputs).

Use one output file per condition (mode x variant x flip); resume is keyed
on concat_id within a file.

ADDED: cue_mode="emotion" for the emotion-cue datasets (emotion_builder.py
output). Mirrors "gender" mode's structure exactly, with the cue changed to
speaker emotional tone. Same "no extra vocabulary" rule applies -- the
emotion prompts never mention gender, even though emotion_builder.py's
segments also carry a gender attribute (kept out of scope here, consistent
with this mode populating only its own target_emotion/distractor_emotion
fields, same as how "gender" mode never populates target_bg/distractor_bg
and vice versa).

REFACTORED for background-noise compatibility: cue_mode="bg_noise" is the
properly-adapted mode for bg_noise_builder.py's output (as opposed to
"acoustic_cue", which stays untouched for whichever earlier env-only
dataset it was originally built for). Differences from "acoustic_cue":
  * Prompt: IDENTICAL wording -- reuses build_index_prompt directly (the
    cue is the same thing, a background sound category), so no new prompt
    function was needed.
  * Output field naming: target_bg_category/distractor_bg_category, to
    match bg_noise_builder.py's own top-level field names and the
    convention already used by bgnoise_math_eval.py /
    bgnoise_trivia_qa_eval.py / evaluate_dual_bgnoise_snr.py (rather than
    "acoustic_cue" mode's shorter target_bg/distractor_bg).
  * Informational-only passthrough: target_gender/distractor_gender,
    target_emotion/distractor_emotion, and snr_db are populated for
    "bg_noise" mode (bg_noise_builder.py's segments carry gender/emotion
    left over from the emotion-TTS step, plus the noise-mixing SNR) --
    never mentioned in the prompt, never affect grading, same "always
    surface secondary axes" convention used everywhere else this dataset
    is consumed. "acoustic_cue"/"gender"/"emotion" modes' own output
    fields are completely unchanged by this addition.
  * Summary: a per-bg_category accuracy breakdown (bgnoise_counts,
    populated only from "bg_noise"-mode records) is printed alongside the
    existing per-gender/per-emotion breakdowns.
  * Distractor selection switched from identity comparison
    (`s is not target_segment`) to index arithmetic
    (`record["sequence"][1 - target_index]`) -- equivalent for the
    2-segment case this whole file assumes, just more explicit and
    consistent with the index-based convention already used in
    bgnoise_math_eval.py / bgnoise_trivia_qa_eval.py. Applies uniformly to
    all cue_modes; not a behavior change for any of them.
"""

import json
import os
import random
import re

from tqdm import tqdm


def build_index_prompt(target_bg, prompt_variant=0):
    """Background-sound-cue index prompts. Shared verbatim by cue_mode
    "acoustic_cue" and cue_mode "bg_noise" -- the cue (a background sound
    category) and its framing are identical between those two datasets;
    only the data plumbing feeding target_bg and the output record differ
    (see module docstring). Variant 0 mirrors the compositional baseline's
    framing sentence-for-sentence; 1 and 2 are information-equivalent
    paraphrases (no added labels, no added hints)."""
    if prompt_variant == 0:
        return (
            "Listen to the audio carefully. It contains exactly TWO distinct "
            "spoken segments played sequentially, each mixed with a different "
            "background sound.\n"
            f"Find the segment that is accompanied by the '{target_bg}' sound.\n"
            "Output ONLY its index (1 for the first segment, 2 for the second "
            "segment) inside <ans> and </ans> tags. "
            "Do not output any reasoning or other text."
        )
    if prompt_variant == 1:
        return (
            "The audio contains two spoken segments, one after the other, and "
            "each segment has a different background sound.\n"
            f"Which segment has the '{target_bg}' sound in the background?\n"
            "Answer with ONLY the number 1 (first segment) or 2 (second "
            "segment) inside <ans> and </ans> tags, and nothing else."
        )
    if prompt_variant == 2:
        return (
            "You will hear exactly two spoken segments in sequence. Each one "
            "is mixed with a different background sound.\n"
            f"Identify the position of the segment whose background sound is "
            f"'{target_bg}'.\n"
            "Respond with ONLY <ans>1</ans> if it is the first segment or "
            "<ans>2</ans> if it is the second segment."
        )
    raise ValueError(f"Unknown prompt_variant: {prompt_variant}")


GENDER_NOUN = {"male": "man", "female": "woman"}


def build_gender_index_prompt(target_gender, prompt_variant=0):
    """Gender-cue index prompts. Mirror the acoustic-cue variants with the
    cue changed to speaker gender; audio is clean concatenation, so no
    background/noise mention. Target gender only -- no mention of the other
    voice's identity beyond 'a male and a female voice' framing (parity with
    the gender compositional prompt)."""
    noun = GENDER_NOUN[target_gender]
    if prompt_variant == 0:
        return (
            "Listen to the audio carefully. It contains exactly TWO distinct "
            "spoken segments played sequentially, one spoken by a male voice "
            "and one spoken by a female voice.\n"
            f"Find the segment that is spoken by the {noun}.\n"
            "Output ONLY its index (1 for the first segment, 2 for the second "
            "segment) inside <ans> and </ans> tags. "
            "Do not output any reasoning or other text."
        )
    if prompt_variant == 1:
        return (
            "The audio contains two spoken segments, one after the other -- "
            "one in a male voice and one in a female voice.\n"
            f"Which segment is spoken by the {noun}?\n"
            "Answer with ONLY the number 1 (first segment) or 2 (second "
            "segment) inside <ans> and </ans> tags, and nothing else."
        )
    if prompt_variant == 2:
        return (
            "You will hear exactly two spoken segments in sequence, one from "
            "a male speaker and one from a female speaker.\n"
            f"Identify the position of the segment spoken by the {noun}.\n"
            "Respond with ONLY <ans>1</ans> if it is the first segment or "
            "<ans>2</ans> if it is the second segment."
        )
    raise ValueError(f"Unknown prompt_variant: {prompt_variant}")


# Same phrasing map as emotion_eval.py / evaluate_conditional_asr_emotion.py
# -- ADJUST IF NEEDED to match how your emo_ref_audio clips were actually
# performed/labeled (e.g. "scared"/"afraid" instead of "fearful").
EMOTION_ADJ = {
    "angry": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "sad": "sad",
}


def build_emotion_index_prompt(target_emotion, prompt_variant=0):
    """Emotion-cue index prompts. Mirror the acoustic-cue/gender variants
    with the cue changed to speaker emotional tone; audio is clean
    concatenation, so no background/noise mention. Target emotion only --
    gender is never mentioned, even though emotion_builder.py's segments
    also carry a gender attribute (parity with the emotion compositional
    prompt in emotion_eval.py, which makes the same choice)."""
    adj = EMOTION_ADJ.get(target_emotion, target_emotion)
    if prompt_variant == 0:
        return (
            "Listen to the audio carefully. It contains exactly TWO distinct "
            "spoken segments played sequentially, each spoken with a "
            "different emotional tone of voice.\n"
            f"Find the segment that is spoken by the {adj}-sounding speaker.\n"
            "Output ONLY its index (1 for the first segment, 2 for the second "
            "segment) inside <ans> and </ans> tags. "
            "Do not output any reasoning or other text."
        )
    if prompt_variant == 1:
        return (
            "The audio contains two spoken segments, one after the other, "
            "each spoken with a different emotional tone of voice.\n"
            f"Which segment is spoken by the {adj}-sounding speaker?\n"
            "Answer with ONLY the number 1 (first segment) or 2 (second "
            "segment) inside <ans> and </ans> tags, and nothing else."
        )
    if prompt_variant == 2:
        return (
            "You will hear exactly two spoken segments in sequence, each "
            "spoken with a different emotional tone of voice.\n"
            f"Identify the position of the segment spoken by the {adj}-sounding speaker.\n"
            "Respond with ONLY <ans>1</ans> if it is the first segment or "
            "<ans>2</ans> if it is the second segment."
        )
    raise ValueError(f"Unknown prompt_variant: {prompt_variant}")


def build_position_concept_prompt(ask_first):
    """Control: same output contract, no acoustic-cue demand. Tests whether
    the positional-report concept itself is the bottleneck."""
    which = "FIRST" if ask_first else "SECOND"
    return (
        "Listen to the audio carefully. It contains exactly TWO distinct "
        "spoken segments played sequentially.\n"
        f"Which segment is played {which}?\n"
        "Output ONLY its index (1 for the first segment, 2 for the second "
        "segment) inside <ans> and </ans> tags. "
        "Do not output any reasoning or other text."
    )


def strict_extract_index(raw):
    """strict_extract philosophy: exactly one well-formed non-empty <ans> pair.
    Returns (format_compliant, predicted_index or None).
    predicted_index is 1/2 only if the tag content parses cleanly."""
    matches = re.findall(r"<ans>(.*?)</ans>", raw, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) != 1:
        return False, None
    content = matches[0].strip()
    if not content:
        return False, None
    # Tag pair is well-formed and non-empty -> format compliant.
    # Index validity is checked separately (content axis vs format axis).
    m = re.fullmatch(r"(?:segment\s*)?([12])\.?", content, flags=re.IGNORECASE)
    return True, int(m.group(1)) if m else None


def evaluate_segment_index(
    model,
    dataset_dir,
    metadata_path,
    output_path,
    prompt_variant=0,
    flip_target=False,
    cue_mode="acoustic_cue",
):
    """cue_mode: "acoustic_cue" (env-sound cue, original env-only dataset),
    "bg_noise" (background-noise cue, bg_noise_builder.py dataset -- same
    cue concept as "acoustic_cue" but adapted output-field naming and
    gender/emotion/snr_db informational passthrough, see module
    docstring), "gender" (speaker-gender cue, for the gender datasets),
    "emotion" (speaker-emotion cue, for the emotion datasets), or
    "position_concept" (control, cue-free).
    flip_target: cue the OTHER segment on the same items (counterfactual run).
    """
    assert cue_mode in ("acoustic_cue", "bg_noise", "gender", "emotion", "position_concept")
    print(
        f"\n--- Starting Segment Index Evaluation "
        f"(mode={cue_mode}, variant={prompt_variant}, flip={flip_target}) ---"
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]

    # ==========================================
    # Checkpoint and resume logic (same pattern as baseline)
    # ==========================================
    processed_ids = set()
    correct_count = 0
    total_tested = 0
    # 2x2 contingency: counts[target_index_1based][predicted_index or 0]
    counts = {1: {0: 0, 1: 0, 2: 0}, 2: {0: 0, 1: 0, 2: 0}}
    format_ok_count = 0
    gender_counts = {}  # target_gender -> [n_correct, n_total]
    emotion_counts = {}  # target_emotion -> [n_correct, n_total]
    bgnoise_counts = {}  # target_bg_category -> [n_correct, n_total] (cue_mode="bg_noise" only)

    def _tally(data):
        nonlocal correct_count, total_tested, format_ok_count
        total_tested += 1
        if data.get("is_correct"):
            correct_count += 1
        if data.get("format_compliant"):
            format_ok_count += 1
        t = data.get("expected_index")
        p = data.get("predicted_index") or 0
        if t in (1, 2):
            counts[t][p if p in (1, 2) else 0] += 1
        g = data.get("target_gender")
        if g:
            gender_counts.setdefault(g, [0, 0])
            gender_counts[g][0] += 1 if data.get("is_correct") else 0
            gender_counts[g][1] += 1
        e = data.get("target_emotion")
        if e:
            emotion_counts.setdefault(e, [0, 0])
            emotion_counts[e][0] += 1 if data.get("is_correct") else 0
            emotion_counts[e][1] += 1
        b = data.get("target_bg_category")
        if b:
            bgnoise_counts.setdefault(b, [0, 0])
            bgnoise_counts[b][0] += 1 if data.get("is_correct") else 0
            bgnoise_counts[b][1] += 1

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                    _tally(data)
                except json.JSONDecodeError:
                    continue

    if processed_ids:
        print(f"Detected historical progress! {len(processed_ids)} tests completed.")
        print(
            f"Current historical accuracy: "
            f"{correct_count / total_tested:.2%} ({correct_count}/{total_tested})"
        )
        print("Automatically skipping completed items...\n")
    # ==========================================

    for record in tqdm(records, desc=f"Segment Index ({cue_mode})"):
        concat_id = record["concat_id"]
        if concat_id in processed_ids:
            continue

        audio_path = os.path.join(dataset_dir, record["audio_path"])

        # Prefer the pre-assigned balanced target from the dataset metadata
        # (gender/emotion/bg_noise datasets carry position x cue-attribute
        # fully crossed targets); fall back to the seeded-random convention
        # (item-paired with the env-sound baseline runs).
        if "target_index" in record:
            primary_index = record["target_index"]
        else:
            rng = random.Random(concat_id)
            primary_index = record["sequence"].index(
                rng.choice(record["sequence"]))
        target_index = (1 - primary_index) if flip_target else primary_index
        target_segment = record["sequence"][target_index]

        target_bg = target_segment.get("bg_category")
        target_gender = target_segment.get("gender")
        target_emotion = target_segment.get("emotion")
        expected_index = target_index + 1  # 1-based, matches output contract
        # Index-based exclusion (equivalent to identity comparison for this
        # file's assumed 2-segment sequences; see module docstring).
        distractor = record["sequence"][1 - target_index]

        if cue_mode == "position_concept":
            # Deterministic per-item split of first/second questions.
            ask_first = random.Random(f"{concat_id}_poscon").random() < 0.5
            prompt = build_position_concept_prompt(ask_first)
            expected_index = 1 if ask_first else 2
        elif cue_mode == "gender":
            prompt = build_gender_index_prompt(target_gender, prompt_variant)
        elif cue_mode == "emotion":
            prompt = build_emotion_index_prompt(target_emotion, prompt_variant)
        else:
            # "acoustic_cue" and "bg_noise" share the exact same prompt --
            # both cue on a background sound category (see module docstring).
            prompt = build_index_prompt(target_bg, prompt_variant)

        prediction_raw = model.query(audio_path, prompt).strip()
        format_compliant, predicted_index = strict_extract_index(prediction_raw)

        is_correct = predicted_index == expected_index

        # Outcome category (format axis kept separate from content axis)
        if not format_compliant:
            outcome = "Format_Noncompliant"
        elif predicted_index is None:
            outcome = "Invalid_Index"  # well-formed tag, uninterpretable content
        elif is_correct:
            outcome = "Correct"
        else:
            outcome = "Wrong_Segment"

        result_dict = {
            "concat_id": concat_id,
            "cue_mode": cue_mode,
            "prompt_variant": prompt_variant,
            "flip_target": flip_target,
            "target_bg": target_bg if cue_mode == "acoustic_cue" else None,
            "distractor_bg": distractor.get("bg_category")
            if cue_mode == "acoustic_cue"
            else None,
            # bg_noise mode uses bg_noise_builder.py's own field naming
            # (target_bg_category/distractor_bg_category), matching
            # bgnoise_math_eval.py / bgnoise_trivia_qa_eval.py /
            # evaluate_dual_bgnoise_snr.py -- kept separate from
            # "acoustic_cue"'s target_bg/distractor_bg above.
            "target_bg_category": target_bg if cue_mode == "bg_noise" else None,
            "distractor_bg_category": distractor.get("bg_category")
            if cue_mode == "bg_noise"
            else None,
            "target_gender": target_gender if cue_mode in ("gender", "bg_noise") else None,
            "distractor_gender": distractor.get("gender")
            if cue_mode in ("gender", "bg_noise")
            else None,
            "target_emotion": target_emotion if cue_mode in ("emotion", "bg_noise") else None,
            "distractor_emotion": distractor.get("emotion")
            if cue_mode in ("emotion", "bg_noise")
            else None,
            # Informational only, bg_noise mode only -- see module docstring.
            "snr_db": target_segment.get("snr_db") if cue_mode == "bg_noise" else None,
            "target_index": target_index,  # 0-based position
            "expected_index": expected_index,  # 1-based ground truth
            "prediction_raw": prediction_raw,
            "predicted_index": predicted_index,
            "format_compliant": format_compliant,
            "is_correct": is_correct,
            "outcome": outcome,
        }

        _tally(result_dict)
        processed_ids.add(concat_id)

        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        if cue_mode == "gender":
            cue_display = target_gender
        elif cue_mode == "emotion":
            cue_display = target_emotion
        else:
            cue_display = target_bg
        print(
            f"\n[Concat ID: {concat_id} | Cue: {cue_display}] "
            f"Expected: {expected_index} | Pred: {predicted_index} | "
            f"{outcome}"
        )

    # ---------------- Summary: 2AFC decomposition ----------------
    if not total_tested:
        print("No items evaluated.")
        return

    print(f"\nOverall accuracy: {correct_count / total_tested:.2%} "
          f"({correct_count}/{total_tested})   [chance = 50%]")
    print(f"Format compliance rate: {format_ok_count / total_tested:.2%}")

    n1 = sum(counts[1].values())
    n2 = sum(counts[2].values())
    if n1 and n2:
        acc1 = counts[1][1] / n1
        acc2 = counts[2][2] / n2
        print(f"P(correct | target=1st): {acc1:.2%}  (n={n1})")
        print(f"P(correct | target=2nd): {acc2:.2%}  (n={n2})")
        print(f"Balanced accuracy: {(acc1 + acc2) / 2:.2%}")
        chose1 = counts[1][1] + counts[2][1]
        chose2 = counts[1][2] + counts[2][2]
        answered = chose1 + chose2
        if answered:
            print(
                f"Choice bias: P(predict 1 | answered) = {chose1 / answered:.2%} "
                f"({chose1}/{answered})  [50% = unbiased]"
            )
        print(
            "Contingency (rows=target, cols=predicted 1/2/none): "
            f"t1: {counts[1][1]}/{counts[1][2]}/{counts[1][0]}  "
            f"t2: {counts[2][1]}/{counts[2][2]}/{counts[2][0]}"
        )
    for g, (ok, n) in gender_counts.items():
        if n:
            print(f"P(correct | target={g} voice): {ok / n:.2%}  (n={n})")
    for e, (ok, n) in emotion_counts.items():
        if n:
            print(f"P(correct | target={e} tone): {ok / n:.2%}  (n={n})")
    for b, (ok, n) in bgnoise_counts.items():
        if n:
            print(f"P(correct | target={b} sound): {ok / n:.2%}  (n={n})")


# Thin wrappers matching the per-task naming convention.

def evaluate_segment_index_gender(model, dataset_dir, metadata_path,
                                  output_path, prompt_variant=0,
                                  flip_target=False):
    return evaluate_segment_index(
        model, dataset_dir, metadata_path, output_path,
        prompt_variant=prompt_variant, flip_target=flip_target,
        cue_mode="gender")


def evaluate_segment_index_emotion(model, dataset_dir, metadata_path,
                                   output_path, prompt_variant=0,
                                   flip_target=False):
    return evaluate_segment_index(
        model, dataset_dir, metadata_path, output_path,
        prompt_variant=prompt_variant, flip_target=flip_target,
        cue_mode="emotion")


def evaluate_segment_index_env(model, dataset_dir, metadata_path,
                                   output_path, prompt_variant=0,
                                   flip_target=False):
    """Segment index prediction for the bg_noise_builder.py dataset --
    see cue_mode="bg_noise" in module docstring / evaluate_segment_index."""
    return evaluate_segment_index(
        model, dataset_dir, metadata_path, output_path,
        prompt_variant=prompt_variant, flip_target=flip_target,
        cue_mode="bg_noise")