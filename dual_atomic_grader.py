"""
dual_atomic_grader.py

Standalone re-grader for the "atomic dual" background-sound / gender /
emotion classification result files -- evaluate_dual_bgnoise_snr.py and
evaluate_dual_gender_emotion.py (attribute="gender" or "emotion"). Re-
extracts each record's tagged answer(s) from its ALREADY-SAVED
prediction_raw text and recomputes every grading field FROM SCRATCH, using
only what's already in the result JSONL (ground truth + prediction_raw) --
no model access, no re-querying. Mirrors grader.py's separation between
"run the model" (the evaluate_*.py scripts) and "grade the output" (this
file), applied to the dual-classification tasks.

ONE ATOMIC ABILITY PER EXECUTION -- deliberately NOT auto-detected or
mixed. Each of the three entry points (grade_dual_bgnoise, grade_dual_gender,
grade_dual_emotion) grades exactly the ONE ability its name says, assumes
every record in the input file is already that ability's schema, and
touches nothing else. There is no "figure out which schema each record
happens to be" mode and no combined env+gender+emotion regrading here --
run the matching function/CLI --mode once per ability, once per result
file.

EXTRACTION/GRADING LOGIC is copied verbatim from the evaluating scripts
that produced these files, so regrading reproduces exactly what a fresh
model-query run would have graded, given the same prediction_raw:
  * grade_dual_bgnoise: extract_dual_environment_events (ASSUMED
    EXTERNALLY DEFINED, same as evaluate_dual_bgnoise_snr.py -- pass it in
    directly, or via --extractor_module on the CLI, default
    src.evaluators.extractor).
  * grade_dual_gender / grade_dual_emotion: extract_dual_tagged_values
    (self-contained generic two-tag extractor, same copy as
    evaluate_dual_gender_emotion.py), sharing one internal implementation
    (_regrade_and_write) via _make_tag_extractor -- same "graders cannot
    drift apart" convention used throughout this project.

Grading fields recomputed per record: extracted_answers, is_correct_1/2,
correct_count, is_exact_match, is_partial_match -- identical field names to
the evaluate_dual_*.py scripts' own output, so a regraded file is drop-in
interchangeable with a freshly-evaluated one.

NEVER OVERWRITES THE INPUT FILE -- always writes to a new output path.
Prints TWO things after regrading: (1) the actual grading RESULTS -- exact/
partial/no-match rates, per-segment-position accuracy, and a per-attribute-
value accuracy breakdown (grouped by target_bg_category / target_gender /
target_emotion, whichever applies -- already present in the input file's
records, same informational fields the evaluate_dual_*.py scripts wrote);
and (2) a flip report against whatever grading was already in the file --
useful when checking whether a parsing fix changed anything, but the
grading-results block is there regardless of whether anything flipped.
"""

import argparse
import importlib
import json
import re
from collections import defaultdict


def extract_dual_tagged_values(raw, tag_prefix):
    """
    Generic two-tag extractor for <{tag_prefix}_1>/<{tag_prefix}_2>, same
    copy as evaluate_dual_gender_emotion.py. Lowercased, stripped content;
    None if a tag is missing or malformed.
    """
    results = []
    for i in (1, 2):
        m = re.search(
            rf"<{tag_prefix}_{i}>(.*?)</{tag_prefix}_{i}>",
            raw, re.DOTALL | re.IGNORECASE,
        )
        results.append(m.group(1).strip().lower() if m else None)
    return results


def _normalize_attribute(x):
    if x is None:
        return ""

    x = x.strip().lower()

    # Remove one or more enclosing square brackets.
    while len(x) >= 2 and x.startswith("[") and x.endswith("]"):
        x = x[1:-1].strip()

    return x


def _grade_pair(extracted, ground_truth):
    pred1 = _normalize_attribute(extracted[0])
    pred2 = _normalize_attribute(extracted[1])

    gt1 = _normalize_attribute(ground_truth[0])
    gt2 = _normalize_attribute(ground_truth[1])

    c1 = pred1 == gt1
    c2 = pred2 == gt2

    cnt = int(c1) + int(c2)

    return c1, c2, cnt, (cnt == 2), (cnt == 1)


_SNAPSHOT_KEYS = ["is_correct_1", "is_correct_2", "is_exact_match", "is_partial_match"]


def _correctness_snapshot(record):
    return {k: record[k] for k in _SNAPSHOT_KEYS if k in record}


def _regrade_and_write(input_path, output_path, ability_label, extract_fn, target_key):
    """
    Shared regrading loop for a SINGLE atomic ability -- every record in
    input_path is assumed to already be that ability's schema
    (ground_truth + prediction_raw present); no schema detection is done,
    and no other ability is ever touched in this call.

    target_key is the record field holding the ground-truth attribute
    value ("target_bg_category" / "target_gender" / "target_emotion"),
    already present in the evaluate_dual_*.py output -- used only to
    print a per-value accuracy breakdown in the grading-results summary,
    never for grading itself.
    """
    flip_counts = defaultdict(int)
    n_records = 0
    n_exact = 0
    n_partial = 0
    n_none = 0
    n_correct_pos1 = 0
    n_correct_pos2 = 0
    group_counts = defaultdict(lambda: [0, 0])  # target value -> [n_exact_match, n_total]

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_records += 1

            if "ground_truth" not in record or "prediction_raw" not in record:
                raise ValueError(
                    f"Record {record.get('concat_id')} is missing 'ground_truth' or "
                    f"'prediction_raw' -- is this actually a {ability_label} result file?"
                )

            old_snapshot = _correctness_snapshot(record)

            extracted = extract_fn(record["prediction_raw"])
            c1, c2, cnt, exact, partial = _grade_pair(extracted, record["ground_truth"])

            new_record = dict(record)
            new_record.update({
                "extracted_answers": extracted,
                "is_correct_1": c1,
                "is_correct_2": c2,
                "correct_count": cnt,
                "is_exact_match": exact,
                "is_partial_match": partial,
            })

            new_snapshot = _correctness_snapshot(new_record)
            for key, new_val in new_snapshot.items():
                old_val = old_snapshot.get(key)
                if old_val is not None and old_val != new_val:
                    flip_counts[key] += 1

            # Tally the FRESH grading results for the summary report below.
            if exact:
                n_exact += 1
            elif partial:
                n_partial += 1
            else:
                n_none += 1
            if c1:
                n_correct_pos1 += 1
            if c2:
                n_correct_pos2 += 1
            group_val = record.get(target_key)
            if group_val:
                group_counts[group_val][0] += 1 if exact else 0
                group_counts[group_val][1] += 1

            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    print(f"Regraded {n_records} {ability_label} records -> {output_path}")

    if n_records:
        n_total_segments = n_records * 2
        n_correct_segments = n_correct_pos1 + n_correct_pos2
        print(f"\n--- Grading Results ({ability_label}) ---")
        print(f"Exact match (2/2 correct):   {n_exact / n_records:.2%}  ({n_exact}/{n_records})")
        print(f"Partial match (1/2 correct): {n_partial / n_records:.2%}  ({n_partial}/{n_records})")
        print(f"No match (0/2 correct):      {n_none / n_records:.2%}  ({n_none}/{n_records})")
        # Micro-averaged (segment-level) accuracy -- every individual
        # segment classification counted once, as opposed to the
        # item-level "Exact match" rate above (which requires BOTH
        # segments correct). Equivalent to (n_exact*2 + n_partial) /
        # (n_records*2).
        print(f"Overall segment-level accuracy: {n_correct_segments / n_total_segments:.2%}  "
              f"({n_correct_segments}/{n_total_segments})")
        print(f"P(correct | segment 1): {n_correct_pos1 / n_records:.2%}")
        print(f"P(correct | segment 2): {n_correct_pos2 / n_records:.2%}")
        if group_counts:
            print(f"\nPer-{target_key} exact-match accuracy:")
            for val, (ok, n) in sorted(group_counts.items()):
                if n:
                    print(f"  {val}: {ok / n:.2%}  (n={n})")
    else:
        print("No records to grade.")

    if flip_counts:
        print("\nGrading changed for some records vs. the original file (old -> new differs):")
        for key, n in sorted(flip_counts.items()):
            print(f"  field={key}: {n} records flipped")
    else:
        print("\nNo grading changes vs. the original file.")


def _make_tag_extractor(tag_prefix):
    return lambda raw: extract_dual_tagged_values(raw, tag_prefix)


def grade_dual_bgnoise(input_path, output_path, extract_dual_environment_events):
    """
    Regrades a background-noise-only dual-classification result file
    (evaluate_dual_bgnoise_snr.py output) and prints the resulting grading
    (exact/partial/no-match rates, per-segment accuracy, and per-
    target_bg_category accuracy). extract_dual_environment_events must be
    supplied (assumed externally defined, same as the evaluating script --
    see module docstring). Grades ONLY background-noise -- nothing else,
    even if the file happens to also carry gender/emotion informational
    fields (per_segment_gender/per_segment_emotion).
    """
    if extract_dual_environment_events is None:
        raise ValueError(
            "grade_dual_bgnoise requires extract_dual_environment_events -- "
            "pass it explicitly, or use --extractor_module on the CLI."
        )
    _regrade_and_write(input_path, output_path, "background-noise", extract_dual_environment_events, "target_bg_category")


def grade_dual_gender(input_path, output_path):
    """
    Regrades a gender-only dual-classification result file
    (evaluate_dual_gender_emotion.py, attribute="gender", output) and
    prints the resulting grading (exact/partial/no-match rates,
    per-segment accuracy, and per-target_gender accuracy). Grades ONLY
    gender -- nothing else, even if the file happens to also carry an
    emotion informational field (per_segment_emotion).
    """
    _regrade_and_write(input_path, output_path, "gender", _make_tag_extractor("gender"), "target_gender")


def grade_dual_emotion(input_path, output_path):
    """
    Regrades an emotion-only dual-classification result file
    (evaluate_dual_gender_emotion.py, attribute="emotion", output) and
    prints the resulting grading (exact/partial/no-match rates,
    per-segment accuracy, and per-target_emotion accuracy). Grades ONLY
    emotion -- nothing else, even if the file happens to also carry a
    gender informational field (per_segment_gender).
    """
    _regrade_and_write(input_path, output_path, "emotion", _make_tag_extractor("emotion"), "target_emotion")


def main():
    parser = argparse.ArgumentParser(
        description="Regrade an already-collected atomic dual-classification result file -- "
                     "exactly ONE ability (bg_noise, gender, or emotion) per run."
    )
    parser.add_argument("--mode", required=True, choices=["bg_noise", "gender", "emotion"],
                         help="Which single atomic ability this input file's records are for.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", default="./atomic_results.jsonl", help="Never overwrites input_path.")
    parser.add_argument("--extractor_module", default="src.evaluators.extractor",
                         help="Only used for --mode bg_noise: module to import "
                              "extract_dual_environment_events from.")
    args = parser.parse_args()

    if args.mode == "bg_noise":
        extract_dual_environment_events = None
        try:
            mod = importlib.import_module(args.extractor_module)
            extract_dual_environment_events = getattr(mod, "extract_dual_environment_events", None)
        except ImportError:
            pass
        grade_dual_bgnoise(args.input_path, args.output_path, extract_dual_environment_events)
    elif args.mode == "gender":
        grade_dual_gender(args.input_path, args.output_path)
    else:
        grade_dual_emotion(args.input_path, args.output_path)


if __name__ == "__main__":
    main()