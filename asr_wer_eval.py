"""
asr_wer_eval.py

Evaluates ASR (WER-based) tasks with the SAME strict-vs-lenient duality
grader.py already applies to math/trivia -- both bases are computed HERE,
per record, from the same two inputs grader.py reads:

  - STRICT: independently re-parses prediction_raw with STRICT tag rules
    (strict_extract, copied verbatim from grader.py -- a tag must appear
    exactly once, be well-formed, and have non-empty content, or that
    segment's strict candidate is "" (treated as a maximal-error/deletion
    hypothesis, never silently excluded -- see STRICT MODE below).
  - LENIENT: reads record["llm_extracted_answers"][idx] -- the SAME field
    name/shape grader.py expects, produced upstream by an LLM-based parser
    (not part of this file). For ASR this list holds transcript text
    instead of a short answer, but the shape and the optional
    llm_extraction_successful[idx] gating are identical to grader.py's
    llm_is_correct().

The only difference from grader.py is the METRIC: math/trivia grader.py
computes a boolean match (_answers_match) per candidate; this file computes
WER (a continuous score) per candidate via jiwer, but the strict/lenient
SOURCING of the candidate -- and the point of comparing them -- is exactly
the same idea: strict WER conflates "wrong content" with "right content in
the wrong/missing tags", and the strict-vs-lenient WER delta isolates the
pure format-compliance cost, holding content fixed via the LLM-parsed
transcript, mirroring grader.py's "delta" exactly (just: lower is better
for WER, so delta here is strict_wer - lenient_wer, and should essentially
always be >= 0 -- a NEGATIVE delta would mean the LLM parser's transcript
is somehow WORSE than the raw tag content, worth inspecting the same way
grader.py flags parser_disagreement).

A record with NO llm_extracted_answers field at all -- i.e. the LLM parser
never ran on it, as opposed to running and failing -- is skipped ENTIRELY,
from both bases, for the whole record (not just its lenient side), exactly
matching grader.py's gradeFile(). This is what keeps STRICT and LENIENT
computed over the identical sample pool, so the delta can't go spuriously
negative just because some records never had LLM parsing at all.

STRICT MODE (carried over from the previous concat_asr revision): once a
record has passed that llm_extracted_answers check, a sample/segment is
only excluded from WER if its GROUND TRUTH is empty -- that's the one case
jiwer can't score at all. A missing/malformed tag (strict candidate) or a
failed-but-attempted LLM extraction (lenient candidate, e.g.
llm_extraction_successful[idx] is False) is NOT excluded -- it's scored as
an empty hypothesis, i.e. 100% deletions for that reference's words, so
format failures actively hurt WER on either basis instead of silently
vanishing from the denominator.

TAGS: --tags controls the tag name(s) prediction_raw is parsed for.
Defaults: "asr_1", "asr_2" for --mode concat_asr (one query, two tagged
segments, same one-query-multiple-tags convention as every other dual-
segment task in this project); "asr" for --mode conditional_asr (one
query, one target segment). Override if your actual tag names differ.

BACKWARD-COMPAT FALLBACK: this only ever applies to a record that DID pass
the llm_extracted_answers check above but has no prediction_raw field (an
older result file that had LLM parsing but never captured the raw tagged
text) -- its strict candidate falls back to the pre-extracted field
directly (prediction_asr_1/prediction_asr_2 for concat_asr,
prediction_clean for conditional_asr -- matching this file's own history,
see the commented-out line this replaced) and is used as-is, since there
is no raw text left to apply strict tag rules to.

Two modes, same duality applied to each:
  --mode concat_asr: two-segment sequential ASR (ground_truth_1/2).
    Reports STRICT/LENIENT WER per segment, overall, and the delta.
  --mode conditional_asr: single target-segment ASR with a distractor,
    classified into Target_Focused / Distractor_Attended / Both_Recognized
    / Other_Hallucination by minimum relative WER (unchanged classification
    logic from the previous revision) -- run ONCE per basis (strict,
    lenient), producing two full parallel reports via the same
    _classify_conditional/_print_conditional_report helpers, so you can see
    whether the model's classified *behavior* (not just its WER) changes
    once format failures are given the benefit of LLM-recovered content.
"""

import argparse
import json
import re

from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR performance (strict tag-parsed vs LLM-parsed WER), "
                     "the WER analog of grader.py's strict/lenient duality."
    )
    parser.add_argument("--mode", type=str, choices=["concat_asr", "conditional_asr"],
                         default="concat_asr", help="Evaluation mode to run.")
    parser.add_argument("--results_path", type=str, required=True,
                         help="Path to the JSONL file containing ASR results for evaluation.")
    parser.add_argument("--tags", nargs="+", default=None,
                         help="Tag name(s) prediction_raw is parsed for. Default: "
                              "'asr_1 asr_2' for concat_asr, 'asr' for conditional_asr.")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Strict/lenient candidate extraction -- strict_extract copied verbatim
# from grader.py; lenient_extract mirrors grader.py's llm_is_correct()
# input handling, returning the raw candidate STRING instead of a bool.
# ----------------------------------------------------------------------

def _find_tag_span(text, tag):
    """Locates a single well-formed <tag>...</tag> span. Returns
    (open_match, close_match) or (None, None) if the tag does not appear
    exactly once in well-formed (open-before-close) order."""
    opens = list(re.finditer(rf"<{tag}>", text, flags=re.IGNORECASE))
    closes = list(re.finditer(rf"</{tag}>", text, flags=re.IGNORECASE))
    if len(opens) != 1 or len(closes) != 1:
        return None, None
    o, c = opens[0], closes[0]
    if o.end() > c.start():
        return None, None
    return o, c


def strict_extract(prediction_raw, tag_names):
    """
    Minimal standalone STRICT extraction (identical to grader.py's): for
    each tag in tag_names, returns its content string if the tag appears
    exactly once, well-formed, with non-empty content -- otherwise None
    for that tag. No fallback, no recovery.

    Returns: dict {tag_name: content_str_or_None}
    """
    result = {t: None for t in tag_names}
    if not isinstance(prediction_raw, str) or prediction_raw.strip() == "":
        return result

    for tag in tag_names:
        o, c = _find_tag_span(prediction_raw, tag)
        if o is None:
            continue
        content = prediction_raw[o.end():c.start()].strip()
        if content == "":
            continue
        result[tag] = content
    return result


def lenient_extract(record, idx):
    """
    LENIENT candidate: record["llm_extracted_answers"][idx], respecting an
    optional record["llm_extraction_successful"][idx] flag -- mirrors
    grader.py's llm_is_correct() input handling exactly, just returning the
    raw candidate STRING here instead of a match/no-match verdict (WER
    needs the actual transcript text).

    Returns "" (not None) if llm_extracted_answers is missing/short, or if
    llm_extraction_successful explicitly marks this segment as failed --
    "" is the correct "no lenient candidate" value here, same STRICT MODE
    convention as the strict side (see module docstring): an absent
    transcript is scored as 100% deletions, never silently excluded.
    """
    llm_extracted = record.get("llm_extracted_answers")
    if not llm_extracted or idx >= len(llm_extracted):
        return ""

    llm_extraction_successful = record.get("llm_extraction_successful")
    if llm_extraction_successful is not None and idx < len(llm_extraction_successful):
        if not llm_extraction_successful[idx]:
            return ""

    candidate = llm_extracted[idx]
    return candidate if isinstance(candidate, str) else ""


# ----------------------------------------------------------------------
# concat_asr: two-segment sequential ASR
# ----------------------------------------------------------------------

def evaluate_concat_asr(results_path, tag_names=("asr_1", "asr_2")):
    """
    Evaluates two-segment sequential ASR performance, reporting STRICT
    (tag-parsed) and LENIENT (LLM-parsed) WER side by side, per segment
    and overall, plus the delta. See module docstring.
    """
    print("Loading data and initializing EnglishTextNormalizer...")

    normalizer = EnglishTextNormalizer()

    refs = {1: [], 2: []}
    hyps_strict = {1: [], 2: []}
    hyps_lenient = {1: [], 2: []}

    lack_strict = {1: 0, 2: 0}
    lack_lenient = {1: 0, 2: 0}
    # strict candidate empty but lenient candidate non-empty -- i.e. the
    # model DID say something recoverable, it just violated the tag format.
    format_recovery_count = {1: 0, 2: 0}

    processed_count = 0
    missing_llm_extraction = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Matches grader.py's gradeFile() exactly: a record with NO
            # llm_extracted_answers field at all was never touched by the
            # LLM parser, which is a different situation from "the parser
            # ran but failed to extract anything" -- there is no lenient
            # candidate to even compare against, so the record is skipped
            # from BOTH bases (not scored as a lenient failure), keeping
            # strict and lenient computed over the IDENTICAL sample pool
            # so the delta stays apples-to-apples and can't go spuriously
            # negative just because some records never had LLM parsing.
            if "llm_extracted_answers" not in data:
                missing_llm_extraction += 1
                continue

            gt = {1: data.get("ground_truth_1", ""), 2: data.get("ground_truth_2", "")}

            prediction_raw = data.get("prediction_raw")
            if prediction_raw is not None:
                tag_contents = strict_extract(prediction_raw, tag_names)
                strict_cand = {
                    1: tag_contents.get(tag_names[0]) or "",
                    2: tag_contents.get(tag_names[1]) or "",
                }
            else:
                # Backward-compat fallback -- see module docstring.
                strict_cand = {
                    1: data.get("prediction_asr_1", "") or "",
                    2: data.get("prediction_asr_2", "") or "",
                }

            lenient_cand = {
                1: lenient_extract(data, 0),
                2: lenient_extract(data, 1),
            }

            for seg in (1, 2):
                norm_gt = normalizer(gt[seg])
                norm_strict = normalizer(strict_cand[seg])
                norm_lenient = normalizer(lenient_cand[seg])

                # STRICT MODE: only skip when the REFERENCE is empty.
                if not norm_gt:
                    continue

                refs[seg].append(norm_gt)
                hyps_strict[seg].append(norm_strict)
                hyps_lenient[seg].append(norm_lenient)

                if not norm_strict:
                    lack_strict[seg] += 1
                if not norm_lenient:
                    lack_lenient[seg] += 1
                if not norm_strict and norm_lenient:
                    format_recovery_count[seg] += 1

            processed_count += 1

    def _wer_or_none(ref_list, hyp_list):
        return wer(ref_list, hyp_list) if ref_list else None

    strict_wer = {seg: _wer_or_none(refs[seg], hyps_strict[seg]) for seg in (1, 2)}
    lenient_wer = {seg: _wer_or_none(refs[seg], hyps_lenient[seg]) for seg in (1, 2)}

    overall_ref = refs[1] + refs[2]
    overall_strict_wer = _wer_or_none(overall_ref, hyps_strict[1] + hyps_strict[2])
    overall_lenient_wer = _wer_or_none(overall_ref, hyps_lenient[1] + hyps_lenient[2])

    print("\n=== Sequential ASR Performance Analysis (STRICT tag-parsed vs LENIENT LLM-parsed) ===")
    print(f"Total valid samples processed: {processed_count}")
    if missing_llm_extraction:
        print(f"[WARN] {missing_llm_extraction} record(s) had no llm_extracted_answers field "
              f"(did you run the LLM parser on this file first?) -- left OUT of both STRICT and "
              f"LENIENT WER (matches grader.py's handling of the same situation), so the two "
              f"bases stay computed over the identical sample pool.")

    if overall_strict_wer is not None:
        print(f"\nOverall STRICT  WER (tag-parsed from prediction_raw):    {overall_strict_wer:.2%}")
    if overall_lenient_wer is not None:
        print(f"Overall LENIENT WER (LLM-parsed llm_extracted_answers):  {overall_lenient_wer:.2%}")
    if overall_strict_wer is not None and overall_lenient_wer is not None:
        print(f"Delta (strict - lenient, format-compliance WER cost):   {overall_strict_wer - overall_lenient_wer:+.2%}")

    print("\n--- Breakdown by Sequence Position ---")
    for seg in (1, 2):
        label = "First Question" if seg == 1 else "Second Question"
        s, l = strict_wer[seg], lenient_wer[seg]
        print(f"Segment {seg} ({label}):")
        print(f"  STRICT  WER: {s:.2%}" if s is not None else "  STRICT  WER: N/A")
        print(f"  LENIENT WER: {l:.2%}" if l is not None else "  LENIENT WER: N/A")
        if s is not None and l is not None:
            print(f"  Delta:       {s - l:+.2%}")

    print("\n--- Missing / Recovered Predictions ---")
    for seg in (1, 2):
        print(f"Segment {seg}: missing STRICT candidate: {lack_strict[seg]}  |  "
              f"missing LENIENT candidate: {lack_lenient[seg]}  |  "
              f"format-recoverable (strict empty, lenient non-empty): {format_recovery_count[seg]}")


# ----------------------------------------------------------------------
# conditional_asr: single target-segment ASR with distractor classification
# ----------------------------------------------------------------------

def _classify_conditional(entries):
    """
    entries: list of (n_target, n_dist, n_both, n_pred) -- ALREADY
    normalized strings, n_pred possibly empty. Runs the same min-WER-based
    classification as the original evaluate_conditional_asr_cases
    (unchanged thresholds/logic), factored out so it can be run once per
    candidate basis (strict, lenient).

    Returns (categories, overall_ref, overall_hyp, lack_pred_count, total_processed).
    """
    categories = {
        "Target_Focused": {"refs": [], "hyps": []},
        "Distractor_Attended": {"refs": [], "hyps": []},
        "Both_Recognized": {"refs": [], "hyps": []},
        "Other_Hallucination": {"refs": [], "hyps": []},
    }
    overall_ref, overall_hyp = [], []
    lack_pred_count = 0
    total_processed = 0

    for n_target, n_dist, n_both, n_pred in entries:
        if not n_target or not n_dist or not n_both:
            continue

        # Same jiwer empty-hypothesis guard as the original scalar-mode
        # classification calls below (a single space, not "", to avoid a
        # jiwer edge case in scalar/single-string comparisons specifically
        # -- distinct from concat_asr's list-mode WER, which handles ""
        # fine, per the STRICT MODE fix already verified there).
        if not n_pred:
            lack_pred_count += 1
        effective_pred = n_pred if n_pred else " "

        wer_target_relative = wer(n_target, effective_pred)
        wer_dist_relative = wer(n_dist, effective_pred)
        wer_both_relative = wer(n_both, effective_pred)

        min_wer = min(wer_target_relative, wer_dist_relative, wer_both_relative)

        if min_wer > 0.8:
            case = "Other_Hallucination"
        elif min_wer == wer_both_relative and wer_both_relative < wer_target_relative:
            case = "Both_Recognized"
        elif min_wer == wer_dist_relative and wer_dist_relative < wer_target_relative:
            case = "Distractor_Attended"
        else:
            case = "Target_Focused"

        categories[case]["refs"].append(n_target)
        categories[case]["hyps"].append(effective_pred)
        overall_ref.append(n_target)
        overall_hyp.append(effective_pred)
        total_processed += 1

    return categories, overall_ref, overall_hyp, lack_pred_count, total_processed


def _print_conditional_report(basis_label, categories, overall_ref, overall_hyp, lack_pred_count, total_processed):
    print(f"=== Conditional ASR Behavior Analysis ({basis_label}) ===")
    print(f"Total Valid Samples Processed: {total_processed}")
    print(f"Missing Predictions Count: {lack_pred_count}\n")
    if overall_ref:
        print(f"Overall WER vs Target: {wer(overall_ref, overall_hyp):.2%}\n")
    print("-" * 50)

    for case_name, lists in categories.items():
        count = len(lists["refs"])
        if count == 0:
            print(f"[{case_name}]")
            print("  -> Count: 0 (0.0%)\n")
            continue

        case_wer_vs_target = wer(lists["refs"], lists["hyps"])
        percentage = (count / total_processed) * 100 if total_processed else 0.0

        print(f"[{case_name}]")
        print(f"  -> Count: {count} ({percentage:.2f}%)")
        print(f"  -> WER vs Target: {case_wer_vs_target:.2%}\n")


def evaluate_conditional_asr_cases(results_path, tag_names=("asr",)):
    """
    Evaluates single target-segment ASR with distractor-classification,
    run TWICE -- once against the STRICT (tag-parsed) candidate, once
    against the LENIENT (LLM-parsed) candidate -- producing two full
    parallel reports via _classify_conditional/_print_conditional_report.
    See module docstring.
    """
    print("Loading data and initializing EnglishTextNormalizer...\n")
    normalizer = EnglishTextNormalizer()

    strict_entries = []
    lenient_entries = []
    missing_llm_extraction = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Same rule as evaluate_concat_asr / grader.py's gradeFile():
            # no llm_extracted_answers field at all means the LLM parser
            # never touched this record, which is a different situation
            # from "the parser ran but failed" -- skip it from BOTH the
            # strict and lenient entry lists so the two bases stay
            # computed over the identical sample pool.
            if "llm_extracted_answers" not in data:
                missing_llm_extraction += 1
                continue

            prediction_raw = data.get("prediction_raw")
            if prediction_raw is not None:
                tag_contents = strict_extract(prediction_raw, tag_names)
                strict_cand = tag_contents.get(tag_names[0]) or ""
            else:
                # Backward-compat fallback -- this is exactly the field
                # this function used to read directly (see the commented-
                # out "pred = data.get('prediction_clean', '')" line this
                # replaced) before prediction_raw/tag parsing existed here.
                strict_cand = data.get("prediction_clean", "") or ""

            lenient_cand = lenient_extract(data, 0)

            gt_target = data.get("ground_truth_text", "")
            distractors = data.get("all_distractors_info", [])
            gt_distractor = distractors[0]["text"] if distractors else ""
            gt_1 = data.get("ground_truth_1", "")
            gt_2 = data.get("ground_truth_2", "")
            gt_both = f"{gt_1} {gt_2}".strip()

            n_target = normalizer(gt_target)
            n_dist = normalizer(gt_distractor)
            n_both = normalizer(gt_both)

            strict_entries.append((n_target, n_dist, n_both, normalizer(strict_cand)))
            lenient_entries.append((n_target, n_dist, n_both, normalizer(lenient_cand)))

    if missing_llm_extraction:
        print(f"[WARN] {missing_llm_extraction} record(s) had no llm_extracted_answers field -- "
              f"left OUT of both STRICT and LENIENT classification (matches grader.py's handling "
              f"of the same situation), so the two bases stay computed over the identical sample pool.\n")

    strict_cat, strict_oref, strict_ohyp, strict_lack, strict_total = _classify_conditional(strict_entries)
    _print_conditional_report("STRICT tag-parsed", strict_cat, strict_oref, strict_ohyp, strict_lack, strict_total)

    print("=" * 50 + "\n")

    lenient_cat, lenient_oref, lenient_ohyp, lenient_lack, lenient_total = _classify_conditional(lenient_entries)
    _print_conditional_report("LENIENT LLM-parsed", lenient_cat, lenient_oref, lenient_ohyp, lenient_lack, lenient_total)


def main():
    args = parse_args()
    if args.mode == "concat_asr":
        tag_names = tuple(args.tags) if args.tags else ("asr_1", "asr_2")
        evaluate_concat_asr(args.results_path, tag_names=tag_names)
    elif args.mode == "conditional_asr":
        tag_names = tuple(args.tags) if args.tags else ("asr",)
        evaluate_conditional_asr_cases(args.results_path, tag_names=tag_names)


if __name__ == "__main__":
    main()
