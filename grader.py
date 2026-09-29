"""
grader.py: computes the strict-vs-lenient accuracy delta per record. It is
fully self-contained -- there is no separate llm_score_eval.py file. Both
sides of the comparison are computed here:

  - STRICT: independently re-parses prediction_raw with STRICT tag rules
    (via strict_extract) and compares the tag content against ground truth.
  - LENIENT: compares the already-LLM-extracted answer (llm_extracted_answers,
    produced upstream by an LLM-based parser and expected to already be
    present on each input record) against ground truth, using the SAME
    matching/normalization rule as the strict side for a given answer_type
    (_answers_match) -- so "strict vs lenient" differs only in WHICH
    candidate string is being compared (tag-extracted vs LLM-extracted),
    not in how the comparison itself is done. This function is
    llm_is_correct(candidate, record, idx, num_segments, answer_type) below.

IMPORTANT: grader.py deliberately does NOT trust any upstream is_correct /
is_correct_1 / is_correct_2 fields that may already be present in the
input file. Those were computed by the original evaluator, which applies
its own fallback/lenient parsing when extracting an answer from
prediction_raw -- so they do not represent genuinely STRICT (tag-only,
zero-tolerance) correctness. Using them here would silently launder format
failures the whole point of this file is to expose. Instead, grader.py
runs its own minimal, standalone strict_extract directly on prediction_raw:
a tag must appear exactly once, be well-formed (open before close), and
have non-empty content, or that segment is strictly WRONG -- no fallback,
no recovery, no leniency of any kind.

Motivation (why the delta is meaningful): strict accuracy conflates two
independent failure modes -- wrong content, and right content wrapped in
the wrong/missing tags. The delta isolates the pure instruction-following
cost: how much accuracy is lost purely to format non-compliance, holding
content-correctness fixed via the LLM-parsed answer. This is reported
ALONGSIDE strict/lenient accuracy and format-compliance rate (from
format_compliance_eval.py), never as a replacement for either.

Two answer types, two different normalization/matching rules:
  - math:   pure whitespace-strip + exact string match. No lowercasing, no
            punctuation/article stripping -- kept maximally strict/literal.
  - trivia: normalize_answer() (whisper.normalizers.EnglishTextNormalizer +
            article/punctuation/whitespace stripping + lowercasing) applied
            to BOTH the candidate answer and each ground-truth alias before
            comparing, so trivial surface-form differences ("The answer is
            42." vs "42", "Einstein" vs "einstein") don't count as wrong.

Ground-truth SHAPE differs by answer type and segment count (per your
schema):
  - math,   num_segments == 1 (select tasks):  ground_truth is a bare string
  - math,   num_segments  > 1 (dual-base):      ground_truth is list[str], one per segment
  - trivia, num_segments == 1 (select tasks):  ground_truth_norm is a bare list[str] of aliases
                                                 (ground_truth_raw also present but ignored --
                                                  ground_truth_norm is preferred per your convention)
  - trivia, num_segments  > 1 (dual-base):      ground_truth_norm is list[list[str]], one alias-list per segment
grader.py branches on (answer_type, num_segments) to pick the right access
pattern -- for num_segments == 1 the whole field IS segment 1's ground
truth already (no indexing into a per-segment list).

ALSO SUPPORTED -- the newer gender-cue-subset schema (e.g.
gender_math_000000.jsonl, gender_trivia_000000.jsonl), which differs from
the above in two ways, both handled transparently without any new CLI
flag or code path selection -- you use grader.py exactly the same way
regardless of which schema a given input file is in:
  - math,   num_segments == 1: ground_truth is a single-element list
            (["47.8"]) instead of a bare string ("47.8"). _answers_match's
            math branch now accepts either shape (list -> OR-match across
            elements, string -> direct compare).
  - trivia, num_segments == 1: there is no ground_truth_norm key at all --
            the alias list lives directly under ground_truth instead.
            _segment_ground_truth's existing gt_norm-then-ground_truth
            fallback already resolves this correctly.
  - all_distractors_info, math: entries use {"gender": ..., "expected_answers": [...]}
            instead of the original {"bg_category": ..., "answer": "807"}
            -- the category key (bg_category / gender) is never read for
            grading either way; _distractor_answers now checks for
            expected_answers first and only falls back to the legacy
            "answer" key for old-format math records.
Two-segment (dual-base) records are unaffected by any of this -- their
shape is unchanged between schema versions.

Sanity check: strict_is_correct_i should be True only if llm_is_correct_i
is also True for that segment (a strictly-compliant, correct answer must
also be extractable and correct under the more lenient LLM parser). Any
violation (strict correct but LLM-parsed incorrect) is flagged as a
"parser_disagreement" -- most likely an LLM-parser extraction bug (e.g. the
compound-answer truncation bug found during development) rather than a
genuine instruction-following effect, and should be inspected before
trusting the delta on affected records.

DISTRACTOR accuracy (one-segment / select tasks only): these records carry
all_distractors_info, whose shape differs by answer_type:
  - math:   [{"bg_category": ..., "answer": "807"}, ...]           (single string per distractor)
  - trivia: [{"bg_category": ..., "expected_answers": [...]}, ...] (alias list per distractor)
grader.py adds TWO parallel fields, on two different bases -- never just
one, per the "always report both axes, never substitute" pattern used
throughout this pipeline:
  - is_correct_vs_any_distractor_strict_1: candidate is the STRICT <ans>-tag
    content (the same value used for strict_is_correct_1). False if the tag
    is missing/malformed -- you cannot be "distracted" if you produced no
    parseable strict answer at all.
  - is_correct_vs_any_distractor_llm_1: candidate is the LLM-parsed
    extracted answer (lenient, content-focused signal), independent of tag
    compliance.
Both are True if that candidate matches ANY distractor's answer(s) (using
the matching rule for answer_type) -- i.e. the model answered as if it had
grounded to a wrong (distractor) segment instead of the target. These can
diverge: a model might be strictly non-compliant (no tag) yet its
underlying (LLM-recovered) content still matches a distractor's answer, or
vice versa. Two-segment records have no distractor concept and are skipped
for these fields.

Input: a JSONL file with, per record:
  - prediction_raw: the raw model response (re-parsed here, strictly)
  - ground_truth / ground_truth_norm: as used elsewhere in this pipeline
  - llm_extracted_answers: the LLM-parser's extracted answer(s) -- produced
    upstream by an LLM-based extraction step (not part of this file);
    grader.py itself computes correctness from these, it does not extract
    them
  - (select tasks only) all_distractors_info

Output: the same records, additively augmented with:
  - strict_is_correct_i: bool per segment (i=1..num_segments)
  - llm_is_correct_i: bool per segment -- computed HERE (see llm_is_correct()),
    not read from any pre-existing field
  - strict_correct_count: sum across segments
  - delta_i: int per segment, llm_is_correct_i - strict_is_correct_i (0 or 1;
    should essentially never be -1 -- see parser_disagreement flag)
  - parser_disagreement_i: bool per segment, True if strict_is_correct_i is
    True but llm_is_correct_i is False (the "delta < 0" sanity-check flag)
  - is_correct_vs_any_distractor_strict_1 / is_correct_vs_any_distractor_llm_1:
    bool (select/one-segment tasks only, see above)

Usage as a module:

    from grader import gradeFile

    summary = gradeFile(
        input_file="results_v2/qwen25_omni_..._math_dual_base.llm_extracted.jsonl",
        output_file="results_v2/qwen25_omni_..._math_dual_base.graded.jsonl",
        tag_names=["ans_1", "ans_2"],
        num_segments=2,
        answer_type="math",
    )
    print(summary["strict_accuracy"], summary["lenient_accuracy"], summary["delta"])

Usage from the command line:

    python grader.py \
        --input_file results_v2/qwen25_omni_env_math_concat_snr_10_math_dual_base.llm_extracted.jsonl \
        --output_file results_v2/qwen25_omni_env_math_concat_snr_10_math_dual_base.graded.jsonl \
        --tags ans_1 ans_2 --answer_type math --num_segments 2

    python grader.py \
        --input_file results_v2/qwen25_omni_env_math_concat_snr_10_env_math_select.llm_extracted.jsonl \
        --output_file results_v2/qwen25_omni_env_math_concat_snr_10_env_math_select.graded.jsonl \
        --tags ans --answer_type math --num_segments 1
"""

import argparse
import json
import re
import string

try:
    from whisper.normalizers import EnglishTextNormalizer
    _whisper_normalizer = EnglishTextNormalizer()
except ImportError:
    _whisper_normalizer = None


def whisper_normalize(s):
    """Wraps whisper.normalizers.EnglishTextNormalizer. Raises a clear error
    at call time (not import time) if the whisper package isn't installed,
    so grader.py can still be imported/used for math-only files without it."""
    if _whisper_normalizer is None:
        raise ImportError(
            "whisper.normalizers.EnglishTextNormalizer is required for "
            "answer_type='trivia' normalization but the 'whisper' package "
            "(openai-whisper) is not installed. Install it or use "
            "answer_type='math' if this file doesn't need trivia normalization."
        )
    return _whisper_normalizer(s)


def normalize_answer(s):
    """
    Hybrid normalization for trivia-QA answer matching (per your
    normalize_answer): whisper's EnglishTextNormalizer, then article
    removal, punctuation removal, lowercasing, whitespace collapsing.
    NOT used for math -- math uses pure whitespace-strip exact match.
    """
    if not isinstance(s, str):
        return ""

    s = whisper_normalize(s)

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


def normalize_math(s):
    """Math answer normalization: pure whitespace-strip only. No
    lowercasing, no punctuation/article stripping -- kept maximally
    strict/literal, per your instruction."""
    if not isinstance(s, str):
        return ""
    return s.strip()


def _answers_match(candidate, ground_truth_or_aliases, answer_type):
    """
    Compares a candidate answer string against ground truth, using the
    normalization rule for answer_type. ground_truth_or_aliases may be
    EITHER a single string OR a list of acceptable strings -- if it's a
    list, candidate matches if it equals ANY element (OR-match).

    This list-or-string flexibility matters because your two math input
    schemas disagree on shape for single-segment (select) ground truth: the
    original schema stores a bare string ("47.8"), while the newer
    gender-cue-subset schema stores a single-element list (["47.8"]) --
    both are handled identically here, with no schema-version special-casing
    needed anywhere else in this file. Trivia already always used a list of
    aliases in both schemas, so this was already list-aware for trivia;
    this just extends the same handling to math.

    Returns False if candidate is None or ground_truth_or_aliases is None.
    """
    if candidate is None or ground_truth_or_aliases is None:
        return False

    if answer_type == "math":
        if isinstance(ground_truth_or_aliases, str):
            acceptable = [ground_truth_or_aliases]
        else:
            acceptable = list(ground_truth_or_aliases)
        candidate_norm = normalize_math(candidate)
        return any(candidate_norm == normalize_math(a) for a in acceptable)

    if answer_type == "trivia":
        candidate_norm = normalize_answer(candidate)
        if isinstance(ground_truth_or_aliases, str):
            aliases = [ground_truth_or_aliases]
        else:
            aliases = list(ground_truth_or_aliases)
        return any(candidate_norm == normalize_answer(a) for a in aliases)

    raise ValueError(f"Unknown answer_type {answer_type!r}; expected 'math' or 'trivia'")


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
    Minimal standalone STRICT extraction: for each tag in tag_names, returns
    its content string if the tag appears exactly once, well-formed, with
    non-empty content -- otherwise None for that tag (treated as WRONG by
    the caller). No fallback, no recovery -- deliberately zero-leniency,
    independent of whatever fallback parsing the original evaluator used to
    populate its own is_correct fields.

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


def _segment_ground_truth(record, idx, num_segments):
    """
    Returns the ground-truth value for segment idx (0-based), respecting
    the schema quirk that num_segments==1 records store the WHOLE field as
    segment 1's ground truth (no per-segment list wrapping), while
    num_segments>1 records store a per-segment list.

    Prefers ground_truth_norm (trivia alias lists) over ground_truth when
    both are present (ground_truth_raw is intentionally ignored -- it is
    the pre-normalization/original-language version kept for reference,
    not meant for scoring). The newer gender-cue-subset schema has no
    ground_truth_norm key at all (trivia aliases live directly under
    ground_truth instead) -- the fallback to gt_plain below already
    handles that transparently.

    Returns: for math, either a bare string (original schema) or a list of
    acceptable strings (gender-cue-subset schema, e.g. ["47.8"]) -- both
    are accepted as-is here and disambiguated downstream by
    _answers_match's now list-or-string-aware math branch, not here. For
    trivia, a list of alias strings, regardless of schema version. None if
    not resolvable.
    """
    gt_norm = record.get("ground_truth_norm")
    gt_plain = record.get("ground_truth")
    field = gt_norm if gt_norm is not None else gt_plain
    if field is None:
        return None

    if num_segments == 1:
        # Whole field IS segment 1's ground truth already -- no indexing.
        return field

    if idx < len(field):
        return field[idx]
    return None


def llm_is_correct(record, idx, num_segments, answer_type):
    """
    Fuses in what used to be llm_score_eval.py's job: computes lenient
    correctness for segment idx (0-based) by comparing the LLM-extracted
    answer (record["llm_extracted_answers"][idx]) against ground truth,
    using the SAME _answers_match rule as the strict side for this
    answer_type. This is the only difference between strict and lenient
    scoring in this file: which candidate string is being compared, not how
    the comparison itself works.

    Returns False (not an error) if llm_extracted_answers is missing/short,
    or if extraction_successful (when present) is False for this segment --
    an LLM parser that reports it failed to extract anything should not be
    credited with a correct answer.
    """
    llm_extracted = record.get("llm_extracted_answers")
    if not llm_extracted or idx >= len(llm_extracted):
        return False

    llm_extraction_successful = record.get("llm_extraction_successful")
    if llm_extraction_successful is not None and idx < len(llm_extraction_successful):
        if not llm_extraction_successful[idx]:
            return False

    candidate = llm_extracted[idx]
    gt_for_segment = _segment_ground_truth(record, idx, num_segments)
    return _answers_match(candidate, gt_for_segment, answer_type)


def _distractor_answers(distractor_info, answer_type):
    """
    Extracts the answer(s) to compare against from a single entry of
    all_distractors_info. Two schema variants seen so far, handled
    uniformly here:
      - trivia (both schema versions), and math under the newer
        gender-cue-subset format:
            {"<category-key>": ..., "expected_answers": [...]}  -> [...]
        (the category key itself varies -- "bg_category" in the original
        background-noise subset, "gender" in the gender-cue subset -- and
        is irrelevant to grading, so it is never read here)
      - math under the original background-noise-subset format only:
            {"bg_category": ..., "answer": "807"}  -> ["807"]
    "expected_answers" is checked first; only math records lacking it fall
    back to the legacy "answer" key. Returns a list of alias strings
    (possibly length 1).
    """
    if "expected_answers" in distractor_info:
        return list(distractor_info.get("expected_answers") or [])
    if answer_type == "math" and "answer" in distractor_info:
        ans = distractor_info.get("answer")
        return [ans] if ans is not None else []
    raise ValueError(
        f"Could not find distractor answer(s) in {distractor_info!r} for "
        f"answer_type={answer_type!r} (expected an 'expected_answers' list, "
        f"or 'answer' for legacy math records)"
    )


def grade_record(record, tag_names, num_segments, answer_type):
    """
    Computes strict_is_correct_i / llm_is_correct_i / delta_i /
    parser_disagreement_i (and, for one-segment records with distractor
    info, is_correct_vs_any_distractor_strict_i / _llm_i) for a single
    record.

    STRICT: independently re-parses prediction_raw via strict_extract
    (ignoring any upstream is_correct/is_correct_i fields).
    LENIENT: computes llm_is_correct_i here (fusing in what used to be a
    separate llm_score_eval.py) by comparing record["llm_extracted_answers"]
    against ground truth via the same _answers_match rule used for strict.

    Returns a dict with just the new fields to merge into the record
    (does not mutate the input record).
    """
    prediction_raw = record.get("prediction_raw")
    tag_contents = strict_extract(prediction_raw, tag_names)

    new_fields = {}
    strict_flags = {}
    for i in range(1, num_segments + 1):
        idx = i - 1
        tag = tag_names[idx] if idx < len(tag_names) else None
        content = tag_contents.get(tag) if tag is not None else None

        gt_for_segment = _segment_ground_truth(record, idx, num_segments)
        strict_ok = _answers_match(content, gt_for_segment, answer_type)
        strict_flags[i] = strict_ok
        new_fields[f"strict_is_correct_{i}"] = strict_ok

        llm_ok = llm_is_correct(record, idx, num_segments, answer_type)
        new_fields[f"llm_is_correct_{i}"] = llm_ok
        delta = int(llm_ok) - int(strict_ok)
        new_fields[f"delta_{i}"] = delta
        # parser_disagreement: strict says correct, LLM-parsed says incorrect.
        # This should essentially never happen -- a strictly-compliant correct
        # answer must also be correctly recoverable by the (more lenient) LLM
        # parser. If it does happen, treat it as a signal to inspect the LLM
        # parser's behavior on that record (e.g. truncation/extraction bugs),
        # not as evidence of a real instruction-following effect.
        new_fields[f"parser_disagreement_{i}"] = bool(strict_ok and not llm_ok)

        # Distractor accuracy: only meaningful for one-segment (select) tasks
        # that carry all_distractors_info. Reported on BOTH bases, side by
        # side, per the "always report both axes, never substitute" pattern
        # used throughout this pipeline:
        #   - _strict: candidate is the STRICT <ans>-tag content (`content`,
        #     the same value used for strict_is_correct_i above). If the tag
        #     is missing/malformed, content is None and this is False -- you
        #     cannot be "distracted" if you produced no parseable strict
        #     answer at all.
        #   - _llm: candidate is the LLM-parsed extracted answer (lenient,
        #     content-focused signal), independent of tag compliance.
        # These can diverge: a model might be strictly non-compliant (no
        # tag) yet its underlying (LLM-recovered) content still matches a
        # distractor's answer, or vice versa.
        all_distractors_info = record.get("all_distractors_info")
        if num_segments == 1 and all_distractors_info is not None:
            def _matches_any_distractor(candidate):
                for d in all_distractors_info:
                    d_answers = _distractor_answers(d, answer_type)
                    if any(_answers_match(candidate, d_ans, answer_type) for d_ans in d_answers):
                        return True
                return False

            llm_extracted = record.get("llm_extracted_answers", [])
            llm_candidate = llm_extracted[0] if llm_extracted else None

            new_fields[f"is_correct_vs_any_distractor_strict_{i}"] = _matches_any_distractor(content)
            new_fields[f"is_correct_vs_any_distractor_llm_{i}"] = _matches_any_distractor(llm_candidate)

    new_fields["strict_correct_count"] = sum(1 for v in strict_flags.values() if v)
    return new_fields


def gradeFile(input_file, output_file, tag_names, num_segments, answer_type):
    """
    Reads a JSONL file and writes a graded JSONL file with
    strict_is_correct_i / llm_is_correct_i / delta_i / parser_disagreement_i
    (and, where applicable, is_correct_vs_any_distractor_strict_1 /
    is_correct_vs_any_distractor_llm_1) added to every record. Additive
    only -- all existing fields (including any upstream is_correct/
    is_correct_i fields) are preserved untouched -- grader.py just doesn't
    READ them for its own strict determination.

    Both strict AND lenient (llm_is_correct_i) correctness are computed
    HERE (grader.py is self-contained; there is no separate
    llm_score_eval.py) -- the only external input required per record is
    llm_extracted_answers (assumed already produced by an upstream LLM
    parser) plus prediction_raw and ground_truth(_norm).

    Returns an aggregate summary dict (strict accuracy, lenient accuracy,
    delta, parser-disagreement count, distractor-match rates on both the
    strict and llm bases) computed over all segments across all records,
    plus per-record results for inspection/dumping.
    """
    if answer_type not in ("math", "trivia"):
        raise ValueError(f"answer_type must be 'math' or 'trivia', got: {answer_type!r}")

    with open(input_file, "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    missing_llm_extraction = 0
    n_segments_total = 0
    n_strict_correct = 0
    n_llm_correct = 0
    n_parser_disagreements = 0
    n_distractor_eligible = 0
    n_distractor_matched_strict = 0
    n_distractor_matched_llm = 0
    per_record_results = []

    graded_records = []
    for record in records:
        if "llm_extracted_answers" not in record:
            missing_llm_extraction += 1
            graded_records.append(record)
            continue

        new_fields = grade_record(record, tag_names, num_segments, answer_type)
        record = dict(record)
        record.update(new_fields)
        graded_records.append(record)

        for i in range(1, num_segments + 1):
            n_segments_total += 1
            if record[f"strict_is_correct_{i}"]:
                n_strict_correct += 1
            if record[f"llm_is_correct_{i}"]:
                n_llm_correct += 1
            if record[f"parser_disagreement_{i}"]:
                n_parser_disagreements += 1
                # print(record["concat_id"])
            if f"is_correct_vs_any_distractor_strict_{i}" in record:
                n_distractor_eligible += 1
                if record[f"is_correct_vs_any_distractor_strict_{i}"]:
                    n_distractor_matched_strict += 1
                if record[f"is_correct_vs_any_distractor_llm_{i}"]:
                    n_distractor_matched_llm += 1

        per_record_results.append({
            "concat_id": record.get("concat_id"),
            **{k: v for k, v in new_fields.items()},
        })

    with open(output_file, "w", encoding="utf-8") as fout:
        for record in graded_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if missing_llm_extraction:
        print(f"[WARN] {missing_llm_extraction} record(s) had no llm_extracted_answers field "
              f"(did you run the LLM parser on this file first?) -- left ungraded.")

    strict_accuracy = n_strict_correct / n_segments_total if n_segments_total else float("nan")
    lenient_accuracy = n_llm_correct / n_segments_total if n_segments_total else float("nan")
    delta = lenient_accuracy - strict_accuracy if n_segments_total else float("nan")
    parser_disagreement_rate = n_parser_disagreements / n_segments_total if n_segments_total else float("nan")
    distractor_match_rate_strict = (
        n_distractor_matched_strict / n_distractor_eligible if n_distractor_eligible else float("nan")
    )
    distractor_match_rate_llm = (
        n_distractor_matched_llm / n_distractor_eligible if n_distractor_eligible else float("nan")
    )

    summary = {
        "input_file": input_file,
        "output_file": output_file,
        "answer_type": answer_type,
        "num_segments": num_segments,
        "n_records": len(records),
        "n_segments_total": n_segments_total,
        "n_strict_correct": n_strict_correct,
        "n_llm_correct": n_llm_correct,
        "strict_accuracy": strict_accuracy,
        "lenient_accuracy": lenient_accuracy,
        "delta": delta,
        "n_parser_disagreements": n_parser_disagreements,
        "parser_disagreement_rate": parser_disagreement_rate,
        "n_distractor_eligible": n_distractor_eligible,
        "n_distractor_matched_strict": n_distractor_matched_strict,
        "n_distractor_matched_llm": n_distractor_matched_llm,
        "distractor_match_rate_strict": distractor_match_rate_strict,
        "distractor_match_rate_llm": distractor_match_rate_llm,
        "per_record_results": per_record_results,
    }
    return summary


def _print_summary(summary):
    print(f"Input:  {summary['input_file']}")
    print(f"Output: {summary['output_file']}")
    print(f"Answer type: {summary['answer_type']}  |  Segments per sample: {summary['num_segments']}")
    print(f"Records: {summary['n_records']}  |  Segments scored: {summary['n_segments_total']}")
    print()
    print(f"Strict accuracy  (independent tag re-parse, incomplete tag = wrong): "
          f"{summary['n_strict_correct']}/{summary['n_segments_total']} "
          f"({summary['strict_accuracy']:.2%})")
    print(f"Lenient accuracy (LLM-extracted answer vs ground truth):            "
          f"{summary['n_llm_correct']}/{summary['n_segments_total']} "
          f"({summary['lenient_accuracy']:.2%})")
    print(f"Delta (lenient - strict, instruction-following cost):               "
          f"{summary['delta']:+.2%}")
    print()
    if summary["n_parser_disagreements"] > 0:
        print(f"[WARN] parser_disagreement: {summary['n_parser_disagreements']} segment(s) "
              f"({summary['parser_disagreement_rate']:.2%}) were strict_is_correct=True but "
              f"llm_is_correct=False. This should essentially never happen -- inspect these "
              f"records for LLM-parser extraction bugs before trusting the delta.")
    else:
        print("No parser disagreements found (strict-correct segments are always a subset "
              "of LLM-parsed-correct segments, as expected).")

    if summary["n_distractor_eligible"] > 0:
        print()
        print(f"Distractor-match rate, STRICT basis (<ans>-tag content matches a distractor's answer): "
              f"{summary['n_distractor_matched_strict']}/{summary['n_distractor_eligible']} "
              f"({summary['distractor_match_rate_strict']:.2%})")
        print(f"Distractor-match rate, LLM basis    (LLM-parsed answer matches a distractor's answer): "
              f"{summary['n_distractor_matched_llm']}/{summary['n_distractor_eligible']} "
              f"({summary['distractor_match_rate_llm']:.2%})")


def main():
    ap = argparse.ArgumentParser(description="Compute strict-vs-lenient accuracy delta (instruction-following cost) and, for select tasks, distractor-match rate")
    ap.add_argument("--input_file", required=True, help="JSONL file with prediction_raw, ground_truth(_norm), and llm_extracted_answers per record")
    ap.add_argument("--output_file", default="graded.jsonl")
    ap.add_argument("--tags", nargs="+", required=True, help="Tag names in segment order, e.g. --tags ans_1 ans_2")
    ap.add_argument("--answer_type", required=True, choices=["math", "trivia"])
    ap.add_argument(
        "--num_segments",
        type=int,
        choices=(1, 2),
        required=True,
        help="1 for compositional/select tasks, 2 for atomic dual-base tasks",
    )
    args = ap.parse_args()

    if len(args.tags) != args.num_segments:
        ap.error(
            f"--tags expects exactly {args.num_segments} value(s) when "
            f"--num_segments={args.num_segments}; received {len(args.tags)}"
        )

    summary = gradeFile(
        input_file=args.input_file,
        output_file=args.output_file,
        tag_names=args.tags,
        num_segments=args.num_segments,
        answer_type=args.answer_type,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
