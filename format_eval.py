"""
Format-compliance evaluator for LALM compositional-gap experiments.

Computes format-compliance rate per task from existing result JSONL files
(field: "prediction_raw"), at TWO levels of granularity:

  - JOINT compliance (primary/headline metric): the entire response must
    satisfy every tag requirement simultaneously (all required tags present
    exactly once, well-formed, non-empty, nothing else in the response).
    This is the conservative, all-or-nothing metric and should be reported
    as the primary strict number.

  - PER-TAG compliance (diagnostic/secondary metric): each required tag is
    scored independently for count/well-formedness/non-empty-content, so a
    response like "<ans_1>42</ans_1><ans_2>" is scored as ans_1=PASS,
    ans_2=FAIL, rather than collapsing the whole response to a single fail.
    Report this ALONGSIDE the joint metric, never as a replacement for it,
    so per-tag numbers read as diagnostic decomposition ("why is joint
    compliance low?") rather than metric-shopping.

Per-tag "no outside leakage" is defined relative to that tag's own
boundary: content immediately adjacent to a tag (before the first required
tag's open, between consecutive required tags, and after the last required
tag's close) must be whitespace-only. This still catches prose wrapped
around answers (e.g. "The answer is <ans>42</ans>.") without penalizing an
individual tag for a *different* tag being missing entirely.

  - TAG-COUNT MATCH (diagnostic/tertiary metric): per tag, whether the tag
    appears exactly once as an open marker and exactly once as a close
    marker -- independent of ordering, leakage, or empty content. This is
    STRICTLY MORE LENIENT than per-tag compliance: a tag can count-match
    while still failing overall compliance (e.g. open-after-close, empty
    content, or leaked prose around it). It isolates one narrow question --
    "did the model even attempt this tag's markers the right number of
    times" -- from everything else that per-tag compliance also checks.
    Report alongside, never in place of, the joint and per-tag metrics.

Usage:
    python format_compliance_eval.py --task env_dual_base --results_file path/to/results.jsonl
    python format_compliance_eval.py --task env_math_select --results_file path/to/results.jsonl --dump_failures failures.jsonl
"""

import argparse
import json
import re
from collections import defaultdict


TASK_TAG_SCHEMA = {
    "env_dual_base":        ["event_1", "event_2"],
    "math_dual_base":       ["ans_1", "ans_2"],
    "trivia_qa_dual_base":  ["ans_1", "ans_2"],
    "asr_dual_base":        ["asr_1", "asr_2"],

    "env_asr_select":       ["asr"],
    "env_math_select":      ["ans"],
    "env_trivia_qa_select": ["ans"],
}


def _find_tag_span(text, tag):
    """
    Returns (open_match_or_None, close_match_or_None, well_formed, reason,
    count_match).

    count_match is TRUE whenever the tag appears exactly once as an open
    marker and exactly once as a close marker -- independent of ordering,
    leakage, or empty content. This isolates "did the model even attempt
    this tag's markers the right number of times" as its own diagnostic
    signal, separate from well-formedness (open-before-close) and from the
    leakage/empty-content checks applied later in check_tag_compliance.
    """
    opens = list(re.finditer(rf"<{tag}>", text, flags=re.IGNORECASE))
    closes = list(re.finditer(rf"</{tag}>", text, flags=re.IGNORECASE))
    count_match = len(opens) == 1 and len(closes) == 1
    if not count_match:
        reason = f"tag_count_mismatch:{tag}:opens={len(opens)}:closes={len(closes)}"
        return None, None, False, reason, count_match
    o, c = opens[0], closes[0]
    if o.end() > c.start():
        return None, None, False, f"open_after_close:{tag}", count_match
    return o, c, True, None, count_match


def check_tag_compliance(prediction_raw, tag_names):
    result = {
        "joint_compliant": False,
        "per_tag": {
            t: {"compliant": False, "content": None, "failure_reasons": [], "count_match": False}
            for t in tag_names
        },
        "failure_reasons": [],
    }

    if not isinstance(prediction_raw, str) or prediction_raw.strip() == "":
        reason = "empty_or_non_string_response"
        result["failure_reasons"].append(reason)
        for t in tag_names:
            result["per_tag"][t]["failure_reasons"].append(reason)
        return result

    text = prediction_raw
    spans = {}

    for tag in tag_names:
        o, c, well_formed, reason, count_match = _find_tag_span(text, tag)
        result["per_tag"][tag]["count_match"] = count_match
        if not well_formed:
            result["per_tag"][tag]["failure_reasons"].append(reason)
            result["failure_reasons"].append(reason)
        else:
            spans[tag] = (o, c)

    for tag, (o, c) in spans.items():
        content = text[o.end():c.start()].strip()
        result["per_tag"][tag]["content"] = content
        if content == "":
            reason = f"empty_content:{tag}"
            result["per_tag"][tag]["failure_reasons"].append(reason)
            result["failure_reasons"].append(reason)

    # Strip out any text that matches ANY required tag's open/close pattern
    # (including tags that failed well-formedness elsewhere, e.g. a lone
    # unclosed "<ans_2>") before checking for leakage around a given tag.
    # This ensures a broken NEIGHBOR tag doesn't get counted as "prose
    # leakage" against a tag that is itself fine — each tag's compliance
    # should depend only on genuine surrounding prose/reasoning text, not
    # on debris left behind by a different tag's failure.
    all_tag_marker_pattern = re.compile(
        "|".join(rf"</?{re.escape(t)}>" for t in tag_names), flags=re.IGNORECASE
    )

    def _strip_other_tag_markers(segment):
        return all_tag_marker_pattern.sub("", segment)

    ordered_present = [t for t in tag_names if t in spans]
    for idx, tag in enumerate(ordered_present):
        o, c = spans[tag]
        region_start = 0 if idx == 0 else spans[ordered_present[idx - 1]][1].end()
        region_end = len(text) if idx == len(ordered_present) - 1 else spans[ordered_present[idx + 1]][0].start()
        before_raw = text[region_start:o.start()]
        after_raw = text[c.end():region_end]
        before = _strip_other_tag_markers(before_raw)
        after = _strip_other_tag_markers(after_raw)
        if before.strip() != "" or after.strip() != "":
            reason_bits = []
            if before.strip():
                reason_bits.append(f"before='{before.strip()[:40]}'")
            if after.strip():
                reason_bits.append(f"after='{after.strip()[:40]}'")
            reason = f"local_leakage:{tag}:" + ",".join(reason_bits)
            result["per_tag"][tag]["failure_reasons"].append(reason)
            result["failure_reasons"].append(reason)

    for tag in tag_names:
        result["per_tag"][tag]["compliant"] = len(result["per_tag"][tag]["failure_reasons"]) == 0

    result["joint_compliant"] = all(result["per_tag"][t]["compliant"] for t in tag_names)
    return result


def evaluate_file(results_path, task):
    if task not in TASK_TAG_SCHEMA:
        raise ValueError(f"Unknown task '{task}'. Known tasks: {list(TASK_TAG_SCHEMA.keys())}")
    tag_names = TASK_TAG_SCHEMA[task]

    n_total = 0
    n_joint_compliant = 0
    per_tag_totals = {t: {"n": 0, "compliant": 0} for t in tag_names}
    tag_count_match_totals = {t: {"n": 0, "count_match": 0} for t in tag_names}
    joint_failure_reason_counts = defaultdict(int)
    per_record_results = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at line {line_num}: {e}")
                continue

            prediction_raw = record.get("prediction_raw")
            check = check_tag_compliance(prediction_raw, tag_names)

            n_total += 1
            if check["joint_compliant"]:
                n_joint_compliant += 1
            else:
                for t in tag_names:
                    if not check["per_tag"][t]["compliant"]:
                        first_reason = check["per_tag"][t]["failure_reasons"][0]
                        category = first_reason.split(":")[0]
                        joint_failure_reason_counts[category] += 1
                        break

            for t in tag_names:
                per_tag_totals[t]["n"] += 1
                if check["per_tag"][t]["compliant"]:
                    per_tag_totals[t]["compliant"] += 1

                tag_count_match_totals[t]["n"] += 1
                if check["per_tag"][t]["count_match"]:
                    tag_count_match_totals[t]["count_match"] += 1

            per_record_results.append({
                "concat_id": record.get("concat_id"),
                "prediction_raw": prediction_raw,
                **check,
            })

    per_tag_rates = {}
    for t, v in per_tag_totals.items():
        if v["n"]:
            per_tag_rates[t] = v["compliant"] / v["n"]
        else:
            per_tag_rates[t] = float("nan")

    # tag_count_match: per-tag rate of "tag appears exactly once open + once
    # close", independent of ordering/leakage/empty-content. A separate,
    # more lenient diagnostic axis than per-tag compliance -- isolates
    # whether the model even attempted each tag's markers the right number
    # of times, decoupled from whether it then used them correctly.
    tag_count_match_rates = {}
    for t, v in tag_count_match_totals.items():
        if v["n"]:
            tag_count_match_rates[t] = v["count_match"] / v["n"]
        else:
            tag_count_match_rates[t] = float("nan")

    n_tag_count_match_total = sum(v["count_match"] for v in tag_count_match_totals.values())
    n_tag_slots_total = sum(v["n"] for v in tag_count_match_totals.values())

    output = {}
    output["task"] = task
    output["n_total"] = n_total
    output["n_joint_compliant"] = n_joint_compliant
    if n_total:
        output["joint_compliance_rate"] = n_joint_compliant / n_total
    else:
        output["joint_compliance_rate"] = float("nan")
    output["per_tag_totals"] = per_tag_totals
    output["per_tag_compliance_rate"] = per_tag_rates
    output["tag_count_match_totals"] = tag_count_match_totals
    output["tag_count_match_rate"] = tag_count_match_rates
    output["n_tag_count_match_total"] = n_tag_count_match_total
    output["n_tag_slots_total"] = n_tag_slots_total
    output["tag_count_match_rate_overall"] = (
        n_tag_count_match_total / n_tag_slots_total if n_tag_slots_total else float("nan")
    )
    output["joint_failure_reason_counts"] = dict(joint_failure_reason_counts)
    output["per_record_results"] = per_record_results
    return output


def main():
    parser = argparse.ArgumentParser(description="Format-compliance evaluator (joint + per-tag)")
    parser.add_argument("--task", required=True, choices=list(TASK_TAG_SCHEMA.keys()))
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--dump_failures", default=None)
    args = parser.parse_args()

    summary = evaluate_file(args.results_file, args.task)

    print("Task:", summary["task"])
    print("Total records:", summary["n_total"])
    print()
    print("[PRIMARY] Joint strict compliance:", summary["n_joint_compliant"],
          "(" + format(summary["joint_compliance_rate"], ".2%") + ")")
    print("Joint failure breakdown (attributed to first non-compliant tag, schema order):")
    sorted_reasons = sorted(summary["joint_failure_reason_counts"].items(), key=lambda kv: kv[1], reverse=True)
    for category, count in sorted_reasons:
        pct = count / summary["n_total"] if summary["n_total"] else 0
        print("  " + category.ljust(22) + ":", str(count).rjust(5), "(" + format(pct, ".2%") + ")")

    print()
    print("[SECONDARY / diagnostic] Per-tag compliance rate:")
    for tag, rate in summary["per_tag_compliance_rate"].items():
        n = summary["per_tag_totals"][tag]["n"]
        c = summary["per_tag_totals"][tag]["compliant"]
        print("  " + tag.ljust(12) + ":", str(c).rjust(5) + "/" + str(n).rjust(5), "(" + format(rate, ".2%") + ")")

    print()
    print("[DIAGNOSTIC] Tag-count match rate (tag appears exactly once open + once close,")
    print("             independent of ordering/leakage/empty-content):")
    for tag, rate in summary["tag_count_match_rate"].items():
        n = summary["tag_count_match_totals"][tag]["n"]
        c = summary["tag_count_match_totals"][tag]["count_match"]
        print("  " + tag.ljust(12) + ":", str(c).rjust(5) + "/" + str(n).rjust(5), "(" + format(rate, ".2%") + ")")
    print("  " + "TOTAL".ljust(12) + ":",
          str(summary["n_tag_count_match_total"]).rjust(5) + "/" + str(summary["n_tag_slots_total"]).rjust(5),
          "(" + format(summary["tag_count_match_rate_overall"], ".2%") + ")")

    if args.dump_failures:
        failures = [r for r in summary["per_record_results"] if not r["joint_compliant"]]
        with open(args.dump_failures, "w", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print()
        print("Dumped", len(failures), "joint-non-compliant records to", args.dump_failures)


if __name__ == "__main__":
    main()
