"""
Grader for segment-index results (outputs of evaluate_segment_index.py).

Philosophy (mirrors grader.py): do NOT trust upstream is_correct /
predicted_index fields -- re-parse prediction_raw independently, on two axes:

  strict  : exactly one well-formed non-empty <ans> pair whose content parses
            to 1 or 2. (Format + content must both hold.)
  lenient : fallback chain that recovers an index from non-compliant output.
            Content-correctness, format-independent.

strict acc - lenient acc = IF-delta for the index task (same construct as the
QA IF-delta: how much accuracy is lost to format non-compliance alone).

Reported per input file (= per condition): strict/lenient accuracy,
format-compliance rate, position-conditioned accuracies, balanced accuracy,
choice bias, 2x2 contingency, d' and criterion c (log-linear corrected).
If files contain both flip_target=False and flip_target=True runs of the same
concat_ids, cue-flip consistency is reported (prediction should flip).

Usage:
    python grade_segment_index.py results_main.jsonl [results_flip.jsonl ...]
"""

import json
import re
import sys
from collections import defaultdict
from statistics import NormalDist

_ORDINAL = {
    "1": 1, "2": 2, "1st": 1, "2nd": 2,
    "one": 1, "two": 2, "first": 1, "second": 2,
}


def _content_to_index(content):
    """Map tag content / short phrase to 1 or 2, else None."""
    c = content.strip().lower().rstrip(".")
    c = re.sub(r"^(the\s+)?(segment\s*)?", "", c)
    c = re.sub(r"\s*(segment|one|problem)$", lambda m: ""
               if m.group(1) == "segment" or m.group(1) == "problem" else m.group(0),
               c).strip()
    return _ORDINAL.get(c)


def strict_parse(raw):
    """(format_compliant, index or None). Exactly one well-formed non-empty
    <ans> pair; content must parse to 1/2 for the index to count."""
    matches = re.findall(r"<ans>(.*?)</ans>", raw, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) != 1 or not matches[0].strip():
        return False, None
    return True, _content_to_index(matches[0])


def lenient_parse(raw):
    """Fallback chain. Returns (index or None, method, ambiguous).

    1. tag_strict   : the strict parse succeeded.
    2. tag_multi    : >=1 <ans> pairs; all parseable contents agree.
    3. text_pattern : phrases in tag-free text ('segment 2', 'the second
                      segment', 'answer is 1', standalone 1/2 ...).
    If mentions conflict, take the LAST one and mark ambiguous=True
    (models typically state their final choice last).
    """
    compliant, idx = strict_parse(raw)
    if idx is not None:
        return idx, "tag_strict", False

    # 2) any number of tag pairs
    tag_contents = re.findall(r"<ans>(.*?)</ans>",
                              raw, flags=re.DOTALL | re.IGNORECASE)
    tag_indices = [i for c in tag_contents
                   if (i := _content_to_index(c)) is not None]
    if tag_indices:
        distinct = set(tag_indices)
        if len(distinct) == 1:
            return tag_indices[0], "tag_multi", False
        return tag_indices[-1], "tag_multi", True

    # 3) tag-free text patterns, in order of appearance
    text = re.sub(r"</?ans>", " ", raw, flags=re.IGNORECASE)
    candidates = []
    patterns = [
        (r"\bsegment\s*(?:number\s*)?([12])\b", lambda m: int(m.group(1))),
        (r"\b(first|second)\s+segment\b", lambda m: _ORDINAL[m.group(1)]),
        (r"\bthe\s+(first|second)\s+one\b", lambda m: _ORDINAL[m.group(1)]),
        (r"\b(?:answer|ans|index)\s*(?:is|:)?\s*([12])\b",
         lambda m: int(m.group(1))),
        (r"\bit\s+is\s+(?:segment\s*)?([12])\b", lambda m: int(m.group(1))),
        (r"\bthe\s+(first|second)\b", lambda m: _ORDINAL[m.group(1)]),
        (r"(?<![\d.])([12])(?![\d.])", lambda m: int(m.group(1))),
    ]
    lowered = text.lower()
    for pat, conv in patterns:
        for m in re.finditer(pat, lowered):
            candidates.append((m.start(), conv(m)))
        if candidates:
            break  # use the highest-priority pattern class that matched
    if not candidates:
        return None, "none", False
    candidates.sort(key=lambda t: t[0])
    values = [v for _, v in candidates]
    ambiguous = len(set(values)) > 1
    return values[-1], "text_pattern", ambiguous


def _dprime(hit, fa, n_signal, n_noise):
    """d' and criterion c with log-linear (Hautus) correction."""
    h = (hit + 0.5) / (n_signal + 1.0)
    f = (fa + 0.5) / (n_noise + 1.0)
    z = NormalDist().inv_cdf
    return z(h) - z(f), -0.5 * (z(h) + z(f))


def summarize(records, label):
    n = len(records)
    if not n:
        print(f"\n=== {label}: no records ===")
        return

    strict_results = []
    lenient_results = []
    for r in records:
        raw = r.get("prediction_raw", "")
        compliant, s_idx = strict_parse(raw)
        l_idx, method, ambiguous = lenient_parse(raw)
        strict_results.append((compliant, s_idx))
        lenient_results.append((l_idx, method, ambiguous))

    expected = [r["expected_index"] for r in records]

    def _metrics(pred):
        correct = sum(p == e for p, e in zip(pred, expected))
        by_pos = {1: {"n": 0, "ok": 0, "p1": 0},
                  2: {"n": 0, "ok": 0, "p1": 0}}
        for p, e in zip(pred, expected):
            by_pos[e]["n"] += 1
            by_pos[e]["ok"] += p == e
            by_pos[e]["p1"] += p == 1
        return correct, by_pos

    s_pred = [idx for _, idx in strict_results]
    l_pred = [idx for idx, _, _ in lenient_results]
    s_correct, s_pos = _metrics(s_pred)
    l_correct, l_pos = _metrics(l_pred)
    compliant_n = sum(c for c, _ in strict_results)

    print(f"\n=== {label} (n={n}) ===")
    print(f"Format compliance:  {compliant_n / n:.2%}")
    print(f"Strict accuracy:    {s_correct / n:.2%}   [chance = 50%]")
    print(f"Lenient accuracy:   {l_correct / n:.2%}")
    print(f"IF-delta (strict - lenient): {(s_correct - l_correct) / n:+.2%}")

    methods = defaultdict(int)
    amb_count = 0
    for idx, method, ambiguous in lenient_results:
        methods[method] += 1
        amb_count += ambiguous
    print(f"Lenient parse methods: {dict(methods)}  (ambiguous: {amb_count})")

    for name, pred_pos in (("strict", s_pos), ("lenient", l_pos)):
        n1, n2 = pred_pos[1]["n"], pred_pos[2]["n"]
        if not (n1 and n2):
            continue
        a1 = pred_pos[1]["ok"] / n1
        a2 = pred_pos[2]["ok"] / n2
        # signal-detection convention: 'signal' = target is segment 1
        dp, c = _dprime(pred_pos[1]["p1"], pred_pos[2]["p1"], n1, n2)
        p1_rate = (pred_pos[1]["p1"] + pred_pos[2]["p1"]) / (n1 + n2)
        print(
            f"[{name}] P(ok|t=1): {a1:.2%} (n={n1})  "
            f"P(ok|t=2): {a2:.2%} (n={n2})  "
            f"balanced: {(a1 + a2) / 2:.2%}  "
            f"P(pred=1): {p1_rate:.2%}  d': {dp:.2f}  c: {c:+.2f}"
        )


def flip_consistency(records):
    """Pairs (concat_id) present with both flip_target False and True.
    A cue-following model should flip its (lenient) prediction."""
    by_id = defaultdict(dict)
    for r in records:
        by_id[r["concat_id"]][bool(r.get("flip_target"))] = r
    pairs = [v for v in by_id.values() if False in v and True in v]
    if not pairs:
        return
    flipped = same = unparseable = 0
    for v in pairs:
        a, _, _ = lenient_parse(v[False].get("prediction_raw", ""))
        b, _, _ = lenient_parse(v[True].get("prediction_raw", ""))
        if a is None or b is None:
            unparseable += 1
        elif a != b:
            flipped += 1
        else:
            same += 1
    n = len(pairs)
    print(f"\n=== Cue-flip consistency (paired items: {n}) ===")
    print(f"Prediction flipped (cue-following): {flipped / n:.2%}")
    print(f"Prediction unchanged (cue-ignoring): {same / n:.2%}")
    print(f"Unparseable in at least one run:     {unparseable / n:.2%}")


def main(paths):
    all_records = []
    for path in paths:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        summarize(records, path)
        all_records.extend(records)

    flip_consistency(all_records)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])