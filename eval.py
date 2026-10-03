"""Evaluate detector output against hand-labelled ground truth.

a predicted gap matches a truth gap of the SAME type when their turn ids
intersect (one-to-one greedy). unmatched predictions are false positives,
unmatched truths are false negatives.

run: python eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from gapdetect import analyze

DATA = Path(__file__).parent / "data"
#ground truth sits in its own folder, same stems as the transcripts
TRUTH = DATA / "ground_truth"


def load_case(stem: str):
    case = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    truth = json.loads((TRUTH / f"{stem}.json").read_text(encoding="utf-8"))

    raw = "\n".join(f"{t['speaker']}: {t['text']}" for t in case["transcript"])
    return case.get("title", stem), raw, truth.get("gaps", [])


def evidence_integrity(report) -> tuple[int, int]:
    """(gaps with valid cited ids and verbatim quotes, total gaps)"""
    by_id = {u.id: u for u in report.transcript}
    norm = lambda s: " ".join(s.split())  #noqa: E731

    ok = 0
    for g in report.gaps:
        if not g.message_ids:
            continue
        if not all(tid in by_id for tid in g.message_ids):
            continue

        #every quote has to turn up in one of the turns we cite
        if not all(
            any(norm(q) in norm(by_id[tid].text) for tid in g.message_ids)
            for q in g.evidence_quotes
        ):
            continue
        ok += 1
    return ok, len(report.gaps)


def evaluate(stem: str) -> dict:
    title, raw, truths = load_case(stem)
    report = analyze(raw, title=title)

    tp = fp = fn = 0

    details = []
    used: set[int] = set()

    for g in report.gaps:
        lo, hi = g.turn_range
        pred = set(range(lo, hi + 1))
        hit = None
        #greedy: first unused truth of the same type that overlaps wins
        for i, t in enumerate(truths):
            if i in used or t["type"] != g.type.value:
                continue
            if pred & set(t["turns"]):
                hit = i
                break
        if hit is None:
            fp += 1
            #print("fp:", g.description)   # debug
            details.append(("FP", g.type.value, list(g.turn_range), g.description))
        else:
            used.add(hit)
            tp += 1
            details.append(("TP", g.type.value, list(g.turn_range), g.description))

    for i, t in enumerate(truths):
        if i not in used:
            fn += 1

            details.append(("FN", t["type"], t["turns"], t.get("note", "")))

    return {
        "title": title,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "details": details,
        "n_truth": len(truths),
        "n_pred": len(report.gaps),
        "integrity": evidence_integrity(report),
    }


def main() -> None:
    #sorted so the report reads in a stable order run to run
    cases = [p.stem for p in sorted(DATA.glob("*.json"))]
    agg = {"tp": 0, "fp": 0, "fn": 0}
    integ_ok = integ_total = 0

    for stem in cases:
        r = evaluate(stem)
        for k in ("tp", "fp", "fn"):
            agg[k] += r[k]
        integ_ok += r["integrity"][0]
        integ_total += r["integrity"][1]
        print(f"\n== {r['title']} ({stem})  truth={r['n_truth']}  predicted={r['n_pred']}")

        for kind, gtype, span, desc in r["details"]:
            print(f"   [{kind}] {gtype:<24} {str(span):<12} {desc[:80]}")
        if r["tp"] + r["fp"] == 0 and r["fn"] == 0:
            print("   (no gaps detected, no gaps expected)")

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    print("\n" + "=" * 56)
    print(f"TOTAL   TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={p:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(
        f"Evidence integrity: {integ_ok}/{integ_total} gaps "
        f"(cited message_ids exist + verbatim quotes)"
    )
    print("=" * 56)


if __name__ == "__main__":
    main()
