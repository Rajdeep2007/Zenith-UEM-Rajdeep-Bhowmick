"""Train the gap-window classifier (phase 5).

no hand-labelled ML dataset needed:
  1. real positives   : gap windows the detectors emit on the hand-labelled
                        demo transcripts (labels verified by eval.py)
  2. synthetic        : inject each gap pattern into the clean transcript at
                        several positions, re-run the pipeline, keep the
                        window where the expected detector fired
  3. negatives        : sliding windows from the clean meeting + windows in
                        the demo transcripts that do NOT overlap any
                        ground-truth gap

model: TF-IDF (1,2-grams) over act-sequence + message text, LogisticRegression,
evaluated with StratifiedGroupKFold (groups = source conversation, so windows
from the same conversation never leak across folds), then fit on all data and
saved to artifacts/gap_classifier.joblib.

run: python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from gapdetect import analyze
from gapdetect.ml import METRICS_PATH, MODEL_PATH, window_features

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CASES = ["clean_meeting", "team_standup", "design_review"]

# synthetic injections: (expected GapType, positions, inserted turns).
# turns are (speaker, text), speakers come from the clean meeting.
FILLER_SETS = [
    [
        ("Nikhil", "The corridor lights on floor two keep flickering at night."),
        ("Tanvi", "Room B is booked for tomorrow afternoon."),
        ("Vikram", "Someone left a laptop charger in the meeting room."),
    ],
    [
        ("Vikram", "The printer on level three needs a new cartridge."),
        ("Nikhil", "The parking lot gate card reader is acting up again."),
        ("Meera", "Facilities posted a maintenance notice for the lifts."),
    ],
]

# filler turns appended after an injected ignored-response statement: the first
# is an ANSWER-act turn so any nearby pre-existing clean question still finds
# an answer (the injection must not create a stray unanswered-question gap),
# none of them reference or acknowledge the statement.
IGNORED_FILLERS = [
    ("Nikhil", "I checked the storage quotas, nothing unusual."),
    ("Tanvi", "The freight elevator booking sheet is out of date."),
    ("Vikram", "I swapped the dead bulb in the stairwell light."),
]

INJECTIONS: list[tuple[str, list[int], list[tuple[str, str]]]] = [
    # unanswered question: question + 3 filler turns (guaranteed non-answer window)
    *[
        (
            "unanswered_question",
            [3, 10],
            [question, *fillers],
        )
        for question, fillers in zip(
            [
                ("Meera", "Nikhil, do you know if the staging environment is back up?"),
                ("Vikram", "Can someone confirm whether the design tokens library was published?"),
                ("Tanvi", "Hey Meera, did the client approve the revised launch date yet?"),
            ],
            FILLER_SETS + [FILLER_SETS[0]],
        )
    ],
    # ignored response: an addressed statement nobody reacts to,
    # followed by injected filler turns (positive window = fully synthetic)
    (
        "ignored_response",
        [0, 1, 14],
        [
            ("Meera", "Tanvi, the API key rotation is done, you can verify it whenever you like."),
            *IGNORED_FILLERS,
        ],
    ),
    (
        "ignored_response",
        [0, 14],
        [
            ("Nikhil", "Meera, the invoice export is fixed, you can retry it now."),
            *IGNORED_FILLERS,
        ],
    ),
    # repeated clarification: two adjacent clarifications about the same referent
    # (phrasing avoids the word "clarify" so the cluster cannot merge with the
    # clean meeting's own clarification turn, which contains it)
    (
        "repeated_clarification",
        [5, 8],
        [
            ("Tanvi", "I still do not understand the roll-out window - can you explain it?"),
            ("Vikram", "Can you explain the roll-out window again? I really do not get it."),
        ],
    ),
    (
        "repeated_clarification",
        [5, 8],
        [
            ("Nikhil", "I still do not understand the deploy rollback plan - can you explain it?"),
            ("Meera", "Can you explain the deploy rollback plan again? I really do not get it."),
        ],
    ),
    # unresolved topic: an open, decision-relevant thread appended at the end
    (
        "unresolved_topic",
        [17],
        [
            ("Meera", "I think the new navigation is a regression - users get lost."),
            ("Vikram", "The old navigation was clearer for first-time users anyway."),
            ("Tanvi", "The navigation choice needs a decision this week."),
            ("Nikhil", "We should decide the navigation owner and pick one option."),
            ("Meera", "Until then the navigation changes stay blocked."),
        ],
    ),
    (
        "unresolved_topic",
        [17],
        [
            ("Vikram", "Nobody seems sure about the roll-out window scope at all."),
            ("Meera", "The roll-out window keeps changing and nobody knows why."),
            ("Tanvi", "We need an owner for the roll-out window definition."),
            ("Nikhil", "Someone has to decide the roll-out window rules this week."),
            ("Vikram", "Until then the roll-out window stays undefined."),
        ],
    ),
    (
        "unresolved_topic",
        [17],
        [
            ("Tanvi", "The API versioning strategy is still completely open."),
            ("Meera", "Nobody picked an API versioning approach for the gateway yet."),
            ("Vikram", "We need to decide the API versioning scheme before the release."),
            ("Nikhil", "The API versioning owner is still unknown."),
            ("Tanvi", "No one can ship the API versioning changes as it stands."),
        ],
    ),
]

# swept for the negative windows; tried 6, no gain
SIZES = (2, 3, 4, 5)


def case_raw(stem: str) -> str:
    case = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    # same "Speaker: text" join ingest.parse expects
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in case["transcript"])


def build_dataset() -> tuple[list[str], list[int], list[str]]:
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []

    # 1) real gap windows + 3) negative windows from labelled cases
    for stem in CASES:
        report = analyze(case_raw(stem), title=stem)
        # gaps on labelled cases are verified true positives (eval.py F1=1.000);
        # windows overlapping them are positive, the rest are negative
        spans = [g.turn_range for g in report.gaps]
        for g in report.gaps:
            texts.append(window_features(report.transcript, g.turn_range))
            labels.append(1)
            groups.append(stem)
        n = len(report.transcript)

        for size in SIZES:
            for start in range(n - size + 1):
                span = (start, start + size - 1)
                if any(
                    not (span[1] < lo or span[0] > hi) for lo, hi in spans
                ):
                    continue
                texts.append(window_features(report.transcript, span))
                labels.append(0)
                # clean negatives contain no synthetic text (positives are 100%
                # injected), so they can be split into regional sub-groups to
                # balance the folds; standup/design negatives must stay with
                # their own conversation's positives (same text, both labels)
                if stem == "clean_meeting":
                    groups.append(f"clean_region_{start // 5}")
                else:
                    groups.append(stem)

    # 2) synthetic gap windows
    base = json.loads((DATA / "clean_meeting.json").read_text(encoding="utf-8"))["transcript"]
    kept = skipped = 0
    for gtype, positions, turns in INJECTIONS:
        for pos in positions:
            if pos > len(base):
                # past the end, nothing to splice into
                continue
            data = (
                base[:pos]
                + [{"speaker": s, "text": t} for s, t in turns]
                + base[pos:]
            )
            report = analyze(json.dumps(data), title="synthetic")
            lo, hi = pos, pos + len(turns) - 1
            match = next(
                (
                    g
                    for g in report.gaps
                    if g.type.value == gtype
                    and not (g.turn_range[1] < lo or g.turn_range[0] > hi)
                ),
                None,
            )

            if match is None:
                skipped += 1
                #print(gtype, pos, "no match")   # debug
                continue
            texts.append(window_features(report.transcript, match.turn_range))
            labels.append(1)
            groups.append(f"synth_{gtype}_{pos}_{kept}")
            kept += 1
    # kept = detector fired where we expected, skipped = the pattern didnt land
    print(f"synthetic injections: kept={kept}  skipped(no matching detector)={skipped}")

    return texts, labels, groups


def make_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    # C=5.0 beats 1.0 here, min_df=1 cos the corpus is tiny
                    C=5.0,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    texts, labels, groups = build_dataset()
    n_pos = sum(labels)

    n_neg = len(labels) - n_pos
    print(f"dataset: {n_pos} positive / {n_neg} negative windows, "
          f"{len(set(groups))} groups")

    # too few of either class and the stratified folds come back degenerate
    if n_pos < 5 or n_neg < 5:
        raise SystemExit("not enough data to train")

    pipe = make_pipeline()
    # groups = source conversation: every window from the same conversation
    # (real positives, negatives, or one synthetic variant) stays in one fold.
    # Positive windows are 100% injected text, so they never collide with the
    # clean-meeting negative windows across folds.
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1 = cross_val_score(pipe, texts, labels, cv=cv, groups=groups, scoring="f1")
    prec = cross_val_score(pipe, texts, labels, cv=cv, groups=groups, scoring="precision")
    rec = cross_val_score(pipe, texts, labels, cv=cv, groups=groups, scoring="recall")
    print(f"CV (StratifiedGroupKFold, {n_splits} folds): "
          f"F1={f1.mean():.3f}\u00b1{f1.std():.3f}  "
          f"P={prec.mean():.3f}  R={rec.mean():.3f}")
    print("per-fold F1:", " ".join(f"{v:.3f}" for v in f1))

    pipe.fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # imported here so a missing joblib only breaks the save step
    import joblib

    joblib.dump(pipe, MODEL_PATH)
    metrics = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_groups": n_groups,
        "cv_folds": n_splits,
        "cv_f1_mean": round(float(f1.mean()), 4),
        "cv_f1_std": round(float(f1.std()), 4),
        "cv_precision": round(float(prec.mean()), 4),
        "cv_recall": round(float(rec.mean()), 4),
        "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved: {MODEL_PATH}")
    print(f"metrics: {METRICS_PATH}")

    # sanity: score the model on the labelled demo gaps
    from gapdetect.ml import confidence as conf  #noqa: E402
    from gapdetect.ml import _model  # noqa: E402

    # cache_clear so we pick up the model we just wrote
    _model.cache_clear()
    assert _model() is not None, "model failed to load"
    for stem in CASES:
        report = analyze(case_raw(stem), title=stem)

        scores = []
        for g in report.gaps:
            c = conf(g, report.transcript)
            if c is not None:
                scores.append(f"{g.type.value}={c:.0%}")
        print(f"  {stem}: {', '.join(scores) or '(no gaps)'}")


if __name__ == "__main__":
    main()
