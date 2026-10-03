"""Communication Health Score, 0-100, four sub-scores.

    responsiveness  35%  did the questions get answers
    closure         25%  did threads end on a decision
    inclusion       20%  even participation (entropy) minus ignored turns
    tone            20%  frustration / urgency / uncertainty wording
"""

from __future__ import annotations

import math
from collections import Counter

from .detectors.unresolved import segment_topics
from .models import Act, GapType, Utterance
from .tone import tone_of

#these weights came from the spec, they sum to 1 so dont "fix" them
WEIGHTS = {
    "responsiveness": 0.35,
    "closure": 0.25,
    "inclusion": 0.20,
    "tone": 0.20,
}

_TONE_PENALTY = {"frustration": 12, "urgency": 6, "uncertainty": 4}
_IGNORED_PENALTY = 8
# was 15, crushed quiet calls


def grade_for(score: float) -> str:
    #letter cutoffs, 50 is the pass line
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _responsiveness(utterances: list[Utterance], q_ranges) -> tuple[float, dict]:
    questions = [u for u in utterances if u.is_question_like]
    #no questions at all is a perfect score, not a zero
    if not questions:
        return 100.0, {"questions": 0, "unanswered": 0}
    unanswered = sum(
        1
        for q in questions
        if any(lo <= q.id <= hi for lo, hi in q_ranges)
    )
    # print(len(questions), unanswered)  # debug while tuning the window

    answered = len(questions) - unanswered
    return 100.0 * answered / len(questions), {
        "questions": len(questions),
        "unanswered": unanswered,
    }


def _closure(utterances: list[Utterance]) -> tuple[float, dict]:
    threads = segment_topics(utterances)
    if not threads:
        return 100.0, {"threads": 0, "closed": 0}
    # a CLOSURE inside the thread or on the very next turn both count
    closed = sum(
        1
        for lo, hi in threads
        if any(u.act == Act.CLOSURE for u in utterances[lo : hi + 1])
        or (hi + 1 < len(utterances) and utterances[hi + 1].act == Act.CLOSURE)
    )
    return 100.0 * closed / len(threads), {
        "threads": len(threads),
        "closed": closed,
    }


def _inclusion(utterances: list[Utterance], n_ignored: int) -> tuple[float, dict]:
    counts = Counter(u.speaker for u in utterances)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        balance = 100.0  # one voice is "balanced" by definition
    else:
        # shannon entropy, normalised by log(k) so it lands in 0..1
        h = -sum(
            (c / total) * math.log(c / total) for c in counts.values()
        ) / math.log(len(counts))
        balance = 100.0 * h
    # tried a flat 5 per ignored turn, should of been gentler
    score = max(0.0, balance - n_ignored * _IGNORED_PENALTY)
    return score, {"speakers": len(counts), "ignored": n_ignored}


def _tone(utterances: list[Utterance]) -> tuple[float, dict]:
    penalty = 0.0
    counts = {"frustration": 0, "urgency": 0, "uncertainty": 0}
    for u in utterances:
        t = tone_of(u.text)
        if t in counts:
            counts[t] += 1
            penalty += _TONE_PENALTY[t]
    #cap keeps a rant from zeroing the sub-score (was 50, too harsh)
    score = max(0.0, 100.0 - min(penalty, 70.0))
    return score, counts


def compute_health(
    utterances: list[Utterance],
    gaps: list | None = None,
) -> dict:
    """Health report for a conversation: overall grade plus the four sub-scores."""
    gaps = gaps or []
    q_ranges = [
        g.turn_range for g in gaps if g.type is GapType.UNANSWERED_QUESTION
    ]
    n_ignored = sum(1 for g in gaps if g.type is GapType.IGNORED_RESPONSE)

    resp, resp_d = _responsiveness(utterances, q_ranges)
    clos, clos_d = _closure(utterances)
    incl, incl_d = _inclusion(utterances, n_ignored)
    tone, tone_d = _tone(utterances)
    # print(resp, clos, incl, tone)  # raw floats before rounding

    subs = {
        "responsiveness": round(resp),
        "closure": round(clos),
        "inclusion": round(incl),
        "tone": round(tone),
    }
    # subs get rounded first, then weighted; the dashboard shows the rounded ones
    overall = round(sum(subs[k] * w for k, w in WEIGHTS.items()))
    return {
        "overall": overall,
        "grade": grade_for(overall),
        "subscores": subs,
        "details": {
            "responsiveness": resp_d,
            "closure": clos_d,
            "inclusion": incl_d,
            "tone": tone_d,
        },
    }
