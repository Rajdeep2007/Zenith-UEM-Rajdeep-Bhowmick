"""Detector 1: a question nobody answers inside the reply window."""

from __future__ import annotations

import re

from ..evidence import score_severity
from ..models import Act, Gap, GapType, Utterance
from ..textutil import content_words, overlap_score
from ..tone import tone_for_turns

# replies that signal "someone is on it"
_COMMITMENT = re.compile(
    r"\b(?:let\s+me|i\s*'?ll|i\s+will|let'?s\s+(?:check|look)|"
    r"on\s+it|looking\s+into|will\s+check|checking\s+now|"
    r"i\s+can\s+check|let\s+us\s+check|will\s+(?:find|confirm|verify))\b",
    re.IGNORECASE,
)


def _is_response(reply: Utterance, question: Utterance, immediate: bool) -> bool:
    """does this turn actually answer it?

    "let me check" style cues only count on the very next turn, so a late
    commitment about something else shouldnt close this question.
    """
    if reply.speaker == question.speaker:
        return False
    if reply.act in (Act.ANSWER, Act.CLOSURE):
        return True
    if reply.act == Act.CLARIFY:
        return False
    #seperate real answers from the "ill check" deferrals
    if immediate and _COMMITMENT.search(reply.text):
        return True

    q = content_words(question.text)
    r = content_words(reply.text)
    #tried 0.4 first, too strict
    return overlap_score(q, r) >= 0.25


def detect(utterances: list[Utterance], window: int = 3) -> list[Gap]:
    gaps: list[Gap] = []
    if not utterances:
        return gaps
    last_id = utterances[-1].id

    for i, u in enumerate(utterances):
        if u.act != Act.QUESTION:
            continue
        following = utterances[i + 1: i + 1 + window]
        # any turn in the window can still defuse it
        #window = 2   # too twitchy on the standup sample
        if any(
            _is_response(v, u, immediate=(j == 0))
            for j, v in enumerate(following)
        ):
            continue

        end = min(last_id, u.id + window)
        n = end - u.id
        #print(u.id, u.speaker, n)   # debug
        gaps.append(
            Gap(
                type=GapType.UNANSWERED_QUESTION,
                turn_range=(u.id, end),
                participants=[u.speaker],
                evidence_quotes=[u.text],
                message_ids=[u.id],
                severity=score_severity(
                    gap_turns=n,
                    silence=n,
                    repeats=0,
                    texts=[u.text],
                ),
                description=(
                    f"Question from {u.speaker} at turn {u.id} received no reply "
                    f"in the following {n} turn(s)."
                ),
                tone=tone_for_turns([u.text] + [v.text for v in following]),
            )
        )

    #merge consecutive unanswered questions from the same speaker
    merged: list[Gap] = []
    for g in gaps:
        prev = merged[-1] if merged else None
        # fold it in only when it abuts the previous gap from the same person
        if (
            prev is None
            or prev.participants != g.participants
            or g.turn_range[0] > prev.turn_range[1] + 1
        ):
            merged.append(g)
            continue

        prev.turn_range = (prev.turn_range[0], max(prev.turn_range[1], g.turn_range[1]))
        for q, qid in zip(g.evidence_quotes, g.message_ids):
            if q not in prev.evidence_quotes:
                prev.evidence_quotes.append(q)
                prev.message_ids.append(qid)
        prev.description = (
            f"Question(s) from {prev.participants[0]} starting at turn "
            f"{prev.turn_range[0]} received no reply for "
            f"{prev.turn_range[1] - prev.turn_range[0]} turn(s)."
        )
    return merged
