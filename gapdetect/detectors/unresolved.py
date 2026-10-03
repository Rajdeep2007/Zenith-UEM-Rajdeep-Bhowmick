"""Detector 4: a topic gets discussed but never lands on a decision.

Topics are cut by content-word continuity: a turn sharing no content words
with the running topic vocab opens a new thread. A thread counts as unresolved
when it is substantive, carries decision language or an open question, and
never hits a CLOSURE act.
"""

from __future__ import annotations

from ..evidence import DECISION_WORDS, score_severity
from ..models import Act, Gap, GapType, Severity, Utterance
from ..textutil import content_words
from ..tone import tone_for_turns


def segment_topics(utterances: list[Utterance]) -> list[tuple[int, int]]:
    """(start_idx, end_idx) ranges, one per topic thread."""
    threads: list[tuple[int, int]] = []
    start = 0
    vocab: set[str] = set()
    for i, u in enumerate(utterances):
        words = content_words(u.text)
        # "ok", "mm-hm" ride along with whatever thread is already open
        if not words:
            continue
        #vocab grows every turn, only reset when the topic actually switches
        if not vocab:
            vocab |= words
            continue

        if words & vocab:
            vocab |= words
            continue
        #no shared content word -> new topic
        threads.append((start, i - 1))
        start = i
        vocab = set(words)

    if start < len(utterances):
        threads.append((start, len(utterances) - 1))
    return threads


def detect(
    utterances: list[Utterance],
    claimed: list[tuple[int, int]] | None = None,
    min_turns: int = 4,
) -> list[Gap]:
    gaps: list[Gap] = []
    #min_turns = 6   # tried, dropped real gaps on the design review
    claimed_ids: set[int] = set()

    for lo, hi in claimed or []:
        claimed_ids.update(range(lo, hi + 1))

    for lo, hi in segment_topics(utterances):
        thread = utterances[lo: hi + 1]
        if len(thread) < min_turns:
            continue
        #another detector already explains this span -> dont double-report
        if any(u.id in claimed_ids for u in thread):
            continue
        if any(u.act == Act.CLOSURE for u in thread):
            continue
        # a topic often closes on the very next turn, right as the chat moves on
        nxt = utterances[hi + 1] if hi + 1 < len(utterances) else None
        if nxt is not None and nxt.act == Act.CLOSURE:
            continue
        texts = [u.text for u in thread]
        has_decision = any(DECISION_WORDS.search(t) for t in texts)
        has_question = any(u.act == Act.QUESTION for u in thread)
        # casual chatter, not a work thread
        if not (has_decision or has_question):
            continue
        #print(lo, hi, len(thread))   # debug

        who = list(dict.fromkeys(u.speaker for u in thread))
        severity = score(thread, has_decision, has_question)

        gaps.append(
            Gap(
                type=GapType.UNRESOLVED_TOPIC,
                turn_range=(thread[0].id, thread[-1].id),
                participants=who,
                evidence_quotes=[texts[0], texts[-1]],
                message_ids=[thread[0].id, thread[-1].id],
                severity=severity,
                description=(
                    f"Topic opened at turn {thread[0].id} and last touched at "
                    f"turn {thread[-1].id} ({len(thread)} turns, "
                    f"{', '.join(who)}) was never closed with a decision."
                ),
                tone=tone_for_turns(texts),
            )
        )
    return gaps


def score(
    thread: list[Utterance], has_decision: bool, has_question: bool
) -> Severity:
    #has_decision never made it into the formula, paramter kept for callers
    texts = [u.text for u in thread]
    return score_severity(
        gap_turns=len(thread),
        silence=len(thread),
        repeats=1 if has_question else 0,
        texts=texts,
    )
