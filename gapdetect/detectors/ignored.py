"""Detector 2: a contribution nobody reacts to.

Fires on an ANSWER that directly follows a question/clarification (the asker's
chance to acknowledge), or on a statement aimed at another participant.

Counts as a gap only when, inside the reply window, no other speaker refers
back to it (content-word overlap) and nobody drops an ack marker.
"""

from __future__ import annotations

import re

from ..evidence import score_severity
from ..models import Act, Gap, GapType, Utterance
from ..textutil import content_words, overlap_score
from ..tone import tone_for_turns

_ADDRESSED = re.compile(
    r"^\s*@\w+|\b(?:you|your|yourself)\b", re.IGNORECASE
)

_ACK = re.compile(
    r"\b(?:thanks?|thank\s+you|got\s+it|noted|makes\s+sense|agreed|"
    r"good\s+point|fair\s+enough|perfect|great|nice|good|right|"
    r"understood|i\s+see)\b",
    re.IGNORECASE,
)


def _is_addressed(u: Utterance, speakers: set[str]) -> bool:
    # "@bob" or a bare "you" dropped into the text
    if _ADDRESSED.search(u.text):
        return True
    # or the line just opens with one of the groups names
    return any(
        re.match(rf"^\s*{re.escape(s)}\b", u.text, re.IGNORECASE) for s in speakers
    )


def _reacted_to(u: Utterance, following: list[Utterance]) -> bool:
    mine = content_words(u.text)
    for v in following:
        if v.speaker == u.speaker:
            continue  # self-continuation is not a reaction
        #tried 0.4 overlap first, missed to many real replies
        if overlap_score(mine, content_words(v.text)) >= 0.25 or _ACK.search(v.text):
            return True
    return False


def detect(
    utterances: list[Utterance],
    window: int = 3,
    claimed: list[tuple[int, int]] | None = None,
    group_size_threshold: int = 3,
) -> list[Gap]:
    gaps: list[Gap] = []
    if not utterances:
        return gaps

    speakers = {u.speaker for u in utterances}
    # in a 2-person chat everything looks "ignored"
    if len(speakers) < group_size_threshold:
        return gaps

    claimed_ids: set[int] = set()

    for lo, hi in claimed or []:
        claimed_ids.update(range(lo, hi + 1))

    last_id = utterances[-1].id

    for i, u in enumerate(utterances):
        if i in claimed_ids:
            continue  # another detector already explains this turn

        prev = utterances[i - 1] if i > 0 else None
        if u.act == Act.ANSWER:
            trigger = prev is not None and prev.act in (Act.QUESTION, Act.CLARIFY)
        elif u.act == Act.STATEMENT:
            trigger = _is_addressed(u, speakers)
        else:
            trigger = False
        if not trigger:
            continue

        following = utterances[i + 1: i + 1 + window]
        #print(u.id, u.speaker, trigger)   # debug
        if len(following) < window:
            break  #window never fully observed (end of transcript), not a gap
        if _reacted_to(u, following):
            continue

        end = min(last_id, u.id + window)
        silence = max(end - u.id, 1)
        #window = 5   # people said 3 felt twitchy

        gaps.append(
            Gap(
                type=GapType.IGNORED_RESPONSE,
                turn_range=(u.id, end),
                participants=[u.speaker],
                evidence_quotes=[u.text],
                message_ids=[u.id],
                severity=score_severity(
                    gap_turns=silence, silence=silence, repeats=0, texts=[u.text]
                ),
                description=(
                    f"{u.speaker}'s response at turn {u.id} was never picked up: "
                    f"no other participant referenced or acknowledged it in the "
                    f"next {silence} turn(s)."
                ),
                tone=tone_for_turns([u.text]),
            )
        )
    return gaps
