"""Detector 3: the same point keeps getting re-asked and never lands."""

from __future__ import annotations

from ..evidence import score_severity
from ..models import Act, Gap, GapType, Utterance
from ..textutil import content_words, overlap_score
from ..tone import tone_for_turns


def detect(
    utterances: list[Utterance],
    window: int = 6,
    min_cluster: int = 2,
    similarity: float = 0.5,
) -> list[Gap]:
    gaps: list[Gap] = []
    # indexes, not the turns themselves, the window math needs positions
    asks = [i for i, u in enumerate(utterances) if u.act == Act.CLARIFY]

    #min_cluster=1 floods small meetings with noise, dont go there
    if len(asks) < min_cluster:
        return gaps

    # cluster clarifications that are close together AND about the same thing
    #window = 8   # tried wider, way too noisy
    clusters: list[list[int]] = []
    for i in asks:
        words = content_words(utterances[i].text)
        placed = False
        for cl in clusters:
            last_i = cl[-1]
            if i - last_i > window:
                continue
            ref = content_words(utterances[last_i].text)
            # 0.5 overlap, or just one shared word, the set check is generous on purpose
            if overlap_score(words, ref) >= similarity or (words & ref):
                cl.append(i)
                placed = True
                break

        if not placed:
            clusters.append([i])

    for cl in clusters:
        if len(cl) < min_cluster:
            continue
        turns = [utterances[k] for k in cl]
        start, end = turns[0].id, turns[-1].id

        #is it actually resolved? a CLOSURE between first and last clarify = resolved
        if any(u.act == Act.CLOSURE for u in utterances if start <= u.id <= end):
            continue

        who = list(dict.fromkeys(u.speaker for u in turns))
        texts = [u.text for u in turns]
        #silence and repeats land on the same count anyway, scorer reads both
        n_rep = len(cl) - 1
        severity = score_severity(
            gap_turns=end - start,
            silence=n_rep,
            repeats=n_rep,
            texts=texts,
        )
        #print(who, start, end)   # debug
        gaps.append(
            Gap(
                type=GapType.REPEATED_CLARIFICATION,
                turn_range=(start, end),
                participants=who,
                evidence_quotes=texts,
                message_ids=[t.id for t in turns],
                severity=severity,
                description=(
                    f"The same point was asked about {len(cl)} times "
                    f"(turns {start}-{end}) by {', '.join(who)} and was "
                    f"never resolved."
                ),
                tone=tone_for_turns(texts),
            )
        )
    return gaps
