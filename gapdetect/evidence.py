"""Evidence verification and severity scoring.

Every evidence quote must be a VERBATIM substring of the source transcript.
If it isn't, it is dropped (or trimmed to the verifiable core) — so a report
can never contain an invented quote.
"""

from __future__ import annotations

import re
from .models import Severity , Utterance


def verify_quote(quote: str, utterances: list[Utterance]) -> str | None:
    """Return the quote if it appears verbatim in the transcript, else None."""
    q=" ".join(quote.split() )
    if not q :
        return None


    for u in utterances:
        if q in " ".join( u.text.split ( ) ) :
            return q
    #try a smaller window of the quote (first clause)
    for clause in re.split( r"[.;!?]|\s-\s" ,q):

        clause = clause.strip()
        if len(clause) < 12:
            continue
        for u in utterances:
            if clause in " ".join (u.text.split( )) :


                return clause
    return None


def verify_all(quotes: list[str], utterances: list[Utterance]) -> list[str]:
    out :list[str]=[]
    for q in quotes:
        v=verify_quote (q ,utterances)

        if v and v not in out:
            out.append(v)
    return out


DECISION_WORDS = re.compile(
    r"\b(?:deadline|decide|decision|ship|launch|budget|owner|"
    r"approve|scope|release|priority|assign|next\s+step|"
    r"who|when|date|confirm)\b",
    re.IGNORECASE ,
)


def _norm(s: str) -> str:
    return " ".join(s.split( ))


def verify_evidence(gap, utterances) -> bool:
    """Full evidence-integrity check for one gap. Mutates the gap.

    Guarantees on success:
      * every message_id exists in the transcript (ID integrity)
      * every quote is a VERBATIM substring of the message it cites
        (re-anchoring the quote to its true message if the detector
        paired them loosely)
      * message_ids are unique, sorted and inside the transcript bounds
      * turn_range is clamped to valid message ids

    Returns False when nothing can be substantiated → caller drops the gap.
    """

    if not utterances :
        return False
    by_id={u.id : u for u in utterances }
    last = utterances[-1].id

    lo,hi =gap.turn_range
    gap.turn_range = (max(0, min(lo, last)), max(0, min(hi, last)))

    pairs: list[ tuple [ int,str]]= [ ]
    quotes= [_norm( q ) for q in gap.evidence_quotes ]
    ids = list(gap.message_ids )

    # 1) paired (id, quote) checks — verify against THAT message
    for q, tid in zip(quotes, ids):
        u =by_id.get (tid)
        if u is not None and q and q in _norm ( u.text ) :
            pairs.append((tid, q))
            continue
        # 2) re-anchor: quote must exist somewhere verbatim
        if q:
            anchor = next (
                (v.id for v in utterances if q in _norm(v.text )),None
            )
            if anchor is not None:
                pairs.append((anchor, q))
        # 3) otherwise the quote is rejected outright

    #legacy fallback: quotes without declared ids
    if not pairs and not ids :
        for q in quotes:

            anchor = next(( v.id for v in utterances if q in _norm( v.text) ) ,None)
            if anchor is not None :
                pairs.append((anchor, q))

    if not pairs :
        return False

    #dedupe (id, quote), keep quote order stable, then sort by message id
    seen :set [tuple[int, str ] ]=set()
    unique: list[tuple[int, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    unique.sort ( key=lambda p : p[0 ] )

    gap.evidence_quotes= [q for _ ,q in unique]
    gap.message_ids =[tid for tid,_ in unique ]



    # every cited id must exist — final integrity assertion
    if not all (tid in by_id for tid in gap.message_ids ) :
        return False
    # cited ids must belong inside the reported span
    lo ,hi= gap.turn_range
    if any (not(lo <= tid <= hi) for tid in gap.message_ids):
        #widen the span rather than cite outside it
        gap.turn_range= (
            min ( [lo ,* gap.message_ids]) ,
            max([ hi , * gap.message_ids ]),
        )
    return True


def score_severity(
    *, gap_turns: int, silence: int, repeats: int, texts: list[str]
) -> Severity :
    """Heuristic severity for a gap.

    gap_turns : how many turns the gap spans
    silence   : how many turns of silence follow the trigger
    repeats   : how many times the same thing was re-raised
    texts     : involved utterance texts (for decision-critical keywords)
    """
    score =0
    score +=min ( silence, 6 )
    score+=min (gap_turns,6)
    score += 2 * min(repeats, 3)
    if any(DECISION_WORDS.search(t) for t in texts):
        score+=3
    if score >= 10:
        return Severity.HIGH
    if score >= 5:
        return Severity.MEDIUM
    return Severity.LOW
