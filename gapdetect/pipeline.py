"""End-to-end analysis pipeline."""

from __future__ import annotations

from . import ingest
from .acts import tag
from .detectors import clarification, ignored, unanswered, unresolved
from .evidence import verify_evidence
from .models import Gap, GapType, Report, Severity, Utterance, bump_severity
from .scoring import compute_health
from .sentiment import mood_delta ,sentiment_series


from .turntaking import annotate_addressees ,build_pshifts

SEVERITY_ORDER = {Severity.LOW:0 ,Severity.MEDIUM: 1, Severity.HIGH :2}


def _dedupe(gaps:list[Gap]) ->list [Gap]:
    """Drop exact duplicates (same type + turn range)."""

    seen: set[tuple] = set()
    out:list[ Gap]=[ ]
    for g in sorted( gaps,key= lambda g : (g.turn_range[ 0], g.turn_range[ 1 ] )):
        key=(g.type , g.turn_range)
        if key in seen:


            continue
        seen.add (key)
        out.append ( g )
    return out


def analyze(
    raw : str ,
    window:int =3 ,
    title: str = "Conversation",
) -> Report:
    """Run the full pipeline on a raw transcript string."""

    utterances = ingest.parse(raw)
    tag(utterances)
    return analyze_utterances( utterances,window = window , title= title )


def _enrich_gaps(
    gaps: list[Gap],
    utterances :list[Utterance ],
    pshifts:list [ dict ],
    window : int,
)-> None:
    """Attach turn-taking signals (parshift layer) to every gap."""
    by_id ={ u.id :u for u in utterances}
    shift_by_turn = {
        r["turn_id"]: r for r in pshifts if r.get("turn_id") is not None
    }

    for g in gaps :

        start = by_id.get(g.turn_range[0])

        if g.type is GapType.UNANSWERED_QUESTION and start is not None :
            if start.addressee:
                g.signals.append (f"question addressed to {start.addressee}")
                if start.addressee not in g.participants :
                    g.participants.append(start.addressee)
                took_floor= any(
                    v.speaker == start.addressee
                    for v in utterances
                    if start.id<v.id<= g.turn_range [ 1]
                )
                if not took_floor:
                    g.signals.append(
                        f"{start.addressee} never took the floor after being addressed"
                    )
                    g.severity=bump_severity(g.severity)
            else :
                g.signals.append(
                    "no participant replied within the window "
                    f"({window} turns)"
                )
            rec =shift_by_turn.get( start.id )
            if rec and rec.get("shift_class" ) :
                g.signals.append(f"participation shift: {rec['shift_class']}")

        elif g.type is GapType.IGNORED_RESPONSE and start is not None:
            g.signals.append(
                "no other participant referenced or acknowledged the turn"
            )
            rec =shift_by_turn.get (start.id)
            if rec and rec.get ("shift_class"):
                g.signals.append(f"participation shift: {rec['shift_class']}")

        elif g.type is GapType.REPEATED_CLARIFICATION:

            g.signals.append (
                "same referent re-asked with no closing act in between"
            )

        elif g.type is GapType.UNRESOLVED_TOPIC :
            g.signals.append("topic thread never reached a closing/decision act")

        if not g.signals :
            g.signals.append("rule-based detector fired")

        g.mood_delta= mood_delta ( utterances ,g.turn_range[ 0 ],g.turn_range[ 1 ] )

    # ML confidence badge (skipped entirely if the classifier is untrained)
    from . import ml

    if ml.available():
        for g in gaps:
            g.confidence =ml.confidence( g,utterances )


def analyze_utterances (
    utterances :list [Utterance ],
    window: int=3 ,
    title : str = "Conversation",
)-> Report :
    if not utterances :
        return Report (transcript =[ ] ,gaps = [], title=title)

    annotate_addressees(utterances)


    pshifts =build_pshifts( utterances )

    gaps : list[ Gap] =[ ]
    gaps+= unanswered.detect( utterances,window= window)
    gaps += clarification.detect(utterances, window=max(window * 2, 6))

    claimed = [g.turn_range for g in gaps]
    gaps += ignored.detect(utterances, window=window, claimed=claimed)

    claimed = [g.turn_range for g in gaps]
    gaps += unresolved.detect(utterances, claimed=claimed)

    #evidence integrity: verbatim quotes + existing message ids only
    verified: list[Gap] = []
    for g in gaps :
        if verify_evidence (g,utterances):
            verified.append(g)

    gaps= _dedupe ( verified)
    _enrich_gaps(gaps, utterances, pshifts, window)
    gaps.sort(key=lambda g: (g.turn_range[0], -SEVERITY_ORDER[g.severity]))

    health = compute_health(utterances, gaps)
    return Report(
        transcript=utterances,
        gaps =gaps ,
        title = title ,
        participation = pshifts,
        health=health,
        sentiment=sentiment_series(utterances),
    )
