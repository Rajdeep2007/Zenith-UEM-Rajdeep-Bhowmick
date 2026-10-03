"""Per-turn sentiment polarity for the emotion-trajectory chart.

Blends TextBlob polarity (offline) with our own lexical tone cues, so affect
markers like "Hello? Anyone?" still show up as a mood dip even though
TextBlob scores them as neutral.
"""

from __future__ import annotations

from .models import Utterance
from .tone import tone_of

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None  # type: ignore
#textblob is optional, without it we lean entirely on the lexicon

_LEXICON_POLARITY = {
    "frustration": -0.6,
    "urgency": -0.3,
    "uncertainty": -0.25,
    "neutral": 0.0,
}

_TEXTBLOB_WEIGHT = 0.55

# they sum to 1, moving one means moving the other
_LEXICON_WEIGHT = 0.45


def polarity(text: str) -> float:
    """Polarity in [-1, 1] for a single message."""
    tb = 0.0

    if TextBlob is not None:
        try:
            tb = float(TextBlob(text).sentiment.polarity)
        except Exception:
            #textblob chokes on some inputs, dont let it kill the turn
            tb = 0.0
    # "Hello? Anyone?" is neutral to textblob, the lexicon catches it
    lex = _LEXICON_POLARITY.get(tone_of(text), 0.0)

    # print(tb, lex)   # debug, blend looked too textblob-heavy at first
    blended = _TEXTBLOB_WEIGHT * tb + _LEXICON_WEIGHT * lex
    # textblob promises [-1,1] so this shouldnt fire, clamping anyway
    return max(-1.0, min(1.0, blended))


def sentiment_series(utterances: list[Utterance]) -> list[dict]:
    """One record per turn: {turn, speaker, polarity}."""
    return [
        {"turn": u.id, "speaker": u.speaker, "polarity": round(polarity(u.text), 3)}
        for u in utterances
    ]


def mood_delta(utterances: list[Utterance], lo: int, hi: int) -> float | None:
    """Polarity change from before the gap to inside the gap.

    Positive = mood improved, negative = mood dipped during the gap.
    """
    # two turns back is the baseline; wider than that and normal chit-chat drowns it
    before = [polarity(u.text) for u in utterances if lo - 2 <= u.id < lo]
    during = [polarity(u.text) for u in utterances if lo <= u.id <= hi]
    # print(len(before), len(during))   # counts while tuning the window
    if not before or not during:
        return None
    # None here means the chart just leaves the marker off
    return round(sum(during) / len(during) - sum(before) / len(before), 3)
