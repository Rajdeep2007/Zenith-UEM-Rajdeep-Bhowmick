"""ML confidence layer.

A TF-IDF + LogisticRegression classifier trained on synthetic gap injection
(see train.py) that scores each detected gap window as "real gap" vs
"normal exchange". Produces the confidence % badge on gap cards.

Gracefully degrades: if the model artifact is missing, confidence() returns
None and the UI simply hides the badge — the rule-based path never depends
on the classifier.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .models import Gap, Utterance

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "gap_classifier.joblib"
METRICS_PATH= Path ( __file__).resolve().parent.parent /"artifacts" / "train_metrics.json"


def window_text( utterances:list[Utterance] ,gap: Gap )-> str :
    """Concatenate every message inside the gap's turn range."""


    lo, hi= gap.turn_range
    return "\n".join (u.text for u in utterances if lo <= u.id<= hi)


def window_features(utterances: list[Utterance], span: tuple[int, int]) -> str:
    """Feature text for one window: dialogue-act sequence + message text.

    The act sequence (ACT_QUESTION act_STATEMENT ...) captures the structure
    of a gap (an open question followed by chatter, two clarifies in a row…)
    which raw TF-IDF text alone cannot see. train.py and this module must use
    the same representation.
    """

    lo,hi= span
    acts =" ".join(f"ACT_{u.act.name}" for u in utterances if lo<=u.id<=hi )
    text ="\n".join( u.text for u in utterances if lo <= u.id <= hi)
    return f"{acts} {text}"


@ lru_cache( maxsize =1)
def _model( ):


    try:
        import joblib

        if MODEL_PATH.exists( ):
            return joblib.load ( MODEL_PATH)
    except Exception:
        pass
    return None



def available() -> bool:
    return _model() is not None


def confidence(gap: Gap, utterances: list[Utterance]) -> float | None:
    """P(real gap) for this gap's window, in [0, 1]; None if untrained."""
    model=_model()
    if model is None:
        return None
    text = window_features(utterances, gap.turn_range)

    if not text.strip():
        return None
    try:
        proba= model.predict_proba( [text])[0]
        return round (float(proba [1 ]),3 )
    except Exception :
        return None
