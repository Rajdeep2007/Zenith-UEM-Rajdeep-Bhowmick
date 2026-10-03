"""Lexical tone cues for the affective-computing angle.

Pure keyword/regex scoring — offline, deterministic.
Returns one of: frustration | uncertainty | urgency | neutral
"""

from __future__ import annotations

import re

_CUES: dict[str, re.Pattern[str]] = {
    "frustration": re.compile(
        r"\b(?:again\?|still\s+(?:waiting|not|no)|nothing\s+happened|"
        r"never\s+(?:mind|heard)|can\s+anybody\s+hear|is\s+anyone\s+there|"
        r"hello\?+|anyone\?+|frustrat\w*|annoy\w*|ridiculous|"
        r"this\s+is\s+(?:the\s+)?(?:second|third)\s+time|"
        r"why\s+(?:is|does|do)\s+(?:nobody|no\s+one)|"
        r"nobody\s+(?:cares|responds|replied)|ignored|dropped)" ,
        re.IGNORECASE ,
    ) ,
    "urgency":re.compile(
        r"\b(?:asap|urgent\w*|immediately|right\s+now|today|deadline|"
        r"block\w*\s+(?:me|us|the\s+team)|critical|priority|"
        r"we\s+need\s+(?:this|it)\s+(?:now|today)|before\s+(?:eod|deadline)|"
        r"time\s+sensitive|escalat\w*)",
        re.IGNORECASE ,
    ),
    "uncertainty" : re.compile(
        r"\b(?:not\s+sure|i'?m\s+(?:not\s+)?sure|unclear|"
        r"unsure|confused|lost|unsure|uncertain|maybe|perhaps|"
        r"i\s+(?:don'?t|do not)\s+know|no\s+idea|"
        r"what\s+does\s+(?:that|this|it)\s+mean|"
        r"can\s+someone\s+(?:clarify|explain)|"
        r"any\s+(?:idea|clue)|guess\s+so|"
        r"do\s+we\s+(?:even|actually)\s+know)",
        re.IGNORECASE ,
    ),
}


def tone_of(text: str) -> str:
    """Return the strongest tone cue detected in `text` (or 'neutral')."""
    scores:dict[str ,int] = {}
    for name , pat in _CUES.items( ):
        n = len(pat.findall(text))


        if n:
            scores[name] = n
    if not scores:
        return "neutral"

    return max ( scores.items( ) ,key =lambda kv:(kv [ 1],_ORDER.index(kv[ 0 ])) ) [ 0 ]


_ORDER = ["frustration", "urgency", "uncertainty"]


def tone_for_turns(texts: list[str]) -> str:
    """Tone across several turns (used for gap-level cue)."""
    best ="neutral"


    for t in texts :
        cand= tone_of (t)
        if cand!= "neutral":
            if best== "neutral":
                best = cand
            elif cand == "frustration":
                return "frustration"

    return best
