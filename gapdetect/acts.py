"""rule-based dialogue-act tagging.

labels: QUESTION, CLARIFY, ANSWER, CLOSURE, STATEMENT
"""

from __future__ import annotations

import re

from .models import Act, Utterance

# interrogative starters, the "can you / does anyone" openers
_STARTERS = re.compile(
    r"^(?:can|could|should|would|will|shall|do|does|did|is|are|was|were|"
    r"has|have|had|may|might|anyone|anybody|someone|somebody|everyone|"
    r"nobody|team|guys|all|please)\b",
    re.IGNORECASE,
)

#wh-words only count next to a starter, otherwise "the reason why we did it" trips us
_WH_RE = re.compile(r"\b(?:who|what|when|where|why|which|how|whose|whom)\b", re.IGNORECASE)

#order matters here: clarify beats question beats closure beats ack/answer
_CLARIFY_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+mean|can\s+you\s+clarify|clarify|"
    r"i\s+(?:didn'?t|did not)\s+(?:understand|get)|"
    r"could\s+you\s+(?:explain|rephrase|elaborate)|"
    r"can\s+you\s+(?:explain|rephrase|elaborate|repeat|clarify)|"
    r"what\s+(?:exactly|precisely|does\s+\w+\s+mean)|"
    r"i'?m\s+(?:a\s+bit\s+)?(?:confused|lost|unsure)|"
    r"not\s+sure\s+(?:what|how|why)|"
    r"say\s+that\s+again|come\s+again|"
    r"can\s+someone\s+(?:explain|clarify)|"
    r"to\s+clarify|just\s+to\s+be\s+clear|"
    r"what\s+does\s+(?:that|this|it)\s+mean|"
    r"i\s+(?:don'?t|do not)\s+(?:follow|understand))",
    re.IGNORECASE,
)

_CLOSURE_RE = re.compile(
    r"\b(?:so\s+(?:we\s+)?(?:are\s+)?(?:all\s+)?agreed|"
    r"we\s+(?:are\s+)?agreed|"
    r"(?:to\s+)?(?:summar(?:y|ize)|summing\s+up)|"
    r"final\s+(?:decision|call|word)|"
    r"let'?s\s+(?:lock|confirm|close|go\s+with)|"
    r"decision\s+(?:is|made)|"
    r"we\s+(?:will|shall|are\s+going\s+to)\s+"
    r"(?:go\s+with|use|ship|proceed)|"
    r"agreed|approved|sign[ -]?off|"
    r"action\s+item|"
    r"to\s+conclude|in\s+conclusion)",
    re.IGNORECASE,
)

_ANSWER_RE = re.compile(
    r"^(?:yes|no|yeah|nope|sure|ok(?:ay)?|right|correct|"
    r"it'?s|it\s+is|it\s+was|they'?re|the\s+answer|"
    r"i\s+(?:think|believe|thought|remember|recall|guess|know|finished|"
    r"have|had|would|will|can|tested|checked|fixed|see|mean)|"
    r"we\s+(?:should|can|will|need)|"
    r"because|since|according\s+to)",
    re.IGNORECASE,
)

#short acks close the question loop, so they count as answers
_ACK_RE = re.compile(
    r"^(?:got\s+it|makes\s+sense|noted|thanks|thank\s+you|"
    r"fair\s+enough|good|great|cool|ack|acknowledged|"
    r"i\s+see|understood)\b",
    re.IGNORECASE,
)

# "i was wondering ..." / "just asking" with no question mark at all
_WONDER_RE = re.compile(r"\b(?:wondering|wonder|asking)\b", re.IGNORECASE)


def _has_question_cue(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        #any trailing ? is enough, dont overthink it
        return True
    if _WH_RE.search(t) and _STARTERS.search(t):
        return True

    # "I was wondering ..." style
    return bool(_WONDER_RE.search(t))


def tag_act(u: Utterance) -> Act:
    t = u.text.strip()

    if _CLARIFY_RE.search(t):
        return Act.CLARIFY
    # a turn ending in '?' wins over closure words, "approved?" is still a question
    if _has_question_cue(t):
        return Act.QUESTION
    if _CLOSURE_RE.search(t):
        return Act.CLOSURE
    if _ACK_RE.search(t):
        return Act.ANSWER

    if _ANSWER_RE.search(t):
        return Act.ANSWER
    return Act.STATEMENT


def tag(utterances: list[Utterance]) -> list[Utterance]:
    """tag every utterance in place, returns the same list"""
    for u in utterances:
        #print(u.id, u.act)   # debug, useful when reordering the checks
        u.act = tag_act(u)
    return utterances
