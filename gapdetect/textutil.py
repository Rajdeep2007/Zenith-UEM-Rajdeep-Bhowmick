"""Small text helpers shared by detectors."""

from __future__ import annotations

import re

STOPWORDS = set(
    """a an the and or but if then than so of to in on at for with from by as
    is are was were be been being do does did doing have has had having
    i you he she it we they me him her us them my your his its our their
    this that these those there here what which who whom whose when where why
    how not no nor too very can could should would will shall may might must
    about into over under again once all any both each few more most other
    some such only own same s t don now up down out off above below
    um uh like just really going get got go one two also need needs
    think thought know want let lets right ok okay yeah yes guy guys
    team please thanks thank sounds going""".split()
)

_TOKEN_RE = re.compile( r"[a-z0-9']+" )


def stem (word:str )-> str :
    """Very light suffix stripping so 'finish'~'finished', 'test'~'tests'."""
    if len( word)>4 and word.endswith("ies" ):
        return word[ : -3 ] +"y"


    if len(word) > 5 and word.endswith(("ing", "ed")):
        return word [: - 3]
    if len ( word) >4 and word.endswith("es") :
        return word [:-2]

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[: - 1]
    return word


def tokens( text : str ) ->list [str ] :
    return _TOKEN_RE.findall(text.lower())




def content_words(text: str)-> set[ str ]:
    return{stem(w)for w in tokens(text) if w not in STOPWORDS and len( w)>2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)

    if not inter:
        return 0.0
    return inter/len (a|b )


def overlap_score(a: set[str], b: set[str]) -> float:
    """Overlap relative to the smaller set — catches 'budget' vs 'budget numbers'."""
    if not a or not b:
        return 0.0
    inter= len(a& b )

    if not inter :
        return 0.0
    return inter/min (len( a) , len (b))
