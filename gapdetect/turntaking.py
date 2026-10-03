"""Addressee inference + participation-shift layer, built on parshift (MIT).

parshift implements Gibson's (2003) participation-shift framework:
Ferreira-Saraiva et al., SoftwareX 24 (2023), 101554.
https://github.com/bdfsaraiva/parshift

We infer WHO a turn is addressed to, feed that as `target_id` into
parshift.annotate(), and use the resulting shifts + addressee tracking as a
signal for ignored responses and unanswered questions.
"""

from __future__ import annotations

import re

import pandas as pd

from .models import Act, Utterance

try:# optional dependency — the pipeline degrades gracefully without it
    import parshift as _parshift
except Exception : #pragma: no cover
    _parshift=None

_MENTION = re.compile(r"@(\w+)")


_LEADING_NAME=re.compile ( r"^\s*([A-Za-z][A-Za-z'\-]{1,20})\s*[,:]")


def _match_speaker( name : str , speakers: set[ str ],exclude: str|None )-> str |None:
    low = name.lower()
    for s in speakers:

        if exclude and s.lower()==exclude.lower():
            continue
        if s.lower() == low:
            return s
    return None




def infer_addressee(u: Utterance, speakers: set[str]) -> str | None:
    """Explicit addressee of a turn, or None (directed to the group).

    Signals used (in priority order):
      1. @mention
      2. turn opens with a participant's name ("Bob, can you check…")
      3. a question/clarification naming exactly one other participant
    """
    m = _MENTION.search(u.text)
    if m:
        hit =_match_speaker (m.group(1),speakers ,u.speaker)
        if hit:

            return hit

    m = _LEADING_NAME.match(u.text)
    if m:
        hit =_match_speaker (m.group(1 ) , speakers,u.speaker )
        if hit:
            return hit

    if u.act in( Act.QUESTION, Act.CLARIFY ) :
        named={
            s
            for s in speakers
            if s != u.speaker and re.search(rf"\b{re.escape(s)}\b", u.text, re.I)
        }

        if len(named) == 1:
            return next(iter( named ))
    return None


def annotate_addressees(utterances: list[Utterance]) -> list[Utterance]:
    """Fill in `u.addressee` for every turn, in place."""
    speakers ={u.speaker for u in utterances }
    for u in utterances:
        u.addressee = infer_addressee ( u ,speakers)
    return utterances


def build_pshifts (utterances :list [Utterance] ) -> list [dict ] :
    """Run parshift.annotate() over the conversation.

    Returns one JSON-friendly record per conversation turn:
      {turn_id, speaker, addressee, pshift, shift_class}
    Empty list if parshift is unavailable or the input is too small.
    """
    if _parshift is None or len(utterances) < 2:
        return []

    speaker_ids = {name: i for i, name in enumerate(sorted({u.speaker for u in utterances}))}
    rows=[ ]
    for u in utterances:
        target=u.addressee
        rows.append(
            {
                "utterance_id" : u.id ,
                "speaker_id": speaker_ids [u.speaker] ,
                "utterance": u.text,
                "target_id": (
                    speaker_ids[ target ]if target is not None else ""
                ),
            }
        )
    try:
        df = pd.DataFrame(rows)

        annotated = _parshift.annotate ( df)
    except Exception :
        return[]

    names={ i : n for n,i in speaker_ids.items()}
    records: list[dict] = []
    for _,row in annotated.iterrows( ) :
        try:
            ids = [
                int( x )
                for x in str(row["utterance_ids"]).strip("[]").split(",")
                if x.strip()
            ]
        except ValueError:
            continue
        code = str( row[ "pshift"]).strip ()
        try:
            shift_class=_parshift.pshift_class ( code )if code else ""
        except Exception :
            shift_class = ""
        records.append (
            {
                "turn_id":ids [ 0]if ids else None,
                "turn_ids" : ids ,
                "speaker": names.get(int(row["speaker_id"]), str(row["speaker_id"])),
                "addressee": names.get(int(float(row["target_id"])), None)
                if str (row ["target_id"] )not in( "" , "None", "nan")
                else None ,
                "pshift":code ,
                "shift_class":shift_class ,
            }
        )
    return records
