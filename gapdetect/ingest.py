"""Parse conversation transcripts into Utterance objects.

Accepted inputs:
  * Plain text, one turn per line:  "Alice: hello there"
  * Lines with timestamps:          "[00:01:12] Alice: hello there"
  * JSON list of {"speaker":..., "text":..., "timestamp":...}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Utterance

_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<ts>[^\]]+)\]\s*)?(?P<speaker>[^:]{1,60}?)\s*:\s*(?P<text>.+)$"
)


def parse_text(raw: str) -> list[Utterance]:
    """Parse 'Speaker: text' lines (optionally time-stamped)."""


    utterances: list [Utterance ]=[]
    for line in raw.splitlines() :
        line=line.rstrip ( )
        if not line.strip( ) :

            continue
        m = _LINE_RE.match(line)
        if not m:
            # continuation of the previous turn
            if utterances:
                utterances [ -1 ].text +=" " +line.strip()


            continue
        utterances.append(
            Utterance(
                id = len(utterances),
                speaker=m.group("speaker").strip(),
                text=m.group ( "text").strip( ) ,
                timestamp = m.group ("ts" ),
            )
        )
    return utterances


def parse_json(raw: str) -> list[Utterance]:
    """Parse a JSON list of turn dicts."""
    data = json.loads(raw)

    if isinstance(data, dict):
        data = data.get("transcript", [])
    utterances : list[Utterance] =[]
    for i, item in enumerate(data):
        utterances.append(
            Utterance (
                id= int (item.get("id",i)),
                speaker=str (item.get ("speaker","unknown")).strip( ),
                text=str (item.get ("text","" ) ).strip (),
                timestamp =item.get( "timestamp") ,
            )
        )
    for i, u in enumerate(utterances):
        u.id = i
    return utterances



def parse ( raw:str)->list [ Utterance] :
    """Auto-detect format and parse."""
    stripped = raw.lstrip()
    if stripped.startswith(( "[" , "{" ) ) :
        try:
            return parse_json(raw )
        except( json.JSONDecodeError, KeyError,TypeError ) :
            pass
    return parse_text(raw)


def load ( path: str | Path)->list [Utterance]:
    """Load a transcript from a file (json or txt)."""
    p = Path(path)
    return parse ( p.read_text (encoding = "utf-8"))
