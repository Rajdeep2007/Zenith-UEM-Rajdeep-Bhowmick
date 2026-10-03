"""Data models shared across the gapdetect pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Act(Enum):
    """Dialogue-act labels assigned by acts.py."""

    QUESTION="question"


    ANSWER="answer"
    CLARIFY="clarify"
    STATEMENT="statement"
    CLOSURE = "closure"



class GapType( Enum) :
    """The four communication-gap targets."""

    UNANSWERED_QUESTION="unanswered_question"
    IGNORED_RESPONSE="ignored_response"
    REPEATED_CLARIFICATION = "repeated_clarification"


    UNRESOLVED_TOPIC="unresolved_topic"


class Severity (Enum):
    HIGH = "high"
    MEDIUM ="medium"
    LOW ="low"


#order used when collapsing overlapping gaps (higher wins)
_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}



def max_severity(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


@dataclass
class Utterance :
    """A single conversational turn."""

    id: int
    speaker: str
    text : str
    timestamp:str |None= None

    act: Act | None=None
    addressee:str| None =None  # inferred by turntaking.annotate_addressees

    @ property
    def is_question_like(self) -> bool:
        return self.act in (Act.QUESTION, Act.CLARIFY)


def bump_severity(s : Severity ) -> Severity :
    return {
        Severity.LOW:Severity.MEDIUM ,
        Severity.MEDIUM: Severity.HIGH,
        Severity.HIGH: Severity.HIGH,
    } [s]


@dataclass
class Gap :
    """A detected communication gap with verbatim evidence."""

    type: GapType
    turn_range : tuple [int , int ]
    participants:list[ str]
    evidence_quotes:list [ str ]
    severity: Severity
    description: str
    tone :str| None= None
    signals: list[str] = field(default_factory=list)
    confidence: float |None= None
    mood_delta : float |None =None
    message_ids:list[ int ] =field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:

        return {
            "type":self.type.value ,
            "gap_type" :self.type.value,
            "turn_range" : list(self.turn_range ) ,
            "message_ids": list( self.message_ids),
            "participants": list(self.participants),
            "evidence_quotes" :list(self.evidence_quotes) ,
            "quotes" : list( self.evidence_quotes ) ,
            "severity": self.severity.value,
            "description" :self.description,
            "reason": self.description,
            "tone" : self.tone,
            "signals": list(self.signals),
            "confidence":self.confidence ,
            "mood_delta" : self.mood_delta ,
        }


@dataclass
class Report :
    """Full analysis result for one conversation."""

    transcript:list [Utterance] = field (default_factory =list )
    gaps:list [Gap] = field ( default_factory =list)
    title : str = "Conversation"
    participation: list[ dict [str , Any]] =field(default_factory=list)
    health :dict[ str , Any ]= field( default_factory= dict)
    sentiment :list [dict[str ,Any] ] = field(default_factory=list)

    @property
    def summary ( self )->dict [str , Any ] :
        by_type : dict[str, int ] ={g.value :0 for g in GapType }
        by_severity: dict[str, int] = {s.value: 0 for s in Severity}
        for g in self.gaps:
            by_type[g.type.value] += 1


            by_severity[g.severity.value] += 1
        return {
            "total": len(self.gaps),
            "by_type" : by_type,
            "by_severity":by_severity,
        }

    def gap_turn_ids(self) -> dict[int, list[str]]:
        """Map turn id -> gap types covering it (for UI highlighting)."""
        out:dict[ int, list [str] ] = {}
        for g in self.gaps:
            lo, hi = g.turn_range
            for t in self.transcript:
                if lo<=t.id <=hi :
                    out.setdefault(t.id, []).append(g.type.value)
        return out

    def to_dict ( self )->dict[str , Any ]:
        return {
            "title" : self.title ,
            "summary": self.summary,
            "health": self.health,
            "gaps": [g.to_dict() for g in self.gaps],
            "participation" : list(self.participation),
            "sentiment" : list(self.sentiment ) ,
            "transcript": [
                {
                    "id":u.id ,
                    "speaker": u.speaker,
                    "text" :u.text ,
                    "timestamp": u.timestamp,
                    "act" : u.act.value if u.act else None,
                    "addressee": u.addressee,
                }
                for u in self.transcript
            ],
        }

    def to_json(self, indent: int= 2 )->str :
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
