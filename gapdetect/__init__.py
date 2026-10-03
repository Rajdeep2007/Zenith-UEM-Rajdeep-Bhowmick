"""gapdetect: find communication gaps in multi-person conversations."""

from .models import Act, Gap, GapType, Report, Severity, Utterance
from .pipeline import analyze, analyze_utterances

#re-exported here so callers dont need the submodules
__version__ = "1.0.0"
#bump when the public surface changes
__all__ = [
    "Act",
    "Gap",
    "GapType",
    "Report",
    "Severity",
    "Utterance",
    "analyze",
    "analyze_utterances",
]
