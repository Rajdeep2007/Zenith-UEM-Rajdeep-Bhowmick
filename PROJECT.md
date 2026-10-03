# GapSense — Detecting Communication Gaps in Group Conversations

**Event:** CEREBRO Hackathon, Dept. of CSE, Institute of Engineering & Management (IEM), Kolkata
**Problem Statement:** Detecting Communication Gaps in Group Conversations
**Track:** NLP / Affective Computing
**Stack:** Python 3.14 · Streamlit · scikit-learn · parshift · TextBlob · pure rule-based core, fully offline

---

## 1. The Problem (slide 1)

In team meetings, standups and group chats, communication silently fails:
questions get no answers, replies get ignored, people re-ask the same thing,
and topics are opened but never closed. These **gaps cause delays, rework and
conflict** — yet nobody notices them in real time.

**Goal:** A system that analyses a multi-person conversation, identifies
communication gaps, **highlights where the gap occurred**, and provides
**evidence quoted directly from the conversation** — with verifiable message
IDs, never invented quotes.

---

## 2. Gap Taxonomy (slide 2)

| # | Gap Type | Signal | Evidence Shown |
|---|----------|--------|----------------|
| 1 | **Unanswered Question** | A question receives no real reply inside the reply window | The exact question turn(s) |
| 2 | **Ignored Response** | An answer/statement that nobody references or acknowledges | The ignored turn + silent window |
| 3 | **Repeated Clarification** | The same point re-asked ≥ 2 times, never resolved | All re-ask turns |
| 4 | **Unresolved Topic** | A topic discussed but never closed with a decision | First & last turn of the thread |

Each gap carries: **severity** (high/medium/low), **tone cue**
(frustration/urgency/uncertainty), **turn range**, **cited message IDs**,
**participants**, **signals** (why the detector fired), **mood delta**, and
an ML **confidence %**.

### Theoretical grounding
- **Adjacency pairs** (Schegloff & Sacks): a question expects an answer → a gap is a *broken adjacency pair*.
- **Grounding theory** (Clark & Brennan): mutual understanding must be established → failure appears as repeated clarification.
- **Gibson's participation shifts** (2003, via ParShift): speaker → target → non-target transitions reveal *who was addressed and whether they ever took the floor*.
- **Topic structure**: open → develop → close; a gap is a thread that never closes.

---

## 3. Architecture (slide 3 — the pipeline diagram)

```
chat file (Speaker: text / JSON / paste)
   │
   ▼
Ingest ──► Dialogue-act tagging (QUESTION / CLARIFY / ANSWER / CLOSURE / STATEMENT)
   │
   ├─► Addressee inference (@mention, leading name, named-in-question)
   │        └─► ParShift (MIT, pip) → Gibson participation-shift codes
   │                └─► signals for ignored / unanswered detectors
   │
   ├─► 4 detectors (rule engines over turn windows)
   │      unanswered · ignored · clarification · unresolved
   │
   ▼
EVIDENCE ENGINE (ours)
   merge candidates → verbatim-quote verification → message-ID integrity
   → turn-taking signals → severity scoring → tone cue
   │
   ▼
ENRICHMENT
   mood_delta (sentiment) · ML confidence (ml.py) · health score (scoring.py)
   │
   ▼
Report JSON  {gap_type, severity, turn_range, message_ids, quotes, reason,
              signals, tone, mood_delta, confidence} + health + sentiment
   │
   ▼
Streamlit dashboard
   highlighted transcript · gap evidence cards · health score ·
   emotion trajectory · live simulation · JSON export
```

### The evidence engine (our core IP — what makes it trustworthy)
1. **Verbatim verification** — every quote must be an exact substring of the source transcript; otherwise it is dropped. *No hallucinated evidence, ever.*
2. **Message-ID integrity** — every gap cites real turn IDs; quotes are re-anchored to the exact message they came from (or rejected), IDs are deduped/sorted, ranges clamped to valid bounds. `eval.py` prints **Evidence integrity: 6/6** alongside P/R/F1.
3. **Cross-signal check** — gaps carry the signals that produced them (rule fired + participation shift + addressee behaviour), so every verdict is auditable.
4. **Deterministic, offline** — no API keys, no network, reproducible results (unlike LLM-as-verifier, which can itself hallucinate).

### The confidence layer (Phase 5 — how the ML number is honest)
Rule-based detectors stay the source of truth; a classifier only *scores* their windows.
`train.py` builds its own labelled dataset with **synthetic gap injection**: each of the 4
gap patterns is injected into the clean meeting at several positions (18/18 detected by the
real detectors), plus real detector windows (positives) and sliding non-gap windows
(negatives) → 24 pos / 74 neg. Features = dialogue-act sequence + TF-IDF text;
LogisticRegression; evaluated with **5-fold StratifiedGroupKFold grouped by source
conversation (no window from the same conversation in two folds)** →
**F1 = 0.823 ± 0.216, Precision = 0.900, Recall = 0.800**. The fitted model is saved to
`artifacts/gap_classifier.joblib` and shown as a per-gap **confidence badge** (green ≥80%,
amber ≥60%). Missing artifact ⇒ badges are hidden, detection unaffected.

---

## 4. Detection rules (how each gap is found)

| Detector | Logic |
|---|---|
| Unanswered question | Question turn → look ahead *window* turns for a real response: answer/closure act, ≥25% content-word overlap, or an immediate commitment ("let me check"). If the question was addressed to a named person, the engine records whether that person ever took the floor (and bumps severity if not). Consecutive unanswered questions by the same speaker merge into one gap. |
| Ignored response | Trigger: an answer that directly follows a question, or a statement aimed at someone (@mention / "you" / name). Condition: within the window **no other speaker references it** (content overlap) and **no other speaker acknowledges it** (ack lexicon). Full window must be observed (end-of-chat is not a gap). |
| Repeated clarification | Turns tagged CLARIFY clustered by content-word overlap (with light stemming) within a window; skipped if a CLOSURE act appears between the first and last ask. |
| Unresolved topic | Topics segmented by content-word continuity; flagged when substantive (≥4 turns), decision-relevant or questioned, and never closed by a CLOSURE act (including the turn right after the thread). Spans already explained by other detectors are excluded (no double-reporting). |

**Supporting layers:** light stemmer + stopword filter + overlap scoring;
dialogue-act lexicons; lexical tone cues; parshift participation shifts.

---

## 5. Technology stack (slide 4)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.14 | NLP standard |
| Detection core | Rule-based (regex + lexical cues + word-overlap clustering) | Offline, deterministic, explainable, zero training data needed for v1 |
| Turn-taking | **ParShift** (`pip install parshift`, MIT) | Gibson participation-shift framework, peer-reviewed (SoftwareX 2023) |
| ML layer | scikit-learn: TF-IDF (act sequence + text) + LogisticRegression trained on **synthetic gap injection** — grouped CV **F1 = 0.823, P = 0.900, R = 0.800** (5-fold StratifiedGroupKFold, no conversation leaks across folds) | Confidence % per gap without any hand-labelled dataset |
| Sentiment | TextBlob + tone-lexicon blend | Offline per-turn polarity for the emotion trajectory |
| UI | Streamlit 1.58 | Fast interactive dashboard, single file |
| Eval | Custom P/R/F1 harness vs hand-labelled ground truth | |

**Key differentiator:** no black-box model on the critical path — every flagged
gap traces back to an exact turn number, an exact quote, and the signals that
produced it.

---

## 6. Innovations roadmap (slides: build progress)

| Phase | Feature | Status |
|---|---|---|
| 1 | **ParShift turn-taking layer** — addressee inference + Gibson participation-shift codes as detector signals | ✅ done |
| 2 | **Communication Health Score 0–100** — responsiveness / closure / inclusion / tone sub-scores + gauge | ✅ done |
| 3 | **Emotion trajectory chart** — per-speaker sentiment over turns with gap spans shaded + "mood delta" per gap | ✅ done |
| 4 | **Evidence engine hardening** — `message_ids`, `signals`, ID-integrity checks in output contract | ✅ done |
| 5 | **Trained ML classifier** — auto-injected synthetic gaps → TF-IDF + LogisticRegression (CV F1 reported) → confidence % badges | ✅ done |
| 6 | **Live simulation** — play/pause streaming of the conversation, gaps pop as alerts in real time | ✅ done |
| stretch | Agree/disagree lexicon (ADR-inspired) · WhatsApp export parser | ⬜ |

---

## 7. Output format (machine-readable — exactly what `analyze()` returns)

```json
{
  "title": "Team Standup",
  "summary": {
    "total": 4,
    "by_type": {"unanswered_question": 1, "ignored_response": 2, "repeated_clarification": 1, "unresolved_topic": 0},
    "by_severity": {"high": 1, "medium": 3, "low": 0}
  },
  "health": {
    "overall": 53,
    "grade": "D",
    "subscores": {"responsiveness": 71, "closure": 0, "inclusion": 81, "tone": 60}
  },
  "gaps": [
    {
      "type": "unanswered_question",
      "gap_type": "unanswered_question",
      "turn_range": [4, 7],
      "message_ids": [4],
      "participants": ["Alice", "Dave"],
      "evidence_quotes": ["Dave, good - do you know if the staging environment is back up?"],
      "quotes": ["Dave, good - do you know if the staging environment is back up?"],
      "severity": "high",
      "description": "Question from Alice at turn 4 received no reply in the following 3 turn(s).",
      "reason": "Question from Alice at turn 4 received no reply in the following 3 turn(s).",
      "tone": "frustration",
      "signals": ["question addressed to Dave", "Dave never took the floor after being addressed", "participation shift: Turn Claiming"],
      "confidence": 0.734,
      "mood_delta": -0.224
    }
  ],
  "participation": [{"turn_id": 0, "turn_ids": [0], "speaker": "Alice", "addressee": null, "pshift": "", "shift_class": ""}],
  "sentiment": [{"turn": 0, "speaker": "Alice", "polarity": 0.183}],
  "transcript": ["…every turn, id + speaker + text + act + addressee…"]
}
```

---

## 8. The app (demo script for judges, slide 5)

1. Sidebar: pick **Team Standup** (4 gaps, health 53/D) · **Design Review** (2 gaps, health 36/F) · **Clean Meeting** (0 gaps = false-positive control, health 88/B) · **Paste your own** (`Speaker: text`).
2. Adjust reply window (2–6 turns).
3. **Highlighted transcript** tab — gap turns color-coded by type.
4. **Gap report** tab — evidence cards: type badge, severity, tone, turn range, **cited message IDs**, verbatim quotes, detector signals, **mood delta**, **confidence badge**.
5. **Emotion** tab — per-speaker sentiment lines with gap spans shaded; callout of the largest mood drop.
6. **Live simulation** tab — ▶ Play streams the conversation turn by turn; alerts pop the moment each gap becomes detectable (⏸ Pause / ⏭ Step / ⟲ Reset).
7. **JSON** tab — full structured report + download.

*Suggested demo flow: Standup (frustrated "Hello? Anyone?" + named addressee ignored, health drops to 53) → Live simulation replay of the same chat with alerts popping → Design Review (unresolved debate, health 36, mood dip) → Clean Meeting (proves zero false alarms, health 88) → paste a real chat.*

---

## 9. Evaluation (slide 6)

- 3 hand-labelled transcripts with **injected, known gaps** + 1 **clean control**.
- Matching: predicted vs labelled gap of the same type whose turn IDs intersect (one-to-one).
- Current result:

```
TOTAL   TP=6  FP=0  FN=0
Precision=1.000  Recall=1.000  F1=1.000
```

- Clean meeting: **0 predicted gaps** (no false positives).
- Evidence integrity: **6/6** — `eval.py` re-checks that every cited message ID exists and every quote is verbatim in the message it cites.
- ML layer: **grouped 5-fold CV F1 = 0.823 (P = 0.900, R = 0.800)** on synthetic gap injection — reported by `python train.py`; reproducible end-to-end on Kaggle (`kaggle_notebook.ipynb` + `gapsense_kaggle.zip`).

---

## 10. Why it matters / novelty (slide 7)

- Turns invisible meeting failures into **actionable, evidenced reports** — with the exact quotes a moderator can act on.
- **Theory-grounded**: adjacency pairs + grounding theory + Gibson participation shifts (ParShift), not just regex heuristics.
- Works **offline, instantly, zero training data** for the rule core; ML layer trains on *auto-generated* labels.
- Evidence-first design → results are **auditable and trustworthy** (quote verification + message-ID integrity).
- Tone layer bridges NLP with **affective computing** (CEREBRO theme); emotion trajectory shows mood dips at gaps.

---

## 11. Future scope (slide 8)

- LLM-assisted coreference ("who was addressing whom") + optional narrative meeting brief (feature-flagged)
- Live meeting integration (Zoom/Teams transcription)
- Action-item / commitment tracking with kept-or-dropped status
- Multilingual & Hinglish code-mixed conversations
- Before/after intervention comparison (health score over time)

---

## 12. Acknowledgements & licenses (slide footer — judges check this)

- **ParShift** — https://github.com/bdfsaraiva/parshift — **MIT License**, integrated via PyPI.
  Ferreira-Saraiva, B.D., Matos-Carvalho, J.P., Fachada, N. & Pita, M. (2023).
  *ParShift: A Python package to study order and differentiation in group conversations.* SoftwareX 24, 101554.
- **Gibson (2003)** participation-shift framework (theory).
- Inspirations only, **no code used**: *groupchat-decoder* (no license — feature-list inspiration only),
  *Agreement-and-Disagreement-Recognition* (MPL-2.0 — signal idea only),
  *chat-analytics* (parsing approach only).

---

## 13. Repo layout

```
gapdetect/
  models.py       Utterance, Gap, Report, enums
  ingest.py       transcript parsing (Speaker: text / JSON)
  acts.py         dialogue-act tagging
  turntaking.py   addressee inference + ParShift participation shifts
  tone.py         lexical tone cues
  sentiment.py    per-turn polarity (TextBlob + tone blend), mood deltas
  scoring.py      communication health score
  ml.py           confidence classifier wrapper (loads artifacts/)
  evidence.py     verbatim quote verification + message-ID integrity + severity
  textutil.py     tokenising, stemming, overlap scoring
  pipeline.py     end-to-end analyse()
  detectors/      unanswered, ignored, clarification, unresolved
data/             3 demo transcripts + ground_truth/ (hand-labelled)
app.py            Streamlit dashboard (5 tabs incl. live simulation)
eval.py           P/R/F1 + evidence-integrity harness
train.py          synthetic gap injection → classifier training
requirements.txt  runtime dependencies
artifacts/        gap_classifier.joblib + train_metrics.json (created by train.py)
kaggle_notebook.ipynb + gapsense_kaggle.zip   reproducible Kaggle training run
README.md         usage + full credits
PROJECT.md        this document (source for the PPT)
```
