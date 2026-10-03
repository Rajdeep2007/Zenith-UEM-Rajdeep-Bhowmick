# GapSense

**Communication Gap Detector** for group conversations — CEREBRO hackathon, IEM Kolkata.

Analyses a multi-person conversation, finds **communication gaps**, shows **where** they
happened in the transcript, and quotes **verbatim evidence** — every flagged gap traces
back to an exact turn number, an exact quote, and the signals that produced it.

- **Offline & deterministic** — no API keys, no LLM on the critical path
- **Evidence-first** — quotes are verified verbatim against cited message IDs; nothing invented
- **Evaluated** — Precision / Recall / F1 = **1.000** on hand-labelled ground truth, evidence integrity **6/6**

---

## Architecture

```
   INPUT
   "Speaker: text" lines · JSON · Streamlit paste box
        │
        ▼
   ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
   │  ingest.py  │    │     acts.py          │    │    turntaking.py        │
   │ parse turns │───▶│ dialogue-act tagging │    │ addressee inference     │
   │ (+ JSON)    │    │ QUESTION  ANSWER     │    │ (@mention / leading     │
   └─────────────┘    │ CLARIFY   CLOSURE    │    │  name / named-in-q)     │
                      │ STATEMENT            │    │ + ParShift codes (MIT)  │
                      └──────────┬───────────┘    └───────────┬─────────────┘
                                 │                            │
                                 ▼                            │
   ┌──────────────────────────────────────────────────────────┘
   │  4 DETECTORS  (gapdetect/detectors/, run independently over turn windows)
   │
   │   unanswered_question   question → look ahead N turns for a real response
   │                         (answer/closure act · commitment cue · ≥25% content overlap)
   │   ignored_response      answer or addressed statement → nobody references
   │                         or acknowledges it inside the window
   │   repeated_clarification  same referent re-asked ≥2× clustered by content-word
   │                         overlap, skipped if a closure act lands between asks
   │   unresolved_topic      topics segmented by content-word continuity → flagged
   │                         when substantive, decision-relevant, never closed
   │                         (spans already covered by other detectors excluded)
   └───────────────┬─────────────────────────────────────────────┘
                   ▼
   ┌─────────────────────────────────────────────┐
   │  EVIDENCE ENGINE  (evidence.py)  ← core IP  │
   │  · every quote must be VERBATIM in the      │
   │    message it cites (re-anchor or reject)   │
   │  · message_ids must exist, sorted, deduped  │
   │  · turn range clamped to valid bounds       │
   │  · unsubstantiated gaps are DROPPED         │
   └───────────────┬─────────────────────────────┘
                   ▼
   ┌─────────────────────────────────────────────┐
   │  ENRICHMENT  (pipeline.py + helpers)        │
   │  · signals: rule fired · parshift shift ·   │
   │    addressee never took the floor           │
   │  · severity score (silence · repeats ·      │
   │    decision keywords) + tone cue            │
   │  · mood_delta from sentiment.py (TextBlob   │
   │    + tone-lexicon blend)                    │
   │  · confidence from ml.py (TF-IDF + LR       │
   │    classifier, trained by train.py)         │
   │  · health score from scoring.py (0–100)     │
   └───────────────┬─────────────────────────────┘
                   ▼
   REPORT JSON
   {gap_type, severity, turn_range, message_ids, quotes, reason,
    signals, tone, mood_delta, confidence} + health + sentiment + participation
                   │
                   ▼
   ┌─────────────────────────────────────────────┐
   │  STREAMLIT DASHBOARD  (app.py)              │
   │  highlighted transcript · evidence cards ·  │
   │  health gauge · emotion trajectory ·        │
   │  live simulation · JSON export              │
   └─────────────────────────────────────────────┘
```

**Design principle:** rule-based detectors own every decision; the ML model only *scores*
windows the rules already produced; the evidence engine can veto any gap. That is what
makes the output auditable — unlike an LLM verifier, this pipeline cannot hallucinate a
quote, and `eval.py` proves it on every run.

### Module map

| Layer | Modules |
|---|---|
| Data model | `gapdetect/models.py` — `Utterance`, `Gap`, `Report`, enums |
| Parsing | `ingest.py` (text/JSON) · `textutil.py` (stemming, overlap scoring) |
| Understanding | `acts.py` (dialogue acts) · `tone.py` (affect lexicon) · `turntaking.py` (addressee + ParShift) |
| Detection | `detectors/unanswered.py` · `ignored.py` · `clarification.py` · `unresolved.py` |
| Trust | `evidence.py` (verbatim + ID integrity + severity) |
| Scoring | `scoring.py` (health) · `sentiment.py` (mood) · `ml.py` (confidence) |
| Orchestration | `pipeline.py` — `analyze()` end to end |
| Surfaces | `app.py` (UI) · `eval.py` (P/R/F1) · `train.py` (classifier) |

---

## Gap types

| Type | What it catches |
|------|-----------------|
| `unanswered_question` | A question that receives no reply inside the reply window |
| `ignored_response` | An answer/statement nobody references or acknowledges |
| `repeated_clarification` | The same point re-asked ≥ 2 times, never resolved |
| `unresolved_topic` | A topic discussed but never closed with a decision |

Each gap carries **severity** (high/medium/low), a **tone cue** (frustration / urgency /
uncertainty / neutral), **cited message IDs**, verified **verbatim quotes**, a **mood
delta**, and an ML **confidence %**.

---

## Quick start

```bash
pip install -r requirements.txt

streamlit run app.py      # interactive demo  → http://localhost:8501
python eval.py            # precision / recall / F1 vs ground truth
python train.py           # (re)train the confidence classifier
```

The classifier artifact (`artifacts/gap_classifier.joblib`) is optional — without it the
app simply hides confidence badges; detection is unaffected.

## Using the app

1. Pick a demo in the sidebar (**Team Standup**, **Design Review**, **Clean Meeting**)
   or **Paste your own** (`Speaker: text` lines).
2. Adjust the **reply window** — how many turns of silence count as a gap.
3. Tabs:
   - **Highlighted transcript** — every gap turn color-coded in place
   - **Gap report** — evidence cards: type, severity, tone, cited message IDs,
     verbatim quotes, detector signals, mood delta, confidence badge
   - **Emotion** — per-participant sentiment trajectory with gap spans shaded
   - **Live simulation** — ▶ Play streams the chat turn by turn; alerts pop the
     moment each gap becomes detectable (⏸ Pause / ⏭ Step / ⟲ Reset)
   - **JSON** — full machine-readable report + download

---

## Evaluation

`python eval.py` runs the three labelled transcripts (gaps hand-injected into realistic
chats, plus **Clean Meeting** as the zero-gap control):

```
TOTAL   TP=6  FP=0  FN=0
Precision=1.000  Recall=1.000  F1=1.000
Evidence integrity: 6/6 gaps (cited message_ids exist + verbatim quotes)
```

- Matching: predicted vs labelled gap of the **same type** whose turn IDs intersect.
- The integrity check re-verifies, for every predicted gap, that all cited IDs exist and
  every quote is verbatim in the message it cites.

### Health score sanity check

| Scenario | Score | Grade |
|---|---|---|
| Clean Meeting | 88 | B |
| Team Standup | 53 | D |
| Design Review | 36 | F |

---

## ML confidence layer

No hand-labelled ML dataset — `train.py` manufactures one:

1. **Real positives** — detector windows on the labelled transcripts
2. **Synthetic positives** — each of the 4 gap patterns injected into the clean meeting
   at several positions; a window counts only if the *real detector* fires there
   (**18/18 injections detected**)
3. **Negatives** — sliding windows that overlap no gap

Features: dialogue-act sequence + TF-IDF text → LogisticRegression. Evaluated with
**5-fold StratifiedGroupKFold grouped by source conversation** (no window from the same
conversation leaks across folds):

```
dataset: 24 positive / 74 negative windows, 24 groups
CV (StratifiedGroupKFold, 5 folds): F1=0.823±0.216  P=0.900  R=0.800
```

The fitted model lands in `artifacts/gap_classifier.joblib`; gap cards show a
**confidence badge** (green ≥ 80%, amber ≥ 60%). A ready-to-run **Kaggle notebook**
(`kaggle_notebook.ipynb` + `gapsense_kaggle.zip`) reproduces the whole training run.

---

## Example output

```json
{
  "title": "Team Standup",
  "summary": {"total": 4, "by_type": {"unanswered_question": 1, "ignored_response": 2, "repeated_clarification": 1, "unresolved_topic": 0}},
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
      "evidence_quotes": ["Dave, good - do you know if the staging environment is back up?", "Hello? Is staging up? Anyone?"],
      "quotes": ["Dave, good - do you know if the staging environment is back up?", "Hello? Is staging up? Anyone?"],
      "severity": "high",
      "tone": "frustration",
      "reason": "Question(s) from Alice starting at turn 4 received no reply for 3 turn(s).",
      "signals": ["question addressed to Dave", "Dave never took the floor after being addressed", "participation shift: Turn Claiming"],
      "mood_delta": -0.224,
      "confidence": 0.734
    }
  ]
}
```

---

## Project layout

```
gapdetect/
  models.py        Utterance, Gap, Report, enums
  ingest.py        transcript parsing (text / JSON)
  acts.py          dialogue-act tagging (regex + lexical cues)
  tone.py          affect tone cues (frustration/urgency/uncertainty)
  textutil.py      tokenising, light stemming, overlap scoring
  evidence.py      verbatim-quote verification, ID integrity, severity
  turntaking.py    addressee inference + ParShift participation shifts
  sentiment.py     per-turn polarity + mood deltas
  scoring.py       communication health score (0–100)
  ml.py            confidence classifier wrapper
  pipeline.py      end-to-end analyse()
  detectors/       unanswered · ignored · clarification · unresolved
data/              demo transcripts + ground_truth/
app.py             Streamlit dashboard (5 tabs + live simulation)
eval.py            P/R/F1 + evidence-integrity harness
train.py           synthetic gap injection → classifier training
requirements.txt   runtime dependencies
kaggle_notebook.ipynb + gapsense_kaggle.zip   reproducible Kaggle run
PROJECT.md         full project write-up (source for slides)
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3 | NLP standard |
| Detection core | Rules (regex + lexical cues + word-overlap clustering) | Offline, deterministic, explainable, zero training data |
| Turn-taking | **ParShift** (MIT, `pip install parshift`) | Gibson participation-shift framework, peer-reviewed (SoftwareX 2023) |
| ML layer | scikit-learn: TF-IDF + LogisticRegression | Confidence % from synthetic gap injection, grouped CV reported |
| Sentiment | TextBlob + tone-lexicon blend | Offline per-turn polarity for the emotion trajectory |
| UI | Streamlit + Plotly | Interactive dashboard in a single file |
| Eval | Custom P/R/F1 harness vs hand labels | Regression gate: must stay 1.000 |

---

## Acknowledgements

- **ParShift** — [bdfsaraiva/parshift](https://github.com/bdfsaraiva/parshift) (**MIT**),
  used for the participation-shift / turn-taking layer.
  Ferreira-Saraiva, B.D., Matos-Carvalho, J.P., Fachada, N. & Pita, M. (2023).
  *ParShift: A Python package to study order and differentiation in group
  conversations.* SoftwareX 24, 101554. https://doi.org/10.1016/j.softx.2023.101554
- **Gibson's (2003) participation-shift framework** — the theory behind the
  speaker / target / non-target turn codes.
- Inspirations (no code used): *groupchat-decoder* (dashboard feature set),
  *Agreement-and-Disagreement-Recognition* (agree/disagree signal idea),
  *chat-analytics* (chat export parsing approach).

## Extending

- **Add a detector**: drop a module in `gapdetect/detectors/` returning `list[Gap]`,
  wire it into `pipeline.py`.
- **New demo**: add `data/<name>.json` + `data/ground_truth/<name>.json`.
- **LLM assist**: `analyze()` output can be sent to a model for adjudication — the
  verbatim evidence check keeps any downstream summary grounded.
