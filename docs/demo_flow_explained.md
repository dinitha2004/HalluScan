# What happens in the demo, start to end (beginner guide)

This follows **one question** through HalluScan — from the moment you type it to the moment you see the coloured result. No prior knowledge needed.

---

## The two halves of the app

- **Frontend** (the web page you see) — a React app in `frontend/`. Runs in your browser.
- **Backend** (the brain) — a Python server in `backend/app.py`. Runs on a machine with a GPU (your PC, or a Colab GPU shared over the internet via ngrok).

They talk over the internet using simple web requests. The frontend never runs the AI model itself — it just asks the backend and draws the answer.

---

## Before you even type — the "is it ready?" heartbeat

As soon as the page loads, the frontend quietly asks the backend **"are you ready?"** every 2.5 seconds by calling `GET /status` ([App.jsx:62](../frontend/src/App.jsx#L62)). The reply tells it:

- is a model loaded? (drives the green online dot in the navbar)
- which models are available (1B / 8B) and which one is active (fills the model dropdown)
- is a model currently switching? (shows "Switching…")

This is why the page knows the backend's state without you doing anything.

---

## The journey of one question (step by step)

### 1. You type and press send
Your text goes into the chat box ([ChatInterface.jsx](../frontend/src/components/ChatInterface.jsx)). Pressing send calls `handleSendMessage` in [App.jsx:89](../frontend/src/App.jsx#L89).

### 2. The frontend sends it to the backend
It adds your message to the chat, shows the "Generating + checking…" spinner, and sends:
```
POST  <backend-url>/infer     body: { question: "...", model: "1b" or "8b" }
```
([App.jsx:93](../frontend/src/App.jsx#L93)). That's the only thing the frontend does — wait for the reply.

### 3. The backend receives it
`/infer` in [app.py:288](../backend/app.py#L288) picks the chosen model. If that model isn't in the GPU yet, it loads it first (and unloads the other to save memory). Then it hands the question to the scoring pipeline:
```python
result = pipe.score_with_sentences(question, max_new_tokens, use_claim_filter)
```

### 4. The model writes an answer
Inside [pipeline.py:117](../hallking/pipeline.py#L117) `score_with_sentences`, the model generates a normal answer — written as **one fact per short sentence** so each piece can be checked separately (`engine.generate_demo`).

### 5. The answer is checked sentence by sentence
This is the core. `localize()` ([localize.py:81](../hallking/localize.py#L81)) does:
1. **Split** the answer into sentences.
2. **Filter**: a claim detector (DeBERTa NLI) decides which sentences are real *factual claims*. Greetings, filler, and questions are skipped — they never get a score.
3. **Score each claim sentence** with the three detectors:
   - **SEP** — reads the model's internal "mind‑snapshot" at that sentence.
   - **HalluShift** — looks at how shifty the model's internals were over that sentence.
   - **TSV** — checks whether the sentence leans "truthful" or "made‑up".
4. **Fuse**: the fusion model ([fusion.py](../hallking/fusion.py)) blends those three numbers into one **hallucination probability** for the sentence, and `risk.py` turns it into a colour tier (green / yellow / red).

### 6. One headline number for the whole answer
`_aggregate_from_sentences` ([pipeline.py:176](../hallking/pipeline.py#L176)) rolls the sentences up into the headline:
- **fused** = the **2nd‑worst** claim sentence (so one stray bad sentence makes it "Uncertain", not "Likely Hallucinated").
- the SEP / HalluShift / TSV bars = the **average** across claim sentences.
- **n_flagged / n_sentences** = how many sentences were high‑risk.

### 7. The backend replies
It logs the scores (for debugging) and sends back JSON:
```json
{
  "answer": "…the model's answer…",
  "aggregate": { "fused": 0.07, "tier": "ok", "label": "Reliable",
                 "sep_entropy": …, "hallushift": …, "tsv_margin": …,
                 "n_flagged": 0, "n_sentences": 3 },
  "sentences": [ { "sentence": "...", "is_claim": true, "fused": …, "tier": "...",
                   "sep_entropy": …, "hallushift": …, "tsv_margin": … }, … ],
  "model": "1b"
}
```

### 8. The frontend draws the result
Back in [App.jsx:94](../frontend/src/App.jsx#L94) the reply is stored and shown:
- **The answer**, with each sentence **highlighted** by its risk (red = high, yellow = medium, none = safe/filler). Hovering a sentence shows its SEP / HalluShift / TSV / FUSED numbers ([ChatInterface.jsx](../frontend/src/components/ChatInterface.jsx)).
- **A badge** under the answer: 🛡️ Reliable / ⚠️ Uncertain / 🚨 Likely Hallucinated.
- **The right panel** ([App.jsx:164](../frontend/src/App.jsx#L164)): the big fused % , three detector bars, and "N of M sentences flagged".
- **The line chart** ([UncertaintyChart.jsx](../frontend/src/components/UncertaintyChart.jsx)): the risk of this answer added as a new point across your conversation.

The spinner turns off. The app is ready for your next question.

---

## Switching models (optional)
If you pick a different model in the dropdown, the frontend calls `POST /select-model` ([app.py:239](../backend/app.py#L239)). The backend unloads the current model and loads the new one in the GPU; meanwhile `/status` reports "loading", so the UI shows "Switching…" until it's ready.

---

## The whole trip in one line

> **You type → frontend POSTs `/infer` → backend runs the model → splits the answer into sentences → 3 detectors score each claim sentence → fusion turns them into one risk → headline = 2nd‑worst sentence → JSON back → frontend paints the colours, badge, bars and chart.**

| Stage | Where it lives |
|---|---|
| Type / send / draw result | `frontend/src/App.jsx`, `components/ChatInterface.jsx` |
| Risk chart | `frontend/src/components/UncertaintyChart.jsx` |
| `/infer`, `/status`, `/select-model` | `backend/app.py` |
| Generate + orchestrate scoring | `hallking/pipeline.py` (`score_with_sentences`) |
| Per‑sentence split + score | `hallking/localize.py` |
| The 3 detectors | `hallking/{sep,hallushift,tsv}_adapter.py` |
| Combine into one risk | `hallking/fusion.py` |
| Risk colour bands | `hallking/risk.py` |
