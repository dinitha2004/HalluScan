# demo එකේ මුල සිට අග දක්වා සිදුවන දේ (මුල සිට පැහැදිලි කිරීම)

මෙය **එක් ප්‍රශ්නයක්** HalluScan හරහා ගමන් කරන ආකාරය — ඔබ එය type කරන මොහොතේ සිට වර්ණ ගැන්වූ ප්‍රතිඵලය දකින මොහොත දක්වා — අනුගමනය කරයි. පූර්ව දැනුමක් අවශ්‍ය නැත.

---

## යෙදුමේ කොටස් දෙක

- **Frontend** (ඔබ දකින වෙබ් පිටුව) — `frontend/` හි React යෙදුමක්. ඔබේ බ්‍රවුසරයේ ක්‍රියාත්මක වේ.
- **Backend** (මොළය) — `backend/app.py` හි Python සේවාදායකයක්. GPU සහිත යන්ත්‍රයක ක්‍රියාත්මක වේ (ඔබේ පරිගණකය, හෝ ngrok හරහා අන්තර්ජාලයෙන් බෙදාගත් Colab GPU එකක්).

මේ දෙක සරල වෙබ් ඉල්ලීම් මඟින් කතා කරයි. frontend එක කිසිදා AI මොඩලය ක්‍රියාත්මක නොකරයි — එය backend එකෙන් අසා පිළිතුර අඳියි පමණි.

---

## ඔබ type කරන්නටත් පෙර — "සූදානම්ද?" හද ගැස්ම

පිටුව load වූ සැණින්, frontend එක තත්පර 2.5කට වරක් backend එකෙන් **"ඔබ සූදානම්ද?"** කියා නිහඬව අසයි — `GET /status` ([App.jsx:62](../frontend/src/App.jsx#L62)) කැඳවීමෙන්. පිළිතුර මඟින් කියයි:

- මොඩලයක් load වී තිබේද? (navbar එකේ කොළ online තිතට බලපායි)
- කුමන මොඩල තිබේද (1B / 8B) සහ ක්‍රියාත්මක කුමක්ද (model dropdown පුරවයි)
- මොඩලයක් දැන් මාරු වෙමින් තිබේද? ("Switching…" පෙන්වයි)

මේ නිසා ඔබ කිසිවක් නොකර backend එකේ තත්ත්වය පිටුව දනී.

---

## එක් ප්‍රශ්නයක ගමන (පියවරෙන් පියවර)

### 1. ඔබ type කර send කරයි
ඔබේ පෙළ chat box එකට යයි ([ChatInterface.jsx](../frontend/src/components/ChatInterface.jsx)). send එබීමෙන් [App.jsx:89](../frontend/src/App.jsx#L89) හි `handleSendMessage` කැඳවේ.

### 2. frontend එක backend එකට යවයි
එය ඔබේ පණිවිඩය chat එකට එක් කර, "Generating + checking…" spinner එක පෙන්වා, මෙය යවයි:
```
POST  <backend-url>/infer     body: { question: "...", model: "1b" or "8b" }
```
([App.jsx:93](../frontend/src/App.jsx#L93)). frontend කරන එකම දේ එයයි — පිළිතුරට බලා සිටීම.

### 3. backend එක එය ලබා ගනී
[app.py:288](../backend/app.py#L288) හි `/infer` තෝරාගත් මොඩලය තෝරයි. එම මොඩලය තවම GPU එකේ නැතිනම්, මුලින් එය load කරයි (මතකය ඉතිරි කිරීමට අනෙක unload කරයි). ඉන්පසු ප්‍රශ්නය scoring pipeline එකට දෙයි:
```python
result = pipe.score_with_sentences(question, max_new_tokens, use_claim_filter)
```

### 4. මොඩලය පිළිතුරක් ලියයි
[pipeline.py:117](../hallking/pipeline.py#L117) `score_with_sentences` තුළ, මොඩලය සාමාන්‍ය පිළිතුරක් generate කරයි — **කෙටි වාක්‍යයකට එක් කරුණක්** ලෙස ලියා, එක් එක් කොටස වෙන වෙනම පරීක්ෂා කළ හැකි වන පරිදි (`engine.generate_demo`).

### 5. පිළිතුර වාක්‍යයෙන් වාක්‍යය පරීක්ෂා කරයි
මෙය හරයයි. `localize()` ([localize.py:81](../hallking/localize.py#L81)) කරන්නේ:
1. පිළිතුර වාක්‍ය වලට **බෙදීම**.
2. **පෙරීම**: claim detector එකක් (DeBERTa NLI) කුමන වාක්‍ය සැබෑ *කරුණු ප්‍රකාශ* ද යන්න තීරණය කරයි. ආචාර කිරීම්, filler, සහ ප්‍රශ්න මඟ හැරේ — ඒවාට ලකුණක් නොලැබේ.
3. **එක් එක් claim වාක්‍යය ලකුණු කිරීම** detector තුනෙන්:
   - **SEP** — එම වාක්‍යයේ මොඩලයේ අභ්‍යන්තර "මනස් ඡායාරූපය" කියවයි.
   - **HalluShift** — එම වාක්‍යය පුරා මොඩලයේ අභ්‍යන්තරය කොතරම් shifty වුණාද බලයි.
   - **TSV** — වාක්‍යය "සත්‍ය" ද "ගොතන ලද" ද කියා නැඹුරුව බලයි.
4. **Fuse**: fusion මොඩලය ([fusion.py](../hallking/fusion.py)) එම අගයන් තුන වාක්‍යයට එක් **වැරදි තොරතුරු සම්භාවිතාවක්** බවට මිශ්‍ර කරයි, සහ `risk.py` එය වර්ණ tier එකක් බවට හරවයි (කොළ / කහ / රතු).

### 6. මුළු පිළිතුරටම එක් මාතෘකා අගයක්
`_aggregate_from_sentences` ([pipeline.py:176](../hallking/pipeline.py#L176)) වාක්‍ය මාතෘකාවට ගුළි කරයි:
- **fused** = **දෙවන‑නරකම** claim වාක්‍යය (එක් වැරදි වාක්‍යයක් එය "Uncertain" කරයි, "Likely Hallucinated" නොකරයි).
- SEP / HalluShift / TSV තීරු = claim වාක්‍ය හරහා **සාමාන්‍යය**.
- **n_flagged / n_sentences** = කොපමණ වාක්‍ය ඉහළ අවදානම්ද.

### 7. backend එක පිළිතුරු දෙයි
එය ලකුණු log කර (debugging සඳහා) JSON එකක් ආපසු යවයි:
```json
{
  "answer": "…මොඩලයේ පිළිතුර…",
  "aggregate": { "fused": 0.07, "tier": "ok", "label": "Reliable",
                 "sep_entropy": …, "hallushift": …, "tsv_margin": …,
                 "n_flagged": 0, "n_sentences": 3 },
  "sentences": [ { "sentence": "...", "is_claim": true, "fused": …, "tier": "...", … }, … ],
  "model": "1b"
}
```

### 8. frontend එක ප්‍රතිඵලය අඳියි
[App.jsx:94](../frontend/src/App.jsx#L94) හි පිළිතුර ගබඩා කර පෙන්වයි:
- **පිළිතුර**, එක් එක් වාක්‍යය එහි අවදානම අනුව **highlight** කර (රතු = ඉහළ, කහ = මධ්‍යම, කිසිවක් නැත = ආරක්ෂිත/filler). වාක්‍යයක් මත hover කළ විට එහි SEP / HalluShift / TSV / FUSED අගයන් පෙන්වයි ([ChatInterface.jsx](../frontend/src/components/ChatInterface.jsx)).
- **badge එකක්** පිළිතුර යට: 🛡️ Reliable / ⚠️ Uncertain / 🚨 Likely Hallucinated.
- **දකුණු පැනලය** ([App.jsx:164](../frontend/src/App.jsx#L164)): විශාල fused %, detector තීරු තුන, සහ "N of M sentences flagged".
- **රේඛා ප්‍රස්තාරය** ([UncertaintyChart.jsx](../frontend/src/components/UncertaintyChart.jsx)): මෙම පිළිතුරේ අවදානම ඔබේ සංවාදය පුරා නව ලක්ෂ්‍යයක් ලෙස එක් කරයි.

spinner එක නිවේ. යෙදුම ඔබේ ඊළඟ ප්‍රශ්නයට සූදානම්.

---

## මොඩල මාරු කිරීම (විකල්ප)
ඔබ dropdown එකෙන් වෙනත් මොඩලයක් තෝරන්නේ නම්, frontend එක `POST /select-model` ([app.py:239](../backend/app.py#L239)) කැඳවයි. backend එක වත්මන් මොඩලය unload කර නව එක GPU එකට load කරයි; මේ අතර `/status` "loading" වාර්තා කරයි, එම නිසා UI එක සූදානම් වන තෙක් "Switching…" පෙන්වයි.

---

## මුළු ගමන එක් පේළියකින්

> **ඔබ type කරයි → frontend `/infer` POST කරයි → backend මොඩලය ක්‍රියාත්මක කරයි → පිළිතුර වාක්‍ය වලට බෙදයි → detector 3ක් එක් එක් claim වාක්‍යය ලකුණු කරයි → fusion ඒවා එක් අවදානමකට හරවයි → මාතෘකාව = දෙවන‑නරකම වාක්‍යය → JSON ආපසු → frontend වර්ණ, badge, තීරු සහ ප්‍රස්තාරය අඳියි.**

| අදියර | ඇත්තේ කොහේද |
|---|---|
| type / send / ප්‍රතිඵලය ඇඳීම | `frontend/src/App.jsx`, `components/ChatInterface.jsx` |
| අවදානම් ප්‍රස්තාරය | `frontend/src/components/UncertaintyChart.jsx` |
| `/infer`, `/status`, `/select-model` | `backend/app.py` |
| generate + scoring සම්බන්ධීකරණය | `hallking/pipeline.py` (`score_with_sentences`) |
| වාක්‍යයෙන් වාක්‍යය බෙදීම + ලකුණු | `hallking/localize.py` |
| detector 3 | `hallking/{sep,hallushift,tsv}_adapter.py` |
| එක් අවදානමකට මිශ්‍ර කිරීම | `hallking/fusion.py` |
| අවදානම් වර්ණ band | `hallking/risk.py` |
