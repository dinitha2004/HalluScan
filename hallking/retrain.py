"""Unified re-training of the three lightweight detector HEADS on ONE shared dataset + config
(fp16 everywhere), with the base LLM frozen. This makes all three in-distribution together so
the fusion is legitimate. Heavy parts (generation, TSV training) are GPU; SEP/HalluShift head
fits are CPU on cached features.

Functions:
  gen_and_cache(...)         -> (df[q,answer,hs_feat_*,bleurt,hallucination], sep_feats[N,135168])  [GPU, Instruct fp16]
  retrain_sep(feats, y)      -> SEP probe pkl (list of one dict)                                     [CPU]
  retrain_hallushift(df, y)  -> (state_dict, StandardScaler)                                         [CPU]
  train_tsv(df, ...)         -> (checkpoint_dict, per_row_tsv_margin[N])                             [GPU, base fp16]
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from engine import HallKingEngine
from sep_adapter import SEPAdapter
from hallushift_adapter import HalluShiftAdapter
from run_dataset import load_qa, INSTRUCT_MODEL, BASE_MODEL
from gt_bleurt import bleurt_labels, DEFAULT_BLEURT_PY
from llm_layers import add_tsv_layers, get_layers
from tsv_train_utils import compute_ot_loss_cos, update_centroids_ema_hard

TSV_PROMPT = "Answer the question concisely. Q: {question} A:{answer}"


# ---------------------------------------------------------------- GPU: generate + cache features
def gen_and_cache(dataset_name="triviaqa", n=1200, offset=1000, max_new_tokens=64,
                  instruct_model=INSTRUCT_MODEL, bleurt_python=DEFAULT_BLEURT_PY,
                  regime="short", label_method="bleurt", drop_refusals=False, verbose=True):
    """Generate answers + cache RAW per-head features (SEP 135168-d, HalluShift 71-d) + labels.

    regime       : "short"    -> base QA prompt, ~2-word answers (Option A; keeps the eval AUROCs valid)
                   "sentence" -> Instruct chat template, ONE factual sentence (Option B; train == demo)
    label_method : "bleurt"    -> BLEURT(answer, ref) > 0.5  (subprocess)
                   "reference" -> any normalized alias is a substring of the answer (cheap, 1-pass)
    drop_refusals: skip "I don't know"/refusal answers (not claims, not hallucinations) before labeling.
    Returns (df[question, answer, hs_feat_00..70, bleurt, hallucination], sep_feats[N, 135168]).
    """
    questions, refs = load_qa(dataset_name, n=n, offset=offset)
    eng = HallKingEngine(model_name=instruct_model, fp16_nonquant=True).load()  # fp16 (matches SEP)
    sep = SEPAdapter(eng)                       # features() needs no probe
    hs = HalluShiftAdapter(eng); hs.num_layers = eng.num_layers  # features() needs no trained model
    from tqdm.auto import tqdm   # live progress bar (count + rate + ETA) so the cell visibly advances
    from claim_filter import is_refusal
    kept_q, kept_refs, answers, hs_feats, sep_feats = [], [], [], [], []
    n_refused = 0
    for q, rf in tqdm(list(zip(questions, refs)), desc=f"generate+cache ({dataset_name}, {regime})", unit="q"):
        gen = (eng.generate_sentence(q, max_new_tokens=max_new_tokens) if regime == "sentence"
               else eng.generate(q, max_new_tokens=max_new_tokens))
        ans = gen["answer_clean"]
        if drop_refusals and is_refusal(ans):
            n_refused += 1
            continue
        kept_q.append(q); kept_refs.append(rf); answers.append(ans)
        hs_feats.append(hs.features(gen))
        sep_feats.append(sep.features(gen).astype(np.float16))
    eng.unload(); del eng, sep, hs
    if drop_refusals and verbose:
        print(f"  dropped {n_refused} refusal answers (left the model's 'I don't know' alone)", flush=True)
    if not answers:
        raise RuntimeError("no answers left after generation / refusal-filtering")

    if label_method == "reference":
        from claim_label import label_by_reference_match
        labels = label_by_reference_match(answers, kept_refs)
        bleurt = np.full(len(answers), np.nan)
    else:
        if verbose:
            print(f"  scoring BLEURT ground-truth labels for {len(answers)} answers "
                  f"(bleurt_env subprocess; ~1-2 min, no per-item bar) ...", flush=True)
        labels, bleurt = bleurt_labels(answers, kept_refs, threshold=0.5, bleurt_python=bleurt_python)

    df = pd.DataFrame({"question": kept_q, "answer": answers})
    hs_arr = np.stack(hs_feats)
    for j in range(hs_arr.shape[1]):
        df[f"hs_feat_{j:02d}"] = hs_arr[:, j]
    df["bleurt"] = bleurt
    df["hallucination"] = np.asarray(labels).astype(int)
    if verbose:
        y = df["hallucination"].to_numpy()
        print(f"  labels ({label_method}): truthful={int((y==0).sum())} halluc={int(y.sum())} "
              f"({y.mean()*100:.1f}%) over {len(df)} answers", flush=True)
    return df, np.stack(sep_feats)


# ---------------------------------------------------------------- CPU: SEP probe re-fit
def retrain_sep(sep_feats, y, name="retrained"):
    """LogReg probes on the flattened all-layer SLT features. Returns SEP-format probe list."""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(sep_feats, dtype=np.float32)
    y = np.asarray(y).astype(int)
    halluc = LogisticRegression(max_iter=2000, C=0.1).fit(X, y)            # P(hallucinated)
    truth = LogisticRegression(max_iter=2000, C=0.1).fit(X, 1 - y)         # P(truthful)
    return [{"name": name, "s_bmodel": halluc, "s_amodel": truth,
             "sep_layer_range": (0, 1000), "ap_layer_range": (0, 1000)}]


# ---------------------------------------------------------------- CPU: HalluShift MLP re-fit
def retrain_hallushift(df, y, num_layers=32, epochs=300, lr=1e-3, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from classifier import CombinedNN, AccuracyImprovementLossBinary
    cols = [f"hs_feat_{j:02d}" for j in range(71)]
    X = df[cols].to_numpy(dtype=np.float64); y = np.asarray(y).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.tensor(scaler.transform(Xtr), dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev).unsqueeze(1)
    Xte_t = torch.tensor(scaler.transform(Xte), dtype=torch.float32, device=dev)
    torch.manual_seed(seed)  # seed the MLP init so OOF features (and the fusion) are reproducible
    model = CombinedNN(num_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = AccuracyImprovementLossBinary().to(dev)
    cw = torch.tensor([len(ytr) / max((ytr == 0).sum(), 1), len(ytr) / max((ytr == 1).sum(), 1)],
                      dtype=torch.float32, device=dev)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Xtr_t)
        w = ytr_t * cw[1] + (1 - ytr_t) * cw[0]
        loss = (crit(out, ytr_t) * w).mean()
        loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xte_t)).cpu().numpy().ravel()
            auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else 0.5
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"  [HalluShift retrain] best val AUROC={best_auc:.4f}")
    return best_state, scaler


# ---------------------------------------------------------------- GPU: TSV head re-train
def _collate(batch_ids, pad_id, device):
    L = max(x.size(0) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :x.size(0)] = x; mask[i, :x.size(0)] = 1
    return ids.to(device), mask.to(device)


def _last_tok(h, mask):
    lengths = mask.sum(1)
    return torch.stack([h[i, lengths[i] - 1, :] for i in range(h.size(0))])


def train_tsv(df, base_model=BASE_MODEL, epochs=40, batch_size=16, lr=5e-3, lam=5.0,
              cos_temp=0.1, ema_decay=0.99, str_layer=9, seed=42, tr_idx=None, te_idx=None, verbose=True):
    """Supervised TSV: train steering vector + EMA centroids. Returns (checkpoint, margins[N])
    aligned to df rows (margin = cos_halluc - cos_truth, higher = hallucinated).

    Pass `tr_idx`/`te_idx` to reuse a SHARED held-out split (so SEP/HalluShift/TSV report per-head
    AUROC on the same test questions); otherwise an internal stratified 75/25 split is used."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    a = SimpleNamespace(cos_temp=cos_temp, ema_decay=ema_decay, str_layer=str_layer,
                        component="res", model_name="llama3.1-8B", lam=lam)
    df = df.reset_index(drop=True)
    truthful = (1 - df["hallucination"]).astype(int).to_numpy()  # class 1 = truthful
    if tr_idx is None or te_idx is None:
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.25,
                                          stratify=truthful, random_state=seed)
    else:
        tr_idx, te_idx = np.asarray(tr_idx), np.asarray(te_idx)

    eng = HallKingEngine(model_name=base_model, fp16_nonquant=True).load()
    model, tok, device = eng.model, eng.tokenizer, eng.model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    hidden = model.config.hidden_size
    num_layers = len(get_layers(model))

    def prompts(idx):
        out = []
        for i in idx:
            ans = df.answer[i] if str(df.answer[i]).startswith(" ") else " " + str(df.answer[i])
            out.append(tok(TSV_PROMPT.format(question=df.question[i], answer=ans),
                           return_tensors="pt").input_ids[0])
        return out
    all_ids = prompts(range(len(df)))

    # Keep steering params in float32 so torch.cuda.amp.GradScaler can unscale their grads.
    # TSVLayer casts to .half() internally, so the model's forward stays fp16.
    tsv = nn.ParameterList([nn.Parameter(torch.zeros(hidden), requires_grad=True) for _ in range(num_layers)])
    tsv.to(device)
    add_tsv_layers(model, tsv, [lam], a)
    opt = torch.optim.AdamW(list(tsv.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    centroids = F.normalize(torch.randn(2, hidden).half().to(device), p=2, dim=1)
    y_tr = torch.tensor(truthful[tr_idx], dtype=torch.long)

    @torch.no_grad()
    def margins(idx):
        out = []
        for s in range(0, len(idx), batch_size):
            ids, mask = _collate([all_ids[i] for i in idx[s:s + batch_size]], pad_id, device)
            h = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            rep = F.normalize(_last_tok(h, mask).float(), p=2, dim=-1)
            cen = F.normalize(centroids.float(), p=2, dim=-1)
            cos = torch.matmul(rep, cen.T)
            out.append((cos[:, 0] - cos[:, 1]).cpu().numpy())   # cos_halluc - cos_truth
        return np.concatenate(out)

    model.eval()
    best_auc, best = -1, None
    for ep in range(epochs):
        perm = np.random.RandomState(seed + ep).permutation(len(tr_idx))
        running = 0.0
        for s in range(0, len(perm), batch_size):
            bidx = [tr_idx[i] for i in perm[s:s + batch_size]]
            ids, mask = _collate([all_ids[i] for i in bidx], pad_id, device)
            yb = y_tr[perm[s:s + batch_size]].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                h = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                rep = _last_tok(h, mask)
                yoh = F.one_hot(yb, num_classes=2)
                loss, _ = compute_ot_loss_cos(rep, centroids, yoh, ids.size(0), a)
                with torch.no_grad():
                    centroids = update_centroids_ema_hard(centroids, rep, yoh, a)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            running += loss.item() * ids.size(0)
        m_te = margins(list(te_idx))
        auc = roc_auc_score(df["hallucination"].to_numpy()[te_idx], m_te) if len(set(truthful[te_idx])) > 1 else 0.5
        if verbose:
            print(f"  [TSV ep {ep+1}/{epochs}] loss={running/len(tr_idx):.4f} test AUROC={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best = {"tsv": [t.detach().float().cpu() for t in tsv],
                    "centroids": centroids.detach().float().cpu(), "str_layer": str_layer,
                    "lam": lam, "component": "res", "cos_temp": cos_temp,
                    "model_name": "llama3.1-8B", "hidden_size": hidden,
                    "best_test_auroc": float(best_auc), "best_epoch": ep}
    # Restore the BEST-epoch checkpoint into the live params before final scoring. The optimizer keeps
    # updating tsv/centroids every epoch, so without this all_margins would reflect the LAST epoch
    # (often overfit) while `best` holds the early-stopped weights — the fusion/scored parquet would then
    # disagree with the saved checkpoint. copy_ is in-place so TSVLayer's reference into `tsv` stays valid.
    if best is not None:
        with torch.no_grad():
            for p, bw in zip(tsv, best["tsv"]):
                p.copy_(bw.to(p.device, p.dtype))
        centroids = best["centroids"].half().to(device)
    all_margins = margins(list(range(len(df))))
    if verbose:
        chk_auc = (roc_auc_score(df["hallucination"].to_numpy()[te_idx], all_margins[te_idx])
                   if len(set(truthful[te_idx])) > 1 else 0.5)
        print(f"  [TSV retrain] best-checkpoint test AUROC={chk_auc:.4f} (restored from epoch {best['best_epoch']+1})",
              flush=True)
    eng.unload()
    print(f"  [TSV retrain] best test AUROC={best_auc:.4f}")
    return best, all_margins
