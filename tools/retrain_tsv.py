"""Re-train the TSV head (steering vector + centroids) on the shared dataset, FULLY SUPERVISED.

This is the faithful TSV objective (Park et al.) using the labels we already have for every
example (the paper's strong "TSV-with-full-labels" setting): a trainable steering vector at
layer `str_layer` reshapes the latent space while class centroids are EMA-updated, optimised by
the cosine/vMF NLL loss. The frozen base LLM stays frozen; only the ~4K steering vector trains.

Input  : data/<dataset>_fusion.parquet  (needs columns: question, answer, hallucination)
Output : artifacts/tsv/best_checkpoint_retrained.pt   (same format the TSV adapter loads)

GPU job — run in se_probes_env:
    python tools/retrain_tsv.py --data data/triviaqa_fusion.parquet --epochs 40
"""
import argparse, os, sys
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from engine import HallKingEngine
from llm_layers import add_tsv_layers, get_layers
from tsv_train_utils import compute_ot_loss_cos, update_centroids_ema_hard

TSV_PROMPT = "Answer the question concisely. Q: {question} A:{answer}"


def build_prompts(tok, df):
    ids = []
    for r in df.itertuples():
        a = r.answer if str(r.answer).startswith(" ") else " " + str(r.answer)
        p = TSV_PROMPT.format(question=r.question, answer=a)
        ids.append(tok(p, return_tensors="pt").input_ids[0])
    return ids


def collate(batch_ids, pad_id, device):
    L = max(x.size(0) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :x.size(0)] = x
        mask[i, :x.size(0)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def reps_for(model, ids_list, pad_id, device, bs=16):
    """Last-layer, last-real-token reps for a list of sequences (no grad; for eval)."""
    out = []
    for s in range(0, len(ids_list), bs):
        ids, mask = collate(ids_list[s:s + bs], pad_id, device)
        h = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        lengths = mask.sum(1)
        out.append(torch.stack([h[i, lengths[i] - 1, :] for i in range(ids.size(0))]).float())
    return F.normalize(torch.cat(out), p=2, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)   # small batches fit 12 GB
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--cos_temp", type=float, default=0.1)
    ap.add_argument("--ema_decay", type=float, default=0.99)
    ap.add_argument("--str_layer", type=int, default=9)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "tsv", "best_checkpoint_retrained.pt"))
    args = ap.parse_args()
    a = SimpleNamespace(cos_temp=args.cos_temp, ema_decay=args.ema_decay, str_layer=args.str_layer,
                        component="res", model_name="llama3.1-8B", lam=args.lam)

    df = pd.read_parquet(args.data)
    # TSV class convention: class 1 = TRUTHFUL (centroids row 1). hallucination=1 -> truthful=0.
    df = df.assign(truthful=(1 - df["hallucination"]).astype(int))
    tr, te = train_test_split(df, test_size=0.25, stratify=df["truthful"], random_state=42)
    print(f"[retrain_tsv] train={len(tr)} test={len(te)}")

    eng = HallKingEngine(model_name=args.base_model, fp16_nonquant=True).load()
    model, tok, device = eng.model, eng.tokenizer, eng.model.device
    pad_id = tok.pad_token_id
    hidden = model.config.hidden_size
    num_layers = len(get_layers(model))

    # float32 steering params so GradScaler can unscale grads (TSVLayer casts to fp16 internally).
    tsv = nn.ParameterList([nn.Parameter(torch.zeros(hidden), requires_grad=True) for _ in range(num_layers)])
    tsv.to(device)
    add_tsv_layers(model, tsv, [args.lam], a)
    optimizer = torch.optim.AdamW(list(tsv.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    centroids = F.normalize(torch.randn(2, hidden).half().to(device), p=2, dim=1)

    tr_ids = build_prompts(tok, tr); tr_y = torch.tensor(tr["truthful"].values, dtype=torch.long)
    te_ids = build_prompts(tok, te); te_y = te["truthful"].values

    best_auroc, best = -1.0, None
    for ep in range(args.epochs):
        model.eval()
        perm = torch.randperm(len(tr_ids))
        running = 0.0
        for s in range(0, len(tr_ids), args.batch_size):
            idx = perm[s:s + args.batch_size]
            ids, mask = collate([tr_ids[i] for i in idx], pad_id, device)
            yb = tr_y[idx].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                h = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                lengths = mask.sum(1)
                rep = torch.stack([h[i, lengths[i] - 1, :] for i in range(ids.size(0))])
                yoh = F.one_hot(yb, num_classes=2)
                loss, _ = compute_ot_loss_cos(rep, centroids, yoh, ids.size(0), a)
                with torch.no_grad():
                    centroids = update_centroids_ema_hard(centroids, rep, yoh, a)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            running += loss.item() * ids.size(0)
        # eval AUROC (P(truthful)); detection target = hallucinated, so use 1 - P(truthful)
        rep_te = reps_for(model, te_ids, pad_id, device)
        cen = F.normalize(centroids.float(), p=2, dim=-1)
        p_truth = torch.softmax(torch.matmul(rep_te, cen.T) / args.cos_temp, dim=-1)[:, 1].cpu().numpy()
        auroc = roc_auc_score(1 - te_y, 1 - p_truth)  # 1=hallucinated
        print(f"epoch {ep+1}/{args.epochs}  loss={running/len(tr_ids):.4f}  test AUROC(halluc)={auroc:.4f}", flush=True)
        if auroc > best_auroc:
            best_auroc = auroc
            best = {"tsv": [t.detach().float().cpu() for t in tsv],
                    "centroids": centroids.detach().float().cpu(),
                    "str_layer": args.str_layer, "lam": args.lam, "component": "res",
                    "cos_temp": args.cos_temp, "model_name": "llama3.1-8B",
                    "hidden_size": hidden, "best_test_auroc": float(auroc), "best_epoch": ep}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(best, args.out)
    print(f"\n[retrain_tsv] saved {args.out}  best test AUROC(halluc)={best_auroc:.4f}")


if __name__ == "__main__":
    main()
