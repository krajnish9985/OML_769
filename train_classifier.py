"""Module C: fine-tune MuRIL (or IndicBERT v2) to tell clean prefixes from corrupted ones.

    export OMP_NUM_THREADS=2
    python train_classifier.py --gpu 0 --limit 20000          # quick test (a few minutes)
    python train_classifier.py --gpu 0                        # full: all languages, 1 epoch -> data/classifier/

One model for all three languages. Input: data/corrupt/{train,dev}.jsonl from build_corrupt.py.
Plain PyTorch loop on ONE GPU (the HF Trainer would spread over every visible GPU).
Long prefixes are cut from the LEFT, so the last word (the one being judged) is always kept.
Prints dev accuracy per language and corruption type, and the false-veto rate: the share of CLEAN
dev prefixes with P(ok) below tau, i.e. how often module C would reject correct text at that tau.
"""
import argparse
import json
import random
import time
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import DATA, SEED

OUT = DATA / "classifier"
TAUS = (0.05, 0.1, 0.2, 0.3, 0.5)


def read(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def batches(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


@torch.no_grad()
def p_ok(model, tok, texts, device, max_len, batch=256):
    """P(label 1 = grammatical) for each text."""
    out = []
    for b in batches(texts, batch):
        enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits
        out += torch.softmax(logits.float(), -1)[:, 1].tolist()
    return out


def evaluate(model, tok, dev, device, max_len):
    model.eval()
    probs = p_ok(model, tok, [r["text"] for r in dev], device, max_len)
    model.train()
    acc = defaultdict(lambda: [0, 0])
    for r, p in zip(dev, probs):
        ok = (p >= 0.5) == (r["label"] == 1)
        for key in ("all", r["lang"], f"{r['type']}"):
            acc[key][0] += ok
            acc[key][1] += 1
    clean = [p for r, p in zip(dev, probs) if r["label"] == 1]
    bad = [p for r, p in zip(dev, probs) if r["label"] == 0]
    line = "  ".join(f"{k} {c / n:.3f}" for k, (c, n) in acc.items())
    fv = "  ".join(f"tau {t}: clean vetoed {100 * sum(p < t for p in clean) / len(clean):.1f}% / "
                   f"bad caught {100 * sum(p < t for p in bad) / len(bad):.1f}%" for t in TAUS)
    return acc["all"][0] / acc["all"][1], line, fv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased", help="or ai4bharat/IndicBERTv2-MLM-only")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, help="use only the first N training rows (quick test)")
    ap.add_argument("--eval-every", type=int, default=2000, help="steps")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(args.threads)
    device = f"cuda:{args.gpu}"

    train = read(DATA / "corrupt" / "train.jsonl", args.limit)
    dev = read(DATA / "corrupt" / "dev.jsonl")
    print(f"train {len(train)} rows, dev {len(dev)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")

    rng = random.Random(SEED)
    step, t0, best = 0, time.time(), -1.0
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for b in batches(train, args.batch):
            enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            labels = torch.tensor([r["label"] for r in b], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**enc, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps}  loss {loss.item():.4f}  {step / (time.time() - t0):.1f} it/s", flush=True)
            if step % args.eval_every == 0 or step == steps:
                acc, line, fv = evaluate(model, tok, dev, device, args.max_len)
                print(f"[dev] step {step}  acc  {line}\n[dev] {fv}", flush=True)
                if acc > best:
                    best = acc
                    model.save_pretrained(args.out)
                    tok.save_pretrained(args.out)
                    print(f"  saved {args.out} (best dev acc {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
