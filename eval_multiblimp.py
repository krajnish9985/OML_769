"""MultiBLiMP minimal pairs: do the models and modules catch a wrong verb form?

    export OMP_NUM_THREADS=2
    python eval_multiblimp.py --gpu 0 --limit 50          # quick test
    python eval_multiblimp.py --gpu 0                     # all pairs, all languages, modules S2 S3 S4

Each pair is a prefix + the correct verb vs the same prefix + a wrong verb (agreement in number,
gender or person). For every pair this measures:
  1. LM preference   target and draft: is log P(correct verb | prefix) > log P(wrong verb | prefix)?
  2. Module L        L(correct) vs L(wrong) for module A (S2), B (S3), C (S4), and how often each
                     verb gets L < tau (wrong caught / correct wrongly vetoed)
  3. SD acceptance   if the draft proposes the verb, the chance that verification accepts all of
                     its tokens: prod min(1, p/q) for S1, times [L >= tau] with a module (hard).
                     "bad accepted" = this chance for the wrong verb: the bad-token acceptance rate.
                     It is computed exactly from p and q, no sampling.
Model pair from config.py (`export PAIR=qwen` for Qwen; prefixes are scored as plain text, no chat template).
Pairs whose prefix is empty (verb is the first word) are skipped: the modules need context.
Writes per-pair numbers to results/multiblimp/{lang}.jsonl (results/multiblimp_T1.0/ for --temperature 1.0) (results_qwen/ for Qwen) and prints one summary block per language.
"""
import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA, DRAFT, DTYPE, LANGS, PAIR, RESULTS, TARGET


SENT_KEYS = [("sen", "wrong_sen"), ("sentence_good", "sentence_bad"), ("good_sen", "bad_sen"), ("good", "bad")]


def split_pair(good_sen, bad_sen):
    """Prefix and the two differing words, from the first word where the two sentences differ."""
    g, b = good_sen.split(), bad_sen.split()
    for i, (x, y) in enumerate(zip(g, b)):
        if x != y:
            return " ".join(g[:i]), x, y
    return "", "", ""


def load_pairs(lang, limit):
    with open(DATA / "multiblimp" / f"{lang}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    pairs = []
    for r in rows:
        prefix, good, bad = ((r.get(k) or "").strip() for k in ("prefix", "verb", "wrong_verb"))
        if not (good and bad):  # no verb fields: take the first differing word of the two sentences
            ka, kb = next(((a, b) for a, b in SENT_KEYS if r.get(a) and r.get(b)), (None, None))
            if ka:
                prefix, good, bad = split_pair(r[ka], r[kb])
        if prefix and good and bad and good != bad:
            phen = r.get("phenomenon") or r.get("phen") or r.get("feature") or "all"
            pairs.append({"prefix": prefix, "good": good, "bad": bad, "phen": str(phen)})
    if not pairs:
        raise SystemExit(f"{lang}: no usable pairs. Fields in the file: {sorted(rows[0]) if rows else 'file is empty'}\n"
                         f"First row: {json.dumps(rows[0], ensure_ascii=False)[:600] if rows else ''}")
    return pairs[:limit]


def make_scorer(system, tok, lang, args):
    if system == "S2":
        from modules.rules import RuleScorer
        return RuleScorer(tok, lang, groups="all")
    if system == "S3":
        from modules.stat import StatScorer
        return StatScorer(tok, lang)
    from modules.neural import NeuralScorer
    return NeuralScorer(tok, lang, model_path=args.clf)


@torch.no_grad()
def verb_probs(model, prefix_ids, verb_ids, temperature, vocab):
    """(log P(verb | prefix) at T=1, [prob of each verb token at the sampling temperature])."""
    x = torch.tensor([prefix_ids + verb_ids], device=model.device)
    logits = model(x).logits[0, len(prefix_ids) - 1:-1, :vocab].float()  # positions that predict the verb tokens
    # (:vocab drops embedding padding, e.g. Qwen2.5 7B has 128 more rows than 0.5B, as in sd.py)
    idx = torch.tensor(verb_ids, device=logits.device)[:, None]
    logp = torch.log_softmax(logits, -1).gather(1, idx).sum().item()
    probs = torch.softmax(logits / temperature, -1).gather(1, idx)[:, 0].tolist()
    return logp, probs


def sd_accept(p, q, ok=True):
    """Chance that verification accepts every drafted token (left to right)."""
    a = 1.0
    for pi, qi in zip(p, q):
        a *= min(1.0, pi / qi) if qi > 0 else 0.0
    return a if ok else 0.0


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--systems", nargs="*", default=["S2", "S3", "S4"], choices=["S2", "S3", "S4"])
    ap.add_argument("--tau-s2", type=float, default=0.5)
    ap.add_argument("--tau-s3", type=float, default=0.05)
    ap.add_argument("--tau-s4", type=float, default=0.3)
    ap.add_argument("--clf", help="S4 classifier folder (default data/classifier)")
    ap.add_argument("--temperature", type=float, default=0.7, help="same as in generation")
    ap.add_argument("--limit", type=int, help="first N pairs per language")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    taus = {"S2": args.tau_s2, "S3": args.tau_s3, "S4": args.tau_s4}

    pairs_by_lang = {lang: load_pairs(lang, args.limit) for lang in args.lang}  # fail fast, before loading models
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(args.threads)
    print(f"pair {PAIR}: draft {DRAFT}, target {TARGET}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    load_kw = dict(device_map={"": device}, dtype=getattr(torch, DTYPE))
    target = AutoModelForCausalLM.from_pretrained(TARGET, **load_kw).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, **load_kw).eval()
    vocab = min(m.get_output_embeddings().weight.shape[0] for m in (draft, target))
    end_id = tok(" .", add_special_tokens=False).input_ids[:1]  # lets the modules see the verb as a finished word

    out_dir = RESULTS / ("multiblimp" if args.temperature == 0.7 else f"multiblimp_T{args.temperature}")  # keep T=0.7 files
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.lang:
        pairs = pairs_by_lang[lang]
        scorers = {}
        for s in args.systems:
            try:
                scorers[s] = make_scorer(s, tok, lang, args)
            except Exception as e:  # e.g. no n-gram model or classifier yet
                print(f"{lang}: skipping {s} ({type(e).__name__}: {e})", flush=True)
        res = []
        for n, pr in enumerate(pairs):
            prefix_ids = tok(pr["prefix"]).input_ids
            rec = {**pr}
            for kind in ("good", "bad"):
                verb_ids = tok(" " + pr[kind], add_special_tokens=False).input_ids
                lp_t, p = verb_probs(target, prefix_ids, verb_ids, args.temperature, vocab)
                lp_d, q = verb_probs(draft, prefix_ids, verb_ids, args.temperature, vocab)
                rec[f"logp_target_{kind}"], rec[f"logp_draft_{kind}"] = lp_t, lp_d
                rec[f"acc_S1_{kind}"] = sd_accept(p, q)
                for s, sc in scorers.items():
                    sc.reset([])
                    L = min(sc.score(prefix_ids, verb_ids + end_id)[: len(verb_ids)])
                    rec[f"L_{s}_{kind}"] = L
                    rec[f"acc_{s}_{kind}"] = sd_accept(p, q, ok=L >= taus[s])
            res.append(rec)
            if n % 200 == 0:
                print(f"{lang} {n}/{len(pairs)}", flush=True)

        with open(out_dir / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n=== {lang}: {len(res)} pairs ===")
        for m in ("target", "draft"):
            print(f"  {m:6s} prefers correct verb: {pct([r[f'logp_{m}_good'] > r[f'logp_{m}_bad'] for r in res]):.1f}%")
        by_phen = defaultdict(list)
        for r in res:
            by_phen[r["phen"]].append(r["logp_target_good"] > r["logp_target_bad"])
        if len(by_phen) > 1:
            print("  target by phenomenon: " + "  ".join(f"{k} {pct(v):.1f}% (n={len(v)})" for k, v in sorted(by_phen.items())))
        print(f"  {'system':6s} {'bad accepted':>12s} {'good accepted':>13s}   L: prefers good / tie / prefers bad   "
              f"bad caught (L<tau)   good vetoed")
        print(f"  {'S1':6s} {pct([r['acc_S1_bad'] for r in res]):11.1f}% {pct([r['acc_S1_good'] for r in res]):12.1f}%")
        for s in scorers:
            g, b = [r[f"L_{s}_good"] for r in res], [r[f"L_{s}_bad"] for r in res]
            print(f"  {s:6s} {pct([r[f'acc_{s}_bad'] for r in res]):11.1f}% {pct([r[f'acc_{s}_good'] for r in res]):12.1f}%"
                  f"   {pct([x > y for x, y in zip(g, b)]):5.1f} / {pct([x == y for x, y in zip(g, b)]):5.1f} / "
                  f"{pct([x < y for x, y in zip(g, b)]):5.1f}%"
                  f"         {pct([y < taus[s] for y in b]):5.1f}%            {pct([x < taus[s] for x in g]):5.1f}%"
                  f"   (tau {taus[s]})")
        print(f"  wrote {out_dir / f'{lang}.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
