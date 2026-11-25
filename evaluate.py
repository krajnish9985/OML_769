"""Score generation outputs written by run_generation.py.

    python evaluate.py results/S1                              # cheap metrics only (CPU, seconds)
    python evaluate.py results/S1 results/S2 --stanza --gpu 0  # + agreement errors (Stanza parse)
    python evaluate.py results/S1 --judge sarvamai/sarvam-1 --gpu 0   # + perplexity under a judge LM
    bash eval_all.sh 0                                         # final S1-S4 table (Aya + Dolly), GPU 0

Per output: words, native-script share, english flag, repetition, chrF++ (files with references: Dolly, FLORES;
sentence mean and corpus-level),
agreement errors, judge perplexity, plus the SD stats saved by run_generation.py.
Writes {results_dir}/eval_{data}_{lang}.jsonl (per output) and prints one summary row per file.
"""
import argparse
import json
import math
from pathlib import Path

from agreement import agreement_errors, load_stanza as _load_stanza
from config import DATA, LANG_2, SCRIPT_RANGE

REPETITIVE = 0.3  # share of repeated word 4-grams above which an output counts as looping


# ---------- cheap metrics ----------

def script_share(text, lang):
    lo, hi = SCRIPT_RANGE[lang]
    letters = [c for c in text if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) if letters else 0.0


def repetition(words, n=4):
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def distinct2(words):
    grams = list(zip(words, words[1:]))
    return len(set(grams)) / len(grams) if grams else 0.0


# ---------- agreement errors (Stanza): see agreement.py ----------

def load_stanza(lang, gpu):
    return _load_stanza(LANG_2[lang], use_gpu=gpu is not None)


# ---------- judge perplexity ----------

class Judge:
    def __init__(self, name, gpu):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": f"cuda:{gpu}"}, dtype=torch.float16).eval()

    def ppl(self, prompt, output):
        """Perplexity of the output tokens given the prompt."""
        torch = self.torch
        p = self.tok(prompt).input_ids
        o = self.tok(output, add_special_tokens=False).input_ids
        if not o:
            return float("nan")
        x = torch.tensor([p + o], device=self.model.device)
        with torch.no_grad():
            logits = self.model(x).logits[0, len(p) - 1:-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, len(p):])
        return math.exp(nll.item())


# ---------- main ----------

def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def references(data, lang):
    path = DATA / "prompts" / f"{data}_{lang}.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r.get("reference") for r in map(json.loads, f)}


def evaluate_file(path, use_stanza, judge, gpu, parsers):
    data, lang = path.stem.rsplit("_", 1)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    refs = references(data, lang)
    if use_stanza and lang not in parsers:
        parsers[lang] = load_stanza(lang, gpu)
    chrf = None
    if any(refs.get(r["id"]) for r in rows):
        from sacrebleu.metrics import CHRF
        chrf = CHRF(word_order=2)  # chrF++

    out = []
    for r in rows:
        text = r["output"].strip()
        words = text.split()
        m = {"id": r["id"], "words": len(words), "native": script_share(text, lang),
             "repetition": repetition(words), "distinct2": distinct2(words), **r["stats"]}
        m["english"] = m["native"] < 0.5
        m["looping"] = m["repetition"] > REPETITIVE
        ref = refs.get(r["id"])
        if chrf and ref:
            m["chrf++"] = chrf.sentence_score(text, [ref]).score
        if use_stanza and words:
            n_err, ex = agreement_errors(parsers[lang](text), lang)
            m["agr_errors"], m["agr_examples"] = n_err, ex
            m["agr_per_100w"] = 100 * n_err / len(words)
        if judge:
            m["judge_ppl"] = judge.ppl(r["prompt"], text)
        out.append(m)

    with open(path.parent / f"eval_{path.stem}.jsonl", "w", encoding="utf-8") as f:
        for m in out:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    hindi_like = [m for m in out if not m["english"] and m["words"]]
    summary = {
        "n": len(out),
        "words": mean([m["words"] for m in out]),
        "%english": 100 * mean([m["english"] for m in out]),
        "%looping": 100 * mean([m["looping"] for m in out]),
        "distinct2": mean([m["distinct2"] for m in out]),
        "accept": mean([m["acceptance_rate"] for m in out]),
        "tok/s": mean([m["tok_per_s"] for m in out]),
        "tok/step": mean([m["tokens_per_step"] for m in out]),
    }
    if chrf:
        summary["chrF++"] = mean([m.get("chrf++") for m in out])
        pairs = [(r["output"].strip(), refs[r["id"]]) for r in rows if refs.get(r["id"])]
        summary["chrF++corpus"] = chrf.corpus_score([h for h, _ in pairs], [[t for _, t in pairs]]).score
    if use_stanza:  # computed on native-script outputs only
        summary["agr/100w"] = mean([m.get("agr_per_100w") for m in hindi_like])
    if judge:
        summary["judge_ppl"] = mean([m.get("judge_ppl") for m in hindi_like])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path, help="results/<system> folders")
    ap.add_argument("--data", nargs="+", help="only these datasets, e.g. aya dolly (default: all files)")
    ap.add_argument("--stanza", action="store_true", help="agreement errors via Stanza parse")
    ap.add_argument("--judge", help="HF model id for perplexity, e.g. sarvamai/sarvam-1")
    ap.add_argument("--gpu", type=int, help="GPU index for Stanza/judge (default: CPU for Stanza)")
    ap.add_argument("--csv", type=Path, help="also write all summary rows to this CSV file")
    args = ap.parse_args()

    if args.gpu is not None:
        import torch
        torch.cuda.set_device(args.gpu)
    judge = Judge(args.judge, args.gpu or 0) if args.judge else None
    parsers = {}

    table = []  # (system, data, lang, summary)
    for d in args.dirs:
        for path in sorted(d.glob("*.jsonl")):
            if path.name.startswith("eval_"):
                continue
            data, lang = path.stem.rsplit("_", 1)
            if args.data and data not in args.data:
                continue
            s = evaluate_file(path, args.stanza, judge, args.gpu, parsers)
            table.append((d.name, data, lang, s))
            cells = "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items())
            print(f"{d.name:8s} {path.stem:12s} {cells}", flush=True)

    if len(args.dirs) > 1:  # side-by-side: one block per dataset/language, one row per system
        keys = list(dict.fromkeys(k for *_, s in table for k in s))
        print("\n" + f"{'system':22s}" + "".join(f"{k:>11s}" for k in keys))
        for data, lang in dict.fromkeys((t[1], t[2]) for t in table):
            print(f"--- {data}_{lang}")
            for name, d2, l2, s in table:
                if (d2, l2) == (data, lang):
                    print(f"{name:22s}" + "".join(f"{s[k]:11.2f}" if isinstance(s.get(k), float) else f"{s.get(k, ''):>11}"
                                                  for k in keys))
    if args.csv:
        import csv
        keys = list(dict.fromkeys(k for *_, s in table for k in s))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["system", "data", "lang"] + keys)
            for name, data, lang, s in table:
                w.writerow([name, data, lang] + [round(s[k], 4) if isinstance(s.get(k), float) else s.get(k, "") for k in keys])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
