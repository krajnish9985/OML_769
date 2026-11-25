# """Generate outputs for one system on the test prompts.

#     python run_generation.py --system S1 --gpu 0                    # Aya + Dolly, all languages
#     python run_generation.py --system S1 --data aya --lang hin --limit 20   # quick test
#     python run_generation.py --system S2 --variant hard --tau 0.3 --tag hard_tau0.3
#     python run_generation.py --system S3 --variant hard --tau 0.05 --tag hard_tau0.05
#     python run_generation.py --system S4 --variant hard --tau 0.1 --tag hard_tau0.1
#     python run_generation.py --system S1 --script-guard --tag guard     # any system + script guard
#     python run_generation.py --system S1 --data flores --lang mar       # English -> Marathi translation
#     python run_generation.py --system S1 --temperature 1.0 --tag T1.0   # higher temperature

# Always start with OMP_NUM_THREADS=2 (about 3x faster on a shared server).
# Model pair: BLOOMZ by default; `export PAIR=qwen` switches to Qwen2.5 0.5B -> 7B Instruct (config.py),
# with the chat template, its own lengths and outputs under results_qwen/ instead of results/.

# Systems: S1 = standard SD (no module), S2 = module A (rules), S3 = module B (tag 4-gram),
# S4 = module C (MuRIL classifier, data/classifier/).
# Output: results/{system}[_tag]/{data}_{lang}.jsonl, one line per prompt with the output text,
# per-prompt stats and a per-drafted-token log (p, q, L, accepted, reason).
# """
# import argparse
# import json

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from config import DATA, DRAFT, DTYPE, GEN_LEN, LANGS, MATCH_LEN, PAIR, PAIRS, RESULTS, TARGET, build_prompt, eos_ids
# from sd import SDConfig, SpeculativeDecoder

# def _rules(tok, lang, args):
#     from modules.rules import RuleScorer
#     return RuleScorer(tok, lang, groups=args.rules)


# def _stat(tok, lang, args):
#     from modules.stat import StatScorer
#     return StatScorer(tok, lang)


# def _neural(tok, lang, args):
#     from modules.neural import NeuralScorer
#     return NeuralScorer(tok, lang, model_path=args.clf)


# MODULES = {"S1": None, "S2": _rules, "S3": _stat, "S4": _neural}  # system -> factory(tok, lang, args) returning a Scorer

# # data name -> (file pattern, prompt field, min_new_tokens, max_new_tokens)
# DATASETS = {
#     "aya": ("prompts/aya_{lang}.jsonl", "prompt", *GEN_LEN),
#     "dolly": ("prompts/dolly_{lang}.jsonl", "prompt", *GEN_LEN),
#     "flores": ("prompts/flores_{lang}.jsonl", "prompt", 0, 160),  # one sentence: no forced length, stops at EOS
#     "blimp": ("multiblimp/{lang}.jsonl", "prefix", 0, 32),  # short continuation after the prefix
# }


# def load(name, lang):
#     path, field, _, _ = DATASETS[name]
#     with open(DATA / path.format(lang=lang), encoding="utf-8") as f:
#         rows = [json.loads(line) for line in f]
#     for i, r in enumerate(rows):
#         r.setdefault("id", f"{name}_{lang}_{i}")
#         r["_prompt"] = (r.get(field) or "").strip()
#     return [r for r in rows if r["_prompt"]]  # some MultiBLiMP prefixes are empty (verb is the first word)


# def token_ratio(tok, ref_tok, rows):
#     """Tokens of this pair's tokenizer per BLOOM token, over all prompts of one file."""
#     texts = [r["_prompt"] for r in rows]
#     n = sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
#     n_ref = sum(len(ref_tok(t, add_special_tokens=False).input_ids) for t in texts)
#     return n / max(n_ref, 1)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--system", default="S1", choices=list(MODULES))
#     ap.add_argument("--data", nargs="+", default=["aya", "dolly"], choices=list(DATASETS),
#                     help="blimp generation was dropped after S1 (too short to judge); add it explicitly if needed")
#     ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS)
#     ap.add_argument("--limit", type=int, help="first N prompts per file")
#     ap.add_argument("--gpu", type=int, default=0)
#     ap.add_argument("--k", type=int, default=4)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--variant", default="soft", choices=["soft", "hard"])
#     ap.add_argument("--alpha", type=float, default=1.0)
#     ap.add_argument("--tau", type=float, default=0.3)
#     ap.add_argument("--min-new-tokens", type=int, help="override the per-dataset minimum length")
#     ap.add_argument("--rules", default="all", choices=["all", "surface", "grammar"],
#                     help="S2 only: which rule group is active")
#     ap.add_argument("--script-guard", action="store_true",
#                     help="penalize drift into Latin script on every token (draft, target, replacement, bonus)")
#     ap.add_argument("--guard-budget", type=float, default=0.2, help="allowed Latin share for a native prompt")
#     ap.add_argument("--guard-strength", type=float, default=0.05, help="Latin tokens x this when over budget")
#     ap.add_argument("--clf", help="S4 only: classifier folder (default data/classifier)")
#     ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (keep small on shared machines)")
#     ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. alpha1")
#     args = ap.parse_args()

#     device = f"cuda:{args.gpu}"
#     torch.cuda.set_device(args.gpu)  # Stanza (module A) uses the current device
#     torch.set_num_threads(args.threads)  # the per-token CPU work is tiny; many threads only fight each other
#     print(f"pair {PAIR}: draft {DRAFT}, target {TARGET}", flush=True)
#     tok = AutoTokenizer.from_pretrained(TARGET)
#     load_kw = dict(device_map={"": device}, dtype=getattr(torch, DTYPE))
#     target = AutoModelForCausalLM.from_pretrained(TARGET, **load_kw).eval()
#     draft = AutoModelForCausalLM.from_pretrained(DRAFT, **load_kw).eval()

#     out_dir = RESULTS / (args.system + (f"_{args.tag}" if args.tag else ""))
#     vocab = min(m.get_output_embeddings().weight.shape[0] for m in (draft, target))  # logits are cut to this
#     ref_tok = AutoTokenizer.from_pretrained(PAIRS["bloomz"]["target"]) if MATCH_LEN else None  # tokenizer only
#     eos = eos_ids(tok, target)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for lang in args.lang:
#         make = MODULES[args.system]
#         scorer = make(tok, lang, args) if make else None
#         guard = None
#         if args.script_guard:
#             from modules.script_guard import ScriptGuard
#             guard = ScriptGuard(tok, lang, vocab, budget=args.guard_budget, strength=args.guard_strength)
#         for name in args.data:
#             _, _, min_new, max_new = DATASETS[name]
#             all_rows = load(name, lang)
#             if ref_tok is not None and name != "blimp":  # same answer length in text as the BLOOMZ runs
#                 ratio = token_ratio(tok, ref_tok, all_rows)  # all prompts, so --limit does not change it
#                 min_new, max_new = round(min_new * ratio), round(max_new * ratio)
#                 print(f"{name}/{lang}: {ratio:.2f} tokens per BLOOM token -> min {min_new}, max {max_new} new tokens",
#                       flush=True)
#             if args.min_new_tokens is not None:
#                 min_new = args.min_new_tokens
#             cfg = SDConfig(k=args.k, temperature=args.temperature, min_new_tokens=min_new, max_new_tokens=max_new,
#                            variant=args.variant, alpha=args.alpha, tau=args.tau)
#             sd = SpeculativeDecoder(draft, target, eos, scorer, cfg, guard=guard)
#             rows = all_rows[: args.limit]
#             path = out_dir / f"{name}_{lang}.jsonl"
#             with open(path, "w", encoding="utf-8") as f:
#                 for i, r in enumerate(rows):
#                     user_ids = tok(r["_prompt"], add_special_tokens=False).input_ids  # without the chat template
#                     res = sd.generate(build_prompt(tok, r["_prompt"]), seed=i, user_ids=user_ids)
#                     rec = {"id": r["id"], "lang": lang, "prompt": r["_prompt"],
#                            "output": tok.decode(res.ids, skip_special_tokens=True),
#                            "stats": res.stats(), "tokens": res.tokens,
#                            "rules_fired": dict(getattr(scorer, "fired", {})),
#                            "guard": dict(guard.stats) if guard else None,
#                            "pair": PAIR, "config": vars(cfg)}
#                     f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#                     f.flush()
#                     if i % 20 == 0:
#                         s = rec["stats"]
#                         print(f"{name}/{lang} {i}/{len(rows)}  acc {s['acceptance_rate']:.2f}  "
#                               f"{s['tok_per_s']:.1f} tok/s  {rec['output'][:60]!r}", flush=True)
#             print(f"wrote {path}", flush=True)


# if __name__ == "__main__":
#     main()

"""Generate outputs for one system on the test prompts.

    python run_generation.py --system S1 --gpu 0                    # Aya + Dolly, all languages
    python run_generation.py --system S1 --data aya --lang hin --limit 20   # quick test
    python run_generation.py --system S2 --variant hard --tau 0.3 --tag hard_tau0.3
    python run_generation.py --system S3 --variant hard --tau 0.05 --tag hard_tau0.05
    python run_generation.py --system S4 --variant hard --tau 0.1 --tag hard_tau0.1
    python run_generation.py --system S1 --script-guard --tag guard     # any system + script guard
    python run_generation.py --system S1 --data flores --lang mar       # English -> Marathi translation
    python run_generation.py --system S1 --temperature 1.0 --tag T1.0   # higher temperature
    python run_generation.py --system S4 --variant hard --tau 0.3 --check-target --tag hard_tau0.3_ct  # judge target tokens too

Always start with OMP_NUM_THREADS=2 (about 3x faster on a shared server).
Model pair: BLOOMZ by default; `export PAIR=qwen` switches to Qwen2.5 0.5B -> 7B Instruct (config.py),
with the chat template, its own lengths and outputs under results_qwen/ instead of results/.

Systems: S1 = standard SD (no module), S2 = module A (rules), S3 = module B (tag 4-gram),
S4 = module C (MuRIL classifier, data/classifier/).
Output: results/{system}[_tag]/{data}_{lang}.jsonl, one line per prompt with the output text,
per-prompt stats and a per-drafted-token log (p, q, L, accepted, reason).
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA, DRAFT, DTYPE, GEN_LEN, LANGS, MATCH_LEN, PAIR, PAIRS, RESULTS, TARGET, build_prompt, eos_ids
from sd import SDConfig, SpeculativeDecoder

def _rules(tok, lang, args):
    from modules.rules import RuleScorer
    return RuleScorer(tok, lang, groups=args.rules)


def _stat(tok, lang, args):
    from modules.stat import StatScorer
    return StatScorer(tok, lang)


def _neural(tok, lang, args):
    from modules.neural import NeuralScorer
    return NeuralScorer(tok, lang, model_path=args.clf)


MODULES = {"S1": None, "S2": _rules, "S3": _stat, "S4": _neural}  # system -> factory(tok, lang, args) returning a Scorer

# data name -> (file pattern, prompt field, min_new_tokens, max_new_tokens)
DATASETS = {
    "aya": ("prompts/aya_{lang}.jsonl", "prompt", *GEN_LEN),
    "dolly": ("prompts/dolly_{lang}.jsonl", "prompt", *GEN_LEN),
    "flores": ("prompts/flores_{lang}.jsonl", "prompt", 0, 160),  # one sentence: no forced length, stops at EOS
    "blimp": ("multiblimp/{lang}.jsonl", "prefix", 0, 32),  # short continuation after the prefix
}


def load(name, lang):
    path, field, _, _ = DATASETS[name]
    with open(DATA / path.format(lang=lang), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for i, r in enumerate(rows):
        r.setdefault("id", f"{name}_{lang}_{i}")
        r["_prompt"] = (r.get(field) or "").strip()
    return [r for r in rows if r["_prompt"]]  # some MultiBLiMP prefixes are empty (verb is the first word)


def token_ratio(tok, ref_tok, rows):
    """Tokens of this pair's tokenizer per BLOOM token, over all prompts of one file."""
    texts = [r["_prompt"] for r in rows]
    n = sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
    n_ref = sum(len(ref_tok(t, add_special_tokens=False).input_ids) for t in texts)
    return n / max(n_ref, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="S1", choices=list(MODULES))
    ap.add_argument("--data", nargs="+", default=["aya", "dolly"], choices=list(DATASETS),
                    help="blimp generation was dropped after S1 (too short to judge); add it explicitly if needed")
    ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--limit", type=int, help="first N prompts per file")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--variant", default="soft", choices=["soft", "hard"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--min-new-tokens", type=int, help="override the per-dataset minimum length")
    ap.add_argument("--rules", default="all", choices=["all", "surface", "grammar"],
                    help="S2 only: which rule group is active")
    ap.add_argument("--script-guard", action="store_true",
                    help="penalize drift into Latin script on every token (draft, target, replacement, bonus)")
    ap.add_argument("--guard-budget", type=float, default=0.2, help="allowed Latin share for a native prompt")
    ap.add_argument("--guard-strength", type=float, default=0.05, help="Latin tokens x this when over budget")
    ap.add_argument("--check-target", action="store_true",
                    help="S2-S4: also judge tokens the target samples itself (replacement, bonus); see sd.py")
    ap.add_argument("--max-target-retries", type=int, default=3,
                    help="with --check-target: resamples per position before a failing target token is kept")
    ap.add_argument("--clf", help="S4 only: classifier folder (default data/classifier)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (keep small on shared machines)")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. alpha1")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)  # Stanza (module A) uses the current device
    torch.set_num_threads(args.threads)  # the per-token CPU work is tiny; many threads only fight each other
    print(f"pair {PAIR}: draft {DRAFT}, target {TARGET}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    load_kw = dict(device_map={"": device}, dtype=getattr(torch, DTYPE))
    target = AutoModelForCausalLM.from_pretrained(TARGET, **load_kw).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, **load_kw).eval()

    out_dir = RESULTS / (args.system + (f"_{args.tag}" if args.tag else ""))
    vocab = min(m.get_output_embeddings().weight.shape[0] for m in (draft, target))  # logits are cut to this
    ref_tok = AutoTokenizer.from_pretrained(PAIRS["bloomz"]["target"]) if MATCH_LEN else None  # tokenizer only
    eos = eos_ids(tok, target)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.lang:
        make = MODULES[args.system]
        scorer = make(tok, lang, args) if make else None
        guard = None
        if args.script_guard:
            from modules.script_guard import ScriptGuard
            guard = ScriptGuard(tok, lang, vocab, budget=args.guard_budget, strength=args.guard_strength)
        for name in args.data:
            _, _, min_new, max_new = DATASETS[name]
            all_rows = load(name, lang)
            if ref_tok is not None and name != "blimp":  # same answer length in text as the BLOOMZ runs
                ratio = token_ratio(tok, ref_tok, all_rows)  # all prompts, so --limit does not change it
                min_new, max_new = round(min_new * ratio), round(max_new * ratio)
                print(f"{name}/{lang}: {ratio:.2f} tokens per BLOOM token -> min {min_new}, max {max_new} new tokens",
                      flush=True)
            if args.min_new_tokens is not None:
                min_new = args.min_new_tokens
            cfg = SDConfig(k=args.k, temperature=args.temperature, min_new_tokens=min_new, max_new_tokens=max_new,
                           variant=args.variant, alpha=args.alpha, tau=args.tau,
                           check_target=args.check_target, max_target_retries=args.max_target_retries)
            sd = SpeculativeDecoder(draft, target, eos, scorer, cfg, guard=guard)
            rows = all_rows[: args.limit]
            path = out_dir / f"{name}_{lang}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    user_ids = tok(r["_prompt"], add_special_tokens=False).input_ids  # without the chat template
                    res = sd.generate(build_prompt(tok, r["_prompt"]), seed=i, user_ids=user_ids)
                    rec = {"id": r["id"], "lang": lang, "prompt": r["_prompt"],
                           "output": tok.decode(res.ids, skip_special_tokens=True),
                           "stats": res.stats(), "tokens": res.tokens,
                           "rules_fired": dict(getattr(scorer, "fired", {})),
                           "guard": dict(guard.stats) if guard else None,
                           "pair": PAIR, "config": vars(cfg)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 20 == 0:
                        s = rec["stats"]
                        print(f"{name}/{lang} {i}/{len(rows)}  acc {s['acceptance_rate']:.2f}  "
                              f"{s['tok_per_s']:.1f} tok/s  {rec['output'][:60]!r}", flush=True)
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
