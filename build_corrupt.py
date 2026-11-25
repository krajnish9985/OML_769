"""Module C data: clean vs corrupted sentence prefixes from IndicCorp.

    python build_corrupt.py                    # 200k pairs per language -> data/corrupt/{train,dev}.jsonl
    python build_corrupt.py --pairs 2000       # quick look

At decoding time module C judges the word that was just finished, given the sentence so far.
So each training pair is one prefix cut after word j, in two versions:
    label 1: words[:j+1]                        (the real text)
    label 0: words[:j] + [corrupted word j]     (same prefix, last word broken)
Corruptions of the last word:
    suffix    gender/number/person ending swapped   (करता -> करती, வந்தான் -> வந்தாள்)
    word      auxiliary swapped                      (है -> हैं, था -> थे, आहे -> आहेत)
    drop      word j skipped (word j+1 takes its place)
    dup       previous word repeated
Prefixes use the same tokenization as decoding: whitespace words, punctuation stripped.
Some corruptions give a correct sentence by chance (e.g. a gender swap with an ambiguous subject):
that label noise is expected and small.
"""
import argparse
import json
import random

from config import DATA, LANGS, SEED
from modules.common import words_of

# Only swaps that the LEFT context can show to be wrong (agreement with an earlier subject).
# Not used: postposition swaps (राम ने / राम को), genitive का/की/के, Marathi चा/ची/चे and Tamil case
# endings, because which one is right depends on words that come AFTER them, so a prefix can't judge it.
# endings that can replace each other
SUFFIXES = {
    "hin": [["ता", "ती", "ते"], ["ेगा", "ेगी", "ेंगे"], ["ूंगा", "ूंगी"]],
    "mar": [["तो", "ती", "ते", "तात"], ["ला", "ली", "ले"], ["ेल", "तील"]],
    "tam": [["கிறான்", "கிறாள்", "கிறார்", "கிறது", "கிறார்கள்"], ["ார்கள்", "ான்", "ாள்", "ார்"]],
}
# whole function words that can replace each other
WORDS = {
    "hin": [["है", "हैं"], ["था", "थी", "थे"], ["रहा", "रही", "रहे"], ["गया", "गई", "गए"],
            ["सकता", "सकती", "सकते"], ["हूं", "हो", "हैं"]],
    "mar": [["आहे", "आहेत"], ["होता", "होती", "होते", "होत्या"], ["नाही", "नाहीत"]],
    "tam": [["இருக்கிறது", "இருக்கின்றன"], ["உள்ளது", "உள்ளன"], ["இருந்தது", "இருந்தன"]],
}
MIN_STEM = 2  # letters that must stay in front of a swapped ending


def morph_options(word, lang):
    """All corrupted versions of a word: (type, new word)."""
    out = []
    for group in WORDS[lang]:
        if word in group:
            out += [("word", w) for w in group if w != word]
    endings = sorted({e for g in SUFFIXES[lang] for e in g}, key=len, reverse=True)
    for e in endings:  # longest matching ending only
        if word.endswith(e) and len(word) - len(e) >= MIN_STEM:
            group = next(g for g in SUFFIXES[lang] if e in g)
            out += [("suffix", word[: -len(e)] + x) for x in group if x != e]
            break
    return out


def make_pair(words, lang, rng, p_morph=0.7):
    """One (clean, corrupted, type) prefix pair from a sentence, or None."""
    n = len(words)
    morph = [(j, o) for j in range(1, n) for o in morph_options(words[j], lang)]
    if morph and rng.random() < p_morph:
        j, (kind, new) = rng.choice(morph)
    elif rng.random() < 0.5 and n >= 3:
        j = rng.randrange(1, n - 1)
        kind, new = "drop", words[j + 1]
    else:
        j = rng.randrange(2, n) if n >= 3 else 1
        kind, new = "dup", words[j - 1]
    if new == words[j]:
        return None
    return " ".join(words[: j + 1]), " ".join(words[:j] + [new]), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--pairs", type=int, default=200_000, help="training pairs per language")
    ap.add_argument("--dev", type=int, default=2000, help="dev pairs per language")
    args = ap.parse_args()

    rng = random.Random(SEED)
    out = DATA / "corrupt"
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "dev": []}
    for lang in args.lang:
        need = args.pairs + args.dev
        pairs, kinds = [], {}
        with open(DATA / "indiccorp" / f"{lang}.jsonl", encoding="utf-8") as f:
            for line in f:
                words = words_of(json.loads(line)["text"])
                if len(words) < 3:
                    continue
                p = make_pair(words, lang, rng)
                if p:
                    pairs.append(p)
                    kinds[p[2]] = kinds.get(p[2], 0) + 1
                if len(pairs) >= need:
                    break
        rng.shuffle(pairs)
        for split, part in (("dev", pairs[: args.dev]), ("train", pairs[args.dev:])):
            for clean, bad, kind in part:
                rows[split].append({"text": clean, "label": 1, "lang": lang, "type": kind})
                rows[split].append({"text": bad, "label": 0, "lang": lang, "type": kind})
        print(f"{lang}: {len(pairs)} pairs  types {kinds}", flush=True)
        for clean, bad, kind in pairs[:3]:
            print(f"  [{kind}]  ok: ...{clean[-50:]}\n          bad: ...{bad[-50:]}", flush=True)
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out / f'{split}.jsonl'} ({len(rs)} rows)", flush=True)


if __name__ == "__main__":
    main()
