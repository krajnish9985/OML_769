"""Build module B's tag 4-gram model from IndicCorp.

    export OMP_NUM_THREADS=2
    python build_ngram.py tag --lang hin --n 1000000 --gpu 0   # Stanza POS+morph tags -> data/ngram/hin.tags
    python build_ngram.py train --lang hin                     # Kneser-Ney 4-gram    -> data/ngram/hin.pkl

`train` also stores how real held-out words score, so module B can report L as a percentile of real text.

Tagging uses the same tokenization as the decoding loop (whitespace words, punctuation stripped).
`tag` appends and resumes: re-running continues after the sentences already tagged.
"""
import argparse
import json
import random
import time

from config import DATA, LANG_2, LANGS, SEED

OUT = DATA / "ngram"


def tag(lang, n, gpu, batch, threads):
    import torch
    from agreement import load_stanza
    from modules.common import words_of
    from ngram import word_tag

    torch.cuda.set_device(gpu)  # one GPU only
    torch.set_num_threads(threads)  # many CPU threads only fight each other on a shared machine
    nlp = load_stanza(LANG_2[lang], use_gpu=True, pretokenized=True, pos_only=True, pos_batch_size=5000)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{lang}.tags"
    done = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
    with open(DATA / "indiccorp" / f"{lang}.jsonl", encoding="utf-8") as f:
        sents = [w for w in (words_of(json.loads(line)["text"]) for line in f) if len(w) >= 3][:n]
    print(f"{lang}: {len(sents)} sentences, {done} already tagged", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as out:
        for start in range(done, len(sents), batch):
            doc = nlp(sents[start:start + batch])
            for s in doc.sentences:
                out.write(" ".join(word_tag(w) for w in s.words) + "\n")
            k = min(start + batch, len(sents))
            print(f"  {k}/{len(sents)}  {(k - done) / (time.time() - t0):.0f} sent/s", flush=True)


def train(lang, order, heldout):
    from ngram import KNLM

    with open(OUT / f"{lang}.tags", encoding="utf-8") as f:
        sents = [line.split() for line in f if line.strip()]
    random.Random(SEED).shuffle(sents)
    test, train_ = sents[:heldout], sents[heldout:]
    t0 = time.time()
    lm = KNLM(order).fit(train_)
    print(f"{lang}: {len(train_)} sentences, {len(lm.vocab)} tags, "
          f"{sum(len(v) for v in lm.levels[order].values())} {order}-gram types, {time.time() - t0:.0f}s", flush=True)
    print(f"  held-out perplexity ({len(test)} sentences): {lm.perplexity(test):.2f}", flush=True)
    t0 = time.time()
    lm.calibrate(test)
    c = lm.calib
    print(f"  calibration on held-out words ({time.time() - t0:.0f}s): raw score at percentile "
          f"1% {c[len(c) // 100]:.4f}, 5% {c[len(c) // 20]:.4f}, 25% {c[len(c) // 4]:.4f}, 50% {c[len(c) // 2]:.4f}",
          flush=True)
    lm.save(OUT / f"{lang}.pkl")
    print(f"  wrote {OUT / f'{lang}.pkl'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tag", "train"])
    ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--n", type=int, default=1_000_000, help="tag: sentences per language")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--heldout", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=2, help="tag: CPU threads for torch")
    args = ap.parse_args()
    for lang in args.lang:
        if args.step == "tag":
            tag(lang, args.n, args.gpu, args.batch, args.threads)
        else:
            train(lang, args.order, args.heldout)
