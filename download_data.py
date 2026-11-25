"""Download and prepare all datasets into data/.

    python download_data.py                     # everything
    python download_data.py aya dolly blimp ud  # pick some
    python download_data.py indiccorp --n-sents 1000000

Output layout:
    data/prompts/aya_{lang}.jsonl         ~300 native prompts per language
    data/prompts/dolly_{lang}.jsonl       ~200 prompts, same source_id across languages
    data/prompts/flores_{lang}.jsonl      1012 FLORES-200 devtest sentences, English -> lang (reference = lang)
    data/multiblimp/{lang}.jsonl          minimal pairs (prefix, verb, wrong_verb, ...)
    data/indiccorp/{lang}.jsonl           clean sentences {"text": ...}
    data/ud/{lang}/*.conllu + *.jsonl     UD treebank splits
"""
import argparse
import json
import random
import urllib.request

from datasets import get_dataset_config_names, load_dataset

from config import ANSWER_HINT, DATA, LANG_2, LANG_NAME, LANGS, SCRIPT_RANGE, SEED, UD, UD_TAG


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.relative_to(DATA.parent)}: {len(rows)} rows")


def lang_match(value, lang):
    v = str(value).lower()
    return v in {lang, LANG_2[lang], LANG_NAME[lang].lower()} or v.startswith(lang + "_")


def script_ratio(text, lang):
    lo, hi = SCRIPT_RANGE[lang]
    letters = [c for c in text if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) if letters else 0.0


def aya(n=300, max_chars=400):
    """Native-script, open-ended prompts only: drops English prompts, MCQs (options or >2 line breaks) and fill-in-the-blanks."""
    print("aya_dataset")
    ds = load_dataset("CohereLabs/aya_dataset", split="train")
    for lang in LANGS:
        sub = ds.filter(lambda x: x["language_code"] == lang)
        seen, rows = set(), []
        for x in sub:
            q = x["inputs"].strip()
            if q in seen or not (10 <= len(q) <= max_chars):
                continue
            if script_ratio(q, lang) < 0.8 or "__" in q or "(A)" in q or "(a)" in q or q.count("\n") > 2:
                continue
            seen.add(q)
            rows.append({"input": q, "prompt": f"{q}\n{ANSWER_HINT[lang]}", "reference": x["targets"].strip()})
        random.Random(SEED).shuffle(rows)
        rows = [dict(id=f"aya_{lang}_{i}", lang=lang, **r) for i, r in enumerate(rows[:n])]
        write_jsonl(DATA / "prompts" / f"aya_{lang}.jsonl", rows)


def dolly(n=200):
    print("aya_evaluation_suite / dolly_machine_translated")
    ds = load_dataset("CohereLabs/aya_evaluation_suite", "dolly_machine_translated", split="test")
    key = "source_id" if "source_id" in ds.column_names else "id"
    per_lang = {}
    for lang in LANGS:
        sub = ds.filter(lambda x: lang_match(x["language"], lang))
        per_lang[lang] = {x[key]: x for x in sub}
        print(f"  {lang}: {len(sub)} rows")
    shared = sorted(set.intersection(*(set(d) for d in per_lang.values())), key=str)
    random.Random(SEED).shuffle(shared)
    shared = shared[:n]
    for lang in LANGS:
        rows = [{"id": f"dolly_{sid}", "source_id": sid, "lang": lang,
                 "prompt": per_lang[lang][sid]["inputs"].strip(),
                 "reference": per_lang[lang][sid]["targets"].strip()} for sid in shared]
        write_jsonl(DATA / "prompts" / f"dolly_{lang}.jsonl", rows)


FLORES_CODE = {"hin": "hin_Deva", "mar": "mar_Deva", "tam": "tam_Taml"}
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"  # public, ~25 MB, CC BY-SA 4.0


def _flores_devtest(code):
    """Lines of one FLORES-200 devtest file: the public tarball first, the gated HF copy (flores_plus) as fallback."""
    import io
    import tarfile
    try:
        cache = DATA / "flores200_dataset.tar.gz"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(FLORES_URL, cache)
        with tarfile.open(cache) as tar:
            f = tar.extractfile(f"flores200_dataset/devtest/{code}.devtest")
            return io.TextIOWrapper(f, encoding="utf-8").read().splitlines()
    except Exception as e:
        (DATA / "flores200_dataset.tar.gz").unlink(missing_ok=True)  # may be a half download
        print(f"  tarball failed ({type(e).__name__}: {e}); trying openlanguagedata/flores_plus (accept its terms on HF)")
        ds = load_dataset("openlanguagedata/flores_plus", code, split="devtest")
        return [x["text"] for x in sorted(ds, key=lambda x: x["id"])]


def flores():
    """English -> Indic translation prompts. Same sentence ids in every language, so files are parallel."""
    print("flores-200 devtest")
    src = _flores_devtest("eng_Latn")
    for lang in LANGS:
        ref = _flores_devtest(FLORES_CODE[lang])
        assert len(ref) == len(src), (lang, len(ref), len(src))
        rows = [{"id": f"flores_{i}", "lang": lang, "source": s.strip(),
                 "prompt": f"Translate to {LANG_NAME[lang]}: {s.strip()}",  # BLOOMZ model-card prompt style
                 "reference": r.strip()} for i, (s, r) in enumerate(zip(src, ref))]
        write_jsonl(DATA / "prompts" / f"flores_{lang}.jsonl", rows)


def blimp():
    print("multiblimp")
    configs = get_dataset_config_names("jumelet/multiblimp")
    for lang in LANGS:
        cfg = next((c for c in configs if lang_match(c, lang)), None)
        if cfg is None:
            raise SystemExit(f"no MultiBLiMP config for {lang}; available: {configs}")
        ds = load_dataset("jumelet/multiblimp", cfg)
        split = next(iter(ds))  # single split
        rows = [{"lang": lang, **x} for x in ds[split]]  # dataset may have its own lang column
        write_jsonl(DATA / "multiblimp" / f"{lang}.jsonl", rows)


def indiccorp(n_sents=1_000_000, langs=LANGS, min_words=5, max_words=50, min_script=0.9):
    """Stream IndicCorp v2 and keep clean, deduplicated sentences in the right script.

    Only the first part of each language file is read (streaming), not the whole multi-GB file.
    """
    print("IndicCorp v2")
    import re
    from huggingface_hub import list_repo_files

    repo = "ai4bharat/IndicCorpV2"
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".txt")]
    for lang in langs:
        pat = re.compile(rf"^({LANG_2[lang]}|{lang})([._-]|$)", re.I)  # hi.txt, hi-1.txt, hin_Deva.txt
        mine = sorted(f for f in files if pat.match(f.rsplit("/", 1)[-1]))
        if not mine:
            raise SystemExit(f"no IndicCorp file for {lang}; files: {files}")
        print(f"  {lang}: {mine}")
        urls = [f"hf://datasets/{repo}/{f}" for f in mine]  # streamed, never fully downloaded
        stream = load_dataset("text", data_files=urls, split="train", streaming=True)
        seen, rows = set(), []
        for x in stream:
            s = x["text"].strip()
            if not s or s in seen or not (min_words <= len(s.split()) <= max_words):
                continue
            if script_ratio(s, lang) < min_script:
                continue
            seen.add(s)
            rows.append({"text": s})
            if len(rows) >= n_sents:
                break
        write_jsonl(DATA / "indiccorp" / f"{lang}.jsonl", rows)


def ud():
    print(f"UD treebanks ({UD_TAG})")
    import conllu

    for lang in LANGS:
        repo, prefix = UD[lang]
        out = DATA / "ud" / lang
        out.mkdir(parents=True, exist_ok=True)
        for split in ["train", "dev", "test"]:
            name = f"{prefix}-ud-{split}.conllu"
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{UD_TAG}/{name}"
            text = urllib.request.urlopen(url).read().decode("utf-8")
            (out / name).write_text(text, encoding="utf-8")
            rows = []
            for sent in conllu.parse(text):
                toks = [t for t in sent if isinstance(t["id"], int)]  # skip multiword ranges
                rows.append({
                    "sent_id": sent.metadata.get("sent_id"),
                    "text": sent.metadata.get("text"),
                    "tokens": [t["form"] for t in toks],
                    "lemmas": [t["lemma"] for t in toks],
                    "upos": [t["upos"] for t in toks],
                    "feats": [t["feats"] or {} for t in toks],
                    "head": [t["head"] for t in toks],
                    "deprel": [t["deprel"] for t in toks],
                })
            write_jsonl(out / f"{split}.jsonl", rows)


STEPS = {"aya": aya, "dolly": dolly, "flores": flores, "blimp": blimp, "indiccorp": indiccorp, "ud": ud}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", help=f"any of {list(STEPS)} (default: all)")
    ap.add_argument("--n-sents", type=int, default=1_000_000, help="IndicCorp sentences per language")
    ap.add_argument("--lang", nargs="+", default=LANGS, choices=LANGS, help="IndicCorp languages")
    args = ap.parse_args()
    for s in args.steps or STEPS:
        if s not in STEPS:
            ap.error(f"unknown step {s!r}")
        STEPS[s](n_sents=args.n_sents, langs=args.lang) if s == "indiccorp" else STEPS[s]()
