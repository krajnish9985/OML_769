"""Agreement checks on a Stanza parse, shared by evaluate.py (metric) and modules/rules.py (module A).

find_errors(sentence, lang) returns a list of
    {"rule": ..., "dep": word, "head": word, "feats": [...]}
for these rules:
    subj_verb   nominal subject vs its verb (features from the verb, then its aux/cop)
    adj_noun    inflecting adjective (lemma in -ा) vs its noun; not Tamil
    ne_object   Hindi: subject + ने, perfective verb must agree with an unmarked object
    oblique     Hindi: masculine -ा noun directly before a postposition must be oblique (लड़का को -> लड़के को)

Skipped to limit parser noise: proper nouns, numbers/abbreviations, words in lists (coordination
is the most common parser failure), and subjects marked by a postposition (ने, को).
This is parser-based: parser mistakes also show up as errors.
"""
import unicodedata

AGR_FEATS = ["Gender", "Number", "Person"]
OBLIQUE_CASES = ("Erg", "Dat", "Acc", "Gen", "Ins", "Loc")
# masculine nouns in -ा that do not change before a postposition
NON_INFLECTING_A = {"राजा", "पिता", "दादा", "चाचा", "नाना", "मामा", "देवता", "नेता", "योद्धा", "आत्मा",
                    "काका", "बाबा", "भैया", "मुखिया", "दरोगा", "अभिनेता", "दाता", "कर्ता", "वक्ता"}


def load_stanza(lang2, use_gpu, pretokenized=False, pos_only=False, **extra):
    """Stanza pipeline; pos_only=True skips lemma + depparse (POS and morph features only).

    Uses the models already downloaded, offline (no update check over the network, which can fail
    on the server); only if a model is missing does it go online to fetch it.
    """
    import stanza
    from stanza.pipeline.core import DownloadMethod
    kw = dict(lang=lang2, use_gpu=use_gpu, verbose=False, tokenize_pretokenized=pretokenized, **extra)
    rest = "pos" if pos_only else "pos,lemma,depparse"
    err = None
    for download in (None, DownloadMethod.REUSE_RESOURCES):  # None = offline
        for procs in (f"tokenize,mwt,{rest}", f"tokenize,{rest}"):  # Hindi has no MWT model
            try:
                return stanza.Pipeline(processors=procs, download_method=download, **kw)
            except Exception as e:
                err = e
    raise err


def feats(word):
    return dict(f.split("=", 1) for f in word.feats.split("|")) if word.feats else {}


def is_word(text):
    """Letters and combining marks (matras) only: excludes numbers, abbreviations, punctuation."""
    return all(c.isalpha() or unicodedata.category(c).startswith("M") for c in text)


def _verb_feats(head, words):
    fh = {}
    for v in [head] + [c for c in words if c.head == head.id and c.deprel in ("aux", "aux:pass", "cop")]:
        for k, val in feats(v).items():
            fh.setdefault(k, val)
    return fh


def _mismatch(fa, fb, keys):
    return [k for k in keys if fa.get(k) and fb.get(k) and fa[k] != fb[k]]


def find_errors(sent, lang):
    words = sent.words
    in_list = {x.id for x in words if x.deprel == "conj"} | {x.head for x in words if x.deprel == "conj"}
    errors = []

    def clean(w, h):
        return not (w.id in in_list or h.id in in_list or "PROPN" in (w.upos, h.upos)
                    or not (is_word(w.text) and is_word(h.text)) or min(len(w.text), len(h.text)) < 2)

    for w in words:
        if w.head == 0:
            continue
        head = words[w.head - 1]
        fw = feats(w)
        case_children = [c for c in words if c.head == w.id and c.deprel == "case"]

        # oblique noun before a postposition (Hindi)
        if (lang == "hin" and w.deprel == "case" and w.upos == "ADP" and head.id == w.id - 1
                and head.upos == "NOUN" and head.text.endswith("ा") and head.text not in NON_INFLECTING_A
                and feats(head).get("Gender") == "Masc" and feats(head).get("Number", "Sing") == "Sing"
                and is_word(head.text)):
            errors.append({"rule": "oblique", "dep": head, "head": w, "feats": ["Case"]})
            continue

        if not clean(w, head):
            continue

        if w.deprel in ("nsubj", "nsubj:pass") and w.upos in ("NOUN", "PRON") and head.upos in ("VERB", "AUX", "ADJ"):
            ne = any(c.lemma == "ने" or c.text == "ने" for c in case_children)
            if lang == "hin" and ne and head.upos == "VERB":
                # ने + perfective: verb agrees with an object that has no postposition
                for o in words:
                    if (o.head == head.id and o.deprel == "obj" and o.upos == "NOUN" and clean(o, head)
                            and not any(c.head == o.id and c.deprel == "case" for c in words)):
                        bad = _mismatch(feats(o), _verb_feats(head, words), ["Gender", "Number"])
                        if bad:
                            errors.append({"rule": "ne_object", "dep": o, "head": head, "feats": bad})
                continue
            if case_children or w.text.endswith(("ने", "को")) or fw.get("Case") in OBLIQUE_CASES:
                continue
            if fw.get("PronType") in ("Int", "Ind", "Rel"):  # क्या/என்ன "what", कोई ...
                continue
            if lang == "tam" and feats(head).get("VerbForm") != "Fin":  # infinitives don't agree
                continue
            skip = {"Number", "Person"} if w.upos == "PRON" else set()  # honorific आप, ambiguous हैं
            if lang in ("hin", "mar") and fw.get("Gender") == "Masc" and not w.text.endswith("ा"):
                skip.add("Number")  # खिलाड़ी, वैज्ञानिक: same form in singular and plural
            if lang == "tam" and fw.get("Gender", "Neut") == "Neut":
                skip.add("Number")  # neuter plural + singular verb (-கிறது) is common modern Tamil
            bad = _mismatch(fw, _verb_feats(head, words), [k for k in AGR_FEATS if k not in skip])
            if bad:
                errors.append({"rule": "subj_verb", "dep": w, "head": head, "feats": bad})

        elif (w.deprel == "amod" and w.upos == "ADJ" and head.upos == "NOUN" and lang != "tam"
              and (w.lemma or w.text).endswith("ा")):  # बड़ा/बड़ी/बड़े inflect; भारतीय, सरकारी do not
            bad = _mismatch(fw, feats(head), ["Gender", "Number"])
            if bad:
                errors.append({"rule": "adj_noun", "dep": w, "head": head, "feats": bad})
    return errors


def agreement_errors(doc, lang):
    """(count, up to 3 examples) over a parsed document; used by evaluate.py."""
    n, examples = 0, []
    for sent in doc.sentences:
        for e in find_errors(sent, lang):
            n += 1
            if len(examples) < 3:
                examples.append(f"{e['rule']}: {e['dep'].text}->{e['head'].text} "
                                f"({','.join(e['feats'])}): {sent.text[:80]}")
    return n, examples
