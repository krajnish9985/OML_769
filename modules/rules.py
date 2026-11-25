"""Module A: rule-based linguistic score L for speculative decoding (system S2).

Surface rules, checked on every drafted token (no parser, all languages):
    broken_matra  vowel sign with no consonant before it, or two vowel signs in a row
    script_mix    Latin letters while the output so far is in the native script
    repeat        the last 4 words already appeared earlier in the output (looping)
Parser rules (agreement.py), checked when a drafted token ends a word, i.e. the next drafted
token starts with a space or punctuation (the last token of a draft block is not parser-checked):
    subj_verb, adj_noun, ne_object (Hindi), oblique (Hindi)

L = 1 if no rule fires, else the lowest SEVERITY among the rules that fired.
"""
import re
import unicodedata
from collections import Counter

from agreement import find_errors, load_stanza
from config import LANG_2, SCRIPT_RANGE
from modules.common import decode_pieces, word_end_sentences
from sd import Scorer

SEVERITY = {"broken_matra": 0.0, "subj_verb": 0.1, "adj_noun": 0.1, "ne_object": 0.1,
            "oblique": 0.2, "script_mix": 0.3, "repeat": 0.3}
VOWEL_SIGNS = {"hin": (0x093E, 0x094C), "mar": (0x093E, 0x094C), "tam": (0x0BBE, 0x0BCC)}
class RuleScorer(Scorer):
    def __init__(self, tok, lang, groups="all", use_gpu=True):
        """groups: "surface" (no parser), "grammar" (parser rules only) or "all"."""
        self.groups = groups
        use_parser = groups in ("grammar", "all")
        self.tok, self.lang = tok, lang
        self.lo, self.hi = SCRIPT_RANGE[lang]
        self.vlo, self.vhi = VOWEL_SIGNS[lang]
        self.parser = load_stanza(LANG_2[lang], use_gpu, pretokenized=True) if use_parser else None
        self.n_prompt, self.fired = 0, Counter()
        self.last_rules, self._latin = [], None

    def reset(self, prompt_ids):
        self.n_prompt, self.fired = len(prompt_ids), Counter()

    # ---- helpers ----
    def _native_share(self, text):
        letters = [c for c in text if c.isalpha()]
        return sum(self.lo <= ord(c) <= self.hi for c in letters) / len(letters) if letters else 1.0

    def _is_vowel_sign(self, c):
        return self.vlo <= ord(c) <= self.vhi

    def _surface(self, before, piece):
        rules = []
        if piece:
            prev = before[-1] if before else " "
            if unicodedata.category(piece[0]).startswith("M") and not (prev.isalpha() or unicodedata.category(prev).startswith("M")):
                rules.append("broken_matra")
            elif self._is_vowel_sign(prev) and self._is_vowel_sign(piece[0]):
                rules.append("broken_matra")
            if re.search(r"[A-Za-z]", piece) and (not before.strip() or self._native_share(before) >= 0.5):
                rules.append("script_mix")
        words = (before + piece).split()
        if len(words) >= 8:
            last = tuple(words[-4:])
            if any(tuple(words[j:j + 4]) == last for j in range(len(words) - 7)):
                rules.append("repeat")
        return rules

    def _parse_rules(self, sentences):
        """sentences: list of word lists; returns rules fired on the last word of each."""
        if not sentences:
            return []
        doc = self.parser(sentences)
        out = []
        for sent in doc.sentences:
            last = {w.id for w in sent.tokens[-1].words}
            fired = []
            for e in find_errors(sent, self.lang):
                # the new word is the dependent, the head, or an aux/cop of the head (बिछा रहा है)
                aux = {w.id for w in sent.words if w.head == e["head"].id and w.deprel in ("aux", "aux:pass", "cop")}
                if last & ({e["dep"].id, e["head"].id} | aux):
                    fired.append(e["rule"])
            out.append(fired)
        return out

    def banned_ids(self, i):
        """After a script-mixing veto, exclude every Latin-script token, not just the drafted one."""
        if "script_mix" not in self.last_rules[i]:
            return []
        if self._latin is None:  # computed once; a tensor so masking ~100k ids is a single op
            import torch
            self._latin = torch.tensor([t for t in range(len(self.tok)) if re.search(r"[A-Za-z]", self.tok.decode([t]))])
        return self._latin

    # ---- Scorer API ----
    def score(self, prefix_ids, cand_ids):
        texts, pieces = decode_pieces(self.tok, list(prefix_ids[self.n_prompt:]), cand_ids)

        if self.groups == "grammar":
            rules = [[] for _ in cand_ids]
        else:
            rules = [self._surface(texts[i], pieces[i]) for i in range(len(cand_ids))]

        if self.parser is not None:
            idx, sents = word_end_sentences(texts, pieces)
            for i, r in zip(idx, self._parse_rules(sents)):
                rules[i] += r

        self.last_rules = rules
        L = []
        for r in rules:
            self.fired.update(r)
            L.append(min((SEVERITY[x] for x in r), default=1.0))
        return L
