"""Module B: statistical linguistic score L from a tag 4-gram model (system S3).

When a drafted token ends a word (same detection as module A), the current sentence is POS+morph
tagged with Stanza and the new word's tag is scored given the 3 tags before it:
    L = P(tag | previous 3 tags) / max_t P(t | previous 3 tags)        (ngram.KNLM.normalized)
1 means the most likely tag in that context, near 0 a very unlikely one. If the word splits into
several syntactic words (Tamil MWT), L is the minimum over them. Tokens that do not end a word,
and tags never seen in training, get L = 1.
"""
from collections import Counter

from agreement import load_stanza
from config import DATA, LANG_2
from modules.common import decode_pieces, word_end_sentences
from ngram import KNLM, word_tag
from sd import Scorer


class StatScorer(Scorer):
    def __init__(self, tok, lang, model_path=None, use_gpu=True):
        self.tok, self.lang = tok, lang
        self.lm = KNLM.load(model_path or DATA / "ngram" / f"{lang}.pkl")
        self.tagger = load_stanza(LANG_2[lang], use_gpu, pretokenized=True, pos_only=True)
        self.n_prompt, self.fired = 0, Counter()

    def reset(self, prompt_ids):
        self.n_prompt, self.fired = len(prompt_ids), Counter()

    def _scores(self, sentences):
        """L for the last word of each sentence (word lists)."""
        if not sentences:
            return []
        out = []
        for sent in self.tagger(sentences).sentences:
            tags = [word_tag(w) for w in sent.words]
            n_last = len(sent.tokens[-1].words)
            ls = []
            for j in range(len(tags) - n_last, len(tags)):
                l = self.lm.normalized(tags[:j], tags[j])
                if l is not None:
                    ls.append(l)
            self.fired["checked"] += 1
            if not ls:
                self.fired["unseen_tag"] += 1
            out.append(min(ls, default=1.0))
        return out

    def score(self, prefix_ids, cand_ids):
        texts, pieces = decode_pieces(self.tok, list(prefix_ids[self.n_prompt:]), cand_ids)
        L = [1.0] * len(cand_ids)
        idx, sents = word_end_sentences(texts, pieces)
        for i, l in zip(idx, self._scores(sents)):
            L[i] = l
            for t in (0.5, 0.3, 0.1):
                if l < t:
                    self.fired[f"L<{t}"] += 1
        return L
"""Module B: statistical linguistic score L from a tag 4-gram model (system S3).

When a drafted token ends a word (same detection as module A), the current sentence is POS+morph
tagged with Stanza and the new word's tag is scored given the 3 tags before it:
    raw = P(tag | previous 3 tags) / max_t P(t | previous 3 tags)      (ngram.KNLM.normalized)
    L   = share of real held-out words whose raw score is <= raw       (ngram.KNLM.percentile)
So L = 0.01 means "this tag is less likely than for 99% of words in real text". The raw ratio alone
was too harsh: about half of all words (also correct ones) got raw < 0.5, because the most likely
tag is often not the only good one. Use the hard variant: tau is then the share of real text that
would be vetoed (tau 0.01 -> 1%). If the .pkl has no calibration (old model), L = raw.
If the word splits into several syntactic words (Tamil MWT), L is the minimum over them.
Tokens that do not end a word, and tags never seen in training, get L = 1.
"""
from collections import Counter

from agreement import load_stanza
from config import DATA, LANG_2
from modules.common import decode_pieces, word_end_sentences
from ngram import KNLM, word_tag
from sd import Scorer


class StatScorer(Scorer):
    def __init__(self, tok, lang, model_path=None, use_gpu=True):
        self.tok, self.lang = tok, lang
        self.lm = KNLM.load(model_path or DATA / "ngram" / f"{lang}.pkl")
        self.tagger = load_stanza(LANG_2[lang], use_gpu, pretokenized=True, pos_only=True)
        self.n_prompt, self.fired = 0, Counter()

    def reset(self, prompt_ids):
        self.n_prompt, self.fired = len(prompt_ids), Counter()

    def _scores(self, sentences):
        """L for the last word of each sentence (word lists)."""
        if not sentences:
            return []
        out = []
        for sent in self.tagger(sentences).sentences:
            tags = [word_tag(w) for w in sent.words]
            n_last = len(sent.tokens[-1].words)
            ls = []
            for j in range(len(tags) - n_last, len(tags)):
                l = self.lm.normalized(tags[:j], tags[j])
                if l is not None:
                    ls.append(self.lm.percentile(l))
            self.fired["checked"] += 1
            if not ls:
                self.fired["unseen_tag"] += 1
            out.append(min(ls, default=1.0))
        return out

    def score(self, prefix_ids, cand_ids):
        texts, pieces = decode_pieces(self.tok, list(prefix_ids[self.n_prompt:]), cand_ids)
        L = [1.0] * len(cand_ids)
        idx, sents = word_end_sentences(texts, pieces)
        for i, l in zip(idx, self._scores(sents)):
            L[i] = l
            for t in (0.1, 0.05, 0.01):
                if l < t:
                    self.fired[f"L<{t}"] += 1
        return L
