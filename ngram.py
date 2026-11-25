"""Tag n-gram model for module B: interpolated Kneser-Ney over POS+morph tags.

A tag is UPOS plus the agreement-relevant features that are present, e.g.
    VERB|Fem|Sing|3   NOUN|Masc|Plur|Acc   ADP
Written in plain Python (no KenLM build needed): the tag vocabulary is a few hundred symbols,
so counting 4-grams over ~1M sentences fits in memory and queries are dictionary lookups.
"""
import bisect
import math
import pickle
from collections import defaultdict

TAG_FEATS = ["Gender", "Number", "Person", "Case"]
BOS, EOS = "<s>", "</s>"


def word_tag(word):
    feats = dict(f.split("=", 1) for f in word.feats.split("|")) if word.feats else {}
    return "|".join([word.upos] + [feats[k] for k in TAG_FEATS if k in feats])


class KNLM:
    def __init__(self, order=4, discount=0.75):
        self.order, self.d = order, discount
        self.vocab = {}  # tag -> id
        self.levels = {}  # n -> {ctx tuple: {w: count}}
        self.stats = {}  # n -> {ctx tuple: (total, n_types)}
        self.calib = []  # sorted normalized() values of real held-out words (see calibrate)
        self._cache = {}

    def _id(self, tag, add=False):
        if tag not in self.vocab:
            if not add:
                return None
            self.vocab[tag] = len(self.vocab)
        return self.vocab[tag]

    def fit(self, sentences):
        """sentences: iterable of tag lists."""
        n = self.order
        bos, eos = self._id(BOS, True), self._id(EOS, True)
        top = defaultdict(lambda: defaultdict(int))
        for tags in sentences:
            seq = [bos] * (n - 1) + [self._id(t, True) for t in tags] + [eos]
            for i in range(n - 1, len(seq)):
                top[tuple(seq[i - n + 1:i])][seq[i]] += 1
        self.levels[n] = top
        for k in range(n - 1, 0, -1):  # continuation counts: distinct left extensions
            lower = defaultdict(lambda: defaultdict(int))
            for ctx, ws in self.levels[k + 1].items():
                for w in ws:
                    lower[ctx[1:]][w] += 1
            self.levels[k] = lower
        self.levels = {k: {c: dict(ws) for c, ws in lv.items()} for k, lv in self.levels.items()}
        self.stats = {k: {c: (sum(ws.values()), len(ws)) for c, ws in lv.items()} for k, lv in self.levels.items()}
        return self

    def _p(self, k, ctx, w):
        if k == 0:
            return 1.0 / len(self.vocab)
        ws = self.levels[k].get(ctx)
        if not ws:
            return self._p(k - 1, ctx[1:], w)
        total, types = self.stats[k][ctx]
        return max(ws.get(w, 0) - self.d, 0) / total + self.d * types / total * self._p(k - 1, ctx[1:], w)

    def _ctx(self, history):
        ids = [self._id(t) for t in history][-(self.order - 1):]
        ids = [self.vocab[BOS]] * (self.order - 1 - len(ids)) + ids
        return tuple(ids)

    def prob(self, history, tag):
        """P(tag | last order-1 tags of history); None if a tag was never seen in training."""
        w = self._id(tag)
        ctx = self._ctx(history)
        if w is None or None in ctx:
            return None
        return self._p(self.order, ctx, w)

    def normalized(self, history, tag):
        """P(tag | history) / max over tags: 1 for the most likely tag, near 0 for very unlikely ones."""
        p = self.prob(history, tag)
        if p is None:
            return None
        ctx = self._ctx(history)
        if ctx not in self._cache:
            self._cache[ctx] = max(self._p(self.order, ctx, v) for v in self.vocab.values() if v != self.vocab[BOS])
        return p / self._cache[ctx]

    def calibrate(self, sentences, max_values=20000):
        """Store the normalized() scores of real (held-out) words, so a score can be turned into a percentile."""
        vals = []
        for tags in sentences:
            for j, t in enumerate(tags):
                v = self.normalized(tags[:j], t)
                if v is not None:
                    vals.append(v)
        vals.sort()
        step = max(1, len(vals) // max_values)
        self.calib = vals[::step]
        return self

    def percentile(self, value):
        """Share of real words that score <= value: 0.01 = rarer than 99% of real text, 1 = as typical as it gets."""
        if not self.calib:
            return value
        return bisect.bisect_right(self.calib, value) / len(self.calib)

    def perplexity(self, sentences):
        logp, n = 0.0, 0
        for tags in sentences:
            hist = []
            for t in tags + [EOS]:
                p = self.prob(hist, t)
                if p is not None:
                    logp += math.log(p)
                    n += 1
                hist.append(t)
        return math.exp(-logp / max(n, 1))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"order": self.order, "d": self.d, "vocab": self.vocab,
                         "levels": self.levels, "stats": self.stats,
                         "calib": self.calib}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            s = pickle.load(f)
        lm = cls(s["order"], s["d"])
        lm.vocab, lm.levels, lm.stats = s["vocab"], s["levels"], s["stats"]
        lm.calib = s.get("calib", [])
        return lm
