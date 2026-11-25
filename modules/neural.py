"""Module C: neural linguistic score L from a fine-tuned classifier (system S4).

When a drafted token ends a word (same detection as modules A and B), the sentence so far is
scored by the classifier from train_classifier.py:
    L = P(grammatical | sentence prefix up to and including the new word)
All word ends in one draft block go to the classifier in a single batch.
Tokens that do not end a word get L = 1.
"""
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DATA
from modules.common import decode_pieces, word_end_sentences
from sd import Scorer


class NeuralScorer(Scorer):
    def __init__(self, tok, lang, model_path=None, max_len=128):
        path = model_path or DATA / "classifier"
        self.tok, self.lang, self.max_len = tok, lang, max_len
        self.clf_tok = AutoTokenizer.from_pretrained(path)
        self.clf_tok.truncation_side = "left"  # keep the last word
        self.device = f"cuda:{torch.cuda.current_device()}"
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16).to(self.device).eval()
        self.n_prompt, self.fired = 0, Counter()

    def reset(self, prompt_ids):
        self.n_prompt, self.fired = len(prompt_ids), Counter()

    @torch.no_grad()
    def _scores(self, sentences):
        enc = self.clf_tok([" ".join(s) for s in sentences], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
        return torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].tolist()

    def score(self, prefix_ids, cand_ids):
        texts, pieces = decode_pieces(self.tok, list(prefix_ids[self.n_prompt:]), cand_ids)
        L = [1.0] * len(cand_ids)
        idx, sents = word_end_sentences(texts, pieces)
        if not sents:
            return L
        for i, l in zip(idx, self._scores(sents)):
            L[i] = l
            self.fired["checked"] += 1
            for t in (0.5, 0.2, 0.1, 0.05):
                if l < t:
                    self.fired[f"L<{t}"] += 1
        return L
