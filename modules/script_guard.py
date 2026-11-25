"""Script guard: stop drift into English without banning English.

The linguistic modules only judge DRAFTED tokens. Tokens the target samples itself (the replacement
after a rejection and the bonus token) are never checked, and that is how Dolly answers drift into
English. The guard works one level lower: it reshapes the next-token distribution of BOTH models at
every position, so every emitted token is covered, and speculative decoding stays exact with respect
to the guarded target distribution.

Budget from the prompt:
    budget = max(base_budget, Latin share of the prompt + 0.1)
    An English prompt (Latin share ~1) gets budget >= 1: the guard is OFF and English is answered as before.
    Latin tokens that appear in the prompt (names, terms) are never penalized.

Modes, decided per position from the answer so far (native prompts only):
    start  fewer than `start_letters` native letters written yet -> Latin tokens BANNED.
           Drift happens at the start: once an answer starts in English the model is ~99% sure of each
           next English token and a soft penalty cannot turn it (seen on Dolly). Once it starts native,
           it mostly stays native.
    force  still in start, but more than `start_other` digits/symbols written: with Latin banned the
           model escaped into digit/symbol loops ("1,000,000,000,..."), so now every token without a
           native letter is banned (EOS excepted) until the answer starts in native script.
    hard   Latin share (with one more English word) > hard_factor x budget -> BANNED.
    soft   Latin share > budget -> Latin tokens x strength (a confident term like "API" still comes out).
    else   untouched: English words inside a native answer are fine.

"Latin" here means letters of any non-native script (English, but also Chinese, which Qwen can drift into).
Byte pieces (tokens that decode to U+FFFD, part of one character) are neutral: Qwen writes Indic letters with them.
"""
import math

import torch

from config import SCRIPT_RANGE

BAN = -1e4  # large negative instead of -inf: never NaN, even if every token were masked


class ScriptGuard:
    def __init__(self, tok, lang, vocab_size, budget=0.2, strength=0.05, smooth=10, start_letters=20, hard_factor=2.0,
                 start_other=8):
        self.tok, self.budget0, self.smooth = tok, budget, smooth
        self.start_letters, self.hard_factor, self.start_other = start_letters, hard_factor, start_other
        self.lo, self.hi = SCRIPT_RANGE[lang]
        latin, foreign = torch.zeros(vocab_size), torch.zeros(vocab_size)
        for t in range(len(tok)):
            s = tok.decode([t])
            if "\ufffd" in s:
                continue  # a byte piece of a longer character (Qwen splits Indic letters this way): neutral
            if not any(self.lo <= ord(c) <= self.hi for c in s):
                foreign[t] = 1.0  # no native letter
                if any(c.isalpha() for c in s):
                    latin[t] = 1.0  # letters of another script (Latin, or e.g. Chinese for Qwen), no native ones
        special = [t for t in tok.all_special_ids if t < vocab_size]
        if special:  # special tokens (EOS "</s>", "<|im_end|>") are never blocked: the answer must be able to stop
            latin[special] = 0.0
            foreign[special] = 0.0
        self.latin, self.foreign = latin, foreign
        self.log_s = math.log(strength)
        self.stats = {}

    def _count(self, text):
        """(letters of other scripts, native characters). "Latin" in this file means any non-native script."""
        native = sum(1 for c in text if self.lo <= ord(c) <= self.hi)
        latin = sum(1 for c in text if c.isalpha() and not self.lo <= ord(c) <= self.hi)
        return latin, native

    def _other(self, text):
        """Non-space characters that are neither Latin nor native letters (digits, symbols)."""
        return sum(1 for c in text if not c.isspace() and c != "\ufffd" and not c.isalpha()
                   and not self.lo <= ord(c) <= self.hi)

    def reset(self, prompt_ids):
        lat, nat = self._count(self.tok.decode(prompt_ids))
        self.budget = max(self.budget0, lat / max(lat + nat, 1) + 0.1)
        keep = torch.ones_like(self.latin)
        exempt = [t for t in set(prompt_ids) if t < len(keep) and self.latin[t]]
        if exempt:
            keep[exempt] = 0.0  # words copied from the prompt are fine
        mask = self.latin * keep
        self.pen = {"soft": mask * self.log_s, "ban": mask * BAN, "force": self.foreign * BAN}
        self._dev = {}
        self.stats = {"budget": round(self.budget, 3), "start": 0, "force": 0, "hard": 0, "soft": 0}

    def mode(self, gen_ids):
        if self.budget >= 1:
            return None
        text = self.tok.decode(gen_ids)
        lat, nat = self._count(text)
        if nat < self.start_letters:
            return "force" if self._other(text) > self.start_other else "start"
        share = (lat + 5) / (lat + nat + 5 + self.smooth)
        if share > self.hard_factor * self.budget:
            return "hard"
        return "soft" if share > self.budget else None

    def apply(self, logits, gen_ids):
        """Add the penalty in place to one row of logits (float tensor, any device)."""
        m = self.mode(gen_ids)
        if m is None:
            return
        self.stats[m] += 1
        key = ({"soft": "soft", "force": "force"}.get(m, "ban"), logits.device)
        if key not in self._dev:
            self._dev[key] = self.pen[key[0]].to(logits.device)
        pen = self._dev[key]
        n = min(logits.shape[-1], pen.shape[-1])
        logits[..., :n] += pen[:n]
