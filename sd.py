# """Speculative decoding with a linguistic score L in the verification step.

# Acceptance of drafted token x (p = target prob, q = draft prob):
#     standard : a = min(1, p(x)/q(x))
#     soft     : a = min(1, p(x)/q(x)) * L^alpha
#     hard     : a = 0 if L < tau else min(1, p(x)/q(x))
# On rejection the replacement is sampled from the residual max(0, p - q). If the rejection
# was caused by L, x itself (plus scorer.banned_ids) is removed from that distribution, so the veto sticks.
# With scorer=None (L = 1) this is standard speculative decoding (system S1).
# An optional guard (modules/script_guard.py) reshapes the draft AND target distributions at every
# position before sampling/verification, so it also covers tokens the target samples itself.

# temperature=0 gives greedy decoding: accept x iff x == argmax p (and the L check passes).
# """
# import time
# from dataclasses import dataclass, field

# import torch
# from transformers import DynamicCache


# class Scorer:
#     """Base class for modules A/B/C.

#     score(prefix_ids, cand_ids) returns one L in [0, 1] per drafted token:
#     L[i] judges cand_ids[i] given prefix_ids + cand_ids[:i].
#     """

#     def reset(self, prompt_ids: list[int]) -> None:
#         """Called once per prompt before generation (e.g. to know where the output starts)."""

#     def score(self, prefix_ids: list[int], cand_ids: list[int]) -> list[float]:
#         raise NotImplementedError

#     def banned_ids(self, i: int) -> list[int]:
#         """Extra token ids to exclude from the replacement when drafted token i (from the last
#         score() call) is vetoed, e.g. all Latin-script tokens after a script-mixing veto."""
#         return []


# @dataclass
# class SDConfig:
#     k: int = 4                  # draft tokens per step
#     temperature: float = 0.7    # 0 = greedy
#     max_new_tokens: int = 256
#     min_new_tokens: int = 0     # EOS is masked in both draft and target until this many tokens
#     variant: str = "soft"       # "soft" or "hard"
#     alpha: float = 1.0          # soft: exponent on L
#     tau: float = 0.3            # hard: veto threshold


# @dataclass
# class SDResult:
#     ids: list[int]                                # generated token ids (prompt excluded)
#     tokens: list[dict] = field(default_factory=list)  # one entry per drafted token
#     n_steps: int = 0
#     time_total: float = 0.0
#     time_scorer: float = 0.0

#     def stats(self) -> dict:
#         n = len(self.tokens)
#         acc = sum(t["accepted"] for t in self.tokens)
#         ling = sum(t["reason"] == "ling" for t in self.tokens)
#         return {
#             "n_new": len(self.ids),
#             "n_steps": self.n_steps,
#             "n_drafted": n,
#             "acceptance_rate": acc / n if n else 0.0,
#             "ling_rejections": ling,
#             "tokens_per_step": len(self.ids) / self.n_steps if self.n_steps else 0.0,
#             "tok_per_s": len(self.ids) / self.time_total if self.time_total else 0.0,
#             "scorer_time_per_step": self.time_scorer / self.n_steps if self.n_steps else 0.0,
#         }


# def _probs(logits, temperature):
#     logits = logits.float()
#     if temperature == 0:
#         return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
#     return torch.softmax(logits / temperature, dim=-1)


# def _sample(probs, gen):
#     return int(torch.multinomial(probs, 1, generator=gen))


# class _Model:
#     """Wraps a causal LM with a KV cache that can be rolled back."""

#     def __init__(self, model, vocab=None):
#         self.model, self.vocab = model, vocab  # vocab: keep only the first `vocab` logits
#         self.cache = DynamicCache()

#     def reset(self):
#         self.cache = DynamicCache()

#     def __len__(self):
#         return self.cache.get_seq_length()

#     @torch.no_grad()
#     def forward(self, new_ids):
#         """Feed tokens not yet in the cache; return logits for each of them."""
#         x = torch.tensor([new_ids], device=self.model.device)
#         out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
#         self.cache = out.past_key_values
#         return out.logits[0, :, : self.vocab] if self.vocab else out.logits[0]

#     def crop(self, n):
#         if len(self) > n:
#             self.cache.crop(n)


# class SpeculativeDecoder:
#     def __init__(self, draft, target, eos_id, scorer: Scorer | None = None, cfg: SDConfig = SDConfig(), guard=None):
#         """eos_id: one id or a list (Qwen instruct stops on <|im_end|> and <|endoftext|>).
#         Draft and target logits are cut to the smaller vocab size: same tokenizer, but the embedding
#         matrices can carry different padding (Qwen2.5 0.5B 151936 vs 7B 152064 rows)."""
#         size = lambda m: m.get_output_embeddings().weight.shape[0]
#         vocab = min(size(draft), size(target))
#         self.draft, self.target = _Model(draft, vocab), _Model(target, vocab)
#         self.eos = [eos_id] if isinstance(eos_id, int) else list(eos_id)
#         self.scorer, self.cfg, self.guard = scorer, cfg, guard

#     def _accept_probs(self, p_x, q_x, L):
#         """Return (a_base, a_final): acceptance prob without and with the linguistic score."""
#         c = self.cfg
#         if c.temperature == 0:
#             a = float(p_x == 1)
#         else:
#             a = min(1.0, p_x / q_x) if q_x > 0 else 0.0
#         if self.scorer is None:
#             return a, a
#         if c.variant == "hard":
#             return a, 0.0 if L < c.tau else a
#         return a, a * L ** c.alpha

#     @torch.no_grad()
#     def generate(self, prompt_ids: list[int], seed: int = 0, user_ids: list[int] | None = None) -> SDResult:
#         """user_ids: the user's own prompt without the chat template, for the guard's budget."""
#         c = self.cfg
#         gen = torch.Generator(device="cpu").manual_seed(seed)
#         self.draft.reset()
#         self.target.reset()
#         ids = list(prompt_ids)
#         n_prompt = len(ids)
#         if self.scorer is not None:
#             self.scorer.reset(ids)
#         if self.guard is not None:
#             self.guard.reset(user_ids if user_ids is not None else ids)
#         res = SDResult(ids=[])
#         t0 = time.time()

#         while len(ids) - n_prompt < c.max_new_tokens:
#             k = min(c.k, c.max_new_tokens - (len(ids) - n_prompt))
#             n0 = len(ids)

#             # 1. draft k tokens
#             drafted, q = [], []
#             feed = ids[len(self.draft):]
#             for j in range(k):
#                 lg = self.draft.forward(feed)[-1].float()
#                 if n0 + j - n_prompt < c.min_new_tokens:
#                     lg[self.eos] = float("-inf")
#                 if self.guard is not None:
#                     self.guard.apply(lg, ids[n_prompt:] + drafted)
#                 q_i = _probs(lg, c.temperature).cpu()
#                 x = _sample(q_i, gen)
#                 drafted.append(x)
#                 q.append(q_i)
#                 feed = [x]

#             # 2. target scores all drafted positions (+1 bonus) in one pass
#             logits = self.target.forward(ids[len(self.target):] + drafted)[-(k + 1):].float()
#             n_block = max(0, min(k + 1, c.min_new_tokens - (n0 - n_prompt)))
#             logits[:n_block, self.eos] = float("-inf")
#             if self.guard is not None:
#                 for i in range(k + 1):
#                     self.guard.apply(logits[i], ids[n_prompt:] + drafted[:i])
#             p = _probs(logits, c.temperature).cpu()

#             # 3. linguistic scores
#             if self.scorer is not None:
#                 ts = time.time()
#                 L = self.scorer.score(ids, drafted)
#                 res.time_scorer += time.time() - ts
#             else:
#                 L = [1.0] * k

#             # 4. verify left to right
#             new, n_acc = [], 0
#             for i, x in enumerate(drafted):
#                 p_x, q_x = float(p[i, x]), float(q[i][x])
#                 a_base, a = self._accept_probs(p_x, q_x, L[i])
#                 u = torch.rand(1, generator=gen).item()
#                 ok = u < a
#                 ling = not ok and u < a_base  # standard SD would have accepted: L caused the rejection
#                 res.tokens.append({"pos": n0 + i - n_prompt, "id": x, "p": p_x, "q": q_x, "L": L[i],
#                                    "accepted": ok, "reason": "ok" if ok else "ling" if ling else "prob"})
#                 if ok:
#                     new.append(x)
#                     n_acc += 1
#                     if x in self.eos:
#                         break
#                     continue
#                 veto = ling or (self.scorer is not None and c.variant == "hard" and L[i] < c.tau)
#                 extra = self.scorer.banned_ids(i) if veto else []  # list or LongTensor of ids

#                 if c.temperature == 0:  # greedy: best target token (excluding banned tokens if vetoed)
#                     lg = logits[i].cpu()
#                     if veto:
#                         lg[x] = float("-inf")
#                         if len(extra):
#                             lg[extra] = float("-inf")
#                     new.append(int(lg.argmax()))
#                 else:
#                     resid = (p[i] - q[i]).clamp(min=0)
#                     if veto:
#                         resid[x] = 0
#                         if len(extra):
#                             resid[extra] = 0
#                     if resid.sum() <= 0:  # nothing left: fall back to target distribution
#                         resid = p[i].clone()
#                         if veto:
#                             resid[x] = 0
#                             if len(extra):
#                                 resid[extra] = 0
#                         if resid.sum() <= 0:
#                             resid = p[i].clone()
#                     new.append(_sample(resid / resid.sum(), gen))
#                 break
#             else:
#                 new.append(_sample(p[k], gen))  # all accepted: bonus token from target

#             # 5. roll back caches to the accepted prefix
#             self.target.crop(n0 + n_acc)
#             self.draft.crop(n0 + n_acc)
#             res.n_steps += 1

#             stop = [j for j, t in enumerate(new) if t in self.eos]
#             if stop:
#                 new = new[: stop[0] + 1]
#             ids += new
#             if ids[-1] in self.eos:
#                 break

#         ids = ids[: n_prompt + c.max_new_tokens]
#         res.ids = [t for t in ids[n_prompt:] if t not in self.eos]
#         res.time_total = time.time() - t0
#         return res

"""Speculative decoding with a linguistic score L in the verification step.

Acceptance of drafted token x (p = target prob, q = draft prob):
    standard : a = min(1, p(x)/q(x))
    soft     : a = min(1, p(x)/q(x)) * L^alpha
    hard     : a = 0 if L < tau else min(1, p(x)/q(x))
On rejection the replacement is sampled from the residual max(0, p - q). If the rejection
was caused by L, x itself (plus scorer.banned_ids) is removed from that distribution, so the veto sticks.
With scorer=None (L = 1) this is standard speculative decoding (system S1).

check_target=True also judges the tokens the TARGET samples itself (the replacement after a rejection
and the bonus token), because we do not trust the target's Indic grammar either. Modules judge a word
only once the next token shows it has ended, so these tokens are judged at the start of the next step,
together with the new draft block (which gives the lookahead). The last two tokens of each block are
re-judged this way: the target token, and the token before it (its lookahead changed or was missing).
If one fails (hard: L < tau; soft: kept with prob L^alpha), generation rolls back to it and a new token
is sampled from the TARGET distribution at that position with the failed token (and banned_ids) removed;
the new token is judged again, at most max_target_retries times per position (then it is kept).
check_target=False (default) gives exactly the old behaviour.
An optional guard (modules/script_guard.py) reshapes the draft AND target distributions at every
position before sampling/verification, so it also covers tokens the target samples itself.

temperature=0 gives greedy decoding: accept x iff x == argmax p (and the L check passes).
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


class Scorer:
    """Base class for modules A/B/C.

    score(prefix_ids, cand_ids) returns one L in [0, 1] per drafted token:
    L[i] judges cand_ids[i] given prefix_ids + cand_ids[:i].
    """

    def reset(self, prompt_ids: list[int]) -> None:
        """Called once per prompt before generation (e.g. to know where the output starts)."""

    def score(self, prefix_ids: list[int], cand_ids: list[int]) -> list[float]:
        raise NotImplementedError

    def banned_ids(self, i: int) -> list[int]:
        """Extra token ids to exclude from the replacement when drafted token i (from the last
        score() call) is vetoed, e.g. all Latin-script tokens after a script-mixing veto."""
        return []


@dataclass
class SDConfig:
    k: int = 4                  # draft tokens per step
    temperature: float = 0.7    # 0 = greedy
    max_new_tokens: int = 256
    min_new_tokens: int = 0     # EOS is masked in both draft and target until this many tokens
    variant: str = "soft"       # "soft" or "hard"
    alpha: float = 1.0          # soft: exponent on L
    tau: float = 0.3            # hard: veto threshold
    check_target: bool = False  # also judge target-sampled tokens (replacement, bonus); see module doc
    max_target_retries: int = 3  # per position, then the target token is kept


@dataclass
class SDResult:
    ids: list[int]                                # generated token ids (prompt excluded)
    tokens: list[dict] = field(default_factory=list)  # one entry per drafted token
    n_steps: int = 0
    time_total: float = 0.0
    time_scorer: float = 0.0
    target_checks: list[dict] = field(default_factory=list)  # one entry per re-judged target-side token

    def stats(self) -> dict:
        n = len(self.tokens)
        acc = sum(t["accepted"] for t in self.tokens)
        ling = sum(t["reason"] == "ling" for t in self.tokens)
        return {
            "n_new": len(self.ids),
            "n_steps": self.n_steps,
            "n_drafted": n,
            "acceptance_rate": acc / n if n else 0.0,
            "ling_rejections": ling,
            "tokens_per_step": len(self.ids) / self.n_steps if self.n_steps else 0.0,
            "tok_per_s": len(self.ids) / self.time_total if self.time_total else 0.0,
            "scorer_time_per_step": self.time_scorer / self.n_steps if self.n_steps else 0.0,
            "target_checked": len(self.target_checks),
            "target_vetoes": sum(not t["accepted"] for t in self.target_checks),
            "target_forced": sum(t["forced"] for t in self.target_checks),
        }


def _probs(logits, temperature):
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs, gen):
    return int(torch.multinomial(probs, 1, generator=gen))


class _Model:
    """Wraps a causal LM with a KV cache that can be rolled back."""

    def __init__(self, model, vocab=None):
        self.model, self.vocab = model, vocab  # vocab: keep only the first `vocab` logits
        self.cache = DynamicCache()

    def reset(self):
        self.cache = DynamicCache()

    def __len__(self):
        return self.cache.get_seq_length()

    @torch.no_grad()
    def forward(self, new_ids):
        """Feed tokens not yet in the cache; return logits for each of them."""
        x = torch.tensor([new_ids], device=self.model.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out.logits[0, :, : self.vocab] if self.vocab else out.logits[0]

    def crop(self, n):
        if len(self) > n:
            self.cache.crop(n)


class SpeculativeDecoder:
    def __init__(self, draft, target, eos_id, scorer: Scorer | None = None, cfg: SDConfig = SDConfig(), guard=None):
        """eos_id: one id or a list (Qwen instruct stops on <|im_end|> and <|endoftext|>).
        Draft and target logits are cut to the smaller vocab size: same tokenizer, but the embedding
        matrices can carry different padding (Qwen2.5 0.5B 151936 vs 7B 152064 rows)."""
        size = lambda m: m.get_output_embeddings().weight.shape[0]
        vocab = min(size(draft), size(target))
        self.draft, self.target = _Model(draft, vocab), _Model(target, vocab)
        self.eos = [eos_id] if isinstance(eos_id, int) else list(eos_id)
        self.scorer, self.cfg, self.guard = scorer, cfg, guard

    def _accept_probs(self, p_x, q_x, L):
        """Return (a_base, a_final): acceptance prob without and with the linguistic score."""
        c = self.cfg
        if c.temperature == 0:
            a = float(p_x == 1)
        else:
            a = min(1.0, p_x / q_x) if q_x > 0 else 0.0
        if self.scorer is None:
            return a, a
        if c.variant == "hard":
            return a, 0.0 if L < c.tau else a
        return a, a * L ** c.alpha

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], seed: int = 0, user_ids: list[int] | None = None) -> SDResult:
        """user_ids: the user's own prompt without the chat template, for the guard's budget."""
        c = self.cfg
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.draft.reset()
        self.target.reset()
        ids = list(prompt_ids)
        n_prompt = len(ids)
        if self.scorer is not None:
            self.scorer.reset(ids)
        if self.guard is not None:
            self.guard.reset(user_ids if user_ids is not None else ids)
        res = SDResult(ids=[])
        t0 = time.time()
        check = c.check_target and self.scorer is not None
        pend = []  # tail tokens of ids still to be judged: {"pos", "logits" (target row), "banned", "tries"}

        while len(ids) - n_prompt < c.max_new_tokens:
            k = min(c.k, c.max_new_tokens - (len(ids) - n_prompt))
            n0 = len(ids)

            # 1. draft k tokens
            drafted, q = [], []
            feed = ids[len(self.draft):]
            for j in range(k):
                lg = self.draft.forward(feed)[-1].float()
                if n0 + j - n_prompt < c.min_new_tokens:
                    lg[self.eos] = float("-inf")
                if self.guard is not None:
                    self.guard.apply(lg, ids[n_prompt:] + drafted)
                q_i = _probs(lg, c.temperature).cpu()
                x = _sample(q_i, gen)
                drafted.append(x)
                q.append(q_i)
                feed = [x]

            # 2. target scores all drafted positions (+1 bonus) in one pass
            logits = self.target.forward(ids[len(self.target):] + drafted)[-(k + 1):].float()
            n_block = max(0, min(k + 1, c.min_new_tokens - (n0 - n_prompt)))
            logits[:n_block, self.eos] = float("-inf")
            if self.guard is not None:
                for i in range(k + 1):
                    self.guard.apply(logits[i], ids[n_prompt:] + drafted[:i])
            p = _probs(logits, c.temperature).cpu()

            # 3. linguistic scores (with check_target: the pending tail tokens first, then the drafts)
            start = pend[0]["pos"] if pend else len(ids)
            m_p = len(ids) - start
            if self.scorer is not None:
                ts = time.time()
                L_all = self.scorer.score(ids[:start], ids[start:] + drafted)
                res.time_scorer += time.time() - ts
            else:
                L_all = [1.0] * (m_p + k)
            L = L_all[m_p:]

            # 3b. judge the pending target-side tokens; on a failure roll back and resample from the target
            vetoed = False
            for j, e in enumerate(pend):
                x, l = ids[e["pos"]], L_all[j]
                fail = l < c.tau if c.variant == "hard" else torch.rand(1, generator=gen).item() >= l ** c.alpha
                forced = fail and e["tries"] >= c.max_target_retries  # out of retries: kept although it failed
                fail = fail and not forced
                res.target_checks.append({"pos": e["pos"] - n_prompt, "id": x, "L": l, "accepted": not fail,
                                          "tries": e["tries"], "forced": forced})
                if not fail:
                    continue
                extra = self.scorer.banned_ids(j)
                banned = e["banned"] + [x] + [int(t) for t in extra]
                lg = e["logits"].clone()
                lg[banned] = float("-inf")
                if torch.isinf(lg).all():  # everything banned: keep the plain target distribution
                    lg = e["logits"].clone()
                    lg[x] = float("-inf")
                y = int(lg.argmax()) if c.temperature == 0 else _sample(_probs(lg, c.temperature), gen)
                ids = ids[: e["pos"]] + [y]
                self.target.crop(e["pos"])
                self.draft.crop(e["pos"])
                pend = [{"pos": e["pos"], "logits": e["logits"], "banned": banned, "tries": e["tries"] + 1}]
                res.n_steps += 1
                vetoed = True
                break
            if vetoed:
                if ids[-1] in self.eos:
                    break
                continue
            pend = []

            # 4. verify left to right
            new, n_acc = [], 0
            for i, x in enumerate(drafted):
                p_x, q_x = float(p[i, x]), float(q[i][x])
                a_base, a = self._accept_probs(p_x, q_x, L[i])
                u = torch.rand(1, generator=gen).item()
                ok = u < a
                ling = not ok and u < a_base  # standard SD would have accepted: L caused the rejection
                res.tokens.append({"pos": n0 + i - n_prompt, "id": x, "p": p_x, "q": q_x, "L": L[i],
                                   "accepted": ok, "reason": "ok" if ok else "ling" if ling else "prob"})
                if ok:
                    new.append(x)
                    n_acc += 1
                    if x in self.eos:
                        break
                    continue
                veto = ling or (self.scorer is not None and c.variant == "hard" and L[i] < c.tau)
                extra = self.scorer.banned_ids(m_p + i) if veto else []  # list or LongTensor of ids

                if c.temperature == 0:  # greedy: best target token (excluding banned tokens if vetoed)
                    lg = logits[i].cpu()
                    if veto:
                        lg[x] = float("-inf")
                        if len(extra):
                            lg[extra] = float("-inf")
                    new.append(int(lg.argmax()))
                else:
                    resid = (p[i] - q[i]).clamp(min=0)
                    if veto:
                        resid[x] = 0
                        if len(extra):
                            resid[extra] = 0
                    if resid.sum() <= 0:  # nothing left: fall back to target distribution
                        resid = p[i].clone()
                        if veto:
                            resid[x] = 0
                            if len(extra):
                                resid[extra] = 0
                        if resid.sum() <= 0:
                            resid = p[i].clone()
                    new.append(_sample(resid / resid.sum(), gen))
                break
            else:
                new.append(_sample(p[k], gen))  # all accepted: bonus token from target

            # 5. roll back caches to the accepted prefix
            self.target.crop(n0 + n_acc)
            self.draft.crop(n0 + n_acc)
            res.n_steps += 1

            stop = [j for j, t in enumerate(new) if t in self.eos]
            if stop:
                new = new[: stop[0] + 1]
            elif check:  # last two tokens of the block (the target-sampled one and the one before it)
                pend = [{"pos": n0 + j, "logits": logits[j].cpu(), "banned": [], "tries": 0}
                        for j in range(max(0, len(new) - 2), len(new))]
            ids += new
            if ids[-1] in self.eos:
                break

        ids = ids[: n_prompt + c.max_new_tokens]
        res.ids = [t for t in ids[n_prompt:] if t not in self.eos]
        res.time_total = time.time() - t0
        return res
