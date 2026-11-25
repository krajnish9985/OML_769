import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import torch
from transformers import BloomConfig, BloomForCausalLM
from sd import SDConfig, SpeculativeDecoder, Scorer

torch.manual_seed(0)
V = 64
def model(layers, h):
    return BloomForCausalLM(BloomConfig(vocab_size=V, hidden_size=h, n_layer=layers, n_head=4)).eval()
target, draft = model(2, 64), model(1, 32)
EOS = 2
prompt = [5, 9, 11, 3]

# 1. greedy, no scorer == target greedy
for k in [1, 3, 5]:
    sd = SpeculativeDecoder(draft, target, EOS, None, SDConfig(k=k, temperature=0, max_new_tokens=30))
    r = sd.generate(prompt)
    ref = target.generate(torch.tensor([prompt]), max_new_tokens=30, do_sample=False, eos_token_id=EOS, pad_token_id=0)[0, 4:].tolist()
    ref = [t for t in ref if t != EOS]
    assert r.ids == ref, (k, r.ids, ref)
    print("greedy match k=%d" % k, r.stats()["acceptance_rate"])

# 2. draft == target -> everything accepted
sd = SpeculativeDecoder(target, target, EOS, None, SDConfig(k=4, temperature=0, max_new_tokens=20))
r = sd.generate(prompt); assert all(t["accepted"] for t in r.tokens), r.tokens; print("self-draft all accepted")

# 3. hard veto of a token that greedy output contains
ref_ids = SpeculativeDecoder(draft, target, EOS, None, SDConfig(k=3, temperature=0, max_new_tokens=30)).generate(prompt).ids
bad = ref_ids[0]
class Ban(Scorer):
    def score(self, prefix, cands): return [0.0 if c == bad else 1.0 for c in cands]
for variant in ["hard", "soft"]:
    for T in [0, 0.8]:
        sd = SpeculativeDecoder(draft, target, EOS, Ban(), SDConfig(k=3, temperature=T, max_new_tokens=30, variant=variant, alpha=1))
        for seed in range(5):
            r = sd.generate(prompt, seed=seed)
            # banned token can still come from correction/bonus sampling only if L never saw it; drafted ones must never be accepted
            assert not any(t["accepted"] and t["id"] == bad for t in r.tokens)
        print(variant, T, "ban ok", r.stats())

# 4. sampling: SD output distribution of first token ~ target distribution
T = 1.0
sd = SpeculativeDecoder(draft, target, EOS, None, SDConfig(k=3, temperature=T, max_new_tokens=1))
cnt = torch.zeros(V)
N = 3000
for s in range(N):
    ids = sd.generate(prompt, seed=s).ids; cnt[ids[0] if ids else EOS] += 1
p = torch.softmax(target(torch.tensor([prompt])).logits[0, -1] / T, -1)
tv = 0.5 * (cnt / N - p).abs().sum()
print("TV distance first token:", float(tv))
assert tv < 0.08
# 5. padded target vocab (like Qwen2.5 0.5B vs 7B) and two EOS ids: SD still equals target greedy
big = BloomForCausalLM(BloomConfig(vocab_size=V + 8, hidden_size=64, n_layer=2, n_head=4)).eval()
EOS2 = [EOS, 7]
r = SpeculativeDecoder(draft, big, EOS2, None, SDConfig(k=3, temperature=0, max_new_tokens=30)).generate(prompt)
x, ref = list(prompt), []  # greedy target over the shared first V ids (padding rows are cut)
with torch.no_grad():
    for _ in range(30):
        t = int(big(torch.tensor([x])).logits[0, -1, :V].argmax())
        if t in EOS2:
            break
        ref.append(t)
        x.append(t)
assert r.ids == ref, (r.ids, ref)
print("padded vocab + 2 EOS ids ok")
# 6. check_target: the banned token never appears in the output (target-sampled tokens are judged too),
#    except possibly in the last 2 positions (still pending when generation stops)
for T in [0, 0.8]:
    for seed in range(5):
        sd = SpeculativeDecoder(draft, target, EOS, Ban(), SDConfig(k=3, temperature=T, max_new_tokens=30, variant="hard",
                                                                    tau=0.5, check_target=True))
        r = sd.generate(prompt, seed=seed)
        assert bad not in r.ids[:-2], (T, seed, r.ids)
print("check_target ok", r.stats())
print("ALL OK")
