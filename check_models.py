"""Download draft/target of the current pair, check they share a tokenizer, run a Hindi SD test.

    python check_models.py --gpu 2               # BLOOMZ pair (default)
    PAIR=qwen python check_models.py --gpu 2     # Qwen2.5 0.5B -> 7B Instruct
    python check_models.py --8bit                # load target in 8-bit (needs bitsandbytes)

Prints: tokens per word per language, embedding sizes, EOS ids,
then greedy target vs greedy SD with our own loop (sd.py). They match for the first tokens and can
then split at a near-tie: SD scores k+1 tokens per pass, the target alone one, and bf16 rounds the
two differently. tests/test_sd.py (float32) is the exact check.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ANSWER_HINT, DRAFT, DTYPE, GEN_LEN, PAIR, TARGET, build_prompt, eos_ids
from sd import SDConfig, SpeculativeDecoder

PROMPT = "भारत की राजधानी के बारे में 5-6 वाक्यों में बताइए।"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--8bit", dest="eightbit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_LEN[1])
    ap.add_argument("--gpu", type=int, default=0, help="GPU index; both models go on this one GPU")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: reinstall torch with the wheel matching your driver (README step 2)")
    print(f"pair {PAIR}: draft {DRAFT}, target {TARGET}, gen_len {GEN_LEN}")

    tok_d = AutoTokenizer.from_pretrained(DRAFT)
    tok = AutoTokenizer.from_pretrained(TARGET)
    assert tok_d.get_vocab() == tok.get_vocab(), "draft and target vocabularies differ"
    ids = build_prompt(tok, PROMPT)
    assert ids == build_prompt(tok_d, PROMPT)
    print(f"tokenizer OK: vocab {len(tok)}, prompt -> {len(ids)} tokens (with chat template if any)")
    for lang, text in [("hin", PROMPT)] + [(l, t) for l, t in ANSWER_HINT.items() if l != "hin"]:
        n = len(tok(text, add_special_tokens=False).input_ids)
        print(f"  {lang}: {n / len(text.split()):.1f} tokens per word")

    device = f"cuda:{args.gpu}"
    kw = dict(device_map={"": device}, dtype=getattr(torch, DTYPE))
    if args.eightbit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET, **kw).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, device_map={"": device}, dtype=getattr(torch, DTYPE)).eval()
    for name, m in [("draft", draft), ("target", target)]:
        print(f"{name}: output embeddings {m.get_output_embeddings().weight.shape[0]} (cut to the smaller one), "
              f"device {m.device}")
    eos = eos_ids(tok, target)
    print(f"EOS ids: {eos} {tok.convert_ids_to_tokens(eos)}")

    # greedy target alone, with the same cut vocab and EOS set as SD
    vocab = min(m.get_output_embeddings().weight.shape[0] for m in (draft, target))
    sd = SpeculativeDecoder(draft, target, eos, None, SDConfig(k=4, temperature=0, max_new_tokens=args.max_new_tokens))

    @torch.no_grad()
    def target_greedy():
        x = torch.tensor([ids], device=device)
        out = target.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                              do_sample=False, eos_token_id=eos, pad_token_id=eos[0],
                              suppress_tokens=list(range(vocab, target.get_output_embeddings().weight.shape[0])) or None)
        return [t for t in out[0, len(ids):].tolist() if t not in eos]

    for _ in range(2):  # first round = warm-up
        torch.cuda.synchronize(); t = time.time()
        ref = target_greedy()
        torch.cuda.synchronize(); t_ar = time.time() - t
        t = time.time()
        res = sd.generate(ids)
        torch.cuda.synchronize(); t_sd = time.time() - t

    print(f"\nprompt: {PROMPT}\noutput: {tok.decode(res.ids, skip_special_tokens=True)}\n")
    print(f"generated tokens: {len(res.ids)}  ({len(res.ids) / max(len(tok.decode(res.ids).split()), 1):.1f} per word)")
    print(f"target alone : {len(ref) / t_ar:.1f} tok/s")
    print(f"our SD       : {len(res.ids) / t_sd:.1f} tok/s, acceptance {res.stats()['acceptance_rate']:.2f}")
    # Greedy SD must reproduce greedy target output; small fp16 drift can break exact equality late on.
    same = next((i for i, (a, b) in enumerate(zip(ref, res.ids)) if a != b), min(len(ref), len(res.ids)))
    print(f"greedy outputs identical: {ref == res.ids}  (first {same} tokens match)")


if __name__ == "__main__":
    main()
