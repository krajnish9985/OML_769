import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Model pair, picked with the PAIR environment variable (default bloomz):
#     export PAIR=qwen      # every script then uses the Qwen pair and results_qwen/
# chat:     wrap each prompt in the model's chat template (instruct models)
# gen_len:  (min_new_tokens, max_new_tokens) for Aya/Dolly, in BLOOM tokens.
# match_len: Qwen splits Indic text into many more tokens than BLOOM (hin 4.6, mar 7.0, tam 11.8 tokens
#           per word), so run_generation.py scales gen_len per data/language file by
#           (Qwen tokens / BLOOM tokens) over that file's prompts: answers get the same length in text.
# dtype:    Qwen2.5 is trained in bf16 and can overflow in fp16; BLOOMZ stays fp16 as in all earlier runs.
PAIRS = {
    "bloomz": dict(draft="bigscience/bloomz-560m", target="bigscience/bloomz-7b1-mt", chat=False, gen_len=(64, 256), match_len=False,
                   dtype="float16"),
    "qwen": dict(draft="Qwen/Qwen2.5-0.5B-Instruct", target="Qwen/Qwen2.5-7B-Instruct", chat=True, gen_len=(64, 256), match_len=True,
                 dtype="bfloat16"),
}
PAIR = os.environ.get("PAIR", "bloomz")
if PAIR not in PAIRS:
    raise SystemExit(f"PAIR={PAIR!r}: must be one of {sorted(PAIRS)}")
DRAFT, TARGET, CHAT, GEN_LEN, MATCH_LEN, DTYPE = (PAIRS[PAIR][k] for k in
                                                  ("draft", "target", "chat", "gen_len", "match_len", "dtype"))
RESULTS = ROOT / ("results" if PAIR == "bloomz" else f"results_{PAIR}")  # BLOOM results are never overwritten


def build_prompt(tok, text):
    """Token ids of one prompt: raw text (BLOOMZ) or the chat template with one user turn (Qwen)."""
    if CHAT:
        text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False).input_ids
    return tok(text).input_ids


def eos_ids(tok, model):
    """Every id that ends generation (Qwen instruct: <|im_end|> and <|endoftext|>)."""
    ids = {tok.eos_token_id}
    gen = getattr(model, "generation_config", None)
    extra = getattr(gen, "eos_token_id", None)
    ids.update(extra if isinstance(extra, list) else [extra])
    return sorted(i for i in ids if i is not None)

LANGS = ["hin", "mar", "tam"]
LANG_2 = {"hin": "hi", "mar": "mr", "tam": "ta"}
LANG_NAME = {"hin": "Hindi", "mar": "Marathi", "tam": "Tamil"}

# "Answer in 5-6 sentences", appended to Aya prompts
ANSWER_HINT = {
    "hin": "कृपया 5-6 वाक्यों में उत्तर दें।",
    "mar": "कृपया 5-6 वाक्यांमध्ये उत्तर द्या.",
    "tam": "தயவுசெய்து 5-6 வாக்கியங்களில் பதிலளிக்கவும்.",
}

# Unicode block per language, used to filter IndicCorp lines
SCRIPT_RANGE = {"hin": (0x0900, 0x097F), "mar": (0x0900, 0x097F), "tam": (0x0B80, 0x0BFF)}

UD = {  # lang: (repo, file prefix)
    "hin": ("UD_Hindi-HDTB", "hi_hdtb"),
    "mar": ("UD_Marathi-UFAL", "mr_ufal"),
    "tam": ("UD_Tamil-TTB", "ta_ttb"),
}
UD_TAG = "r2.15"

SEED = 0
