# Setup: environment, models, data

## What each file does

| File | Contents |
|---|---|
| `environment.yml` | Conda env `lingsd`: Python 3.10 + transformers, datasets, stanza, sacrebleu etc. No PyTorch (installed in step 2). |
| `config.py` | Shared settings: model pair (`PAIR` env var: `bloomz` default, `qwen`), model names, languages (hin/mar/tam), data folder, "answer in 5-6 sentences" text per language. |
| `check_models.py` | Downloads the pair's draft and target, checks they share a tokenizer, prints tokens per word, runs one Hindi prompt with the target alone and with our SD (greedy, must match), prints speed. |
| `sd.py` | Our speculative decoding loop. Accepts drafted tokens with min(1, p/q)·L^α (soft) or vetoes when L < τ (hard). With no module (L = 1) it is standard SD. Modules subclass `Scorer`. |
| `run_generation.py` | Runs one system over the prompts and saves outputs plus a per-token log to `results/`. |
| `agreement.py` | Agreement checks on a Stanza parse (subject–verb, adjective–noun, Hindi ने and oblique). Used by `evaluate.py` and module A. |
| `modules/rules.py` | Module A (S2): rule-based L. Surface rules (broken matra, script mixing, repetition) on every token, agreement rules when a token ends a word. |
| `ngram.py` | Kneser-Ney n-gram model over POS+morph tags (e.g. `VERB|Fem|Sing|3`), plain Python. |
| `build_ngram.py` | Module B data: tags IndicCorp with Stanza (`tag`), trains the tag 4-gram (`train`). |
| `modules/stat.py` | Module B (S3): L = percentile (vs real text) of P(tag of new word | 3 previous tags). |
| `build_corrupt.py` | Module C data: clean vs corrupted sentence prefixes from IndicCorp (agreement swaps, dropped/repeated words). |
| `train_classifier.py` | Module C: fine-tunes MuRIL on those prefixes (one model, all languages) -> `data/classifier/`. |
| `modules/script_guard.py` | Optional for any system (`--script-guard`): penalizes drift into English on every emitted token, with a Latin budget taken from the prompt. |
| `modules/neural.py` | Module C (S4): L = P(grammatical | sentence so far) from the classifier. |
| `modules/common.py` | Shared by the modules: token text and word-end detection. |
| `eval_multiblimp.py` | Minimal pairs: does the target/draft prefer the correct verb, does each module give the wrong verb a lower L, and how often SD accepts a wrong verb (S1 vs S2/S3/S4). |
| `evaluate.py` | Scores outputs in `results/`: length, % English, looping, chrF++ (Dolly), agreement errors (Stanza), judge perplexity, acceptance rate, tok/s. |
| `run_all.sh` | Launches all five systems (S1, S2 surface/all, S3, S4) for the current pair, with or without the guard. |
| `download_data.py` | Downloads all datasets into `data/` as jsonl. Steps: `aya`, `dolly`, `blimp`, `indiccorp`, `ud`. |

## Steps

**Model pair.** Everything below runs the BLOOMZ pair by default. `export PAIR=qwen` switches every script to
Qwen2.5-0.5B-Instruct -> Qwen2.5-7B-Instruct (chat template, bf16, longer token limits) and writes to
`results_qwen/`, so BLOOMZ results are never touched. Modules B and C (n-gram, MuRIL) do not depend on the pair.

**1. Create the environment**
```bash
conda env create -f environment.yml
conda activate lingsd
```

**2. Install PyTorch with CUDA**

Run `nvidia-smi` and read "CUDA Version" in the top right. Pick the line that is not higher than it:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA Version 12.8 or higher
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA Version 12.6 - 12.7
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA Version 12.4 - 12.5
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA Version 12.1 - 12.3
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA Version 11.8 - 12.0
```
Check it (must print `True`):
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**3. Check the models** (downloads ~17 GB, needs ~16 GB GPU memory)
```bash
python check_models.py --gpu 0      # both models on one GPU; pick a free one from nvidia-smi
```
You should see `tokenizer OK`, a Hindi answer, and two speeds (target alone vs speculative decoding).

**4. Download the small datasets** (a few minutes)
```bash
python download_data.py aya dolly blimp ud
```

**5. Download IndicCorp v2** (public; only needed for modules B and C, see step 8)

**6. Run the S1 baseline** (standard speculative decoding)
```bash
python run_generation.py --system S1 --data aya --lang hin --limit 10 --gpu 0   # quick test first
python run_generation.py --system S1 --gpu 0                                    # everything
```
Writes `results/S1/{data}_{lang}.jsonl`: output text, stats (acceptance rate, tok/s) and every drafted token with p, q, L and whether it was accepted.

**7. Run S2 (module A, rules)**
```bash
python run_generation.py --system S2 --data aya --lang hin --limit 10 --gpu 0 --variant hard --tau 0.3 --tag hard_tau0.3   # quick test
# rule-group ablation (hard veto, tau 0.5 so every fired rule vetoes)
for g in grammar surface all; do python run_generation.py --system S2 --rules $g --variant hard --tau 0.5 --tag ${g}_hard --gpu 0; done
```
Each output line also has `rules_fired`, e.g. `{"script_mix": 3, "subj_verb": 1}`.

**8. Module B (S3): IndicCorp + tag 4-gram**

IndicCorp v2 is public (no login needed). Do not use `load_dataset("ai4bharat/IndicCorpV2")`: it downloads every language.
```bash
python download_data.py indiccorp --n-sents 1000000     # streams ~1M clean sentences per language -> data/indiccorp/
export OMP_NUM_THREADS=2                                # do this in every shell you run things from
python build_ngram.py tag --lang hin --n 10000 --gpu 0  # speed test first; it resumes, so just rerun with --n 1000000
python build_ngram.py tag --n 1000000 --gpu 0           # all languages -> data/ngram/{lang}.tags
python build_ngram.py train                             # -> data/ngram/{lang}.pkl, prints held-out perplexity
python run_generation.py --system S3 --data aya --lang hin --limit 20 --variant hard --tau 0.01 --tag hard_tau0.01 --gpu 0
```
L is a percentile of real text: L = 0.01 means the new word's tag is less likely than for 99% of words in held-out IndicCorp. Use `--variant hard --tau 0.01` (or 0.05); τ is then roughly the share of real text that would be vetoed. `rules_fired` counts checked words and how many got L < 0.1 / 0.05 / 0.01.

**8b. Module C (S4): MuRIL classifier**
```bash
export OMP_NUM_THREADS=2
python build_corrupt.py                                  # 200k pairs/lang -> data/corrupt/{train,dev}.jsonl
python train_classifier.py --gpu 0 --limit 20000         # quick test: dev accuracy should rise well above 0.5
python train_classifier.py --gpu 0                       # full run (1.2M rows, 1 epoch) -> data/classifier/
python run_generation.py --system S4 --data aya --lang hin --limit 20 --variant hard --tau 0.1 --tag hard_tau0.1 --gpu 0
```
`train_classifier.py` prints, per τ, how much clean text the classifier would veto and how many corruptions it catches: pick τ where clean vetoed is about 1-5%.

**8c. Script guard (any system)**
```bash
python run_generation.py --system S2 --rules surface --variant hard --tau 0.5 --script-guard --tag surface_hard_guard --gpu 0
bash run_guard_all.sh          # all systems with the guard, one GPU per system -> results/*_guard
bash eval_all.sh 0 both        # table with and without the guard
```
Native prompt: about 20% Latin letters allowed (`--guard-budget`); an English or mixed prompt raises the budget, so English prompts are not affected. Latin words copied from the prompt are always allowed.

**9. Evaluate**
```bash
python evaluate.py results/S1                               # cheap metrics, seconds
python evaluate.py results/S1 --stanza --gpu 0              # + agreement errors per 100 words
python evaluate.py results/S1 --judge sarvamai/sarvam-1 --gpu 0   # + perplexity under a judge LM
```
Prints one summary row per file and writes per-output scores to `results/S1/eval_*.jsonl`.

**10. Minimal pairs (MultiBLiMP)**
```bash
python eval_multiblimp.py --gpu 0 --limit 50     # quick test
python eval_multiblimp.py --gpu 0                # all pairs; tau per module: --tau-s2 0.5 --tau-s3 0.05 --tau-s4 0.3
```
The main grammar metric for Marathi and Tamil (parser counts are unreliable there). "bad accepted" is the bad-token acceptance rate.

## Output in `data/`

| Path | Content |
|---|---|
| `prompts/aya_{lang}.jsonl` | 300 native prompts per language |
| `prompts/dolly_{lang}.jsonl` | 200 prompts, the same questions in all three languages, with references |
| `multiblimp/{lang}.jsonl` | agreement minimal pairs (prefix, correct verb, wrong verb) |
| `ud/{lang}/` | UD treebanks: `.conllu` files plus `.jsonl` (tokens, POS, features, heads) |
| `indiccorp/{lang}.jsonl` | clean sentences for modules B and C |

If a Hugging Face step stops with "no config" or "no file", paste the message to me: it lists the real names.
