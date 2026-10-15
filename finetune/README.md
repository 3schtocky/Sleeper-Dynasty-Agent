# Fine-tuning the chat model

This folder is the recipe for fine-tuning the chat model (`qwen3:4b-instruct`) with QLoRA on a single consumer GPU. **Status: work in progress.** The default model stays the base model until a fine-tuned one passes every gate below.

## The short version

```
pip install -e ".[finetune]"                # on an NVIDIA GPU, install the CUDA build of PyTorch first
dynasty-agent finetune all --llama-cpp <folder with convert_hf_to_gguf.py and llama-quantize>
```

`finetune all` runs the four stages in order (`data`, `train`, `package`, `gates`). Each can be run alone (`dynasty-agent finetune train`), and `--dry-run` prints the exact commands without running anything. Work goes in `finetune-work/` (git-ignored). The sections below explain each stage and the scripts behind it.

## Why fine-tune at all

The base model already passes both evals once the Python guards are in place (39 of 39 routing, 57 of 57 conversation steps). Fine-tuning is not there to fix those. It is there to make the model do the right thing *without* a guard stepping in: route a plain trade question on the first try, stop at the right length, and never write the sentence a guard would have to drop. A guard that rarely fires is a better guard than one that fires every turn. If a fine-tune does not beat the base model on the gates, **keep the base model**. That is a valid result.

## Nobody's league goes into the model

This model is meant to be published, so its training data must contain nothing private. It is built from a *synthetic league*, not a real one:

1. `make_synthetic_league.py` copies a real database and replaces everything league-specific: manager and team names, user, league and draft IDs, win-loss records and points. Players are shuffled among roster rows with the same position and slot kind, so every roster keeps a legal shape while nobody owns who they really own. Matchups, traded picks, transactions and every cached Sleeper response are deleted, and the file is vacuumed.
2. It then scans every text column of every table, and the raw bytes of the file, for each identifier the real league had. If one remains, it deletes the output and fails. The scanner was tested against the real database to confirm it finds leaks there.
3. Player names, NFL stats and FantasyCalc values are public and stay.

No real chat log is used. The dataset is generated from templates over the synthetic league plus replies sampled from the base model.

## The dataset (`build_dataset.py`)

About 1,280 examples across the four jobs the chat gives the model, each in the exact message format the chat sends (same system prompts, same tool schemas, same Qwen tool-call format), so training matches inference:

| Job | Share | How the answers are made |
|---|---|---|
| **route**: a question to one of six tool calls, or a short no-tool reply | ~28% | Right by construction. The sides of every trade come from who owns whom in the synthetic league, and each trade is run through the real tool to prove it is valid |
| **topic**: `fantasy`, `smalltalk` or `other` | ~10% | Right by construction from labeled lists |
| **take**: two to four friendly sentences from a tool's facts | ~40% | Sampled from the base model (3 candidates) and kept only if the project's own checks pass |
| **talk**: free fantasy conversation, some multi-turn | ~23% | Same: sampled, then filtered |

A sampled reply is kept only if **every** check passes: each number is in the tool's output (`grounding.ungrounded`), no markdown, list, emoji or em dash, the sentence limit holds, the injury-risk and fact-contradiction checks pass (`risk_talk`), and the trade finder never tells the user to make a trade. This is rejection-sampling fine-tuning: the student learns the teacher's *best* behavior, filtered by code, so it is a consistency gain rather than new knowledge.

**Contamination is blocked.** Any question that appears in the repo's evals is excluded from training, so the gates stay honest. Ten percent of examples are held out for validation, including every example that mentions a held-out set of players the model never trains on.

Build it:

```
dynasty-agent finetune data                 # needs Ollama running the base model; takes hours
```

or run the three scripts by hand:

```
python finetune/make_synthetic_league.py data/dynasty.db work/synthetic.db
python finetune/build_dataset.py work/synthetic.db work/dataset        # needs Ollama running the base model
python finetune/make_probes.py work/synthetic.db work/dataset          # fresh prompts, none in the training data
```

`build_dataset.py` is resumable: rerun it and it skips finished work. It asks the base model for its three candidate replies at the same time, which Ollama serves in parallel on one GPU for about 2.3 times the throughput.

## Training (`train.py`, `dynasty-agent finetune train`)

QLoRA on [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (the same weights Ollama calls `qwen3:4b-instruct`). The base is loaded in 4-bit (NF4, double quantization, bfloat16 compute) and small low-rank adapters are trained on top.

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 on every attention and MLP projection |
| Sequence length | 2048 (drop to 1536 if long rows stall on an 8 GB card) |
| Batch | 1 per device x 16 gradient accumulation = 16 |
| Optimizer / learning rate | paged AdamW 8-bit, 1e-4, cosine, 3% warmup, grad-norm clip 1.0 |
| Epochs | up to 3, evaluating every 50 steps, keeping the lowest eval loss, stopping early after 3 evaluations that get worse |
| Loss | **assistant tokens only** (system, user and tool turns are masked) |

Two details matter:

- **Assistant-only loss is built by hand.** Qwen3's chat template has no `{% generation %}` markers, which the usual trainer option needs. `train.py` finds each assistant turn by character offsets and masks everything else.
- **`lossfn.py` saves memory.** Plain training builds a `[sequence x 152,000-word vocabulary]` logits tensor, which alone overflows an 8 GB card at 2048 tokens. The lean loss runs the output layer only on assistant positions, in checkpointed chunks of 512, and computes the identical cross-entropy.

```
python finetune/check_data.py work/dataset/train.jsonl work/dataset/eval.jsonl     # validates before you spend hours
python finetune/train.py --train work/dataset/train.jsonl --eval work/dataset/eval.jsonl
```

It watches GPU temperature and pauses above 83 C.

## Turning an adapter into something Ollama can run

```
python finetune/merge.py <run>/best_adapter work/merged           # merge the LoRA into the bf16 base, on CPU
python llama.cpp/convert_hf_to_gguf.py work/merged --outfile work/model-f16.gguf --outtype f16
llama.cpp/llama-quantize work/model-f16.gguf dynasty-qwen3-4b-v2.Q4_K_M.gguf Q4_K_M
ollama create dynasty-qwen3-4b:v2 -f finetune/Modelfile
```

Ollama imports adapters directly only for some model families, and Qwen3 is not one of them, so the adapter is merged first. The Modelfile copies the base model's template and stop tokens, so prompts are formatted exactly as in training.

## The gates

A fine-tuned model replaces the default only if **all** of these hold, measured on the same machine:

1. `dynasty-agent chat-eval --model dynasty-qwen3-4b:v2` routes 39 of 39.
2. `dynasty-agent chat-eval --conversations --hard --model dynasty-qwen3-4b:v2` passes every step.
3. `score_probes.py` on the held-out probes is no worse than the base model on routing, topic, grounded takes and talk, and no reply contains markdown, a list or an emoji.
4. Generation speed stays within about 10% of the base model.

`dynasty-agent finetune gates` measures the base model and the new one on all four, prints PASS or FAIL for each gate, writes the numbers to `finetune-work/gates_report.json` and exits non-zero if any gate fails. The probes alone:

```
python finetune/score_probes.py work/dataset/probes.jsonl work/probes_out.jsonl --model dynasty-qwen3-4b:v2
```

## What gets published

The recipe and the evidence go in the repo. A quantized model is about 2.3 GiB, which is over GitHub's limit for a release file (2 GiB) and far over its limit for a file in git (100 MB). So the weights are hosted separately and linked from here. Nothing is published unless the model passes the gates.
