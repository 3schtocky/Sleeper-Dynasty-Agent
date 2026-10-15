"""QLoRA fine-tune of Qwen3-4B-Instruct-2507. Loss on assistant tokens only, no packing."""
import argparse, json, os, subprocess, sys, time, datetime
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback,
                          Trainer, TrainerCallback, TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
p.add_argument("--train", default="data/train.jsonl")
p.add_argument("--eval", default="data/eval.jsonl")
p.add_argument("--runs-dir", default="runs")
p.add_argument("--max-len", type=int, default=2048)  # with lossfn.lean_loss: 7.0 GB allocated / 7.8 GB reserved at 2048; drop to 1536 if long rows stall
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--max-steps", type=int, default=-1)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--grad-accum", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup-ratio", type=float, default=0.03)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--eval-steps", type=int, default=50)
p.add_argument("--save-steps", type=int, default=50)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-temp", type=int, default=83)
p.add_argument("--resume", nargs="?", const="auto", default=None,
               help="--resume (latest checkpoint in --run-dir) or --resume <checkpoint path>")
p.add_argument("--run-dir", default=None, help="reuse an existing run dir (needed for --resume)")
args = p.parse_args()

run_dir = args.run_dir or os.path.join(args.runs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(run_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=2)
print("run dir:", run_dir)

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def norm_messages(msgs):
    """Tool-call arguments must be dicts for the Qwen template (a string would be double-encoded)."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append(c)
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def encode(ex):
    """Tokenize one example; labels are -100 except on assistant turns (through <|im_end|>)."""
    msgs = norm_messages(ex["messages"])
    tools = ex.get("tools") or None
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=False)
        if not (full.startswith(upto) and upto.startswith(prefix)):
            raise ValueError("chat template is not prefix-stable for this example")
        end = len(upto)
        if upto.endswith("\n"):  # leave the trailing newline after <|im_end|> unlabelled
            end -= 1
        spans.append((len(prefix), end))
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=args.max_len)
    labels = []
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        labels.append(tid if any(s >= a and e <= b for a, b in spans) else -100)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels,
            "truncated": len(tok(full, add_special_tokens=False)["input_ids"]) > args.max_len}


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list([encode(r) for r in rows])
    n_trunc = sum(ds["truncated"])
    if n_trunc:
        print(f"WARNING: {n_trunc}/{len(ds)} examples in {path} exceed {args.max_len} tokens and were truncated")
    return ds.remove_columns("truncated")


train_ds, eval_ds = load(args.train), load(args.eval)
print(f"train {len(train_ds)} examples, eval {len(eval_ds)} examples")
tok_count = {"n": 0}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    lab = torch.full((len(batch), n), -100)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        att[i, :L] = 1
        lab[i, :L] = torch.tensor(b["labels"])
        tok_count["n"] += L
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0},
                                             dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def gpu_temp():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception:
        return 0


class Monitor(TrainerCallback):
    def __init__(self):
        self.t0 = time.time(); self.last_t = None; self.last_tok = 0; self.log = []

    def on_step_end(self, a, state, control, **kw):
        now = time.time()
        dt = now - (self.last_t or self.t0)
        dtok = tok_count["n"] - self.last_tok
        self.last_t, self.last_tok = now, tok_count["n"]
        peak = torch.cuda.max_memory_allocated() / 1e9
        temp = gpu_temp()
        print(f"[step {state.global_step}] {dt:.1f}s/step  {dtok / dt:.0f} tok/s  peak {peak:.2f} GB  gpu {temp}C", flush=True)
        self.log.append({"step": state.global_step, "sec": dt, "tok": dtok, "peak_gb": peak, "temp": temp})
        if temp > args.max_temp:
            print(f"!!! GPU at {temp}C (> {args.max_temp}C): PAUSING training until it cools below {args.max_temp - 8}C", flush=True)
            open(os.path.join(run_dir, "TEMP_PAUSE.txt"), "a").write(f"step {state.global_step}: {temp}C\n")
            while gpu_temp() > args.max_temp - 8:
                time.sleep(10)
            print("GPU cooled, resuming", flush=True)

    def on_train_end(self, a, state, control, **kw):
        json.dump(self.log, open(os.path.join(run_dir, "step_log.json"), "w"))


targs = TrainingArguments(
    output_dir=run_dir, logging_steps=5, seed=args.seed,
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=args.warmup_ratio, max_grad_norm=1.0,
    optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps,
    save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    report_to="none", remove_unused_columns=False, dataloader_num_workers=0)

from lossfn import lean_loss


class LeanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = lean_loss(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"], num_items_in_batch)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = lean_loss(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        return (loss.detach(), None, None)


trainer = LeanTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collate,
                  callbacks=[Monitor(), EarlyStoppingCallback(early_stopping_patience=args.patience)])
resume = None
if args.resume:
    resume = True if args.resume == "auto" else args.resume
trainer.train(resume_from_checkpoint=resume)

best = os.path.join(run_dir, "best_adapter")
trainer.model.save_pretrained(best)
tok.save_pretrained(best)
hist = [h for h in trainer.state.log_history]
json.dump({"best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "log_history": hist},
          open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print("best checkpoint:", trainer.state.best_model_checkpoint, "eval loss:", trainer.state.best_metric)
print("adapter saved to", best)
