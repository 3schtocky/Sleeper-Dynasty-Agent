"""Merge a LoRA adapter into the bf16 base on CPU and save safetensors. Usage: merge.py <adapter_dir> <out_dir>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out = sys.argv[1], sys.argv[2]
BASE = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
print("merged model saved to", out)
