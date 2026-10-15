"""Memory-lean causal LM loss: lm_head runs only on labelled (assistant) positions, in checkpointed chunks,
so the full [seq, 152k-vocab] logits tensor is never materialised."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK = 512


def _chunk_loss(h, w, y):
    return F.cross_entropy((h @ w.t()).float(), y, reduction="sum")


def lean_loss(peft_model, input_ids, attention_mask, labels, num_items=None):
    """Mean over labelled tokens, or sum / num_items when the trainer passes the accumulated-batch token count."""
    inner = peft_model.base_model.model          # Qwen3ForCausalLM
    hidden = inner.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    h = hidden[:, :-1][labels[:, 1:] != -100]    # [N, H] assistant positions only
    y = labels[:, 1:][labels[:, 1:] != -100]
    if y.numel() == 0:
        return hidden.sum() * 0.0
    w = inner.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), CHUNK):
        total = total + checkpoint(_chunk_loss, h[i:i + CHUNK], w, y[i:i + CHUNK], use_reentrant=False)
    return total / (num_items if num_items is not None else y.numel())
