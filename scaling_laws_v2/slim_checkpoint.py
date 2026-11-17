"""Shrink finished checkpoints in place: drop the optimizer state and frozen retrieval-encoder weights.

    python scaling_laws_v2/slim_checkpoint.py CKPT [CKPT ...]

A retrieval checkpoint holds the frozen 151,669 x 1,024 Qwen3 token table (~620 MB), which is the
same in every run and is reloaded from Hugging Face when the model is built; with DOCS=last or mix
it holds the frozen Qwen3 model (~1.2 GB) instead. The layer-mix weights and layer embedding are trained and kept. What remains (trained
weights) is enough to warm-start from or to inspect; re-testing needs the table restored first
(load_state_dict(strict=False) on a freshly built model).
"""

import sys

import torch

for path in sys.argv[1:]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    ckpt["state_dict"] = {k: v for k, v in state.items() if ".retrieval_encoder." not in f".{k}" or k.endswith(("layer_weights", "gamma", "layer_embedding"))}
    for key in ("optimizer_states", "lr_schedulers"):
        ckpt.pop(key, None)
    torch.save(ckpt, path)
    print(f"{path}: kept {len(ckpt['state_dict'])} of {len(state)} tensors")
