# mimic_iv

MIMIC-IV mortality prediction experiments for the initial MedRAP paper.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

Run in roughly this order:

1. **`scripts/create_tasks.sh`** — extract mortality prediction task labels from the MIMIC-IV MEDS cohort
2. **`scripts/prepare_retrieval_slurm.sh`** — build the HF retrieval corpus and FAISS index
3. Training (pick one variant):
   - `scripts/run_rope_cross_attention.sh` — RoPE encoder + cross-attention fusion (main paper model)
   - `scripts/run_retrieval_only.sh` / `run_retrieval_only_local.sh` — retrieval-only ablation
   - `scripts/train_multitask_rope_cross_attention_slurm.sh` — multitask variant (SLURM)
   - `scripts/train_multitask_rope_no_retrieval_slurm.sh` — no-retrieval baseline (SLURM)
4. **`scripts/sweep_slurm.sh`** / **`scripts/sweep_local.sh`** — hyperparameter sweep over k, lr, embed dims
