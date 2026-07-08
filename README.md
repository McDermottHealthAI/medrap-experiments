# medrap-experiments

Paper experiments built on top of [MedRAP](https://github.com/McDermottHealthAI/MedRAP) and
[medrap-analysis](https://github.com/McDermottHealthAI/medrap-analysis).

## Structure

Each experiment lives in its own directory:

```
medrap-experiments/
└── <experiment_name>/
    ├── requirements.txt   # pins exact commits/tags of MedRAP, medrap-analysis, etc.
    └── scripts/           # training, sweep, and data-prep scripts for this experiment
```

## Reproducing an experiment

```bash
cd <experiment_name>
pip install -r requirements.txt
# then follow the scripts/ README or comments inside each script
```

## Experiments

| Directory | Description |
|---|---|
| [`mimic_iv/`](mimic_iv/) | MIMIC-IV mortality prediction — initial MedRAP paper experiments (RoPE, cross-attention, retrieval ablations, hyperparameter sweeps) |
| [`mimic_iv_smoke/`](mimic_iv_smoke/) | Smoke test validating the post-refactor MedRAP CLI/structure (flat entrypoints, `medrap-preprocess`) on the same MIMIC-IV pipeline |
