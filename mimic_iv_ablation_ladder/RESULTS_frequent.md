# Ablation ladder -- 4-column AUROC table (code selection: `frequent`)

Generated : 2026-07-31 15:42:16
Outputs   : `/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs`
Readout   : final logged `test/auroc/mean` (the single row each medrap-eval eval_mode=test run logs).
Split     : `test` -- every column is `medrap-eval eval_mode=test` on the MEDS held_out split.
Cell key  : (code selection, N), collapsing path components matching `(seed|rep)\d+` -- i.e. averaged over the 5 model seeds and, for `frozen_random` only, the 3 inference-time reps.

## Coverage

Grid: 2 code selections x 5 model seeds x 8 N values. This report covers selection=`frequent` only.
Expected cells per arm: **40** (seed, N) [`frozen_random` additionally x3 reps = 120 runs].

    model seeds : 1001, 2002, 3003, 4004, 5005
    N values    : 1, 2, 4, 8, 16, 32, 64, 128

| arm | column | runs | (seed, N) cells | status | resolved glob |
| --- | --- | --- | --- | --- | --- |
| `patient_only` | patient_only | 40 | 40/40 | complete | `/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_patient_only/frequent/seed*/n*` |
| `real_docs` | marginalized (binary) | 40 | 40/40 | complete | `/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed*/n*` |
| `random_docs` | frozen_random | 120 | 40/40 | complete | `/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_random_docs/frequent/seed*/n*/rep*` |
| `null_docs` | frozen_null | 40 | 40/40 | complete | `/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed*/n*` |

No missing cells: every arm covers the full 5 x 8 grid.

## Acceptance gates

A failing run is reported and excluded from the headline aggregate, but still appears in the table:

* `n_valid_tasks == n_tasks   -- no task in the label set was unscorable`
* `effective_k_mean >= 2.0    -- retrieval did not collapse onto a single document (checked only where the column exists, i.e. on the marginalized arms)`
* `top1_mode_frac <= 0.5      -- fewer than half of patients retrieve the same top document (a gate only where retrieval reaches the loss; informational elsewhere)`

```

--- acceptance gates: patient_only ---
    all 40 run(s) pass.
    informational (these runs still count toward the headline aggregate):
        NOTE frequent/n1/seed1001: retrieval/test/top1_mode_frac=0.7734 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n1/seed2002: retrieval/test/top1_mode_frac=0.7893 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n1/seed3003: retrieval/test/top1_mode_frac=0.9268 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n1/seed4004: retrieval/test/top1_mode_frac=0.8888 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n1/seed5005: retrieval/test/top1_mode_frac=0.8983 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n2/seed1001: retrieval/test/top1_mode_frac=0.8988 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n2/seed2002: retrieval/test/top1_mode_frac=0.9997 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n2/seed3003: retrieval/test/top1_mode_frac=0.9991 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n2/seed4004: retrieval/test/top1_mode_frac=0.5933 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n2/seed5005: retrieval/test/top1_mode_frac=0.8193 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n4/seed1001: retrieval/test/top1_mode_frac=0.7580 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n4/seed2002: retrieval/test/top1_mode_frac=0.7835 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n4/seed3003: retrieval/test/top1_mode_frac=0.9152 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n4/seed4004: retrieval/test/top1_mode_frac=0.5889 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n4/seed5005: retrieval/test/top1_mode_frac=0.7013 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n8/seed1001: retrieval/test/top1_mode_frac=0.7678 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n8/seed2002: retrieval/test/top1_mode_frac=0.6327 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n8/seed3003: retrieval/test/top1_mode_frac=0.6105 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n8/seed4004: retrieval/test/top1_mode_frac=0.5776 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n8/seed5005: retrieval/test/top1_mode_frac=0.9155 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n16/seed1001: retrieval/test/top1_mode_frac=0.8497 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n16/seed2002: retrieval/test/top1_mode_frac=0.5871 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n16/seed3003: retrieval/test/top1_mode_frac=0.7571 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n16/seed4004: retrieval/test/top1_mode_frac=0.6450 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n16/seed5005: retrieval/test/top1_mode_frac=0.6072 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n32/seed1001: retrieval/test/top1_mode_frac=0.7679 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n32/seed2002: retrieval/test/top1_mode_frac=0.6546 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n32/seed3003: retrieval/test/top1_mode_frac=0.7074 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n32/seed4004: retrieval/test/top1_mode_frac=0.8472 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n32/seed5005: retrieval/test/top1_mode_frac=0.9546 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n64/seed1001: retrieval/test/top1_mode_frac=0.8065 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n64/seed2002: retrieval/test/top1_mode_frac=0.5718 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n64/seed3003: retrieval/test/top1_mode_frac=0.7067 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n64/seed4004: retrieval/test/top1_mode_frac=0.5889 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n64/seed5005: retrieval/test/top1_mode_frac=0.7685 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n128/seed1001: retrieval/test/top1_mode_frac=0.5701 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n128/seed2002: retrieval/test/top1_mode_frac=0.6866 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n128/seed3003: retrieval/test/top1_mode_frac=0.7258 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n128/seed4004: retrieval/test/top1_mode_frac=0.5772 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]
        NOTE frequent/n128/seed5005: retrieval/test/top1_mode_frac=0.6197 > 0.5 -- most patients retrieve the same top document [no differentiable-retrieval column in this run, so retrieval is logged but does not reach the loss -- informational, NOT a gate failure]

--- acceptance gates: marginalized (binary) ---
    EXCLUDED frequent/n1/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed2002/n1)
             reason: retrieval/test/top1_mode_frac=0.6364 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n1/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed5005/n1)
             reason: retrieval/test/top1_mode_frac=0.5960 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed2002/n2)
             reason: retrieval/test/top1_mode_frac=0.5215 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed3003/n2)
             reason: retrieval/test/top1_mode_frac=0.5634 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed4004/n2)
             reason: retrieval/test/top1_mode_frac=0.7667 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed5005/n2)
             reason: retrieval/test/top1_mode_frac=0.5970 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_real_docs/frequent/seed1001/n4)
             reason: retrieval/test/top1_mode_frac=0.5166 > 0.5 -- most patients retrieve the same top document
    7/40 run(s) excluded from the headline aggregate.

--- acceptance gates: frozen_random ---
    all 120 run(s) pass.

--- acceptance gates: frozen_null ---
    EXCLUDED frequent/n1/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n1)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n1/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n1)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n1/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n1)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n1/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n1)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n1/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n1)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n2)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n2)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n2)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n2)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n2/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n2)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n4)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n4)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n4)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n4)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n4/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n4)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n8/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n8)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n8/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n8)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n8/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n8)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n8/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n8)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n8/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n8)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n16/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n16)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n16/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n16)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n16/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n16)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n16/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n16)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n16/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n16)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n32/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n32)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n32/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n32)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n32/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n32)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n32/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n32)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n32/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n32)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n64/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n64)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n64/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n64)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n64/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n64)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n64/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n64)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n64/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n64)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n128/seed1001  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed1001/n128)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n128/seed2002  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed2002/n128)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n128/seed3003  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed3003/n128)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n128/seed4004  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed4004/n128)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    EXCLUDED frequent/n128/seed5005  (/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/outputs/eval_null_docs/frequent/seed5005/n128)
             reason: retrieval/test/top1_mode_frac=1.0000 > 0.5 -- most patients retrieve the same top document
    40/40 run(s) excluded from the headline aggregate.
    *** NO RUN PASSES THE GATES. The headline aggregate is empty; every number
        below is DESCRIPTIVE ONLY and must not be read as an architecture result. ***
```

## Results

**What "valid task" means:** a sampled task counts as valid only if the scored split contains at least one positive *and* one negative example -- AUROC is undefined otherwise, so those tasks are excluded from the mean rather than scored as 0.

Comparisons (all three share the `marginalized (binary)` column as their reference):

* **C1** = `marginalized (binary)` − `patient_only` -- between-arm (separately trained models); does retrieval help at all?
* **C2** = `frozen_random` − `marginalized (binary)` -- within-checkpoint (weights frozen, documents swapped); is retrieval SELECTING useful documents, or just adding machinery?
* **C3** = `frozen_null` − `marginalized (binary)` -- within-checkpoint, deterministic (content-free corpus); do document CONTENTS contribute anything?

| N | patient_only | marginalized (binary) | frozen_random | frozen_null | Δ C1 | Δ C2 | Δ C3 | valid tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8858 | 0.8790 | 0.8787 | 0.8787 | -0.0068 | -0.0003 | -0.0003 | 1/1 |
| 2 | 0.7624 | 0.7464 | 0.7459 | 0.7449 | -0.0160 | -0.0005 | -0.0015 | 2/2 |
| 4 | 0.8229 | 0.8190 | 0.8156 | 0.8167 | -0.0039 | -0.0034 | -0.0023 | 4/4 |
| 8 | 0.8394 | 0.8366 | 0.8330 | 0.8342 | -0.0028 | -0.0036 | -0.0024 | 8/8 |
| 16 | 0.8458 | 0.8447 | 0.8427 | 0.8432 | -0.0011 | -0.0020 | -0.0015 | 16/16 |
| 32 | 0.8368 | 0.8363 | 0.8338 | 0.8337 | -0.0005 | -0.0025 | -0.0026 | 32/32 |
| 64 | 0.8398 | 0.8368 | 0.8352 | 0.8344 | -0.0030 | -0.0016 | -0.0024 | 64/64 |
| 128 | 0.8422 | 0.8362 | 0.8343 | 0.8331 | -0.0060 | -0.0019 | -0.0031 | 128/128 |

Values are the mean final `test/auroc/mean` over the 5 model seeds (and over the 3 inference reps for `frozen_random`). Deltas are the difference of the two displayed 4-decimal values (repo convention); the paired statistics below are computed at full precision, so they can differ in the last digit.

**Gate failures in cells: 1, 2, 4, 8, 16, 32, 64, 128** -- these rows are reported for reproducibility but at least one of their runs failed an acceptance gate; see the gate report above for the per-run reason. They are excluded from the headline paired statistics below.

## Paired comparisons

The headline test blocks on N: one paired observation per (code selection, N) cell, each side already averaged over its seeds and reps. The secondary test un-collapses the seed dimension and pairs at (seed, N), which is the 40-pair structure the grid was sized for -- it has more power but assumes the seeds are exchangeable between the two arms, which holds for C2 and C3 (same checkpoint) and is an assumption for C1 (separately trained).

```
=== C1: marginalized (binary) - patient_only ===
    question  : does retrieval help at all?
    structure : between-arm (separately trained models)
    matched cells: 8
    paired delta over N blocks, gate-passing only (HEADLINE): n=5
        mean delta = -0.0027   sd = 0.0021   se = 0.0010
        paired t(4) = -2.804   two-sided p = 0.0486
        95% t-CI   = [-0.0053, -0.0000]
        5/5 blocks negative (treatment worse), 0/5 non-negative
    paired delta over N blocks, ALL matched (descriptive): n=8
        mean delta = -0.0050   sd = 0.0049   se = 0.0017
        paired t(7) = -2.887   two-sided p = 0.0234
        95% t-CI   = [-0.0091, -0.0009]
        8/8 blocks negative (treatment worse), 0/8 non-negative
    paired delta over (seed, N) blocks, ALL matched (secondary): n=40
        mean delta = -0.0050   sd = 0.0062   se = 0.0010
        paired t(39) = -5.137   two-sided p = 0.0000
        95% t-CI   = [-0.0070, -0.0030]
        36/40 blocks negative (treatment worse), 4/40 non-negative

=== C2: frozen_random - marginalized (binary) ===
    question  : is retrieval SELECTING useful documents, or just adding machinery?
    structure : within-checkpoint (weights frozen, documents swapped)
    matched cells: 8
    paired delta over N blocks, gate-passing only (HEADLINE): n=5
        mean delta = -0.0023   sd = 0.0008   se = 0.0004
        paired t(4) = -6.453   two-sided p = 0.0030
        95% t-CI   = [-0.0033, -0.0013]
        5/5 blocks negative (treatment worse), 0/5 non-negative
    paired delta over N blocks, ALL matched (descriptive): n=8
        mean delta = -0.0020   sd = 0.0012   se = 0.0004
        paired t(7) = -4.751   two-sided p = 0.0021
        95% t-CI   = [-0.0030, -0.0010]
        8/8 blocks negative (treatment worse), 0/8 non-negative
    paired delta over (seed, N) blocks, ALL matched (secondary): n=40
        mean delta = -0.0020   sd = 0.0017   se = 0.0003
        paired t(39) = -7.476   two-sided p = 0.0000
        95% t-CI   = [-0.0025, -0.0014]
        37/40 blocks negative (treatment worse), 3/40 non-negative

=== C3: frozen_null - marginalized (binary) ===
    question  : do document CONTENTS contribute anything?
    structure : within-checkpoint, deterministic (content-free corpus)
    matched cells: 8
    paired delta over N blocks, gate-passing only (HEADLINE): n=0
        nothing to test.
    paired delta over N blocks, ALL matched (descriptive): n=8
        mean delta = -0.0020   sd = 0.0009   se = 0.0003
        paired t(7) = -6.525   two-sided p = 0.0003
        95% t-CI   = [-0.0027, -0.0013]
        8/8 blocks negative (treatment worse), 0/8 non-negative
    paired delta over (seed, N) blocks, ALL matched (secondary): n=40
        mean delta = -0.0020   sd = 0.0015   se = 0.0002
        paired t(39) = -8.696   two-sided p = 0.0000
        95% t-CI   = [-0.0025, -0.0015]
        37/40 blocks negative (treatment worse), 3/40 non-negative

```

## Realized cross-seed sigma

Training noise does not shrink when the labels improve, and it was unmeasured under these labels before this run. Below is the sd of the per-seed cell value across the model seeds, within each (selection, N) cell: `mean sd` averages those sds, `pooled sd` is the square root of the mean variance. Right-size any follow-up experiment against `pooled sd`, not against the old 0.011 figure (which included estimator noise these labels largely remove).

```
    arm                    cells  seeds/cell  mean sd  pooled sd  min sd  max sd
    ---------------------  -----  ----------  -------  ---------  ------  ------
    patient_only           8      5.0         0.0019   0.0025     0.0008  0.0059
    marginalized (binary)  8      5.0         0.0026   0.0033     0.0008  0.0072
    frozen_random          8      5.0         0.0030   0.0035     0.0013  0.0072
    frozen_null            8      5.0         0.0031   0.0038     0.0011  0.0078
```

