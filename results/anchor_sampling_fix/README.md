# Anchor-sampling fix: results summary

**Setup (constant across all three runs below):** N=25 randomly-drawn task
codes, 5 independent random draws (seeds 101/202/303/404/505), two fixed
occurrence windows (7-day, 30-day), `patient_only` vs. `marginalized(binary,
k=4)` architectures trained on identical labels within each (duration,
draw) pair, 3 epochs. Only the **anchor-sampling logic** — how the
prediction-time "now" is chosen for each labeled example — changes between
the three columns below. Full scripts/configs live in `mimic_iv_sweep/`.

| Version | How the anchor (prediction time) is chosen |
| --- | --- |
| **Original** | Uniform random draw from *continuous calendar time* within the patient's valid window. Could land in a "dead" gap between visits — a timestamp where nothing was recorded. |
| **MedRAP#99** (my fix) | Uniform random draw over the patient's *real event timestamps* instead of continuous time — every anchor now coincides with an actual observation. |
| **MedRAP#100** (Zach's refinement) | Same idea as #99, plus excludes synthetic `TIMELINE//` boundary-marker tokens from the anchor candidate pool, so an anchor can never land on an artifact instead of a real clinical event. |

## Mean AUROC across the 5 draws

| Duration | Architecture | Original (continuous-time) | #99 (real-event) | #100/Zach (real-event, TIMELINE-excluded) |
| --- | --- | --- | --- | --- |
| 7d  | patient_only   | 0.9443 | 0.8773 | **0.8997** |
| 7d  | marginalized   | 0.9388 | 0.8748 | **0.8868** |
| 30d | patient_only   | 0.9429 | 0.8949 | **0.9037** |
| 30d | marginalized   | 0.9367 | 0.8913 | **0.8954** |

### How to read this

1. **Original → #99: a real drop, and it's expected, not a regression.**
   The original continuous-time anchors could fall in silent gaps between
   visits, which made the prediction task artificially easy (little real
   signal to confuse the model). Anchoring on real events instead makes
   every prediction a genuine clinical decision point — a harder, more
   honest task. AUROC dropping ~0.05–0.07 here reflects the task getting
   harder, not the model getting worse.
2. **#99 → #100/Zach: a small but consistent *improvement*.** Zach's
   refinement closes a gap in #99: #99's anchors could still land exactly
   on a synthetic `TIMELINE//` marker (not a real clinical event). Removing
   that leak recovers some AUROC (+0.01 to +0.02 across the board) — in the
   opposite direction from the original→#99 drop, and for a different
   reason (fixing a residual artifact, not making the task easier).
3. **Draw-to-draw variance also shrank substantially from Original → #99**
   (e.g. patient_only std at 7d: 0.088 → 0.042), making results far more
   reproducible. Under #100/Zach, variance shrinks further at 7d (→0.030)
   but not at 30d (0.022 → 0.028) — with only 5 draws this is plausibly
   noise and worth revisiting with more draws.
4. **The core scientific conclusion is unchanged across all three
   versions**: `marginalized(binary)` retrieval shows **no consistent
   AUROC benefit** over `patient_only` — the architecture gap stays small
   and slightly negative (best case around -0.003, worst case around
   -0.013), always smaller than the noise from draw to draw. Fixing the
   anchor-sampling bug changed the absolute difficulty and reproducibility
   of the task, but not the answer to the retrieval question.

## Full per-draw results and links

Full per-draw AUROC tables (with W&B run links for every run):

- Original vs. #99: [`mimic_iv_sweep/README.md` — "Anchor-sampling fix rerun"](../../mimic_iv_sweep/README.md#anchor-sampling-fix-rerun-generate_labels_duration_variance_n25sh--sweep_patient_only_duration_variance_n25sh--sweep_marginalized_binary_duration_variance_n25sh-medrap99)
- #99 vs. #100/Zach: [`mimic_iv_sweep/README.md` — "Zach's anchor refinement"](../../mimic_iv_sweep/README.md#zachs-anchor-refinement-medrap100-excluding-timeline-tokens-from-anchor-candidates-generate_labels_zach_uniform_event_n25_730dsh--sweep_patient_only_zach_uniform_event_n25_730dsh--sweep_marginalized_binary_zach_uniform_event_n25_730dsh)

## Code references

- Original `patient_only`/`marginalized` comparison: `mimic_iv_sweep/README.md`, "Duration x variance study"
- Anchor fix (#99): [MedRAP PR #99](https://github.com/McDermottHealthAI/MedRAP/pull/99)
- Zach's refinement (#100): [MedRAP PR #100](https://github.com/McDermottHealthAI/MedRAP/pull/100), integrated on `MedRAP@experiment/zach-uniform-event-plus-stack`
