"""Preliminary 4-column ablation-ladder table from whatever eval cells exist on disk.

Pairs strictly on (seed, N) so every reported delta is within-checkpoint where it
should be. random_docs may carry multiple reps per cell; they are averaged first.
"""

import csv
import glob
import statistics as st

ARMS = ["patient_only", "real_docs", "random_docs", "null_docs"]


def load(arm):
    """Return {(seed, n): auroc}, averaging reps when an arm has them."""
    acc = {}
    for p in glob.glob(f"outputs/eval_{arm}/*/*/**/loggers/csv/version_0/metrics.csv", recursive=True):
        parts = p.split("/")
        seed, n = parts[3], parts[4]
        try:
            rows = list(csv.DictReader(open(p)))
        except OSError:
            continue
        if not rows:
            continue
        key = next((c for c in rows[0] if c.endswith("test/auroc/mean")), None)
        if not key:
            continue
        vals = [float(r[key]) for r in rows if r.get(key)]
        if vals:
            acc.setdefault((seed, n), []).append(vals[-1])
    return {k: st.mean(v) for k, v in acc.items()}


data = {a: load(a) for a in ARMS}
cells = sorted(set.intersection(*(set(data[a]) for a in ARMS)), key=lambda x: (int(x[1][1:]), x[0]))
print(f"paired cells present in all four arms: {len(cells)} of 40\n")
if not cells:
    raise SystemExit("no fully-paired cells yet")

hdr = f"{'N':>5} {'patient_only':>13} {'marginalized':>13} {'random':>10} {'null':>10}"
hdr += f" {'C1':>9} {'C2':>9} {'C3':>9} {'seeds':>6}"
print(hdr)
print("-" * len(hdr))

pooled = {"C1": [], "C2": [], "C3": []}
for n in sorted({c[1] for c in cells}, key=lambda x: int(x[1:])):
    sub = [c for c in cells if c[1] == n]
    m = {a: st.mean([data[a][c] for c in sub]) for a in ARMS}
    # C1 = does retrieval help; C2 = does doc *selection* matter; C3 = do doc *contents* matter
    c1 = st.mean([data["real_docs"][c] - data["patient_only"][c] for c in sub])
    c2 = st.mean([data["real_docs"][c] - data["random_docs"][c] for c in sub])
    c3 = st.mean([data["real_docs"][c] - data["null_docs"][c] for c in sub])
    for k, v in (("C1", c1), ("C2", c2), ("C3", c3)):
        pooled[k].append(v)
    print(
        f"{n:>5} {m['patient_only']:>13.4f} {m['real_docs']:>13.4f} "
        f"{m['random_docs']:>10.4f} {m['null_docs']:>10.4f} "
        f"{c1:>+9.4f} {c2:>+9.4f} {c3:>+9.4f} {len(sub):>6}"
    )

print()
labels = {
    "C1": "real - patient_only  (does retrieval help at all?)",
    "C2": "real - random_docs   (does document SELECTION matter?)",
    "C3": "real - null_docs     (do document CONTENTS matter?)",
}
for k in ("C1", "C2", "C3"):
    per_cell = []
    for c in cells:
        base = data["real_docs"][c]
        other = {"C1": "patient_only", "C2": "random_docs", "C3": "null_docs"}[k]
        per_cell.append(base - data[other][c])
    m = st.mean(per_cell)
    sd = st.stdev(per_cell) if len(per_cell) > 1 else 0.0
    wins = sum(1 for x in per_cell if x > 0)
    print(f"{labels[k]}")
    print(f"    mean {m:+.4f}  sd {sd:.4f}  retrieval wins {wins}/{len(per_cell)} cells")
