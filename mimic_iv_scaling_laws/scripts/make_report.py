"""Write a self-contained HTML overview of the scaling-law experiments and their results.

Run from the directory plot_results.py wrote into (it holds outputs/, figures/, tables/ and,
for the 625-task grid, t625/): every experiment gets a short description, its config and its
completion status; results get headline numbers, paired retrieval - patient_only tables with
95% CIs over draws, and the figures. Re-run after syncing new results to refresh it.

Usage:
    cd ~/Desktop/scaling_laws_results
    <venv>/bin/python <repo>/mimic_iv_scaling_laws/scripts/plot_results.py --out .
    <venv>/bin/python <repo>/mimic_iv_scaling_laws/scripts/plot_results.py --out t625 --main-tasks 625
    <venv>/bin/python <repo>/mimic_iv_scaling_laws/scripts/make_report.py   # -> report.html
"""

import datetime
import html
import math
import re
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import ABLATION_DIM, CONTROL_ARMS, CONTROL_NUM_TASKS, collect_results, encoder_n_params

DIMS = (4, 8, 16, 32, 64, 128)
NS = (1000, 3000, 10000, 30000, 100000)
ABLATION_TASKS = (5, 25, 125, 625, 2500)
CONTROL_DIMS, CONTROL_NS = (4, 16, 64), (1000, 10000, 100000)
ARM_LABEL = {
    "patient_only": "Patient only",
    "retrieval": "Retrieval",
    "retrieval_random_docs": "Retrieval · random docs",
    "retrieval_null_docs": "Retrieval · null docs",
}
BLUE, ORANGE, GRAY = (0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0xF0, 0xEF, 0xEC)


def compact(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else str(int(n))


def per_draw(results: pl.DataFrame, col: str = "test_metric__auroc") -> pl.DataFrame:
    dims = {encoder_n_params(d): d for d in DIMS}
    return results.select(
        "arm",
        pl.col("model_size__n_params").replace_strict(dims).alias("dim"),
        pl.col("dataset_size__n_train_patients").alias("n"),
        pl.col("num_tasks").alias("t"),
        pl.col("optimizer_seed").alias("draw"),
        pl.col(col).alias("auroc"),
    )


def paired(draws: pl.DataFrame, arm: str, ref: str = "patient_only") -> dict:
    """(dim, n, t) -> (mean diff, ci half-width, wins, n_draws) for arm - ref over matched draws."""
    a = draws.filter(pl.col("arm") == arm).drop("arm")
    b = draws.filter(pl.col("arm") == ref).drop("arm").rename({"auroc": "ref"})
    j = a.join(b, on=["dim", "n", "t", "draw"]).with_columns((pl.col("auroc") - pl.col("ref")).alias("d"))
    out = {}
    for (dim, n, t), g in j.group_by(["dim", "n", "t"]):
        d = g["d"].to_list()
        k = len(d)
        hw = stats.t.ppf(0.975, k - 1) * (pl.Series(d).std() / k**0.5) if k > 1 else float("nan")
        out[(dim, n, t)] = (sum(d) / k, hw, sum(x > 0 for x in d), k)
    return out


def tint(v: float, lim: float = 0.2) -> str:
    """Diverging cell background: blue (patient only better) -> gray -> orange (retrieval better).

    A negative lim flips the sign (for loss, where lower is better); abs(lim) < 0.01 is
    read as a log scale so small loss gaps stay visible next to large ones.
    """
    if abs(lim) < 0.01:
        mag = min(1.0, max(0.0, (math.log10(abs(v) + 1e-12) + 4) / 3.5))  # 1e-4 -> 0, ~3e-1 -> 1
        x = math.copysign(mag, v * lim)
    else:
        x = max(-1.0, min(1.0, v / lim))
    end = ORANGE if x > 0 else BLUE
    w = abs(x) * 0.75
    r, g, b = (round(GRAY[i] * (1 - w) + end[i] * w) for i in range(3))
    return f"rgb({r},{g},{b})"


def diff_table(stats_: dict, t: int, dims, ns, caption: str, lim: float = 0.2, dec: int = 3) -> str:
    head = "".join(f"<th>{compact(n)}</th>" for n in ns)
    rows = []
    for dim in dims:
        cells = []
        for n in ns:
            s = stats_.get((dim, n, t))
            if s is None:
                cells.append('<td class="pending">pending</td>')
                continue
            m, hw, wins, k = s
            if lim < 0:  # lower is better: a retrieval "win" is a negative difference
                wins = k - wins
            sig = "sig" if m - hw > 0 or m + hw < 0 else ""
            lo, hi = f"{m - hw:+.{dec}f}", f"{m + hw:+.{dec}f}"
            cells.append(
                f'<td class="num tinted {sig}" style="background:{tint(m, lim)}" title="95% CI [{lo}, {hi}], '
                f'{wins}/{k} draws won by retrieval"><b>{m:+.{dec}f}</b><span>[{lo}, {hi}] · {wins}/{k}</span></td>'
            )
        rows.append(f"<tr><th>dim {dim}</th>{''.join(cells)}</tr>")
    return (
        f'<figure class="tbl"><figcaption>{caption}</figcaption><div class="scroll"><table class="grid">'
        f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></figure>"
    )


def mean_table(draws: pl.DataFrame, t: int, dims, ns, arms, caption: str) -> str:
    agg = (
        draws.filter(pl.col("t") == t)
        .group_by("arm", "dim", "n")
        .agg(pl.col("auroc").mean(), pl.col("auroc").std().alias("sd"))
    )
    look = {(r["arm"], r["dim"], r["n"]): (r["auroc"], r["sd"]) for r in agg.iter_rows(named=True)}
    head = "".join(f"<th>{compact(n)}</th>" for n in ns)
    body = []
    for dim in dims:
        for i, arm in enumerate(arms):
            cells = "".join(
                f'<td class="num">{look[(arm, dim, n)][0]:.3f}<span>± {look[(arm, dim, n)][1] or 0:.3f}</span></td>'
                if (arm, dim, n) in look
                else '<td class="pending">pending</td>'
                for n in ns
            )
            lead = f'<th rowspan="{len(arms)}">dim {dim}</th>' if i == 0 else ""
            body.append(
                f'<tr class="{"first" if i == 0 else ""}">{lead}<td class="arm"><i class="sw {arm}"></i>{ARM_LABEL[arm]}</td>{cells}</tr>'
            )
    return (
        f'<figure class="tbl"><figcaption>{caption}</figcaption><div class="scroll"><table class="grid">'
        f"<thead><tr><th></th><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></figure>"
    )


def figure(path: str, caption: str) -> str:
    if not Path(path).exists():
        return f'<div class="fig-pending">{html.escape(caption)}: pending</div>'
    return f'<figure class="fig"><a href="{path}"><img src="{path}" alt="{html.escape(caption)}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>'


def config(pairs: list[tuple[str, str]]) -> str:
    return '<dl class="cfg">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def status(done: int, total: int) -> str:
    cls = "done" if done >= total else "run" if done else "todo"
    word = "complete" if done >= total else "running" if done else "queued"
    return f'<span class="pill {cls}">{word} · {done}/{total} runs</span>'


def fit_summary(path: Path) -> str:
    if not path.exists():
        return "<p class='muted'>Fit pending (needs the full grid).</p>"
    text = path.read_text()
    arms = re.findall(
        r"## (.+?)\n.*?test_metric__auroc[^:]*: E=([\d.e+-]+), A=([\d.e+-]+), alpha=([\d.e+-]+), B=([\d.e+-]+), beta=([\d.e+-]+).*?R2=([\d.]+)",
        text,
        re.DOTALL,
    )
    rows = "".join(
        f"<tr><td>{a}</td><td>{float(al):.4f}</td><td>{float(be):.3f}</td><td>{float(e):.1f}</td><td>{float(r2):.2f}</td></tr>"
        for a, e, _A, al, _B, be, r2 in arms
    )
    weak = "weakly_identified_exponent" in text
    note = (
        '<p class="callout warn"><b>Not identifiable.</b> Both fits put the asymptote E on its bound and the width '
        "exponent α near 0, with 95% intervals spanning orders of magnitude. AUROC still rises roughly linearly in "
        "log(patients) over 1K–100K, so there is no plateau to pin a power law to. Read the curves as smooth "
        "descriptions of the measured range, not as exponents to extrapolate.</p>"
        if weak
        else ""
    )
    return (
        '<div class="scroll"><table class="grid small"><thead><tr><th>Arm</th><th>α (width)</th><th>β (data)</th>'
        f"<th>E (logit)</th><th>R²</th></tr></thead><tbody>{rows}</tbody></table></div>{note}"
    )


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--accent:#2a78d6;
--po:#2a78d6;--ret:#eb6834;--rnd:#1baf7a;--nul:#eda100;--ok:#0ca30c;--warn:#b87700;--run:#2a78d6;
--code:#f0efec;--shadow:0 1px 2px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.04)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#97968e;--line:#2e2e2b;--accent:#3987e5;--code:#232321;--shadow:none}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#97968e;--line:#2e2e2b;
--accent:#3987e5;--code:#232321;--shadow:none}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 16px 80px}
header h1{font-size:28px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em}
header p{margin:0;color:var(--ink2);max-width:780px}
.meta{color:var(--muted);font-size:13px;margin-top:10px}
nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin:24px -16px 0;padding:10px 16px;
display:flex;gap:18px;overflow-x:auto;font-size:14px}
nav a{color:var(--ink2);text-decoration:none;white-space:nowrap}nav a:hover{color:var(--ink)}
section{margin-top:44px}h2{font-size:21px;margin:0 0 4px}h3{font-size:16px;margin:0}
.lead{color:var(--ink2);margin:0 0 18px;max-width:820px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}
.card .top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px}
.card p{margin:6px 0 12px;color:var(--ink2);font-size:14px}
.pill{font-size:12px;border-radius:999px;padding:2px 10px;white-space:nowrap;border:1px solid currentColor}
.pill.done{color:var(--ok)}.pill.run{color:var(--run)}.pill.todo{color:var(--muted)}
.cfg{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0;font-size:13px}
.cfg dt{color:var(--muted)}.cfg dd{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;overflow-wrap:anywhere}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:12px;margin-bottom:22px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .v{font-size:26px;font-weight:600;letter-spacing:-.01em}.tile .l{font-size:13px;color:var(--ink2)}
.tile .s{font-size:12px;color:var(--muted);margin-top:2px}
.scroll{overflow-x:auto}
table.grid{border-collapse:collapse;font-size:13px;min-width:100%}
.grid th,.grid td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:middle}
.grid thead th{color:var(--muted);font-weight:500}
.grid td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink)}
.grid td.num span{display:block;font-size:11px;color:var(--ink2)}
.grid td.tinted{color:#0b0b0b}.grid td.tinted span{color:#52514e}
.grid.small td{white-space:nowrap}
.grid td.sig b{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
.grid td.pending{color:var(--muted);font-style:italic;text-align:center}
.grid tr.first th,.grid tr.first td{border-top:1px solid var(--line)}
.grid td.arm{white-space:nowrap;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:0}
.sw.patient_only{background:var(--po)}.sw.retrieval{background:var(--ret)}
.sw.retrieval_random_docs{background:var(--rnd);border-radius:2px}.sw.retrieval_null_docs{background:var(--nul);clip-path:polygon(50% 0,100% 100%,0 100%);border-radius:0}
figure{margin:0}.tbl{margin:14px 0 26px}.tbl figcaption,.fig figcaption{font-size:13px;color:var(--ink2);margin:6px 0 8px}
.fig{background:#fcfcfb;border:1px solid var(--line);border-radius:12px;padding:10px;margin:14px 0 22px}
.fig img{display:block;width:100%;height:auto;border-radius:6px}.fig figcaption{padding:0 4px;color:#52514e}
.fig-pending{border:1px dashed var(--line);border-radius:12px;padding:28px;text-align:center;color:var(--muted);margin:14px 0}
.callout{border-left:3px solid var(--accent);background:var(--surface);padding:10px 14px;border-radius:0 8px 8px 0;font-size:14px;color:var(--ink2)}
.callout.warn{border-color:var(--warn)}.muted{color:var(--muted)}
ul.notes{padding-left:18px;color:var(--ink2);font-size:14px}ul.notes li{margin:6px 0}
pre{background:var(--code);border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.5}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:4px 0 6px}
.key{font-size:12px;color:var(--muted);margin-top:-14px;margin-bottom:18px}
"""


def main() -> None:
    results = collect_results()
    draws = per_draw(results)
    have = lambda f: draws.filter(f).height
    main25 = have((pl.col("t") == 25) & pl.col("arm").is_in(["patient_only", "retrieval"]))
    main625 = have((pl.col("t") == 625) & pl.col("arm").is_in(["patient_only", "retrieval"]))
    abl = have((pl.col("dim") == ABLATION_DIM) & pl.col("arm").is_in(["patient_only", "retrieval"]))
    ctrl = have(pl.col("arm").is_in(list(CONTROL_ARMS)))

    ret = paired(draws, "retrieval")
    sig25 = [k for k, (m, hw, *_r) in ret.items() if k[2] == 25 and m - hw > 0]
    cells25 = [k for k in ret if k[2] == 25]
    best = max(((k, v) for k, v in ret.items() if k[2] == 25), key=lambda kv: kv[1][0], default=None)
    big = [v for k, v in ret.items() if k[2] == 25 and k[0] >= 64]
    po100 = draws.filter((pl.col("arm") == "patient_only") & (pl.col("t") == 25) & (pl.col("n") == 100000))[
        "auroc"
    ]
    re100 = draws.filter((pl.col("arm") == "retrieval") & (pl.col("t") == 25) & (pl.col("n") == 100000))[
        "auroc"
    ]

    tiles = []
    if best:
        (bd, bn, _), (bm, *_r) = best
        tiles.append(
            (f"+{bm:.3f}", "Largest retrieval gain (AUROC)", f"dim {bd}, {compact(bn)} patients, 25 tasks")
        )
    if cells25:
        tiles.append(
            (
                f"{len(sig25)}/{len(cells25)}",
                "Cells where retrieval is significantly better",
                "paired over 5 draws, 95% CI > 0",
            )
        )
    if big:
        tiles.append(
            (
                f"{max(abs(v[0]) for v in big):.3f}",
                "Largest |gap| at dim 64–128",
                "no cell significant either way",
            )
        )
    if len(po100) and len(re100):
        tiles.append(
            (
                f"{po100.mean():.2f} / {re100.mean():.2f}",
                "Mean AUROC at 100K patients",
                "patient only / retrieval, all widths",
            )
        )
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{lbl}</div><div class="s">{s}</div></div>'
        for v, lbl, s in tiles
    )

    experiments = [
        (
            "Main scaling grid · 25 tasks",
            "Every encoder width × training-set size for both arms. This is the grid the scaling law is fit on, and it matches the earlier N=25 results.",
            [
                ("arms", "patient_only, retrieval"),
                ("encoder.embedding_dim", "4, 8, 16, 32, 64, 128"),
                ("training patients", "1K, 3K, 10K, 30K, 100K"),
                ("tasks", "25 random codes, 30-day horizon"),
                ("draws", "5 (task seed 101–505, train seed = draw)"),
            ],
            status(main25, 300),
        ),
        (
            "Task-count ablation · dim 32",
            "Does retrieval help more or less as the shared encoder trains on more tasks? Fixed width, every data size.",
            [
                ("arms", "patient_only, retrieval"),
                ("encoder.embedding_dim", "32"),
                ("training patients", "1K – 100K (5 sizes)"),
                ("tasks", "5, 25, 125, 625, 2,500"),
                ("draws", "5 per cell"),
            ],
            status(abl, 250),
        ),
        (
            "Main scaling grid · 625 tasks",
            "The same grid at 625 tasks, where the task draw adds 3–5× less noise across draws. This will be the cleaner version of the scaling surface.",
            [
                ("arms", "patient_only, retrieval"),
                ("encoder.embedding_dim", "4 – 128 (6 widths)"),
                ("training patients", "1K – 100K (5 sizes)"),
                ("tasks", "625 random codes"),
                ("draws", "5 (dim-32 cells shared with the ablation)"),
            ],
            status(main625, 300),
        ),
        (
            "Knowledge vs capacity control",
            "Is the gain the documents' content, or the retrieval arm's extra 256-d fusion and head? The same retrieval model is trained <b>and</b> tested with documents that carry no knowledge.",
            [
                ("arms", "retrieval_random_docs, retrieval_null_docs"),
                ("random docs", "retriever.ablation_mode=random_docs"),
                ("null docs", "16 empty docs, zero embeddings"),
                ("slice", "dim 4, 16, 64 × 1K, 10K, 100K"),
                ("tasks / draws", "625 / 5"),
            ],
            status(ctrl, 90),
        ),
    ]
    exp_html = "".join(
        f'<div class="card"><div class="top"><h3>{t}</h3>{s}</div><p>{d}</p>{config(c)}</div>'
        for t, d, c, s in experiments
    )

    arms_html = (
        '<div class="cards">'
        '<div class="card"><div class="top"><h3><i class="sw patient_only"></i>Patient only</h3></div>'
        "<p>Codes → RoPE transformer encoder → masked-mean pool → linear head (one logit per task).</p>"
        + config(
            [
                ("encoder", "rope · 1 layer · 1 head · ff_dim = 2·dim"),
                ("fusion", "passthrough"),
                ("head", "linear, in_dim = dim"),
                ("loss", "multitask_binary_bce"),
            ]
        )
        + '</div><div class="card"><div class="top"><h3><i class="sw retrieval"></i>Retrieval</h3></div>'
        "<p>Same encoder. A learned query fetches 4 textbook passages, and per-document cross-attention fuses them; predictions are marginalized over documents.</p>"
        + config(
            [
                ("query", "sequence_mean → linear → 1024-d"),
                ("corpus", "MedRAG/textbooks · Qwen3-Embedding-0.6B · FAISS"),
                ("retriever", "hf_dataset · k = 4"),
                ("fusion", "cross_attention_perdoc_medium (d 256)"),
                ("loss", "multitask_binary_bce_marginalized"),
            ]
        )
        + '</div><div class="card"><div class="top"><h3>Shared training</h3></div>'
        "<p>Identical across arms and cells, apart from the per-size schedule.</p>"
        + config(
            [
                ("cohort", "MIMIC-IV MEDS, rebuilt from raw (min 100 subjects/code, 10 events/subject)"),
                ("context", "last 256 events"),
                ("optimizer", "lr 1e-3 · cosine with warmup · clip 1.0 · batch 32"),
                ("schedule (epochs / warmup)", "1K 10/60 · 3K 7/100 · 10K 5/200 · 30K 4/200 · 100K 3/200"),
                ("eval", "last.ckpt on held_out, mean task AUROC"),
            ]
        )
        + "</div></div>"
    )

    t625_ready = Path("t625/figures/fig3_retrieval_advantage.png").exists() and main625 >= 300
    abl_table = (
        '<figure class="tbl"><figcaption>Task-count ablation, dim 32: retrieval − patient only, paired over draws</figcaption>'
        '<div class="scroll"><table class="grid"><thead><tr><th></th>'
        + "".join(f"<th>{compact(n)}</th>" for n in NS)
        + "</tr></thead><tbody>"
        + "".join(
            f"<tr><th>{t:,} tasks</th>"
            + "".join(
                (
                    f'<td class="num tinted {"sig" if ret[(ABLATION_DIM, n, t)][0] - ret[(ABLATION_DIM, n, t)][1] > 0 or ret[(ABLATION_DIM, n, t)][0] + ret[(ABLATION_DIM, n, t)][1] < 0 else ""}" '
                    f'style="background:{tint(ret[(ABLATION_DIM, n, t)][0])}"><b>{ret[(ABLATION_DIM, n, t)][0]:+.3f}</b>'
                    f"<span>[{ret[(ABLATION_DIM, n, t)][0] - ret[(ABLATION_DIM, n, t)][1]:+.3f}, {ret[(ABLATION_DIM, n, t)][0] + ret[(ABLATION_DIM, n, t)][1]:+.3f}] · {ret[(ABLATION_DIM, n, t)][2]}/{ret[(ABLATION_DIM, n, t)][3]}</span></td>"
                )
                if (ABLATION_DIM, n, t) in ret
                else '<td class="pending">pending</td>'
                for n in NS
            )
            + "</tr>"
            for t in ABLATION_TASKS
        )
        + "</tbody></table></div></figure>"
    )

    body = f"""
<div class="wrap">
<header>
  <h1>Retrieval vs patient-only: scaling laws on MIMIC-IV</h1>
  <p>Does retrieving medical-textbook passages help a patient-sequence model, and does that help shrink, hold or grow
  as the encoder and the training set get bigger? Each grid cell trains both arms on the same tasks and patients, then compares test AUROC.</p>
  <div class="meta">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} · {results.height} test evaluations · medrap-experiments PR #47 (<code>mimic_iv_scaling_laws/</code>)</div>
</header>
<nav><a href="#experiments">Experiments</a><a href="#arms">Models &amp; configs</a><a href="#headline">Headline</a>
<a href="#main25">Scaling grid (25 tasks)</a><a href="#ablation">Task-count ablation</a><a href="#main625">625 tasks</a>
<a href="#control">Knowledge control</a><a href="#notes">Caveats</a><a href="#repro">Reproduce</a></nav>

<section id="experiments"><h2>Experiments</h2>
<p class="lead">Four experiments share one cohort, one task-generation recipe and one training setup. Each cell runs 5 draws, and both arms use the same draws, so their differences are paired.</p>
<div class="cards">{exp_html}</div></section>

<section id="arms"><h2>Models &amp; configs</h2>
<p class="lead">Encoder width is the only capacity knob, so the parameter count N is dominated by the 65,536 × dim code-embedding table.
Tasks ask whether code X occurs within 30 days of a prediction time sampled uniformly over the patient's clinical events. Codes are drawn uniformly at random, and any code that is all-positive or all-negative in a split is redrawn.</p>
{arms_html}</section>

<section id="headline"><h2>Headline</h2>
<p class="lead">From the 25-task grid. Retrieval helps a lot when the encoder or the data is small, and the gap closes as both grow.</p>
<div class="tiles">{tiles_html}</div></section>

<section id="main25"><h2>Scaling grid · 25 tasks</h2>
<p class="lead">Each cell shows retrieval − patient-only test AUROC, averaged over draws, with its 95% CI and the number of draws retrieval won. Underlined values are significant.</p>
<div class="legend"><span><i class="sw retrieval"></i>orange: retrieval better</span><span><i class="sw patient_only"></i>blue: patient only better</span></div>
{diff_table(ret, 25, DIMS, NS, "Retrieval − patient only, 25 tasks (paired, 95% CI · draws won)")}
{figure("figures/fig1_data_scaling.png", "AUROC vs training patients by encoder width. Points are draw means ± 1 std; lines are the fitted separable power law (median, 95% band).")}
{figure("figures/fig2_capacity_scaling.png", "AUROC vs patient-encoder parameters, by training-set size.")}
{mean_table(draws, 25, DIMS, NS, ["patient_only", "retrieval"], "Test AUROC per arm (mean ± std over 5 draws), 25 tasks")}
<h3 style="margin-top:8px">Fitted law</h3>
<p class="lead" style="margin-top:6px">AUROC ≈ E + A·N<sup>−α</sup> + B·D<sup>−β</sup> per arm, on the logit scale (simple-scaling-laws, wild cluster bootstrap).</p>
{fit_summary(Path("tables/table4_fitted_laws.md"))}
<h3 style="margin-top:26px">Test loss</h3>
<p class="lead" style="margin-top:6px">Both arms log the same test loss: binary cross-entropy of their final predicted probability
(retrieval's is the marginal over its 4 documents), averaged over patient–task pairs. Lower is better, so <b>negative = retrieval better</b>.
Retrieval's loss stays flat at ~0.03–0.04 in every cell. Patient only starts far higher for small models on little data, then
falls to meet it at dim 64–128 or ~100K patients. Loss also rewards calibration, which is why the small-model gap is much larger here than in AUROC.</p>
{diff_table(paired(per_draw(results, "test_loss__bce"), "retrieval"), 25, DIMS, NS, "Retrieval − patient only, test loss, 25 tasks (paired, 95% CI · draws retrieval won)", lim=-0.005, dec=4)}
{figure("figures/fig1_data_scaling_loss.png", "Test loss vs training patients by encoder width (log y-axis; means joined, since the power-law fit to loss is poor).")}
{figure("figures/fig2_capacity_scaling_loss.png", "Test loss vs patient-encoder parameters, by training-set size.")}
</section>

<section id="ablation"><h2>Task-count ablation · dim 32</h2>
<p class="lead">Each task count draws its own random codes. Patient-only spread across draws falls from ~0.1 at 5 tasks to ~0.01 at 625+, so the many-task rows are the trustworthy ones.</p>
{abl_table}
{figure("figures/fig4a_task_count_ablation.png", "Both arms vs training patients, one panel per task count.")}
{figure("figures/fig4b_task_count_advantage.png", "Retrieval advantage vs training patients, one line per task count.")}
</section>

<section id="main625"><h2>Scaling grid · 625 tasks</h2>
<p class="lead">The same grid with less task-sampling noise. {"" if t625_ready else "Still running; the table fills in as results are synced and the report is regenerated."}</p>
{diff_table(ret, 625, DIMS, NS, "Retrieval − patient only, 625 tasks (paired, 95% CI · draws won)")}
{figure("t625/figures/fig1_data_scaling.png", "AUROC vs training patients by encoder width, 625 tasks") if t625_ready else ""}
{figure("t625/figures/fig3_retrieval_advantage.png", "Retrieval advantage, 625 tasks") if t625_ready else ""}
</section>

<section id="control"><h2>Knowledge vs capacity control</h2>
<p class="lead">If the random-docs and null-docs arms match full retrieval, the gain comes from extra capacity. If they fall back to patient only, it comes from what the documents contain.</p>
{mean_table(draws, CONTROL_NUM_TASKS, CONTROL_DIMS, CONTROL_NS, ["patient_only", "retrieval", *CONTROL_ARMS], f"Test AUROC (mean ± std over 5 draws), {CONTROL_NUM_TASKS} tasks")}
{figure("figures/fig5_knowledge_control.png", "Retrieval vs its no-knowledge controls")}
</section>

<section id="notes"><h2>Caveats</h2><ul class="notes">
<li><b>Draws mix noise sources.</b> Each draw changes the task codes, the training-patient subsample and the training seed together, so task-choice noise and optimization noise can't be separated.</li>
<li><b>Capacity confound.</b> The retrieval arm's 256-d fusion and head are fixed across the grid and dwarf small encoders. The knowledge control above tests this directly.</li>
<li><b>Rare tasks.</b> Uniform random codes are mostly rare lab-value bins: median ~0.5–1.7% positives, and in a sample run some tasks had only a few held-out positives. Small-N cells and few-task draws are noisy.</li>
<li><b>Schedule.</b> Epochs scale with N, so the data axis also changes optimizer steps. Patient only stalls at 3K patients with 125+ tasks (≈0.58, then 0.75 at 10K), which creates the 3K spike in the ablation.</li>
<li><b>Not parameter-matched.</b> N counts only the patient encoder, and most of it is the embedding table (65,536 rows, ~10K of them used).</li>
</ul></section>

<section id="repro"><h2>Reproduce</h2>
<pre>cd mimic_iv_scaling_laws &amp;&amp; uv sync
ln -s …/mimic_iv_sweep/data/retrieval_db data/retrieval_db
.venv/bin/python scripts/make_null_retrieval_db.py
scripts/submit_tasks.sh                     # rebuild cohort from raw MEDS, then generate tasks
scripts/submit_sweep.sh afterok:&lt;ids&gt;      # every cell; reruns only fill in missing draws
.venv/bin/python scripts/plot_results.py --out . &amp;&amp; .venv/bin/python scripts/make_report.py</pre>
</section>
</div>"""
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Retrieval Scaling Laws</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    Path("report.html").write_text(doc)
    print(f"wrote {Path('report.html').resolve()}")


if __name__ == "__main__":
    main()
