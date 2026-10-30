"""Make polars.scan_parquet default to parallel="columns" for the cohort build.

polars 1.30.x (pinned by MEDS-transforms 0.5.3 / meds-torch-data 0.6.6) picks its
"prefiltered" parallel strategy for an ``is_null`` filter, and on the normalized MIMIC
shards that path panics with "validity mask length must match the number of values".
meds-torch-data's tokenization stage hits it via ``filter(pl.col("time").is_null())``.
"columns" (and "none") read the same files correctly; polars >= 1.32 fixes the bug but
is incompatible with MEDS-transforms 0.5.3. `run.sh cohort` puts this directory on
PYTHONPATH so the patch applies to every pipeline subprocess, and nothing else.
"""

try:
    import functools

    import polars as pl
except ImportError:  # a python without polars (e.g. a system tool) -- nothing to patch
    pass
else:
    _scan_parquet = pl.scan_parquet

    @functools.wraps(_scan_parquet)
    def scan_parquet(*args, **kwargs):
        kwargs.setdefault("parallel", "columns")
        return _scan_parquet(*args, **kwargs)

    pl.scan_parquet = scan_parquet
