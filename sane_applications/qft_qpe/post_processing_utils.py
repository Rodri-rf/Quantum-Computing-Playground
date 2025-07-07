
from __future__ import annotations
import argparse, pathlib, re, json
import numpy  as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, Dropdown

# ─────────────────────────────────────────────────────────────────────────────
# A)  bit-strings  →  eigenvalues  (vectorised, no for-loops, no expansion)
# ─────────────────────────────────────────────────────────────────────────────
import ast

def counts_to_evals(counts: str | dict,
                    num_ancilla: int,
                    time: float) -> np.ndarray:
    """
    Turn the summary frequency dictionary into a flat numpy array.
    It does so by "unpacking" the dictionary as follows: {"result a":5, "results b":2,...}
    turns to [a,a,a,a,a,b,b]
    Note: we assume counts already come as dict, not just strings
    
    """
    # convert bit-strings → decimal vectorised
    bit_strings = np.array(list(counts.keys()))
    shots       = np.fromiter(counts.values(), int)

    decimals = np.array([int(b, 2) for b in bit_strings]) / 2**num_ancilla

    phases = np.where(decimals >= .5, decimals - 1.0, decimals)
    energies = 2 * np.pi  * phases / time          # E = 2π·phase / t

    # replicate according to shot counts and return as 1-D array
    return np.repeat(energies, shots)

###############################################################################
# 1. I/O helpers
###############################################################################

def read_experiment_csv(fname: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(fname)
    df.rename(columns=str.strip, inplace=True) # Make life easier
    df["Exact Eigenvalue"] = df["Exact Eigenvalue"].apply(lambda x : complex(re.sub(r'[()]', '', x)))
    df["Raw results"] = df["Raw results"].apply(lambda x : ast.literal_eval(x))
    return df

###############################################################################
# 2. Summary-statistics builder
###############################################################################

STAT_FUNS = {
    "mean":   np.mean,
    "std":    np.std,
    "median": np.median,
    "q25":    lambda x: np.quantile(x, 0.25),
    "q75":    lambda x: np.quantile(x, 0.75),
    "min":    np.min,
    "max":    np.max,
}

# ─────────────────────────────────────────────────────────────────────────────
# B)  summary *from raw*  (drop-in replacement for build_summary)
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_from_raw(
        df,
        group_cols=("QDRIFT Implementation", "Num Ancilla", "Time"),
        raw_col="Raw results") -> pd.DataFrame:
    """
    1. For every *row* compute stats (mean, std, median, …) from the full
       eigenvalue distribution that `raw_col` encodes.
    2. Group those row-wise stats again by (impl, n, t) – useful if you ran
       several random realisations per setting.
    """

    # explode→stats  (one new DataFrame per row)
    stat_frames = []
    for idx, row in df.iterrows():
        evals = counts_to_evals(row[raw_col], row["Num Ancilla"], row["Time"])
        stats = {f"Eigenvalue_{name}": func(evals) for name, func in STAT_FUNS.items()}
        # stats is smth like { "Eigenvalue_mean":   2.37, "Eigenvalue_std":    0.15,  "Eigenvalue_median": 2.35, ...}
        stat_frames.append(pd.Series(stats, name=idx))

    stats_df = pd.concat(stat_frames, axis=1).T
    merged   = pd.concat([df[group_cols], stats_df], axis=1)

    # ➋ now aggregate in case we had >1 replicate for the same setting
    '''
    agg_dict = {c: (c, np.mean) for c in stats_df.columns}     # mean across runs
    summary  = merged.groupby(list(group_cols)).agg(**agg_dict).reset_index()
    '''
    return merged

def build_summary(
        df,
        group_cols=("QDRIFT Implementation", "Num Ancilla", "Time"),
        metric="Eigenvalue Error"):

    # ➊ build a dict   {"Eigenvalue Error_mean":  (metric, np.mean), ...}
    agg_dict = {f"{metric}_{name}": (metric, func)
                for name, func in STAT_FUNS.items()}

    # ➋ run the group-by using **named** aggregation syntax
    out = df.groupby(list(group_cols)).agg(**agg_dict).reset_index()
    return out


###############################################################################
# 3. Plotting utilities
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# C)  upgraded interactive plot
# ─────────────────────────────────────────────────────────────────────────────
from functools import partial   # small helper

def plot_stat_vs_time(summary: pd.DataFrame, log_x=False):

    impls = summary["QDRIFT Implementation"].unique()
    ns    = sorted(summary["Num Ancilla"].unique())

    # pick all columns that look like   "Eigenvalue_<something>"
    stat_cols = [c for c in summary.columns if c.startswith("Eigenvalue_")]

    @interact(implementation = Dropdown(options=impls, description="Channel"),
              n              = Dropdown(options=ns,    description="Ancilla"),
              stat           = Dropdown(options=stat_cols, description="Statistic"))
    def _view(implementation, n, stat):
        df = summary[(summary["QDRIFT Implementation"] == implementation) &
                     (summary["Num Ancilla"] == n)]
        fig = px.line(df, x="Time", y=stat,
                      title=f"{stat} vs t  (n={n}, {implementation})",
                      markers=True,
                      log_x=log_x)
        fig.update_layout(yaxis_title=stat, xaxis_title="Evolution time t")
        fig.show()


def histogram_widget(df: pd.DataFrame,
                     group_cols=("Num Ancilla", "Time", "QDRIFT Implementation"),
                     value_col="Eigenvalue Error"):
    """
    Explore the full empirical distribution (histogram) for any triple
    (n, t, implementation).
    """
    times  = sorted(df["Time"].unique())
    ns     = sorted(df["Num Ancilla"].unique())
    impls  = df["QDRIFT Implementation"].unique()

    @interact(t   = Dropdown(options=times, description="Time"),
              n   = Dropdown(options=ns,    description="Ancilla"),
              imp = Dropdown(options=impls, description="Channel"))
    def _hist(t, n, imp):
        sel = df[(df["Time"] == t) &
                 (df["Num Ancilla"] == n) &
                 (df["QDRIFT Implementation"] == imp)]

        fig = px.histogram(sel, x=value_col, nbins=40,
                           title=f"Histogram of {value_col} "
                                 f"(t={t}, n={n}, {imp})")
        fig.show()


###############################################################################
# 4. CLI / notebook entry-points
###############################################################################

def analyse_files(files: list[str | pathlib.Path],
                  save_summary: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load → concat → summarise → save → plot widget.
    Returns (raw_df, summary_df).
    """
    df = pd.concat([read_experiment_csv(f) for f in files], ignore_index=True)
    summary = build_summary(df)

    if save_summary:
        out_name = pathlib.Path(files[0]).with_suffix('').name + "_summary.csv"
        summary.to_csv(out_name, index=False)
        print(f"Summary written to  {out_name}")

    return df, summary


def _main():
    parser = argparse.ArgumentParser(description="Analyse QDRIFT-QPE sweep(s)")
    parser.add_argument("csv", nargs="+", help="one or more experiment CSVs")
    parser.add_argument("--no-save", action="store_true",
                        help="do NOT write …_summary.csv to disk")
    args = parser.parse_args()

    raw, summary = analyse_files(args.csv, save_summary=not args.no_save)
    print(summary.head())

    # Non-interactive fallback plot (one figure per implementation)
    for impl in summary["QDRIFT Implementation"].unique():
        df = summary[summary["QDRIFT Implementation"] == impl]
        fig = px.line(df, x="Time", y="Eigenvalue Error_mean",
                      color="Num Ancilla", error_y="Eigenvalue Error_std",
                      title=f"Mean eigenvalue error vs time ({impl})",
                      markers=True)
        fig.write_html(f"{impl.replace(' ', '_')}_error_vs_time.html")
        print(f"Wrote {impl} plot.")


if __name__ == "__main__":
    _main()