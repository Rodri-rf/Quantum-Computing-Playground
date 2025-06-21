import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Any, TypeVar, Generic, Optional
from dataclasses import dataclass
import pandas as pd


def loglog_error_vs_time_with_regression(
    df: pd.DataFrame,
    title_suffix: str,
    true_eigenval: float = None,
    log_transform_data: bool = True,
    show_plot: bool = True,
    agg: bool = False
) -> dict:
    """
    If log_transform_data is True, performs log-log regression per Num Ancilla group
    and plots transformed data + regression lines.
    
    If False, plots raw data using log-log scales (no transformation, no regression).
    Returns R² values per group if regression was performed; otherwise returns None.
    """

    if log_transform_data:
        df = df.copy()
        df["log_time"] = np.log10(df["Time"])
        df["log_error"] = np.log10(df["Eigenvalue Error"]) if not agg else np.log10(df["Mean_Error"])
        x_col = "log_time"
        y_col = "log_error"

        r2_scores = {}
        if show_plot:
            plt.figure(figsize=(8, 6))
        if "Num Ancilla" in df.columns:
            grouped = df.groupby("Num Ancilla") 
            for group_name, group_data in grouped:
                X = group_data[[x_col]]
                y = group_data[y_col]

                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)

                r2 = r2_score(y, predictions)
                r2_scores[group_name] = r2

                slope = model.coef_[0]
                intercept = model.intercept_
                if show_plot:
                    # Scatter and regression
                    plt.scatter(X, y, label=f"Ancilla={group_name}", alpha=0.6)
                    x_vals = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
                    x_vals_df = pd.DataFrame(x_vals, columns=[x_col])
                    y_vals = model.predict(x_vals_df)
                    plt.plot(x_vals, y_vals, linestyle='--')

                    # Annotate with R²
                    x_mid = X.mean().values[0]
                    y_mid = model.predict([[x_mid]])[0]
                    plt.text(x_mid, y_mid, f"$R^2$={r2:.2f}", fontsize=8)
        else:
            # If no "Num Ancilla" column, treat as single group
            X = df[[x_col]]
            y = df[y_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            r2 = r2_score(y, predictions)
            r2_scores["All"] = r2

            slope = model.coef_[0]
            intercept = model.intercept_
            if show_plot:
                # Scatter and regression
                plt.scatter(X, y, label="All Data", alpha=0.6)
                x_vals = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
                x_vals_df = pd.DataFrame(x_vals, columns=[x_col])
                y_vals = model.predict(x_vals_df)
                plt.plot(x_vals, y_vals, linestyle='--')

                # Annotate with R²
                x_mid = X.mean().values[0]
                y_mid = model.predict([[x_mid]])[0]
                plt.text(x_mid, y_mid, f"$R^2$={r2:.2f}", fontsize=8)

        if show_plot:
            plt.xlabel("log(Time)")
            plt.ylabel("log(Error)")
            plt.title(f"Log-Log Regression: {title_suffix}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return r2_scores

    else:
        # Just plot raw data using log-log axes
        grouped = df.groupby("Num Ancilla")
        if show_plot:
            plt.figure(figsize=(8, 6))
            for group_name, group_data in grouped:
                plt.scatter(group_data["Time"], group_data["Eigenvalue Error"] if not agg else group_data["Mean_Error"],
                            label=f"Ancilla={group_name}", alpha=0.6)

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time")
            plt.ylabel("Eigenvalue Error")
            plt.title(f"Log-Log Plot Only (No Regression): {title_suffix}")
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            plt.show()

        return None

def search_best_time_interval_r2(
    df: pd.DataFrame,
    min_duration: float,
    title_suffix: str = "",
    log_transform_data: bool = True,
    show_plot: bool = True,
    agg: bool = False
):
    best_r2 = -np.inf
    best_df = None
    best_range = (None, None)

    times = sorted(df["Time"].unique())
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            t_min = times[i]
            t_max = times[j]
            if t_max - t_min < min_duration:
                continue

            sub_df = df[(df["Time"] >= t_min) & (df["Time"] <= t_max)]

            r2s = loglog_error_vs_time_with_regression(
                sub_df,
                title_suffix=f"{title_suffix} [{t_min:.2e}, {t_max:.2e}]",
                log_transform_data=log_transform_data,
                show_plot=show_plot,
                agg=agg
            )

            if r2s is not None:
                r2_of_largest_qubits = r2s.get(sub_df["Num Ancilla"].max(), -np.inf)
                if r2_of_largest_qubits > best_r2:
                    best_r2 = r2_of_largest_qubits
                    best_df = sub_df
                    best_range = (t_min, t_max)

    print(f"\n✅ Best R² sum = {best_r2:.4f} in interval {best_range}")
    return best_df, best_range, best_r2


T = TypeVar('T')  # Generic search space element type

@dataclass
class RecursiveInfo(Generic[T]):
    max_overall: float
    max_overall_slice: List[T]
    max_prefix: float
    max_prefix_slice: List[T]
    max_suffix: float
    max_suffix_slice: List[T]
    full_slice: List[T]

    def __repr__(self):
        return (f"RecursiveInfo(max_overall={self.max_overall:.4f}, "
                f"max_prefix={self.max_prefix:.4f}, max_suffix={self.max_suffix:.4f})")


def generalized_max_substructure(
    search_space: List[T],
    partition_fn: Callable[[List[T]], Tuple[List[T], List[T]]],
    base_case_fn: Callable[[List[T]], bool],
    base_case_eval_fn: Callable[[List[T]], RecursiveInfo[T]],
    combine_fn: Callable[[RecursiveInfo[T], RecursiveInfo[T]], RecursiveInfo[T]]
) -> RecursiveInfo[T]:
    """
    Generalized divide-and-conquer solver for contiguous substructure optimization.
    """
    if base_case_fn(search_space):
        return base_case_eval_fn(search_space)

    left, right = partition_fn(search_space)
    left_info = generalized_max_substructure(left, partition_fn, base_case_fn, base_case_eval_fn, combine_fn)
    right_info = generalized_max_substructure(right, partition_fn, base_case_fn, base_case_eval_fn, combine_fn)

    return combine_fn(left_info, right_info)

# specific functions when search spadce is a 2d dataframe with x and y columns,and the function is f: R^2 -> R

def dataframe_to_point_search_space(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sort_by_y: bool = True
) -> List[Tuple[float, float]]:
    """
    Converts a DataFrame with x and y columns into a list of (x, y) points,
    optionally sorted by y value.
    """
    points = list(df[[x_col, y_col]].to_records(index=False))
    return sorted(points, key=lambda pt: pt[1]) if sort_by_y else points

def partition_by_y(search_space: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Partitions the search space into two halves based on the y-coordinate.
    """
    mid_index = len(search_space) // 2
    return search_space[:mid_index], search_space[mid_index:]

def base_case_is_single_point(search_space: List[Tuple[float, float]]) -> bool:
    """
    Base case: check if the search space has only one point.
    """
    return len(search_space) == 1

def r2_of_points(search_space: List[Tuple[float, float]]) -> float:
    """
    Computes the R² value of the linear regression fit for the given points.
    """
    if len(search_space) < 2:
        return 0.0  # Not enough points to compute R²

    X = np.array([[x] for x, _ in search_space])
    y = np.array([y for _, y in search_space])

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    return r2_score(y, predictions)

def base_case_eval_r2(search_space: List[Tuple[float, float]]) -> RecursiveInfo[Tuple[float, float]]:
    """
    Base case evaluation: returns the R² value for a single point.
    """
    r2_value = r2_of_points(search_space)
    return RecursiveInfo(
        max_overall=r2_value,
        max_overall_slice=search_space,
        max_prefix=r2_value,
        max_prefix_slice=search_space,
        max_suffix=r2_value,
        max_suffix_slice=search_space,
        full_slice=search_space
    )

def combine_r2_infos(left_info, right_info):
    combined_full = left_info.full_slice + right_info.full_slice

    # Cross slice (suffix of left + prefix of right)
    cross_slice = left_info.max_suffix_slice + right_info.max_prefix_slice
    cross_r2 = r2_of_points(cross_slice)

    # Determine max overall
    max_val = max(left_info.max_overall, right_info.max_overall, cross_r2)
    if max_val == left_info.max_overall:
        max_slice = left_info.max_overall_slice
        
    elif max_val == right_info.max_overall:
        max_slice = right_info.max_overall_slice
    else:
        max_slice = cross_slice

    # Max prefix
    best_prefix = []
    best_val = float('-inf')
    for i in range(1, len(combined_full)+1):
        curr = combined_full[:i]
        val = r2_of_points(curr)
        if val > best_val:
            best_val = val
            best_prefix = curr
    max_prefix = best_val

    # Max suffix
    best_suffix = []
    best_val = float('-inf')
    for i in range(len(combined_full)):
        curr = combined_full[i:]
        val = r2_of_points(curr)
        if val > best_val:
            best_val = val
            best_suffix = curr
    max_suffix = best_val

    return RecursiveInfo(
        max_overall=max_val,
        max_overall_slice=max_slice,
        max_prefix=max_prefix,
        max_prefix_slice=best_prefix,
        max_suffix=max_suffix,
        max_suffix_slice=best_suffix,
        full_slice=combined_full
    )
