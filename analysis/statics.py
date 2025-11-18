#!/usr/bin/env python3
"""
Statistical analyses for timeline outputs.

This script consumes the CSV artefacts produced by:
  * linked_ratio.py (linked / not-linked relevance ratios)
  * linked_time.py (monthly linked_time averages)

It performs:
  1. Month-to-month group comparison using repo-level averages
     (automatically picks one-way ANOVA when normality + equal-variance
      assumptions look reasonable, otherwise falls back to Kruskal–Wallis).
  2. Trend detection on the averaged time-series via Mann–Kendall test
     (implemented with Kendall's tau against time) and Theil–Sen regression.

Outputs are printed to stdout.
"""

from __future__ import annotations

import argparse
import sys
import warnings
import csv
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore[import]
    import pandas as pd  # type: ignore[import]
    from scipy import stats  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "statistics.py requires numpy, pandas, and scipy. "
        "Please install them in the current environment."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from root_util import ContentType  # noqa: E402


BASE_DIR = SCRIPT_DIR
LINKED_RATIO_DIR = BASE_DIR / "results" / "linked_ratio" / "csv"
LINKED_TIME_DIR = BASE_DIR / "results" / "linked_time" / "csv"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"
SUMMARY_FILES = {
    "per-repo": OUTPUT_DIR / "statistics_summary_per-repo.txt",
    "all-repos": OUTPUT_DIR / "statistics_summary_all-repos.txt",
}
COMBINED_OUTPUT_FILES = {
    "linked_ratio": OUTPUT_DIR / "statistics_summary_linked_ratio.csv",
    "linked_time": OUTPUT_DIR / "statistics_summary_linked_time.csv",
}
METRIC_KEY_MAP = {
    "Linked ratio": "linked_ratio",
    "Linked time": "linked_time",
}


class InsufficientDataError(RuntimeError):
    """Raised when we cannot run a requested statistical test."""


@dataclass(frozen=True)
class GroupTestResult:
    method: str
    statistic: float
    pvalue: float
    group_count: int
    sample_sizes: Dict[str, int]

    def summary(self) -> str:
        samples = ", ".join(f"{key}:{value}" for key, value in self.sample_sizes.items())
        return (
            f"method={self.method}, statistic={self.statistic:.4f}, "
            f"p-value={self.pvalue:.4g}, groups={self.group_count}, sizes=({samples})"
        )


@dataclass(frozen=True)
class TrendResult:
    tau: float
    tau_pvalue: float
    slope_per_year: float
    intercept: float
    slope_ci_per_year: Tuple[float, float]
    n_obs: int

    def summary(self) -> str:
        lower, upper = self.slope_ci_per_year
        return (
            f"tau={self.tau:.4f} (p={self.tau_pvalue:.4g}), "
            f"slope={self.slope_per_year:.4f}/year "
            f"[{lower:.4f}, {upper:.4f}], n={self.n_obs}"
        )


@dataclass(frozen=True)
class SummaryRow:
    metric_key: str
    metric_label: str
    view: str
    content_type: str
    dataset_summary: str
    descriptive_statistics: str
    group_summary: str
    trend_summary: str


BLOCK_HEADER_RE = re.compile(r"^=== (?P<label>.+) \((?P<content_type>UR|PR)\) ===$")


def _list_repo_csvs(directory: Path, suffix: str) -> List[Path]:
    """Yield per-repository CSV files while skipping the aggregated all-repo file."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.glob(f"*_{suffix}.csv")
        if p.is_file() and not p.name.startswith("all_repos_")
    )


def _to_month(ts_value: str) -> pd.Timestamp:
    """Convert YYYY-MM (or YYYY-MM-DD) strings into Timestamp objects (month start)."""
    if pd.isna(ts_value):
        return pd.NaT
    text = str(ts_value)
    return pd.to_datetime(text + "-01" if len(text) == 7 else text, errors="coerce")


def parse_summary_file(path: Path, view: str) -> Iterable[SummaryRow]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        header_match = BLOCK_HEADER_RE.match(lines[index])
        if not header_match:
            index += 1
            continue

        metric_label = header_match.group("label")
        content_type = header_match.group("content_type")
        metric_key = METRIC_KEY_MAP.get(metric_label)
        if metric_key is None:
            raise ValueError(f"Unexpected metric label '{metric_label}' in {path}")

        dataset_summary = ""
        descriptive_statistics = "N/A"
        group_summary = "N/A"
        trend_summary = "N/A"

        search_pos = index + 1
        while search_pos < len(lines):
            line = lines[search_pos].strip()
            if not line:
                search_pos += 1
                continue
            if line.startswith("Dataset summary:"):
                dataset_summary = line.partition(":")[2].strip()
            elif line.startswith("Descriptive statistics:"):
                descriptive_statistics = line.partition(":")[2].strip()
            elif line.startswith("Group comparison"):
                group_summary = line.partition(":")[2].strip()
            elif line.startswith("Trend analysis"):
                trend_summary = line.partition(":")[2].strip()
                search_pos += 1
                break
            search_pos += 1

        if not dataset_summary:
            raise ValueError(f"Missing dataset summary for metric '{metric_label}' in {path}")

        yield SummaryRow(
            metric_key=metric_key,
            metric_label=metric_label,
            view=view,
            content_type=content_type,
            dataset_summary=dataset_summary,
            descriptive_statistics=descriptive_statistics,
            group_summary=group_summary,
            trend_summary=trend_summary,
        )

        index = search_pos


def build_combined_csv_summaries() -> None:
    rows_by_metric: Dict[str, List[SummaryRow]] = {key: [] for key in COMBINED_OUTPUT_FILES}
    for view, summary_path in SUMMARY_FILES.items():
        for row in parse_summary_file(summary_path, view):
            rows_by_metric[row.metric_key].append(row)

    headers = ["metric_label", "view", "content_type", "dataset_summary", 
               "descriptive_statistics", "trend_test_summary", "group_test_summary"]
    for metric_key, rows in rows_by_metric.items():
        output_path = COMBINED_OUTPUT_FILES[metric_key]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows([
                [r.metric_label, r.view, r.content_type, r.dataset_summary,
                 r.descriptive_statistics, r.trend_summary, r.group_summary]
                for r in rows
            ])


def _compute_elapsed_month(df: pd.DataFrame) -> pd.DataFrame:
    """Compute elapsed_month from month column if needed."""
    if "elapsed_month" in df.columns:
        df["elapsed_month"] = pd.to_numeric(df["elapsed_month"], errors="coerce")
    if "elapsed_month" not in df.columns or df["elapsed_month"].isna().all():
        if "month" not in df.columns:
            return df
        df["month"] = df["month"].apply(_to_month)
        df = df.dropna(subset=["month"])
        if not df.empty:
            df = df.sort_values("month")
            df["elapsed_month"] = (
                (df["month"].dt.year - df["month"].iloc[0].year) * 12
                + (df["month"].dt.month - df["month"].iloc[0].month)
            )
    return df


def _load_per_repo_data(
    directory: Path, suffix: str, value_col: str, filter_label: bool = False
) -> List[Dict[str, object]]:
    """Load per-repository data from CSV files."""
    records: List[Dict[str, object]] = []
    for csv_path in _list_repo_csvs(directory, suffix):
        repo_name = csv_path.stem.replace(f"_{suffix}", "")
        df = pd.read_csv(csv_path)
        
        if filter_label and "label" not in df.columns:
            continue
        if not filter_label and value_col not in df.columns:
            continue

        df = _compute_elapsed_month(df)
        df = df.dropna(subset=["elapsed_month"])
        if df.empty:
            continue
        df["elapsed_month"] = df["elapsed_month"].astype(int)

        if filter_label:
            value_source = next(
                (col for col in ["value", "count", "linked_ratio"] if col in df.columns), None
            )
            if value_source is None:
                continue
            df = df[df["label"] == "linked"].dropna(subset=[value_source])
            for _, row in df.iterrows():
                records.append({
                    "repo": repo_name,
                    value_col: float(row[value_source]),
                    "month": int(row["elapsed_month"]),
                    "elapsed_month": int(row["elapsed_month"]),
                })
        else:
            df = df.dropna(subset=[value_col])
            for _, row in df.iterrows():
                records.append({
                    "repo": repo_name,
                    value_col: float(row[value_col]),
                    "month": int(row["elapsed_month"]),
                    "elapsed_month": int(row["elapsed_month"]),
                })
    return records


def _load_all_repos_data(
    csv_path: Path, value_col: str, filter_label: bool = False
) -> pd.DataFrame:
    """Load aggregated all-repos data from CSV file."""
    if not csv_path.exists():
        return pd.DataFrame(columns=["repo", "month", value_col, "elapsed_month"])
    
    df = pd.read_csv(csv_path)
    if filter_label:
        if "label" not in df.columns or "value" not in df.columns:
            return pd.DataFrame(columns=["repo", "month", value_col, "elapsed_month"])
        df = df[df["label"] == "linked"]
        value_col_source = "value"
    else:
        if value_col not in df.columns:
            return pd.DataFrame(columns=["repo", "month", value_col, "elapsed_month"])
        value_col_source = value_col
    
    if "elapsed_month" not in df.columns:
        return pd.DataFrame(columns=["repo", "month", value_col, "elapsed_month"])
    
    df = df.dropna(subset=["elapsed_month", value_col_source]).copy()
    df["elapsed_month"] = pd.to_numeric(df["elapsed_month"], errors="coerce")
    df = df.dropna(subset=["elapsed_month"])
    df["elapsed_month"] = df["elapsed_month"].astype(int)
    df["month"] = df["elapsed_month"]
    df["repo"] = "all_repos"
    df[value_col] = df[value_col_source].astype(float)
    return df[["repo", "month", "elapsed_month", value_col]]


def load_linked_ratio_repo_series(
    content_type: ContentType, *, data_scope: str
) -> pd.DataFrame:
    """Load per-repository linked ratios (repo-month averages) for the given content type."""
    suffix = f"{content_type.value}_linked_ratio"
    value_col = "linked_ratio"
    
    if data_scope == "per-repo":
        records = _load_per_repo_data(LINKED_RATIO_DIR, suffix, value_col, filter_label=True)
    elif data_scope == "all-repos":
        csv_path = LINKED_RATIO_DIR / f"all_repos_{suffix}.csv"
        df = _load_all_repos_data(csv_path, value_col, filter_label=True)
        return df.sort_values(["elapsed_month", "month", "repo"]).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown data scope: {data_scope}")
    
    repo_df = pd.DataFrame(records)
    return repo_df.sort_values(["elapsed_month", "month", "repo"]).reset_index(drop=True) if not repo_df.empty else repo_df


def load_linked_time(
    content_type: ContentType, *, data_scope: str
) -> pd.DataFrame:
    """Load per-repository linked_time values for the given content type."""
    suffix = f"{content_type.value}_linked_time"
    value_col = "mean"
    
    if data_scope == "per-repo":
        records = _load_per_repo_data(LINKED_TIME_DIR, suffix, value_col, filter_label=False)
    elif data_scope == "all-repos":
        csv_path = LINKED_TIME_DIR / f"all_repos_{suffix}.csv"
        df = _load_all_repos_data(csv_path, value_col, filter_label=False)
        return df.sort_values(["elapsed_month", "month", "repo"]).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown data scope: {data_scope}")
    
    repo_df = pd.DataFrame(records)
    return repo_df.sort_values(["elapsed_month", "month", "repo"]).reset_index(drop=True) if not repo_df.empty else repo_df


def prepare_elapsed_month_groups(
    df: pd.DataFrame, value_col: str, *, min_group_size: int
) -> OrderedDict[str, np.ndarray]:
    """Group repo-level values by elapsed month and retain groups with enough data."""
    if df.empty:
        raise InsufficientDataError("No data available to form monthly groups.")

    if "elapsed_month" not in df.columns:
        if "repo" not in df.columns or "month" not in df.columns:
            raise ValueError("Expected 'elapsed_month' or both 'repo' and 'month' columns.")
        df = df.dropna(subset=["month"]).copy()
        if df.empty:
            raise InsufficientDataError("No valid month data available to form groups.")
        df["elapsed_month"] = df.groupby("repo")["month"].transform(
            lambda s: (s.dt.year - s.iloc[0].year) * 12 + (s.dt.month - s.iloc[0].month)
        )

    working_df = df.dropna(subset=["elapsed_month", value_col]).copy()
    if working_df.empty:
        raise InsufficientDataError("No valid elapsed month data available to form groups.")
    
    working_df["elapsed_month"] = working_df["elapsed_month"].astype(int)
    grouped_values = OrderedDict()
    
    for elapsed_month, group in working_df.groupby("elapsed_month"):
        values = group[value_col].dropna().to_numpy(dtype=float)
        if values.size >= min_group_size:
            grouped_values[str(int(elapsed_month))] = values

    if len(grouped_values) < 2:
        raise InsufficientDataError("Not enough months meet the minimum group size requirement.")
    return grouped_values


def _check_normality(groups: Iterable[np.ndarray]) -> bool:
    """Return True if all groups pass the Shapiro-Wilk test (p >= 0.05)."""
    return all(
        sample.size >= 3 and stats.shapiro(sample)[1] >= 0.05
        for sample in groups
    )


def _check_variance_homogeneity(groups: Iterable[np.ndarray]) -> bool:
    """Return True if Levene's test indicates equal variances (p >= 0.05)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stat, pvalue = stats.levene(*groups)
    return pvalue >= 0.05


def run_group_comparison(groups: OrderedDict[str, np.ndarray]) -> GroupTestResult:
    """
    Run one-way ANOVA when assumptions appear reasonable; otherwise Kruskal–Wallis.
    """
    samples = list(groups.values())
    if len(samples) < 2:
        raise InsufficientDataError("Need at least two groups for comparison.")

    sample_sizes = {label: int(values.size) for label, values in groups.items()}
    normal_ok = _check_normality(samples)
    variance_ok = False
    if all(values.size >= 2 for values in samples):
        variance_ok = _check_variance_homogeneity(samples)

    if normal_ok and variance_ok:
        try:
            statistic, pvalue = stats.f_oneway(*samples)
            method = "anova"
        except ValueError:
            statistic, pvalue = stats.kruskal(*samples)
            method = "kruskal-wallis"
    else:
        statistic, pvalue = stats.kruskal(*samples)
        method = "kruskal-wallis"

    return GroupTestResult(
        method=method,
        statistic=float(statistic),
        pvalue=float(pvalue),
        group_count=len(samples),
        sample_sizes=sample_sizes,
    )


def run_trend_analysis(grouped_data: OrderedDict[str, np.ndarray]) -> TrendResult:
    """Compute Mann–Kendall (via Kendall's tau) and Theil–Sen regression on month-averaged values."""
    if not grouped_data:
        raise InsufficientDataError("No grouped data available for trend analysis.")

    months, means = [], []
    for month_key, values in grouped_data.items():
        try:
            month_ordinal = float(int(str(month_key)))
        except ValueError:
            ts = pd.to_datetime(str(month_key) + "-01", errors="coerce")
            if pd.isna(ts):
                raise ValueError(f"Cannot interpret month key '{month_key}' for trend analysis.")
            month_ordinal = float(ts.toordinal())
        months.append(month_ordinal)
        means.append(float(np.mean(values)))

    if len(means) < 3:
        raise InsufficientDataError("Need at least three months for trend analysis.")

    months_arr, means_arr = np.array(months, dtype=float), np.array(means, dtype=float)
    tau, tau_pvalue = stats.kendalltau(months_arr, means_arr, nan_policy="omit")
    slope, intercept, lower, upper = stats.theilslopes(means_arr, months_arr, 0.95)
    
    months_per_year = 12.0
    return TrendResult(
        tau=float(tau),
        tau_pvalue=float(tau_pvalue),
        slope_per_year=float(slope * months_per_year),
        intercept=float(intercept),
        slope_ci_per_year=(float(lower * months_per_year), float(upper * months_per_year)),
        n_obs=len(means_arr),
    )


def describe_dataset(df: pd.DataFrame, value_col: str) -> str:
    """Return a short summary string for the repository-month dataset."""
    if df.empty:
        return "no observations"
    total_repos = df["repo"].nunique()
    total_months = df["month"].nunique()
    return f"{len(df)} samples | {total_repos} repos | {total_months} months"


def compute_descriptive_stats(df: pd.DataFrame, value_col: str) -> str:
    """Compute and return descriptive statistics for the dataset."""
    if df.empty or value_col not in df.columns:
        return "N/A"
    values = df[value_col].dropna()
    if values.empty:
        return "N/A"
    
    mean_val, median_val = float(np.mean(values)), float(np.median(values))
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    iqr = q3 - q1
    variance_val, std_val = float(np.var(values, ddof=0)), float(np.std(values, ddof=0))
    
    return (
        f"mean={mean_val:.4f}, median={median_val:.4f}, "
        f"IQR=[{q1:.4f}, {q3:.4f}] (range={iqr:.4f}), "
        f"variance={variance_val:.4f}, std={std_val:.4f}"
    )


class ResultLogger:
    """Collect log messages, print them, and save them to a file."""
    def __init__(self) -> None:
        self._lines: List[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self._lines.append(message)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._lines)
        path.write_text(content + ("\n" if content and not content.endswith("\n") else ""), encoding="utf-8")


def analyse_content_type(
    *,
    content_type: ContentType,
    dataset_name: str,
    df: pd.DataFrame,
    value_col: str,
    min_group_size: int,
) -> List[str]:
    """Run group comparison and trend analysis for a single metric."""
    lines: List[str] = []
    lines.append("")
    lines.append(f"=== {dataset_name} ({content_type.value.upper()}) ===")
    lines.append(f"Dataset summary: {describe_dataset(df, value_col)}")
    desc_stats = compute_descriptive_stats(df, value_col)
    if desc_stats != "N/A":
        lines.append(f"Descriptive statistics: {desc_stats}")

    try:
        groups = prepare_elapsed_month_groups(
            df,
            value_col,
            min_group_size=min_group_size,
        )
    except InsufficientDataError as exc:
        lines.append(f"Skipping: {exc}")
        return lines

    try:
        group_result = run_group_comparison(groups)
        lines.append(f"Group comparison result: {group_result.summary()}")
    except InsufficientDataError as exc:
        lines.append(f"Group comparison skipped: {exc}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Group comparison failed: {exc}")

    try:
        trend_result = run_trend_analysis(groups)
        lines.append(f"Trend analysis result: {trend_result.summary()}")
    except InsufficientDataError as exc:
        lines.append(f"Trend analysis skipped: {exc}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Trend analysis failed: {exc}")

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical tests for timeline outputs (elapsed month differences and trends)."
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Minimum number of repo observations required per month (default: 5).",
    )
    parser.add_argument(
        "--data-scope",
        choices=["per-repo", "all-repos", "both"],
        default="both",
        help=(
            "Data granularity to analyse. 'per-repo' uses individual repository series "
            "and compares them per month. 'all-repos' uses the aggregated all-repository "
            "time series (requires alignment=elapsed). 'both' runs both analyses and "
            "produces combined CSV output (default)."
        ),
    )
    return parser.parse_args()


def run_scope_analysis(*, data_scope: str, min_group_size: int) -> None:
    logger = ResultLogger()
    logger.log("Running elapsed month difference tests and trend analyses...")
    scope_min_group_size = 1 if data_scope == "all-repos" and min_group_size > 1 else min_group_size
    if scope_min_group_size != min_group_size:
        logger.log("Adjusting minimum repo count per month to 1 for all-repos aggregated data.")
    logger.log(f"Minimum repo count per month: {scope_min_group_size}")
    logger.log(f"Data scope: {data_scope}")

    metrics = [
        ("Linked ratio", load_linked_ratio_repo_series, "linked_ratio"),
        ("Linked time", load_linked_time, "mean"),
    ]
    
    all_lines: List[str] = []
    for content_type in [ContentType.UR, ContentType.PR]:
        for dataset_name, load_func, value_col in metrics:
            df = load_func(content_type, data_scope=data_scope)
            all_lines.extend(analyse_content_type(
                content_type=content_type,
                dataset_name=dataset_name,
                df=df,
                value_col=value_col,
                min_group_size=scope_min_group_size,
            ))

    for line in all_lines:
        logger.log(line)

    output_path = OUTPUT_DIR / f"statistics_summary_{data_scope}.txt"
    logger.dump(output_path)
    logger.log(f"Results saved to: {output_path.resolve()}")


def main() -> None:
    args = parse_args()

    if args.data_scope == "both":
        scopes: Sequence[str] = ("per-repo", "all-repos")
    else:
        scopes = (args.data_scope,)

    for scope in scopes:
        run_scope_analysis(
            data_scope=scope,
            min_group_size=args.min_group_size,
        )

    if "per-repo" in scopes and "all-repos" in scopes:
        try:
            build_combined_csv_summaries()
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to build combined CSV summaries: {exc}", file=sys.stderr)
        else:
            csv_paths = ", ".join(str(path.resolve()) for path in COMBINED_OUTPUT_FILES.values())
            print(f"Combined CSV summaries written to: {csv_paths}")


if __name__ == "__main__":
    main()
