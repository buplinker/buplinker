#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))
from root_util import ContentType, CategoryType  # noqa: E402

# Import from same directory
from analysis.timeline.statistics_types import (  # noqa: E402
    DatasetSummary,
    DescriptiveStatistics,
    TrendResult,
    GroupTestResult,
    SummaryRow,
)

class InsufficientDataError(RuntimeError):
    """Raised when we cannot run a requested statistical test."""


class StatisticsAnalyzer:
    """Statistical analysis for timeline metrics."""
    
    def __init__(self, metric_key: str, limited: bool = False):
        """
        Initialize the statistics analyzer.
        
        Args:
            limited: If True, use limited_years data; otherwise use all_years data.
        """
        self.metric_key = metric_key
        self.limited = limited

        output_root = Path(__file__).parent / "results" / metric_key / ("limited_years" if limited else "all_years")
        self.csv_path = output_root / "csv"
        self.statistics_path = output_root / "statistics"
        self.statistics_path.mkdir(parents=True, exist_ok=True)

    def load_metric_data(self, content_type: ContentType) -> pd.DataFrame:
        """Load metric data from CSV files."""
        suffix = f"{content_type.value}_{self.metric_key}"
        return self._load_csv_data(suffix)
    
    def load_category_data(self, category: CategoryType, content_type: ContentType) -> pd.DataFrame:
        """Load category data from CSV files."""
        suffix = f"{category.value}_{content_type.value}_{self.metric_key}"
        return self._load_csv_data(suffix)
    
    def _load_csv_data(self, suffix: str) -> pd.DataFrame:
        df = self._load_per_repo_data(suffix, value_col="metric")
        if df.empty:
            return df
        return df.sort_values(["elapsed_month", "repo"]).reset_index(drop=True)


    def _load_per_repo_data(self, suffix: str, value_col: str) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        csv_files = sorted(
            p for p in self.csv_path.glob(f"*_{suffix}.csv")
            if p.is_file() and not p.name.startswith("all_repos_")
        )
        for csv_path in csv_files:
            repo_name = csv_path.stem.replace(f"_{suffix}", "")
            category_tokens = (CategoryType.Hedonic.value, CategoryType.Utilitarian.value)
            category = next((part for part in csv_path.stem.split("_") if part in category_tokens), None)
            df = pd.read_csv(csv_path)

            if value_col not in df.columns:
                continue

            df["elapsed_month"] = pd.to_numeric(df["elapsed_month"], errors="coerce")
            df = df.dropna(subset=["elapsed_month", value_col])
            if df.empty:
                continue

            df["elapsed_month"] = df["elapsed_month"].astype(int)
            df["repo"] = repo_name
            df[value_col] = df[value_col].astype(float)
            df["category"] = category

            records.extend(df[["repo", "elapsed_month", "category", value_col]].to_dict("records"))
        
        return pd.DataFrame(records)


    def _load_all_repos_data(self, csv_path: Path, value_col: str) -> pd.DataFrame:
        empty_df = pd.DataFrame(columns=["repo", value_col, "elapsed_month"])
        if not csv_path.exists():
            return empty_df

        df = pd.read_csv(csv_path)
        if value_col not in df.columns or "elapsed_month" not in df.columns:
            return empty_df

        df = df.dropna(subset=["elapsed_month", value_col]).copy()
        df["elapsed_month"] = pd.to_numeric(df["elapsed_month"], errors="coerce")
        df = df.dropna(subset=["elapsed_month"])
        
        df["elapsed_month"] = df["elapsed_month"].astype(int)
        df["repo"] = "all_repos"
        df[value_col] = df[value_col].astype(float)
        
        return df[["repo", "elapsed_month", value_col]]
    

    def analyse_content_type(self, df: pd.DataFrame, content_type: ContentType, category: Optional[CategoryType] = None) -> Optional[SummaryRow]:
        """Analyze content type and return summary row."""
        value_col = "metric"

        dataset_summary = self.describe_dataset(df)
        desc_stats = self.compute_descriptive_stats(df, value_col)

        elapsed_month_groups = self.prepare_compared_value_groups(df, value_col, compared_col="elapsed_month", min_group_size=1)
        trend_result = self.run_trend_analysis(elapsed_month_groups)

        month_comparison_result = self.run_group_comparison(elapsed_month_groups)
        app_groups = self.prepare_compared_value_groups(df, value_col, compared_col="repo", min_group_size=1)
        app_comparison_result = self.run_group_comparison(app_groups)
        category_comparison_result = None
        if category is None:
            print("category is None")
            category_groups = self.prepare_compared_value_groups(df, value_col, compared_col="category", min_group_size=1)
            category_comparison_result = self.run_group_comparison(category_groups)
       
        return SummaryRow(
            metric_key=self.metric_key,
            content_type=content_type.value.upper(),
            category=category.value if category else None,
            dataset_summary=dataset_summary,
            descriptive_statistics=desc_stats,
            trend_result=trend_result,
            month_comparison_result=month_comparison_result,
            app_comparison_result=app_comparison_result,
            category_comparison_result=category_comparison_result,
        )


    def describe_dataset(self, df: pd.DataFrame) -> DatasetSummary:
        if df.empty:
            return DatasetSummary(sample_count=0, total_repos=0, total_months=0)

        sample_count = len(df)
        total_repos = df["repo"].nunique()
        total_months = df["elapsed_month"].nunique()

        return DatasetSummary(
            sample_count=sample_count,
            total_repos=total_repos,
            total_months=total_months
        )

    def compute_descriptive_stats(self, df: pd.DataFrame, value_col: str) -> DescriptiveStatistics:
        if df.empty or value_col not in df.columns:
            return DescriptiveStatistics(
                mean=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0, variance=0.0, std=0.0
            )
        values = df[value_col].dropna()
        if values.empty:
            return DescriptiveStatistics(
                mean=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0, variance=0.0, std=0.0
            )

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        variance_val = float(np.var(values, ddof=0))
        std_val = float(np.std(values, ddof=0))

        return DescriptiveStatistics(
            mean=mean_val,
            median=median_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            variance=variance_val,
            std=std_val
        )

    def prepare_compared_value_groups(self, df: pd.DataFrame, value_col: str, compared_col: str, min_group_size=1) -> OrderedDict[str, np.ndarray]:
        if df.empty:
            raise InsufficientDataError(f"No data available to form {compared_col} groups.")

        if compared_col not in df.columns:
            raise ValueError(f"Expected '{compared_col}' column.")

        working_df = df.dropna(subset=[compared_col, value_col]).copy()
        if working_df.empty:
            raise InsufficientDataError(f"No valid {compared_col} data available to form groups.")

        grouped_values = OrderedDict()
        for compared_value, group in working_df.groupby(compared_col):
            values = group[value_col].dropna().to_numpy(dtype=float)
            if values.size >= min_group_size:
                grouped_values[compared_value] = values

        if len(grouped_values) < 2:
            raise InsufficientDataError(f"Not enough {compared_col} meet the minimum group size requirement.")
        return grouped_values

    def _format_grouped_values(self, grouped_data: OrderedDict[str, np.ndarray]) -> OrderedDict[str, np.ndarray]:
        if not grouped_data:
            raise InsufficientDataError("No grouped data available for trend analysis.")

        months, grouped_values = [], []
        for month_key, values in grouped_data.items():
            try:
                month_ordinal = float(month_key)
            except ValueError:
                ts = pd.to_datetime(str(month_key) + "-01", errors="coerce")
                if pd.isna(ts):
                    raise ValueError(f"Cannot interpret month key '{month_key}' for trend analysis.")
                month_ordinal = float(ts.toordinal())
            months.extend([month_ordinal] * len(values))
            grouped_values.extend(values)

        if len(grouped_values) < 3:
            raise InsufficientDataError("Need at least three months for trend analysis.")

        months_arr, values_arr = np.array(months, dtype=float), np.array(grouped_values, dtype=float)
        
        return months_arr, values_arr

    def run_trend_analysis(self, grouped_data: OrderedDict[str, np.ndarray]) -> TrendResult:
        """Compute Mann–Kendall (via Kendall's tau) and Theil–Sen regression on month-averaged values."""
        months_arr, values_arr = self._format_grouped_values(grouped_data)
        tau, tau_pvalue = stats.kendalltau(months_arr, values_arr, nan_policy="omit")
        slope, intercept, lower, upper = stats.theilslopes(values_arr, months_arr, 0.95)

        return TrendResult(
            tau=float(tau),
            tau_pvalue=float(tau_pvalue),
            slope_per_year=float(slope * 12.0),
            intercept=float(intercept),
            slope_ci_per_year=(float(lower * 12.0), float(upper * 12.0)),
            n_obs=len(values_arr),
        )

    def run_group_comparison(self, groups: OrderedDict[str, np.ndarray]) -> GroupTestResult:
        """Run one-way ANOVA when assumptions appear reasonable; otherwise Kruskal–Wallis."""
        samples = list(groups.values())
        if len(samples) < 2:
            raise InsufficientDataError("Need at least two groups for comparison.")

        sample_sizes = {label: int(values.size) for label, values in groups.items()}
        normal_ok = self._check_normality(samples)
        variance_ok = all(v.size >= 2 for v in samples) and self._check_variance_homogeneity(samples)

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


    def _check_normality(self, groups: Iterable[np.ndarray]) -> bool:
        """Return True if all groups pass the Shapiro-Wilk test (p >= 0.05)."""
        return all(
            sample.size >= 3 and stats.shapiro(sample)[1] >= 0.05
            for sample in groups
        )

    def _check_variance_homogeneity(self, groups: Iterable[np.ndarray]) -> bool:
        """Return True if Levene's test indicates equal variances (p >= 0.05)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _, pvalue = stats.levene(*groups)
        return pvalue >= 0.05


    def build_metric_combined_csv_summaries(self, metric_rows: List[SummaryRow]) -> None:
        """Build and save combined CSV summaries."""
        output_path = self.statistics_path / f"statistics_summary_{self.metric_key}.csv"
        self._build_combined_csv_summaries(metric_rows, output_path)
    
    def build_category_combined_csv_summaries(self, metric_rows: List[SummaryRow]) -> None:
        """Build and save combined CSV summaries."""
        output_path = self.statistics_path / f"statistics_summary_category_{self.metric_key}.csv"
        self._build_combined_csv_summaries(metric_rows, output_path)

    def _build_combined_csv_summaries(self, metric_rows: List[SummaryRow], output_file: Path) -> None:
        if not metric_rows:
            return
        
        headers = ["metric_key", "content_type", "category", "dataset_summary", 
                   "descriptive_statistics", "trend_test_summary", "month_comparison_summary", "app_comparison_summary", "category_comparison_summary"]
        
        with output_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows([
                [r.metric_key, r.content_type, r.category if r.category else None, 
                 r.dataset_summary.summary(), r.descriptive_statistics.summary(), r.trend_result.summary(), 
                 r.month_comparison_result.summary() if r.month_comparison_result else None,
                 r.app_comparison_result.summary() if r.app_comparison_result else None,
                 r.category_comparison_result.summary() if r.category_comparison_result else None]
                for r in metric_rows
            ])
        print(f"CSV saved to: {output_file}")