#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSummary:
    sample_count: int
    total_repos: int
    total_months: int

    def summary(self) -> str:
        return f"{self.sample_count} samples, {self.total_repos} repos, {self.total_months} months"


@dataclass(frozen=True)
class DescriptiveStatistics:
    mean: float
    median: float
    q1: float
    q3: float
    iqr: float
    variance: float
    std: float

    def summary(self) -> str:
        return (
            f"mean={self.mean:.4f}, median={self.median:.4f}, "
            f"IQR=[{self.q1:.4f}, {self.q3:.4f}] (range={self.iqr:.4f}), "
            f"variance={self.variance:.4f}, std={self.std:.4f}"
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
class SummaryRow:
    metric_key: str
    content_type: str
    category: Optional[str]
    dataset_summary: DatasetSummary
    descriptive_statistics: DescriptiveStatistics
    trend_result: TrendResult
    month_comparison_result: Optional[GroupTestResult]
    app_comparison_result: Optional[GroupTestResult]
    category_comparison_result: Optional[GroupTestResult]

