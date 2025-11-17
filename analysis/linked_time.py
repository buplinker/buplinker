#!/usr/bin/env python3
"""
Monthly diff_times mean visualisation.

This script focuses on plotting ONLY the monthly mean of diff_times for UR/PR
content, without any moving-average smoothing. The implementation mirrors the
data-loading utilities from plot_diff_times_metrics.py but strips out weekly
handling and moving-average logic so the resulting charts emphasise the raw
per-month trend.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

# Ensure we can import root_util (same approach as the original script)
sys.path.append(str(Path(__file__).parent.parent))
from root_util import ContentType  # noqa: E402
from util import (  # noqa: E402
    Repository,
    target_repos,
    REFERENCE_YEARS_DEFAULT,
    clamp_elapsed_years,
    compute_elapsed_years,
    normalized_time_ticks,
    CONTENT_TYPE_LINE_COLORS,
    CONTENT_TYPE_LINE_STYLES,
    plot_content_type_lines,
    elapsed_years_label,
    ELAPSED_MONTHS_LABEL,
    FONT_SIZE,
    LEGEND_FONT_SIZE,
    STATS_FONT_SIZE,
    apply_elapsed_years_axis,
    apply_elapsed_month_axis,
    compute_elapsed_month_ticks,
    compute_series_stats,
    format_stats_lines,
    render_stats_box,
    compute_trend_parameters,
    plot_trend_line,
    save_figure,
)

CONTENT_TYPE_ORDER = {
    ContentType.UR: 0,
    ContentType.PR: 1,
}


def parse_diff_times(diff_times_str):
    """Parse diff_times column values into lists of ints."""
    if pd.isna(diff_times_str) or diff_times_str == "":
        return []

    try:
        diff_times_list = ast.literal_eval(diff_times_str)
        if isinstance(diff_times_list, list):
            return [
                int(x)
                for x in diff_times_list
                if isinstance(x, (int, float)) and not pd.isna(x)
            ]
    except (ValueError, SyntaxError):
        return []

    return []


class LinkedTimePlotter:
    """Create monthly mean diff_time plots grouped by repository."""

    def __init__(self):
        base_path = Path(__file__).parent.parent
        self.data_path = base_path / "time_processed_data"

        output_root = base_path / "results" / "linked_time"
        self.csv_path = output_root / "csv"
        self.png_path = output_root / "png"

        self.csv_path.mkdir(parents=True, exist_ok=True)
        self.png_path.mkdir(parents=True, exist_ok=True)

    def _content_type_label(self, content_type: ContentType) -> str:
        if content_type == ContentType.UR:
            return "URs"
        if content_type == ContentType.PR:
            return "PRs"
        return content_type.value.upper()

    def _build_line_entry(
        self,
        metrics: Optional[pd.DataFrame],
        content_type: ContentType,
        reference_years: float = REFERENCE_YEARS_DEFAULT,
        *,
        use_elapsed_months: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if metrics is None or metrics.empty:
            return None
        if "mean" not in metrics.columns:
            return None

        if use_elapsed_months:
            if "elapsed_month" not in metrics.columns:
                return None
            x_values = metrics["elapsed_month"].to_numpy(dtype=float)
            y_values = metrics["mean"].to_numpy(dtype=float)
            if x_values.size == 0 or y_values.size == 0:
                return None
            max_elapsed_month = int(np.nanmax(x_values)) if x_values.size > 0 else 0
            ticks, labels = compute_elapsed_month_ticks(max_elapsed_month)
            return {
                "x_values": x_values,
                "linked_ratio": y_values,
                "xticks": ticks,
                "xticklabels": labels,
                "xlabel": ELAPSED_MONTHS_LABEL,
                "title_suffix": self._content_type_label(content_type),
                "axis_type": "months",
                "max_elapsed_month": max_elapsed_month,
            }

        if "elapsed_years" not in metrics.columns:
            return None

        x_values = metrics["elapsed_years"].to_numpy(dtype=float)
        y_values = metrics["mean"].to_numpy(dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            return None

        ticks, labels = normalized_time_ticks(reference_years)
        return {
            "x_values": x_values,
            "linked_ratio": y_values,
            "xticks": ticks,
            "xticklabels": labels,
            "xlabel": elapsed_years_label(reference_years),
            "title_suffix": self._content_type_label(content_type),
            "axis_type": "years",
        }

    # ------------------------------------------------------------------ #
    # Data loading / preparation
    # ------------------------------------------------------------------ #
    def _load_repo_data(self, repo: Repository, content_type: ContentType) -> pd.DataFrame:
        csv_file = (
            self.data_path
            / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}.csv"
        )
        if not csv_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(csv_file)
        df["parsed_diff_times"] = df["diff_times"].apply(parse_diff_times)

        df = df[(df["label"] == 1) & (df["parsed_diff_times"].apply(len) > 0)]
        if df.empty:
            return pd.DataFrame()

        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    def _explode_and_group_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        exploded = df.explode("parsed_diff_times").dropna(subset=["parsed_diff_times"])
        if exploded.empty:
            return pd.DataFrame()

        exploded["month"] = exploded["created_at"].dt.to_period("M").dt.to_timestamp()
        return exploded

    def _aggregate_monthly_metrics(self, data_df: pd.DataFrame) -> pd.DataFrame:
        monthly = (
            data_df.groupby("month")
            .agg(
                mean=("parsed_diff_times", "mean"),
                median=("parsed_diff_times", "median"),
                std=("parsed_diff_times", "std"),
                count=("parsed_diff_times", "count"),
            )
            .reset_index()
            .sort_values("month")
        )
        monthly["std"] = monthly["std"].fillna(0)
        return monthly

    def calculate_monthly_metrics(
        self, repo: Repository, content_type: ContentType
    ) -> pd.DataFrame:
        df = self._load_repo_data(repo, content_type)
        if df.empty:
            return pd.DataFrame()

        exploded = self._explode_and_group_monthly(df)
        if exploded.empty:
            return pd.DataFrame()

        metrics = self._aggregate_monthly_metrics(exploded)
        if metrics.empty:
            return pd.DataFrame()

        metrics = metrics.sort_values("month").reset_index(drop=True)
        month_periods = metrics["month"].dt.to_period("M")
        if month_periods.empty:
            return pd.DataFrame()

        start_period = month_periods.iloc[0]
        start_year = start_period.year
        start_month = start_period.month

        elapsed_months = (
            (month_periods.map(lambda p: p.year) - start_year) * 12
            + (month_periods.map(lambda p: p.month) - start_month)
        ).astype(int)
        metrics["elapsed_month"] = elapsed_months
        metrics["elapsed_years"] = elapsed_months.astype(float) / 12.0
        metrics = clamp_elapsed_years(
            metrics, "elapsed_years", reference_years=REFERENCE_YEARS_DEFAULT
        )
        max_elapsed_months = int(round(REFERENCE_YEARS_DEFAULT * 12))
        metrics = metrics[
            (metrics["elapsed_month"] >= 0)
            & (metrics["elapsed_month"] < max_elapsed_months)
        ]
        return metrics


    def _compute_repo_elapsed_metrics(
        self, repo: Repository, content_type: ContentType, reference_years: float
    ) -> pd.DataFrame:
        """Return per-repository linked time data with elapsed months/years."""
        df = self._load_repo_data(repo, content_type)
        if df.empty:
            return pd.DataFrame()

        exploded = self._explode_and_group_monthly(df)
        if exploded.empty:
            return pd.DataFrame()

        metrics = self._aggregate_monthly_metrics(exploded)
        if metrics.empty:
            return pd.DataFrame()

        metrics = metrics.sort_values("month").reset_index(drop=True)
        metrics["elapsed_years"] = compute_elapsed_years(metrics["month"])
        metrics = clamp_elapsed_years(
            metrics, "elapsed_years", reference_years=reference_years
        )
        if metrics.empty:
            return pd.DataFrame()

        metrics["elapsed_month"] = (
            metrics["elapsed_years"] * 12.0
        ).round().astype(int)
        max_elapsed_months = int(round(reference_years * 12))
        metrics = metrics[
            (metrics["elapsed_month"] >= 0)
            & (metrics["elapsed_month"] < max_elapsed_months)
        ]
        return metrics.reset_index(drop=True)

    def calculate_all_repos_linked_time(
        self, repos: List[Repository], content_type: ContentType
    ) -> pd.DataFrame:
        reference_years = REFERENCE_YEARS_DEFAULT
        max_elapsed_months = int(round(reference_years * 12))
        if max_elapsed_months <= 0:
            return pd.DataFrame()

        sum_values = np.zeros(max_elapsed_months, dtype=float)
        repo_counts = np.zeros(max_elapsed_months, dtype=int)
        total_counts = np.zeros(max_elapsed_months, dtype=float)

        for repo in repos:
            metrics = self._compute_repo_elapsed_metrics(
                repo, content_type, reference_years
            )
            if metrics.empty:
                continue
            months = metrics["elapsed_month"].to_numpy(dtype=int)
            means = metrics["mean"].to_numpy(dtype=float)
            counts = metrics["count"].to_numpy(dtype=float)

            valid_mask = (
                (months >= 0)
                & (months < max_elapsed_months)
                & np.isfinite(means)
            )
            if not np.any(valid_mask):
                continue

            months_valid = months[valid_mask]
            means_valid = means[valid_mask]
            counts_valid = counts[valid_mask]

            sum_values[months_valid] += means_valid
            repo_counts[months_valid] += 1
            total_counts[months_valid] += counts_valid

        valid_bins = repo_counts > 0
        if not np.any(valid_bins):
            return pd.DataFrame()

        elapsed_months = np.arange(max_elapsed_months, dtype=int)[valid_bins]
        averaged_means = sum_values[valid_bins] / repo_counts[valid_bins]
        elapsed_years = elapsed_months.astype(float) / 12.0

        aggregated = pd.DataFrame(
            {
                "elapsed_month": elapsed_months,
                "elapsed_years": elapsed_years,
                "mean": averaged_means,
                "repo_count": repo_counts[valid_bins],
                "count": total_counts[valid_bins],
            }
        )
        return aggregated.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def _setup_axes(self, ax, title: str, axis_type: str = "years", 
                    max_elapsed_month: Optional[int] = None, 
                    reference_years: float = REFERENCE_YEARS_DEFAULT):
        """Setup axes with common styling."""
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Linked Time (days)", fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=FONT_SIZE, labelrotation=0)
        ax.tick_params(axis="y", labelsize=FONT_SIZE)

        if axis_type == "months" and max_elapsed_month is not None:
            apply_elapsed_month_axis(ax, max_elapsed_month)
        else:
            apply_elapsed_years_axis(ax, reference_years)
            ax.set_xlabel(elapsed_years_label(reference_years), fontsize=FONT_SIZE)

    def _annotate_stats(self, ax, metrics: pd.DataFrame):
        if metrics.empty:
            return

        total_months = len(metrics)
        total_points = int(metrics["count"].sum())
        final_mean = metrics["mean"].iloc[-1]
        overall_mean = float(metrics["mean"].mean())
        overall_median = float(metrics["mean"].median())
        if {"elapsed_years", "mean"}.issubset(metrics.columns):
            slope, _ = compute_trend_parameters(
                metrics["elapsed_years"].to_numpy(dtype=float),
                metrics["mean"].to_numpy(dtype=float),
                axis_type="years",
            )
        else:
            slope = None

        ax.text(
            0.02,
            0.98,
            (
                f"Months: {total_months}\n"
                f"Points: {total_points}\n"
                f"Mean: {overall_mean:.1f}d\n"
                f"Median: {overall_median:.1f}d\n"
                f"Final mean: {final_mean:.1f}d"
                + (f"\nSlope: {slope:.2f}days/year" if slope is not None else "")
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=STATS_FONT_SIZE,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def _save_plot(self, path: Path, fig):
        save_figure(path, fig)
        print(f"Saved linked time plot: {path}")

    def _plot_content_type_with_stats(
        self, ax: plt.Axes, entry: Dict[str, Any], content_type: ContentType,
        title: Optional[str] = None, legend_loc: str = "upper right",
        stats_location: str = "upper left", stats_offsets: Optional[Tuple[float, float]] = None
    ) -> bool:
        """Plot a single content type with stats and trend line."""
        if not entry:
            return False
        
        line_entries = {content_type: entry}
        plotted = plot_content_type_lines(
            ax, line_entries, line_colors=CONTENT_TYPE_LINE_COLORS,
            line_styles=CONTENT_TYPE_LINE_STYLES, default_linewidth=2.5,
            sort_key=lambda ct: CONTENT_TYPE_ORDER.get(ct, float("inf")),
        )
        if not plotted:
            return False
        
        series = entry.get("linked_ratio")
        stats = compute_series_stats(series)
        if stats:
            axis_for_entry = entry.get("axis_type", "years")
            slope, intercept = compute_trend_parameters(
                entry.get("x_values"), series, axis_type=axis_for_entry
            )
            stats_lines = format_stats_lines(
                None, stats, include_count=True, slope_per_year=slope, slope_unit="days"
            )
            x_offset, y_offset = stats_offsets or (0, 0)
            render_stats_box(ax, stats_lines, location=stats_location,
                            x_offset=x_offset, y_offset=y_offset)
            plot_trend_line(ax, entry.get("x_values"), slope, intercept,
                           axis_type=axis_for_entry,
                           color=CONTENT_TYPE_LINE_COLORS.get(content_type))
        
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=legend_loc)
        return True

    def _calculate_unified_y_range(self, entries: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate unified Y-axis range from multiple entries."""
        y_min = None
        y_max = None
        for entry in entries:
            if not entry:
                continue
            series = entry.get("linked_ratio")
            if series is not None:
                series_arr = np.asarray(series, dtype=float)
                finite_values = series_arr[np.isfinite(series_arr)]
                if finite_values.size > 0:
                    entry_min = float(np.min(finite_values))
                    entry_max = float(np.max(finite_values))
                    if y_min is None:
                        y_min, y_max = entry_min, entry_max
                    else:
                        y_min = min(y_min, entry_min)
                        y_max = max(y_max, entry_max)
        
        if y_min is not None and y_max is not None:
            y_range = y_max - y_min
            if y_range > 0:
                y_padding = y_range * 0.05
                return max(0, y_min - y_padding), y_max + y_padding
            return max(0, y_min - 1), y_max + 1
        return None, None

    def _configure_axis_for_entry(self, ax: plt.Axes, entry: Dict[str, Any],
                                  line_entries: Dict[ContentType, Dict[str, Any]],
                                  is_subplot: bool = False):
        """Configure axis based on entry type."""
        axis_type = entry.get("axis_type", "years")
        
        if axis_type == "months":
            ticks = entry.get("xticks")
            labels = entry.get("xticklabels")
            max_elapsed = 0
            for e in line_entries.values():
                if not e:
                    continue
                candidate = e.get("max_elapsed_month")
                if candidate is None:
                    x_vals = e.get("x_values")
                    if isinstance(x_vals, np.ndarray) and x_vals.size > 0:
                        candidate = int(np.nanmax(x_vals))
                if candidate is not None:
                    max_elapsed = max(max_elapsed, int(candidate))
            if ticks is None or labels is None:
                ticks, labels = compute_elapsed_month_ticks(max_elapsed)
            ax.set_xlim(0, max(0, max_elapsed))
            if ticks is not None and labels is not None:
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
            xlabel = entry.get("xlabel") or ELAPSED_MONTHS_LABEL
            if is_subplot:
                ax.set_xlabel("", fontsize=FONT_SIZE)
            else:
                ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
        else:
            ticks = entry.get("xticks")
            labels = entry.get("xticklabels")
            if ticks is None or labels is None:
                ticks, labels = normalized_time_ticks(REFERENCE_YEARS_DEFAULT)
            xlabel = entry.get("xlabel", elapsed_years_label(REFERENCE_YEARS_DEFAULT))
            apply_elapsed_years_axis(ax, REFERENCE_YEARS_DEFAULT,
                                    xticks=ticks, xticklabels=labels,
                                    edge_margin_fraction=0.02)
            if is_subplot:
                ax.set_xlabel("", fontsize=FONT_SIZE)
            else:
                ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
            ax.set_xlim(0, REFERENCE_YEARS_DEFAULT)
        
        return entry.get("xlabel") or (ELAPSED_MONTHS_LABEL if axis_type == "months" 
                                     else elapsed_years_label(REFERENCE_YEARS_DEFAULT))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def create_repo_plot(
        self, repo: Repository, content_type: ContentType, metrics: pd.DataFrame
    ):
        if metrics.empty:
            print(f"No monthly data for {repo.owner}.{repo.name} ({content_type.value})")
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(metrics["elapsed_month"], metrics["mean"],
                markersize=4, linewidth=2, label="Linked time", color="teal")
        
        slope, intercept = compute_trend_parameters(
            metrics["elapsed_month"].to_numpy(dtype=float),
            metrics["mean"].to_numpy(dtype=float), axis_type="months"
        )
        plot_trend_line(ax, metrics["elapsed_month"].to_numpy(dtype=float),
                        slope, intercept, axis_type="months", color="teal")
        
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        self._annotate_stats(ax, metrics)
        title = f"{repo.owner}.{repo.name} - {self._content_type_label(content_type)}"
        max_elapsed_month = int(metrics["elapsed_month"].max()) if not metrics.empty else 0
        self._setup_axes(ax, title, axis_type="months", max_elapsed_month=max_elapsed_month)

        output_file = self.png_path / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_linked_time.png"
        self._save_plot(output_file, fig)

    def save_metrics_csv(
        self, repo: Repository, content_type: ContentType, metrics: pd.DataFrame
    ):
        if metrics.empty:
            return

        output_file = (
            self.csv_path
            / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_linked_time.csv"
        )
        metrics_to_save = metrics.copy()
        preferred_order = [
            "elapsed_month",
            "month",
            "elapsed_years",
            "mean",
            "median",
            "std",
            "count",
        ]
        ordered_columns = [col for col in preferred_order if col in metrics_to_save.columns]
        remaining_columns = [col for col in metrics_to_save.columns if col not in ordered_columns]
        metrics_to_save = metrics_to_save[ordered_columns + remaining_columns]
        metrics_to_save.to_csv(output_file, index=False)
        print(f"Saved linked time CSV: {output_file}")

    def create_all_repos_plot(self, content_type: ContentType, metrics: pd.DataFrame):
        if metrics.empty:
            print(f"No monthly data for all repositories ({content_type.value})")
            return

        reference_years = REFERENCE_YEARS_DEFAULT
        plot_metrics = metrics.sort_values("elapsed_month").reset_index(drop=True)
        if plot_metrics.empty:
            print(f"No monthly data within {reference_years:.1f} years for all repositories ({content_type.value})")
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(plot_metrics["elapsed_years"], plot_metrics["mean"],
                linewidth=2, label="Linked time", color="purple")
        
        slope, intercept = compute_trend_parameters(
            plot_metrics["elapsed_years"].to_numpy(dtype=float),
            plot_metrics["mean"].to_numpy(dtype=float), axis_type="years"
        )
        plot_trend_line(ax, plot_metrics["elapsed_years"].to_numpy(dtype=float),
                        slope, intercept, axis_type="years", color="purple")
        
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        self._annotate_stats(ax, plot_metrics)
        self._setup_axes(ax, f"All repositories - {content_type.value.upper()} - Monthly linked_times mean",
                        reference_years=reference_years)
        ax.set_xlim(0, reference_years)

        output_file = self.png_path / f"all_repos_{content_type.value}_linked_time.png"
        self._save_plot(output_file, fig)

    def _create_combined_plot(
        self, line_entries: Dict[ContentType, Dict[str, Any]], 
        output_file: Path, repo: Optional[Repository] = None
    ):
        """Create a single plot with both UR and PR lines together."""
        if not line_entries:
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        plotted = plot_content_type_lines(
            ax, line_entries, line_colors=CONTENT_TYPE_LINE_COLORS,
            line_styles=CONTENT_TYPE_LINE_STYLES, default_linewidth=2.5,
            sort_key=lambda ct: CONTENT_TYPE_ORDER.get(ct, float("inf")),
        )
        if not plotted:
            plt.close(fig)
            return

        sample_entry = next((e for e in line_entries.values() if e), None)
        if sample_entry:
            xlabel = self._configure_axis_for_entry(ax, sample_entry, line_entries, is_subplot=False)
        
        ax.set_ylabel("Linked Time (days)", fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        stats_lines: List[str] = []
        for content_type, entry in sorted(
            line_entries.items(), key=lambda item: CONTENT_TYPE_ORDER.get(item[0], float("inf"))
        ):
            if not entry:
                continue
            series = entry.get("linked_ratio")
            stats = compute_series_stats(series)
            if not stats:
                continue
            title = entry.get("title_suffix") or self._content_type_label(content_type)
            axis_for_entry = entry.get("axis_type", "years")
            slope, intercept = compute_trend_parameters(
                entry.get("x_values"), series, axis_type=axis_for_entry
            )
            stats_lines.extend(format_stats_lines(title, stats, include_count=True, slope_per_year=slope, slope_unit="days"))
            stats_lines.append("")
            plot_trend_line(ax, entry.get("x_values"), slope, intercept,
                axis_type=axis_for_entry,
                           color=CONTENT_TYPE_LINE_COLORS.get(content_type))
        if stats_lines and stats_lines[-1] == "":
            stats_lines.pop()
        render_stats_box(ax, stats_lines, location="center left")

        self._save_plot(output_file, fig)

    def create_repo_combined_plot(
        self, repo: Repository, line_entries: Dict[ContentType, Dict[str, Any]]
    ):
        """Create a single plot with both UR and PR lines together for a repo."""
        output_file = self.png_path / f"{repo.id}_{repo.owner}.{repo.name}_ur_pr_linked_time.png"
        self._create_combined_plot(line_entries, output_file, repo)

    def create_all_repos_combined_plot(self, line_entries: Dict[ContentType, Dict[str, Any]]):
        """Create a single plot with both UR and PR lines together for all repositories."""
        output_file = self.png_path / "all_repos_ur_pr_linked_time.png"
        self._create_combined_plot(line_entries, output_file)

    def _create_ur_pr_plot(
        self, line_entries: Dict[ContentType, Dict[str, Any]], output_file: Path,
        repo: Optional[Repository] = None
    ):
        """Create a plot with UR and PR in separate subplots (UR on top, PR on bottom)."""
        if not line_entries:
            return

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_ur, ax_pr = axes[0], axes[1]
        ur_entry = line_entries.get(ContentType.UR)
        pr_entry = line_entries.get(ContentType.PR)

        # Plot UR and PR with stats
        ur_title = None
        if repo and ur_entry:
            ur_title = f"{repo.owner}.{repo.name} - {self._content_type_label(ContentType.UR)}"
        self._plot_content_type_with_stats(
            ax_ur, ur_entry, ContentType.UR, title=ur_title,
            legend_loc="lower right", stats_location="upper left",
            stats_offsets=(-0.01, -0.015) if repo else (-0.01, -0.02)
        )

        pr_title = None
        if repo and pr_entry:
            pr_title = f"{repo.owner}.{repo.name} - {self._content_type_label(ContentType.PR)}"
        self._plot_content_type_with_stats(
            ax_pr, pr_entry, ContentType.PR, title=pr_title,
            legend_loc="upper right", stats_location="upper left",
            stats_offsets=(-0.01, -0.015) if repo else (-0.05, -0.02)
        )

        # Calculate and set unified Y-axis range
        y_min_unified, y_max_unified = self._calculate_unified_y_range([ur_entry, pr_entry])
        
        # Configure axes
        sample_entry = ur_entry or pr_entry
        xlabel_text = None
        for ax in [ax_ur, ax_pr]:
            ax.set_ylabel("", fontsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=FONT_SIZE)
            ax.tick_params(axis="y", labelsize=FONT_SIZE)
            if y_min_unified is not None and y_max_unified is not None:
                ax.set_ylim(y_min_unified, y_max_unified)
            if sample_entry:
                xlabel_text = self._configure_axis_for_entry(ax, sample_entry, line_entries, is_subplot=True)

        if xlabel_text:
            fig.supxlabel(xlabel_text, fontsize=FONT_SIZE, y=0.02)
        fig.supylabel("Linked Time (days)", fontsize=FONT_SIZE, x=0.02)
        plt.tight_layout()
        self._save_plot(output_file, fig)

    def create_repo_ur_pr_plot(
        self, repo: Repository, line_entries: Dict[ContentType, Dict[str, Any]]
    ):
        """Create a plot with UR and PR in separate subplots for a repo."""
        output_file = self.png_path / f"{repo.id}_{repo.owner}.{repo.name}_ur_pr_linked_time.png"
        self._create_ur_pr_plot(line_entries, output_file, repo)

    def create_all_repos_ur_pr_plot(self, line_entries: Dict[ContentType, Dict[str, Any]]):
        """Create a plot with UR and PR in separate subplots for all repositories."""
        output_file = self.png_path / "all_repos_ur_pr_linked_time.png"
        self._create_ur_pr_plot(line_entries, output_file)

    def save_all_repos_csv(self, content_type: ContentType, metrics: pd.DataFrame):
        if metrics.empty:
            return

        output_file = (
            self.csv_path / f"all_repos_{content_type.value}_linked_time.csv"
        )
        metrics.to_csv(output_file, index=False)
        print(f"Saved all-repos linked time CSV: {output_file}")

    def generate_all_repo_plot(self, repos: List[Repository], content_type: ContentType):
        metrics = self.calculate_all_repos_linked_time(repos, content_type)
        if metrics.empty:
            return pd.DataFrame()

        # Don't create individual UR/PR plots for all repos, only save CSV
        self.save_all_repos_csv(content_type, metrics)
        return metrics

    def generate_repo_plots(self, repo: Repository, content_type: ContentType):
        metrics = self.calculate_monthly_metrics(repo, content_type)
        if metrics.empty:
            return pd.DataFrame()

        self.save_metrics_csv(repo, content_type, metrics)
        return self._compute_repo_elapsed_metrics(
            repo, content_type, REFERENCE_YEARS_DEFAULT
        )


def main():
    plotter = LinkedTimePlotter()

    repos = target_repos()
    per_repo_metrics: Dict[ContentType, List[Tuple[Repository, pd.DataFrame]]] = {
        ContentType.UR: [],
        ContentType.PR: [],
    }
    for repo in repos:
        combined_entries: Dict[ContentType, Dict[str, Any]] = {}
        for content_type in (ContentType.UR, ContentType.PR):
            metrics_df = plotter.generate_repo_plots(repo, content_type)
            entry = plotter._build_line_entry(
                metrics_df,
                content_type,
            )
            if entry:
                combined_entries[content_type] = entry
            if metrics_df is not None and not metrics_df.empty:
                per_repo_metrics[content_type].append((repo, metrics_df))
        if combined_entries:
            # Create UR/PR plot (two subplots)
            plotter.create_repo_ur_pr_plot(repo, combined_entries)

    all_repo_entries: Dict[ContentType, Dict[str, Any]] = {}
    for content_type in (ContentType.UR, ContentType.PR):
        metrics = plotter.generate_all_repo_plot(repos, content_type)
        # Ensure CSV is saved
        if not metrics.empty:
            plotter.save_all_repos_csv(content_type, metrics)
        entry = plotter._build_line_entry(metrics, content_type)
        if entry:
            all_repo_entries[content_type] = entry
    if all_repo_entries:
        # Create UR/PR plot (two subplots)
        plotter.create_all_repos_ur_pr_plot(all_repo_entries)


if __name__ == "__main__":
    main()
