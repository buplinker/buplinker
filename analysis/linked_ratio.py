#!/usr/bin/env python3
"""
Create stacked area plot from the processed timeline data.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from root_util import ContentType, get_first_release_date_from_repository
from util import (  # noqa: E402
    Repository,
    target_repos,
    REFERENCE_YEARS_DEFAULT,
    normalized_time_ticks,
    CONTENT_TYPE_LINE_COLORS,
    CONTENT_TYPE_LINE_STYLES,
    plot_content_type_lines,
    elapsed_years_label,
    FONT_SIZE,
    LEGEND_FONT_SIZE,
    apply_date_axis,
    apply_elapsed_years_axis,
    compute_elapsed_month_ticks,
    compute_series_stats,
    format_stats_lines,
    render_stats_box,
    compute_trend_parameters,
    plot_trend_line,
    save_figure,
    write_counts_csv,
)

CONTENT_TYPE_ORDER = {
    ContentType.UR: 0,
    ContentType.PR: 1,
}

MAX_ELAPSED_MONTHS = int(round(REFERENCE_YEARS_DEFAULT * 12))


class LinkedRatioPlotter:
    """Create linked ratio plots grouped by repository."""

    def __init__(self):
        base_path = Path(__file__).parent
        self.data_path = base_path / "time_processed_data"

        output_root = base_path / "results" / "linked_ratio"
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

    # ------------------------------------------------------------------ #
    # Data loading / preparation
    # ------------------------------------------------------------------ #
    def _load_repo_monthly_data(
        self,
        repo: Repository,
        content_type: ContentType,
    ) -> Optional[Dict[str, Any]]:
        """Load per-repository monthly counts aligned to the release window."""
        data_path = (
            self.data_path
            / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}.csv"
        )
        if not data_path.exists():
            return None

        df = pd.read_csv(data_path)
        if df.empty:
            return None

        df["created_at"] = pd.to_datetime(df["created_at"])
        df["year_month"] = df["created_at"].dt.to_period("M")
        monthly_not_linked = (
            df[df["label"] == 0].groupby("year_month").size() if not df.empty else pd.Series(dtype=float)
        )
        monthly_linked = (
            df[df["label"] == 1].groupby("year_month").size() if not df.empty else pd.Series(dtype=float)
        )

        release_period = get_first_release_date_from_repository(repo.id)
        start_date: Optional[pd.Timestamp] = None
        end_date: Optional[pd.Timestamp] = None
        if release_period:
            start_date, end_date = release_period
            all_months: pd.PeriodIndex = pd.period_range(start=start_date, end=end_date, freq="M")
        else:
            month_candidates = sorted(set(monthly_not_linked.index) | set(monthly_linked.index))
            if month_candidates:
                all_months = pd.PeriodIndex(month_candidates, freq="M")
            elif not df.empty:
                min_month = df["year_month"].min()
                max_month = df["year_month"].max()
                all_months = pd.period_range(start=min_month, end=max_month, freq="M")
            else:
                all_months = pd.PeriodIndex([], freq="M")
            if len(all_months) > 0:
                start_date = all_months[0].to_timestamp(how="start")
                end_date = all_months[-1].to_timestamp(how="end")

        if len(all_months) == 0:
            return None

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        monthly_not_linked = monthly_not_linked.reindex(all_months, fill_value=0.0).astype(float)
        monthly_linked = monthly_linked.reindex(all_months, fill_value=0.0).astype(float)
        monthly_total = (monthly_not_linked + monthly_linked).astype(float)
        x_axis_dates = all_months.to_timestamp(how="end")
        elapsed_months = np.arange(len(all_months), dtype=int)

        return {
            "monthly_not_linked": monthly_not_linked,
            "monthly_linked": monthly_linked,
            "monthly_total": monthly_total,
            "all_months": all_months,
            "x_axis_dates": x_axis_dates,
            "elapsed_months": elapsed_months,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _aggregate_entries(
        self,
        entries: List[Dict[str, Any]],
        reference_years: float = REFERENCE_YEARS_DEFAULT,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate per-repository linked ratio entries into averaged series."""
        if not entries:
            return None

        max_elapsed_months = int(round(reference_years * 12))
        if max_elapsed_months <= 0:
            return None

        ratio_sums = np.zeros(max_elapsed_months, dtype=float)
        repo_counts = np.zeros(max_elapsed_months, dtype=int)
        title_suffix = None

        for entry in entries:
            months = entry.get("elapsed_months")
            ratios = entry.get("linked_ratio")
            if months is None or ratios is None:
                continue
            months_arr = np.asarray(months, dtype=int)
            ratios_arr = np.asarray(ratios, dtype=float)
            if months_arr.size == 0 or ratios_arr.size == 0:
                continue
            valid_mask = (
                (months_arr >= 0)
                & (months_arr < max_elapsed_months)
                & np.isfinite(ratios_arr)
            )
            if not np.any(valid_mask):
                continue
            months_valid = months_arr[valid_mask]
            ratios_valid = ratios_arr[valid_mask]
            ratio_sums[months_valid] += ratios_valid
            repo_counts[months_valid] += 1
            if title_suffix is None:
                title_suffix = entry.get("title_suffix")

        valid_bins = repo_counts > 0
        if not np.any(valid_bins):
            return None

        elapsed_months = np.arange(max_elapsed_months, dtype=int)[valid_bins]
        averaged_ratios = ratio_sums[valid_bins] / repo_counts[valid_bins]
        x_values_years = elapsed_months.astype(float) / 12.0

        xticks, xticklabels = normalized_time_ticks(
            reference_years=reference_years, step=0.5
        )
        xlabel = elapsed_years_label(reference_years)

        return {
            "x_values": x_values_years,
            "linked_ratio": averaged_ratios,
            "xticks": xticks,
            "xticklabels": xticklabels,
            "xlabel": xlabel,
            "title_suffix": title_suffix,
            "reference_years": reference_years,
        }

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def _plot_linked_ratio(
        self,
        data_map: Dict[ContentType, Dict[str, Any]],
        output_path: Path,
        title_prefix: str,
        xlabel: str,
        *,
        is_date_axis: bool,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        xticks: Optional[List[float]] = None,
        xticklabels: Optional[List[str]] = None,
        reference_years: float = REFERENCE_YEARS_DEFAULT,
        axis_type: str = "years",
    ) -> None:
        """Plot linked ratios for UR and PR in separate subplots (UR on top, PR on bottom)."""
        valid_entries = {
            ct: data
            for ct, data in data_map.items()
            if data
            and isinstance(data.get("linked_ratio"), np.ndarray)
            and data["linked_ratio"].size > 0
            and isinstance(data.get("x_values"), np.ndarray)
            and data["x_values"].size == data["linked_ratio"].size
        }
        if not valid_entries:
            return

        # Create two subplots: UR on top, PR on bottom
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        ax_ur = axes[0]  # User Reviews (top)
        ax_pr = axes[1]  # Pull Requests (bottom)

        # Process each content type separately
        ur_entry = valid_entries.get(ContentType.UR)
        pr_entry = valid_entries.get(ContentType.PR)

        axis_xticks = xticks
        axis_xticklabels = xticklabels
        max_axis_value = 0.0
        if not is_date_axis and axis_type == "months":
            max_candidates: List[float] = []
            for data in valid_entries.values():
                x_vals = data.get("x_values")
                if isinstance(x_vals, np.ndarray) and x_vals.size > 0:
                    max_candidates.append(float(np.nanmax(x_vals)))
            if max_candidates:
                max_axis_value = max(max_candidates)
            if axis_xticks is None or axis_xticklabels is None:
                computed_ticks, computed_labels = compute_elapsed_month_ticks(int(round(max_axis_value)))
                axis_xticks = computed_ticks
                axis_xticklabels = computed_labels

        # Plot User Reviews (top)
        if ur_entry:
            ur_line_entries = {ContentType.UR: ur_entry}
            plotted_ur = plot_content_type_lines(
                ax_ur,
                ur_line_entries,
                line_colors=CONTENT_TYPE_LINE_COLORS,
                line_styles=CONTENT_TYPE_LINE_STYLES,
                default_linewidth=2.5,
                sort_key=lambda ct: CONTENT_TYPE_ORDER.get(ct, float("inf")),
            )
            if plotted_ur:
                series = ur_entry.get("linked_ratio")
                stats = compute_series_stats(series)
                if stats:
                    title_suffix = ur_entry.get("title_suffix") or ContentType.UR.value.upper()
                    axis_for_entry = ur_entry.get("axis_type", axis_type)
                    slope, intercept = compute_trend_parameters(
                        ur_entry.get("x_values"),
                        series,
                        axis_type=axis_for_entry,
                    )
                    stats_lines = format_stats_lines(
                        None,
                        stats,
                        include_sum=False,
                        mean_decimals=3,
                        median_decimals=3,
                        slope_per_year=slope,
                    )
                    render_stats_box(
                        ax_ur,
                        stats_lines,
                        location="upper left",
                        x_offset=-0.01,
                        y_offset=-0.015
                    )
                    plot_trend_line(
                        ax_ur,
                        ur_entry.get("x_values"),
                        slope,
                        intercept,
                        axis_type=axis_for_entry,
                        color=CONTENT_TYPE_LINE_COLORS.get(ContentType.UR),
                    )
                ax_ur.legend(fontsize=LEGEND_FONT_SIZE)

        # Plot Pull Requests (bottom)
        if pr_entry:
            pr_line_entries = {ContentType.PR: pr_entry}
            plotted_pr = plot_content_type_lines(
                ax_pr,
                pr_line_entries,
                line_colors=CONTENT_TYPE_LINE_COLORS,
                line_styles=CONTENT_TYPE_LINE_STYLES,
                default_linewidth=2.5,
                sort_key=lambda ct: CONTENT_TYPE_ORDER.get(ct, float("inf")),
            )
            if plotted_pr:
                series = pr_entry.get("linked_ratio")
                stats = compute_series_stats(series)
                if stats:
                    title_suffix = pr_entry.get("title_suffix") or ContentType.PR.value.upper()
                    axis_for_entry = pr_entry.get("axis_type", axis_type)
                    slope, intercept = compute_trend_parameters(
                        pr_entry.get("x_values"),
                        series,
                        axis_type=axis_for_entry,
                    )
                    stats_lines = format_stats_lines(
                        None,
                        stats,
                        include_sum=False,
                        mean_decimals=3,
                        median_decimals=3,
                        slope_per_year=slope,
                    )
                    render_stats_box(
                        ax_pr,
                        stats_lines,
                        location="upper left",
                        x_offset=-0.01,
                        y_offset=-0.015
                    )
                    plot_trend_line(
                        ax_pr,
                        pr_entry.get("x_values"),
                        slope,
                        intercept,
                        axis_type=axis_for_entry,
                        color=CONTENT_TYPE_LINE_COLORS.get(ContentType.PR),
                    )
                ax_pr.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right")

        # Configure axes for both subplots
        xlabel_text = None
        for ax in [ax_ur, ax_pr]:
            # Remove individual Y-axis labels (will use common label instead)
            ax.set_ylabel("", fontsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=FONT_SIZE)
            ax.tick_params(axis="y", labelsize=FONT_SIZE)
            ax.set_ylim(0, 1.0)

            if is_date_axis:
                apply_date_axis(ax, start_date, end_date, rotation=45)
                # Remove individual X-axis labels (will use common label instead)
                ax.set_xlabel("", fontsize=FONT_SIZE)
            else:
                if axis_type == "months":
                    upper_bound = int(round(max_axis_value))
                    ax.set_xlim(0, max(0, upper_bound))
                    if axis_xticks is not None and axis_xticklabels is not None:
                        ax.set_xticks(axis_xticks)
                        ax.set_xticklabels(axis_xticklabels)
                    xlabel_text = xlabel
                    # Remove individual X-axis labels (will use common label instead)
                    ax.set_xlabel("", fontsize=FONT_SIZE)
                else:
                    apply_elapsed_years_axis(
                        ax,
                        reference_years,
                        xticks=axis_xticks,
                        xticklabels=axis_xticklabels,
                    )
                    xlabel_text = xlabel
                    # Remove individual X-axis labels (will use common label instead)
                    ax.set_xlabel("", fontsize=FONT_SIZE)
                yticks = ax.get_yticks()
                ax.set_yticks(yticks)

        # Set common X-axis label for the entire figure at the bottom
        if xlabel_text:
            fig.supxlabel(xlabel_text, fontsize=FONT_SIZE, y=0.02)

        # Set common Y-axis label for the entire figure on the left
        fig.supylabel("Linked Ratio", fontsize=FONT_SIZE, x=0.02)

        plt.tight_layout()
        save_figure(output_path, fig)
        print(f"Saved linked ratio plot: {output_path}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_repo_plots(
        self, repo: Repository, content_type: ContentType
    ) -> Optional[Dict[str, Any]]:
        """Create plots for specific data type for a single repository.

        Returns linked-ratio data for downstream combined plotting."""
        monthly_data = self._load_repo_monthly_data(repo, content_type)
        if monthly_data is None:
            print(f"No data found for {repo.owner}.{repo.name} - {content_type.value}")
            return None

        if content_type == ContentType.UR:
            title_suffix = "URs"
        else:
            title_suffix = "PRs"

        monthly_not_linked = monthly_data["monthly_not_linked"]
        monthly_linked = monthly_data["monthly_linked"]
        all_months = monthly_data["all_months"]

        max_elapsed_months = MAX_ELAPSED_MONTHS
        if max_elapsed_months <= 0:
            return None

        x_years = np.arange(max_elapsed_months, dtype=float) / 12.0
        xticks, xticklabels = normalized_time_ticks(
            reference_years=REFERENCE_YEARS_DEFAULT, step=0.5
        )
        month_indices = pd.RangeIndex(max_elapsed_months)

        repo_not_linked = np.zeros(max_elapsed_months, dtype=float)
        repo_linked = np.zeros(max_elapsed_months, dtype=float)

        if monthly_data["start_date"] is not None:
            start_period = pd.Timestamp(monthly_data["start_date"]).to_period("M")
        elif len(all_months) > 0:
            start_period = all_months[0]
        else:
            start_period = None

        end_period_cap = (
            start_period + (max_elapsed_months - 1) if start_period is not None else None
        )

        for period, not_linked_value, linked_value in zip(
            all_months, monthly_not_linked, monthly_linked
        ):
            if start_period is None:
                start_period = period
                end_period_cap = start_period + (max_elapsed_months - 1)
            month_offset = (
                (period.year - start_period.year) * 12
                + (period.month - start_period.month)
            )
            if month_offset < 0 or month_offset >= max_elapsed_months:
                continue
            if end_period_cap is not None and period > end_period_cap:
                continue
            repo_not_linked[month_offset] = float(not_linked_value)
            repo_linked[month_offset] = float(linked_value)

        repo_total = repo_not_linked + repo_linked
        active_mask = repo_total > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            repo_not_linked_ratio = np.divide(
                repo_not_linked,
                repo_total,
                out=np.zeros_like(repo_not_linked),
                where=repo_total > 0,
            )
            repo_linked_ratio = np.divide(
                repo_linked,
                repo_total,
                out=np.zeros_like(repo_linked),
                where=repo_total > 0,
            )
        repo_total_ratio = np.clip(
            repo_not_linked_ratio + repo_linked_ratio, a_min=0.0, a_max=1.0
        )

        csv_path = self.csv_path / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_linked_ratio.csv"
        write_counts_csv(
            [
                ("not_linked", pd.Series(repo_not_linked_ratio, index=month_indices)),
                ("linked", pd.Series(repo_linked_ratio, index=month_indices)),
                ("total", pd.Series(repo_total_ratio, index=month_indices)),
            ],
            index_name="elapsed_month",
            output_csv=csv_path,
            index_formatter=lambda value: int(value) if pd.notna(value) else None,
            enforce_integer=False,
            value_round=6,
        )

        valid_mask = active_mask
        if not np.any(valid_mask):
            return None

        elapsed_months_arr = np.nonzero(valid_mask)[0].astype(int)
        linked_ratio_values = repo_linked_ratio[valid_mask]
        ratio_xticks, ratio_xticklabels = normalized_time_ticks(
            reference_years=REFERENCE_YEARS_DEFAULT, step=0.5
        )
        x_values_years = elapsed_months_arr.astype(float) / 12.0

        return {
            "repo": repo,
            "content_type": content_type,
            "x_values": x_values_years,
            "linked_ratio": linked_ratio_values,
            "title_suffix": title_suffix,
            "elapsed_months": elapsed_months_arr,
            "xticks": ratio_xticks,
            "xticklabels": ratio_xticklabels,
            "xlabel": elapsed_years_label(REFERENCE_YEARS_DEFAULT),
            "reference_years": REFERENCE_YEARS_DEFAULT,
            "axis_type": "years",
        }

    def generate_all_repo_plot(
        self, repos: List[Repository], content_type: ContentType
    ) -> Optional[Dict[str, Any]]:
        """Create combined plots for specific data type across all repositories.

        Returns linked-ratio data for downstream combined plotting."""
        max_elapsed_months = MAX_ELAPSED_MONTHS
        if max_elapsed_months <= 0:
            return None

        if content_type == ContentType.UR:
            title_suffix = "URs"
        else:
            title_suffix = "PRs"

        xlabel = elapsed_years_label(REFERENCE_YEARS_DEFAULT)
        xticks, xticklabels = normalized_time_ticks(
            reference_years=REFERENCE_YEARS_DEFAULT, step=0.5
        )
        x_years = np.arange(max_elapsed_months, dtype=float) / 12.0
        month_indices = pd.RangeIndex(max_elapsed_months)

        aggregated_not_linked = np.zeros(max_elapsed_months, dtype=float)
        aggregated_linked = np.zeros(max_elapsed_months, dtype=float)
        not_linked_sum = np.zeros(max_elapsed_months, dtype=float)
        linked_sum = np.zeros(max_elapsed_months, dtype=float)
        repo_counts = np.zeros(max_elapsed_months, dtype=float)

        any_data = False

        for repo in repos:
            monthly_data = self._load_repo_monthly_data(repo, content_type)
            if monthly_data is None:
                continue
            any_data = True

            monthly_not_linked = monthly_data["monthly_not_linked"]
            monthly_linked = monthly_data["monthly_linked"]
            all_months = monthly_data["all_months"]

            if len(all_months) == 0:
                continue

            if monthly_data["start_date"] is not None:
                start_period = pd.Timestamp(monthly_data["start_date"]).to_period("M")
            else:
                start_period = all_months[0]

            end_period_cap = start_period + (max_elapsed_months - 1)

            repo_not_linked = np.zeros(max_elapsed_months, dtype=float)
            repo_linked = np.zeros(max_elapsed_months, dtype=float)

            for period, not_linked_value, linked_value in zip(
                all_months, monthly_not_linked, monthly_linked
            ):
                if start_period is None:
                    start_period = period
                month_offset = (
                    (period.year - start_period.year) * 12
                    + (period.month - start_period.month)
                )
                if month_offset < 0 or month_offset >= max_elapsed_months:
                    continue
                if period > end_period_cap:
                    continue
                repo_not_linked[month_offset] = float(not_linked_value)
                repo_linked[month_offset] = float(linked_value)

            repo_total = repo_not_linked + repo_linked
            valid_mask = repo_total > 0

            aggregated_not_linked += repo_not_linked
            aggregated_linked += repo_linked

            not_linked_sum[valid_mask] += (
                repo_not_linked[valid_mask] / repo_total[valid_mask]
            )
            linked_sum[valid_mask] += (
                repo_linked[valid_mask] / repo_total[valid_mask]
            )
            repo_counts[valid_mask] += 1.0

        if not any_data:
            print(f"No data found for {content_type.value}")
            return None

        monthly_total_count = aggregated_not_linked + aggregated_linked

        with np.errstate(invalid="ignore", divide="ignore"):
            monthly_not_linked = np.divide(
                not_linked_sum, repo_counts, out=np.zeros_like(not_linked_sum), where=repo_counts > 0
            )
            monthly_linked = np.divide(
                linked_sum, repo_counts, out=np.zeros_like(linked_sum), where=repo_counts > 0
            )
        monthly_total = np.clip(
            monthly_not_linked + monthly_linked, a_min=0.0, a_max=1.0
        )

        repo_count_series = pd.Series(np.round(repo_counts).astype(int), index=month_indices)
        total_count_series = pd.Series(np.round(monthly_total_count).astype(int), index=month_indices)
        extra_series = {
            "repo_count": repo_count_series,
            "count": total_count_series,
        }

        csv_path = self.csv_path / f"all_repos_{content_type.value}_linked_ratio.csv"
        write_counts_csv(
            [
                ("not_linked", pd.Series(monthly_not_linked, index=month_indices)),
                ("linked", pd.Series(monthly_linked, index=month_indices)),
                ("total", pd.Series(monthly_total, index=month_indices)),
            ],
            index_name="elapsed_month",
            output_csv=csv_path,
            index_formatter=lambda value: int(value) if pd.notna(value) else None,
            enforce_integer=False,
            value_round=6,
            extra_series=extra_series,
        )

        valid_bins = repo_counts > 0
        if not np.any(valid_bins):
            return None

        x_values_rel = x_years[valid_bins]
        linked_ratio_values = monthly_linked[valid_bins]

        return {
            "content_type": content_type,
            "x_values": x_values_rel,
            "linked_ratio": linked_ratio_values,
            "xticks": xticks,
            "xticklabels": xticklabels,
            "xlabel": xlabel,
            "title_suffix": title_suffix,
            "reference_years": REFERENCE_YEARS_DEFAULT,
        }

    def create_repo_ur_pr_plot(
        self, repo: Repository, data_map: Dict[ContentType, Dict[str, Any]]
    ) -> None:
        """Plot UR/PR linked ratios in separate subplots for a single repository."""
        sample_ticks = None
        sample_labels = None
        for data in data_map.values():
            if not data:
                continue
            ticks = data.get("xticks")
            labels = data.get("xticklabels")
            if ticks is not None and labels is not None:
                sample_ticks = ticks
                sample_labels = labels
                break

        output_path = (
            self.png_path
            / f"{repo.id}_{repo.owner}.{repo.name}_ur_pr_linked_ratio.png"
        )
        self._plot_linked_ratio(
            data_map,
            output_path,
            title_prefix=f"{repo.owner}.{repo.name}",
            xlabel=elapsed_years_label(REFERENCE_YEARS_DEFAULT),
            is_date_axis=False,
            xticks=sample_ticks,
            xticklabels=sample_labels,
        )

    def create_all_repos_ur_pr_plot(
        self, data_map: Dict[ContentType, Dict[str, Any]]
    ) -> None:
        """Plot UR/PR linked ratios in separate subplots across all repositories."""
        filtered_map = {
            ct: data for ct, data in data_map.items() if data is not None
        }
        if not filtered_map:
            return

        sample_data = next(iter(filtered_map.values()))
        xlabel = sample_data.get(
            "xlabel",
            elapsed_years_label(REFERENCE_YEARS_DEFAULT),
        )
        xticks = sample_data.get("xticks")
        xticklabels = sample_data.get("xticklabels")
        reference_years = sample_data.get("reference_years", REFERENCE_YEARS_DEFAULT)

        output_path = self.png_path / "all_repos_ur_pr_linked_ratio.png"
        self._plot_linked_ratio(
            filtered_map,
            output_path,
            title_prefix="All Repositories",
            xlabel=xlabel,
            is_date_axis=False,
            xticks=xticks,
            xticklabels=xticklabels,
            reference_years=float(reference_years),
        )


def main():
    plotter = LinkedRatioPlotter()

    repos = target_repos()
    entries_map: Dict[ContentType, List[Dict[str, Any]]] = {
        ContentType.UR: [],
        ContentType.PR: [],
    }
    for repo in repos:
        print(f"Processing {repo.id}: {repo.owner}.{repo.name}...")

        data_map: Dict[ContentType, Dict[str, Any]] = {}

        ur_data = plotter.generate_repo_plots(repo, ContentType.UR)
        if ur_data:
            data_map[ContentType.UR] = ur_data
            entries_map[ContentType.UR].append(ur_data)

        pr_data = plotter.generate_repo_plots(repo, ContentType.PR)
        if pr_data:
            data_map[ContentType.PR] = pr_data
            entries_map[ContentType.PR].append(pr_data)

        if data_map:
            plotter.create_repo_ur_pr_plot(repo, data_map)

    # Create combined plots for all repositories
    print("Creating combined plots for all repositories...")
    plotter.generate_all_repo_plot(repos, ContentType.UR)
    plotter.generate_all_repo_plot(repos, ContentType.PR)

    aggregated_map: Dict[ContentType, Optional[Dict[str, Any]]] = {
        ct: plotter._aggregate_entries(entries)
        for ct, entries in entries_map.items()
    }
    filtered_aggregated_map = {
        ct: data for ct, data in aggregated_map.items() if data is not None
    }
    plotter.create_all_repos_ur_pr_plot(filtered_aggregated_map)

    print("Finished processing all repositories...")


if __name__ == "__main__":
    main()
