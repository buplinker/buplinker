from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from root_util import ContentType, CategoryType, get_first_release_date_from_repository  # noqa: E402
from data_fetch.database.tables import Repository
from data_fetch.database.get import repositories
from analysis.timeline.statistics_analyzer import StatisticsAnalyzer

REFERENCE_YEARS_DEFAULT = 3.0
MAX_ELAPSED_MONTHS = int(round(REFERENCE_YEARS_DEFAULT * 12))

REFERENCE_TICK_STEP_DEFAULT = 0.5
REFERENCE_TICK_STEP_INT_DEFAULT = 1.0

FONT_SIZE = 30
LEGEND_FONT_SIZE = max(8, FONT_SIZE - 3)

LINE_STYLE_MAIN = (2.5, 1.0)
LINE_STYLE_INDIVIDUAL = (0.5, 0.3)
LINE_STYLE_THEIL_SEN = (2.0, 0.8)

CONTENT_TYPE_LINE_COLORS: Dict[ContentType, str] = {
    ContentType.UR: "#1f77b4",
    ContentType.PR: "#ff7f0e",
}

CATEGORY_TYPE_LINE_COLORS: Dict[CategoryType, str] = {
    CategoryType.Hedonic: "#e74c3c",
    CategoryType.Utilitarian: "#3498db",
}

CONTENT_TYPE_ORDER = {
    ContentType.UR: 0,
    ContentType.PR: 1,
}

CONTENT_TYPE_LABELS: Dict[ContentType, str] = {
    ContentType.UR: "URs",
    ContentType.PR: "PRs",
}

CATEGORY_TYPE_LABELS: Dict[CategoryType, str] = {
    CategoryType.Hedonic: "Hedonic",
    CategoryType.Utilitarian: "Utilitarian",
}

ELAPSED_YEARS_LABEL = "Elapsed Years Since First Release"

LOCATIONS = {
    "upper left": (0.02, 0.94, "left", "top"),
    "upper right": (0.96, 0.96, "right", "top"),
    "lower left": (0.02, 0.06, "left", "bottom"),
    "lower right": (0.96, 0.04, "right", "bottom"),
}

LEGEND_LOCATIONS: Dict[str, str] = {"stats": "upper left", "labels": "upper right"}


# Helper functions
def get_normalized_time_ticks(max_years: Optional[float] = None) -> Tuple[List[float], List[str]]:
    """Get normalized time ticks for plotting.
    
    Args:
        max_years: Maximum years for the time axis. If None, uses REFERENCE_YEARS_DEFAULT.
    """
    if max_years is None:
        max_years = REFERENCE_YEARS_DEFAULT
    marks = np.arange(0.0, max_years + 1e-9, REFERENCE_TICK_STEP_DEFAULT)
    use_int_format = max_years > REFERENCE_YEARS_DEFAULT
    effective_label_step = REFERENCE_TICK_STEP_INT_DEFAULT if use_int_format else REFERENCE_TICK_STEP_DEFAULT
    def format_label(m: float) -> str:
        if abs(m % effective_label_step) < 1e-9 or abs(m % effective_label_step - effective_label_step) < 1e-9:
            return f"{int(m)}y" if use_int_format else f"{m:.1f}y"
        return ""
    return marks.round(10).tolist(), [format_label(m) for m in marks]

class BaseLinkedPlotter(ABC):
    """Base class for linked time and linked ratio plotters."""

    def __init__(self, metric_name: str, limited: bool):
        """
        Initialize the plotter.
        
        Args:
            metric_name: Name of the metric (e.g., "linked_time" or "linked_ratio")
        """
        base_path = Path(__file__).parent
        self.data_path = base_path / "time_processed_data"/ ("limited_years" if limited else "all_years") / "data"
        self.metric_name = metric_name
        self.limited = limited

        output_root = base_path / "results" / metric_name / ("limited_years" if limited else "all_years")
        self.csv_path = output_root / "csv"
        self.png_path = output_root / "png"

        self.csv_path.mkdir(parents=True, exist_ok=True)
        self.png_path.mkdir(parents=True, exist_ok=True)

        self.statistics_analyzer = StatisticsAnalyzer(metric_name, limited)

    @abstractmethod
    def _load_repo_data(self, repo: Repository, content_type: ContentType) -> pd.DataFrame:
        """Load and preprocess repository data."""
        pass

    @abstractmethod
    def _group_by_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group data by month and compute metrics."""
        pass

    @abstractmethod
    def _get_csv_column_order(self) -> List[str]:
        """Return the preferred column order for CSV output."""
        pass

    @abstractmethod
    def _get_y_axis_label(self) -> str:
        """Return the y-axis label for plots."""
        pass

    @abstractmethod
    def _get_y_axis_limits(
        self, entries: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return the y-axis limits (min, max) for plots."""
        pass

    @abstractmethod
    def _format_stats_lines(
        self, stats: Dict[str, float], slope: Optional[float]
    ) -> List[str]:
        """Format statistics lines for display in plots."""
        pass

    def _get_repositories_by_category(self) -> Dict[CategoryType, List[Repository]]:
        """カテゴリごとにリポジトリを分類する"""
        all_repos = repositories()
        category_repos = {CategoryType.Hedonic: [], CategoryType.Utilitarian: []}
        
        for repo in all_repos:
            if repo.category == CategoryType.Hedonic.value:
                category_repos[CategoryType.Hedonic].append(repo)
            elif repo.category == CategoryType.Utilitarian.value:
                category_repos[CategoryType.Utilitarian].append(repo)

        return category_repos
    
    def save_metrics_csv(
        self, repo: Repository, content_type: ContentType, metrics: pd.DataFrame
    ) -> None:
        """Save metrics to CSV file."""
        if metrics.empty:
            return

        output_file = (
            self.csv_path 
            / f"{repo.id}_{repo.owner}.{repo.name}_{repo.category}_{content_type.value}_{self.metric_name}.csv"
        )
        metrics_to_save = metrics.copy()
        preferred_order = self._get_csv_column_order()
        ordered_columns = [col for col in preferred_order if col in metrics_to_save.columns]
        remaining_columns = [
            col for col in metrics_to_save.columns if col not in ordered_columns
        ]
        metrics_to_save = metrics_to_save[ordered_columns + remaining_columns]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_to_save.to_csv(output_file, index=False)

    def save_all_repos_csv(
        self, content_type: ContentType, metrics_df: pd.DataFrame
    ) -> None:
        """Save aggregated metrics across all repositories to CSV."""
        if metrics_df.empty:
            return

        output_file = (
            self.csv_path / f"all_repos_{content_type.value}_{self.metric_name}.csv"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_file, index=False)
        print(f"Saved all-repos {self.metric_name} CSV: {output_file}")
    
    def save_category_csv(
        self, category: CategoryType, content_type: ContentType, metrics_df: pd.DataFrame
    ) -> None:
        """Save aggregated metrics across all repositories to CSV."""
        if metrics_df.empty:
            return

        output_file = (
            self.csv_path / f"all_repos_{category.value}_{content_type.value}_{self.metric_name}.csv"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_file, index=False)
        print(f"Saved category {category.value} {self.metric_name} CSV: {output_file}")

    def calculate_monthly_metrics(
        self, repo: Repository, content_type: ContentType
    ) -> pd.DataFrame:
        """Calculate monthly metrics for a repository."""
        df = self._load_repo_data(repo, content_type)
        monthly_df = self._group_by_month(df)

        if monthly_df.empty:
            return pd.DataFrame()

        start_date, _ = get_first_release_date_from_repository(repo.id)
        start_period = pd.Timestamp(start_date).to_period("M")

        # Filter out data before the first release date
        start_timestamp = start_period.to_timestamp()
        monthly_df = monthly_df[monthly_df["month"] >= start_timestamp].copy()

        if monthly_df.empty:
            return pd.DataFrame()

        start_year = start_period.year
        start_month = start_period.month

        elapsed_months = (
            (monthly_df["month"].dt.year - start_year) * 12
            + (monthly_df["month"].dt.month - start_month)
        ).astype(int)

        # Limit to MAX_ELAPSED_MONTHS months
        if self.limited:
            valid_mask = (0 <= elapsed_months) & (elapsed_months < MAX_ELAPSED_MONTHS)
        else:
            valid_mask = (0 <= elapsed_months)

        metrics_df = pd.DataFrame(
            {
                "elapsed_month": elapsed_months[valid_mask],
                "elapsed_years": elapsed_months[valid_mask].astype(float) / 12.0,
                "metric": monthly_df["metric"].values[valid_mask],
                "count": monthly_df["count"].values[valid_mask],
            }
        )

        # Add additional columns if they exist in monthly_df
        for col in ["median", "std"]:
            if col in monthly_df.columns:
                metrics_df[col] = monthly_df[col].values[valid_mask]

        return metrics_df.reset_index(drop=True)

    def calculate_all_repos_linked_metrics(
        self, repos: List[Repository], content_type: ContentType
    ) -> pd.DataFrame:
        """Calculate aggregated metrics across all repositories."""
        if self.limited:
            array_size = MAX_ELAPSED_MONTHS
        else:
            # Find maximum elapsed_month across all repos
            max_month = -1
            for repo in repos:
                metrics_df = self.calculate_monthly_metrics(repo, content_type)
                if metrics_df.empty:
                    continue
                months = metrics_df["elapsed_month"].to_numpy(dtype=int)
                if len(months) > 0:
                    max_month = max(max_month, int(np.max(months)))
            array_size = max_month + 1 if max_month >= 0 else MAX_ELAPSED_MONTHS

        sum_values = np.zeros(array_size, dtype=float)
        repo_counts = np.zeros(array_size, dtype=int)
        total_counts = np.zeros(array_size, dtype=int)

        for repo in repos:
            metrics_df = self.calculate_monthly_metrics(repo, content_type)
            if metrics_df.empty:
                continue
            months = metrics_df["elapsed_month"].to_numpy(dtype=int)
            metrics = metrics_df["metric"].to_numpy(dtype=float)
            counts = metrics_df["count"].to_numpy(dtype=int)

            valid_mask = (0 <= months) & (months < array_size)

            months_valid = months[valid_mask]
            metrics_valid = metrics[valid_mask]
            counts_valid = counts[valid_mask]

            sum_values[months_valid] += metrics_valid
            repo_counts[months_valid] += 1
            total_counts[months_valid] += counts_valid

        valid_bins = repo_counts > 0
        if not np.any(valid_bins):
            return pd.DataFrame()

        elapsed_months = np.arange(array_size, dtype=int)[valid_bins]
        averaged_metrics = sum_values[valid_bins] / repo_counts[valid_bins]
        elapsed_years = elapsed_months.astype(float) / 12.0

        aggregated = pd.DataFrame(
            {
                "elapsed_month": elapsed_months,
                "elapsed_years": elapsed_years,
                "metric": averaged_metrics,
                "repo_count": repo_counts[valid_bins],
                "count": total_counts[valid_bins],
            }
        )
        return aggregated.reset_index(drop=True)

    def _build_line_entry(
        self, metrics_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Build line entry dictionary for plotting."""
        if "metric" not in metrics_df.columns or "elapsed_years" not in metrics_df.columns:
            return None

        x_values = metrics_df["elapsed_years"].to_numpy(dtype=float)
        y_values = metrics_df["metric"].to_numpy(dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            return None

        # Calculate max_years based on limited flag
        if self.limited:
            max_years = REFERENCE_YEARS_DEFAULT
        else:
            max_years = float(np.max(x_values)) if len(x_values) > 0 else REFERENCE_YEARS_DEFAULT
        
        ticks, labels = get_normalized_time_ticks(max_years)
        return {
            "x_values": x_values,
            "metric_values": y_values,
            "xticks": ticks,
            "xticklabels": labels,
            "xlabel": ELAPSED_YEARS_LABEL,
            "axis_type": "years",
        }

    def create_repo_ur_pr_plot(
        self, repo: Repository, line_entries: Dict[ContentType, Dict[str, Any]]
    ) -> None:
        """Create UR/PR comparison plot for a single repository."""
        output_file = (
            self.png_path
            / f"{repo.id}_{repo.owner}.{repo.name}_{repo.category}_ur_pr_{self.metric_name}.png"
        )
        self._create_ur_pr_plot(line_entries, output_file)

    def create_all_repos_ur_pr_plot(
        self, line_entries: Dict[ContentType, Dict[str, Any]]
    ) -> None:
        """Create UR/PR comparison plot for all repositories."""
        output_file = self.png_path / f"all_repos_ur_pr_{self.metric_name}.png"
        self._create_ur_pr_plot(line_entries, output_file)
    
    def create_all_repos_with_individual_plot(
        self, 
        repo_entries: Dict[Repository, Dict[ContentType, Dict[str, Any]]],
        combined_metric_df: Dict[ContentType, pd.DataFrame]
    ) -> None:
        """Create UR/PR comparison plot with individual repositories (light) and average (bold)."""
        output_file = self.png_path / f"all_repos_with_individual_{self.metric_name}.png"
        self._create_ur_pr_plot_with_individual(repo_entries, combined_metric_df, output_file)

    def _create_ur_pr_plot(
        self, line_entries: Dict[ContentType, Dict[str, Any]], output_file: Path
    ) -> None:
        """Create UR/PR comparison plot."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_ur, ax_pr = axes[0], axes[1]
        ur_entry = line_entries.get(ContentType.UR)
        pr_entry = line_entries.get(ContentType.PR)

        if not ur_entry or not pr_entry:
            return

        self._plot_content_type_with_stats(ax_ur, ur_entry, ContentType.UR)
        self._plot_content_type_with_stats(ax_pr, pr_entry, ContentType.PR)

        self._configure_dual_plot_axes(fig, [ax_ur, ax_pr], [ur_entry, pr_entry])
        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _create_ur_pr_plot_with_individual(
        self, 
        repo_entries: Dict[int, Dict[ContentType, Dict[str, Any]]], 
        combined_metric_df: Dict[ContentType, pd.DataFrame],
        output_file: Path
    ) -> None:
        """Create UR/PR comparison plot with individual repositories (light) and average (bold)."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_ur, ax_pr = axes[0], axes[1]
        ur_entries, pr_entries = {
            "x_values": [],
            "metric_values": [],
        }, {
            "x_values": [],
            "metric_values": [],
        }

        for _, entry in repo_entries.items():
            ur_entry = entry.get(ContentType.UR)
            pr_entry = entry.get(ContentType.PR)

            if not ur_entry or not pr_entry:
                continue

            ur_entries["x_values"].extend(ur_entry["x_values"])
            ur_entries["metric_values"].extend(ur_entry["metric_values"])
            pr_entries["x_values"].extend(pr_entry["x_values"])
            pr_entries["metric_values"].extend(pr_entry["metric_values"])

            self._plot_content_type_with_stats(ax_ur, ur_entry, ContentType.UR, linewidth=LINE_STYLE_INDIVIDUAL[0], alpha=LINE_STYLE_INDIVIDUAL[1], only_line=True)
            self._plot_content_type_with_stats(ax_pr, pr_entry, ContentType.PR, linewidth=LINE_STYLE_INDIVIDUAL[0], alpha=LINE_STYLE_INDIVIDUAL[1], only_line=True)


        ur_elapsed_month_groups = self.statistics_analyzer.prepare_compared_value_groups(combined_metric_df[ContentType.UR], "metric", compared_col="elapsed_years", min_group_size=1)
        pr_elapsed_month_groups = self.statistics_analyzer.prepare_compared_value_groups(combined_metric_df[ContentType.PR], "metric", compared_col="elapsed_years", min_group_size=1)
        self._plot_content_type_with_stats(ax_ur, ur_elapsed_month_groups, ContentType.UR, view="per-repo")
        self._plot_content_type_with_stats(ax_pr, pr_elapsed_month_groups, ContentType.PR, view="per-repo")

        self._configure_dual_plot_axes(fig, [ax_ur, ax_pr], [ur_entries, pr_entries])
        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def create_category_comparison_plot(
        self,
        category_entries_by_content: Dict[ContentType, Dict[CategoryType, Dict[str, Any]]],
    ) -> None:
        """Create category comparison plot with UR (top) and PR (bottom), each showing Hedonic and Utilitarian."""
        output_file = self.png_path / f"all_repos_category_ur_pr_{self.metric_name}.png"
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_ur, ax_pr = axes[0], axes[1]
        ur_category_entries = category_entries_by_content.get(ContentType.UR)
        pr_category_entries = category_entries_by_content.get(ContentType.PR)
        
        if not ur_category_entries or not pr_category_entries:
            print("No category entries found")
            return
        
        self._plot_categories_with_stats(ax_ur, ur_category_entries, ContentType.UR)
        self._plot_categories_with_stats(ax_pr, pr_category_entries, ContentType.PR)
        
        self._configure_categories_dual_plot_axes(fig, [ax_ur, ax_pr], [ur_category_entries, pr_category_entries])
        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved category comparison plot: {output_file}")
    
    def _configure_dual_plot_axes(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        all_entries: List[Dict[str, Any]],
    ) -> None:
        """Configure axes for dual plot (common logic for UR/PR and category plots)."""
        # Calculate max_years
        if self.limited:
            max_years = REFERENCE_YEARS_DEFAULT
        else:
            max_years = 0.0
            for entry in all_entries:
                if isinstance(entry, dict) and "x_values" in entry:
                    x_vals = entry["x_values"]
                    # Handle both list and numpy array
                    if x_vals is not None:
                        x_vals_array = np.asarray(x_vals)
                        if x_vals_array.size > 0:
                            max_years = max(max_years, float(np.max(x_vals_array)))
            if max_years == 0.0:
                max_years = REFERENCE_YEARS_DEFAULT
        
        self._plot_axes_with_stats(fig, axes, all_entries, max_years)
            
    def _plot_axes_with_stats(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        all_entries: List[Dict[str, Any]],
        max_years: float,
    ) -> None:
        """Plot axes with statistics."""
        y_min_unified, y_max_unified = self._get_y_axis_limits(all_entries)

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=FONT_SIZE)
            ax.tick_params(axis="y", labelsize=FONT_SIZE)
            if y_min_unified is not None and y_max_unified is not None:
                ax.set_ylim(y_min_unified, y_max_unified)
            ticks, labels = get_normalized_time_ticks(max_years)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            if self.limited:
                margin = max(0.0, REFERENCE_YEARS_DEFAULT * 0.02)
                ax.set_xlim(-margin, REFERENCE_YEARS_DEFAULT + margin)
            else:
                margin = max(0.0, max_years * 0.02)
                ax.set_xlim(-margin, max_years + margin)
        
        fig.supxlabel(ELAPSED_YEARS_LABEL, fontsize=FONT_SIZE, y=0.02)
        fig.supylabel(self._get_y_axis_label(), fontsize=FONT_SIZE, x=0.02)

    
    def _configure_categories_dual_plot_axes(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        all_entries: List[Any],
    ) -> None:
        """Configure axes for dual plot (common logic for UR/PR and category plots).
        
        Args:
            all_entries: List of entries. Each entry can be either:
                - Dict[str, Any]: Single entry (for UR/PR plots)
                - Dict[CategoryType, Dict[str, Any]]: Category entries dict (for category plots)
        """
        # Flatten entries: if entry is a dict of dicts (category entries), extract values
        flattened_entries = []
        for entry in all_entries:
            if isinstance(entry, dict):
                # Check if it's a category entries dict (Dict[CategoryType, Dict[str, Any]])
                # by checking if all values are dicts with "x_values" key
                values = list(entry.values())
                if values and all(isinstance(v, dict) and "x_values" in v for v in values):
                    flattened_entries.extend(values)
                else:
                    # Single entry dict
                    flattened_entries.append(entry)
            else:
                flattened_entries.append(entry)
        # Calculate max_years from all flattened entries
        if self.limited:
            max_years = REFERENCE_YEARS_DEFAULT
        else:
            max_years = 0.0
            for entry in flattened_entries:
                if isinstance(entry, dict) and "x_values" in entry:
                    x_vals = entry["x_values"]
                    # Handle both list and numpy array
                    if x_vals is not None:
                        x_vals_array = np.asarray(x_vals)
                        if x_vals_array.size > 0:
                            max_years = max(max_years, float(np.max(x_vals_array)))
            if max_years == 0.0:
                max_years = REFERENCE_YEARS_DEFAULT
        
        self._plot_axes_with_stats(fig, axes, flattened_entries, max_years)

    def _plot_content_type_with_stats(
        self,
        ax: plt.Axes,
        entry: Dict[str, Any],
        content_type: ContentType,
        linewidth: float = LINE_STYLE_MAIN[0],
        alpha: float = LINE_STYLE_MAIN[1],
        only_line: bool = False,
        view: str = "all-repos",
    ) -> None:
        """Plot content type with statistics."""
        if not entry:
            return

        relative_data_map = {content_type: entry}
        items = sorted(relative_data_map.items(), key=lambda kv: CONTENT_TYPE_ORDER.get(kv[0], float("inf")))
    
        for content_type, data in items:
            if view == "per-repo":
                plot_x_vals = [values for values in entry.keys()]
                plot_y_vals = [values.mean() for values in entry.values()]
                x_vals, y_vals = self.statistics_analyzer._format_grouped_values(entry)
            else:
                x_vals = np.asarray(data.get("x_values", []))
                y_vals = np.asarray(data.get("metric_values", []))
                plot_x_vals, plot_y_vals = x_vals, y_vals

            ax.plot(
                plot_x_vals, plot_y_vals,
                label=CONTENT_TYPE_LABELS.get(content_type) if not only_line else None,
                color=CONTENT_TYPE_LINE_COLORS.get(content_type),
                linestyle="-",
                linewidth=linewidth,
                alpha=alpha,
            )

        if only_line:
            return
        
        array = np.asarray(y_vals, dtype=float)
        finite_values = array[np.isfinite(array)]
        stats = {
            "mean": float(np.mean(finite_values)),
            "median": float(np.median(finite_values)),
        }

        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_arr, y_arr = x_vals[mask], y_vals[mask]

        slope, intercept, _, _ = scipy_stats.theilslopes(y_arr, x_arr, 0.95)
        stats_lines = self._format_stats_lines(stats, slope)
        location = LEGEND_LOCATIONS["stats"]

        normalized_location = location.lower().strip()
        x, y, ha, va = LOCATIONS.get(normalized_location, LOCATIONS["upper left"])

        bbox_style = dict(boxstyle="round", facecolor="white", alpha=0.8)

        ax.text(x, y, "\n".join(stats_lines), transform=ax.transAxes, fontsize=LEGEND_FONT_SIZE,
                ha=ha, va=va, bbox=bbox_style)

        x_line = np.array([x_arr.min(), x_arr.max()], dtype=float)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color=CONTENT_TYPE_LINE_COLORS.get(content_type), linestyle="--", linewidth=LINE_STYLE_THEIL_SEN[0], alpha=LINE_STYLE_THEIL_SEN[1], label="Theil-Sen")

        if view == "per-repo":
            ax.plot(x_line, y_line, color=CONTENT_TYPE_LINE_COLORS.get(content_type), linestyle="-", linewidth=LINE_STYLE_INDIVIDUAL[0], alpha=LINE_STYLE_INDIVIDUAL[1], label="App")
        
        legend_loc = LEGEND_LOCATIONS["labels"]
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=legend_loc)
    
    def _plot_categories_with_stats(
        self,
        ax: plt.Axes,
        category_entries: Dict[CategoryType, Dict[str, Any]],
        content_type: ContentType,
    ) -> None:
        """Plot multiple categories (Hedonic and Utilitarian) on the same axes with statistics."""
        # Plot lines for both categories
        for category, entry in category_entries.items():
            if not entry:
                continue
            
            x_vals = np.asarray(entry.get("x_values", []))
            y_vals = np.asarray(entry.get("metric_values", []))

            ax.plot(
                x_vals, y_vals,
                label= f"{CONTENT_TYPE_LABELS.get(content_type)} {CATEGORY_TYPE_LABELS.get(category)}",
                color=CATEGORY_TYPE_LINE_COLORS.get(category),
                linestyle="-",
                linewidth=LINE_STYLE_MAIN[0],
            )

        # Add statistics for each category
        for category, entry in category_entries.items():
            if not entry:
                continue
                
            x_values = np.asarray(entry.get("x_values", []))
            y_values = np.asarray(entry.get("metric_values", []))
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            x_arr, y_arr = x_values[mask], y_values[mask]

            slope, intercept, _, _ = scipy_stats.theilslopes(y_arr, x_arr, 0.95)
            
            x_line = np.array([x_arr.min(), x_arr.max()], dtype=float)
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, color=CATEGORY_TYPE_LINE_COLORS.get(category), 
                    linestyle="--", linewidth=LINE_STYLE_THEIL_SEN[0], alpha=LINE_STYLE_THEIL_SEN[1], label=f"{CATEGORY_TYPE_LABELS.get(category)} Theil-Sen")
        
        legend_loc = LEGEND_LOCATIONS["labels"]
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=legend_loc)

