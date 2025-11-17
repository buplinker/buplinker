#!/usr/bin/env python3
"""
Common utilities for timeline visualisations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from scipy import stats as scipy_stats  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None

from root_util import ContentType

REFERENCE_YEARS_DEFAULT = 3.0
REFERENCE_TICK_STEP_DEFAULT = 0.5

FONT_SIZE = 30
LEGEND_FONT_SIZE = max(8, FONT_SIZE - 3)
STATS_FONT_SIZE = max(7, FONT_SIZE - 3)

CONTENT_TYPE_LINE_COLORS: Dict[ContentType, str] = {
    ContentType.UR: "#1f77b4",
    ContentType.PR: "#ff7f0e",
}

CONTENT_TYPE_LINE_STYLES: Dict[ContentType, str] = {
    ContentType.UR: "-",
    ContentType.PR: "-",
}

ELAPSED_MONTHS_LABEL = "Elapsed Months Since First Release"

@dataclass
class Repository:
    """Simple repository data class to replace the database Repository."""
    id: int
    owner: str
    name: str


def load_repositories_from_csv(csv_path: Optional[Path] = None) -> List[Repository]:
    """Load repositories from repositories.csv file."""
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "repositories.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Repository CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None)
    return [
        Repository(id=int(row.iloc[0]), owner=str(row.iloc[1]), name=str(row.iloc[2]))
        for _, row in df.iterrows()
    ]


def target_repos() -> List[Repository]:
    """Load all repositories from CSV (replaces the database version)."""
    return load_repositories_from_csv()

def normalized_time_label(reference_years: float = REFERENCE_YEARS_DEFAULT) -> str:
    """Return the standard label used for normalized time axes."""
    return f"Normalized Time (years; 100% ≈ {reference_years:.1f}y)"


def normalized_time_ticks(
    reference_years: float = REFERENCE_YEARS_DEFAULT,
    step: float = REFERENCE_TICK_STEP_DEFAULT,
) -> Tuple[List[float], List[str]]:
    """Return tick positions and labels for a normalized timeline."""
    if reference_years <= 0 or step <= 0:
        return [], []
    marks = np.arange(0.0, reference_years + 1e-9, step)
    return marks.round(10).tolist(), [f"{m:.1f}y" for m in marks]


def apply_elapsed_years_axis(
    ax: plt.Axes,
    reference_years: float = REFERENCE_YEARS_DEFAULT,
    *,
    xticks: Optional[List[float]] = None,
    xticklabels: Optional[List[str]] = None,
    edge_margin_fraction: float = 0.02,
) -> Tuple[List[float], List[str]]:
    """Configure an axis to use an elapsed-years scale capped at reference_years."""
    if reference_years <= 0:
        ax.set_xlim(0, 0)
        return [], []

    ticks, labels = (xticks, xticklabels) if xticks and xticklabels else normalized_time_ticks(reference_years)
    margin = max(0.0, reference_years * edge_margin_fraction) if edge_margin_fraction and reference_years > 0 else 0.0

    ax.set_xlim(-margin, reference_years + margin)
    if ticks and labels:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    return ticks, labels


def save_figure(path: Path, fig: Optional[plt.Figure] = None) -> None:
    """
    Apply tight layout, persist the figure, and close it to free resources.
    """
    if fig is None:
        fig = plt.gcf()
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_date_axis(
    ax: plt.Axes,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    *,
    rotation: Optional[int] = 45,
    interval_months: Optional[int] = None,
    date_format: str = "%Y-%m",
) -> None:
    """Configure a matplotlib axis to use monthly date ticks/limits."""
    start = pd.to_datetime(start) if start is not None else None
    end = pd.to_datetime(end) if end is not None else None

    if interval_months is None:
        if start and end:
            total_months = max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
            interval_months = max(1, total_months // 12)
        else:
            interval_months = 1

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    if start and end:
        ax.set_xlim(start, end)
    if rotation is not None:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)


def write_counts_csv(
    labelled_series: List[tuple[str, pd.Series]],
    index_name: str,
    output_csv: Path,
    *,
    index_formatter: Optional[Callable[[Any], Any]] = None,
    enforce_integer: bool = True,
    value_round: Optional[int] = None,
    extra_series: Optional[Dict[str, pd.Series]] = None,
) -> None:
    """Save long-format counts with optional extra aligned columns."""
    normalized_extras = {
        k: pd.Series(v) if not isinstance(v, pd.Series) else v
        for k, v in (extra_series or {}).items() if v is not None
    }
    export_frames: List[pd.DataFrame] = []

    for label_value, series in labelled_series:
        if series is None or series.empty:
            continue
        tmp = series.copy()
        if isinstance(tmp.index, pd.PeriodIndex):
            tmp.index = tmp.index.to_timestamp(how="end")
        tmp.index.name = index_name
        tmp = tmp.fillna(0)
        if enforce_integer:
            try:
                tmp = tmp.astype(int)
            except (ValueError, TypeError):
                tmp = tmp.round().astype(int)
        else:
            tmp = tmp.round(value_round).astype(float) if value_round else tmp.astype(float)
        
        tmp_reset = tmp.reset_index(name="value")
        if index_formatter:
            tmp_reset[index_name] = tmp_reset[index_name].apply(index_formatter)
        for col_name, extra in normalized_extras.items():
            tmp_reset[col_name] = extra.reindex(tmp.index).values
        tmp_reset["label"] = label_value
        export_frames.append(tmp_reset)

    if not export_frames:
        pd.DataFrame(columns=[index_name, "value", "label"]).to_csv(output_csv, index=False)
    else:
        pd.concat(export_frames, ignore_index=True).to_csv(output_csv, index=False)


def normalized_time_tick_pairs(
    num_bins: int,
    reference_years: float = REFERENCE_YEARS_DEFAULT,
    step: float = REFERENCE_TICK_STEP_DEFAULT,
) -> List[Tuple[int, str]]:
    """Generate (position, label) pairs for discrete normalized timelines."""
    if num_bins < 0 or reference_years <= 0 or step <= 0:
        return []

    tick_pairs: List[Tuple[int, str]] = []
    for mark in np.arange(0.0, reference_years + 1e-9, step):
        position = max(0, min(num_bins, int(round(mark / reference_years * num_bins if reference_years > 0 else 0))))
        if not tick_pairs or tick_pairs[-1][0] != position:
            tick_pairs.append((position, f"{mark:.1f}y"))
    return tick_pairs


def elapsed_years_label(reference_years: float = REFERENCE_YEARS_DEFAULT) -> str:
    """Return the standard label for elapsed-years axes (0-based, capped timeline)."""
    return "Elapsed Years Since First Release"


def compute_elapsed_years(timestamps: pd.Series) -> pd.Series:
    """Compute elapsed time in years from the first timestamp in a monthly series."""
    if timestamps.empty:
        return pd.Series([], dtype=float)
    normalized_ts = pd.to_datetime(timestamps)
    month_index = (normalized_ts.dt.year * 12 + normalized_ts.dt.month).astype("int64")
    return ((month_index - month_index.min()).astype(float) / 12.0)


def clamp_elapsed_years(
    df: pd.DataFrame,
    elapsed_years_col: str = "elapsed_years",
    reference_years: float = REFERENCE_YEARS_DEFAULT,
) -> pd.DataFrame:
    """Return a copy of the dataframe limited to the requested reference window."""
    if elapsed_years_col not in df.columns:
        raise KeyError(f"{elapsed_years_col} column not found in dataframe.")

    mask = df[elapsed_years_col] <= reference_years + 1e-9
    return df.loc[mask].copy()


def plot_content_type_lines(
    ax: plt.Axes,
    relative_data_map: Dict[Any, Dict[str, Any]],
    *,
    line_colors: Optional[Dict[Any, str]] = None,
    line_styles: Optional[Dict[Any, str]] = None,
    line_widths: Optional[Dict[Any, float]] = None,
    default_linewidth: float = 2.5,
    sort_key: Optional[Callable[[Any], Any]] = None,
) -> bool:
    """Plot time-series lines for multiple content types on a shared axis."""
    if ax is None:
        raise ValueError("An active matplotlib axis is required.")
    if not relative_data_map:
        return False

    def _resolve(mapping: Optional[Dict[Any, str]], key: Any, default: Optional[str] = None):
        if not mapping:
            return default
        return mapping.get(key) or mapping.get(str(key)) or default

    items = sorted(relative_data_map.items(), key=lambda kv: sort_key(kv[0]) if sort_key else str(kv[0]))
    has_plotted = False
    
    for content_type, data in items:
        if not data:
            continue
        x_vals = np.asarray(data.get("x_values", []))
        y_vals = np.asarray(data.get("linked_ratio", []))
        if x_vals.size == 0 or y_vals.size == 0 or x_vals.size != y_vals.size:
            continue

        ax.plot(
            x_vals, y_vals,
            label=data.get("title_suffix") or getattr(content_type, "value", str(content_type)),
            color=_resolve(line_colors, content_type),
            linestyle=_resolve(line_styles, content_type, "-"),
            linewidth=line_widths.get(content_type) if line_widths and content_type in line_widths else default_linewidth,
        )
        has_plotted = True
    return has_plotted


# ------------------------------------------------------------------ #
# Common statistics and plotting utilities
# ------------------------------------------------------------------ #


def compute_elapsed_month_ticks(max_elapsed_month: int) -> Tuple[List[int], List[str]]:
    """Return tick positions and labels for elapsed-month axes."""
    if max_elapsed_month <= 0:
        return [0], ["0"]

    thresholds = [(24, 3), (60, 6), (120, 12), (240, 24)]
    step = next((s for limit, s in thresholds if max_elapsed_month <= limit), 36)
    step = max(1, step)
    ticks = list(range(0, max_elapsed_month + step, step))
    if ticks[-1] != max_elapsed_month:
        ticks.append(max_elapsed_month)
    return ticks, [str(t) for t in ticks]


def apply_elapsed_month_axis(
    ax: plt.Axes, max_elapsed_month: int
) -> Tuple[List[int], List[str]]:
    """Configure an axis to display elapsed months since first release."""
    ticks, labels = compute_elapsed_month_ticks(max_elapsed_month)
    ax.set_xlim(0, max(0, max_elapsed_month))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(ELAPSED_MONTHS_LABEL, fontsize=FONT_SIZE)
    return ticks, labels


def compute_series_stats(values: Any) -> Optional[Dict[str, float]]:
    """Return basic statistics (mean, median, sum, count) for a numeric sequence."""
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return None
    return {
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "sum": float(np.sum(finite_values)),
        "count": int(finite_values.size),
    }


def format_stats_lines(
    title: Optional[str],
    stats: Dict[str, float],
    *,
    decimals: int = 1,
    mean_decimals: Optional[int] = None,
    median_decimals: Optional[int] = None,
    include_count: bool = False,
    include_sum: bool = False,
    slope_per_year: Optional[float] = None,
    slope_format: str = "year",
    slope_unit: Optional[str] = None,
) -> List[str]:
    """Format statistics for displaying inside a plot."""
    mean_decimals = mean_decimals or decimals
    median_decimals = median_decimals or decimals
    lines = [title] if title else []

    if slope_unit == "days":
        lines.append(f"mean: {stats['mean']:.{mean_decimals}f}days  median: {stats['median']:.{median_decimals}f}days")
        if include_count and "count" in stats:
            lines.append(f"count: {int(stats['count'])}")
        if slope_per_year is not None:
            lines.append(f"slope: {slope_per_year:.2f}days/year")
    else:
        lines.append(f"mean: {stats['mean']:.{mean_decimals}f}  median: {stats['median']:.{median_decimals}f}")
        if include_sum and "sum" in stats:
            lines.append(f"total: {stats['sum']:.0f}")
        if include_count and "count" in stats:
            lines.append(f"count: {int(stats['count'])}")
        if slope_per_year is not None:
            lines.append(f"slope: {slope_per_year:.3f}/year" if slope_format == "year" else f"slope: {slope_per_year:.{mean_decimals}f}")
    return lines


def render_stats_box(
    ax: plt.Axes,
    lines: List[str],
    *,
    location: str = "upper left",
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    edgecolor: Optional[str] = None,
) -> None:
    """Render the provided text lines inside (or just outside) a plot axis."""
    if not lines:
        return

    outside = False
    normalized_location = location.lower().strip()
    outside_prefix = "outside "
    if normalized_location.startswith(outside_prefix):
        outside = True
        normalized_location = normalized_location[len(outside_prefix) :]

    locations = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "center right": (0.98, 0.5, "right", "center"),
        "middle right": (0.98, 0.5, "right", "center"),
        "center left": (0.02, 0.5, "left", "center"),
        "middle left": (0.02, 0.5, "left", "center"),
    }
    simple_locations = {
        "upper left": (0.04, 0.96, "left", "top"),
        "upper right": (0.96, 0.96, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = (
        simple_locations.get(normalized_location) or
        locations.get(normalized_location) or
        locations["upper left"]
    )

    # Apply offsets if provided
    if x_offset is not None:
        x = x + x_offset
        # Ensure x stays within reasonable bounds (with margin for bbox)
        x = max(0.03, min(0.95, x))
    if y_offset is not None:
        y = y + y_offset
        # Ensure y stays within reasonable bounds (with margin for bbox)
        y = max(0.05, min(0.95, y))

    fontsize = STATS_FONT_SIZE

    # Prepare bbox style with optional edgecolor
    bbox_style = dict(boxstyle="round", facecolor="white", alpha=0.8)
    if edgecolor is not None:
        bbox_style["edgecolor"] = edgecolor
        bbox_style["linewidth"] = 2.0

    if outside:
        pad = 0.04
        x = (1.0 + pad if "right" in normalized_location else
             -pad if "left" in normalized_location else 0.5)
        ha = ("left" if "right" in normalized_location else
              "right" if "left" in normalized_location else "center")
        y = (1.0 + pad if "upper" in normalized_location else
             -pad if "lower" in normalized_location else 0.5)
        va = ("bottom" if "upper" in normalized_location else
              "top" if "lower" in normalized_location else "center")
        clip_on = False
    else:
        clip_on = True

    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, bbox=bbox_style, clip_on=clip_on)


def compute_trend_parameters(
    x_values: Any, y_values: Any, *, axis_type: str = "years"
) -> Tuple[Optional[float], Optional[float]]:
    """Return slope per year and intercept for the provided series."""
    if x_values is None or y_values is None:
        return None, None

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    
    if not np.any(mask) or mask.sum() < 2:
        return None, None
    
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    x_years = x_arr / 12.0 if axis_type == "months" else x_arr
    
    if np.allclose(x_years, x_years[0]):
        return None, None

    if scipy_stats is not None:
        try:
            slope, intercept, _, _ = scipy_stats.theilslopes(y_arr, x_years, 0.95)
            return float(slope), float(intercept)
        except Exception:
            pass

    try:
        slope, intercept = np.polyfit(x_years, y_arr, 1)
        return float(slope), float(intercept)
    except Exception:
        return None, None


def plot_trend_line(
    ax: plt.Axes,
    x_values: Any,
    slope_per_year: Optional[float],
    intercept: Optional[float],
    *,
    axis_type: str,
    color: Any,
    label: Optional[str] = "Theil-Sen",
) -> None:
    """Draw a trend line for the provided series."""
    if slope_per_year is None or intercept is None or x_values is None:
        return

    x_arr = np.asarray(x_values, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    if x_arr.size < 2:
        return

    x_line = np.array([x_arr.min(), x_arr.max()], dtype=float)
    if np.allclose(x_line[0], x_line[1]):
        return

    y_line = intercept + (slope_per_year / 12.0 if axis_type == "months" else slope_per_year) * x_line
    ax.plot(x_line, y_line, color=color or "black", linestyle="--", linewidth=2.0, alpha=0.8, label=label)

