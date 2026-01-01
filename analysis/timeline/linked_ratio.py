from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from root_util import ContentType, CategoryType, target_repos  # noqa: E402
from data_fetch.database.tables import Repository
from analysis.timeline.base_plotter import BaseLinkedPlotter
from analysis.timeline.statistics_analyzer import StatisticsAnalyzer  # noqa: E402

class LinkedRatioPlotter(BaseLinkedPlotter):
    def __init__(self, limited: bool):
        super().__init__("linked_ratio", limited)


    def _load_repo_data(self, repo: Repository, content_type: ContentType) -> pd.DataFrame:
        csv_file = (self.data_path / f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}.csv")
        if not csv_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(csv_file)
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    def _group_by_month(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "created_at" not in df.columns:
            return pd.DataFrame()

        df["month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            df.groupby("month")
            .agg(
                not_linked=("label", lambda x: (x == 0).sum()),
                linked=("label", lambda x: (x == 1).sum()),
            )
            .reset_index()
            .sort_values("month")
        )

        monthly["total"] = monthly["not_linked"] + monthly["linked"]
        with np.errstate(invalid="ignore", divide="ignore"):
            monthly["metric"] = np.divide(
                monthly["linked"],
                monthly["total"],
                out=np.zeros_like(monthly["linked"], dtype=float),
                where=monthly["total"] > 0,
            )
        monthly["count"] = monthly["total"].astype(int)

        return monthly.sort_values("month")

    def _get_csv_column_order(self) -> List[str]:
        return [
            "elapsed_month",
            "elapsed_years",
            "metric",
            "count",
        ]

    def _get_y_axis_label(self) -> str:
        return "Linked Ratio"

    def _get_y_axis_limits(
        self, entries: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        return 0.0, 1.0

    def _format_stats_lines(
        self, stats: Dict[str, float], slope: Optional[float]
    ) -> List[str]:
        slope_str = f"{slope:.3f}" if slope is not None else "N/A"
        return [
            f"mean: {stats['mean']:.3f}  median: {stats['median']:.3f}",
            f"slope: {slope_str}/year",
        ]


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--limited", action="store_true", help="Use limited years data")
    args = arg.parse_args()
    print(f"Using limited years data: {args.limited}")
    plotter = LinkedRatioPlotter(args.limited)

    repos = target_repos()
    repo_entries: Dict[int, Dict[ContentType, Dict[str, Any]]] = {}
    combined_metric_df: Dict[ContentType, pd.DataFrame] = {}
    for repo in repos:
        print(f"Processing {repo.id}: {repo.owner}.{repo.name}...")

        # リポジトリごとのプロット
        combined_entries: Dict[ContentType, Dict[str, Any]] = {}
        repo_entries[repo.id] = {}
        for content_type in (ContentType.UR, ContentType.PR):
            metrics_df = plotter.calculate_monthly_metrics(repo, content_type)
            combined_metric_df[content_type] = pd.concat([combined_metric_df.get(content_type, pd.DataFrame()), metrics_df])
            plotter.save_metrics_csv(repo, content_type, metrics_df)
            entry = plotter._build_line_entry(metrics_df)
            combined_entries[content_type] = entry
            repo_entries[repo.id][content_type] = entry
        plotter.create_repo_ur_pr_plot(repo, combined_entries)

    # 全リポジトリのプロット（個別リポジトリを薄く、平均を濃く）
    plotter.create_all_repos_with_individual_plot(repo_entries, combined_metric_df)

    # 全リポジトリのプロット
    all_repo_entries: Dict[ContentType, Dict[str, Any]] = {}
    for content_type in (ContentType.UR, ContentType.PR):
        metrics_df = plotter.calculate_all_repos_linked_metrics(repos, content_type)
        plotter.save_all_repos_csv(content_type, metrics_df)
        entry = plotter._build_line_entry(metrics_df)
        all_repo_entries[content_type] = entry
    plotter.create_all_repos_ur_pr_plot(all_repo_entries)
    
    # カテゴリごとのプロット
    combined_entries: Dict[ContentType, Dict[CategoryType, Dict[str, Any]]] = {}
    for content_type in [ContentType.UR, ContentType.PR]:
        combined_entries.setdefault(content_type, {})
        for category in [CategoryType.Hedonic, CategoryType.Utilitarian]:
            repos = plotter._get_repositories_by_category()[category]
            metrics_df = plotter.calculate_all_repos_linked_metrics(repos, content_type)
            plotter.save_category_csv(category, content_type, metrics_df)
            entry = plotter._build_line_entry(metrics_df)
            combined_entries[content_type][category] = entry
    plotter.create_category_comparison_plot(combined_entries)
    
    # 統計情報の計算
    # アプリカテゴリ間の比較も行う
    analyzer = StatisticsAnalyzer(metric_key="linked_ratio", limited=args.limited)
    metric_rows = []
    for content_type in [ContentType.UR, ContentType.PR]:
        df = analyzer.load_metric_data(content_type)
        row = analyzer.analyse_content_type(df, content_type)
        if row is not None:
            metric_rows.append(row)
    analyzer.build_metric_combined_csv_summaries(metric_rows)

    # アプリカテゴリごとの統計情報の計算
    category_rows = []
    for category in [CategoryType.Hedonic, CategoryType.Utilitarian]:
        for content_type in [ContentType.UR, ContentType.PR]:
            metrics_df = analyzer.load_category_data(category, content_type)
            row = analyzer.analyse_content_type(metrics_df, content_type, category)
            if row is not None:
                category_rows.append(row)
    analyzer.build_category_combined_csv_summaries(category_rows)