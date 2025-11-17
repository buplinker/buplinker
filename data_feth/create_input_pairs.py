"""
Ground Truth用データ抽出スクリプト

各期間からランダムに2つのURを抽出し、それぞれに対して1年以内のPRとのペアを作成する。
"""

import logging
from pathlib import Path
import pandas as pd
import sys
import tqdm
import re
import time
import json
from datetime import datetime
import os
import argparse
import re
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from data_fetch.database.tables import Repository
from data_fetch.database import get as db_get
import project_config as config
from root_util import target_repos, load_data, GroupType, filter_dataframe_by_date_range, get_first_release_date_from_repository
from data_fetch.preprocess.preprocess_pr import preprocess_pr_data

base_path = Path(__file__).parent.parent

NUMBER_OF_DATA_FOR_GROUND_TRUTH = 3
THRESHOLD_DAYS = 365

def setup_repo_logger(output_dir, repo):
    log_file = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_ground_truth.log"
    logger = logging.getLogger(f"{repo.id}_{repo.owner}.{repo.name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    return logger, handler


def extract_data(repo):
    df_release, df_user_review, df_pull_request = load_data(repo.id)

    if df_release.empty or df_user_review.empty or df_pull_request.empty:
        print(f"Skipped ground truth extraction - {repo.owner}.{repo.name} (no data)")
        return

    # Get date range from repository table
    first_release_date, end_date = get_first_release_date_from_repository(repo.id)
    if first_release_date is None:
        print(f"Warning: Could not get first_release_date for {repo.owner}.{repo.name}, skipping filtering")

    output_pairs(repo, df_user_review, df_pull_request, GroupType.UR_PR, date_range=(first_release_date, end_date))
    output_pairs(repo, df_pull_request, df_user_review, GroupType.PR_UR, df_release=df_release, date_range=(first_release_date, end_date))


def compare_versions(ur_version, release_version):
    """バージョン比較を行う（柔軟な形式に対応）"""
    try:
        # バージョン文字列の前処理
        ur_clean = str(ur_version).strip()
        release_clean = str(release_version).strip()
        
        # プレフィックスを削除（v, oc-android-など）
        ur_clean = re.sub(r'^[a-zA-Z-]+-?', '', ur_clean)
        release_clean = re.sub(r'^[a-zA-Z-]+-?', '', release_clean)
        
        # バージョン文字列を分割（.で分割）
        ur_parts = ur_clean.split('.')
        release_parts = release_clean.split('.')
        
        # 各パートを数値に変換（失敗した場合は0）
        def safe_int(part):
            try:
                # 英数字が混在する場合は数字部分のみ抽出
                numbers = re.findall(r'\d+', str(part))
                return int(numbers[0]) if numbers else 0
            except:
                return 0
        
        # バージョン番号を抽出
        ur_nums = [safe_int(part) for part in ur_parts]
        release_nums = [safe_int(part) for part in release_parts]
        
        # パディングして同じ長さにする
        max_len = max(len(ur_nums), len(release_nums))
        ur_nums.extend([0] * (max_len - len(ur_nums)))
        release_nums.extend([0] * (max_len - len(release_nums)))
        
        # バージョン比較
        for ur_num, release_num in zip(ur_nums, release_nums):
            if ur_num > release_num:
                return True
            elif ur_num < release_num:
                return False
        return True  # 完全に同じバージョン
        
    except Exception as e:
        print(f"Version comparison failed: {ur_version} vs {release_version} - {e}")
        return False


def is_valid_pr_ur_link_vectorized(pr_merged_at, ur_created_at_series, ur_review_version_series, df_release_sorted):
    """PR-URのリンクが有効かどうかをベクトル化で判定する
    
    Args:
        pr_merged_at: PRのマージ日
        ur_created_at_series: URの作成日シリーズ
        ur_review_version_series: URのレビューバージョンシリーズ
        df_release: リリースデータフレーム
    
    Returns:
        tuple: (valid_mask: pd.Series, latest_release_date: pd.Timestamp or None)
    """
    try:
        # df_release_sorted は事前ソート・型変換済みを期待
        
        # PRのマージ日以降のリリースを取得
        releases_after_merge = df_release_sorted[df_release_sorted['released_at'] >= pr_merged_at]
        
        if releases_after_merge.empty:
            return pd.Series([False] * len(ur_created_at_series), index=ur_created_at_series.index), None
        
        # 最も直近のリリースを取得
        latest_release = releases_after_merge.iloc[0]
        latest_release_date = latest_release['released_at']
        latest_release_version = latest_release['version']
        
        # 条件1: URの作成日が最も直近のリリース日以降（ベクトル化）
        time_condition = (
            (ur_created_at_series >= latest_release_date) & 
            (ur_created_at_series <= latest_release_date + pd.Timedelta(days=THRESHOLD_DAYS))
        )
        
        # 条件2: バージョン比較（ベクトル化）
        version_condition = _compare_versions_vectorized(ur_review_version_series, latest_release_version)
        
        # 両方の条件を満たすURのマスク
        valid_mask = time_condition & version_condition
        
        return valid_mask, latest_release_date
            
    except Exception as e:
        print(f"Error in is_valid_pr_ur_link_vectorized: {e}")
        return pd.Series([False] * len(ur_created_at_series), index=ur_created_at_series.index), None


def _compare_versions_vectorized(ur_versions, release_version):
    """バージョン比較をベクトル化で実行"""
    
    def safe_int(part):
        try:
            numbers = re.findall(r'\d+', str(part))
            return int(numbers[0]) if numbers else 0
        except:
            return 0
    
    def compare_single_version(ur_version):
        try:
            # バージョン文字列の前処理
            ur_clean = str(ur_version).strip()
            release_clean = str(release_version).strip()
            
            # プレフィックスを削除
            ur_clean = re.sub(r'^[a-zA-Z-]+-?', '', ur_clean)
            release_clean = re.sub(r'^[a-zA-Z-]+-?', '', release_clean)
            
            # バージョン文字列を分割
            ur_parts = ur_clean.split('.')
            release_parts = release_clean.split('.')
            
            # 各パートを数値に変換
            ur_nums = [safe_int(part) for part in ur_parts]
            release_nums = [safe_int(part) for part in release_parts]
            
            # パディングして同じ長さにする
            max_len = max(len(ur_nums), len(release_nums))
            ur_nums.extend([0] * (max_len - len(ur_nums)))
            release_nums.extend([0] * (max_len - len(release_nums)))
            
            # バージョン比較
            for ur_num, release_num in zip(ur_nums, release_nums):
                if ur_num > release_num:
                    return True
                elif ur_num < release_num:
                    return False
            return True  # 完全に同じバージョン
            
        except Exception:
            return False
    
    # ベクトル化されたバージョン比較
    return ur_versions.apply(compare_single_version)


def output_pairs(repo, base_df, target_df, group_type: GroupType, df_release=None, date_range=None) -> None:
    start_time = time.time()

    sub_path = "ground_truth"
    subsub_path = "all"
    output_dir = Path(__file__).parent.parent / "ur_pr_easylink" / sub_path / subsub_path / group_type.value
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, handler = setup_repo_logger(output_dir, repo)

    logger.info(f"Start output_pairs for {repo.id}: {repo.owner}.{repo.name}")

    # Filter base_df by date range
    if date_range is not None:
        start_date, end_date = date_range
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        original_count = len(base_df)
        base_df = filter_dataframe_by_date_range(base_df, start_date_ts, end_date_ts, date_column='created_at')
        logger.info(f"Filtered base_df from {original_count} to {len(base_df)} rows (date range: {start_date} to {end_date})")

    df_all = create_pairs(base_df, target_df, group_type, df_release, logger=logger)
    
    if not df_all.empty:
        df_all = preprocess_pr_data(df_all, logger=logger).drop(columns=['description_html'])
    else:
        logger.info("  No data to preprocess")
   
    if 'description' in df_all.columns:
        df_all['description'] = df_all['description'].astype(str).str.replace('\n', '\\n').str.replace('\r', '\\r')
    
    output_file = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_ground_truth.csv"
    df_all.to_csv(output_file, index=False)
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "repository_id": repo.id,
        "repository_owner": repo.owner,
        "repository_name": repo.name,
        "output_path": str(output_file),
        "group_type": group_type.value,
        "num_rows": len(df_all),
        "runtime_seconds": round(time.time() - start_time, 3),
    }

    json_path = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_ground_truth_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Finish output_pairs for {repo.owner}.{repo.name} - total rows: {len(df_all)}")
    if handler:
        logger.removeHandler(handler)
        handler.close()


def extract_random_data(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, n: int = 2, logger=None) -> pd.DataFrame:
    period_data = df[
        (df['created_at'] >= start_date) & 
        (df['created_at'] <= end_date)
    ]
    
    # データが不足している場合は警告
    if len(period_data) < n:
        logger.warning(f"  Warning: Only {len(period_data)} data available in period({start_date}, {end_date}), requested {n}")
        return period_data
    
    # ランダムにn個抽出
    return period_data.sample(n=n, random_state=42)


def create_pairs(base_df: pd.DataFrame, target_df: pd.DataFrame, group_type: GroupType, df_release: pd.DataFrame=None, logger=None) -> pd.DataFrame:
    output_data = []
    
    # 事前に必要な日付列を型変換
    if 'created_at' in base_df.columns:
        base_df = base_df.copy()
        base_df['created_at'] = pd.to_datetime(base_df['created_at'])
    if 'created_at' in target_df.columns:
        target_df = target_df.copy()
        target_df['created_at'] = pd.to_datetime(target_df['created_at'])

    # リリースは一度だけソート&型変換
    df_release_sorted = None
    if df_release is not None and not df_release.empty:
        df_release_sorted = df_release.copy()
        df_release_sorted['released_at'] = pd.to_datetime(df_release_sorted['released_at'])
        df_release_sorted = df_release_sorted.sort_values('released_at')

    if group_type == GroupType.UR_PR:
        # 高速化: PR側を作成日時でソートし、二分探索で範囲抽出
        pr_df_sorted = target_df.sort_values('created_at').reset_index(drop=True)
        pr_times = pr_df_sorted['created_at'].to_numpy(dtype='datetime64[ns]')

        # ループは itertuples で高速化
        for base_row in tqdm.tqdm(base_df.itertuples(index=False), total=len(base_df)):
            base_date = getattr(base_row, 'created_at')
            end_date = base_date + pd.Timedelta(days=THRESHOLD_DAYS)
            start_idx = np.searchsorted(pr_times, np.datetime64(base_date.to_datetime64()), side='left')
            end_idx = np.searchsorted(pr_times, np.datetime64(end_date.to_datetime64()), side='right')

            if end_idx <= start_idx:
                continue

            candidates = pr_df_sorted.iloc[start_idx:end_idx]
            for pr_row in candidates.itertuples(index=False):
                pair_data = {
                    'repository_id': int(getattr(base_row, 'repository_id')),
                    'ur_id': getattr(base_row, 'id'),
                    'pr_id': getattr(pr_row, 'id'),
                    'create_date': getattr(base_row, 'created_at'),
                    'pull_request_date': getattr(pr_row, 'created_at'),
                    'latest_release_date': None,
                    'review': getattr(base_row, 'content'),
                    'title': getattr(pr_row, 'title'),
                    'description': getattr(pr_row, 'bodyText'),
                    'description_html': getattr(pr_row, 'bodyHtml'),
                    'url': getattr(pr_row, 'url'),
                    'author': getattr(pr_row, 'author'),
                    'target': 0,
                }
                output_data.append(pair_data)
    
    else:
        # PR-URの場合はベクトル化された処理を使用
        # 日付を事前に変換
        target_created_at_series = target_df['created_at']
        target_review_version_series = target_df['review_version']
        
        for base_row in tqdm.tqdm(base_df.itertuples(index=False), total=len(base_df)):
            pr_merged_at = pd.to_datetime(getattr(base_row, 'merged_at'))
            
            # ベクトル化された関数を使用してPR-URリンクをチェック
            valid_mask, latest_release_date = is_valid_pr_ur_link_vectorized(
                pr_merged_at,
                target_created_at_series,
                target_review_version_series,
                df_release_sorted,
            )
            
            # 条件を満たすURを取得
            valid_targets = target_df[valid_mask].copy()
            if latest_release_date is not None:
                valid_targets['latest_release_date'] = latest_release_date
            
            valid_targets = valid_targets.sort_values('created_at')
            selected_targets = valid_targets
                
            for target_row in selected_targets.itertuples(index=False):
                pair_data = {
                    'repository_id': int(getattr(target_row, 'repository_id')),
                    'ur_id': getattr(target_row, 'id'),
                    'pr_id': getattr(base_row, 'id'),
                    'create_date': getattr(target_row, 'created_at'),
                    'pull_request_date': getattr(base_row, 'created_at'),
                    'latest_release_date': getattr(target_row, 'latest_release_date', None),
                    'review': getattr(target_row, 'content'),
                    'title': getattr(base_row, 'title'),
                    'description': getattr(base_row, 'bodyText'),
                    'description_html': getattr(base_row, 'bodyHtml'),
                    'url': getattr(base_row, 'url'),
                    'author': getattr(base_row, 'author'),
                    'target': 0,
                }
                output_data.append(pair_data)
    
    return pd.DataFrame(output_data)


def _process_single_repo_entry(repo):
    extract_data(repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract input pairs')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                       help='Number of parallel worker processes for repositories')
    args = parser.parse_args()
    
    repos = list(target_repos())
    print(f"Target repositories: {len(repos)}; workers={args.workers}")

    if args.workers and args.workers > 1 and len(repos) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(_process_single_repo_entry, repos))
    else:
        for repo in repos:
            _process_single_repo_entry(repo)
            print(f"Finished processing {repo.id}: {repo.owner}.{repo.name}")
