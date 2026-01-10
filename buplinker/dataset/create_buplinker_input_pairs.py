"""
Ground Truth data extraction script

Extract 2 URs randomly from each period and create pairs with PRs within 1 year for each.
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

sys.path.append(str(Path(__file__).parent.parent.parent))

from root_util import target_repos, load_data, GroupType, filter_dataframe_by_date_range, get_first_release_date_from_repository
from buplinker.dataset.preprocess.preprocess_pr import preprocess_pr_data

NUMBER_OF_DATA_FOR_GROUND_TRUTH = 3
THRESHOLD_DAYS = 365

def setup_repo_logger(output_dir, repo):
    log_file = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_input_pairs.log"
    logger = logging.getLogger(f"{repo.id}_{repo.owner}.{repo.name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    return logger, handler


def extract_data(repo, limited_mode=False):
    df_release, df_user_review, df_pull_request = load_data(repo.id)
    df_user_review = df_user_review[(df_user_review['intention'] != None) & (df_user_review['intention'] != "['OTHER']")]

    if df_release.empty or df_user_review.empty or df_pull_request.empty:
        print(f"Skipped input pairs extraction - {repo.owner}.{repo.name} (no data)")
        return

    # Limited mode: Get date range from repository table
    if limited_mode:
        first_release_date, end_date = get_first_release_date_from_repository(repo.id)
        if first_release_date is None:
            print(f"Warning: Could not get first_release_date for {repo.owner}.{repo.name}, skipping limited mode filtering")
            return
        else:
            date_range = (first_release_date, end_date)
    else:
        date_range = None

    output_pairs(repo, df_user_review, df_pull_request, GroupType.UR_PR, date_range=date_range)
    output_pairs(repo, df_pull_request, df_user_review, GroupType.PR_UR, df_release=df_release, date_range=date_range)
   

def compare_versions(ur_version, release_version):
    """Compare versions (supports flexible formats)"""
    try:
        # Preprocess version strings
        ur_clean = str(ur_version).strip()
        release_clean = str(release_version).strip()
        
        # Remove prefixes (v, oc-android-, etc.)
        ur_clean = re.sub(r'^[a-zA-Z-]+-?', '', ur_clean)
        release_clean = re.sub(r'^[a-zA-Z-]+-?', '', release_clean)
        
        # Split version strings (split by .)
        ur_parts = ur_clean.split('.')
        release_parts = release_clean.split('.')
        
        # Convert each part to number (0 if failed)
        def safe_int(part):
            try:
                # Extract only numeric part if alphanumeric mixed
                numbers = re.findall(r'\d+', str(part))
                return int(numbers[0]) if numbers else 0
            except:
                return 0
        
        # Extract version numbers
        ur_nums = [safe_int(part) for part in ur_parts]
        release_nums = [safe_int(part) for part in release_parts]
        
        # Pad to same length
        max_len = max(len(ur_nums), len(release_nums))
        ur_nums.extend([0] * (max_len - len(ur_nums)))
        release_nums.extend([0] * (max_len - len(release_nums)))
        
        # Compare versions
        for ur_num, release_num in zip(ur_nums, release_nums):
            if ur_num > release_num:
                return True
            elif ur_num < release_num:
                return False
        return True  # Completely same version
        
    except Exception as e:
        print(f"Version comparison failed: {ur_version} vs {release_version} - {e}")
        return False


def is_valid_pr_ur_link_vectorized(pr_merged_at, ur_created_at_series, ur_review_version_series, df_release_sorted):
    """Determine if PR-UR link is valid using vectorization
    
    Args:
        pr_merged_at: PR merge date
        ur_created_at_series: UR creation date series
        ur_review_version_series: UR review version series
        df_release: Release dataframe
    
    Returns:
        tuple: (valid_mask: pd.Series, latest_release_date: pd.Timestamp or None)
    """
    try:
        # df_release_sorted is expected to be pre-sorted and type-converted        
        # Get releases after PR merge date
        releases_after_merge = df_release_sorted[df_release_sorted['released_at'] >= pr_merged_at]
        
        if releases_after_merge.empty:
            return pd.Series([False] * len(ur_created_at_series), index=ur_created_at_series.index), None
        
        # Get most recent release
        latest_release = releases_after_merge.iloc[0]
        latest_release_date = latest_release['released_at']
        latest_release_version = latest_release['version']
        
        # Condition 1: UR creation date is after most recent release date (vectorized)
        time_condition = (
            (ur_created_at_series >= latest_release_date) & 
            (ur_created_at_series <= latest_release_date + pd.Timedelta(days=THRESHOLD_DAYS))
        )
        
        # Condition 2: Version comparison (vectorized)
        version_condition = _compare_versions_vectorized(ur_review_version_series, latest_release_version)
        
        # Mask for URs satisfying both conditions
        valid_mask = time_condition & version_condition
        
        return valid_mask, latest_release_date
            
    except Exception as e:
        print(f"Error in is_valid_pr_ur_link_vectorized: {e}")
        return pd.Series([False] * len(ur_created_at_series), index=ur_created_at_series.index), None


def _compare_versions_vectorized(ur_versions, release_version):
    """Execute version comparison using vectorization"""
    
    def safe_int(part):
        try:
            numbers = re.findall(r'\d+', str(part))
            return int(numbers[0]) if numbers else 0
        except:
            return 0
    
    def compare_single_version(ur_version):
        try:
            # Preprocess version strings
            ur_clean = str(ur_version).strip()
            release_clean = str(release_version).strip()
            
            # Remove prefixes
            ur_clean = re.sub(r'^[a-zA-Z-]+-?', '', ur_clean)
            release_clean = re.sub(r'^[a-zA-Z-]+-?', '', release_clean)
            
            # Split version strings
            ur_parts = ur_clean.split('.')
            release_parts = release_clean.split('.')
            
            # Convert each part to number
            ur_nums = [safe_int(part) for part in ur_parts]
            release_nums = [safe_int(part) for part in release_parts]
            
            # Pad to same length
            max_len = max(len(ur_nums), len(release_nums))
            ur_nums.extend([0] * (max_len - len(ur_nums)))
            release_nums.extend([0] * (max_len - len(release_nums)))
            
            # Compare versions
            for ur_num, release_num in zip(ur_nums, release_nums):
                if ur_num > release_num:
                    return True
                elif ur_num < release_num:
                    return False
            return True  # Completely same version
            
        except Exception:
            return False
    
    # Vectorized version comparison
    return ur_versions.apply(compare_single_version)


def output_pairs(repo, base_df, target_df, group_type: GroupType, df_release=None, date_range=None) -> None:
    start_time = time.time()

    subsub_path = "limited_years" if date_range else "all_years"
    output_dir = Path(__file__).parent / "input_pairs" / group_type.value / subsub_path
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, handler = setup_repo_logger(output_dir, repo)

    logger.info(f"Start output_pairs for {repo.id}: {repo.owner}.{repo.name}")

    # Limited mode: Filter base_df by date range
    if date_range:
        start_date, end_date = date_range
        start_date_ts, end_date_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        original_count = len(base_df)
        base_df = filter_dataframe_by_date_range(base_df, start_date_ts, end_date_ts, date_column='created_at')
        logger.info(f"Filtered base_df from {original_count} to {len(base_df)} rows (date range: {start_date} to {end_date})")

    df_all = create_pairs(base_df, target_df, group_type, df_release)
    
    if not df_all.empty:
        df_all = preprocess_pr_data(df_all, logger=logger).drop(columns=['description_html'])
    else:
        logger.info("  No data to preprocess")
   
    if 'description' in df_all.columns:
        df_all['description'] = df_all['description'].astype(str).str.replace('\n', '\\n').str.replace('\r', '\\r')
    
    output_file = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_input_pairs.csv"
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

    json_path = output_dir / f"{repo.id}_{repo.owner}.{repo.name}_input_pairs_summary.json"
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
    
    # Warn if insufficient data
    if len(period_data) < n:
        logger.warning(f"  Warning: Only {len(period_data)} data available in period({start_date}, {end_date}), requested {n}")
        return period_data
    
    # Randomly extract n items
    return period_data.sample(n=n, random_state=42)


def create_pairs(base_df: pd.DataFrame, target_df: pd.DataFrame, group_type: GroupType, df_release: pd.DataFrame=None) -> pd.DataFrame:
    output_data = []
    
    # Type convert date columns
    base_df['created_at'] = pd.to_datetime(base_df['created_at'])
    target_df['created_at'] = pd.to_datetime(target_df['created_at'])

    if group_type == GroupType.UR_PR:
        # Optimization: Sort PR side by creation date and extract range using binary search
        pr_df_sorted = target_df.sort_values('created_at').reset_index(drop=True)
        pr_times = pr_df_sorted['created_at'].to_numpy(dtype='datetime64[ns]')

        # Optimize loop with itertuples
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
        # Use vectorized processing for PR-UR case
        # Pre-convert dates
        target_created_at_series = target_df['created_at']
        target_review_version_series = target_df['review_version']

        # Sort and type-convert releases only once
        df_release_sorted = df_release.copy()
        df_release_sorted['released_at'] = pd.to_datetime(df_release_sorted['released_at'])
        df_release_sorted = df_release_sorted.sort_values('released_at')
        
        for base_row in tqdm.tqdm(base_df.itertuples(index=False), total=len(base_df)):
            pr_merged_at = pd.to_datetime(getattr(base_row, 'merged_at'))
            
            # Check PR-UR links using vectorized function
            valid_mask, latest_release_date = is_valid_pr_ur_link_vectorized(
                pr_merged_at,
                target_created_at_series,
                target_review_version_series,
                df_release_sorted,
            )
            
            # Get URs satisfying conditions
            valid_targets = target_df[valid_mask].copy()
            if latest_release_date is not None:
                valid_targets['latest_release_date'] = latest_release_date
            
            valid_targets = valid_targets.sort_values('created_at')
                
            for target_row in valid_targets.itertuples(index=False):
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


def _process_single_repo_entry(args_tuple):
    repo, limited_mode = args_tuple
    print(f"{repo.id}: {repo.owner}.{repo.name}")
    extract_data(repo, limited_mode=limited_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create BUPLinker input pairs')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                       help='Number of parallel worker processes for repositories')
    parser.add_argument('--limited', action='store_true', 
                       help='Extract limited_years data instead of all_years data')
    args = parser.parse_args()
    
    repos = list(target_repos())
    print(f"Target repositories: {len(repos)}; workers={args.workers}, limited={args.limited}")

    if args.workers and args.workers > 1 and len(repos) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(_process_single_repo_entry, [(repo, args.limited) for repo in repos]))
    else:
        for repo in repos:
            _process_single_repo_entry((repo, args.limited))
            print(f"Finished processing {repo.id}: {repo.owner}.{repo.name}")
