#!/usr/bin/env python3
"""
Timeline data processing script for user reviews with topic classification,
label assignment, and time difference calculation.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import time
import json
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from root_util import target_repos, ContentType, GroupType, load_data
from buplinker.dataset.preprocess.preprocess_pr import _get_template_context

base_path = Path(__file__).parent.parent.parent.parent

def get_cached_pr_descriptions(repository_id: int, pr_ids: List[str]) -> List[str]:
    """キャッシュされた前処理済みPRのdescriptionを取得"""
    try:
        # キャッシュから前処理済みデータを取得
        template_context = _get_template_context(repository_id)
        pr_description_cache = template_context.get("pr_description_without_template", {})
        
        cached_descriptions = []
        for pr_id in pr_ids:
            if pr_id in pr_description_cache:
                cached_descriptions.append(pr_description_cache[pr_id])
            else:
                # キャッシュにない場合は空文字列を返す
                cached_descriptions.append("")
        
        return cached_descriptions
    except Exception as e:
        print(f"Error loading cached PR descriptions: {e}")
        return [""] * len(pr_ids)

def calculate_time_diff(create_time: Union[str, pd.Timestamp], 
                       reference_time: Union[str, pd.Timestamp]) -> Optional[int]:
    """Calculate time difference between two dates in days."""
    try:
        if pd.isna(create_time) or pd.isna(reference_time):
            return None
        create_dt = pd.to_datetime(create_time)
        ref_dt = pd.to_datetime(reference_time)
        return (ref_dt - create_dt).days
    except:
        return None

def process_repository(repo, group_type: GroupType, limited: bool = False):
    start_time = time.time()

    _, df_user_review, df_pull_request = load_data(repo.id)

    year_subdir = "limited_years" if limited else "all_years"
    buplinker_results_path = f"{base_path}/buplinker/code/output/{group_type.value}/{year_subdir}/results/{repo.id}_{repo.owner}.{repo.name}_result_buplinker_5.csv"
    if not os.path.exists(buplinker_results_path):
        print(f"Error: {buplinker_results_path} not found")
        return
    buplinker_df = pd.read_csv(buplinker_results_path)

    if group_type == GroupType.UR_PR:
        df = df_user_review
        id_column = 'ur_id'
        created_at_column = 'create_date'
        related_at_column = 'pull_request_date'
    else: 
        df = df_pull_request
        id_column = 'pr_id'
        created_at_column = 'latest_release_date'
        related_at_column = 'create_date'
    
    # Assign labels
    if id_column not in buplinker_df.columns:
        df['label'] = 0
        df['diff_times'] = None
        print(f"Error: {id_column} not found in buplinker_df")
    else:
        buplinker_ids = set(buplinker_df[id_column].tolist())
        df['label'] = df['id'].apply(lambda x: 1 if x in buplinker_ids else 0)
        
        # Calculate time differences
        diff_times = []
        for _, row in df.iterrows():
            item_id = row['id']
            matching_items = buplinker_df[buplinker_df[id_column] == item_id]
            
            if len(matching_items) > 0:
                create_time = buplinker_df[buplinker_df[id_column] == item_id][created_at_column].values[0]
                time_values = matching_items[related_at_column].tolist()
                diff_time_list = [calculate_time_diff(create_time, time_val) for time_val in time_values]
                # Remove None values
                diff_time_list = [dt for dt in diff_time_list if dt is not None]
                diff_times.append(diff_time_list if diff_time_list else None)
            else:
                diff_times.append(None)
    
        df['diff_times'] = diff_times
    
    # Save results
    output_dir = f"{base_path}/analysis/timeline/time_processed_data/{year_subdir}/"
    content_type_value = ContentType.UR.value if group_type == GroupType.UR_PR else ContentType.PR.value
    output_path = f"{output_dir}/data/{repo.id}_{repo.owner}.{repo.name}_{content_type_value}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "repository_id": repo.id,
        "repository_owner": repo.owner,
        "repository_name": repo.name,
        "group_type": group_type.value,
        "num_rows": len(df),
        "elapsed_time": round(time.time() - start_time, 3),
    }
    json_path = f"{output_dir}/summary/{repo.id}_{repo.owner}.{repo.name}_{group_type.value}_summary.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--limited", action="store_true", help="Use limited years data")
    args = arg.parse_args()
    print(f"Using limited years data: {args.limited}")

    for repo in target_repos():
        print(f"Processing {repo.id}: {repo.owner}.{repo.name}...")
        process_repository(repo, GroupType.UR_PR, args.limited)
        process_repository(repo, GroupType.PR_UR, args.limited)
        print("Done!")