#!/usr/bin/env python3
"""
Script to randomly sample RANDOM_SAMPLE_SIZE rows from all CSV files combined in easylink/ground_truth/ur_pr
and save them grouped by original filename to random/ folder as JSON files.
"""

import pandas as pd
import os
import random
from pathlib import Path
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from root_util import GroupType, get_first_release_date_from_repository, filter_dataframe_by_date_range

RANDOM_SAMPLE_SIZE = 384

def _get_chunk_reader(csv_file_path: Path):
    """
    pandasのバージョンに応じて適切なパラメータでCSVリーダーを取得する
    
    Returns:
        chunk reader iterator
    """
    CHUNK_SIZE = 10000
    
    # Try with newer pandas API (on_bad_lines)
    try:
        return pd.read_csv(csv_file_path, chunksize=CHUNK_SIZE, 
                          on_bad_lines='skip', engine='python')
    except TypeError:
        # Fall back to older pandas API (error_bad_lines)
        try:
            return pd.read_csv(csv_file_path, chunksize=CHUNK_SIZE, 
                             error_bad_lines=False, warn_bad_lines=False, engine='python')
        except TypeError:
            # Very old pandas - no error handling parameter
            return pd.read_csv(csv_file_path, chunksize=CHUNK_SIZE, engine='python')


def _read_rows_from_chunk_reader(chunk_reader, local_indices: set) -> list:
    """
    チャンクリーダーから指定されたインデックスの行を読み込む
    
    Args:
        chunk_reader: pandas chunk reader iterator
        local_indices: 読み込む行インデックスのセット（変更される）
    
    Returns:
        読み込んだ行のリスト
    """
    sampled_rows = []
    current_row_idx = 0
    
    for chunk_df in chunk_reader:
        for idx, row in chunk_df.iterrows():
            if current_row_idx in local_indices:
                sampled_rows.append(row)
                local_indices.remove(current_row_idx)
                if not local_indices:  # All needed rows found
                    return sampled_rows
            current_row_idx += 1
    
    return sampled_rows


def _read_selected_rows_from_file(csv_file_path: Path, local_indices: set) -> list:
    """指定されたCSVファイルから指定されたローカルインデックスの行を読み込む"""
    if not local_indices:
        return []
    
    # Make a copy to avoid modifying the original set
    local_indices_copy = local_indices.copy()
    
    try:
        # Try reading with standard error handling
        chunk_reader = _get_chunk_reader(csv_file_path)
        return _read_rows_from_chunk_reader(chunk_reader, local_indices_copy)
                
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        # Retry with error handling if parser error occurs
        print(f"    - Warning: Parser error, attempting with error handling...")
        try:
            local_indices_copy = local_indices.copy()
            chunk_reader = _get_chunk_reader(csv_file_path)
            return _read_rows_from_chunk_reader(chunk_reader, local_indices_copy)
        except Exception as e:
            print(f"    - Error: Could not process file: {e}")
            return []
    except Exception as e:
        print(f"    - Error processing file: {e}")
        return []


def _read_selected_rows_from_file_with_date_filter(
    csv_file_path: Path, 
    local_indices: set, 
    group_type: GroupType, 
    start_date: datetime, 
    end_date: datetime
) -> list:
    """
    日付フィルタリングを適用しながら、指定されたローカルインデックスの行を読み込む
    （必要な行だけを読み込むため、全データを読み込まない）
    
    Args:
        csv_file_path: CSVファイルのパス
        local_indices: 読み込む行インデックスのセット（日付フィルタリング後のインデックス）
        group_type: GroupType (UR_PR or PR_UR)
        start_date: 開始日（first_release_date）
        end_date: 終了日（first_release_date + 3年）
    
    Returns:
        読み込んだ行のリスト
    """
    if not local_indices:
        return []
    
    date_column = 'create_date' if group_type == GroupType.UR_PR else 'pull_request_date'
    
    filtered_rows = []
    filtered_row_idx = 0  # Index within filtered rows
    
    try:
        chunk_reader = _get_chunk_reader(csv_file_path)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        for chunk_df in chunk_reader:
            # Filter chunk using common function
            filtered_chunk = filter_dataframe_by_date_range(
                chunk_df, start_ts, end_ts, date_column=date_column
            )
            
            # Check if any of the requested indices are in this chunk
            for idx, row in filtered_chunk.iterrows():
                if filtered_row_idx in local_indices:
                    filtered_rows.append(row)
                    local_indices.remove(filtered_row_idx)
                    if not local_indices:  # All needed rows found
                        return filtered_rows
                filtered_row_idx += 1
            
    except Exception as e:
        print(f"    - Error reading file with date filter: {e}")
        return []
    
    return filtered_rows


def _create_random_ground_truth_internal(group_type: GroupType, limited_mode: bool = False):
    """
    内部関数: 全CSVファイルからランダムに384行をサンプリングし、ファイル名ごとにrandomフォルダにJSON形式で保存
    
    Args:
        group_type: GroupType (UR_PR or PR_UR)
        limited_mode: Trueの場合、limited_yearsのstatisticsを使用し、ur_pr_links_total_countとpr_ur_links_total_countの両方が0より大きいリポジトリのみを対象
    """
    random.seed(42)

    base_path = Path(__file__).parent
    
    mode_str = "limited_years" if limited_mode else "all_years"
    statistics_dir = Path(__file__).parent.parent.parent / "data_fetch" / "statistics" / "repository_statistics" / mode_str / "intention"
    dataset_dir = base_path / mode_str / group_type.value

    sub_path = "limited_random" if limited_mode else "random"
    output_dir = base_path / sub_path / group_type.value
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which count field to use based on group_type
    count_field = f"{group_type.value}_links_total_count"
    
    # Scan statistics directory to find all repositories (no database connection needed)
    file_row_counts = []
    
    # Get all statistics files
    stats_files = list(statistics_dir.glob("*_statistics.json"))
    
    for stats_file_path in stats_files:
        try:
            # Parse repository info from filename: {repo.id}_{repo.owner}.{repo.name}_statistics.json
            stats_filename = stats_file_path.name
            # Remove _statistics.json suffix
            repo_info = stats_filename.replace("_statistics.json", "")
            # Split to get repo.id and repo.owner.repo.name
            parts = repo_info.split("_", 1)
            if len(parts) != 2:
                continue
            
            repo_id = parts[0]
            repo_owner_name = parts[1]
            
            # Check if corresponding CSV file exists
            csv_filename = f"{repo_id}_{repo_owner_name}_input_pairs.csv"
            csv_file_path = dataset_dir / csv_filename
            
            if not csv_file_path.exists():
                continue
            
            # Read row count from statistics file
            with open(stats_file_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                
            # Limited mode: Check if both ur_pr_links_total_count and pr_ur_links_total_count are > 0
            if limited_mode:
                ur_pr_count = int(stats.get('ur_pr_links_total_count', 0) or 0)
                pr_ur_count = int(stats.get('pr_ur_links_total_count', 0) or 0)
                if ur_pr_count <= 0 or pr_ur_count <= 0:
                    continue  # Skip repositories where either count is 0
                
                # Get first_release_date and calculate end_date from repository table
                date_range = get_first_release_date_from_repository(int(repo_id))
                if date_range is None:
                    print(f"  - {csv_filename}: Skipping (no first_release_date in repository table)")
                    continue
                
                first_release_date, end_date = date_range
                
                # Use row count from statistics file (already filtered by date range)
                row_count = int(stats.get(count_field, 0) or 0)
                
                if row_count > 0:
                    file_row_counts.append((csv_file_path, row_count, csv_filename, first_release_date, end_date))
                    print(f"  - {csv_filename}: {row_count:,} rows (from statistics, date range: {first_release_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            else:
                row_count = int(stats.get(count_field, 0) or 0)
                
                if row_count > 0:  # Only include files with data
                    file_row_counts.append((csv_file_path, row_count, csv_filename, None, None))
                    print(f"  - {csv_filename}: {row_count:,} rows (from statistics)")
        except Exception as e:
            print(f"  - Error processing {stats_file_path.name}: {e}, skipping")
            continue
    
    if not file_row_counts:
        print(f"No input pairs CSV files found in {dataset_dir} directory")
        return
    
    print(f"\nFound {len(file_row_counts)} input pairs CSV files")
    
    total_rows = sum(file_info[1] for file_info in file_row_counts)
    print(f"\nTotal rows across all files: {total_rows:,}")
    
    if total_rows == 0:
        print("No data found in CSV files")
        return
    
    # Check if we have enough rows to sample RANDOM_SAMPLE_SIZE
    if total_rows < RANDOM_SAMPLE_SIZE:
        print(f"Warning: Only {total_rows} rows available, but {RANDOM_SAMPLE_SIZE} requested. Using all available rows.")
        sample_size = total_rows
    else:
        sample_size = RANDOM_SAMPLE_SIZE
    
    # Select random row indices across all files
    print(f"\nSelecting {sample_size} random row indices from {total_rows:,} total rows...")
    selected_indices = sorted(random.sample(range(total_rows), sample_size))
    
    # Build cumulative row count to map global index to file and local index
    cumulative_counts = []
    current_sum = 0
    for file_info in file_row_counts:
        csv_file_path = file_info[0]
        row_count = file_info[1]
        csv_filename = file_info[2]
        # Get date info if available (for limited mode)
        start_date = file_info[3] if len(file_info) > 3 else None
        end_date = file_info[4] if len(file_info) > 4 else None
        cumulative_counts.append((csv_file_path, current_sum, current_sum + row_count, csv_filename, start_date, end_date))
        current_sum += row_count
    
    # Group selected indices by file
    file_row_map = {}  # {csv_file_path: {'indices': [...], 'filename': ..., 'start_date': ..., 'end_date': ...}}
    for global_idx in selected_indices:
        # Find which file contains this global index
        for csv_file_path, start_idx, end_idx, csv_filename, start_date, end_date in cumulative_counts:
            if start_idx <= global_idx < end_idx:
                local_idx = global_idx - start_idx
                if csv_file_path not in file_row_map:
                    file_row_map[csv_file_path] = {
                        'indices': [], 
                        'filename': csv_filename,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                file_row_map[csv_file_path]['indices'].append(local_idx)
                break
    
    # Create summary of sample sizes per file
    sample_sizes = {}
    for csv_file_path, data in file_row_map.items():
        csv_filename = data['filename']
        json_filename = csv_filename.replace('.csv', '.json')
        sample_sizes[json_filename] = len(data['indices'])
    
    # Save sample sizes to JSON file
    sample_sizes_file = output_dir / "sample_sizes.json"
    with open(sample_sizes_file, 'w', encoding='utf-8') as f:
        json.dump(sample_sizes, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"\nSaved sample sizes to {sample_sizes_file.name}")
    print(f"  - Total files: {len(sample_sizes)}")
    print(f"  - Total samples: {sum(sample_sizes.values())}")
    
    print(f"\nReading selected rows from {len(file_row_map)} files and saving immediately...")
    print("(Processing files with fewer samples first for faster completion...)")
    
    # Sort files by sample size (fewer samples first) for faster processing
    sorted_files = sorted(
        file_row_map.items(),
        key=lambda x: len(x[1]['indices'])
    )
    
    # Read selected rows from each file and save immediately
    total_saved = 0
    files_saved = 0
    
    for csv_file_path, data in sorted_files:
        csv_filename = data['filename']
        local_indices = set(data['indices'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        print(f"Processing {csv_filename}: reading {len(local_indices)} selected rows...")
        
        if not csv_file_path.exists():
            print(f"  - Warning: File does not exist, skipping")
            continue
        
        # Read selected rows (with date filtering if in limited mode)
        if limited_mode and start_date is not None and end_date is not None:
            sampled_rows = _read_selected_rows_from_file_with_date_filter(
                csv_file_path, local_indices.copy(), group_type, start_date, end_date
            )
        else:
            sampled_rows = _read_selected_rows_from_file(csv_file_path, local_indices.copy())
        
        if sampled_rows:
            # Save immediately after reading as JSON
            sampled_df = pd.DataFrame(sampled_rows)
            
            # Convert datetime columns to strings for JSON serialization
            for col in sampled_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sampled_df[col]):
                    sampled_df[col] = sampled_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Change file extension from .csv to .json
            json_filename = csv_filename.replace('.csv', '.json')
            output_file = output_dir / json_filename
            # Use standard json module to avoid escaping slashes in URLs
            # Replace NaN with None for JSON serialization
            data_dict = sampled_df.replace({pd.NA: None, pd.NaT: None}).to_dict('records')
            # Also replace any remaining NaN values
            data_dict = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data_dict]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
            print(f"  - Successfully read and saved {len(sampled_df)} rows to {json_filename}")
            total_saved += len(sampled_df)
            files_saved += 1
        else:
            print(f"  - No rows read")
    
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  - Total rows sampled: {total_saved}")
    print(f"  - Output files: {files_saved}")
    print(f"  - Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Randomly sample ground truth data from CSV files')
    parser.add_argument('--limited', action='store_true', 
                       help='Extract limited_years data instead of all_years data')
    
    args = parser.parse_args()
    
    _create_random_ground_truth_internal(GroupType.UR_PR, limited_mode=args.limited)
    _create_random_ground_truth_internal(GroupType.PR_UR, limited_mode=args.limited)
