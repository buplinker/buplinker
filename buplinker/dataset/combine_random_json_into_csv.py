#!/usr/bin/env python3

import pandas as pd
import json
from pathlib import Path
import sys
import argparse

# Add parent directories to path to import GroupType
sys.path.append(str(Path(__file__).parent.parent.parent))
from root_util import GroupType


def _load_json_to_dataframe(json_file: Path) -> pd.DataFrame | None:
    """
    JSONファイルをDataFrameに読み込む
    
    Args:
        json_file: 読み込むJSONファイルのパス
    
    Returns:
        読み込んだDataFrame、エラーの場合はNone
    """
    # Try reading with pandas read_json first (most efficient)
    try:
        df = pd.read_json(json_file, orient='records')
        if not df.empty:
            return df
        return None
    except Exception:
        # Fallback to json.load + DataFrame conversion if pd.read_json fails
        pass
    
    # Fallback method: use standard json module
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and single dict formats
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return None
        
        if not df.empty:
            return df
        return None
    except Exception as e:
        print(f"    - Error loading {json_file.name}: {e}")
        return None


def combine_json_files(group_type: GroupType, base_dir: Path):
    """指定されたグループタイプのJSONファイルを1つのCSVにまとめる"""
    json_dir = base_dir / group_type.value
    
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist. Skipping...")
        return
    
    # Get all JSON files (exclude sample_sizes.json)
    json_files = [f for f in json_dir.glob("*.json") if f.name != "sample_sizes.json"]
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"\nProcessing {len(json_files)} JSON files from {group_type.value}/...")
    
    # Build DataFrame incrementally for better memory efficiency
    dataframes = []
    total_rows = 0
    
    for json_file in sorted(json_files):
        df_chunk = _load_json_to_dataframe(json_file)
        
        if df_chunk is not None:
            dataframes.append(df_chunk)
            total_rows += len(df_chunk)
            print(f"  - Loaded {json_file.name}: {len(df_chunk)} rows")
    
    if not dataframes:
        print(f"No data found in {group_type.value} JSON files")
        return
    
    # Concatenate all DataFrames efficiently
    # This is more memory efficient than building a large list first
    print(f"\nCombining {len(dataframes)} DataFrames...")
    df = pd.concat(dataframes, ignore_index=True)
    
    # Save to CSV
    output_file = base_dir / group_type.value / "random_input_pairs.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nSaved {len(df)} rows to {output_file.name}")
    print(f"  - Columns: {', '.join(df.columns.tolist())}")
    print(f"  - Shape: {df.shape}")
    
    # Filter rows where target (or label) == 1 and save as JSON
    if 'target' in df.columns:
        df_label1 = df[df['target'] == 1].copy()
        
        if len(df_label1) > 0:                        
            output_csv_file = base_dir / group_type.value / "random_label1_input_pairs.csv"
            df_label1.to_csv(output_csv_file, index=False, encoding='utf-8')
            
            print(f"  - Shape: {df_label1.shape}")
        else:
            print(f"\nNo rows found with target==1, skipping JSON output")
    else:
        print(f"\nWarning: Neither 'target' nor 'label' column found in data. Skipping label==1 JSON output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine JSON files into CSV files')
    parser.add_argument('--limited', action='store_true', default=False,
                        help='Use limited_years data instead of all_years data')
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "limited_random" if args.limited else Path(__file__).parent / "random"
    combine_json_files(GroupType.UR_PR, base_dir)
    combine_json_files(GroupType.PR_UR, base_dir)

