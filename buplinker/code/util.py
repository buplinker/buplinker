import re
import math
from typing import List, Dict
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from root_util import GroupType

def get_group_key(group_type=GroupType.UR_PR.value):
    """Return grouping key according to group_type"""
    return 'ur_id' if group_type == GroupType.UR_PR.value else 'pr_id'

def recall(ground_truth_data_frame, data_frame):
    ground_truth_df = ground_truth_data_frame.loc[ground_truth_data_frame['target']==1] if 'target' in ground_truth_data_frame.columns else ground_truth_data_frame
    df = data_frame.loc[data_frame['label']==1] if 'label' in data_frame.columns else data_frame
    print(f"Recall: {len(df)} / {len(ground_truth_df)} = {round(len(df) / len(ground_truth_df) if len(ground_truth_df) > 0 else 0, 4)}")
    return round(len(df) / len(ground_truth_df) if len(ground_truth_df) > 0 else 0, 4)

def precision(data_frame):
    df = data_frame.loc[data_frame['label']==1] if 'label' in data_frame.columns else data_frame
    print(f"Precision: {len(df)} / {len(data_frame)} = {round(len(df) / len(data_frame) if len(data_frame) > 0 else 0, 4)}")
    return round(len(df) / len(data_frame) if len(data_frame) > 0 else 0, 4)

def F1_score(precision, recall):
    print(f"F1_score: 2 * {precision} * {recall} / ({precision} + {recall}) = {2 * precision * recall / (precision + recall) if precision + recall > 0 else 0}")
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

def NDCG_at_K(data_frame, k=1, group_type=GroupType.UR_PR.value):
    group_by = get_group_key(group_type)
    group_tops = data_frame.groupby(group_by)
    cnt = 0
    dcg_sum = 0
    for group_id, group in group_tops:
        rank = 0
        for index, row in group.head(k).iterrows():
            rank += 1
            # Check if it's the correct PR (may need adjustment based on actual data)
            if hasattr(row, 'label') and row['label'] == 1:
                dcg_sum += math.log(2)/math.log(rank+2) 
                break 
        cnt += 1
    print(f"NDCG@K: {dcg_sum} / {cnt} = {round(dcg_sum / cnt if cnt > 0 else 0, 4)}")
    return round(dcg_sum / cnt if cnt > 0 else 0, 4)

def recall_at_K(data_frame, k=1, group_type=GroupType.UR_PR.value):
    group_by = get_group_key(group_type)
    group_tops = data_frame.groupby(group_by)
    df = data_frame.loc[data_frame['label']==1] if 'label' in data_frame.columns else data_frame
    cnt = 0
    recall = 0.0
    for group_id, group in group_tops:
        hits = 0
        tu = df.loc[df[group_by]==group_id] if 'label' in data_frame.columns else group
        for index, row in group.head(k).iterrows():
            hits += 1 if hasattr(row, 'label') and row['label'] == 1 else 0      
        recall += round(hits / len(tu) if len(tu) > 0 else 0, 4)
        cnt +=1
    print(f"Recall@K: {recall} / {cnt} = {round(recall / cnt if cnt > 0 else 0, 4)}")
    return recall/cnt

def precision_at_K(data_frame, k=1, group_type=GroupType.UR_PR.value):
    group_by = get_group_key(group_type)
    group_tops = data_frame.groupby(group_by)
    cnt = 0
    hits = 0
    for group_id, group in group_tops:
        for index, row in group.head(k).iterrows():
            hits += 1 if hasattr(row, 'label') and row['label'] == 1 else 0      
        cnt += k
    print(f"Precision@K: {hits} / {cnt} = {round(hits / cnt if cnt > 0 else 0, 4)}")
    return round(hits / cnt if cnt > 0 else 0, 4)

def Hit_at_K(data_frame, k=1, group_type=GroupType.UR_PR.value):
    group_by = get_group_key(group_type)
    group_tops = data_frame.groupby(group_by)
    cnt = 0
    hits = 0
    for group_id, group in group_tops:
        for index, row in group.head(k).iterrows():
            if hasattr(row, 'label') and row['label'] == 1:
                hits += 1
                break
        cnt += 1
    print(f"Hit@K: {hits} / {cnt} = {round(hits / cnt if cnt > 0 else 0, 4)}")
    return round(hits / cnt if cnt > 0 else 0, 4)

def MRR(data_frame, group_type=GroupType.UR_PR.value):
    group_by = get_group_key(group_type)
    group_tops = data_frame.groupby(group_by)
    mrr_sum = 0
    for group_id, group in group_tops:
        rank = 0
        for i, (index, row) in enumerate(group.iterrows()):
            rank += 1
            if hasattr(row, 'label') and row['label'] == 1:
                mrr_sum += 1.0 / rank
                break
    print(f"MRR: {mrr_sum} / {len(group_tops)} = {round(mrr_sum / len(group_tops) if len(group_tops) > 0 else 0, 4)}")
    return mrr_sum / len(group_tops)

def clean_text(text):
    """Replace multiple spaces, tabs, or newlines with a single space and trim."""
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def results_to_df_db(res: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(res)
    
    group_sort = df.groupby("s_id", group_keys=False).apply(
        lambda x: x.sort_values("pred", ascending=False)
    ).reset_index(drop=True)
    
    return group_sort

def results_to_df_llm(res: List[Dict], group_type: str = GroupType.UR_PR.value) -> pd.DataFrame:
    df = pd.DataFrame(res)
    
    if len(res) > 0:
        group_by = get_group_key(group_type)
        group_sort = df.groupby(group_by, group_keys=False).apply(
            lambda x: x.sort_values("rank", ascending=True)
        ).reset_index(drop=True)
        return group_sort
    else:
        return pd.DataFrame()

def ground_truth_to_df_llm(res: List[Dict], group_type: str = GroupType.UR_PR.value) -> pd.DataFrame:
    df = pd.DataFrame(res)
    
    group_by = get_group_key(group_type)
    group_sort = df.groupby(group_by, group_keys=False).apply(
        lambda x: x.sort_values("rank", ascending=True)
    ).reset_index(drop=True).drop(columns=['rank'])
    
    return group_sort