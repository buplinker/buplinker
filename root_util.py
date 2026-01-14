#!python3
import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import pandas as pd
from typing import Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta

import data_fetch.database.get as data
from data_fetch.database.tables import Repository

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import project_config as config

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LIMITED_YEARS = 3

REPOSITOREIS = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_fetch/database/tables/repositories.csv"))

class ContentType(Enum):
    UR = "ur"
    PR = "pr"
    ISSUE = "issue"

class GroupType(Enum):
    UR_PR = "ur_pr"
    PR_UR = "pr_ur"

class CategoryType(Enum):
    Hedonic = "Hedonic"
    Utilitarian = "Utilitarian"

def target_repos() -> list[Repository]:
    a = input(f"Enter a if all {len(data.repositories())} repos should be targets.")
    if a == "a":
        return data.repositories()
    else:
        repositories = []
        print("Enter y if a repo should be the target.")
        for repo in data.repositories():
            y = input(f"{repo.id}: {repo.owner}.{repo.name}?")
            if y == "y":
                repositories.append(repo)
        return repositories

def target_csv_repos() -> list[Repository]:
    repositories: list[Repository] = []
    
    for _, row in REPOSITOREIS.copy().iterrows():
        repo = Repository(
            id=int(row['id']) if pd.notna(row['id']) else None,
            owner=str(row['owner']) if pd.notna(row['owner']) else None,
            name=str(row['name']) if pd.notna(row['name']) else None,
            github_url=str(row['github_url']) if pd.notna(row['github_url']) else None,
            google_play_store_app_id=str(row['google_play_store_app_id']) if pd.notna(row['google_play_store_app_id']) else None,
            google_play_store_url=str(row['google_play_store_url']) if pd.notna(row['google_play_store_url']) else None,
            released=str(row['released']) if pd.notna(row['released']) else None,
            category=str(row['category']) if pd.notna(row['category']) else None,
        )
        repositories.append(repo)
    
    return repositories

def target_csv_repo_by_id(repo_id: int) -> Repository:
    repo = REPOSITOREIS.copy()[REPOSITOREIS['id'] == repo_id].iloc[0]
    return Repository(
        id=repo['id'],
        owner=repo['owner'],
        name=repo['name'],
        github_url=repo['github_url'],
        google_play_store_app_id=repo['google_play_store_app_id'],
        google_play_store_url=repo['google_play_store_url'],
        released=repo['released'],
        category=repo['category'],
    )

def load_data(repo_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    releases = data.releases(repo_id)
    df_release = pd.DataFrame([
        {
            'id': r.id,
            'repository_id': r.repository_id,
            'version': r.version,
            'created_at': r.created_at,
            'released_at': r.released_at,
        }
        for r in releases if r.created_at <= config.CUT_OFF_DATE
    ])
    
    user_reviews = data.user_reviews(repo_id)
    df_user_review = pd.DataFrame([
        {
            'id': ur.id,
            'repository_id': ur.repository_id,
            'content': ur.content,
            'created_at': ur.created_at,
            'review_version': ur.review_version,
            'intention': ur.intention,
        }
        for ur in user_reviews if ur.created_at <= config.CUT_OFF_DATE
    ])

    pull_requests = data.pull_requests(repo_id)
    df_pull_request = pd.DataFrame([
        {
            'id': pr.id,
            'repository_id': pr.repository_id,
            'title': pr.title,
            'bodyText': pr.bodyText,
            'bodyHtml': pr.bodyHtml,
            'created_at': pr.created_at,
            'merged_at': pr.merged_at,
            'url': pr.url,
            'author': pr.author,
        }
        for pr in pull_requests if pr.merged_at is not None and pr.created_at <= config.CUT_OFF_DATE
    ])
    
    # Convert date columns to datetime type
    if not df_release.empty:
        df_release['created_at'] = pd.to_datetime(df_release['created_at'])
        df_release['released_at'] = pd.to_datetime(df_release['released_at'])
        df_release['release_date'] = df_release['released_at'].fillna(df_release['created_at'])
    
    if not df_user_review.empty:
        df_user_review['created_at'] = pd.to_datetime(df_user_review['created_at'])
    
    if not df_pull_request.empty:
        df_pull_request['created_at'] = pd.to_datetime(df_pull_request['created_at'])
        df_pull_request['merged_at'] = pd.to_datetime(df_pull_request['merged_at'])
            
    return df_release, df_user_review, df_pull_request


def filter_dataframe_by_date_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, date_column: str = 'created_at') -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df
    
    # Convert date column to datetime type (errors='coerce' sets unparseable values to NaN and suppresses warnings)
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Filter by date range (NaN values are excluded)
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask].copy()


def get_first_release_date_from_repository(repo_id: int) -> Tuple[Optional[datetime], Optional[datetime]]:
    try:
        repo = target_csv_repo_by_id(repo_id)
        if not repo or not repo.released or repo.released.strip() == '':
            return None, None
        
        # Convert released column to date
        start_date = pd.to_datetime(repo.released)
        if pd.isna(start_date):
            return None, None
        
        # end_date is 3 years after start_date
        end_date = start_date + timedelta(days=365 * LIMITED_YEARS)
        
        return start_date, end_date
    except Exception as e:
        print(f"Error getting first_release_date from repository: {e}")
        return None, None