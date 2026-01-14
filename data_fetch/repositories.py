#!python3
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_fetch.database.tables import Repository
from data_fetch.database import set as db_set
import root_util


def load_repositories_from_csv():
    """
    Load repository data from CSV file and add to database
    Uses REPOSITOREIS DataFrame from root_util.py
    """
    repositories_df = root_util.REPOSITOREIS.copy()
    
    for _, row in repositories_df.iterrows():
        # Ignore CSV id (use database auto-increment)
        # Check required fields
        if pd.isna(row.get('owner')) or pd.isna(row.get('name')):
            print(f"Skipping row: missing owner or name")
            continue
        
        # Create Repository object
        repo = Repository(
            owner=str(row['owner']) if pd.notna(row['owner']) else None,
            name=str(row['name']) if pd.notna(row['name']) else None,
            github_url=str(row['github_url']) if pd.notna(row['github_url']) else None,
            google_play_store_app_id=str(row['google_play_store_app_id']) if pd.notna(row['google_play_store_app_id']) else None,
            google_play_store_url=str(row['google_play_store_url']) if pd.notna(row['google_play_store_url']) else None,
            genre=str(row['genre']) if pd.notna(row['genre']) else None,
            genre_id=str(row['genre_id']) if pd.notna(row['genre_id']) else None,
            score=float(row['score']) if pd.notna(row['score']) else None,
            version=str(row['version']) if pd.notna(row['version']) else None,
            released=str(row['released']) if pd.notna(row['released']) else None,
            installs=str(row['installs']) if pd.notna(row['installs']) else None,
            min_installs=int(row['min_installs']) if pd.notna(row['min_installs']) else None,
            real_installs=int(row['real_installs']) if pd.notna(row['real_installs']) else None,
            ratings=int(row['ratings']) if pd.notna(row['ratings']) else None,
            reviews=int(row['reviews']) if pd.notna(row['reviews']) else None,
            free=bool(int(row['free'])) if pd.notna(row['free']) else None,
            updated=int(row['updated']) if pd.notna(row['updated']) else None,
            category=str(row['category']) if pd.notna(row['category']) else None,
        )
        
        # Add to database
        db_set.add_repository(repo)
        print(f"Added repository: {repo.owner}/{repo.name}")


if __name__ == "__main__":
    load_repositories_from_csv()
