from enum import Enum
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

LIMITED_YEARS = 3

class ContentType(Enum):
    UR = "ur"
    PR = "pr"
    ISSUE = "issue"

class GroupType(Enum):
    UR_PR = "ur_pr"
    PR_UR = "pr_ur"

# Cache for repositories CSV data
_repositories_df: Optional[pd.DataFrame] = None

def _load_repositories_csv() -> pd.DataFrame:
    """Load repositories.csv file and cache it."""
    global _repositories_df
    if _repositories_df is not None:
        return _repositories_df
    
    # Try to find repositories.csv in the parent directory
    current_file = Path(__file__)
    csv_path = current_file.parent / "repositories.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Repository CSV file not found: {csv_path}")
    
    # Read CSV without header (no column names)
    _repositories_df = pd.read_csv(csv_path, header=None)
    return _repositories_df

def get_first_release_date_from_repository(repo_id: int) -> Optional[Tuple[datetime, datetime]]:
    """Get first release date from repositories.csv file."""
    try:
        df = _load_repositories_csv()
        
        # Find repository by id (first column, index 0)
        repo_row = df[df.iloc[:, 0] == repo_id]
        if repo_row.empty:
            return None
        
        # released is in column 10 (index 10)
        released_str = repo_row.iloc[0, 10]
        if pd.isna(released_str) or str(released_str).strip() == '':
            return None
        
        # Parse date string (e.g., "Jul 30, 2023")
        start_date = pd.to_datetime(released_str)
        if pd.isna(start_date):
            return None
        
        # end_date is start_date + 3 years
        end_date = start_date + timedelta(days=365 * LIMITED_YEARS)
        
        return start_date, end_date
    except Exception as e:
        print(f"Error getting first_release_date from repository: {e}")
        return None