"""
Script to label repository with category using genre and save to database
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from root_util import target_repos, CategoryType
from data_fetch.database.tables import Repository, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import update


SANER_CATEGORY_MAP = {
    "Social": CategoryType.Hedonic,
    "Photography": CategoryType.Hedonic,
    "Communication": CategoryType.Hedonic,
    "Role Playing": CategoryType.Hedonic,
    "Action": CategoryType.Hedonic,
    "Video Players & Editors": CategoryType.Hedonic,
    "Travel & Local": CategoryType.Hedonic,
    "Casual": CategoryType.Hedonic,
    "Maps & Navigation": CategoryType.Utilitarian,
    "Business": CategoryType.Utilitarian,
    "Productivity": CategoryType.Utilitarian,
    "Tools": CategoryType.Utilitarian,
    "Education": CategoryType.Utilitarian,
    "Finance": CategoryType.Utilitarian,
}

CATEGORY_MAP = {
    "Family": CategoryType.Hedonic,
    "Gaming": CategoryType.Hedonic,
    "Tools": CategoryType.Utilitarian,
    "Medical": CategoryType.Utilitarian,
    "Lyfestyle": CategoryType.Hedonic,
    "Personalization": CategoryType.Utilitarian,
    "Finance": CategoryType.Utilitarian,
    "Sports": CategoryType.Hedonic,
    "Business": CategoryType.Utilitarian,
    "Photography": CategoryType.Hedonic,
    "Productivity": CategoryType.Utilitarian,
    "Health and fitness": CategoryType.Utilitarian,
    "Communication": CategoryType.Hedonic,
    "Shopping": CategoryType.Hedonic,
    "Dating": CategoryType.Hedonic,
    "Social": CategoryType.Hedonic,
    "News and magazine": CategoryType.Utilitarian,
    "Travel and local": CategoryType.Hedonic,
    "Books and refernence": CategoryType.Utilitarian,
    "Video player": CategoryType.Hedonic,
    "Education": CategoryType.Utilitarian,
    "Maps and navigation": CategoryType.Utilitarian,
    "Entertainment": CategoryType.Hedonic,
    "Food and drink": CategoryType.Hedonic,
    "Auto and vehicles": CategoryType.Utilitarian,
    "Libraries and demo": CategoryType.Utilitarian,
    "Art and design": CategoryType.Hedonic,
    "House and home": CategoryType.Utilitarian,
    "Weather": CategoryType.Utilitarian,
    "Comics": CategoryType.Hedonic,
    "Parenting": CategoryType.Utilitarian,
    "Beauty": CategoryType.Hedonic,
    "Events": CategoryType.Utilitarian,
}


def label_repository_with_category(repo_id: int) -> None:
    # Create database session
    Session = sessionmaker(engine)
    session = Session()
    
    try:
        # Get repository
        repo = session.query(Repository).filter(Repository.id == repo_id).first()
        category = SANER_CATEGORY_MAP[repo.genre]

        try:
            stmt = (
                update(Repository)
                .where(Repository.id == repo_id)
                .values(category=category)
            )
            session.execute(stmt)
            session.commit()
            
            print(f"Updated repository {repo.owner}.{repo.name}: genre '{repo.genre}' -> category '{category}'")
            
        except Exception as e:
            print(f"Error updating repository ID {repo_id}: {e}")
            session.rollback()
            raise
            
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


if __name__ == "__main__":        
    for repo in target_repos():
        print(f"Labeling repository {repo.id}: {repo.owner}.{repo.name}...")
        label_repository_with_category(repo.id)