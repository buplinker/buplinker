"""
Script to label user reviews with ARdoc and save to database
"""
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from root_util import target_repos, load_data
from data_fetch.database.tables import UserReview, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import update
from tqdm import tqdm

# Path to ARdoc Java program
ARDOC_DIR = ROOT_DIR / "ARdoc_API" / "ARdocExample"
ARDOC_CLASSPATH = f"bin:../ARdoc_API.jar:../lib/*"
ARDOC_CLASS = "runARdoc.ExampleOfUseARdoc"


def classify_text_with_ardoc(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    
    # Execute Java program
    # subprocess automatically escapes, so pass directly
    cmd = [
        'java',
        '-cp', ARDOC_CLASSPATH,
        ARDOC_CLASS,
        text
    ]
    
    # Change to ARdoc directory and execute
    original_dir = os.getcwd()
    try:
        os.chdir(ARDOC_DIR)
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            print(f"Error: ARdoc classification failed: {result.stderr}")
            return None
        
        # Get result from stdout
        output = result.stdout.strip()
        if not output:
            return None
        
        return str(list(set(c.strip() for c in output.split(',') if c.strip())))
        
    except Exception as e:
        print(f"Error running ARdoc: {e}")
        return None
    finally:
        os.chdir(original_dir)


def label_user_reviews_with_ardoc(repo_id: int, skip_labeled: bool = True) -> None:    
    # Get user reviews
    _, df_user_review, _ = load_data(repo_id)
    
    if df_user_review.empty:
        print("No user reviews found.")
        return
    
    print(f"Found {len(df_user_review)} user reviews")
    
    # Skip reviews that already have intention set
    if skip_labeled:
        df_user_review = df_user_review[df_user_review['intention'].isna()]
        print(f"After filtering (skipping already labeled): {len(df_user_review)} reviews to process")
    
    if df_user_review.empty:
        print("No reviews to process.")
        return
    
    # Create database session
    Session = sessionmaker(engine)
    session = Session()
    
    try:
        updated_count = 0
        error_count = 0
        
        # Process each review
        for _, row in tqdm(df_user_review.iterrows(), total=len(df_user_review), desc="Labeling reviews"):
            review_id = str(row['id'])
            content = str(row['content']) if pd.notna(row['content']) else ""
            
            if not content or not content.strip():
                # Set None for empty content
                categories = None
            else:
                # Classify with ARdoc
                categories = classify_text_with_ardoc(content)
            
            # Update database
            try:
                stmt = (
                    update(UserReview)
                    .where(UserReview.id == review_id)
                    .values(intention=categories)
                )
                session.execute(stmt)
                updated_count += 1
                
                # Commit every 100 items (for performance improvement)
                if updated_count % 100 == 0:
                    session.commit()
                    
            except Exception as e:
                print(f"Error updating review ID {review_id}: {e}")
                error_count += 1
                session.rollback()
        
        # Commit remaining changes
        session.commit()
        
        print(f"\nCompleted!")
        print(f"  Updated: {updated_count} reviews")
        if error_count > 0:
            print(f"  Errors: {error_count} reviews")
            
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label user reviews with ARdoc')
    parser.add_argument('--skip-labeled', action='store_true', 
                       help='Skip labeled user reviews')
    args = parser.parse_args()
    print("skip_labeled: ", args.skip_labeled)

    for repo in target_repos():
        print(f"Labeling user reviews for repository {repo.id}: {repo.owner}.{repo.name}...")
        label_user_reviews_with_ardoc(repo.id, args.skip_labeled)

