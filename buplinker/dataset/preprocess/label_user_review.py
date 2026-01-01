"""
User reviewsにARdocを使用してラベル付けを行い、データベースに保存するスクリプト
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

# ARdoc Javaプログラムのパス
ARDOC_DIR = ROOT_DIR / "ARdoc_API" / "ARdocExample"
ARDOC_CLASSPATH = f"bin:../ARdoc_API.jar:../lib/*"
ARDOC_CLASS = "runARdoc.ExampleOfUseARdoc"


def classify_text_with_ardoc(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    
    # Javaプログラムの実行
    # subprocessは自動的にエスケープしてくれるので、そのまま渡す
    cmd = [
        'java',
        '-cp', ARDOC_CLASSPATH,
        ARDOC_CLASS,
        text
    ]
    
    # ARdocディレクトリに移動して実行
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
        
        # 標準出力から結果を取得
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
    # user reviewsを取得
    _, df_user_review, _ = load_data(repo_id)
    
    if df_user_review.empty:
        print("No user reviews found.")
        return
    
    print(f"Found {len(df_user_review)} user reviews")
    
    # 既にintentionが設定されているレビューをスキップ
    if skip_labeled:
        df_user_review = df_user_review[df_user_review['intention'].isna()]
        print(f"After filtering (skipping already labeled): {len(df_user_review)} reviews to process")
    
    if df_user_review.empty:
        print("No reviews to process.")
        return
    
    # データベースセッションを作成
    Session = sessionmaker(engine)
    session = Session()
    
    try:
        updated_count = 0
        error_count = 0
        
        # 各レビューを処理
        for _, row in tqdm(df_user_review.iterrows(), total=len(df_user_review), desc="Labeling reviews"):
            review_id = str(row['id'])
            content = str(row['content']) if pd.notna(row['content']) else ""
            
            if not content or not content.strip():
                # 空のコンテンツの場合はNoneを設定
                categories = None
            else:
                # ARdocで分類
                categories = classify_text_with_ardoc(content)
            
            # データベースを更新
            try:
                stmt = (
                    update(UserReview)
                    .where(UserReview.id == review_id)
                    .values(intention=categories)
                )
                session.execute(stmt)
                updated_count += 1
                
                # 100件ごとにコミット（パフォーマンス向上のため）
                if updated_count % 100 == 0:
                    session.commit()
                    
            except Exception as e:
                print(f"Error updating review ID {review_id}: {e}")
                error_count += 1
                session.rollback()
        
        # 残りの変更をコミット
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

