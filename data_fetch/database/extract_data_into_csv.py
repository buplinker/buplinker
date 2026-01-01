"""
MySQLデータをCSVに出力するスクリプト

データベース内のすべてのデータを加工せずにCSVファイルとして出力する。
"""

from pathlib import Path
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data_fetch.database.tables import Repository
import data_fetch.database.get as db_get
import project_config as config

def extract_all_data_into_csv(repo: Repository) -> None:
    output_dir = Path(f"data_fetch/database/repositories_csv/{repo.owner}.{repo.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_repo = pd.DataFrame([{
        'id': repo.id,
        'owner': repo.owner,
        'name': repo.name,
        'github_url': repo.github_url,
        'google_play_store_app_id': repo.google_play_store_app_id,
        'google_play_store_url': repo.google_play_store_url,
        'category': repo.category,
    }])
    df_repo.to_csv(output_dir / "repository.csv", index=False)
    print(f"  - repository.csv: 1 record")
    
    releases = db_get.releases(repo.id)
    if releases:
        df_releases = pd.DataFrame([
            {
                'id': r.id,
                'repository_id': r.repository_id,
                'version': r.version,
                'title': r.title,
                'created_at': r.created_at,
                'released_at': r.released_at,
                'url': r.url,
                'author': r.author,
            }
            for r in releases if r.created_at <= config.CUT_OFF_DATE
        ])
        df_releases.to_csv(output_dir / "releases.csv", index=False)
        print(f"  - releases.csv: {len(df_releases)} records")
    
    pull_requests = db_get.pull_requests(repo.id)
    if pull_requests:
        df_pull_requests = pd.DataFrame([
            {
                'id': pr.id,
                'repository_id': pr.repository_id,
                'url': pr.url,
                'author': pr.author,
                'title': pr.title,
                'bodyText': pr.bodyText,
                'bodyHtml': pr.bodyHtml,
                'created_at': pr.created_at,
                'updated_at': pr.updated_at,
                'merged_at': pr.merged_at,
                'closed_at': pr.closed_at,
                'review_requested_at': pr.review_requested_at,
                'additions': pr.additions,
                'deletions': pr.deletions,
                'commits': pr.commits,
                'changed_files': pr.changed_files,
            }
            for pr in pull_requests if pr.merged_at is not None and pr.created_at <= config.CUT_OFF_DATE
        ])
        df_pull_requests.to_csv(output_dir / "pull_requests.csv", index=False)
        print(f"  - pull_requests.csv: {len(df_pull_requests)} records")

    pull_request_templates = db_get.pull_request_templates(repo.id)
    if pull_request_templates:
        df_pull_request_templates = pd.DataFrame([
            {
                'id': pt.id,
                'repository_id': pt.repository_id,
                'template': pt.template,
                'created_at': pt.created_at,
                'author': pt.author,
            }
            for pt in pull_request_templates
        ])
        df_pull_request_templates.to_csv(output_dir / "pull_request_templates.csv", index=False)
        print(f"  - pull_request_templates.csv: {len(df_pull_request_templates)} records")
    
    issues = db_get.issues(repo.id)
    if issues:
        df_issues = pd.DataFrame([
            {
                'id': i.id,
                'repository_id': i.repository_id,
                'url': i.url,
                'author': i.author,
                'title': i.title,
                'bodyText': i.bodyText,
                'bodyHtml': i.bodyHtml,
                'created_at': i.created_at,
                'closed': i.closed,
                'closed_at': i.closed_at,
            }
            for i in issues if i.created_at <= config.CUT_OFF_DATE
        ])
        df_issues.to_csv(output_dir / "issues.csv", index=False)
        print(f"  - issues.csv: {len(df_issues)} records")
    
    issue_templates = db_get.issue_templates(repo.id)
    if issue_templates:
        df_issue_templates = pd.DataFrame([
            {
                'id': it.id,
                'repository_id': it.repository_id,
                'template': it.template,
                'created_at': it.created_at,
                'author': it.author,
            }
            for it in issue_templates if it.created_at <= config.CUT_OFF_DATE
        ])
        df_issue_templates.to_csv(output_dir / "issue_templates.csv", index=False)
        print(f"  - issue_templates.csv: {len(df_issue_templates)} records")

    user_reviews = db_get.user_reviews(repo.id)
    if user_reviews:
        df_user_reviews = pd.DataFrame([
            {
                'id': ur.id,
                'repository_id': ur.repository_id,
                'review_version': ur.review_version,
                'app_version': ur.app_version,
                'content': ur.content,
                'user': ur.user,
                'created_at': ur.created_at,
                'star_score': ur.star_score,
                'thumbs_up_count': ur.thumbs_up_count,
                'reply': ur.reply,
                'replied_at': ur.replied_at,
            }
            for ur in user_reviews if ur.created_at <= config.CUT_OFF_DATE
        ])
        df_user_reviews.to_csv(output_dir / "user_reviews.csv", index=False)
        print(f"  - user_reviews.csv: {len(df_user_reviews)} records")

if __name__ == "__main__":
    from root_util import target_repos
    
    for repo in target_repos():
        print(f"{repo.id}: {repo.owner}.{repo.name}")
        # TODO: アプリのバージョンと直近のリリース日を入れる
        extract_all_data_into_csv(repo)
        print()