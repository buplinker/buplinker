#!python3
from sqlalchemy import exc
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import sessionmaker

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data_fetch.database import get as data
from data_fetch.database.tables import UserReview, PullRequest, Release, Repository, PullRequestTemplate, Issue, IssueTemplate, engine

# Create session
Session = sessionmaker(engine)
session = Session()


def add_repository(repo: Repository):
    if not data.has_repository(repo):
        session.add(repo)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting Repository:", e)
            session.rollback()

def add_pull_request(repo_id: int, pull_request: PullRequest):
    if not data.has_pull_request(repo_id, pull_request.id):
        session.add(pull_request)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting PullRequest:", e)
            session.rollback()

def add_release(repo_id: int, release: Release):
    if not data.has_release(repo_id, release.id):
        session.add(release)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting Release:", e)
            session.rollback()


def add_pull_request_template(repo_id: int, template: PullRequestTemplate):
    if not data.has_pull_request_template(repo_id, template.id, template.file_path):
        session.add(template)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting PullRequestTemplate:", e)
            session.rollback()

def add_issue(repo_id: int, issue: Issue):
    if not data.has_issue(repo_id, issue.id):
        session.add(issue)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting Issue:", e)
            session.rollback()

def add_issue_template(repo_id: int, template: IssueTemplate):
    if not data.has_issue_template(repo_id, template.id, template.file_path):
        session.add(template)
        try:
            session.commit()
        except exc.SQLAlchemyError as e:
            print("Error inserting IssueTemplate:", e)
            session.rollback()

def add_user_reviews(user_reviews: list[UserReview]):
    try:
        user_review_list = []
        for ur in user_reviews:
            user_review_list.append(
                {
                    "id": ur.id,
                    "repository_id": ur.repository_id,
                    "review_version": ur.review_version,
                    "app_version": ur.app_version,
                    "content": ur.content,
                    "created_at": ur.created_at,
                    "star_score": ur.star_score,
                    "thumbs_up_count": ur.thumbs_up_count,
                }
            )

        objects_per_list = 1000
        list_of_lists = []
        for i in range(0, len(user_review_list), objects_per_list):
            sublist = user_review_list[i : i + objects_per_list]
            list_of_lists.append(sublist)

        conn = engine.connect()
        for i, user_review_list in enumerate(list_of_lists):
            print(i * objects_per_list)
            insert_stmt = insert(UserReview).values(user_review_list)
            on_duplicate_key_stmt = insert_stmt.on_duplicate_key_update(
                repository_id=insert_stmt.inserted.repository_id,
                review_version=insert_stmt.inserted.review_version,
                app_version=insert_stmt.inserted.app_version,
                content=insert_stmt.inserted.content,
                user=insert_stmt.inserted.user,
                created_at=insert_stmt.inserted.created_at,
                star_score=insert_stmt.inserted.star_score,
                thumbs_up_count=insert_stmt.inserted.thumbs_up_count,
            )

            conn.execute(on_duplicate_key_stmt)

    except exc.SQLAlchemyError as e:
        print(e)

def reset_database_connection():
    try:
        if hasattr(engine, 'dispose'):
            engine.dispose()
            print("Database connection reset")
    except Exception as e:
        print(f"Error resetting database connection: {e}")