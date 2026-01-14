#!python3

from sqlalchemy.orm import Query, sessionmaker

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data_fetch.database.tables import UserReview, PullRequest, Release, Repository, PullRequestTemplate, Issue, IssueTemplate, engine

# Create session
Session = sessionmaker(engine)
session = Session()

def reset_session():
    """Reset the database session. Call this in child processes to create a new connection.
    
    In multiprocessing, database connections from parent process are not available 
    in child processes. This function creates a new session with a fresh connection.
    """
    global session
    # Invalidate the session first to avoid rollback attempts on invalid connections
    try:
        session.invalidate()
    except Exception:
        pass
    
    # Close existing session, ignoring any errors from invalid connections
    try:
        session.close()
    except Exception:
        # Ignore errors when closing invalid connections in child processes
        pass
    
    # Dispose of the engine's connection pool to ensure fresh connections
    # This is necessary because the connection pool from parent process is invalid in child processes
    try:
        engine.dispose()
    except Exception:
        # Ignore errors if engine disposal fails
        pass
    
    # Create a new session with a fresh connection from the engine
    session = Session()


def repositories() -> list[Repository]:
    return session.query(Repository).all()

def has_repository(repo: Repository) -> bool:
    return session.query(Repository).filter(
        Repository.owner == repo.owner, Repository.name == repo.name
    ).count() > 0

def repository_by_id(repo_id: int) -> Repository:
    return session.query(Repository).filter(Repository.id == repo_id).first()


def _user_review_query(repo_id: int) -> Query:
    return session.query(UserReview).filter(UserReview.repository_id == repo_id)

def user_reviews(repo_id: int) -> list[UserReview]:
    return _user_review_query(repo_id).order_by(UserReview.created_at).all()

def has_user_reviews(repo_id: int) -> bool:
    return _user_review_query(repo_id).count() > 0


def _release_query(repo_id: int) -> Query:
    return session.query(Release).filter(Release.repository_id == repo_id)

# TODO: released_at can be null according to the github api description
def releases(repo_id: int) -> list[Release]:
    return _release_query(repo_id).order_by(Release.released_at).all()

def release(repo_id: int, version: str) -> Release:
    return _release_query(repo_id).filter(Release.version == version).first()

def has_release(repo_id: int, release_id: str) -> bool:
    return _release_query(repo_id).filter(Release.id == release_id).count() > 0

def has_releases(repo_id: int) -> bool:
    return _release_query(repo_id).count() > 0


def _pull_request_query(repo_id: int) -> Query:
    return session.query(PullRequest).filter(PullRequest.repository_id == repo_id)

def pull_requests(repo_id: int) -> list[PullRequest]:
    return _pull_request_query(repo_id).order_by(PullRequest.merged_at).all()

def has_pull_requests(repo_id: int) -> bool:
    return _pull_request_query(repo_id).count() > 0

def has_pull_request(repo_id: int, pull_request_id: str) -> bool:
    return _pull_request_query(repo_id).filter(PullRequest.id == pull_request_id).count() > 0

def pull_request_by_number(repo_id: int, pr_number: int) -> PullRequest:
    pull_requests = _pull_request_query(repo_id).all()
    for pr in pull_requests:
        if pr.url.endswith(f"/pull/{pr_number}"):
            return pr
    return None


def _pull_request_template_query(repo_id: int) -> Query:
    return session.query(PullRequestTemplate).filter(PullRequestTemplate.repository_id == repo_id)

def pull_request_templates(repo_id: int) -> list[PullRequestTemplate]:
    return _pull_request_template_query(repo_id).order_by(PullRequestTemplate.created_at).all()

def has_pull_request_templates(repo_id: int) -> bool:
    return _pull_request_template_query(repo_id).count() > 0

def has_pull_request_template(repo_id: int, template_id: str, file_path: str) -> bool:
    return (
        _pull_request_template_query(repo_id)
        .filter(PullRequestTemplate.id == template_id, PullRequestTemplate.file_path == file_path)
        .count()
        > 0
    )


def _issue_query(repo_id: int) -> Query:
    return session.query(Issue).filter(Issue.repository_id == repo_id)

def issues(repo_id: int) -> list[Issue]:
    return _issue_query(repo_id).order_by(Issue.created_at).all()

def has_issues(repo_id: int) -> bool:
    return _issue_query(repo_id).count() > 0

def has_issue(repo_id: int, issue_id: str) -> bool:
    return _issue_query(repo_id).filter(Issue.id == issue_id).count() > 0

def issue_by_number(repo_id: int, issue_number: int) -> Issue:
    issues = _issue_query(repo_id).all()
    for issue in issues:
        if issue.url.endswith(f"/issues/{issue_number}"):
            return issue
    return None


def _issue_template_query(repo_id: int) -> Query:
    return session.query(IssueTemplate).filter(IssueTemplate.repository_id == repo_id)

def issue_templates(repo_id: int) -> list[IssueTemplate]:
    return _issue_template_query(repo_id).order_by(IssueTemplate.created_at).all()

def has_issue_templates(repo_id: int) -> bool:
    return _issue_template_query(repo_id).count() > 0

def has_issue_template(repo_id: int, template_id: str, file_path: str) -> bool:
    return (
        _issue_template_query(repo_id)
        .filter(IssueTemplate.id == template_id, IssueTemplate.file_path == file_path)
        .count()
        > 0
    )