#!python3
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Column, UniqueConstraint
from sqlalchemy.types import Boolean, DateTime, Float, Integer, String, Text

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import project_config as config

engine = create_engine(
    f"mysql+mysqlconnector://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}@{config.MYSQL_HOST}/{config.MYSQL_DATABASE}"
)
Base = declarative_base()


class Repository(Base):
    __tablename__ = "repositories"
    __table_args__ = (UniqueConstraint("owner", "name", name="uq_owner_name"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)

    owner = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    github_url = Column(String(255), nullable=False)
    google_play_store_app_id = Column(String(100), nullable=False)
    google_play_store_url = Column(String(255), nullable=False)
    genre = Column(String(100))
    genre_id = Column(String(100))
    score = Column(Float)
    version = Column(String(100))
    released = Column(String(100))
    installs = Column(String(100))
    min_installs = Column(Integer)
    real_installs = Column(Integer)
    ratings = Column(Integer)
    reviews = Column(Integer)
    free = Column(Boolean)
    updated = Column(Integer)
    category = Column(String(100))

    user_reviews = relationship("UserReview", back_populates="repository")
    pull_requests = relationship("PullRequest", back_populates="repository")
    issues = relationship("Issue", back_populates="repository")
    releases = relationship("Release", back_populates="repository")
    pull_request_templates = relationship("PullRequestTemplate", back_populates="repository")
    issue_templates = relationship("IssueTemplate", back_populates="repository")


class Release(Base):
    __tablename__ = "releases"
    __table_args__ = (
        UniqueConstraint("repository_id", "version", name="uq_repository_id_version"),
        {"mysql_charset": "utf8mb4"},
    )
    id = Column(String(100), primary_key=True)

    repository_id = Column(Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(String(100), nullable=False, index=True)
    title = Column(String(255), index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    released_at = Column(DateTime, index=True)
    url = Column(String(255), nullable=False)
    author = Column(String(100))

    repository = relationship("Repository", back_populates="releases")


class PullRequest(Base):
    __tablename__ = "pull_requests"
    __table_args__ = ({"mysql_charset": "utf8mb4"})
    id = Column(String(100), primary_key=True)

    repository_id = Column(
        Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True
    )
    url = Column(String(255), nullable=False)
    author = Column(String(100))
    title = Column(Text, nullable=False)
    bodyText = Column(Text, nullable=False)
    bodyHtml = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    merged_at = Column(DateTime)
    closed_at = Column(DateTime)
    review_requested_at = Column(DateTime)
    additions = Column(Integer, nullable=False)
    deletions = Column(Integer, nullable=False)
    commits = Column(Integer, nullable=False)
    changed_files = Column(Integer, nullable=False)

    repository = relationship("Repository", back_populates="pull_requests")

class PullRequestTemplate(Base):
    __tablename__ = "pull_request_templates"
    __table_args__ = (UniqueConstraint("id", "file_path", name="uq_id_file_path"), {"mysql_charset": "utf8mb4"})
    id = Column(String(100), primary_key=True)
    file_path = Column(String(255), primary_key=True)

    repository_id = Column(
        Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True
    )
    template = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    author = Column(String(100), nullable=False)

    repository = relationship("Repository", back_populates="pull_request_templates")

class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = ({"mysql_charset": "utf8mb4"})
    id = Column(String(100), primary_key=True)

    repository_id = Column(
        Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True
    )
    url = Column(String(255), nullable=False)
    author = Column(String(100))
    title = Column(Text, nullable=False)
    bodyText = Column(Text, nullable=False)
    bodyHtml = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    closed = Column(Integer, nullable=False)
    closed_at = Column(DateTime)

    repository = relationship("Repository", back_populates="issues")

class IssueTemplate(Base):
    __tablename__ = "issue_templates"
    __table_args__ = (UniqueConstraint("id", "file_path", name="uq_id_file_path"), {"mysql_charset": "utf8mb4"})
    id = Column(String(100), primary_key=True)
    file_path = Column(String(255), primary_key=True)

    repository_id = Column(
        Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True
    )
    template = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    author = Column(String(100), nullable=False)

    repository = relationship("Repository", back_populates="issue_templates")

class UserReview(Base):
    __tablename__ = "user_reviews"
    __table_args__ = ( {"mysql_charset": "utf8mb4"})
    id = Column(String(100), primary_key=True)

    repository_id = Column(
        Integer, ForeignKey("repositories.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True
    )
    review_version = Column(String(100), nullable=False, index=True)
    app_version = Column(String(100), nullable=False, index=True)
    content = Column(Text, nullable=False)
    user = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
    star_score = Column(Float)
    thumbs_up_count = Column(Integer)
    intention = Column(String(100))

    repository = relationship("Repository", back_populates="user_reviews")


if __name__ == "__main__":
    # Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
