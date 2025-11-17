#!python3
import time
import requests

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import project_config as config
import database.set as data_setter
from database.tables import Repository, PullRequest, Release, Issue
from root_util import target_repos
from template_fetcher import fetch_pull_request_templates, fetch_issue_templates


def fetch_github_data(
    repository: Repository,
    element_list_key: str,
    element_limit_per_request: int = 100,
    request_limit: int = 1000,
) -> list:
    fetched_data = []
    cursor = None
    request_headers = {"Authorization": f"bearer {config.GITHUB_AUTH_TOKEN}"}

    with open(f"data_fetch/query_templates/{element_list_key}.graphql", "r", encoding="utf-8") as f:
        query_text = f.read()

    for _ in range(request_limit):
        variables = {
            "owner": repository.owner,
            "name": repository.name,
            "first": element_limit_per_request,
            "after": cursor,
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.github.com/graphql",
                    headers=request_headers,
                    json={"query": query_text, "variables": variables},
                    timeout=60,
                )
                response.raise_for_status()
                response_data = response.json()
                break
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                wait = 2**attempt
                print(f"Connection error: {e}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            except Exception as e:
                print(f"Request failed: {e}", file=sys.stderr)
                print(getattr(response, "text", "No response text"), file=sys.stderr)
        else:
            print("Max retries reached, aborting", file=sys.stderr)
        element_list = response_data["data"]["repository"][element_list_key]
        page_info = element_list["pageInfo"]
        fetched_data.extend(element_list["nodes"])
        cursor = page_info["endCursor"]

        if not page_info["hasNextPage"]:
            break

    return fetched_data


def fetch_pull_requests(repo: Repository, element_limit_per_request: int = 50):
    start = time.time()
    pull_requests = fetch_github_data(repo, "pullRequests", element_limit_per_request)
    print(f"{len(pull_requests)} pull requests fetched.")
    print("Inserting pull requests......")
    for i, pull_request in enumerate(pull_requests):
        if i % 1000 == 0:
            print(f"{i}/{len(pull_requests)}")
            
        data_setter.add_pull_request(
            repo.id,
            PullRequest(
                id=pull_request["id"],
                repository_id=repo.id,
                url=pull_request["url"],
                author=pull_request["author"]["login"] if pull_request["author"] else None,
                title=pull_request["title"],
                bodyText=pull_request["bodyText"],
                bodyHtml=pull_request["bodyHTML"],
                created_at=pull_request["createdAt"],
                updated_at=pull_request["updatedAt"],
                merged_at=pull_request["mergedAt"],
                closed_at=pull_request["closedAt"],
                review_requested_at=pull_request["timelineItems"]["nodes"][0]["createdAt"] if pull_request["timelineItems"]["nodes"] else None,
                additions=pull_request["additions"],
                deletions=pull_request["deletions"],
                commits=pull_request["commits"]["totalCount"],
                changed_files=pull_request["changedFiles"],
            ),
        )
    print(f"{len(pull_requests)} pull requests inserted in {((time.time() - start) / 60):.2f} minutes!")


def fetch_releases(repo: Repository, element_limit_per_request: int = 50):
    start = time.time()
    releases = fetch_github_data(repo, "releases", element_limit_per_request)
    print(f"{len(releases)} releases fetched.")
    print("Inserting releases......")
    for i, release in enumerate(releases):
        if i % 100 == 0:
            print(f"{i}/{len(releases)}")
        data_setter.add_release(
            repo.id,
            Release(
                id=release["id"],
                repository_id=repo.id,
                version=release["tag"]["name"],
                title=release["name"],
                created_at=release["createdAt"],
                released_at=release["publishedAt"],
                url=release["url"],
                author=release["author"]["login"] if release["author"] else None,
            )
        )
    print(f"{len(releases)} releases inserted in {((time.time() - start) / 60):.2f} minutes!")

def fetch_issues(repo: Repository, element_limit_per_request: int = 50):
    start = time.time()
    issues = fetch_github_data(repo, "issues", element_limit_per_request)
    print(f"{len(issues)} issues fetched.")
    print("Inserting issues......")
    for i, issue in enumerate(issues):
        if i % 100 == 0:
            print(f"{i}/{len(issues)}")
        
        data_setter.add_issue(
            repo.id, 
            Issue(
                id=issue["id"],
                repository_id=repo.id,
                url=issue["url"],
                author=issue["author"]["login"] if issue["author"] else None,
                title=issue["title"],
                bodyText=issue["bodyText"],
                bodyHtml=issue["bodyHTML"],
                created_at=issue["createdAt"],
                closed=issue["closed"],
                closed_at=issue["closedAt"],
            )
        )
    print(f"{len(issues)} issues inserted in {((time.time() - start) / 60):.2f} minutes!")

if __name__ == "__main__":
    for repo in target_repos():
        print(f"{repo.id}: {repo.owner}.{repo.name}")
        
        print("Fetching pull requests......")
        fetch_pull_requests(repo)
        
        print("Fetching releases......")
        fetch_releases(repo)
        
        print("Fetching pull request templates......")
        fetch_pull_request_templates(repo)

        print("Fetching issues......")
        fetch_issues(repo)

        print("Fetching issue templates......")
        fetch_issue_templates(repo)

    print("Done!")
