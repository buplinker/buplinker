#!python3
from time import sleep
import time
import google_play_scraper.features.reviews as gps_reviews
from google_play_scraper import Sort, app
from pathlib import Path
from requests import Timeout

import sys
sys.path.append(str(Path(__file__).parent.parent))

import database.set as data_setter
from database.tables import UserReview, Repository
from root_util import target_repos


def google_play_scraper(repo: Repository, sleep_milliseconds: int = 0) -> list:
    """Fetch all reviews of an app. Basically this function is same as
    `google_play_scraper.features.reviews.reviews_all`, but retries when it fails to fetch reviews.
    """

    continuation_token = None

    result = []

    while True:
        try:
            _result, continuation_token = gps_reviews.reviews(
                repo.google_play_store_app_id, 
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=gps_reviews.MAX_COUNT_EACH_FETCH, 
                continuation_token=continuation_token, 
            )
            result += _result
            print(f"{len(result)} items fetched")

            if continuation_token.token is None:
                break

            if sleep_milliseconds:
                sleep(sleep_milliseconds / 1000)
        except Timeout as e:
            print(e)
            sleep(5)

    return result

def fetch_reviews(repo: Repository) -> list:
    start = time.time()
    if repo.google_play_store_app_id is not None:
        reviews = google_play_scraper(repo, sleep_milliseconds=0)
        user_reviews: list[UserReview] = []
        for review in reviews:
            if review["reviewCreatedVersion"] and review["content"]:
                # review_version is the version of the app that the user reviewed
                # app_version is the version of the app that the user is using
                # review_version should be used to link to the PR that is related to the review
                user_reviews.append(
                    UserReview(
                        id=review["reviewId"],
                        repository_id=repo.id,
                        review_version=review["reviewCreatedVersion"],
                        app_version=review["appVersion"],
                        content=review["content"],
                        user=review["userName"],
                        created_at=review["at"],
                        star_score=review["score"],
                        thumbs_up_count=review["thumbsUpCount"],
                    )
                )
        print(f"{len(user_reviews)} user reviews fetched.")
        print("Inserting user reviews.....")
        data_setter.add_user_reviews(user_reviews)
        print(f"{len(user_reviews)} user reviews inserted in {((time.time() - start) / 60):.2f} minutes!")


if __name__ == "__main__":
    for repo in target_repos():
        print(f"{repo.id}: {repo.owner}.{repo.name}")
        print("Fetching app details.....")
        fetch_reviews(repo)
    print("Done!")
