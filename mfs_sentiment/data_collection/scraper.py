import os
import time
import pickle
from datetime import datetime, timedelta
from google_play_scraper import reviews, Sort


def scrape_reviews(app_id: str, days: int = 365, lang='en', country='us'):
    """
    Scrape all reviews within a given time horizon for a Play Store app.
    Supports resuming from checkpoint if interrupted.

    :param app_id:   App ID e.g. 'com.global.foodpanda.android'
    :param days:     How many days back to collect reviews (default: 365)
    :param lang:     Language code for reviews (default: 'en')
    :param country:  Country store to query (default: 'us')
    """
    all_reviews = []
    continuation_token = None
    cutoff_date = datetime.now() - timedelta(days=days)
    batch_num = 0
    stop = False

    checkpoint_file = f"{app_id}_{lang}_{country}_checkpoint.pkl"

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        all_reviews = checkpoint["all_reviews"]
        continuation_token = checkpoint["continuation_token"]
        print(f"  → Loaded {len(all_reviews)} reviews from checkpoint")
    else:
        print(f"Starting fresh scrape for: {app_id} | lang={lang} | country={country}")

    print(f"Collecting reviews since: {cutoff_date.strftime('%Y-%m-%d')}\n")

    while not stop:
        batch_num += 1
        print(f"Fetching batch #{batch_num}...")

        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token
            )
        except Exception as e:
            print(f"Error on batch #{batch_num}: {e}")
            # Save checkpoint before exiting on error
            with open(checkpoint_file, "wb") as f:
                pickle.dump({"all_reviews": all_reviews, "continuation_token": continuation_token}, f)
            print("Checkpoint saved. You can resume later.")
            break

        if not result:
            print("No more reviews returned.")
            break

        for review in result:
            review_date = review['at']

            if review_date < cutoff_date:
                stop = True
                break

            all_reviews.append({
                'review_id':       review.get('reviewId'),
                'username':        review.get('userName'),
                'user_image':      review.get('userImage'),
                'rating':          review.get('score'),
                'review_text':     review.get('content'),
                'thumbs_up_count': review.get('thumbsUpCount'),
                'review_date':     review_date.strftime('%Y-%m-%d %H:%M:%S'),
                'app_version':     review.get('reviewCreatedVersion'),
                'replied_at':      review.get('repliedAt').strftime('%Y-%m-%d %H:%M:%S') if review.get('repliedAt') else None,
                'reply_content':   review.get('replyContent'),
                'lang':            lang,
            })

        print(f"  → Collected {len(all_reviews)} reviews so far...")

        # Save checkpoint after every batch
        with open(checkpoint_file, "wb") as f:
            pickle.dump({"all_reviews": all_reviews, "continuation_token": continuation_token}, f)

        time.sleep(1.5)

        if not continuation_token:
            print("No continuation token — reached end of reviews.")
            break

    print(f"\n✅ Done! Total reviews collected: {len(all_reviews)}")

    # Clean up checkpoint once complete
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint cleared.")

    return all_reviews