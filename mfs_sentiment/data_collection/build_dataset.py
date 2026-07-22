from mfs_sentiment.data_collection.scraper import scrape_reviews
import json
import pandas as pd
import time


# Config
APPS = {
    "bKash":  "com.bKash.customerapp",
    "Nagad":  "com.konasl.nagad",
    "Rocket": "com.dbbl.mbs.apps.main",
}

LANGS    = ['bn', 'en']
COUNTRY  = 'bd'
DAYS     = 365

# Scrapping
app_data = {app_name: [] for app_name in APPS}  # stores reviews per app
seen_ids = set()                                  # global dedup across all apps & langs

for app_name, app_id in APPS.items():
    for lang_code in LANGS:
        print(f"\n{'='*50}")
        print(f"Scraping {app_name} | lang: {lang_code}")
        print(f"{'='*50}")

        data = scrape_reviews(app_id, days=DAYS, country=COUNTRY, lang=lang_code)

        for review in data:
            if review["review_id"] not in seen_ids:
                seen_ids.add(review["review_id"])
                review["app_name"] = app_name
                app_data[app_name].append(review)

        time.sleep(3)

# Save
for app_name, app_reviews in app_data.items():
    filename = app_name.lower()

    pd.DataFrame(app_reviews).to_csv(f"{filename}_bd_reviews.csv", index=False, encoding='utf-8-sig')

    with open(f"{filename}_bd_reviews.json", "w", encoding="utf-8") as f:
        json.dump(app_reviews, f, ensure_ascii=False, indent=2)

    print(f"✅ {app_name}: {len(app_reviews)} reviews saved → {filename}_bd_reviews.csv / .json")