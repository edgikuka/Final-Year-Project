"""
STEP 18 (b) — AFTER the scraper has run, standardize the per-user tweet files into
Data/Users_Timelines/<username>.csv and build the timeline statistics + histogram.
Point TIMELINES_DIR at the scraper's output ( .../twitter-scraper-main/data/users ).
Run:  python step18_timeline_stats.py
"""
import os, glob, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
TIMELINES_DIR = os.path.join(HERE, "..", "tools", "twitter-scraper-main", "data", "users")
OUTDIR = os.path.join(HERE, "Data", "Users_Timelines"); os.makedirs(OUTDIR, exist_ok=True)
START, END = "2023-01-01", "2023-10-08"

rows = []
for tw_path in glob.glob(os.path.join(TIMELINES_DIR, "*", "tweets.csv")):
    user = os.path.basename(os.path.dirname(tw_path))
    df = pd.read_csv(tw_path, dtype=str, keep_default_na=False)
    # map scraper columns -> standard schema
    std = pd.DataFrame({
        "tweet_id": df.get("id", ""), "created_at": df.get("publication_date", ""),
        "text": df.get("text", ""), "likes": df.get("likes", ""),
        "retweets": df.get("retweets", ""), "replies": df.get("replies", ""),
        "username": user,
    })
    std["created_at"] = pd.to_datetime(std["created_at"], errors="coerce", utc=True)
    std = std[(std["created_at"] >= START) & (std["created_at"] < END)]   # keep project window
    std.to_csv(os.path.join(OUTDIR, f"{user}.csv"), index=False, encoding="utf-8-sig")
    likes = pd.to_numeric(std["likes"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    rt = pd.to_numeric(std["retweets"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    rows.append({"username": user, "#tweets": len(std),
                 "avg_tweet_length": round(std["text"].astype(str).str.len().mean(), 1) if len(std) else 0,
                 "total_likes": int(likes.sum()), "total_retweets": int(rt.sum())})

stats = pd.DataFrame(rows).sort_values("#tweets", ascending=False)
stats.to_csv(os.path.join(HERE, "timeline_stats.csv"), index=False, encoding="utf-8-sig")
print(f"{len(stats)} users, {stats['#tweets'].sum()} tweets total -> timeline_stats.csv")
plt.figure(figsize=(7,4)); plt.hist(stats["#tweets"], bins=30, color="#3ddc97")
plt.xlabel("tweets per user"); plt.ylabel("users"); plt.title("Tweets collected per target user")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "tweets_per_user_hist.png"), dpi=130)
print("saved tweets_per_user_hist.png")
