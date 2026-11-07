"""
following_followers_statistics.csv : for each POI, how many followers and how many
following accounts were actually collected by the crawler.
follower-side types {followers, verified_followers} -> followers_collected
type == following                                     -> following_collected
"""
import pandas as pd
conn = pd.read_csv("POIs_candidate_connections.csv")
conn["bucket"] = conn["type"].map(lambda t: "following_collected" if t == "following" else "followers_collected")
stats = (conn.groupby(["target_username", "bucket"]).size()
              .unstack(fill_value=0).reset_index()
              .rename(columns={"target_username": "twitter_username"}))
for c in ("followers_collected", "following_collected"):
    if c not in stats: stats[c] = 0
stats = stats[["twitter_username", "followers_collected", "following_collected"]]
stats.to_csv("following_followers_statistics.csv", index=False, encoding="utf-8-sig")
print(f"{len(stats)} POIs -> following_followers_statistics.csv")
