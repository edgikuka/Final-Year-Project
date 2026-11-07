"""
POI_twitter_users_statistics.csv : descriptive metrics over the unique POI Twitter
accounts that have metadata. (Note: the exact historical counts depend on which POI
subset had follower/following collected; this computes the same metrics over the
deduplicated POI metadata table.)
"""
# NOTE: the committed file was computed over the 264 POIs that had connections collected;
# this recomputes the same metrics over the deduplicated POI metadata table.
import re, pandas as pd
def to_num(s):
    s = str(s).strip().replace(",", "")
    if s in ("", "nan"): return None
    m = re.match(r"^([\d.]+)\s*([KkMm]?)$", s)
    if not m:
        try: return float(s)
        except: return None
    v = float(m.group(1)); suf = m.group(2).upper()
    return v * (1000 if suf == "K" else 1_000_000 if suf == "M" else 1)

df = pd.read_csv("../05_metadata/pois_joined_with_metadata.csv", dtype=str, keep_default_na=False)
df = df[df["followers"].str.strip() != ""].drop_duplicates(subset=["x username"])
f = df["followers"].map(to_num).dropna(); g = df["following"].map(to_num).dropna()
out = pd.DataFrame({"metric": ["Total unique users","Average followers","Average Following","Median Followers","Median Following"],
                    "value":  [len(df), round(f.mean(),2), round(g.mean(),2), f.median(), g.median()]})
out.to_csv("POI_twitter_users_statistics.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
