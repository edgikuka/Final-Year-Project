"""
POI_statistics.csv : per source Wikipedia article, how many titles were extracted,
how many were marked relevant / person / organisation, and how many got an X username.
Input is the combined relevancy+username table (V = marked true).
"""
# NOTE: the committed POI_statistics.csv was computed over a larger pre-dedup combined
# source table (edited in Excel) that is not in the repo, so per-source totals differ slightly;
# the columns and definitions computed here are identical.
import pandas as pd
df = pd.read_csv("../04_twitter_usernames/final_POIs_with_usernames_FULL.csv", dtype=str, keep_default_na=False)
df.columns = df.columns.str.strip().str.lower()
g = df.groupby("source")
out = pd.DataFrame({
    "source":        g.size().index,
    "total_rows":    g.size().values,
    "relevant":      g.apply(lambda d: (d["relevant"] == "V").sum()).values,
    "person":        g.apply(lambda d: (d["person"] == "V").sum()).values,
    "organisation":  g.apply(lambda d: (d["organisation"] == "V").sum()).values,
    "x_usernames":   g.apply(lambda d: (d["x username"].str.strip() != "").sum()).values,
})
out.to_csv("POI_statistics.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
