"""
Candidates_statistics.csv : descriptive metrics over the unique candidate pool
(Candidates_user_data.csv). followers/following are stored as text like '6.6M' / '27.2K'
/ '8,565', so they are parsed to numbers first.
"""
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

df = pd.read_csv("Candidates_user_data.csv", dtype=str, keep_default_na=False).drop_duplicates(subset=["user_name"])
f = df["followers"].map(to_num).dropna(); g = df["following"].map(to_num).dropna()
out = pd.DataFrame({"metric": ["Total unique users","Average followers","Average Following","Median Followers","Median Following"],
                    "value":  [len(df), round(f.mean(),2), round(g.mean(),2), f.median(), g.median()]})
out.to_csv("Candidates_statistics.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
