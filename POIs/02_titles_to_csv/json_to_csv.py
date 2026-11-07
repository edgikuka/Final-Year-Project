# import json
# import pandas as pd
#
# inputs = ["המחאה נגד הרפורמה המשפטית_titles.json", "הרפורמה המשפטית בישראל (2023)_titles.json", "התמיכה ברפורמה המשפטית_titles.json"]
# out_csv = "combined_titles.csv"
#
# series_list = []
# for file in inputs:
#     with open(file, "r", encoding="utf-8") as f:
#         arr = json.load(f)
#         if not isinstance(arr, list):
#             raise ValueError(f"{file} is not a json array")
#         series_list.append(pd.DataFrame({"title": arr}))
#
# df = pd.concat(series_list, ignore_index=True)
# df["title_norm"] = df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
# df = df.drop_duplicates(subset=["title_norm"]).drop(columns=["title_norm"])
# df = df.sort_values("title", key=lambda s: s.str.casefold(), ignore_index=True)
#
# df.to_csv(out_csv, index=False, encoding="utf-8-sig")
# print("finished")

import json
import pandas as pd
from pathlib import Path

# ---- configure your inputs ----
inputs = [
    "המחאה נגד הרפורמה המשפטית_titles.json",
    "הרפורמה המשפטית בישראל (2023)_titles.json",
    "התמיכה ברפורמה המשפטית_titles.json",
]
out_csv = "titles_with_sources.csv"
# --------------------------------

frames = []

for fp in inputs:
    with open(fp, "r", encoding="utf-8") as f:
        arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError(f"{fp} is not a JSON array")

        src = Path(fp).stem
        src = src[:-7]

        # make DataFrame with source column
        df_part = pd.DataFrame({
            "source": src,  # file name without .json
            "title": arr
        })
        frames.append(df_part)

# Combine them
df = pd.concat(frames, ignore_index=True)

df["title_norm"] = df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df.drop_duplicates(subset=["title_norm"]).drop(columns=["title_norm"])
# Sort alphabetically by title
df = df.sort_values("title", key=lambda s: s.str.casefold(), ignore_index=True)

# ✅ Export CSV (UTF-8 with BOM so Excel shows Hebrew correctly)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"Finished: wrote {len(df)} unique titles to {out_csv}")
