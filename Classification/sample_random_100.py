"""
Draws the 100 random candidate rows used for iteration-1 manual labeling, and adds the
empty label columns. NOTE: the committed random_100_rows.csv is the actual draw that was
labeled; re-running with a different seed would select a different 100, so only re-run if
you intend to start a fresh sample.
"""
import pandas as pd
SEED = 42
pool = pd.read_csv("../Candidates/Candidates_user_data.csv", dtype=str, keep_default_na=False)
sample = pool.sample(n=100, random_state=SEED).reset_index(drop=True)
for col in ["target_population", "locals_vs_diaspora", "person_vs_organisation", "notes"]:
    sample[col] = ""
sample.to_csv("random_100_rows.csv", index=False, encoding="utf-8-sig")
print("wrote random_100_rows.csv (100 rows + empty label columns)")
