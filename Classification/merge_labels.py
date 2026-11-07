"""
Merge labeled label-files across iterations into one training file (deduped by user).
Run AFTER encode_labels.py has produced numeric versions of each round.

Usage:
    python merge_labels.py iteration_1_labeled_numeric.csv iteration_2_labeled_numeric.csv
    # -> writes labeled_all_numeric.csv  (point the pipeline's LABELED_CSV at this)
Later rounds: add more files on the command line.
"""
import sys, pandas as pd
files = sys.argv[1:] or ["iteration_1_labeled_numeric.csv", "iteration_2_labeled_numeric.csv"]
frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
allrows = pd.concat(frames, ignore_index=True)
# later labels win if a user somehow appears twice
allrows = allrows.drop_duplicates(subset=["user_name"], keep="last")
allrows.to_csv("labeled_all_numeric.csv", index=False, encoding="utf-8-sig")
print(f"merged {len(files)} files -> labeled_all_numeric.csv ({len(allrows)} unique users)")
