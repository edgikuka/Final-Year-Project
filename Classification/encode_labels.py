"""
Convert string labels -> numeric codes, per column.

Usage:
    python encode_labels.py random_100_rows_labeled.csv
    # -> writes random_100_rows_labeled_numeric.csv

Or set INPUT below and just run it.
"""
import sys
import pandas as pd

# ---- edit here if your filename differs ----
INPUT = sys.argv[1] if len(sys.argv) > 1 else "random_100_rows_labeled.csv"
OUTPUT = INPUT.replace(".csv", "_numeric.csv")

# ---- the mappings (first->0, second->1, unknown->2) ----
# NOTE: the project guide (step 12) uses the OPPOSITE for target_population
# (target=1, non_target=0). This script uses what you asked for: target=0,
# non_target=1. To match the guide instead, flip the two lines marked below.
MAPS = {
    "target_population": {
        "target": 0,        # <- guide uses 1 here
        "non_target": 1,    # <- guide uses 0 here
        "unknown": 2,
    },
    "locals_vs_diaspora": {
        "local": 0,
        "diaspora": 1,
        "unknown": 2,
    },
    "person_vs_organisation": {
        "person": 0,
        "organisation": 1,
        "unknown": 2,
    },
}

df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)

for col, mapping in MAPS.items():
    if col not in df.columns:
        print(f"!! column '{col}' not found, skipping")
        continue
    raw = df[col].str.strip()
    # map known values; blanks stay blank; anything unexpected is flagged
    unknown_vals = sorted(set(raw[(raw != "") & (~raw.isin(mapping))]))
    if unknown_vals:
        print(f"!! '{col}': unmapped values left as-is -> {unknown_vals}")
    df[col] = raw.map(lambda v: mapping[v] if v in mapping else (v if v == "" else v))
    # report distribution
    counts = df[col].value_counts(dropna=False).to_dict()
    print(f"   {col}: {counts}")

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nWrote {OUTPUT}  ({len(df)} rows)")
