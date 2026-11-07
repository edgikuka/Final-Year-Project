"""
Step 12 helper: split the labeled file into one CSV per target column,
as the guide requires (iteration_1_target_population.csv, etc.).

Usage:  python split_labeled_by_task.py iteration_1_labeled_numeric.csv 1
        (second arg = iteration number, default 1)
"""
import sys, pandas as pd

INFILE = sys.argv[1] if len(sys.argv) > 1 else "iteration_1_labeled_numeric.csv"
ITER   = sys.argv[2] if len(sys.argv) > 2 else "1"
ID_COLS = ["user_name","name","bio","location","followers","following","join_date",
           "account_location","is_verified","connected_via","username_changes","uses_vpn"]
TASKS = ["target_population","locals_vs_diaspora","person_vs_organisation"]

df = pd.read_csv(INFILE, dtype=str, keep_default_na=False)
for task in TASKS:
    if task not in df.columns:
        print(f"!! {task} not in file, skipping"); continue
    cols = [c for c in ID_COLS if c in df.columns] + [task]
    out = df[df[task] != ""][cols]
    fn = f"iteration_{ITER}_{task}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")
    print(f"wrote {fn}  ({len(out)} labeled rows)")
