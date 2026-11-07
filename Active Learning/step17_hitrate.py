"""
STEP 17 (part 2) — after you fill is_target (1/0) in validation_sample.csv, this computes
the Hit Rate (the threshold's real-world precision) and tells you whether you're done or need
another labeling round.
Run:  python step17_hitrate.py
"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_HIT_RATE = 0.85   # below this -> do one more labeling round and re-run step 17

df = pd.read_csv(os.path.join(HERE,"validation_sample.csv"), dtype=str, keep_default_na=False)
df["is_target"] = pd.to_numeric(df["is_target"], errors="coerce")
checked = df.dropna(subset=["is_target"])
if len(checked)==0:
    raise SystemExit("No rows filled in. Set is_target to 1 or 0 in validation_sample.csv first.")

hits = int((checked.is_target==1).sum()); total = len(checked)
rate = hits/total
print(f"Checked {total} users above the threshold.")
print(f"Genuinely target: {hits}")
print(f"HIT RATE = {hits}/{total} = {rate*100:.1f}%")
if rate >= MIN_HIT_RATE:
    print(f"\n>= {MIN_HIT_RATE*100:.0f}% -> threshold is reliable. DONE with step 17.")
    print("Your final target population is final_target_population.csv; no more labeling needed.")
else:
    print(f"\n< {MIN_HIT_RATE*100:.0f}% -> threshold too loose. Do ONE more labeling round:")
    print("  - sample ~50 users with confidence_level between 0.6 and 0.8 from final_model_predictions.csv,")
    print("  - label them, merge, retrain (iteration 4), then re-run step17_threshold.py and this script.")
