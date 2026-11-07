"""
STEP 17 (part 1) — run the final model's scores through a confidence threshold,
build the confidence table + histogram, define the final target population, and
draw a validation sample for you to hand-check.

Place this in your  Active Learning/  folder (next to the iteration_3 predictions).
Run:  python step17_threshold.py
Then open validation_sample.csv, fill the is_target column (1 = really target, 0 = not),
and run step17_hitrate.py.
"""
import os, glob, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# ---- CONFIG ----
THRESHOLD   = 0.80      # confidence cutoff to include a user in the target population
SAMPLE_N    = 40        # how many above-threshold users to hand-check (guide uses 100; 30-50 is fine)
TARGET_LABEL = "target" # the predicted label that means "in the target population"
SEED = 42
# input: the latest predictions file (last AL iteration). Edit if your name differs.
def find_pred():
    for pat in ["iteration_3_unlabeled_users_predictions.csv",
                "iteration_*_unlabeled_users_predictions.csv",
                "final_model_predictions.csv"]:
        hits = sorted(glob.glob(os.path.join(HERE, pat)))
        if hits: return hits[-1]
    raise FileNotFoundError("No predictions CSV found in this folder.")

def main():
    src = find_pred(); print("reading:", os.path.basename(src))
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    df["confidence_level"] = pd.to_numeric(df["confidence_level"], errors="coerce")
    if "predicted_label" not in df.columns:   # derive from numeric class if needed (target=0)
        df["predicted_label"] = df["predicted_class"].map({"0": "target", "1": "non_target", "2": "unknown"})

    # 1) final_model_predictions.csv (required deliverable)
    uname = "user_name" if "user_name" in df.columns else "username"
    keep = [c for c in [uname,"predicted_class","predicted_label","confidence_level",
                        "prob_0","prob_1","prob_2","uncertainty_score"] if c in df.columns]
    df[keep].rename(columns={uname:"username"}).to_csv(os.path.join(HERE,"final_model_predictions.csv"),
                                                        index=False, encoding="utf-8-sig")

    # 2) confidence distribution table + histogram
    n = len(df)
    rows = [(t, int((df.confidence_level>=t).sum()), round((df.confidence_level>=t).mean()*100,1))
            for t in (0.9,0.8,0.7)]
    tbl = pd.DataFrame(rows, columns=["confidence_>=","num_users","pct_of_all"])
    tbl.to_csv(os.path.join(HERE,"confidence_distribution.csv"), index=False, encoding="utf-8-sig")
    print("\nConfidence distribution:\n", tbl.to_string(index=False))
    plt.figure(figsize=(7,4)); plt.hist(df.confidence_level.dropna(), bins=30, color="#3ddc97")
    plt.axvline(THRESHOLD, color="#c0392b", ls="--", label=f"threshold {THRESHOLD}")
    plt.xlabel("confidence_level"); plt.ylabel("users"); plt.title("Model confidence over all users")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(HERE,"confidence_histogram.png"), dpi=130)

    # 3) final target population = predicted target AND confidence >= threshold
    pop = df[(df.predicted_label==TARGET_LABEL) & (df.confidence_level>=THRESHOLD)].copy()
    ctx = [c for c in [uname,"name","bio","location","followers","confidence_level"] if c in pop.columns]
    pop[ctx].rename(columns={uname:"username"}).to_csv(os.path.join(HERE,"final_target_population.csv"),
                                                       index=False, encoding="utf-8-sig")
    print(f"\nFinal target population: {len(pop)} users "
          f"(predicted '{TARGET_LABEL}' with confidence >= {THRESHOLD})")

    # 4) validation sample to hand-check
    samp = pop.sample(n=min(SAMPLE_N, len(pop)), random_state=SEED).copy()
    samp["profile_url"] = "https://x.com/" + samp[uname].astype(str)
    samp["is_target"] = ""      # <-- you fill: 1 = really target, 0 = not
    samp["notes"] = ""
    cols = [uname,"name","bio","location","confidence_level","profile_url","is_target","notes"]
    samp[[c for c in cols if c in samp.columns]].rename(columns={uname:"username"}).to_csv(
        os.path.join(HERE,"validation_sample.csv"), index=False, encoding="utf-8-sig")
    print(f"\nWrote validation_sample.csv with {len(samp)} users to hand-check.")
    print("Open it, set is_target to 1/0 for each, then run step17_hitrate.py")

if __name__ == "__main__":
    main()
