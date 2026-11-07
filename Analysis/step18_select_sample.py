"""
STEP 18 (a) — pick the sample of target users to scrape, BEFORE running the tweet scraper.
Reads the final target population from step 17 and draws a random sample.
Outputs:
  - sample_usernames.txt   (one username per line)
  - sample_users_config.txt (a ready-to-paste YAML 'users:' block for the scraper config)
Run:  python step18_select_sample.py
"""
import os, glob, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_N = 150          # how many target users to scrape (keep modest — scraping is slow)
SEED = 42

def find_pop():
    for p in ["../Active Learning/final_target_population.csv",
              "final_target_population.csv",
              "../Active Learning/final_model_predictions.csv"]:
        fp = os.path.join(HERE, p)
        if os.path.exists(fp): return fp
    raise FileNotFoundError("final_target_population.csv not found")

df = pd.read_csv(find_pop(), dtype=str, keep_default_na=False)
uname = "username" if "username" in df.columns else "user_name"
# if reading the full predictions file, keep only high-confidence targets
if "predicted_label" in df.columns:
    df["confidence_level"] = pd.to_numeric(df["confidence_level"], errors="coerce")
    df = df[(df["predicted_label"] == "target") & (df["confidence_level"] >= 0.8)]
samp = df.sample(n=min(SAMPLE_N, len(df)), random_state=SEED)
users = samp[uname].astype(str).tolist()

open(os.path.join(HERE,"sample_usernames.txt"),"w",encoding="utf-8").write("\n".join(users))
with open(os.path.join(HERE,"sample_users_config.txt"),"w",encoding="utf-8") as f:
    f.write("users:\n" + "".join(f"  - {u}\n" for u in users))
print(f"Sampled {len(users)} target users -> sample_usernames.txt")
print("Paste the contents of sample_users_config.txt into the scraper's config.yaml 'users:' list.")
