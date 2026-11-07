"""
STEP 20 — merge all standardized timelines into one posts.csv and translate text to English.
Adds 'translated_text_en'. Batches with checkpointing so an interruption resumes.
pip install deep-translator pandas
Run:  python step20_translate.py
"""
import os, glob, pandas as pd
from deep_translator import GoogleTranslator
HERE = os.path.dirname(os.path.abspath(__file__))
TLDIR = os.path.join(HERE, "Data", "Users_Timelines")
OUT = os.path.join(HERE, "posts.csv"); BATCH = 500

def load_all():
    frames = []
    for f in glob.glob(os.path.join(TLDIR, "*.csv")):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        frames.append(d[["tweet_id","created_at","username","text"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["tweet_id","created_at","username","text"])

df = load_all() if not os.path.exists(OUT) else pd.read_csv(OUT, dtype=str, keep_default_na=False)
if "translated_text_en" not in df.columns: df["translated_text_en"] = ""
df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
todo = df.index[df["translated_text_en"].astype(str).str.strip() == ""].tolist()
print(f"{len(df)} posts, {len(todo)} to translate")
tr = GoogleTranslator(source="auto", target="en")
for i in range(0, len(todo), BATCH):
    for idx in todo[i:i+BATCH]:
        t = str(df.at[idx, "text"])[:4900]
        try: df.at[idx, "translated_text_en"] = tr.translate(t) if t.strip() else ""
        except Exception: df.at[idx, "translated_text_en"] = t   # fallback: keep original
    df.to_csv(OUT, index=False, encoding="utf-8-sig")            # checkpoint
    print(f"  translated through {min(i+BATCH, len(todo))}/{len(todo)}")
print("done ->", OUT)
