"""
STEP 22 — emotion (joy/anger/sadness/fear/surprise/disgust/others) with pysentimiento.
pip install pysentimiento pandas matplotlib torch
Run:  python step22_emotion.py
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pysentimiento import create_analyzer
HERE = os.path.dirname(os.path.abspath(__file__))
src = "Posts_sentiment.csv" if os.path.exists(os.path.join(HERE,"Posts_sentiment.csv")) else "posts.csv"
df = pd.read_csv(os.path.join(HERE,src), dtype=str, keep_default_na=False)
an = create_analyzer(task="emotion", lang="en")
EMO = ["joy","anger","sadness","fear","surprise","disgust","others"]
labels = {e: [] for e in EMO}; dom = []
for t in df["translated_text_en"].fillna("").astype(str):
    r = an.predict(t if t.strip() else "others"); dom.append(r.output)
    for e in EMO: labels[e].append(round(r.probas.get(e,0),4))
df["emotion_label"]=dom
for e in EMO: df[f"emotion_{e}"]=labels[e]
df.to_csv(os.path.join(HERE,"Posts_sentiment_emotion.csv"), index=False, encoding="utf-8-sig")
print("dominant emotions:", df["emotion_label"].value_counts(normalize=True).round(3).to_dict())
# monthly mean per emotion
df["created_at"]=pd.to_datetime(df["created_at"], errors="coerce", utc=True)
d=df.dropna(subset=["created_at"]).set_index("created_at")
fig,axes=plt.subplots(len(EMO),1,figsize=(9,12),sharex=True)
for ax,e in zip(axes,EMO):
    d[f"emotion_{e}"].resample("ME").mean().plot(ax=ax); ax.set_ylabel(e,rotation=0,labelpad=30,ha="right")
axes[0].set_title("Emotion intensity over time (monthly mean)")
plt.tight_layout(); plt.savefig(os.path.join(HERE,"emotion_trends.png"), dpi=130)
print("saved Posts_sentiment_emotion.csv + emotion_trends.png")
