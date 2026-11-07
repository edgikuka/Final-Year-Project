"""
STEP 21 — sentiment (POS/NEU/NEG) with pysentimiento on translated_text_en.
pip install pysentimiento pandas matplotlib torch
Run:  python step21_sentiment.py   (first run downloads the model)
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pysentimiento import create_analyzer
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE,"posts.csv"), dtype=str, keep_default_na=False)
an = create_analyzer(task="sentiment", lang="en")
lab, pp, pn = [], [], []
for t in df["translated_text_en"].fillna("").astype(str):
    r = an.predict(t if t.strip() else "neutral")
    lab.append(r.output); pp.append(round(r.probas.get("POS",0),4)); pn.append(round(r.probas.get("NEG",0),4))
df["sentiment_label"]=lab; df["sentiment_positive"]=pp; df["sentiment_negative"]=pn
df.to_csv(os.path.join(HERE,"Posts_sentiment.csv"), index=False, encoding="utf-8-sig")
print("distribution:", df["sentiment_label"].value_counts(normalize=True).round(3).to_dict())
# pie
df["sentiment_label"].value_counts().plot.pie(autopct="%1.0f%%", figsize=(5,5),
    colors=["#3ddc97","#9aa0ad","#ff5d6c"]); plt.ylabel(""); plt.title("Sentiment distribution")
plt.tight_layout(); plt.savefig(os.path.join(HERE,"sentiment_pie.png"), dpi=130); plt.close()
# monthly trend of mean positive prob
df["created_at"]=pd.to_datetime(df["created_at"], errors="coerce", utc=True)
m=df.dropna(subset=["created_at"]).set_index("created_at")["sentiment_positive"].resample("ME").mean()
m.plot(figsize=(9,4), color="#3ddc97"); plt.ylabel("avg positive prob"); plt.title("Sentiment over time (monthly)")
plt.tight_layout(); plt.savefig(os.path.join(HERE,"sentiment_trend.png"), dpi=130)
print("saved Posts_sentiment.csv + sentiment_pie.png + sentiment_trend.png")
