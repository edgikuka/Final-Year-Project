"""
STEP 23 — Topic modeling with BERTopic (FIXED).

Fixes vs. the previous version that collapsed to 2 topics on the full corpus:
  * removes stop-words via a CountVectorizer (the cause of the "the/and/of" mega-topic)
  * cleans docs (strips URLs/mentions/RT, drops < 5-word and link-only tweets)
  * smaller min_topic_size options so the model can form real clusters
  * grid selection that REFUSES to pick a 1-2 topic collapse (picks most-topics if needed)
  * keeps tweet_id / created_at / probability in the output
  * prints the topic count at every step and warns loudly on a collapse

pip install bertopic umap-learn hdbscan scikit-learn pandas matplotlib
Run:  python step23_topics.py     (first run downloads an embedding model)
"""
import os, re, itertools, pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
# ---------------- CONFIG ----------------
MIN_TOPIC_SIZES = [40, 60, 80]      # smaller -> more, finer topics
N_NEIGHBORS     = [10, 15]
MIN_WORDS       = 5                  # drop tweets shorter than this (after cleaning)
MIN_DF          = 2                  # a term must appear in >=2 documents
NGRAM           = (1, 2)
CAP             = 30000             # max docs to model (raise/lower for speed)
MIN_GOOD_TOPICS = 8                 # below this we treat a run as a collapse

def clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", str(t))     # URLs
    t = re.sub(r"@\w+", " ", t)                       # mentions
    t = re.sub(r"^rt\b", " ", t, flags=re.I)          # leading RT
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---- load + clean, keeping tweet_id / created_at aligned ----
src = next((f for f in ["Posts_sentiment_emotion.csv","Posts_sentiment.csv","posts.csv"]
            if os.path.exists(os.path.join(HERE, f))), None)
if src is None:
    raise FileNotFoundError(
        "Missing input corpus in Analysis/: expected Posts_sentiment_emotion.csv, "
        "Posts_sentiment.csv, or posts.csv. Research datasets are not bundled with "
        "the public repository. See docs/REPRODUCIBILITY.md."
    )
print("reading:", src)
df = pd.read_csv(os.path.join(HERE, src), dtype=str, keep_default_na=False)
tid_col  = "tweet_id"   if "tweet_id"   in df.columns else None
date_col = "created_at" if "created_at" in df.columns else None

docs, meta = [], []
for _, r in df.iterrows():
    c = clean(r.get("translated_text_en", ""))
    if len(c.split()) >= MIN_WORDS:
        docs.append(c)
        meta.append((r.get(tid_col, "") if tid_col else "",
                     r.get(date_col, "") if date_col else "",
                     c))
    if len(docs) >= CAP: break
print(f"{len(docs)} documents after cleaning (from {len(df)} rows)")

vectorizer = CountVectorizer(stop_words="english", min_df=MIN_DF, ngram_range=NGRAM)

def build(mts, nn):
    um = UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hd = HDBSCAN(min_cluster_size=mts, metric="euclidean",
                 cluster_selection_method="eom", prediction_data=True)
    return BERTopic(umap_model=um, hdbscan_model=hd, vectorizer_model=vectorizer,
                    calculate_probabilities=False, verbose=False)

# ---- grid search ----
grid = []
for mts, nn in itertools.product(MIN_TOPIC_SIZES, N_NEIGHBORS):
    tm = build(mts, nn)
    topics, _ = tm.fit_transform(docs)
    info = tm.get_topic_info()
    n_topics = int((info.Topic >= 0).sum())
    outlier = round((pd.Series(topics) == -1).mean(), 3)
    grid.append({"min_topic_size": mts, "n_neighbors": nn,
                 "n_topics": n_topics, "outlier_ratio": outlier})
    print(f"   min_size={mts} nn={nn} -> {n_topics} topics, outliers {outlier}")
gdf = pd.DataFrame(grid)
gdf.to_csv(os.path.join(HERE, "topic_grid.csv"), index=False)

# ---- selection: never pick a collapse ----
ok = gdf[(gdf.n_topics >= MIN_GOOD_TOPICS) & (gdf.n_topics <= 60) & (gdf.outlier_ratio <= 0.4)]
if len(ok):
    best = ok.sort_values(["outlier_ratio", "n_topics"], ascending=[True, False]).iloc[0]
else:                                   # fallback: the config with the MOST topics (anti-collapse)
    best = gdf.sort_values("n_topics", ascending=False).iloc[0]
print("\nchosen config:", best.to_dict())
if best.n_topics < MIN_GOOD_TOPICS:
    print(f"  WARNING: best run only produced {int(best.n_topics)} topics. "
          f"Lower MIN_TOPIC_SIZES (e.g. [10,15,20]) and re-run.")

# ---- final model ----
tm = build(int(best.min_topic_size), int(best.n_neighbors))
topics, _ = tm.fit_transform(docs)
try:                                    # pull most topics out of the -1 outlier bucket
    topics = tm.reduce_outliers(docs, topics, strategy="c-tf-idf")
    tm.update_topics(docs, topics=topics, vectorizer_model=vectorizer)
except Exception as e:
    print("  (outlier reduction skipped:", e, ")")

info = tm.get_topic_info()
name_of = info.set_index("Topic")["Name"].to_dict()
final_n = int((info.Topic >= 0).sum())
print(f"\nFINAL: {final_n} topics, outlier ratio "
      f"{round((pd.Series(topics)==-1).mean(),3)}")
if final_n <= 2:
    print("  *** STILL COLLAPSED *** lower MIN_TOPIC_SIZES and MIN_DF, then re-run.")

# ---- save outputs (with tweet_id / created_at / probability) ----
os.makedirs(os.path.join(HERE, "Data", "Topics"), exist_ok=True)
try:    probs = tm.transform(docs)[1]
except Exception: probs = None
out = pd.DataFrame({
    "tweet_id":            [m[0] for m in meta],
    "created_at":          [m[1] for m in meta],
    "text_translated_en":  [m[2] for m in meta],
    "topic_id":            topics,
    "topic_label":         [name_of.get(t, "") for t in topics],
})
out["probability"] = (pd.Series(probs).round(4) if probs is not None else "")
out.to_csv(os.path.join(HERE, "Data", "Topics", "final_topics.csv"),
           index=False, encoding="utf-8-sig")
info.to_csv(os.path.join(HERE, "Data", "Topics", "topic_info.csv"),
            index=False, encoding="utf-8-sig")
try:
    tm.visualize_barchart(top_n_topics=min(12, final_n)).write_html(
        os.path.join(HERE, "Data", "Topics", "topics_barchart.html"))
except Exception:
    pass
print("\nsaved Data/Topics/final_topics.csv, topic_info.csv, topic_grid.csv (+ barchart.html)")
print("top topics:")
for _, r in info[info.Topic >= 0].head(12).iterrows():
    print(f"   T{r.Topic} ({r.Count}): {r.Name}")
