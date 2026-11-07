"""
Active Learning pipeline  -  steps 14-17 of the project guide.
Project: Identifying accounts relevant to the Israeli Judicial Reform discourse.

This pipeline does NOT classify support for or opposition to the reform.

What it does
------------
1. Feature engineering from user metadata (TF-IDF on text + numeric features).
2. Trains 6 classifiers x feature-sets x {K-Fold, LOOCV} x {balanced, unbalanced}
   x {2-class, 3-class}, leakage-free (TF-IDF is fit inside each CV fold).
3. Logs every experiment to experiments_results.csv with the required columns.
4. Picks the best model per task, predicts on the unlabeled pool, and writes a
   predictions file with confidence / probabilities / uncertainty_score.
5. Picks the 100 most-uncertain users for the next labeling round (step 15).
6. Draws the accuracy-vs-iteration graph (grows as you add iterations).

Run:
    python active_learning_pipeline.py
Edit the CONFIG block for the labeled file / iteration number / grid size.
"""

import os, re, warnings, itertools
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # silence joblib/loky worker warnings too
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RND = 42
np.random.seed(RND)

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))   # outputs land next to this script
ITERATION   = 3                                   # bump to 2, 3, ... each round
LABELED_CSV = os.path.join(HERE, "..", "Classification", "labeled_all_numeric.csv")
POOL_CSV    = os.path.join(HERE, "..", "Candidates", "Candidates_user_data.csv")  # full candidate pool
RESULTS_CSV = os.path.join(HERE, "experiments_results.csv")
TARGETS     = ["target_population", "locals_vs_diaspora", "person_vs_organisation"]
PRIMARY     = "target_population"                 # task used to rank who to label next

# feature sets that memorize identity (handle/display name) instead of generalizing.
# They are still TRAINED and LOGGED, but never chosen to score the unlabeled pool.
NO_DEPLOY_FEATURES = {"TFIDF_username", "TFIDF_fullname", "TFIDF_username+fullname"}
MIN_DEPLOY_CLASS = 15   # don't deploy a model trained on a tiny balanced slice

# grid knobs -- set QUICK=False for the full grid the guide asks for
QUICK = False
CLASSIFIERS_ALL = ["LogReg", "DecisionTree", "RandomForest", "SVM", "XGBoost", "AdaBoost"]
FEATURE_SETS_ALL = [
    "TFIDF_description", "TFIDF_username", "TFIDF_fullname",
    "TFIDF_description+username", "TFIDF_description+fullname",
    "TFIDF_username+fullname", "TFIDF_all", "numeric", "TFIDF_all+numeric",
]
if QUICK:
    FEATURE_SETS = ["TFIDF_description", "TFIDF_all", "numeric", "TFIDF_all+numeric"]
    CLASS_MODES  = [2]            # 2-class only (unknown is tiny in this data)
else:
    FEATURE_SETS = FEATURE_SETS_ALL
    CLASS_MODES  = [2, 3]
CLASSIFIERS  = CLASSIFIERS_ALL
VALIDATIONS  = ["K-Fold", "LOOCV"]
BALANCE_MODES = [False, True]
KFOLD_K = 5
MAX_TFIDF = 400

# label name maps (for readable prediction output); codes match encode_labels.py
NAMES = {
    "target_population":      {0: "target", 1: "non_target", 2: "unknown"},
    "locals_vs_diaspora":     {0: "local",  1: "diaspora",   2: "unknown"},
    "person_vs_organisation": {0: "person", 1: "organisation", 2: "unknown"},
}
# ================================================================


# ---------- feature engineering ----------
def parse_count(s):
    s = (str(s) or "").strip().replace(",", "")
    if s == "" or s.lower() == "nan":
        return np.nan
    m = re.match(r"^([\d.]+)\s*([KkMm]?)$", s)
    if not m:
        try: return float(s)
        except: return np.nan
    v = float(m.group(1)); suf = m.group(2).upper()
    return v * (1_000 if suf == "K" else 1_000_000 if suf == "M" else 1)

def build_frame(df):
    """Return a frame with a combined-text helper + numeric features."""
    d = df.copy()
    for c in ["bio", "name", "user_name", "location", "connected_via"]:
        if c not in d: d[c] = ""
        d[c] = d[c].fillna("").astype(str)
    foll = d["followers"].map(parse_count)
    folg = d["following"].map(parse_count)
    d["f_followers"] = foll.fillna(foll.median())
    d["f_following"] = folg.fillna(folg.median())
    d["f_ratio"]     = d["f_followers"] / (d["f_following"] + 1)
    d["f_bio_len"]   = d["bio"].str.len()
    d["f_name_len"]  = d["name"].str.len()
    d["f_has_bio"]   = (d["bio"].str.len() > 0).astype(int)
    d["f_verified"]  = (d.get("is_verified", "").astype(str).str.upper() == "TRUE").astype(int)
    d["f_vpn"]       = (d.get("uses_vpn", "").astype(str).str.lower() == "yes").astype(int)
    d["f_uname_chg"] = pd.to_numeric(d.get("username_changes", 0), errors="coerce").fillna(0)
    d["f_via_israel"]= d["connected_via"].str.contains("Israel", case=False, na=False).astype(int)
    return d

NUMERIC_COLS = ["f_followers", "f_following", "f_ratio", "f_bio_len", "f_name_len",
                "f_has_bio", "f_verified", "f_vpn", "f_uname_chg", "f_via_israel"]

TEXT_FOR = {
    "TFIDF_description": ["bio"], "TFIDF_username": ["user_name"], "TFIDF_fullname": ["name"],
    "TFIDF_description+username": ["bio", "user_name"],
    "TFIDF_description+fullname": ["bio", "name"],
    "TFIDF_username+fullname": ["user_name", "name"],
    "TFIDF_all": ["bio", "user_name", "name"],
    "numeric": [], "TFIDF_all+numeric": ["bio", "user_name", "name"],
}

def make_X(d, feature_set):
    text_fields = TEXT_FOR[feature_set]
    X = pd.DataFrame(index=d.index)
    if text_fields:
        X["text"] = d[text_fields].agg(" ".join, axis=1)
    if feature_set in ("numeric", "TFIDF_all+numeric"):
        for c in NUMERIC_COLS:
            X[c] = d[c].values
    return X

def make_estimator(feature_set, clf):
    transformers = []
    if "text" in TEXT_FOR[feature_set] or TEXT_FOR[feature_set]:
        transformers.append(("txt", TfidfVectorizer(max_features=MAX_TFIDF, min_df=1,
                              ngram_range=(1, 2), sublinear_tf=True), "text"))
    if feature_set in ("numeric", "TFIDF_all+numeric"):
        transformers.append(("num", StandardScaler(), NUMERIC_COLS))
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight=None),
        "DecisionTree": DecisionTreeClassifier(random_state=RND),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RND),
        "SVM": SVC(probability=True, random_state=RND),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.2,
                                 eval_metric="logloss", verbosity=0, random_state=RND),
        "AdaBoost": AdaBoostClassifier(random_state=RND),
    }
    return Pipeline([("pre", pre), ("clf", models[clf])])

def balance(d, y):
    """Random undersample to the smallest class size."""
    idx = []
    n = pd.Series(y).value_counts().min()
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        idx.extend(np.random.choice(cls_idx, n, replace=False))
    idx = np.array(sorted(idx))
    return d.iloc[idx].reset_index(drop=True), y[idx]


# ---------- one experiment ----------
def run_experiment(d_full, target, classes, balanced, clf, feature_set, validation):
    d = d_full[d_full[target].isin(classes)].reset_index(drop=True)
    y = d[target].astype(int).values
    if balanced:
        d, y = balance(d, y)
    counts = pd.Series(y).value_counts()
    min_sz = int(counts.min())
    if len(np.unique(y)) < 2 or min_sz < 2:
        return None
    if validation == "K-Fold":
        k = min(KFOLD_K, min_sz)
        if k < 2: return None
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RND)
        kval = k
    else:
        cv = LeaveOneOut(); kval = ""

    X = make_X(d, feature_set)
    est = make_estimator(feature_set, clf)
    try:
        proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=-1)
        pred = proba.argmax(1)
        classes_sorted = np.unique(y)
        pred = classes_sorted[pred]
        try:
            auc = (roc_auc_score(y, proba[:, 1]) if len(classes_sorted) == 2
                   else roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = np.nan
    except Exception:
        try:
            pred = cross_val_predict(est, X, y, cv=cv, n_jobs=-1); auc = np.nan
        except Exception:
            return None

    fitted_pre = make_estimator(feature_set, clf).named_steps["pre"].fit(X, y)
    n_feat = fitted_pre.transform(X.iloc[:1]).shape[1]

    cc = pd.Series(y).value_counts().to_dict()
    return {
        "iteration": ITERATION, "target_column": target, "#classes": len(classes),
        "#class_0": cc.get(0, 0), "#class_1": cc.get(1, 0),
        "#class_2": cc.get(2, "") if len(classes) == 3 else "",
        "min_class_size": min_sz, "training_type": validation, "K": kval,
        "algorithm": clf, "feature_set": feature_set, "Feaures_count": n_feat,
        "balanced": balanced,
        "accuracy":  round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y, pred, average="macro", zero_division=0), 4),
        "F1":        round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "AUC":       round(auc, 4) if auc == auc else "",
    }


def run_grid(d_full):
    rows, total = [], 0
    grid = list(itertools.product(TARGETS, CLASS_MODES, BALANCE_MODES,
                                  CLASSIFIERS, FEATURE_SETS, VALIDATIONS))
    for target, nclass, bal, clf, fs, val in grid:
        classes = [0, 1] if nclass == 2 else [0, 1, 2]
        total += 1
        r = run_experiment(d_full, target, classes, bal, clf, fs, val)
        if r: rows.append(r)
    print(f"  ran {len(rows)} / {total} experiments (rest skipped as infeasible)")
    return pd.DataFrame(rows)


# ---------- prediction on the unlabeled pool (step 15) ----------
def predict_unlabeled(d_lab, pool, results):
    # Candidate models to DEPLOY: primary task, 2-class, trained on the full
    # (unbalanced) data, using a generalizable feature set, with enough samples.
    cand = results[(results.target_column == PRIMARY) & (results["#classes"] == 2)
                   & (results.balanced == False)
                   & (~results.feature_set.isin(NO_DEPLOY_FEATURES))
                   & (results.min_class_size >= MIN_DEPLOY_CLASS)]
    if cand.empty:   # fall back progressively if the filters are too strict
        cand = results[(results.target_column == PRIMARY) & (results["#classes"] == 2)
                       & (~results.feature_set.isin(NO_DEPLOY_FEATURES))]
    cand = cand.sort_values(["AUC", "accuracy"], ascending=False)

    classes = [0, 1]
    d = d_lab[d_lab[PRIMARY].isin(classes)].reset_index(drop=True)
    y = d[PRIMARY].astype(int).values
    unl = pool[~pool["user_name"].isin(d_lab["user_name"])].reset_index(drop=True)

    chosen, proba, cls = None, None, None
    for _, b in cand.iterrows():
        est = make_estimator(b.feature_set, b.algorithm).fit(make_X(d, b.feature_set), y)
        p = est.predict_proba(make_X(unl, b.feature_set))
        conf = p.max(1)
        # reject degenerate models: one class for everyone, or no spread in confidence
        if len(np.unique(p.argmax(1))) < 2 or conf.std() < 1e-4:
            print(f"  skip {b.algorithm}/{b.feature_set} (degenerate on pool: "
                  f"single class or constant confidence)")
            continue
        chosen, proba, cls = b, p, est.named_steps["clf"].classes_
        break
    if chosen is None:   # last resort: best generalizable model even if flat
        b = cand.iloc[0]; est = make_estimator(b.feature_set, b.algorithm).fit(make_X(d, b.feature_set), y)
        proba = est.predict_proba(make_X(unl, b.feature_set)); cls = est.named_steps["clf"].classes_; chosen = b
    print(f"  deployed {PRIMARY} model: {chosen.algorithm} / {chosen.feature_set} "
          f"(cv acc={chosen.accuracy}, AUC={chosen.AUC}); "
          f"pool confidence spread {proba.max(1).min():.3f}-{proba.max(1).max():.3f}")

    out = unl.copy()
    pred = cls[proba.argmax(1)]
    out["predicted_class"]  = pred
    out["predicted_label"]  = [NAMES[PRIMARY][c] for c in pred]
    out["confidence_level"] = proba.max(1).round(4)
    out["prob_0"] = proba[:, list(cls).index(0)].round(4) if 0 in cls else ""
    out["prob_1"] = proba[:, list(cls).index(1)].round(4) if 1 in cls else ""
    out["prob_2"] = ""
    out["uncertainty_score"] = (1 - proba.max(1)).round(4)
    out = out.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
    return out


# ---------- iteration performance graph (step 15.4) ----------
def performance_graph(results_path=RESULTS_CSV, out=None):
    if out is None: out = os.path.join(HERE, "iteration_performance.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.read_csv(results_path)
    g = res.groupby("iteration")["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g.index, g.values, "o-", lw=2, color="#3ddc97", markersize=8)
    ax.set_xlabel("Active-learning iteration"); ax.set_ylabel("Mean accuracy (all classifiers)")
    ax.set_title("Model performance across iterations")
    ax.set_xticks(sorted(res.iteration.unique())); ax.grid(alpha=.3)
    for x, v in zip(g.index, g.values): ax.annotate(f"{v:.3f}", (x, v),
        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  saved {out}")


def main():
    print(f"Iteration {ITERATION}")
    lab = pd.read_csv(LABELED_CSV, dtype=str, keep_default_na=False)
    pool = pd.read_csv(POOL_CSV, dtype=str, keep_default_na=False)
    for t in TARGETS:
        lab = lab[lab[t] != ""]
    for t in TARGETS:
        lab[t] = lab[t].astype(int)
    lab = build_frame(lab)
    pool_f = build_frame(pool)

    print("Training grid...")
    res = run_grid(lab)

    # append to the cumulative results file
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        prev = prev[prev.iteration != ITERATION]            # replace this iteration
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"  wrote {RESULTS_CSV}")

    print("Predicting on unlabeled pool...")
    preds = predict_unlabeled(lab, pool_f, res[res.iteration == ITERATION])
    pred_file = os.path.join(HERE, f"iteration_{ITERATION}_unlabeled_users_predictions.csv")
    keep = ["user_name", "name", "bio", "location", "followers", "following",
            "predicted_class", "predicted_label", "confidence_level",
            "prob_0", "prob_1", "prob_2", "uncertainty_score"]
    preds[keep].to_csv(pred_file, index=False, encoding="utf-8-sig")
    print(f"  wrote {pred_file}")

    nxt = os.path.join(HERE, f"iteration_{ITERATION}_next_to_label_top100.csv")
    preds[keep].head(100).to_csv(nxt, index=False, encoding="utf-8-sig")
    print(f"  wrote {nxt}  (100 most-uncertain users for the next round)")

    performance_graph(RESULTS_CSV)
    print("Done.")


if __name__ == "__main__":
    main()
