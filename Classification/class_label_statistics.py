"""
Produces stats/iteration_1_<task>_stats.csv : class / count / percentage for each of the
three labeled target columns, from the string-labeled file.
"""
import os, pandas as pd
df = pd.read_csv("iteration_1_labeled.csv", dtype=str, keep_default_na=False)
n = len(df); os.makedirs("stats", exist_ok=True)
order = {"target_population": ["target","non_target","unknown"],
         "locals_vs_diaspora": ["local","diaspora","unknown"],
         "person_vs_organisation": ["person","organisation","unknown"]}
for col, cats in order.items():
    vc = df[col].str.strip().value_counts()
    out = pd.DataFrame({"class": cats,
                        "count": [int(vc.get(c, 0)) for c in cats],
                        "percentage": [f"{vc.get(c,0)/n*100:g}%" for c in cats]})
    out.to_csv(f"stats/iteration_1_{col}_stats.csv", index=False, encoding="utf-8-sig")
    print(col, "->", dict(zip(out["class"], out["count"])))
