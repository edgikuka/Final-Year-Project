import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

def fix_hebrew(text: str) -> str:
    shaped = arabic_reshaper.reshape(text)
    return get_display(shaped)

data = [
    ("המחאה נגד הרפורמה",          "Person",      119,    67),
    ("המחאה נגד הרפורמה",          "Organization",57,     42),

    ("הרפורמה המשפטית (2023)",     "Person",      36,     18),
    ("הרפורמה המשפטית (2023)",     "Organization",2,      2),

    ("התמיכה ברפורמה",             "Person",      45,     26),
    ("התמיכה ברפורמה",             "Organization",6,      6),

    ("חברי כנסת (25)",             "Person",      147,    134),
]

df = pd.DataFrame(data, columns=["SourceHebrew", "EntityType", "Total", "WithX"])

# Simple RTL label for the first two plots
df["Source"] = df["SourceHebrew"].apply(fix_hebrew)

df["WithoutX"] = df["Total"] - df["WithX"]

# --------- Plot 1: Entities per source ---------
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x="Source",
    y="Total",
    hue="EntityType"
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Count")
plt.title("Relevant Entities per Source (People vs Organizations)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1.png",
            dpi=300,
            bbox_inches='tight')
plt.show()

# --------- Long format for X plots ---------
df_long = df.melt(
    id_vars=["SourceHebrew", "Source", "EntityType"],
    value_vars=["WithX", "WithoutX"],
    var_name="XStatusRaw",
    value_name="Count"
)

df_long["XStatus"] = df_long["XStatusRaw"].map({
    "WithX": "With X",
    "WithoutX": "Without X"
})

# --------- Plot 2: X availability per source ---------
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_long,
    x="Source",
    y="Count",
    hue="XStatus"
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Count")
plt.title("X Availability by Source")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2.png",
            dpi=300,
            bbox_inches='tight')
plt.show()

g = sns.catplot(
    data=df_long,
    kind="bar",
    x="Source",      # ← כאן שיניתי מ-SourceWrapped ל-Source
    y="Count",
    hue="XStatus",
    col="EntityType",
    height=4,
    aspect=1.5
)

g.fig.suptitle("X Availability by Source", fontsize=16)
g.fig.subplots_adjust(top=0.88)

# Safe tick styling without warnings
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelrotation=20, labelsize=10, labelright=False)
    for label in ax.get_xticklabels():
        label.set_ha("right")   # יותר קריא לשמות עבריים ארוכים

g.fig.subplots_adjust(bottom=0.25)
plt.savefig(OUTPUT_DIR / "3.png",
            dpi=300,
            bbox_inches='tight')
plt.show()


