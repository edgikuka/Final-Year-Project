import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent


# --- Hebrew font ---
plt.rcParams['font.family'] = 'Arial'  # תחליף לפונט שיש לך במערכת אם צריך

# --- פונקציה לתיקון עברית (RTL) ---
def fix_hebrew(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

# --- Load CSV ---
df = pd.read_csv("titles_with_sources1.csv")

# Normalize boolean fields
df["is_relevant"] = df["relevant"].apply(lambda x: x == "V")
df["is_person"] = df["person"].apply(lambda x: x == "V")
df["is_org"] = df["organisation"].apply(lambda x: x == "V")
df["has_twitter"] = df["x username"].apply(lambda x: isinstance(x, str) and x.strip() != "")

# --- Build graph ---
G = nx.Graph()

# Add source page nodes
sources = df["source"].unique()
for page in sources:
    G.add_node(page, node_type="source")

source_nodes_set = set(df["source"])

for p in source_nodes_set:
    G.add_node(p, node_type="source")

# Add entity nodes + edges
for _, row in df.iterrows():
    title = row["title"]
    source = row["source"]

    if title in source_nodes_set:
        title = title + "_entity"  # למנוע דריסה של source node

    G.add_node(
        title,
        node_type="entity",
        is_relevant=row["is_relevant"],
        is_person=row["is_person"],
        is_org=row["is_org"],
        has_twitter=row["has_twitter"]
    )

    G.add_edge(source, title)

# --- Layout ---
# pos = nx.spring_layout(G, k=0.2, seed=42)
pos = nx.kamada_kawai_layout(G, scale=5)

# --- Colors ---
def entity_color(data):
    if not data["is_relevant"]:
        return "lightgray"  # Not relevant
    if data["has_twitter"]:
        return "tab:cyan"   # Relevant + Twitter
    return "tab:olive"     # Relevant + no Twitter

entity_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "entity"]
source_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "source"]

shrink_factor = 0.7  # 0.7 אומר 70% מהמרחק – כלומר יותר קרוב למרכז

for s in source_nodes:
    pos[s] = pos[s] * shrink_factor


entity_colors = [entity_color(G.nodes[n]) for n in entity_nodes]
source_colors = ["black"] * len(source_nodes)

# Shapes split by person/org
person_nodes = [n for n in entity_nodes if G.nodes[n]["is_person"]]
org_nodes = [n for n in entity_nodes if G.nodes[n]["is_org"]]
other_nodes = [n for n in entity_nodes if not (G.nodes[n]["is_person"] or G.nodes[n]["is_org"])]

person_colors = [entity_color(G.nodes[n]) for n in person_nodes]
org_colors = [entity_color(G.nodes[n]) for n in org_nodes]
other_colors = [entity_color(G.nodes[n]) for n in other_nodes]

plt.figure(figsize=(12, 10))

# Source nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=source_nodes,
    node_size=1000,
    node_color=source_colors,
    node_shape="D",
    edgecolors="white",
    linewidths=1
)

# Person nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=person_nodes,
    node_size=220,
    node_color=person_colors,
    node_shape="o",
    edgecolors="black",
    linewidths=1
)

# Organization nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=org_nodes,
    node_size=300,
    node_color=org_colors,
    node_shape="s",
    edgecolors="black",
    linewidths=1
)

# Others (if exist)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=other_nodes,
    node_size=180,
    node_color=other_colors,
    node_shape="^",
    edgecolors="black",
    linewidths=1
)

# Edges
nx.draw_networkx_edges(G, pos, alpha=0.35)
# --- Legend (מקרא) ---

# אם אתה רוצה שגם הטקסט במקרא יהיה RTL תקין:
def h(txt):
    return fix_hebrew(txt)

# מקרא צורות – סוג הישות
shape_handles = [
    Line2D([0], [0],
           marker='D', color='w',
           markerfacecolor='black', markeredgecolor='white',
           markersize=18, linestyle='None',
           label=h("Source Page")),
    Line2D([0], [0],
           marker='o', color='w',
           markerfacecolor='white', markeredgecolor='black',
           markersize=16, linestyle='None',
           label=h("Person")),
    Line2D([0], [0],
           marker='s', color='w',
           markerfacecolor='white', markeredgecolor='black',
           markersize=16, linestyle='None',
           label=h("Organisation")),
    Line2D([0], [0],
           marker='^', color='w',
           markerfacecolor='lightgray', markeredgecolor='black',
           markersize=16, linestyle='None',
           label=h("Irrelevant")),
]

# מקרא צבעים – רלוונטיות / טוויטר
color_handles = [
    Line2D([0], [0],
           marker='h', color='w',
           markerfacecolor='tab:cyan',
           markersize=16, linestyle='None',
           label=h("Relevant with an X account")),
    Line2D([0], [0],
           marker='h', color='w',
           markerfacecolor='tab:olive',
           markersize=16, linestyle='None',
           label=h("Relevant without an X account")),
]

# איחוד המקרא – צורות + צבעים
all_handles = shape_handles + color_handles

legend = plt.legend(
    handles=all_handles,
    loc='upper left',
    bbox_to_anchor=(0.92, 0.92),   # בצד ימין מחוץ לגרף
    borderaxespad=0.,
    title="Legend",
    fontsize=14,
    title_fontsize=16,
    frameon=True
)


# --- Labels only for source pages, עם עברית תקינה + outline ---
labels = {n: fix_hebrew(n) for n in source_nodes}

text_items = nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_color="white",
    font_size=15
)

# Outline לטקסט
for txt in text_items.values():
    txt.set_path_effects([
        pe.Stroke(linewidth=4, foreground="black"),  # מסגרת שחורה עבה
        pe.Normal()
    ])

plt.title("Preliminary Graph", fontsize=25)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "preliminary_graph.png",
            dpi=300,
            bbox_inches='tight')

plt.show()
