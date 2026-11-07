import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- Load CSV ---
df = pd.read_csv("titles_with_sources1.csv")

# למקרה שיש רווחים/אותיות גדולות בשמות העמודות
df.columns = df.columns.str.strip().str.lower()

# Normalize boolean fields
df["is_relevant"] = df["relevant"].apply(lambda x: x == "V")
df["is_person"] = df["person"].apply(lambda x: x == "V")
df["is_org"] = df["organisation"].apply(lambda x: x == "V")
df["has_twitter"] = df["x username"].apply(
    lambda x: isinstance(x, str) and x.strip() != ""
)

# --- Build graph ---
G = nx.Graph()

# Add source page nodes
sources = df["source"].unique()
for page in sources:
    G.add_node(page, node_type="source")

source_nodes_set = set(df["source"])
for p in source_nodes_set:
    G.add_node(p, node_type="source")  # לוודא שכולם מסומנים כ-source

# Add entity nodes + edges
for _, row in df.iterrows():
    title = row["title"]
    source = row["source"]

    # מניעת דריסה של צומת מקור
    if title in source_nodes_set:
        title = title + "_entity"

    G.add_node(
        title,
        node_type="entity",
        is_relevant=row["is_relevant"],
        is_person=row["is_person"],
        is_org=row["is_org"],
        has_twitter=row["has_twitter"]
    )

    G.add_edge(source, title)

# --- Identify node groups ---
entity_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "entity"]
source_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "source"]

# --- Custom layout with radius_sources + radius_entities ---
pos = {}

# מקומות למקורות – במעגל גדול
sources_list = list(source_nodes)
num_sources = len(sources_list)

radius_sources = 6.0  # ככל שמגדילים – המקורות רחוקים יותר אחד מהשני
angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)

for angle, source in zip(angles, sources_list):
    pos[source] = np.array([
        radius_sources * np.cos(angle),
        radius_sources * np.sin(angle)
    ])

# רדיוס לפיזור הישויות סביב ה-source שלהן
radius_entities = 3.0  # ← זה מה שתגביר/תקטין בשביל "לרווח" את הנקודות

for _, row in df.iterrows():
    raw_title = row["title"]
    source = row["source"]

    title = raw_title
    if title in source_nodes_set:
        title = title + "_entity"

    # במקרה נדיר שמשהו לא נכנס לגרף – מדלגים
    if title not in G or source not in pos:
        continue

    sx, sy = pos[source]

    # זווית רנדומלית סביב ה-source
    angle = np.random.uniform(0, 2 * np.pi)

    ex = sx + radius_entities * np.cos(angle)
    ey = sy + radius_entities * np.sin(angle)

    pos[title] = np.array([ex, ey])

# --- Colors ---
def entity_color(data):
    if not data["is_relevant"]:
        return "lightgray"          # Not relevant
    if data["has_twitter"]:
        return "tab:cyan"           # Relevant + Twitter
    return "tab:olive"              # Relevant + no Twitter

entity_colors = [entity_color(G.nodes[n]) for n in entity_nodes]
source_colors = ["black"] * len(source_nodes)

# Shapes split by person/org
person_nodes = [n for n in entity_nodes if G.nodes[n]["is_person"]]
org_nodes = [n for n in entity_nodes if G.nodes[n]["is_org"]]
other_nodes = [
    n for n in entity_nodes
    if not (G.nodes[n]["is_person"] or G.nodes[n]["is_org"])
]

person_colors = [entity_color(G.nodes[n]) for n in person_nodes]
org_colors = [entity_color(G.nodes[n]) for n in org_nodes]
other_colors = [entity_color(G.nodes[n]) for n in other_nodes]

# --- Draw ---
plt.figure(figsize=(12, 10))

# Source nodes (דפי מקור)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=source_nodes,
    node_size=1000,
    node_color=source_colors,
    node_shape="D",
    edgecolors="white",
    linewidths=2
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

# Labels only for source pages (פחות בלאגן)
nx.draw_networkx_labels(
    G, pos,
    labels={n: n for n in source_nodes},
    font_color="white",
    font_size=16
)

plt.title("Network Graph של אינטרלינקים לפי מקור, סוג, רלוונטיות וטוויטר", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
