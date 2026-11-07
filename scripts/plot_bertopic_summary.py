"""Create a publication-safe aggregate plot from BERTopic's topic_info.csv.

Only aggregate topic counts and names are used. Representative document text,
account identifiers, and individual post data are never written to the chart.

Usage:
    python scripts/plot_bertopic_summary.py path/to/topic_info.csv
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(input_file: Path, output_file: Path) -> None:
    topics = pd.read_csv(input_file, usecols=["Topic", "Count", "Name"])
    assigned = topics.loc[topics["Topic"] >= 0].copy()
    if assigned.empty:
        raise ValueError("No assigned topics found in the source file")

    top = assigned.nlargest(12, "Count").iloc[::-1]
    labels = []
    for _, row in top.iterrows():
        # Names are BERTopic labels such as '0_thank_happy_year_dear'.
        tokens = str(row["Name"]).split("_")[1:4]
        words = ", ".join(tokens)
        labels.append(f"Topic {int(row['Topic'])}  ·  {words}")

    fig, ax = plt.subplots(figsize=(11, 7.0), facecolor="white")
    bars = ax.barh(labels, top["Count"], color="#3775a9", height=0.66)
    ax.bar_label(bars, padding=7, fontsize=9.5, color="#1e293b")
    ax.set_xlim(0, float(top["Count"].max()) * 1.17)
    ax.set_xlabel("Number of posts", fontsize=10, labelpad=10)
    fig.text(0.06, 0.96, "Largest BERTopic clusters", fontsize=18, weight="bold")
    fig.text(
        0.06,
        0.92,
        f"Top 12 of {len(assigned)} topics  ·  {int(assigned['Count'].sum()):,} assigned posts",
        fontsize=10.5,
        color="#64748b",
    )
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0, labelsize=10, pad=8)
    ax.tick_params(axis="x", labelsize=9, colors="#64748b")
    fig.subplots_adjust(left=0.32, right=0.94, top=0.86, bottom=0.12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic_info", type=Path, help="Private BERTopic topic_info.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "bertopic_top_topics.png",
    )
    args = parser.parse_args()
    plot(args.topic_info, args.output)
