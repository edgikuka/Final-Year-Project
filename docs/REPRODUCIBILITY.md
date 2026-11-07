# Reproducibility and dataset availability

This repository contains original research scripts, aggregate visualizations, and the academic report. It is **not** a runnable end-to-end reproduction of the original experiment because account-level datasets and external crawlers have been deliberately omitted.

## Execution order and expected inputs

| Stage | Script(s) | Required input | Output (not committed) |
| --- | --- | --- | --- |
| 1. POI discovery | `POIs/01_wikipedia_extraction/Extract Interlinks.py`, `POIs/02_titles_to_csv/json_to_csv.py` | Public Hebrew Wikipedia pages / extracted title JSON | Source/title CSVs |
| 2. Entity review | `POIs/05_metadata/*.py`, `POIs/06_statistics/*.py` | Manually reviewed POI lists, X handles, metadata | Merged metadata and entity charts |
| 3. Candidate pool | `Candidates/*.py` | POI follower/following exports and account metadata | Candidate records, per-POI statistics |
| 4. Human labeling | `Classification/*.py` | Authorized user-profile records and human labels | Encoded label tables |
| 5. Active learning | `Active Learning/active_learning_pipeline.py` | `Classification/labeled_all_numeric.csv` and `Candidates/Candidates_user_data.csv` | Experiment grid and predictions |
| 6. Target population | `Active Learning/step17_threshold.py`, `step17_hitrate.py` | Predictions and manually checked validation sample | Confidence cutoff files and validation calculation |
| 7. Sample and timelines | `Analysis/step18_select_sample.py`, `step18_timeline_stats.py` | Predicted target population and authorized external scraper output | Sampled account IDs, standardized timelines |
| 8. Translation | `Analysis/step20_translate.py` | `Analysis/Data/Users_Timelines/*.csv` | `Analysis/posts.csv` with translated text |
| 9. Sentiment/emotion | `Analysis/step21_sentiment.py`, `step22_emotion.py` | Translated posts, then sentiment output | Labeled posts and aggregate figures |
| 10. Topics | `Analysis/step23_topics.py` | `Posts_sentiment_emotion.csv` (fallback: `Posts_sentiment.csv` or `posts.csv`) | BERTopic assignments and topic information |

Paths above are relative to the repository root unless noted. Scripts generally read or write files next to their source, and many depend on a particular column schema or working directory. Inspect each source before running it and supply **only data you are authorized to use**. The original external scraper implementations are not bundled and would need separate setup.

## Environment

The research used Python with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `bertopic`, `pysentimiento`, and related visualization libraries. See `requirements.txt` for the dependency inventory. Exact package versions from the original run were not preserved, so these results should not be assumed bit-for-bit reproducible on a new machine. NLP models may download additional weights.

For a lightweight repository check (does not run research computations):

```bash
python -m compileall -q "Active Learning" Analysis Candidates Classification POIs scripts
```

## Public data policy

The original source data included X usernames, account descriptions, profile metadata, connection edges, user timelines, and manual labels. Those records are **not published** here. Graphs and summary counts included in the repository are aggregate artifacts. The plot in `docs/bertopic_top_topics.png` was generated from the aggregate `Topic`, `Count`, and `Name` columns only; no representative posts were copied into the chart.

For an authorized source `topic_info.csv`, the summary chart can be regenerated with:

```bash
python scripts/plot_bertopic_summary.py path/to/topic_info.csv
```

## Methodological caveats

- The target-population model identifies **relevance**, not whether a user supports or opposes the reform.
- The 92.5% figure is 37 correct classifications among 40 manually reviewed high-confidence predictions; it is a sample estimate, not cross-validated accuracy of the deployed model.
- Iteration performance plots compare experimental metrics and should not be interpreted as a formal statistical convergence result.
- Machine-translated Hebrew may distort sentiment, emotion, or topics.
- The 94 collected timelines are not a probability sample of all 1,718 predicted target accounts.
