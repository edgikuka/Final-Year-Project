"""
Builds candidate_usernames.csv: the unique list of users connected to the POIs
(followers / verified_followers / following), drawn from the crawler output
POIs_candidate_connections.csv. These are the candidates for the target population.
"""
import pandas as pd
conn = pd.read_csv("POIs_candidate_connections.csv")          # target_username, other_username, type
unique = (conn["other_username"].dropna().drop_duplicates().reset_index(drop=True))
unique.to_frame("other_username").to_csv("candidate_usernames.csv", index=False, encoding="utf-8-sig")
print(f"{len(unique)} unique candidate usernames -> candidate_usernames.csv")
