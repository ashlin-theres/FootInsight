import os
import requests
import pandas as pd
import numpy as np

PROJECT_ROOT = r"C:\FootInsight"
PROCESSED_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "club_matches_top5.csv")

def append_new_matchday(new_matches_df):
    """
    Simulates or executes continuous ingestion into the FootInsight warehouse.
    Appends new matchday results, deduplicates by match ID/date, and recalculates features.
    """
    if os.path.exists(PROCESSED_FILE):
        existing_df = pd.read_csv(PROCESSED_FILE)
    else:
        existing_df = pd.DataFrame()

    print(f"[*] Ingesting {len(new_matches_df)} newly completed matches...")
    
    # 1. Append and deduplicate
    combined = pd.concat([existing_df, new_matches_df], ignore_index=True)
    combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last", inplace=True)
    
    # 2. Derive outcome for new entries
    combined["outcome"] = np.where(
        combined["home_team_goal"] > combined["away_team_goal"], "HOME_WIN",
        np.where(combined["home_team_goal"] < combined["away_team_goal"], "AWAY_WIN", "DRAW")
    )
    
    # 3. Save updated warehouse state
    combined.to_csv(PROCESSED_FILE, index=False)
    print(f"[+] Total records in warehouse updated to: {len(combined)}")

# Example usage: fetching or loading an incremental batch
if __name__ == "__main__":
    print("Continuous Data Ingestion Daemon Initialized.")
    # In production, this runs on an Apache Airflow or Windows Task Scheduler trigger