import os
import sqlite3
import pandas as pd
import zipfile

# Set absolute directories
PROJECT_ROOT = r"C:\FootInsight"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Locate the actual sqlite database file
db_path = os.path.join(RAW_DIR, "database.sqlite")
if os.path.isdir(db_path):
    # If it is a folder, check inside it
    nested_file = os.path.join(db_path, "database.sqlite")
    if os.path.exists(nested_file):
        db_path = nested_file

# Check for zip extraction if needed
if not os.path.isfile(db_path):
    zip_path = os.path.join(RAW_DIR, "database.sqlite.zip")
    if os.path.exists(zip_path):
        print("Extracting database.sqlite.zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(RAW_DIR)

print(f"Target database file: {db_path}")

# --- 1. Process Club Data ---
if os.path.isfile(db_path):
    print("\n--- 1. Processing Club Data ---")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        m.id AS match_id,
        c.name AS country,
        l.name AS league,
        m.season,
        m.stage,
        m.date,
        ht.team_long_name AS home_team,
        at.team_long_name AS away_team,
        m.home_team_goal,
        m.away_team_goal,
        m.B365H AS odds_home,
        m.B365D AS odds_draw,
        m.B365A AS odds_away
    FROM Match m
    JOIN Country c ON m.country_id = c.id
    JOIN League l ON m.league_id = l.id
    JOIN Team ht ON m.home_team_api_id = ht.team_api_id
    JOIN Team at ON m.away_team_api_id = at.team_api_id
    WHERE l.name IN (
        'England Premier League',
        'Spain LIGA BBVA',
        'Italy Serie A',
        'Germany 1. Bundesliga',
        'France Ligue 1'
    );
    """
    df_club = pd.read_sql_query(query, conn)
    conn.close()

    df_club = df_club.dropna(subset=['home_team_goal', 'away_team_goal'])
    df_club['home_team_goal'] = df_club['home_team_goal'].astype(int)
    df_club['away_team_goal'] = df_club['away_team_goal'].astype(int)
    df_club['date'] = pd.to_datetime(df_club['date']).dt.strftime('%Y-%m-%d')

    def get_outcome(r):
        if r['home_team_goal'] > r['away_team_goal']:
            return 'HOME_WIN'
        elif r['home_team_goal'] < r['away_team_goal']:
            return 'AWAY_WIN'
        return 'DRAW'

    df_club['outcome'] = df_club.apply(get_outcome, axis=1)
    df_club['total_goals'] = df_club['home_team_goal'] + df_club['away_team_goal']
    df_club['goal_difference'] = df_club['home_team_goal'] - df_club['away_team_goal']

    out_club = os.path.join(PROCESSED_DIR, "club_matches_top5.csv")
    df_club.to_csv(out_club, index=False)
    print(f"[SUCCESS] Saved {len(df_club)} club matches to: {out_club}")
else:
    print(f"[ERROR] Could not find the file database.sqlite!")

# --- 2. Process International Data ---
results_path = os.path.join(RAW_DIR, "results.csv")
if os.path.exists(results_path):
    print("\n--- 2. Processing International Matches ---")
    df_res = pd.read_csv(results_path).dropna(subset=['home_score', 'away_score'])
    df_res['home_score'] = df_res['home_score'].astype(int)
    df_res['away_score'] = df_res['away_score'].astype(int)

    target_tournaments = ['FIFA World Cup', 'UEFA Euro', 'Copa América']
    df_intl = df_res[df_res['tournament'].isin(target_tournaments)].copy()

    def get_outcome_intl(r):
        if r['home_score'] > r['away_score']:
            return 'HOME_WIN'
        elif r['home_score'] < r['away_score']:
            return 'AWAY_WIN'
        return 'DRAW'

    df_intl['outcome'] = df_intl.apply(get_outcome_intl, axis=1)
    df_intl['total_goals'] = df_intl['home_score'] + df_intl['away_score']

    out_intl = os.path.join(PROCESSED_DIR, "international_matches.csv")
    df_intl.to_csv(out_intl, index=False)
    print(f"[SUCCESS] Saved {len(df_intl)} international matches to: {out_intl}")

# --- 3. Process Goalscorers ---
scorers_path = os.path.join(RAW_DIR, "goalscorers.csv")
if os.path.exists(scorers_path):
    df_scorers = pd.read_csv(scorers_path).dropna(subset=['scorer', 'team'])
    out_scorers = os.path.join(PROCESSED_DIR, "intl_goalscorers.csv")
    df_scorers.to_csv(out_scorers, index=False)
    print(f"[SUCCESS] Saved {len(df_scorers)} goalscorers to: {out_scorers}")

# --- 4. Process Shootouts ---
shootouts_path = os.path.join(RAW_DIR, "shootouts.csv")
if os.path.exists(shootouts_path):
    df_shootouts = pd.read_csv(shootouts_path).dropna(subset=['winner'])
    out_shootouts = os.path.join(PROCESSED_DIR, "intl_shootouts.csv")
    df_shootouts.to_csv(out_shootouts, index=False)
    print(f"[SUCCESS] Saved {len(df_shootouts)} shootouts to: {out_shootouts}")

print("\n--- Files in data/processed/ ---")
print(os.listdir(PROCESSED_DIR))