import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, avg, sum, round, desc

# 1. Initialize Spark Session (Big Data Engine)
print("Starting Spark Session...")
spark = SparkSession.builder \
    .appName("FootInsight_Analytics") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Disable verbose logging
spark.sparkContext.setLogLevel("WARN")

PROJECT_ROOT = r"C:\FootInsight"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

club_csv = os.path.join(PROCESSED_DIR, "club_matches_top5.csv")
intl_csv = os.path.join(PROCESSED_DIR, "international_matches.csv")

# 2. Load Processed Data into Distributed Spark DataFrames
print("\n--- Loading Data into Spark DataFrames ---")
df_club = spark.read.csv(club_csv, header=True, inferSchema=True)
df_intl = spark.read.csv(intl_csv, header=True, inferSchema=True)

print(f"Club Matches Count: {df_club.count()}")
print(f"International Matches Count: {df_intl.count()}")

# 3. Register as Spark SQL Views (Equivalent to Hive Tables)
df_club.createOrReplaceTempView("club_matches")
df_intl.createOrReplaceTempView("intl_matches")

# --- Query 1: League Summary (Goals and Matches) ---
print("\n[SPARK SQL] 1. League-wise Goal Metrics (HiveQL Equivalent):")
league_summary = spark.sql("""
    SELECT 
        league,
        COUNT(match_id) AS total_matches,
        ROUND(AVG(total_goals), 2) AS avg_goals_per_match,
        SUM(home_team_goal) AS total_home_goals,
        SUM(away_team_goal) AS total_away_goals
    FROM club_matches
    GROUP BY league
    ORDER BY avg_goals_per_match DESC
""")
league_summary.show(truncate=False)

# --- Query 2: Match Outcome Distribution Across Leagues ---
print("\n[SPARK SQL] 2. Match Outcome Distribution (%):")
outcome_summary = spark.sql("""
    SELECT 
        outcome,
        COUNT(*) AS match_count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM club_matches), 2) AS percentage
    FROM club_matches
    GROUP BY outcome
""")
outcome_summary.show()

# --- Query 3: Top Performing Teams (Premier League Example) ---
print("\n[SPARK SQL] 3. Top Winning Teams (English Premier League):")
top_teams = spark.sql("""
    SELECT 
        home_team AS team,
        COUNT(CASE WHEN outcome = 'HOME_WIN' THEN 1 END) AS home_wins,
        COUNT(CASE WHEN outcome = 'DRAW' THEN 1 END) AS draws,
        COUNT(CASE WHEN outcome = 'AWAY_WIN' THEN 1 END) AS home_losses,
        ROUND(AVG(home_team_goal), 2) AS avg_home_goals
    FROM club_matches
    WHERE league = 'England Premier League'
    GROUP BY home_team
    ORDER BY home_wins DESC
    LIMIT 10
""")
top_teams.show(truncate=False)

# --- Query 4: International Tournament Goals & Matches ---
print("\n[SPARK SQL] 4. International Tournaments Performance:")
intl_summary = spark.sql("""
    SELECT 
        tournament,
        COUNT(*) AS matches_played,
        SUM(total_goals) AS total_goals,
        ROUND(AVG(total_goals), 2) AS avg_goals
    FROM intl_matches
    GROUP BY tournament
    ORDER BY total_goals DESC
""")
intl_summary.show()

# --- Query 5: Head-to-Head Analysis (e.g. Barcelona vs Real Madrid) ---
print("\n[SPARK SQL] 5. Head-to-Head Sample (El Clásico):")
h2h_sample = spark.sql("""
    SELECT 
        season,
        date,
        home_team,
        away_team,
        home_team_goal,
        away_team_goal,
        outcome
    FROM club_matches
    WHERE (home_team = 'FC Barcelona' AND away_team = 'Real Madrid CF')
       OR (home_team = 'Real Madrid CF' AND away_team = 'FC Barcelona')
    ORDER BY date DESC
    LIMIT 6
""")
h2h_sample.show(truncate=False)

# Stop Spark session
spark.stop()
print("\nSpark Analytics job completed successfully!")