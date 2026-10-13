-- FootInsight: Apache Pig Latin ETL Script
-- Demonstrates MapReduce style filtering, projection, and grouping

-- 1. Load Cleaned Match Records
raw_matches = LOAD '/footinsight/data/club_matches_top5.csv' USING PigStorage(',') 
    AS (match_id:int, country:chararray, league:chararray, season:chararray, 
        stage:int, match_date:chararray, home_team:chararray, away_team:chararray, 
        home_goal:int, away_goal:int, odds_h:float, odds_d:float, odds_a:float);

-- 2. Filter Header Record
filtered_matches = FILTER raw_matches BY match_id IS NOT NULL;

-- 3. Compute Derived Columns
projected_data = FOREACH filtered_matches GENERATE 
    league, 
    home_team, 
    away_team, 
    (home_goal + away_goal) AS total_goals,
    (CASE 
        WHEN home_goal > away_goal THEN 'HOME_WIN'
        WHEN home_goal < away_goal THEN 'AWAY_WIN'
        ELSE 'DRAW'
     END) AS outcome;

-- 4. Group by League for Aggregation
grouped_by_league = GROUP projected_data BY league;

-- 5. Calculate Average Goals per Match
league_stats = FOREACH grouped_by_league GENERATE 
    group AS league_name, 
    COUNT(projected_data) AS total_matches, 
    AVG(projected_data.total_goals) AS avg_goals;

-- 6. Store Analytical Output
STORE league_stats INTO '/footinsight/output/pig_league_stats' USING PigStorage('\t');