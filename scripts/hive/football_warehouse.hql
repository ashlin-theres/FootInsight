-- FootInsight: Apache Hive Data Warehouse Schema & Analytics

CREATE DATABASE IF NOT EXISTS footinsight_db;
USE footinsight_db;

-- 1. Create External Hive Table over HDFS storage
CREATE EXTERNAL TABLE IF NOT EXISTS club_matches (
    match_id INT,
    country STRING,
    league STRING,
    season STRING,
    stage INT,
    match_date STRING,
    home_team STRING,
    away_team STRING,
    home_team_goal INT,
    away_team_goal INT,
    odds_home FLOAT,
    odds_draw FLOAT,
    odds_away FLOAT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/footinsight/warehouse/club_matches'
TBLPROPERTIES ("skip.header.line.count"="1");

-- 2. Partitioned Analytical Table (Partition by League)
CREATE TABLE IF NOT EXISTS partitioned_league_matches (
    match_id INT,
    season STRING,
    home_team STRING,
    away_team STRING,
    home_team_goal INT,
    away_team_goal INT
)
PARTITIONED BY (league_name STRING)
STORED AS ORC;

-- 3. Analytical Query: Team Win Percentage and Goal Ratio
SELECT 
    home_team,
    COUNT(*) AS total_home_games,
    ROUND(SUM(CASE WHEN home_team_goal > away_team_goal THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS home_win_pct,
    ROUND(AVG(home_team_goal), 2) AS avg_goals_scored
FROM club_matches
GROUP BY home_team
HAVING total_home_games >= 50
ORDER BY home_win_pct DESC
LIMIT 10;