FootInsight: Comprehensive Technical Report & Architecture ReferencePlatform Domain: Big Data Sports Analytics, Distributed Data Engineering & Predictive ModelingLive Production URL: https://foot-insight.vercel.app/Production API Base: https://footinsight-api.onrender.com/Interactive API Documentation: https://footinsight-api.onrender.com/docsSource Repository: https://github.com/ashlin-theres/FootInsight1. Executive Summary & Problem Formulation1.1 Academic MotivationModern association football generates millions of transactional data points spanning match results, in-game metrics, betting market quotes, and tournament rosters. However, conventional statistical analysis and monolithic relational databases struggle to process high-dimensional match records across disparate competitions without performance bottlenecks.1.2 Core Challenges AddressedTemporal Concept Drift & Non-Stationary Dynamics:
Historical sports performance data is inherently non-stationary. A football club's dominance eight seasons ago possesses weak predictive power for next week's fixture due to squad rotation, managerial changes, and tactical evolution. Monolithic models trained indiscriminately on historical logs suffer severe accuracy degradation.Class Imbalance & Inherently Low Margin Outcomes:
Association football exhibits high stochastic variance with low average scoring margins ($\approx 2.6$ to $2.8$ goals per match across Europe's top leagues). Class distributions across three outcomes—Home Win ($H$), Draw ($D$), Away Win ($A$)—are naturally skewed, with home advantage causing $H$ to dominate ($\sim 45\text{--}46\%$), while $D$ occupies $\sim 25\text{--}26\%$, and $A$ occupies $\sim 28\text{--}29\%$.Disparate Heterogeneous Schemas:
Club league competitions (season-long round-robin fixtures with odds data) and international tournaments (knockout and group-stage formats without bookmaker lines) operate under radically divergent relational structures.1.3 FootInsight Solution ArchitectureFootInsight delivers an end-to-end, production-grade Big Data analytics and predictive engine. The platform combines:Distributed batch ETL pipelines utilizing Apache Spark (PySpark SQL), Apache Hive, and Apache Pig paradigms.An efficient-market calibrated Multinomial Logistic Regression Classifier achieving $53.34\%$ test accuracy on out-of-sample matches, outperforming both random guess ($33.33\%$) and historical home bias baselines ($45.8\%$).A low-latency asynchronous FastAPI microservice containerized on Render.An edge-rendered responsive client dashboard hosted on Vercel.2. End-to-End System Architecture                       [ RAW DATA LAKE / INGESTION ]
          ┌──────────────────────────────────┴──────────────────────────────────┐
          ▼                                                                     ▼
[ European Club Data (SQLite DB) ]                           [ International Match Records (CSV) ]
  • Matches, Leagues, Teams                                    • 40,000+ International Fixtures
  • Bookmaker Odds (B365, BW, IW, LB)                          • Goalscorer Timestamps & Tournaments
          │                                                                     │
          └──────────────────────────────────┬──────────────────────────────────┘
                                             ▼
                             [ DATA ENGINEERING & ETL CORE ]
                                scripts/cleaning/prepare_datasets.py
          • Schema-on-Read Normalization & Type Enforcement
          • Pruning Incomplete Betting Odds Records
          • Exponential Rolling Window & Goal Aggregate Computation
                                             │
                                             ▼
                             [ DISTRIBUTED BIG DATA LAYER ]
                             scripts/spark/run_pyspark_analytics.py
          • Apache Spark DataFrame Optimizations (Lazy Evaluation)
          • Spark SQL Grouped Aggregations & Window Functions
          • Hive & Pig Relational Schemas for Warehouse Warehousing
                                             │
                                             ▼
                            [ MACHINE LEARNING PIPELINE ]
                             scripts/models/train_predictor.py
          • Feature Engineering: Market-Implied Probability Extraction
          • Multinomial Logistic Regression Model Training
          • Model & Metadata Serialization -> scripts/models/match_predictor.pkl
                                             │
                                             ▼
                             [ BACKEND MICROSERVICE LAYER ]
                                 scripts/api_server.py
          • FastAPI & Uvicorn Async Server
          • Cross-Origin Resource Sharing (CORS) Configuration
          • Endpoints: /api/predict, /api/h2h, /api/leagues, /api/tournaments
          • Deployed on Cloud Linux Container: Render.com
                                             │
                                             ▼
                             [ FRONTEND SERVING LAYER ]
                                dashboard.html / index.html
          • Single-Page React (ES6+) with Tailwind CSS
          • Dynamic Schema Adapters (Club vs. International)
          • Edge Deployment: Vercel CDN
3. Big Data Engineering & Technologies Used3.1 Apache Spark & PySpark SQL (scripts/spark/run_pyspark_analytics.py)Apache Spark functions as the distributed data processing engine within FootInsight:Partitioned Transformations: Match tables are partitioned logically across league and season dimensions, distributing aggregations for home/away scoring ratios.Spark SQL Catalysts: Computes multi-season goal distributions, club clean-sheet percentages, and cumulative tournament leaderboards using in-memory Spark DataFrame operations.Lazy Evaluation Execution Plan: Optimizes filtering and projection steps before triggering actions (count(), collect(), write.csv()), preventing memory saturation.3.2 Apache Hive Data Warehousing Metastore DesignTo represent FootInsight inside an enterprise Hadoop/Hive warehouse, the raw tables are organized with partitioned external schemas:-- European Club Matches Partitioned by Season
CREATE EXTERNAL TABLE IF NOT EXISTS footinsight_dw.club_matches (
    match_id INT,
    country_id INT,
    league_name STRING,
    match_date STRING,
    home_team_id INT,
    away_team_id INT,
    home_team_goal INT,
    away_team_goal INT,
    b365_home DOUBLE,
    b365_draw DOUBLE,
    b365_away DOUBLE
)
PARTITIONED BY (season STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS PARQUET
LOCATION '/warehouse/footinsight/club_matches/';

-- International Tournament Schema
CREATE EXTERNAL TABLE IF NOT EXISTS footinsight_dw.international_fixtures (
    fixture_date STRING,
    home_team STRING,
    away_team STRING,
    home_score INT,
    away_score INT,
    tournament_name STRING,
    city STRING,
    country STRING,
    neutral_ground BOOLEAN
)
STORED AS ORC
LOCATION '/warehouse/footinsight/international_matches/';
3.3 Apache Pig Latin ETL ScriptFor batch ingestion, schema extraction, and filtering of raw fixture dumps:-- Load raw fixture data using PigStorage
raw_fixtures = LOAD '/data/raw/match_logs.csv' USING PigStorage(',') AS (
    match_date:chararray,
    league:chararray,
    home_team:chararray,
    away_team:chararray,
    home_goals:int,
    away_goals:int,
    odd_h:double,
    odd_d:double,
    odd_a:double
);

-- Prune records with missing goal data or corrupted odds
cleaned_fixtures = FILTER raw_fixtures BY (home_goals IS NOT NULL) 
                   AND (away_goals IS NOT NULL) 
                   AND (odd_h > 1.0) AND (odd_d > 1.0) AND (odd_a > 1.0);

-- Compute home team goal sums by league
grouped_by_league = GROUP cleaned_fixtures BY league;
league_goal_stats = FOREACH grouped_by_league GENERATE 
    group AS league_name,
    COUNT(cleaned_fixtures) AS total_matches,
    SUM(cleaned_fixtures.home_goals) AS total_home_goals,
    AVG(cleaned_fixtures.home_goals) AS avg_home_goals;

-- Store to HDFS storage
STORE league_goal_stats INTO '/data/analytics/league_stats' USING PigStorage('\t');
4. Machine Learning & Predictive Modeling4.1 Feature FormulationPredicting the discrete outcome $Y \in \{\text{HOME\_WIN}, \text{DRAW}, \text{AWAY\_WIN}\}$ relies on an engineered feature vector $\mathbf{x} \in \mathbb{R}^7$:$$\mathbf{x} = \begin{bmatrix} P_{\text{market\_H}} \\ P_{\text{market\_D}} \\ P_{\text{market\_A}} \\ \bar{G}_{\text{scored\_H}} \\ \bar{G}_{\text{scored\_A}} \\ \bar{G}_{\text{conceded\_H}} \\ \bar{G}_{\text{conceded\_A}} \end{bmatrix}$$Where:$P_{\text{market\_H}}, P_{\text{market\_D}}, P_{\text{market\_A}}$ represent normalized closing market probabilities extracted from closing decimal odds $(O_H, O_D, O_A)$:
$$P'_{\text{outcome}} = \frac{1}{O_{\text{outcome}}}, \quad S = \sum_{k \in \{H, D, A\}} \frac{1}{O_k}, \quad P_{\text{market\_outcome}} = \frac{P'_{\text{outcome}}}{S}$$
This normalizes out the bookmaker's overround margin $S - 1$.$\bar{G}_{\text{scored\_H}}$ and $\bar{G}_{\text{conceded\_H}}$ denote the historical average goals scored and conceded by the home club when playing at home.$\bar{G}_{\text{scored\_A}}$ and $\bar{G}_{\text{conceded\_A}}$ denote the historical average goals scored and conceded by the away club on the road.4.2 Algorithm Selection: Multinomial Logistic RegressionFootball match outcomes exhibit subtle statistical margins. Deep networks with numerous non-linear layers easily overfit historical training samples and produce uncalibrated probabilities.FootInsight implements a Multinomial Logistic Regression Classifier with Softmax Link:$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j \in \{H, D, A\}} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$Loss Function: Categorical Cross-Entropy (Log-Loss) with $L_2$ regularization penalty:
$$\mathcal{L}(\mathbf{W}) = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(P(Y = k \mid \mathbf{x}_i)) + \frac{\lambda}{2}\sum_{k=1}^K \Vert{}\mathbf{w}_k\Vert{}_2^2$$Benchmark Results:Test Accuracy: $53.34\%$Baseline (Random Guess): $33.33\%$Baseline (Always Home Win): $45.80\%$Log Loss: $\approx 0.98$ (calibrated against market efficiency)5. File & Directory BreakdownFootInsight/
├── data/
│   ├── processed/
│   │   ├── club_matches_top5.csv       # Cleaned European club match records
│   │   ├── international_matches.csv   # Normalized international tournament logs
│   │   └── intl_goalscorers.csv        # Goalscorer logs with tournament tags
│   └── raw/                            # Raw data (.gitignored to avoid quota issues)
├── scripts/
│   ├── api_server.py                   # FastAPI backend server
│   ├── cleaning/
│   │   └── prepare_datasets.py         # ETL, schema cleanup, and feature extraction
│   ├── models/
│   │   ├── train_predictor.py          # Scikit-learn model training and evaluation
│   │   └── match_predictor.pkl         # Serialized model, features, and team stats
│   ├── spark/
│   │   └── run_pyspark_analytics.py    # PySpark SQL distributed aggregation jobs
│   └── visualizations/
│       └── generate_charts.py          # Analytical figure generation for presentations
├── dashboard.html                      # Single-page React dashboard interface
├── index.html                          # Root entry point configured for Vercel
├── requirements.txt                    # Python cloud dependencies
├── .gitignore                          # Excludes raw database.sqlite, caches, bytecode
├── README.md                           # GitHub project homepage documentation
└── SYSTEM_DOCUMENTATION.md             # Complete technical and architectural reference
6. API Endpoint SpecificationsThe backend service runs via FastAPI (scripts/api_server.py).6.1 GET /api/leaguesReturns all registered competitions across club and international domains.Response Sample:{
  "leagues": ["England Premier League", "Spain LIGA BBVA", "FIFA World Cup", "..."],
  "club_leagues": ["England Premier League", "Germany 1. Bundesliga", "..."],
  "international_tournaments": ["FIFA World Cup", "UEFA Euro", "Copa América"]
}
6.2 GET /api/teams?league={name}Returns all unique participating teams for the selected tournament or league.6.3 POST /api/predictCalculates win/draw/loss probabilities for two teams.Request Payload:{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "league": "England Premier League"
}
Response Payload:{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "competition": "England Premier League",
  "predicted_outcome": "HOME_WIN",
  "probabilities": {
    "home_win": 51.42,
    "draw": 26.15,
    "away_win": 22.43
  }
}
6.4 GET /api/h2h?team1={team1}&team2={team2}Returns cumulative head-to-head records and chronological fixture histories between two teams.7. Cloud Deployment & Production Operations7.1 Backend Microservice (Render)Runtime: Python 3 container running on Linux.Web Server: Uvicorn ASGI server binding to dynamically assigned environment port $PORT:uvicorn scripts.api_server:app --host 0.0.0.0 --port $PORT
Cross-Origin Configuration: CORSMiddleware configured with permissive origins to allow incoming browser requests from the Vercel edge domain.7.2 Frontend Application (Vercel)Architecture: Static Single-Page Application (SPA) driven by React via CDN, Tailwind CSS, and asynchronous fetch workers.Edge CDN Distribution: Global distribution via Vercel Edge Network, routing API calls to https://footinsight-api.onrender.com.8. Continuous Data Ingestion ArchitectureFootInsight addresses non-stationary data drift through an incremental ingestion pipeline:[ REST API / Match Feed ] ──► [ Ingestion Worker ] ──► [ Incremental Lakehouse Append ]
                                                                   │
                                                                   ▼
[ Live Dashboard ] ◄── [ API Cache Invalidation ] ◄── [ PySpark Rolling Window Batch Job ]
Ingestion Worker: Fetches newly completed match records via scheduled cron triggers (e.g., Football-Data.org API).Deduplication: Appends new records using unique (date, home_team, away_team) compound keys.Rolling Averages: Updates exponential time-decay scoring rates ($\lambda = 0.95$).Zero-Downtime Serving: The FastAPI layer re-reads updated team statistics without requiring code redeployment.