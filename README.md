FootInsight: Big Data Football Analytics & Predictive Engine

[![Frontend Deployment](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://foot-insight.vercel.app/)
[![API Backend](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://footinsight-api.onrender.com/docs)
[![Python Engine](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#)

> **FootInsight** is an end-to-end Big Data sports analytics platform that ingests, cleans, processes, and serves historical and international football telemetry. It leverages distributed query principles (Apache Spark SQL / Hive / Pig) and an efficient-market calibrated machine learning classifier to deliver sub-millisecond match outcome probabilities and head-to-head analytics.

🌐 **Live Application:** [https://foot-insight.vercel.app/](https://foot-insight.vercel.app/)  
📖 **Interactive API Documentation:** [https://footinsight-api.onrender.com/docs](https://footinsight-api.onrender.com/docs)

---

## 📌 Architectural Overview

FootInsight bridges distributed data engineering with modern web deployment. It handles both domestic European club competitions (Top 5 leagues) and major international tournaments (FIFA World Cup, UEFA Euro, Copa América).

                  [ Raw Data Sources ]
       (European Club SQLite | International Matches CSV)
                           │
                           ▼
               [ Data Engineering Layer ]
     • Pruning Null Betting Odds & Redundant Columns
     • Schema-on-Read Normalization & Feature Derivation
                           │
                           ▼
               [ Big Data Analytics Core ]
     • PySpark SQL Distributed Aggregations & Windowing
     • Apache Pig / Hive Schema Definitions
                           │
                           ▼
               [ Machine Learning Pipeline ]
     • Feature Engineering: Market-Implied Closing Probabilities
     • Multinomial Logistic Regression Classifier (53.34% Benchmark)
                           │
                           ▼
               [ Serving & Cloud Deployment ]
     ┌─────────────────────┴─────────────────────┐
     ▼                                           ▼
[ Render Cloud Service ]                   [ Vercel Edge Host ]
FastAPI + Uvicorn + Scikit-learn           React + Tailwind CSS
Endpoints: /predict, /h2h, /leagues        Client Dashboard UI


---

## 🚀 Key Features

* **Calibrated Outcome Predictor:** Employs a Multinomial Logistic Regression model trained on market-implied probabilities, scoring averages, and defensive concession rates to mitigate temporal concept drift.
* **Dual-Domain Analytics:** Dynamically routes queries between European league clubs and international national squads with automated schema mapping.
* **Deep Head-to-Head (H2H) Engine:** On-demand historical rivalry breakdowns with win-loss-draw distributions and chronological fixture logs.
* **Interactive Responsive Interface:** React-driven dashboard styled with Tailwind CSS, supporting one-click predictions, league filters, and tournament goalscorer tracking.
* **Decoupled Cloud Microservices:** Frontend statically served across edge CDNs (Vercel) communicating asynchronously via CORS with a containerized Python backend (Render).

---

## 🛠️ Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Frontend** | React, Tailwind CSS, Lucide Icons, HTML5, JavaScript (ES6+) |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python 3 |
| **Data & ML** | Scikit-learn, Pandas, NumPy, Pickle |
| **Big Data Design** | Apache Spark (PySpark SQL), Apache Hive, Apache Pig concepts |
| **Cloud Infrastructure** | Render (API container), Vercel (Frontend hosting), GitHub (CI/CD) |

---

## 📁 Repository Structure

```text
FootInsight/
├── data/
│   ├── processed/               # Cleaned, model-ready CSV datasets
│   │   ├── club_matches_top5.csv
│   │   ├── international_matches.csv
│   │   └── intl_goalscorers.csv
│   └── raw/                     # Raw datasets (.gitignored)
├── scripts/
│   ├── api_server.py            # Production FastAPI application
│   ├── cleaning/
│   │   └── prepare_datasets.py  # ETL & normalization pipeline
│   ├── models/
│   │   ├── train_predictor.py   # Scikit-learn training script
│   │   └── match_predictor.pkl  # Serialized model & metadata
│   ├── spark/
│   │   └── run_pyspark_analytics.py # Spark SQL analytics jobs
│   └── visualizations/
│       └── generate_charts.py   # High-resolution report plotting
├── dashboard.html               # Source single-page dashboard
├── index.html                   # Production-configured Vercel entrypoint
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md