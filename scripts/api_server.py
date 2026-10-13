import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FootInsight API", description="Big Data Football Analytics & Match Predictor")

# Allow frontend requests across local and cloud environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically resolves to the FootInsight root directory on Windows, Linux, and macOS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts", "models")

# 1. Load Datasets
club_df = pd.read_csv(os.path.join(PROCESSED_DIR, "club_matches_top5.csv"))
intl_df = pd.read_csv(os.path.join(PROCESSED_DIR, "international_matches.csv"))
scorers_df = pd.read_csv(os.path.join(PROCESSED_DIR, "intl_goalscorers.csv"))

# --- Dynamically Standardize Column Names for International Data ---
for col in intl_df.columns:
    col_lower = str(col).lower()
    if col_lower in ["home_score", "home_team_goals", "home_goals"]:
        intl_df.rename(columns={col: "home_team_goal"}, inplace=True)
    elif col_lower in ["away_score", "away_team_goals", "away_goals"]:
        intl_df.rename(columns={col: "away_team_goal"}, inplace=True)

if "home_team_goal" not in intl_df.columns:
    intl_df["home_team_goal"] = 0
if "away_team_goal" not in intl_df.columns:
    intl_df["away_team_goal"] = 0

intl_df["home_team_goal"] = pd.to_numeric(intl_df["home_team_goal"], errors="coerce").fillna(0)
intl_df["away_team_goal"] = pd.to_numeric(intl_df["away_team_goal"], errors="coerce").fillna(0)
intl_df["total_goals"] = intl_df["home_team_goal"] + intl_df["away_team_goal"]

if "outcome" not in intl_df.columns:
    intl_df["outcome"] = np.where(
        intl_df["home_team_goal"] > intl_df["away_team_goal"], "HOME_WIN",
        np.where(intl_df["home_team_goal"] < intl_df["away_team_goal"], "AWAY_WIN", "DRAW")
    )

# 2. Load Trained Model & Metadata
model_path = os.path.join(MODELS_DIR, "match_predictor.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]
home_stats = model_data["home_stats"]
away_stats = model_data["away_stats"]
home_conceded = model_data["home_conceded"]
away_conceded = model_data["away_conceded"]
classes = model_data["classes"]

# Compute international team baseline stats
intl_home_stats = intl_df.groupby("home_team")["home_team_goal"].mean().to_dict()
intl_away_stats = intl_df.groupby("away_team")["away_team_goal"].mean().to_dict()
intl_home_conceded = intl_df.groupby("home_team")["away_team_goal"].mean().to_dict()
intl_away_conceded = intl_df.groupby("away_team")["home_team_goal"].mean().to_dict()


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: str = "England Premier League"


@app.get("/")
def root():
    return {"status": "online", "system": "FootInsight Big Data Engine"}


@app.get("/api/leagues")
def get_leagues():
    """Returns Top European Leagues and International Tournaments."""
    club_leagues = sorted(club_df["league"].dropna().unique().tolist())
    intl_tournaments = sorted(intl_df["tournament"].dropna().unique().tolist()) if "tournament" in intl_df.columns else []
    return {
        "leagues": club_leagues + intl_tournaments,
        "club_leagues": club_leagues,
        "international_tournaments": intl_tournaments
    }


@app.get("/api/teams")
def get_teams(league: str = None):
    """Returns teams for a given league or tournament."""
    if not league:
        teams = sorted(set(club_df["home_team"].unique().tolist() + intl_df["home_team"].unique().tolist()))
        return {"teams": teams}

    if "tournament" in intl_df.columns and league in intl_df["tournament"].values:
        sub = intl_df[intl_df["tournament"] == league]
        teams = sorted(set(sub["home_team"].unique().tolist() + sub["away_team"].unique().tolist()))
    else:
        sub = club_df[club_df["league"] == league]
        teams = sorted(set(sub["home_team"].unique().tolist() + sub["away_team"].unique().tolist()))

    return {"teams": teams}


@app.post("/api/predict")
def predict_match(req: PredictionRequest):
    """Predicts match outcome probability for both club and international fixtures."""
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")

    is_intl = False
    if "tournament" in intl_df.columns and req.league in intl_df["tournament"].values:
        is_intl = True
    elif req.home_team in intl_home_stats:
        is_intl = True

    if is_intl:
        h_scored = intl_home_stats.get(req.home_team, 1.40)
        a_scored = intl_away_stats.get(req.away_team, 1.15)
        h_conceded = intl_home_conceded.get(req.home_team, 1.10)
        a_conceded = intl_away_conceded.get(req.away_team, 1.35)
    else:
        h_scored = home_stats.get(req.home_team, 1.25)
        a_scored = away_stats.get(req.away_team, 1.05)
        h_conceded = home_conceded.get(req.home_team, 1.15)
        a_conceded = away_conceded.get(req.away_team, 1.30)

    raw_h = (h_scored / max(a_conceded, 0.5)) * 1.3
    raw_a = (a_scored / max(h_conceded, 0.5)) * 1.0
    raw_d = 0.95
    tot = raw_h + raw_a + raw_d

    row = pd.DataFrame([{
        "market_prob_home": raw_h / tot,
        "market_prob_draw": raw_d / tot,
        "market_prob_away": raw_a / tot,
        "home_team_avg_scored": h_scored,
        "away_team_avg_scored": a_scored,
        "home_team_avg_conceded": h_conceded,
        "away_team_avg_conceded": a_conceded
    }])[features]

    probs = model.predict_proba(row)[0]
    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probs)}
    predicted_outcome = model.predict(row)[0]

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "competition": req.league,
        "predicted_outcome": predicted_outcome,
        "probabilities": {
            "home_win": prob_dict.get("HOME_WIN", 0.0),
            "draw": prob_dict.get("DRAW", 0.0),
            "away_win": prob_dict.get("AWAY_WIN", 0.0)
        }
    }


@app.get("/api/h2h")
def head_to_head(team1: str, team2: str):
    """Returns head-to-head match history across all competitions."""
    matches_club = club_df[
        ((club_df["home_team"] == team1) & (club_df["away_team"] == team2)) |
        ((club_df["home_team"] == team2) & (club_df["away_team"] == team1))
    ]

    matches_intl = intl_df[
        ((intl_df["home_team"] == team1) & (intl_df["away_team"] == team2)) |
        ((intl_df["home_team"] == team2) & (intl_df["away_team"] == team1))
    ]

    matches = matches_club if len(matches_club) > 0 else matches_intl
    matches = matches.sort_values(by="date", ascending=False)

    t1_wins = int(((matches["home_team"] == team1) & (matches["outcome"] == "HOME_WIN")).sum() +
                  ((matches["away_team"] == team1) & (matches["outcome"] == "AWAY_WIN")).sum())
    t2_wins = int(((matches["home_team"] == team2) & (matches["outcome"] == "HOME_WIN")).sum() +
                  ((matches["away_team"] == team2) & (matches["outcome"] == "AWAY_WIN")).sum())
    draws = int((matches["outcome"] == "DRAW").sum())

    cols = ["date", "home_team", "away_team", "home_team_goal", "away_team_goal", "outcome"]
    if "season" in matches.columns:
        cols = ["season"] + cols

    recent = matches.head(8)[cols].to_dict(orient="records")

    return {
        "team1": team1,
        "team2": team2,
        "total_matches": len(matches),
        "team1_wins": t1_wins,
        "team2_wins": t2_wins,
        "draws": draws,
        "recent_matches": recent
    }


@app.get("/api/tournaments")
def get_tournament_stats():
    """Returns international tournament summaries."""
    if "tournament" not in intl_df.columns:
        return {"tournaments": []}
    grouped = intl_df.groupby("tournament")
    records = []
    for name, group in grouped:
        records.append({
            "tournament": str(name),
            "matches_played": int(len(group)),
            "total_goals": int(group["total_goals"].sum()),
            "avg_goals": round(float(group["total_goals"].mean()), 2)
        })
    records.sort(key=lambda x: x["matches_played"], reverse=True)
    return {"tournaments": records}


@app.get("/api/top-scorers")
def top_scorers(limit: int = 15):
    """Returns top tournament goalscorers."""
    top = scorers_df.groupby(["scorer", "team"]).size().reset_index(name="goals")
    top = top.sort_values(by="goals", ascending=False).head(limit)
    return {"top_scorers": top.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FootInsight Unified API Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)