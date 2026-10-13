import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("[1/5] Loading processed match data...")
PROJECT_ROOT = r"C:\FootInsight"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

csv_path = os.path.join(PROCESSED_DIR, "club_matches_top5.csv")
df = pd.read_csv(csv_path)

# Derive outcome if missing
if 'outcome' not in df.columns:
    df['outcome'] = np.where(
        df['home_team_goal'] > df['away_team_goal'], 'HOME_WIN',
        np.where(df['home_team_goal'] < df['away_team_goal'], 'AWAY_WIN', 'DRAW')
    )

# Filter rows with available odds
df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away']).copy()

print(f"[2/5] Engineering features for {len(df)} matches...")
raw_h = 1.0 / df['odds_home']
raw_d = 1.0 / df['odds_draw']
raw_a = 1.0 / df['odds_away']
total_prob = raw_h + raw_d + raw_a

df['market_prob_home'] = raw_h / total_prob
df['market_prob_draw'] = raw_d / total_prob
df['market_prob_away'] = raw_a / total_prob

home_stats = df.groupby('home_team')['home_team_goal'].mean().to_dict()
away_stats = df.groupby('away_team')['away_team_goal'].mean().to_dict()
home_conceded = df.groupby('home_team')['away_team_goal'].mean().to_dict()
away_conceded = df.groupby('away_team')['home_team_goal'].mean().to_dict()

df['home_team_avg_scored'] = df['home_team'].map(home_stats).fillna(1.2)
df['away_team_avg_scored'] = df['away_team'].map(away_stats).fillna(1.0)
df['home_team_avg_conceded'] = df['home_team'].map(home_conceded).fillna(1.1)
df['away_team_avg_conceded'] = df['away_team'].map(away_conceded).fillna(1.3)

features = [
    'market_prob_home', 'market_prob_draw', 'market_prob_away',
    'home_team_avg_scored', 'away_team_avg_scored',
    'home_team_avg_conceded', 'away_team_avg_conceded'
]

X = df[features]
y = df['outcome']

print("[3/5] Splitting into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[4/5] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("\n==========================================")
print(f" Model Accuracy: {acc * 100:.2f}%")
print("==========================================")
print(classification_report(y_test, model.predict(X_test)))

print("[5/5] Saving model artifact...")
metadata = {
    'model': model,
    'features': features,
    'home_stats': home_stats,
    'away_stats': away_stats,
    'home_conceded': home_conceded,
    'away_conceded': away_conceded,
    'classes': list(model.classes_)
}

model_file = os.path.join(MODELS_DIR, "match_predictor.pkl")
with open(model_file, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Saved to: {model_file}")
print("Training complete!")