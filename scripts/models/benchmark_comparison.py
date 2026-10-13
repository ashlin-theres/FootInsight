import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = r"C:\FootInsight"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, "club_matches_top5.csv"))
if 'outcome' not in df.columns:
    df['outcome'] = np.where(
        df['home_team_goal'] > df['away_team_goal'], 'HOME_WIN',
        np.where(df['home_team_goal'] < df['away_team_goal'], 'AWAY_WIN', 'DRAW')
    )

df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away']).copy()

raw_h = 1.0 / df['odds_home']
raw_d = 1.0 / df['odds_draw']
raw_a = 1.0 / df['odds_away']
tot = raw_h + raw_d + raw_a
df['market_prob_home'] = raw_h / tot
df['market_prob_draw'] = raw_d / tot
df['market_prob_away'] = raw_a / tot

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
}

names = []
accuracies = []
losses = []

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    probs = m.predict_proba(X_test)
    names.append(name)
    accuracies.append(accuracy_score(y_test, preds) * 100)
    losses.append(log_loss(y_test, probs))

# Plot Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
bars1 = ax1.bar(names, accuracies, color=['#2b5c8f', '#d95f02', '#7570b3'], width=0.5)
ax1.set_title("Model Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_ylim(45, 60)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

# Log Loss plot
bars2 = ax2.bar(names, losses, color=['#2b5c8f', '#d95f02', '#7570b3'], width=0.5)
ax2.set_title("Log Loss (Lower is Better Calibration)", fontsize=12, fontweight='bold')
ax2.set_ylim(0.95, 0.99)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(REPORTS_DIR, "model_benchmark_comparison.png")
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"Benchmark chart saved to: {chart_path}")
