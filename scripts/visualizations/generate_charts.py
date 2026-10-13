import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = r"C:\FootInsight"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load match dataset
df = pd.read_csv(os.path.join(PROCESSED_DIR, "club_matches_top5.csv"))
df['total_goals'] = df['home_team_goal'] + df['away_team_goal']

# 1. League Goal Scoring Distribution Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='league', y='total_goals', data=df, palette='Set2')
plt.title("Goal Distribution Across Top 5 European Leagues", fontsize=14, fontweight='bold')
plt.xlabel("League", fontsize=11)
plt.ylabel("Total Goals per Match", fontsize=11)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "league_goal_distribution.png"), dpi=300)
plt.close()

# 2. Match Outcome Proportions
outcome_counts = df['outcome'].value_counts(normalize=True) * 100
plt.figure(figsize=(7, 5))
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = plt.bar(outcome_counts.index, outcome_counts.values, color=colors, width=0.5)
plt.title("Global Match Outcome Distribution (Home Advantage)", fontsize=13, fontweight='bold')
plt.ylabel("Percentage (%)", fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 60)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "outcome_distribution.png"), dpi=300)
plt.close()

print(f"Figures successfully generated and saved to: {REPORTS_DIR}")