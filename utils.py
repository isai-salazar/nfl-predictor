import pandas as pd
import joblib
import numpy as np

def load_models():
    """Load 3 XGBoost models"""
    model_win = joblib.load('models/model_win.pkl')
    model_spread = joblib.load('models/model_spread.pkl')
    model_total = joblib.load('models/model_total.pkl')
    return (model_win['model'], model_win['features'], model_spread['model'], model_spread['features'], model_total['model'], model_total['features'])

def get_team_features(home_team, away_team, gold_df, vegas_spread, vegas_total):
    """
    Build REAL inference features using most recent available game
    for each team in the dataset.
    """

    # Sort by date to get latest game
    gold_df = gold_df.sort_values("game_date")

    # Get most recent game where team was home or away
    home_row = gold_df[
        (gold_df["home_team"] == home_team) | (gold_df["away_team"] == home_team)
    ].iloc[-1]

    away_row = gold_df[
        (gold_df["home_team"] == away_team) | (gold_df["away_team"] == away_team)
    ].iloc[-1]

    # Extract feature columns
    home_features = {col: home_row[col] for col in gold_df.columns if col.startswith("home_")}
    away_features = {col: away_row[col] for col in gold_df.columns if col.startswith("away_")}

    # Merge
    features = {}
    features.update(home_features)
    features.update(away_features)

    # Add odds
    features["spread_favorite"] = vegas_spread
    features["over_under_line"] = vegas_total
    features["is_favorite_home"] = 1 if vegas_spread < 0 else 0

    return pd.DataFrame([features])

def betting_recommendation(pred_win_home, pred_spread, pred_total, vegas_spread, vegas_total):
    """Generate betting recommendations based on predictions vs. Vegas odds"""
    recs = []
    
    # Win bet
    if pred_win_home > 0.60:
        recs.append("🏆 BET HOME")
    elif pred_win_home < 0.40:
        recs.append("🏆 BET AWAY")
    
    # Spread
    if pred_spread > vegas_spread + 1.5:
        recs.append(f"📈 BET { '+' if pred_spread > 0 else ''}{int(pred_spread)}")
    elif pred_spread < vegas_spread - 1.5:
        recs.append(f"📉 BET OPPOSITE SPREAD")
    
    # Total
    if pred_total > vegas_total + 2:
        recs.append("🔢 BET OVER")
    elif pred_total < vegas_total - 2:
        recs.append("🔢 BET UNDER")
    
    return recs if recs else ["⚖️ NO CLEAR EDGE"]
