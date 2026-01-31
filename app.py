import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_models, get_team_features, betting_recommendation

TEAM_LOGOS = {
    "ARI": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "BUF": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "CAR": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "GB": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LAR": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "LV": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "SF": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "TB": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nfl/500/was.png",
}


# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="NFL Predictor (Alpha)",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Header
# ------------------------
st.title("NFL Predictor (Alpha)")
st.caption("⚠️ This is a very early prototype built for experimentation and portfolio purposes. "
    "Predictions are not production-grade and should not be used for real betting decisions. ⚠️")

st.markdown(
    """
    End-to-end NFL analytics system combining **data engineering pipelines**, 
    **feature engineering**, and **machine learning models** to forecast 
    win probability, expected home-margin, and total points.

    **Model performance** -> **Test Set (2024-2025 seasons)**:
    - 🏆 Home Win: **61.9%** accuracy
    - 📊 Spread: **10.6 pts** MAE  
    - 🔢 Total: **11.2 pts** MAE
    """
)

# ------------------------
# Sidebar - game setup
# ------------------------
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]
st.sidebar.header("Game Setup")
home_team = st.sidebar.selectbox("🏠 Home Team", NFL_TEAMS)
away_options = [t for t in NFL_TEAMS if t != home_team] # Exclude home team
away_team = st.sidebar.selectbox("✈️ Away Team", away_options)
vegas_spread = st.sidebar.slider(
    "📊 Vegas Spread (Home team)", 
    min_value=-20.5, max_value=20.5, value=-3.5, step=0.5
)

# Total POINTS
vegas_total = st.sidebar.slider(
    "🔢 Vegas Over/Under", 
    min_value=30.5, max_value=65.5, value=47.5, step=0.5
)

# ------------------------
# Load models and data
# ------------------------
@st.cache_resource 
def load_all_models():
    model_win, feat_names_win, model_spread, feat_names_spread, model_total, feat_names_total = load_models()
    return model_win, feat_names_win, model_spread, feat_names_spread, model_total, feat_names_total

model_win, feat_names_win, model_spread, feat_names_spread, model_total, feat_names_total = load_all_models()

@st.cache_data
def load_gold_data():
    return pd.read_parquet("data/gold/games/")

gold_df = load_gold_data()

# ------------------------
# Preiction action
# ------------------------
if st.sidebar.button("🚀 PREDICT MATCHUP", type="primary", key="predict_btn"):
    with st.spinner("Calculating predictions..."):
        # Matchup features, in a real scenario these would be dynamic, retrieved from a DB or API. In this case we use the last recorded stats for each selected team.
        X_pred = get_team_features(home_team, away_team, gold_df, vegas_spread, vegas_total)
        X_pred = X_pred[feat_names_win] # Ensure correct feature order for win model
        
        # 3 PREDICTIONS
        prob_home_win = model_win.predict_proba(X_pred)[0, 1]
        pred_spread = model_spread.predict(X_pred)[0]
        pred_total = model_total.predict(X_pred)[0]
        
        recs = betting_recommendation(prob_home_win, pred_spread, pred_total, vegas_spread, vegas_total)
    
    # ------------------------
    # Main results
    # ------------------------
    st.divider()
    st.markdown("### Matchup Forecast")
    st.markdown(f"""
    <div style="
        display:flex;
        align-items:center;
        gap:10px;
        margin-bottom:20px;
    ">
        <div style="display:flex; flex-direction:column; align-items:center;">
            <img src="{TEAM_LOGOS[home_team]}" width="100"/>
            <strong>{home_team}</strong>
        </div>
        <span style="margin:0 6px; opacity:0.6;">vs</span>
        <div style="display:flex; flex-direction:column; align-items:center;">
            <img src="{TEAM_LOGOS[away_team]}" width="100"/>
            <strong>{away_team}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Home win probability",
            f"{prob_home_win:.1%}",
            help="Probability that the home team wins the game outright."
        )

    with col2:
        st.metric(
            "Expected home margin",
            f"{pred_spread:+.1f}",
            help="Predicted final score margin from the home team's perspective."
        )

    with col3:
        st.metric(
            "Expected total points",
            f"{pred_total:.1f}",
            help="Predicted combined points scored by both teams."
        )

    st.caption("Positive margin → home team favored" )
    st.caption("Negative margin → away team favored")

    st.divider()
    
     # ------------------------
    # Betting recommendations
    # ------------------------
    st.subheader("Market Edge Signals")

    if recs:
        for rec in recs:
            st.text(rec)
    else:
        st.text("No significant edge detected against current market lines.")


st.markdown("---")

st.caption("Built by Diego Isaí Salazar Rico | GitHub: [isai-salazar](https://https://github.com/isai-salazar)")
st.caption("Powered by EPA rolling stats + Vegas odds | Built with XGBoost")
st.markdown("""
<div style="font-size:0.8rem;color:#777;text-align:left;">
    Data sources: 
    <a href="https://github.com/nflverse/nflverse-data/releases" target="_blank">NFLverse Data</a> · 
    <a href="https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data" target="_blank">Betting Odds</a>
</div>
""", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display:flex; margin-top:12px">
        <a href="https://github.com/isai-salazar/nfl-predictor" target="_blank"
           style="padding:6px 12px; border-radius:16px; background:#f1f3f6; 
                  text-decoration:none; color:#333; font-size:13px;">
           🔗 Github repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)