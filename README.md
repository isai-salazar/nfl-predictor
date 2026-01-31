
# NFL Predictor

An end-to-end data engineering and machine learning pipeline for predicting NFL game outcomes. This project combines **data engineering pipelines** (Bronze → Silver → Gold medallion architecture), **feature engineering**, and **machine learning models** to forecast home win probability, expected point spread, and total game points.

> Spanish demo: https://www.loom.com/share/9e5835d7f0074573abe111e1e1dc0c30

## 🎯 Project Overview

This system processes historical NFL play-by-play data and Vegas betting odds through Apache Spark, engineers meaningful features from the data, and trains XGBoost models to predict:

- **Home Win Probability**: Probability that the home team wins outright
- **Expected Point Spread**: Predicted margin of victory (home team perspective)
- **Total Points**: Combined points scored by both teams

### Model Performance (Test Set: 2024-2025 Seasons)

- 🏆 **Home Win Classification**: 61.9% accuracy
- 📊 **Spread Prediction**: 10.6 points MAE (Mean Absolute Error)
- 🔢 **Total Points Prediction**: 11.2 points MAE

<img width="1887" height="898" alt="nfl-predictor" src="https://github.com/user-attachments/assets/8481d008-ecb3-4783-af17-850a1d289195" />

### ⚠️ Disclaimer

> This is an **early-stage prototype** built for portfolio and experimentation purposes. Predictions are **not production-grade** and should **not** be used for real betting decisions.

---

## 📊 Architecture

### Data Pipeline (Medallion Architecture)

```
Bronze Layer (Raw Data)
    ├── play_by_play/ (CSV files 2020-2025)
    └── odds/ (Betting odds from Spreadspoke)
           ↓
Silver Layer (Cleaned & Transformed)
    ├── play_by_play/ (Parquet, per season)
    └── odds/ (Parquet, per season)
           ↓
Gold Layer (Business Ready)
    ├── games/ (Game-level aggregations with features)
    └── plays/ (Enhanced play-level data)
           ↓
ML Models
    ├── model_win.pkl (Home win classifier)
    ├── model_spread.pkl (Point spread regressor)
    └── model_total.pkl (Total points regressor)
           ↓
Streamlit App
    └── Interactive predictions & betting recommendations
```

![nfl_predictor](https://github.com/user-attachments/assets/43c68da0-a991-44a2-8af3-bc9a1892590e)

### Data Sources

- **Play-by-Play Data**: [NFLverse Data](https://github.com/nflverse/nflverse-data/releases) - Comprehensive NFL game data including EPA (Expected Points Added)
- **Betting Odds**: [Kaggle NFL Scores & Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) - Vegas spreads and over/under lines

### Key Features Engineering

The manual pipeline creates game-level features from play-by-play data:

**Offensive Stats (Home & Away Teams)**
- EPA (Expected Points Added) rolling averages (last 4 games & season)
- Pass/rush yards rolling averages
- Points scored rolling averages
- Total plays rolling averages

**Defensive Stats (Home & Away Teams)**
- Defensive EPA rolling averages
- Points allowed rolling averages
- Pass/rush yards allowed rolling averages

**Vegas Odds**
- Spread favorite indicator
- Over/under line
- Is favorite home indicator

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Anaconda/Miniconda** installed
- **Java 11+** (required for Spark)
- **Git**

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/nfl-predictor.git
cd nfl-predictor
```

### Step 2: Create Conda Environment

```bash
conda env create -f nfl-predictor-environment.yml
conda activate nfl-predictor
```

This creates an environment with:
- PySpark 4.0.0
- XGBoost
- Streamlit
- Jupyter/JupyterLab
- All required data science libraries

### Step 3: Verify Java Installation

Spark requires Java 17 or higher

### Step 4: Download Raw Data

Create the `data/bronze/` directory structure and populate it with:

1. **Play-by-Play Data** (2020-2025 seasons)
   - **📥 Download**: [NFLverse Releases - Play by Play Data](https://github.com/nflverse/nflverse-data/releases)
   - Look for files named `pbp_*_*.csv` or similar
   - Place CSV files in `data/bronze/play_by_play/`
   - Files should be named: `play_by_play_2020.csv`, `play_by_play_2021.csv`, etc.

2. **Betting Odds Data**
   - **📥 Download**: [Kaggle - NFL Scores and Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
   - Download `spreadspoke_scores.csv`
   - Place in `data/bronze/odds/spreadspoke_scores.csv`

```
data/
├── bronze/
│   ├── play_by_play/
│   │   ├── play_by_play_2020.csv
│   │   ├── play_by_play_2021.csv
│   │   ├── play_by_play_2022.csv
│   │   ├── play_by_play_2023.csv
│   │   ├── play_by_play_2024.csv
│   │   └── play_by_play_2025.csv
│   └── odds/
│       └── spreadspoke_scores.csv
├── silver/
├── gold/
```

### Step 5: Run ETL Pipeline

The pipeline consists of three stages:

#### 5.1 Transform Play-by-Play to Silver

```bash
python src/etl/transform_pbp_silver.py
```

This script:
- Reads raw play-by-play CSVs from bronze layer
- Cleans and type-casts columns
- Filters for relevant play types (pass, rush, punt, field goal, extra point)
- Outputs to `data/silver/play_by_play/` (partitioned by season)

**Expected output**: Parquet files for each season

#### 5.2 Transform Odds to Silver

```bash
python src/etl/transform_odds_silver.py
```

This script:
- Reads raw betting odds CSV
- Maps full team names to abbreviations
- Converts playoff weeks to numeric format
- Outputs to `data/silver/odds/` (partitioned by season)

**Expected output**: Clean odds parquet files with Vegas lines

#### 5.3 Build Gold Games Dataset

```bash
python src/etl/build_gold_games.py
```

This is the core feature engineering script that:
- Aggregates play-by-play data to game level
- Calculates team offensive and defensive statistics
- Computes rolling averages (last 4 games and season-to-date)
- Joins with Vegas odds
- Outputs to `data/gold/games/` (partitioned by season)

**Expected output**: Game-level features

### Step 6: Train Models (Optional - Pre-trained Models Included)

Pre-trained XGBoost models are included in `models/`:
- `model_win.pkl` 
- `model_spread.pkl`
- `model_total.pkl`

To retrain models, use the Jupyter notebooks:

```bash
jupyter lab notebooks/02-model-xgboost.ipynb
```

The notebook trains three separate XGBoost models:
- Win probability classifier (binary classification)
- Spread predictor (regression)
- Total points predictor (regression)

Models are validated on 2024-2025 seasons and saved with their feature names for proper inference.

### Step 7: Run Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

**Features:**
- Select home and away teams
- Input Vegas spread and over/under
- View model predictions:
  - Home win probability
  - Expected point spread
  - Expected total points
- Get betting recommendations based on market edges

---

## 📁 Project Structure

```
nfl-predictor/
├── app.py                              # Streamlit web application
├── utils.py                            # Model loading & inference utilities
├── requirements.txt                    # Pip dependencies
├── nfl-predictor-environment.yml       # Conda environment specification
├── config/
│   └── paths.yml                       # Data path configuration
├── data/
│   ├── bronze/                         # Raw data (input)
│   │   ├── play_by_play/
│   │   └── odds/
│   ├── silver/                         # Cleaned data (intermediate)
│   │   ├── play_by_play/
│   │   └── odds/
│   └── gold/                           # Feature-engineered data (output)
│       └── games/
├── src/
│   └── etl/
│       ├── transform_pbp_silver.py     # Bronze → Silver for play-by-play
│       ├── transform_odds_silver.py    # Bronze → Silver for odds
│       └── build_gold_games.py         # Silver → Gold (feature engineering)
├── models/
│   ├── model_win.pkl                   # Trained win probability model
│   ├── model_spread.pkl                # Trained spread prediction model
│   └── model_total.pkl                 # Trained total points model
├── notebooks/
│   ├── 01-eda-pbp.ipynb                # Exploratory Data Analysis
│   └── 02-model-xgboost.ipynb          # Model training & validation
└── README.md                           # This file
```

---

## 🛠️ Technology Stack

### Data Engineering
- **Apache Spark 4.0.0**: Distributed data processing
- **PySpark**: Python API for Spark
- **YAML**: Configuration management

### Machine Learning
- **XGBoost**: Gradient boosting models
- **scikit-learn**: Model evaluation metrics
- **joblib**: Model serialization

### Visualization & Web
- **Streamlit**: Interactive web applications
- **Plotly**: Interactive charts
- **Pandas**: Data manipulation

### Development
- **Jupyter/JupyterLab**: Interactive notebooks
- **Python 3.10**: Programming language

---

## 📈 Model Details

### Models Overview

All three models are **XGBoost** and use the same 50+ engineered features:

| Model | Task | Target | Metric | Performance |
|-------|------|--------|--------|-------------|
| `model_win` | Classification | home_win (0/1) | Accuracy | 61.9% |
| `model_spread` | Regression | final_home_margin | MAE | 10.6 pts |
| `model_total` | Regression | final_total_score | MAE | 11.2 pts |

Confussion matrix

<img width="590" height="496" alt="image" src="https://github.com/user-attachments/assets/1fd4daca-2244-4a9c-b3cc-51cc1e753718" />

Spread MAE

<img width="955" height="602" alt="image" src="https://github.com/user-attachments/assets/449893d7-0929-45c4-97b9-ab51886dba27" />

### Feature Input

Model inference requires:
- **Game-level features** (team offensive/defensive stats)
- **Vegas odds** (spread, over/under, favorite indicator)
- **Team season rosters** (implicitly via rolling stats)

Features are constructed from the **most recent available game** for each team in the inference dataset.

### Prediction Workflow

```
Home Team (e.g., KC) → Get most recent game stats → Extract features
Away Team (e.g., BUF) → Get most recent game stats → Extract features
Vegas Odds (spread, total) → Add market context

Combine features → [Multi-dimensional vector]
                  ↓
              XGBoost Models
                  ↓
         [Win Prob, Spread, Total]
                  ↓
         Generate Betting Recommendations
```

### Betting Recommendations Logic

The app compares model predictions vs. Vegas lines to identify market edges:

- **Win Edge**: If predicted home win prob > 60%, recommend betting home
- **Spread Edge**: If predicted spread differs from Vegas by >1.5 pts, flag opportunity
- **Total Edge**: If predicted total differs from Vegas by >2 pts, flag opportunity

---

## 🐛 Troubleshooting

### Issue: "Java not found" error when running ETL scripts

**Solution:**
```bash
conda install -c conda-forge openjdk=11
```

### Issue: "Path does not exist" error for data/

**Solution:**
Create the directory structure:
```bash
mkdir -p data/bronze/play_by_play
mkdir -p data/bronze/odds
mkdir -p data/silver/play_by_play
mkdir -p data/silver/odds
mkdir -p data/gold/games
```

### Issue: Streamlit app crashes with "models not found"

**Solution:**
Ensure the `models/` directory contains the three `.pkl` files:
- `model_win.pkl`
- `model_spread.pkl`
- `model_total.pkl`

If missing, run the model training notebook: `notebooks/02-model-xgboost.ipynb`

### Issue: ETL scripts run very slowly or hang

**Solution:**
This typically indicates memory issues with Spark. Adjust Spark config in ETL scripts:
```python
spark = (SparkSession.builder
         .appName("transform_pbp_silver")
         .master('local[2]')  # Reduce from 4 to 2 cores
         .config('spark.driver.memory', '4g')  # Reduce memory
         .getOrCreate())
```

---

## 👤 Author

Built by Diego Isaí Salazar Rico

- **GitHub**: [isai-salazar](https://github.com/isai-salazar)

## 📜 License

This project is provided as-is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- **NFLverse** for comprehensive play-by-play data
- **Kaggle** community for betting odds dataset
- **Apache Spark**, **XGBoost**, and **Streamlit** communities
