import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def load_config():
    with open("config/paths.yml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():

    config = load_config()
    team_mappings = {
        "San Francisco 49ers": "SF",
        "Tennessee Titans": "TEN",
        "Houston Texans": "HOU",
        "Jacksonville Jaguars": "JAX",
        "Arizona Cardinals": "ARI",
        "Tampa Bay Buccaneers": "TB",
        "New England Patriots": "NE",
        "Kansas City Chiefs": "KC",
        "Buffalo Bills": "BUF",
        "Indianapolis Colts": "IND",
        "Philadelphia Eagles": "PHI",
        "Cleveland Browns": "CLE",
        "New York Jets": "NYJ",
        "Chicago Bears": "CHI",
        "Miami Dolphins": "MIA",
        "Atlanta Falcons": "ATL",
        "Green Bay Packers": "GB",
        "Carolina Panthers": "CAR",
        "Denver Broncos": "DEN",
        "Dallas Cowboys": "DAL", 
       	"Los Angeles Chargers":"LAC", 
        "Las Vegas Raiders":"LV", 
        "New Orleans Saints":"NO", 
        "Detroit Lions":"DET", 
        "Los Angeles Rams":"LA", 
        "Seattle Seahawks":"SEA", 
        "Baltimore Ravens":"BAL", 
        "Pittsburgh Steelers":"PIT", 
        "Minnesota Vikings":"MIN", 
        "Cincinnati Bengals":"CIN", 
        "New York Giants":"NYG", 
        "Washington Commanders":"WAS", 
        "Washington Football Team":"WAS"
    }

    spark = (SparkSession.builder
        .appName("transform_odds_silver")
        .master('local[4]')
        .getOrCreate())

    # https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
    input_path = f"{config['bronze']['odds']}/spreadspoke_scores.csv"
    output_path = config['silver']['odds']

    df_odds = (spark.read.option("header", "true").option("inferSchema", "true").csv(input_path))
    # Map team names to abbreviations
    df_mapped = (df_odds.filter(col("schedule_season") > 2019)
                .replace(team_mappings, subset=["team_home", "team_away"])
                .withColumnRenamed("schedule_season", "season")
                .withColumnRenamed("schedule_week", "week")
                .withColumnRenamed("team_home", "home_team")
                .withColumnRenamed("team_away", "away_team")
                .select(
                    "season", 
                    when(col("week") == "Wildcard", 18)
                        .otherwise(
                            when(col("week") == "Division", 19)
                            .otherwise(when(col("week") == "Conference", 20)
                                .otherwise(when(col("week") == "Superbowl", 21)
                                    .otherwise(col("week").cast(IntegerType()))
                                )
                            )
                        ).alias("week"), 
                    "home_team", 
                    "away_team",
                    when(col("team_favorite_id") == col("home_team"), 1).otherwise(0).alias("is_favorite_home"),
                    "spread_favorite", 
                    "over_under_line"
                    )
                )
    
    df_mapped.write.mode('overwrite').partitionBy("season").parquet(output_path)
    spark.stop()

if __name__ == "__main__":
    main()