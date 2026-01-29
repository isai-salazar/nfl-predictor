import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def load_config():
    with open("config/paths.yml", "r") as f:
        config = yaml.safe_load(f)
    return config

''' Transform play-by-play data from bronze to silver layer'''
def main():
    spark = (SparkSession.builder
             .appName("transform_pbp_silver")
             .master('local[4]')
             .config('spark.sql.adaptive.enabled', 'true')
             .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
             .getOrCreate())
    
    config = load_config()
    input_path = f"{config['bronze']['pbp']}/play_by_play_2023.csv"
    output_path = config['silver']['pbp']
    
    # Basic cleanup and some aggregations
    # Note: Remember to include home_team, away_team, season_type which are already strings in the csv in gold layer
    print("Reading raw pbp data from bronze layer")
    df = spark.read.option('header', 'true').csv(input_path)
    print(f"Initial dataframe with {df.count()} rows and {len(df.columns)} columns")
    df_clean = (df.withColumn("game_date", to_date(col("game_date"), "yyyy-MM-dd"))
                .withColumn("season", col("season").cast(IntegerType()))
                .withColumn("week", col("week").cast(IntegerType()))
                .filter(col("play_type").isin("pass", "rush", "punt", "field_goal", "extra_point"))
                .withColumn("air_yards", col("air_yards").cast(IntegerType()))
                .withColumn("yards_after_catch", col("yards_after_catch").cast(IntegerType()))
                .withColumn("yards_gained", col("yards_gained").cast(IntegerType()))
                .withColumn("epa", col("epa").cast(DoubleType()))
                .withColumn("total_home_score", col("total_home_score").cast(IntegerType()))
                .withColumn("total_away_score", col("total_away_score").cast(IntegerType()))
                .withColumn("div_game", col("div_game").cast(IntegerType())) # 0/1 as int
                .withColumnRenamed("season_type", "is_playoffs") # 1=playoffs, 0=regular season
                .withColumn("is_playoffs", when(col("is_playoffs") == "REG", 0).otherwise(1))
                .dropDuplicates(["game_id", "play_id"])
                .withColumnRenamed("result", "home_margin") #  spread from home team perspective
                .withColumn("home_margin", col("home_margin").cast(IntegerType()))
                )
    print(f"Dataframe with {df_clean.count()} rows and {len(df_clean.columns)} columns after cleaning")

    # Save 
    # This requires hadoop winutils.exe in PATH to be able to write to local filesystem on Windows
    df_clean.write.mode('overwrite').parquet(f"{output_path}/season=2023")

    spark.stop()

if __name__ == "__main__":
    main()