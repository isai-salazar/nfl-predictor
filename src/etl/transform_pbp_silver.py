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
    df = spark.read.option('header', 'true').csv(input_path)
    print(f"Dataframe with {df.count()} rows and {len(df.columns)} columns\n")
    df_clean = (df.withColumn("game_date", to_date(col("game_date"), "yyyy-MM-dd"))
                .withColumn("season", col("season").cast(IntegerType()))
                .withColumn("week", col("week").cast(IntegerType()))
                .withColumn("yards_gained", col("yards_gained").cast(IntegerType()))
                .withColumn("epa", col("epa").cast(DoubleType()))
                .withColumn("total_home_score", col("total_home_score").cast(IntegerType()))
                .withColumn("total_away_score", col("total_away_score").cast(IntegerType()))
                .filter(col("play_type").isin("pass", "rush", "punt", "field_goal", "extra_point"))
                .dropDuplicates(["game_id", "play_id"]))
    print(f"Dataframe with {df_clean.count()} rows and {len(df_clean.columns)} columns\n")

    # Save 
    # This requires hadoop winutils.exe in PATH to be able to write to local filesystem on Windows
    df_clean.write.mode('overwrite').parquet(f"{output_path}/season=2023")

    spark.stop()

if __name__ == "__main__":
    main()