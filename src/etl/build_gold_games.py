import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

def load_config():
    with open("config/paths.yml", "r") as f:
        config = yaml.safe_load(f)
    return config

''' Transform play-by-play data from silver to gold layer adding game-level aggregations'''
def main():
    spark = (SparkSession.builder
             .appName("transform_pbp_gold")
             .master('local[4]')
             .config('spark.sql.adaptive.enabled', 'true')
             .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
             .getOrCreate())
    
    config = load_config()

    silver_pbp_path = f"{config['silver']['pbp']}/season=2023"
    df_silver = spark.read.option('header', 'true').parquet(silver_pbp_path)
    output_path = config['gold']['games']

    # 1. Base game-level ML targets and aggregations
    df_games_base = (df_silver
                .groupBy("game_id", "season", "game_date", "week", "home_team", "away_team", "div_game", "is_playoffs")
                .agg(
                    max("total_home_score").alias("final_home_score"),
                    max("total_away_score").alias("final_away_score"),
                    (max("total_home_score") + max("total_away_score")).alias("final_total_score"),
                    max("home_margin").alias("final_home_margin"),
                    sum(when(col("play_type") == "pass", 1).otherwise(0)).alias("total_pass_plays"),
                    sum(when(col("play_type") == "rush", 1).otherwise(0)).alias("total_rush_plays"),
                    sum(when(col("play_type").isin("pass", "rush"), col("yards_gained")).otherwise(0)).alias("total_offensive_yards"),
                    avg("epa").alias("avg_epa_per_play")
                )
            )

    # 2. Team stats per game. We create this smaller df to make WINDOW sums and averages easier to debug later
    # This creates 2 rows per game_id, one for each team as offense
    temp_df_team_stats = (df_silver
        .filter(col("posteam").isNotNull())
        .groupBy("game_id", "season", "week", "posteam", "defteam", "home_team", "away_team")
        .agg(
            avg("epa").alias("off_epa_avg"), 
            sum(when(col("play_type") == "pass", col("yards_gained")).otherwise(0)).alias("pass_yds"),
            sum(when(col("play_type") == "run", col("yards_gained")).otherwise(0)).alias("rush_yds"),
            sum("yards_gained").alias("total_yds"),
            count("play_id").alias("team_plays"),
            # Get total points scored by posteam to later add rolling avg of points scored
            max(when(col("posteam") == col("home_team"), col("total_home_score"))
                .when(col("posteam") == col("away_team"), col("total_away_score"))
                .otherwise(0)
            ).alias("off_final_points_scored")
        )
    )

    # 3. Add rolling averages using pyspark windows
    window_recent = Window.partitionBy("posteam", "season").orderBy("week").rowsBetween(-4, -1)
    window_season = Window.partitionBy("posteam", "season").orderBy("week").rowsBetween(Window.unboundedPreceding, -1)
    window_defense_recent = Window.partitionBy("defteam", "season").orderBy("week").rowsBetween(-4, -1)
    window_defense_season = Window.partitionBy("defteam", "season").orderBy("week").rowsBetween(Window.unboundedPreceding, -1)

    df_team_stats = (temp_df_team_stats
                    .withColumn("off_epa_recent_avg", avg("off_epa_avg").over(window_recent))
                    .withColumn("off_epa_season_avg", avg("off_epa_avg").over(window_season))
                    .withColumn("team_plays_recent_avg", avg("team_plays").over(window_recent))
                    .withColumn("team_plays_season_avg", avg("team_plays").over(window_season))
                    .withColumn("points_scored_recent_avg", avg("off_final_points_scored").over(window_recent))
                    .withColumn("points_scored_season_avg", avg("off_final_points_scored").over(window_season))
                    .withColumn("pass_yds_recent_avg", avg("pass_yds").over(window_recent))
                    .withColumn("pass_yds_season_avg", avg("pass_yds").over(window_season))
                    .withColumn("rush_yds_recent_avg", avg("rush_yds").over(window_recent))
                    .withColumn("rush_yds_season_avg", avg("rush_yds").over(window_season))
                    # defensive stats
                    # DEFENSIVE EPA SHOULD BE MULTIPLIED BY -1
                    .withColumn("def_epa_recent_avg", avg(col("off_epa_avg") * -1).over(window_defense_recent))
                    .withColumn("def_epa_season_avg", avg(col("off_epa_avg") * -1).over(window_defense_season))
                    .withColumn("def_allowed_recent_avg", avg("off_final_points_scored").over(window_defense_recent))
                    .withColumn("def_allowed_season_avg", avg("off_final_points_scored").over(window_defense_season))
                    .withColumn("def_allowed_pass_yds_recent_avg", avg("pass_yds").over(window_defense_recent))
                    .withColumn("def_allowed_pass_yds_season_avg", avg("pass_yds").over(window_defense_season))
                    .withColumn("def_allowed_rush_yds_recent_avg", avg("rush_yds").over(window_defense_recent))
                    .withColumn("def_allowed_rush_yds_season_avg", avg("rush_yds").over(window_defense_season))
    )

    # 4. Pivot & join home/away
    # stats per team
    df_home_stats = df_team_stats.filter(col("posteam") == col("home_team")).select(
                "game_id",
                col("off_epa_recent_avg").alias("home_off_epa_last_4_games_avg"),
                col("off_epa_season_avg").alias("home_off_epa_season_avg"),
                col("team_plays_recent_avg").alias("home_team_plays_last_4_games_avg"),
                col("team_plays_season_avg").alias("home_team_plays_season_avg"),
                col("points_scored_recent_avg").alias("home_points_scored_last_4_games_avg"),
                col("points_scored_season_avg").alias("home_points_scored_season_avg"),
                col("pass_yds_recent_avg").alias("home_pass_yds_last_4_games_avg"),
                col("pass_yds_season_avg").alias("home_pass_yds_season_avg"),
                col("rush_yds_recent_avg").alias("home_rush_yds_last_4_games_avg"),
                col("rush_yds_season_avg").alias("home_rush_yds_season_avg"),
                col("def_epa_recent_avg").alias("home_def_epa_last_4_games_avg"),
                col("def_epa_season_avg").alias("home_def_epa_season_avg"),
                col("def_allowed_recent_avg").alias("home_def_allowed_last_4_games_avg"),
                col("def_allowed_season_avg").alias("home_def_allowed_season_avg"),
                col("def_allowed_pass_yds_recent_avg").alias("home_def_allowed_pass_yds_last_4_games_avg"),
                col("def_allowed_pass_yds_season_avg").alias("home_def_allowed_pass_yds_season_avg"),
                col("def_allowed_rush_yds_recent_avg").alias("home_def_allowed_rush_yds_last_4_games_avg"),
                col("def_allowed_rush_yds_season_avg").alias("home_def_allowed_rush_yds_season_avg")
    )

    # stats per team
    df_away_stats = df_team_stats.filter(col("posteam") == col("away_team")).select(
                "game_id",
                col("off_epa_recent_avg").alias("away_off_epa_last_4_games_avg"),
                col("off_epa_season_avg").alias("away_off_epa_season_avg"),
                col("team_plays_recent_avg").alias("away_team_plays_last_4_games_avg"),
                col("team_plays_season_avg").alias("away_team_plays_season_avg"),
                col("points_scored_recent_avg").alias("away_points_scored_last_4_games_avg"),
                col("points_scored_season_avg").alias("away_points_scored_season_avg"),
                col("pass_yds_recent_avg").alias("away_pass_yds_last_4_games_avg"),
                col("pass_yds_season_avg").alias("away_pass_yds_season_avg"),
                col("rush_yds_recent_avg").alias("away_rush_yds_last_4_games_avg"),
                col("rush_yds_season_avg").alias("away_rush_yds_season_avg"),
                col("def_epa_recent_avg").alias("away_def_epa_last_4_games_avg"),
                col("def_epa_season_avg").alias("away_def_epa_season_avg"),
                col("def_allowed_recent_avg").alias("away_def_allowed_last_4_games_avg"),
                col("def_allowed_season_avg").alias("away_def_allowed_season_avg"),
                col("def_allowed_pass_yds_recent_avg").alias("away_def_allowed_pass_yds_last_4_games_avg"),
                col("def_allowed_pass_yds_season_avg").alias("away_def_allowed_pass_yds_season_avg"),
                col("def_allowed_rush_yds_recent_avg").alias("away_def_allowed_rush_yds_last_4_games_avg"),
                col("def_allowed_rush_yds_season_avg").alias("away_def_allowed_rush_yds_season_avg")
    )

    # 5. Final join
    df_games_final = df_games_base.join(df_home_stats, "game_id", "left").join(df_away_stats, "game_id", "left")
    df_games_final.write.mode('overwrite').partitionBy("season").parquet(f"{output_path}/season=2023")

    spark.stop()

if __name__ == "__main__":
    main()