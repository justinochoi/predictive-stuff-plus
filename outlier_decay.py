import polars as pl 
from sklearn.ensemble import IsolationForest
import seaborn as sns 
import matplotlib.pyplot as plt 

statcast20 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_20.parquet')
statcast21 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_21.parquet')
statcast22 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_22.parquet')
statcast23 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet')

# first step: create pitch outlier scores for 2020-22 
# will help us learn the factors that cause outlier scores to decrease 

df = pl.concat([statcast20, statcast21, statcast22, statcast23], how = 'vertical_relaxed') 
del statcast20, statcast21, statcast22, statcast23 

cols = [
    'player_name','pitcher','game_date','p_throws','pitch_type',
    'release_speed','pfx_x','pfx_z',
    'release_extension','arm_angle' 
]

feats = ['release_speed','pfx_x','pfx_z']

df_reduced = (
    df.select(
        cols
    ).drop_nulls(
        pl.all()
    ).with_columns(
        pl.when(pl.col('p_throws') == 'R')
        .then(pl.col('pfx_x') * -1) 
        .otherwise(pl.col('pfx_x'))
        .alias('pfx_x_adj'), 
        pl.col('game_date').dt.year().alias('season')
    )
)

# calculate the outlier scores for each year 
# we're only allowed to use current year to fit the IsoForest 

def generate_outlier_scores(df, feats): 

    dfs = [] 
    for year in [2020, 2021, 2022, 2023]: 
        year_data = df.filter(pl.col('season') == year) 
        iforest = IsolationForest(n_estimators = 500).fit(year_data[feats]) 
        scores = iforest.decision_function(year_data[feats])
        year_data = year_data.with_columns(outlier_score = scores)
        dfs.append(year_data)

    return pl.concat(dfs)

df_reduced = generate_outlier_scores(df_reduced, feats)

# how do sweeper outlier scores change over the years? 
sweeper_scores = (
    df_reduced.filter(
        pl.col('pitch_type') == 'ST'
    ).group_by(
        'season'
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('outlier_score').mean().alias('mean_outlier_score')
    )
)

# clear upward trend over time (higher score = less likely to be outlier) 
sns.lineplot(sweeper_scores, x = 'season', y = 'mean_outlier_score')

# is there a way to predict future outlier scores? 
# assume we only have 2020-22 data 

# saving data for now 
df_reduced.write_parquet('df_with_outlier_scores.parquet')
