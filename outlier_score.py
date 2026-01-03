
import polars as pl 
from sklearn.ensemble import IsolationForest 

df_reduced = pl.read_parquet('df_reduced.parquet')

# calculate the outlier scores for each year 
# we're only allowed to use current year to fit the IsoForest 
def generate_outlier_scores(df, feats): 

    dfs = [] 
    years = df['season'].unique() 
    for year in years: 
        year_data = df.filter(pl.col('season') == year) 
        iforest = IsolationForest(n_estimators = 500, random_state=76).fit(year_data[feats]) 
        scores = iforest.decision_function(year_data[feats])
        year_data = year_data.with_columns(outlier_score = scores)
        dfs.append(year_data)

    return pl.concat(dfs)

df_reduced = generate_outlier_scores(df_reduced, feats = ['release_speed','pfx_x_adj','pfx_z'])

# KNOWING 2023 OUTLIER SCORE IN ADVANCE IS DATA LEAKAGE!!!! 
# we need a model that can predict 2023 outlier score 

def create_outlier_dataset(df): 

    dfs = [] 
    years = df['season'].unique() 
    for year in years[:-1]: 

        cur = df.filter(pl.col('season') == year) 
        fut = df.filter(pl.col('season') == year + 1) 
        
        cur_agg = cur.group_by(
            ['player_name','pitcher','p_throws','pitch_type','season']
        ).agg(
            pl.len().alias('num_pitches'), 
            pl.col('release_speed').mean().alias('velo'), 
            pl.col('pfx_x_adj').mean().alias('pfx_x_adj'), 
            pl.col('pfx_z').mean().alias('pfx_z'), 
            pl.col('outlier_score').mean().alias('outlier_score'),
            pl.col('delta_pitcher_run_exp').mean().alias('rv')
        )

        fut_agg = fut.group_by(
            ['player_name','pitcher','p_throws','pitch_type','season']
        ).agg(
            pl.len().alias('num_pitches'), 
            pl.col('release_speed').mean().alias('velo'), 
            pl.col('pfx_x_adj').mean().alias('pfx_x_adj'), 
            pl.col('pfx_z').mean().alias('pfx_z'), 
            pl.col('outlier_score').mean().alias('outlier_score'), 
            pl.col('delta_pitcher_run_exp').mean().alias('rv')
        )

        data = cur_agg.join(
            fut_agg, 
            on = ['player_name','pitcher','pitch_type','p_throws'], 
            how = 'inner', 
            suffix = '_fut'
        )

        dfs.append(data)

    return pl.concat(dfs)

outlier_df = create_outlier_dataset(df_reduced)
outlier_df.write_parquet('outlier_df.parquet')