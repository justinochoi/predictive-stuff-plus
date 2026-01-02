import polars as pl 
from sklearn.ensemble import IsolationForest 
from sklearn.metrics import mean_absolute_error
from scipy import stats 
import matplotlib.pyplot as plt 
import seaborn as sns 
import bambi as bmb 
import arviz as az 

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


outlier_df = pl.read_parquet('outlier_df.parquet')
outlier_train = outlier_df.filter(pl.col('season').is_between(2020, 2021))
outlier_test = outlier_df.filter(pl.col('season') == 2022)

rv_mod_control = bmb.Model(
    'rv_fut ~ velo + pfx_x_adj + pfx_z + (1 | pitcher)', 
    data = outlier_train.to_pandas(), family = 't'
)

rv_control_idata = rv_mod_control.fit(
    draws = 1500, tune = 1000, inference_method='numpyro', 
    chains = 4, cores = 4, random_seed = 76, 
    idata_kwargs=dict(log_likehood=True)
)

control_summary = az.summary(rv_control_idata)

# obtain out-of-sample predictions for both models
pred_rv_control = rv_mod_control.predict(
    rv_control_idata, kind = 'response', 
    data = outlier_test, inplace=False, 
    sample_new_groups=True, random_seed=76
)

pred_rv_control_means = az.extract(pred_rv_control, group = 'posterior_predictive', combined = True)['rv_fut'].mean(dim='sample').values 
outlier_test = outlier_test.with_columns(pred_rv_control = pred_rv_control_means) 

validation_bmb = outlier_test.filter(
    pl.col('num_pitches') >= 200, 
    pl.col('num_pitches_fut') >= 200, 
)

mean_absolute_error(validation_bmb['rv_fut'], validation_bmb['pred_rv_control'])
stats.spearmanr(validation_bmb['rv_fut'], validation_bmb['pred_rv_control'])

fig, ax = plt.subplots() 
sns.scatterplot(validation_bmb, x='pred_rv_control', y='rv_fut')
