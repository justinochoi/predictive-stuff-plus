import polars as pl 
import numpy as np 
from sklearn.ensemble import IsolationForest 
from sklearn.metrics import mean_squared_error
import seaborn as sns 
import matplotlib.pyplot as plt 
import bambi as bmb 
import arviz as az 

statcast20 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_20.parquet')
statcast21 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_21.parquet')
statcast22 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_22.parquet')
statcast23 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet')

# first step: create pitch outlier scores for 2020-23 
# will help us learn the factors that cause outlier scores to decrease 

df = pl.concat([statcast20, statcast21, statcast22, statcast23], how = 'vertical_relaxed') 
del statcast20, statcast21, statcast22, statcast23 

cols = [
    'player_name','pitcher','game_date','p_throws','pitch_type',
    'release_speed','pfx_x','pfx_z','delta_run_exp'
]

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
        pl.when(pl.col('pitch_type').is_in(['KC','SV']))
        .then(pl.col('pitch_type') == 'CU')
        .otherwise(pl.col('pitch_type')), 
        pl.col('game_date').dt.year().alias('season')
    ).filter(
        pl.col('pitch_type').is_in(['FF','FC','SI','SL','ST','CU','CH','FS'])
    )
)

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

# how have outlier scores changed over the years by pitch type? 
score_by_ptype = (
    df_reduced.group_by(
        ['season','pitch_type']
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('outlier_score').mean().alias('mean_outlier_score')
    )
)

# for most pitches, change isn't that extreme 
# the sweeper revolution is very noticeable though 
# both four-seam and sinkers become more 'outlier-ish' due to reduced usage 
fig, ax = plt.subplots() 
sns.lineplot(score_by_ptype, x = 'season', y = 'mean_outlier_score', hue='pitch_type')

# is there a way to predict future outlier scores? 
# assume we only have 2020-22 data 
# create dataset that can be used to build outlier score prediction model 
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
            pl.col('release_speed').mean().alias('mean_velo'), 
            pl.col('pfx_x_adj').mean().alias('mean_pfx_x_adj'), 
            pl.col('pfx_z').mean().alias('mean_pfx_z'), 
            pl.col('outlier_score').mean().alias('mean_outlier_score'),
            pl.col('delta_run_exp').mean().alias('mean_rv')
        )

        fut_agg = fut.group_by(
            ['player_name','pitcher','p_throws','pitch_type','season']
        ).agg(
            pl.len().alias('num_pitches'), 
            pl.col('release_speed').mean().alias('mean_velo'), 
            pl.col('pfx_x_adj').mean().alias('mean_pfx_x_adj'), 
            pl.col('pfx_z').mean().alias('mean_pfx_z'), 
            pl.col('outlier_score').mean().alias('mean_outlier_score'), 
            pl.col('delta_run_exp').mean().alias('mean_rv')
        )

        data = cur_agg.join(
            fut_agg, 
            on = ['player_name','pitcher','pitch_type','p_throws'], 
            how = 'inner', 
            suffix = '_fut'
        ).with_columns(
            score_change = pl.col('mean_outlier_score_fut') - pl.col('mean_outlier_score')
        )

        dfs.append(data)

    return pl.concat(dfs)

outlier_df = create_outlier_dataset(df_reduced)

# league-wide average score change is 0, which is what you'd expect 
outlier_df['score_change'].mean() 

# also as expected, sweepers' outlier score changes are skewed positively 
outlier_df_st = outlier_df.filter(pl.col('pitch_type') == 'ST')
fig, ax = plt.subplots() 
sns.histplot(outlier_df_st['score_change'])

# split into train (2020-22) and test (2023) 
train = outlier_df.filter(pl.col('season').is_between(2020, 2021), pl.col('season_fut').is_between(2021, 2022))
test = outlier_df.filter(pl.col('season') == 2022, pl.col('season_fut') == 2023)

# now time to build the model? 
# num pitches + velo + movement + pitch type

outlier_mod = bmb.Model(
    'mean_outlier_score_fut ~ mean_outlier_score + mean_velo + mean_pfx_x_adj*pitch_type + mean_pfx_z + (1 | pitcher)', 
    data = train.to_pandas(), family = 't'
)

outlier_mod_idata = outlier_mod.fit(
    tune=1000, draws=1500, chains=4, cores=4, 
    random_seed=76, idata_kwargs={'log_likelihood': True}
)

# model summary and trace plots 
az.summary(outlier_mod_idata, var_names = ['mean_velo', 'mean_pfx_x_adj', 'mean_outlier_score']) 
az.plot_trace(outlier_mod_idata, var_names = ['mean_velo', 'mean_pfx_x_adj', 'mean_outlier_score'])

# fitted values on training set 
fitted = outlier_mod.predict(
    outlier_mod_idata, data = train, kind = "response", 
    include_group_specific=False, inplace=False
)
fitted_means = az.extract(fitted, group='posterior_predictive', num_samples=500, combined=True)['mean_outlier_score_fut'].mean(dim='sample').values 
train = train.with_columns(pred_score = fitted_means)

# looks good 
fig, ax = plt.subplots() 
sns.scatterplot(train, x='pred_score', y='mean_outlier_score_fut') 

# predictions on test set 
# remove the player random effect to prevent data leakage 
preds = outlier_mod.predict(
    outlier_mod_idata, data = test, kind = "response", 
    include_group_specific=False, inplace=False
)
preds_means = az.extract(preds, group='posterior_predictive', num_samples=500, combined=True)['mean_outlier_score_fut'].mean(dim='sample').values 
test = test.with_columns(pred_score = preds_means)

# also looks good 
fig, ax = plt.subplots() 
sns.scatterplot(test, x='pred_score', y='mean_outlier_score_fut') 

# future run value model 
# compare using actual past outlier score vs. predicted future outlier score 
# also include control 

rv_mod_control = bmb.Model(
    'mean_rv_fut ~ mean_velo + mean_pfx_x_adj*pitch_type + mean_pfx_z + (1 | pitcher)', 
    data = train.to_pandas(), family = 't'
)

rv_mod_control_idata = rv_mod_control.fit(
    draws=1500, tune=1000, chains=4, cores=4, 
    random_seed=76, idata_kwargs={'log_likelihood': True}
)

rv_mod_basic = bmb.Model(
    'mean_rv_fut ~ mean_outlier_score + mean_velo + mean_pfx_x_adj*pitch_type + mean_pfx_z + (1 | pitcher)', 
    data = train.to_pandas(), family = 't'
)

rv_mod_basic_idata = rv_mod_basic.fit(
    draws=1500, tune=1000, chains=4, cores=4, 
    random_seed=76, idata_kwargs={'log_likelihood': True}
)

rv_mod_pred = bmb.Model(
    'mean_rv_fut ~ pred_score + mean_velo + mean_pfx_x_adj*pitch_type + mean_pfx_z + (1 | pitcher)', 
    data = train.to_pandas(), family = 't'
)

rv_mod_pred_idata = rv_mod_pred.fit(
    draws=1500, tune=1000, chains=4, cores=4, 
    random_seed=76, idata_kwargs={'log_likelihood': True}
)

# compare the three models 
compare_dict = {
    'control': rv_mod_control_idata, 
    'basic': rv_mod_basic_idata, 
    'with_predicted_score': rv_mod_pred_idata
}

# we do beat the control 
# the model with the predicted score is ever-so-slightly better 
# but well within the margin of error... 
compare_results = az.compare(compare_dict)
az.plot_compare(compare_results)

# get oos predictions for all mdoels 
rv_control_preds = rv_mod_control.predict(
    rv_mod_control_idata, kind = 'response', data = test, 
    inplace=False, include_group_specific=False
)
rv_basic_preds = rv_mod_basic.predict(
    rv_mod_basic_idata, kind = 'response', data = test, 
    inplace=False, include_group_specific=False
)
rv_pred_preds = rv_mod_pred.predict(
    rv_mod_pred_idata, kind = 'response', data = test, 
    inplace=False, include_group_specific=False
)

rv_control_means = az.extract(rv_control_preds, group='posterior_predictive', combined=True)['mean_rv_fut'].mean(dim='sample').values 
rv_basic_means = az.extract(rv_basic_preds, group='posterior_predictive', combined=True)['mean_rv_fut'].mean(dim='sample').values 
rv_pred_means = az.extract(rv_pred_preds, group='posterior_predictive', combined=True)['mean_rv_fut'].mean(dim='sample').values 
test = test.with_columns(
    control_rv = rv_control_means, 
    basic_rv = rv_basic_means, 
    pred_rv = rv_pred_means
)

# calculating rmse 
# basically no improvement at all 
np.sqrt(mean_squared_error(test['mean_rv_fut'], test['control_rv']))
np.sqrt(mean_squared_error(test['mean_rv_fut'], test['basic_rv']))
np.sqrt(mean_squared_error(test['mean_rv_fut'], test['pred_rv']))

# any hope for just sweepers? 
st_test = test.filter(pl.col('pitch_type') == 'ST') 
np.sqrt(mean_squared_error(st_test['mean_rv_fut'], st_test['basic_rv']))
np.sqrt(mean_squared_error(st_test['mean_rv_fut'], st_test['pred_rv']))
# nope... 