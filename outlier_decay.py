import polars as pl 
import numpy as np 
from sklearn.ensemble import IsolationForest 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import seaborn as sns 
import matplotlib.pyplot as plt 
import xgboost as xgb 
import optuna 

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
    'release_speed','pfx_x','pfx_z','release_extension','arm_angle','delta_pitcher_run_exp'
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
    ).sort(pl.col('game_date'))
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
df_reduced.write_parquet('df_with_outlier_scores.parquet')

# KNOWING 2023 OUTLIER SCORE IN ADVANCE IS DATA LEAKAGE!!!! 

# stuff+ model with outlier score included 
train = df_reduced.filter(pl.col('season').is_between(2020, 2022))
test = df_reduced.filter(pl.col('season') == 2023)

feats = ['release_speed','pfx_x_adj','pfx_z','release_extension','arm_angle','outlier_score']
target = ['delta_pitcher_run_exp'] 

X = train[feats].to_pandas() 
y = train[target].to_pandas() 

def xgb_objective(trial): 
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20), 
    }
    
    cv = TimeSeriesSplit(n_splits=5)
    cv_scores = [] 

    for train_idx, val_idx in cv.split(X, y): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

        model = xgb.XGBRegressor(**params, random_state=76) 
        model.fit(X_train, y_train, verbose = 0) 
        val_preds = model.predict(X_val) 
        rmse = np.sqrt(mean_squared_error(y_val, val_preds)) 
        cv_scores.append(rmse) 

    return np.mean(cv_scores)

study = optuna.create_study(direction='minimize')
study.optimize(xgb_objective, n_trials = 20) 
best_params = study.best_params 

xgb_spec = xgb.XGBRegressor(**best_params) 
xgb_spec.fit(X, y)

rv_preds = xgb_spec.predict(test[feats])  
test = test.with_columns(predicted_rv = rv_preds) 

# check model calibration by grouping pitcher + pitch type 
validation = (
    test.group_by(
        ['player_name','pitcher','pitch_type']
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('delta_pitcher_run_exp').mean().alias('mean_actual_rv'), 
        pl.col('predicted_rv').mean().alias('mean_pred_rv') 
    ).filter(
        pl.col('num_pitches') >= 200
    )
)

# not the best, but not the worst either 
fig, ax = plt.subplots() 
sns.scatterplot(validation, x='mean_actual_rv', y='mean_pred_rv')

monthly_st = (
    test.filter(
        pl.col('pitch_type') == 'ST'
    ).group_by(
        pl.col('game_date').dt.month().alias('month'), 
        maintain_order=True 
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('delta_pitcher_run_exp').mean().alias('mean_actual_rv'), 
        pl.col('predicted_rv').mean().alias('mean_pred_rv'), 
    ).filter(
        pl.col('month').is_between(4,9)
    ).unpivot(
        index = ['month','num_pitches'], 
        on = ['mean_actual_rv', 'mean_pred_rv'], 
        variable_name = 'type', 
        value_name = 'run_value'
    ).with_columns(
        run_value = pl.col('run_value') * 100
    )
)

fig, ax = plt.subplots()
sns.lineplot(monthly_st, x='month', y='run_value', hue='type') 

pitch_type = (
    test.group_by(
        'pitch_type'
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('delta_pitcher_run_exp').mean().alias('mean_actual_rv'), 
        pl.col('predicted_rv').mean().alias('mean_pred_rv'), 
    ).with_columns(
        rv_diff_per_100 = (pl.col('mean_pred_rv') - pl.col('mean_actual_rv')) * 100,
    ).filter(
        pl.col('pitch_type').is_in(['FF','SI','FC','SL','CU','CH','ST'])
    ).sort('rv_diff_per_100', descending=True)
)

# very interesting as sweepers are no longer over-predicted 
fig, ax = plt.subplots()  
fig.suptitle('A Somewhat Unavoidable Pitfall of Stuff+', fontsize=16, fontweight='bold') 
ax.set_xlabel('Run Value Residuals per 100'), 
ax.set_ylabel('Pitch Type')
ax.set_title('Model trained on 2020-22, predictions for 2023')
ax.text(
    -0.15, 7.5, "Under-predicted", size=9, ha='center', va='center',
    bbox=dict(boxstyle='larrow', fc='lightblue', ec='steelblue', lw=2)
)
ax.text(
    0.075, 7.5, "Over-predicted", size=9, ha='center', va='center',
    bbox=dict(boxstyle='rarrow', fc='lightblue', ec='steelblue', lw=2)
)
sns.barplot(pitch_type, x='rv_diff_per_100', y='pitch_type', hue='pitch_type')

# save model 
xgb_spec.save_model('xgb_with_outlier_score.json')


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
