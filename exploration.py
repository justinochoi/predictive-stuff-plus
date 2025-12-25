import polars as pl 
import numpy as np 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb 
import optuna 
import seaborn as sns 

statcast20 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_20.parquet')
statcast21 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_21.parquet')
statcast22 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_22.parquet')
statcast23 = pl.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet')

# first, try to build basic stuff model using 2020-22 data, predict on 2023 
# main q: will this model overvalue 2023 sweepers? 

# get required columns 
cols = [
    'player_name','pitcher','game_date','p_throws','pitch_type','release_speed','pfx_x','pfx_z',
    'release_extension','arm_angle','delta_pitcher_run_exp'
]

df = pl.concat([statcast20, statcast21, statcast22, statcast23], how = 'vertical_relaxed')

del statcast20, statcast21, statcast22, statcast23 

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

train = df_reduced.filter(pl.col('season').is_between(2020, 2022))
test = df_reduced.filter(pl.col('season') == 2023)

feats = ['release_speed','pfx_x_adj','pfx_z','release_extension','arm_angle']
target = ['delta_pitcher_run_exp'] 

X = train[feats].to_pandas() 
y = train[target].to_pandas() 
pitchers = train['pitcher'].to_pandas() 

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
    
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=76)
    cv_scores = [] 

    for train_idx, val_idx in cv.split(X, y, groups = pitchers): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

        model = xgb.XGBRegressor(**params, random_state=76) 
        model.fit(X_train, y_train, verbose = 0) 
        val_preds = model.predict(X_val) 
        rmse = np.sqrt(mean_squared_error(y_val, val_preds)) 
        cv_scores.append(rmse) 

    return np.mean(cv_scores)

study = optuna.create_study(direction='minimize')
study.optimize(xgb_objective, n_trials = 10) 
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
        pl.col('num_pitches') >= 100
    )
)

# not the best, but not the worst either 
sns.scatterplot(validation, x='mean_actual_rv', y='mean_pred_rv')

# now checking for model drift 
# sweepers only, look at monthly performance 
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

# very noisy 
# probably overlapping with seasonal effect  
sns.lineplot(monthly_st, x='month', y='run_value', hue='type') 

# what about for each pitch type on the year? 
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
    ).sort('rv_diff_per_100')
)

# over-valued: four-seam, changeups, sweepers 
# under-valued: cutters, sinkers 
# roughly equal: sliders, curveballs 
sns.barplot(
    pitch_type, x='pitch_type', y='rv_diff_per_100'
)

# save model 
xgb_spec.save_model('base_xgb.json')
