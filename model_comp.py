import polars as pl 
import numpy as np 
import xgboost as xgb 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

df_reduced = pl.read_parquet('df_with_outlier_score.parquet')
train = df_reduced.filter(pl.col('season').is_between(2020, 2022))
test = df_reduced.filter(pl.col('season') == 2023)
feats = ['release_speed','pfx_x_adj','pfx_z','release_extension','arm_angle'] 
extra = ['outlier_score']

base_xgb = xgb.XGBRegressor() 
base_xgb.load_model('base_xgb.json') 
outlier_xgb = xgb.XGBRegressor() 
outlier_xgb.load_model('xgb_with_outlier_score.json')

base_preds = base_xgb.predict(test[feats]) 
outlier_preds = outlier_xgb.predict(test[feats + extra]) 
test = test.with_columns(
    base_preds = base_preds, 
    outlier_preds = outlier_preds 
)

validation = (
    test.group_by(
        ['player_name','pitcher','pitch_type']
    ).agg(
        pl.len().alias('num_pitches'), 
        pl.col('delta_pitcher_run_exp').mean().alias('mean_actual_rv'), 
        pl.col('base_preds').mean().alias('mean_base_rv'), 
        pl.col('outlier_preds').mean().alias('mean_outlier_rv')
    ).filter(
        pl.col('num_pitches') >= 200
    )
)

fig, axs = plt.subplots(1, 2)
sns.scatterplot(
    data=validation,
    x='mean_actual_rv',
    y='mean_base_rv',
    ax=axs[0]
)
sns.scatterplot(
    data=validation,
    x='mean_actual_rv',
    y='mean_outlier_rv',
    ax=axs[1]
)

# same rmse, outlier score model has higher correlation coefficient 
base_rmse = np.sqrt(mean_squared_error(validation['mean_actual_rv'], validation['mean_base_rv']))
outlier_rmse = np.sqrt(mean_squared_error(validation['mean_actual_rv'], validation['mean_outlier_rv']))
base_corr = np.corrcoef(validation['mean_actual_rv'], validation['mean_base_rv'])[0,1]
outlier_corr = np.corrcoef(validation['mean_actual_rv'], validation['mean_outlier_rv'])[0,1]
