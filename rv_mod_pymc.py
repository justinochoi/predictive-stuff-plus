import polars as pl 
import pandas as pd 
import numpy as np 
import pymc as pm 
import pymc_bart as pmb
import arviz as az 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr 

# load everything 
outlier_df = pl.read_parquet('outlier_df.parquet') 

# change to rv/100 for interpretability 
outlier_df = outlier_df.with_columns(pl.col('rv_fut') * 100, pl.col('rv') * 100)

# train/test split
outlier_train = outlier_df.filter(pl.col('season').is_between(2020, 2021))
outlier_test = outlier_df.filter(pl.col('season') == 2022)
train_idx, train_pitcher = pd.factorize(outlier_train['pitcher']) 

# model features 
feats = ['velo','pfx_x_adj','pfx_z'] 
extra = ['outlier_score']

# setting up coords 
coords = {'pitcher': train_pitcher}

# extract covariates, center them 
X_train = outlier_train[feats].to_numpy() 
y_train = outlier_train['rv_fut'].to_numpy() 
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_centered = (X_train - X_mean) / X_std 

# model 
with pm.Model(coords=coords) as linear: 
    # data 
    X = pm.Data("X", X_train_centered) 
    y = pm.Data('y', y_train)
    pitcher_idx = pm.Data("pitcher_idx", train_idx) 

    # priors 
    sigma = pm.HalfStudentT("sigma", nu=4, sigma=0.05)
    sigma_p = pm.HalfNormal("sigma_p", 2)
    intercept = pm.Normal('intercept', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=0.5, shape=len(feats)) 
    nu = pm.Gamma('nu', 2, 0.1)

    # random intercept 
    z = pm.Normal('z', mu=0, sigma=1, dims='pitcher')
    alpha = pm.Deterministic('alpha', intercept + z * sigma_p)

    # fixed + random
    mu = pm.Deterministic("mu", pm.math.dot(X, beta) + alpha[pitcher_idx])

    # likelihood 
    rv_hat = pm.StudentT("rv_hat", mu=mu, nu=nu, sigma=sigma, observed=y) 

pm.model_to_graphviz(linear) 

with linear: 
    idata_lin = pm.sample(random_seed=76, tune=1000, draws=1500, cores=4) 

lin_summary = az.summary(idata_lin, var_names=['intercept','beta','sigma','sigma_p','nu'])

with linear: 
    in_sample_ppc = pm.sample_posterior_predictive(idata_lin, random_seed=76)

in_sample_means = in_sample_ppc.posterior_predictive.rv_hat.mean(dim=('chain','draw')).values
outlier_train = outlier_train.with_columns(fitted_rv = in_sample_means)

fig, ax = plt.subplots() 
sns.scatterplot(data=outlier_train.filter(pl.col('num_pitches') >= 200, pl.col('num_pitches_fut') >= 200), x='fitted_rv', y='rv_fut')

# out-of-sample prediction 
existing_p = set(outlier_train['pitcher']).intersection(outlier_test['pitcher'])
new_p = set(outlier_test['pitcher']).difference(outlier_train['pitcher'])

test_existing = outlier_test.filter(pl.col('pitcher').is_in(existing_p))
test_new = outlier_test.filter(pl.col('pitcher').is_in(new_p))

# correct index values for test_existing? 
pitcher_to_idx = dict(zip(train_pitcher, range(len(train_pitcher))))
test_existing_idx = test_existing['pitcher'].to_pandas().map(pitcher_to_idx).to_numpy() 

X_test_existing = test_existing[feats].to_numpy() 
X_test_existing_centered = (X_test_existing - X_mean) / X_std 

with linear: 
    pm.set_data({
        'X': X_test_existing_centered, 
        'y': np.zeros(len(X_test_existing_centered)), 
        'pitcher_idx': test_existing_idx 
    })

    oos_ppc_existing = pm.sample_posterior_predictive(idata_lin, predictions=True, random_seed=76)

pred_existing = oos_ppc_existing.predictions.rv_hat.mean(dim=('chain','draw')).values 
test_existing = test_existing.with_columns(pred_rv_fut = pred_existing)

# now let's deal with entirely new pitchers 
X_test_new = test_new[feats].to_numpy() 
X_test_new_centered = (X_test_new - X_mean) / X_std 
y_test_new = test_new['rv_fut'].to_numpy() 
test_new_idx, test_new_pitcher = pd.factorize(test_new['pitcher']) 

with linear: 
    linear.add_coord('new_pitcher', test_new_pitcher)

    z_new = pm.Normal('z_new', mu=0, sigma=1, dims='new_pitcher')
    alpha_new = pm.Deterministic('alpha_new', intercept + z_new * sigma_p)
    mu_new = pm.Deterministic('mu_new', pm.math.dot(X_test_new_centered, beta) + alpha_new[test_new_idx])
    rv_hat_new = pm.StudentT('rv_hat_new', mu=mu_new, nu=nu, sigma=sigma, observed=y_test_new) 

pm.model_to_graphviz(linear)

with linear: 
    oos_ppc_new = pm.sample_posterior_predictive(idata_lin, predictions=True, var_names=['rv_hat_new'], random_seed=76)

pred_new = oos_ppc_new.predictions.rv_hat_new.mean(dim=('chain','draw')).values 
test_new = test_new.with_columns(pred_rv_fut = pred_new) 


validation = pl.concat([test_existing, test_new], how='vertical').filter(pl.col('num_pitches') >= 200, pl.col('num_pitches_fut') >= 200)
mean_absolute_error(validation['rv_fut'], validation['pred_rv_fut'])
spearmanr(validation['rv_fut'], validation['pred_rv_fut'])

fig, ax = plt.subplots()
sns.scatterplot(validation, x='pred_rv_fut', y='rv_fut')