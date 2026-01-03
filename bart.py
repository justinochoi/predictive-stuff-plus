import polars as pl 
import pandas as pd 
import pymc as pm 
import pymc_bart as pmb 
import arviz as az 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# load the data 
df = pl.read_parquet('outlier_df.parquet')
train = df.filter(pl.col('season').is_between(2020, 2021))
test = df.filter(pl.col('season') == 2022)

feats = ['velo','pfx_x_adj','pfx_z'] 
extra = ['outlier_score']

train_idx, train_pitcher = pd.factorize(train['pitcher']) 
coords = {'pitcher': train_pitcher}

X_train = train[feats].to_numpy() 
X_train_out = train[feats + extra].to_numpy() 
num_pitches_train = train['num_pitches'].to_numpy()
y_train = train['rv_fut'].to_numpy() 

# we should center our covariates 
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_mean_out = X_train_out.mean(axis=0)
X_std_out = X_train_out.std(axis=0)

X_train_centered = (X_train - X_mean) / X_std
X_train_out_centered = (X_train_out - X_mean_out) / X_std_out 

with pm.Model(coords=coords) as control_mod: 
    # data 
    X = pm.Data("X", X_train_centered) 
    y = pm.Data('y', y_train)
    num_pitches = pm.Data("num_pitches", num_pitches_train)
    pitcher_idx = pm.Data("pitcher_idx", train_idx) 

    # priors 
    s0 = pm.Normal('s0', mu=0, sigma=1) 
    s1 = pm.Normal('s1', mu=-1, sigma=1)
    sigma_p = pm.HalfNormal('sigma_p', sigma=0.5)

    # heteroskedasticity 
    sigma_hat = pm.Deterministic('sigma_hat', s0 + s1 * pm.math.log(num_pitches))

    # pitcher random effect 
    z = pm.Normal('z', mu=0, sigma=1, dims = 'pitcher')
    alpha = pm.Deterministic('alpha', z * sigma_p) 

    # bart 
    mu_bart = pmb.BART('mu_bart', X=X, Y=y, m=20)

    # likelihood 
    rv_hat = pm.Normal(
        "rv_hat", 
        mu = mu_bart + alpha[pitcher_idx], 
        sigma = pm.math.exp(sigma_hat), 
        observed = y
    )

pm.model_to_graphviz(control_mod)

with pm.Model(coords=coords) as outlier_mod: 
    # data 
    X = pm.Data("X", X_train_out_centered) 
    y = pm.Data('y', y_train)
    num_pitches = pm.Data("num_pitches", num_pitches_train)
    pitcher_idx = pm.Data("pitcher_idx", train_idx) 

    # priors 
    s0 = pm.Normal('s0', mu=0, sigma=1) 
    s1 = pm.Normal('s1', mu=-1, sigma=1)
    sigma_p = pm.HalfNormal('sigma_p', sigma=0.5)

    # heteroskedasticity 
    sigma_hat = pm.Deterministic('sigma_hat', s0 + s1 * pm.math.log(num_pitches))

    # pitcher random effect 
    z = pm.Normal('z', mu=0, sigma=1, dims = 'pitcher')
    alpha = pm.Deterministic('alpha', z * sigma_p) 

    # bart 
    mu_bart = pmb.BART('mu_bart', X=X, Y=y, m=20)

    # likelihood 
    rv_hat = pm.Normal(
        "rv_hat", 
        mu = mu_bart + alpha[pitcher_idx], 
        sigma = pm.math.exp(sigma_hat), 
        observed = y
    )

pm.model_to_graphviz(outlier_mod)

with control_mod: 
    control_idata = pm.sample(
        tune = 1000, draws = 1500, chains = 4, cores = 4, 
        target_accept = 0.95, random_seed = 76
    )

control_summary = az.summary(control_idata, var_names=['s0','s1','sigma_p'])

with outlier_mod: 
    outlier_idata = pm.sample(
        tune = 1000, draws = 1500, chains = 4, cores = 4, 
        target_accept = 0.95, random_seed = 76
    )

outlier_summary = az.summary(outlier_idata, var_names=['s0','s1','sigma_p'])

with control_mod: 
    control_pp = pm.sample_posterior_predictive(control_idata, random_seed=76)

az.plot_ppc(control_pp)

with outlier_mod: 
    outlier_pp = pm.sample_posterior_predictive(outlier_idata, random_seed=76)

az.plot_ppc(outlier_pp)

# out-of-sample predictions for existing pitchers 
existing_p = set(train['pitcher']).intersection(test['pitcher'])
new_p = set(test['pitcher']).difference(train['pitcher'])

test_existing = test.filter(pl.col('pitcher').is_in(existing_p))
test_new = test.filter(pl.col('pitcher').is_in(new_p))

pitcher_to_idx = dict(zip(train_pitcher, range(len(train_pitcher))))
test_existing_idx = test_existing['pitcher'].to_pandas().map(pitcher_to_idx).to_numpy() 

X_test_existing = test_existing[feats].to_numpy() 
X_test_existing_centered = (X_test_existing - X_mean) / X_std 

X_test_existing_out = test_existing[feats + extra].to_numpy() 
X_test_existing_out_centered = (X_test_existing_out - X_mean_out) / X_std_out  

num_pitches_test_existing = test_existing['num_pitches'].to_numpy() 
y_test_existing = test_existing['rv_fut'].to_numpy() 

with control_mod: 
    pm.set_data({
        'X': X_test_existing_centered, 
        'y': y_test_existing, 
        'num_pitches': num_pitches_test_existing, 
        'pitcher_idx': test_existing_idx
    })

    oos_pp_existing = pm.sample_posterior_predictive(
        control_idata, predictions=True, var_names=['rv_hat'], random_seed=76
    )

preds_existing = oos_pp_existing.predictions.rv_hat.mean(dim=('chain','draw')).values 

with outlier_mod: 
    pm.set_data({
        'X': X_test_existing_out_centered, 
        'y': y_test_existing, 
        'num_pitches': num_pitches_test_existing, 
        'pitcher_idx': test_existing_idx
    })

    oos_pp_existing_out = pm.sample_posterior_predictive(
        outlier_idata, predictions=True, var_names=['rv_hat'], random_seed=76
    )

preds_existing_out = oos_pp_existing_out.predictions.rv_hat.mean(dim=('chain','draw')).values 

test_existing = test_existing.with_columns(
    pred_rv_fut = preds_existing, 
    pred_rv_fut_outlier = preds_existing_out
) 

# out-of-sample predictions for brand new pitchers 
# we're only going to use the fixed-effect component 

X_test_new = test_new[feats].to_numpy() 
X_test_new_centered = (X_test_new - X_mean) / X_std

X_test_new_out = test_new[feats + extra].to_numpy() 
X_test_new_out_centered = (X_test_new_out - X_mean_out) / X_std_out

num_pitches_test_new = test_new['num_pitches'].to_numpy() 
y_test_new = test_new['rv_fut'].to_numpy() 
test_new_idx, test_new_pitcher = pd.factorize(test_new['pitcher']) 

with control_mod: 
    pm.set_data({
        'X': X_test_new_centered, 
        'y': y_test_new, 
        'num_pitches': num_pitches_test_new, 
        'pitcher_idx': test_new_idx
    })

    rv_hat_new = pm.Normal('rv_hat_new', mu=mu_bart, sigma=pm.math.exp(sigma_hat), observed=y)

with control_mod: 
    oos_ppc_new = pm.sample_posterior_predictive(
        control_idata, predictions=True, var_names=['rv_hat_new'], random_seed=76
    )

preds_new = oos_ppc_new.predictions.rv_hat_new.mean(dim=('chain','draw')).values 

with outlier_mod: 
    pm.set_data({
        'X': X_test_new_out_centered, 
        'y': y_test_new, 
        'num_pitches': num_pitches_test_new, 
        'pitcher_idx': test_new_idx
    })

    rv_hat_new = pm.Normal('rv_hat_new', mu=mu_bart, sigma=pm.math.exp(sigma_hat), observed=y)

with outlier_mod: 
    oos_ppc_new_out = pm.sample_posterior_predictive(
        outlier_idata, predictions=True, var_names=['rv_hat_new'], random_seed=76
    )

preds_new_out = oos_ppc_new_out.predictions.rv_hat_new.mean(dim=('chain','draw')).values 

test_new = test_new.with_columns(
    pred_rv_fut = preds_new, 
    pred_rv_fut_outlier = preds_new_out
) 

validation = (
    pl.concat([test_existing, test_new], how='vertical')
    .filter(pl.col('num_pitches') >= 200, pl.col('num_pitches_fut') >= 200)
)
# comparing the two different versions 
# including the outlier score improves the model! 
mean_absolute_error(validation['rv_fut'], validation['pred_rv_fut'])
mean_absolute_error(validation['rv_fut'], validation['pred_rv_fut_outlier'])
spearmanr(validation['rv_fut'], validation['pred_rv_fut'])
spearmanr(validation['rv_fut'], validation['pred_rv_fut_outlier'])

# much more 'even' spread of data points 
fig, ax = plt.subplots() 
sns.scatterplot(validation, x='pred_rv_fut_outlier', y='rv_fut')

# what about sweepers specifically? 
# we do noticeably better! 
st = validation.filter(pl.col('pitch_type') == 'ST')
mean_absolute_error(st['rv_fut'], st['pred_rv_fut']) * 100 
mean_absolute_error(st['rv_fut'], st['pred_rv_fut_outlier']) * 100 
spearmanr(st['rv_fut'], st['pred_rv_fut'])
spearmanr(st['rv_fut'], st['pred_rv_fut_outlier'])
