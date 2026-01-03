import polars as pl 
import pandas as pd 
import pymc as pm 
import bambi as bmb 
import arviz as az 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pl.read_parquet('outlier_df.parquet').filter(pl.col('pitch_type') == 'FF')
train = df.filter(pl.col('season').is_between(2020, 2021))
test = df.filter(pl.col('season') == 2022)

# we'll build a model to predict future four-seam run value using previous season velo 
# goal is to explore differences between pymc and bambi 

# first specifying the bambi model 
bmb_mod = bmb.Model(
    'rv_fut ~ velo + (1 | pitcher)', 
    data = train.to_pandas(), family = 'gaussian', 
    priors = {
        'Intercept': bmb.Prior('Normal', mu=0, sigma=2), 
        'velo': bmb.Prior('Normal', mu=0, sigma=0.5), 
        '1|pitcher': bmb.Prior('Normal', mu=0, sigma=bmb.Prior('HalfNormal', sigma=0.01)), 
        'sigma': bmb.Prior('HalfStudentT', nu=4, sigma=0.05)
    }, 
    center_predictors=False
)

bmb_idata = bmb_mod.fit(chains=4, cores=4, random_seed=76) 

# next we'll specify the pymc model 
# we need the identities of pitchers and index mapping
train_idx, train_pitcher = pd.factorize(train['pitcher']) 
coords = {'pitcher': train_pitcher}
X_train = train['velo'].to_numpy() 
y_train = train['rv_fut'].to_numpy() 

with pm.Model(coords=coords) as pymc_mod: 
    # data 
    X = pm.Data("X", X_train) 
    y = pm.Data('y', y_train)
    pitcher_idx = pm.Data("pitcher_idx", train_idx) 

    # priors 
    sigma = pm.HalfStudentT("sigma", nu=4, sigma=0.05)
    sigma_p = pm.HalfNormal("sigma_p", 0.01)
    intercept = pm.Normal('intercept', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=0.5) 

    # pitcher random effect 
    z = pm.Normal('z', mu=0, sigma=1, dims='pitcher')
    alpha = pm.Deterministic('alpha', z * sigma_p)

    # mean component (fixed + random)
    mu = pm.Deterministic('mu', intercept + alpha[pitcher_idx] + beta * X)

    # likelihood 
    rv_hat = pm.Normal('rv_hat', mu=mu, sigma=sigma, observed=y)

pm.model_to_graphviz(pymc_mod)

with pymc_mod: 
    pymc_idata = pm.sample(chains=4, cores=4, random_seed=76)

# these should be the same 
bmb_summary = az.summary(bmb_idata)
pymc_summary = az.summary(pymc_idata, var_names=['intercept','beta','sigma','sigma_p'])

# posterior means are identical, differences in ESS and sd probably noise? 
# we should also check that they return the same predictions 
bmb_ppc = bmb_mod.predict(bmb_idata, kind='response_params', inplace=False, random_seed=76)
fitted_bmb = bmb_ppc.posterior['mu'].mean(dim=('chain','draw')).values 
fitted_pymc = pymc_idata.posterior['mu'].mean(dim=('chain','draw')).values 

# basically identical! that's what we want 
fig, ax = plt.subplots() 
sns.scatterplot(x=fitted_bmb, y=fitted_pymc)

train = train.with_columns(
    fitted_rv_bmb = fitted_bmb, 
    fitted_rv_pymc = fitted_pymc
)

# step 2: pitchers who are in both train AND test 
# all this set up is for pymc lol 
existing_p = set(train['pitcher']).intersection(test['pitcher'])
new_p = set(test['pitcher']).difference(train['pitcher'])

test_existing = test.filter(pl.col('pitcher').is_in(existing_p))
test_new = test.filter(pl.col('pitcher').is_in(new_p))

pitcher_to_idx = dict(zip(train_pitcher, range(len(train_pitcher))))
test_existing_idx = test_existing['pitcher'].to_pandas().map(pitcher_to_idx).to_numpy() 

X_test_existing = test_existing['velo'].to_numpy() 
y_test_existing = test_existing['rv_fut'].to_numpy() 

with pymc_mod: 
    pm.set_data({
        'X': X_test_existing, 
        'y': y_test_existing, 
        'pitcher_idx': test_existing_idx
    }) 
    oos_ppc_existing = pm.sample_posterior_predictive(
        pymc_idata, predictions=True, random_seed=76
    )

pymc_preds_existing = oos_ppc_existing.predictions.rv_hat.mean(dim=('chain','draw')).values 

# for bambi, the setup is much much easier 
oos_ppc_bambi = bmb_mod.predict(
    bmb_idata, kind = 'response', 
    data = test_existing, inplace=False, 
    random_seed=76
)

bmb_preds_existing = oos_ppc_bambi.posterior_predictive.rv_fut.mean(dim=('chain','draw')).values 

# also basically identical 
fig, ax = plt.subplots() 
sns.scatterplot(x=pymc_preds_existing, y=bmb_preds_existing)

test_existing = test_existing.with_columns(
    pred_rv_fut_bambi = bmb_preds_existing, 
    pred_rv_fut_pymc = pymc_preds_existing
)

# step 3: predictions for entirely new pitchers 
X_test_new = test_new['velo'].to_numpy() 
y_test_new = test_new['rv_fut'].to_numpy() 
test_new_idx, test_new_pitcher = pd.factorize(test_new['pitcher']) 

with pymc_mod: 
    pymc_mod.add_coord('new_pitcher', test_new_pitcher)

    z_new = pm.Normal('z_new', mu=0, sigma=1, dims='new_pitcher')
    alpha_new = pm.Deterministic('alpha_new', z_new * sigma_p)
    mu_new = pm.Deterministic('mu_new', intercept + pm.math.dot(X_test_new, beta) + alpha_new[test_new_idx])
    rv_hat_new = pm.Normal('rv_hat_new', mu=mu_new, sigma=sigma, observed=y_test_new) 

with pymc_mod: 
    oos_ppc_new = pm.sample_posterior_predictive(
        pymc_idata, predictions=True, var_names=['rv_hat_new'], 
        random_seed=76
    )

pymc_preds_new = oos_ppc_new.predictions.rv_hat_new.mean(dim=('chain','draw')).values 

# bambi setup is again easier 
# we specify sample_new_groups = True 
oos_ppc_bambi = bmb_mod.predict(
    bmb_idata, kind = 'response', 
    data = test_new, inplace=False, 
    sample_new_groups=True, random_seed=76
)

bmb_preds_new = oos_ppc_bambi.posterior_predictive.rv_fut.mean(dim=('chain','draw')).values 

# a little noisier, but basically co-linear 
# i'm pretty sure it's because bambi has a different sampling mechanism 
# these would be the same if we only included fixed effects 
fig, ax = plt.subplots() 
sns.scatterplot(x=pymc_preds_new, y=bmb_preds_new)

test_new = test_new.with_columns(
    pred_rv_fut_bambi = bmb_preds_new, 
    pred_rv_fut_pymc = pymc_preds_new
)

validation = (
    pl.concat([test_existing, test_new], how='vertical')
    .filter(pl.col('num_pitches') >= 200, pl.col('num_pitches_fut') >= 200)
)

# not bad! 
fig, ax = plt.subplots()
sns.scatterplot(validation, x='pred_rv_fut_bambi', y='rv_fut')