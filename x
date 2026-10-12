"""
The script measures the relative sensitivity of series"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA


# =============================================================================
# LR
# =============================================================================
# measure betas from linear regression model
def measure_LR_betas(X, y, calculate_residuals=True):
    linearModel = linear_model.LinearRegression(fit_intercept=True)
    linearModel.fit(X, y)
    beta = linearModel.coef_[0]
    rsq = linearModel.score(X, y)
    if calculate_residuals:
        y_hat = linearModel.predict(X)
        resid = y - y_hat
        resid_x = X
    else:
        resid, resid_x = None, None
    return beta, rsq, resid, resid_x


# measure betas from linear regression model
def measure_LR_parameters(X, y):
    linearModel = linear_model.LinearRegression(fit_intercept=True)
    linearModel.fit(X, y)
    intercept = linearModel.intercept_[0]
    beta = linearModel.coef_[0][0]
    rsq = linearModel.score(X, y)
    return intercept, beta, rsq


# calculate rolling sensitivity and returns betas, rsq, residuals
def measure_rolling_LR_betas(X, y, nObs=60):
    betas = pd.DataFrame(np.nan * np.ones(shape=X.shape), index=X.index, columns=X.columns)
    RSQs = pd.DataFrame(np.nan * np.ones(len(y)), index=X.index, columns=["RSQ"])
    nObs = min(nObs, len(X) - 1)
    is_last_instance = False
    for idx in np.arange(nObs, len(X)):
        upToObs = len(X) - idx
        X_batch = X.iloc[-nObs - upToObs : -upToObs]
        y_batch = y.iloc[-nObs - upToObs : -upToObs]
        if idx == len(X) - 1:
            is_last_instance = True
        beta, rsq, resid, resid_x = measure_LR_betas(
            X_batch, y_batch, calculate_residuals=is_last_instance
        )
        betas.iloc[idx] = beta
        RSQs.iloc[idx] = rsq
    return betas, RSQs, resid, resid_x


# calculate rolling sensitivity, but starting from a DataFrame and with labels of explanatory
# and target variables
def measure_rolling_LR_betas_from_DF(df, target, explanatory, nObs=60):
    # calculate rolling sensitivity from a dataframe
    df = df[[target] + explanatory].copy()
    df = df.dropna()
    X = df[explanatory] if type(explanatory) == list else df[[explanatory]]
    y = df[[target]]
    betas, RSQs, resid, resid_x = measure_rolling_LR_betas(X, y, nObs=nObs)
    return betas, RSQs, resid, resid_x


# =============================================================================
# PCA
# =============================================================================
# runs the PCA analysis of a DataFrame and returns the PCA-adjusted series + the pca_vector
def measure_PCA(df, component=-1):
    df = df.dropna().copy()
    pca = PCA()
    pca.fit(df)
    pca_vector = pca.components_[component]
    explained_var = pca.explained_variance_ / pca.explained_variance_.sum()
    return pca_vector, explained_var, pca.components_


# calculate rolling PCA from a dataframe
def measure_rolling_PCA_from_DF(df, nObs=125, component=-1):
    global all_components
    df = df.dropna().copy()
    eigenvecs = pd.DataFrame(
        np.nan * np.ones(shape=df.shape), index=df.index, columns=df.columns
    )
    explained_variances = pd.DataFrame(
        np.nan * np.ones(shape=df.shape),
        index=df.index,
        columns=np.arange(len(df.columns)) + 1,
    )
    nObs = min(nObs, len(df) - 1)
    for idx in np.arange(nObs, len(df)):
        upToObs = len(df) - idx
        df_batch = df.iloc[-nObs - upToObs : -upToObs]
        pca_vector, explained_var, all_components = measure_PCA(df_batch, component=component)
        eigenvecs.iloc[idx] = pca_vector
        explained_variances.iloc[idx] = explained_var
    return eigenvecs, explained_variances, all_components


# runs the PCA analysis of a DataFrame and returns the PCA-adjusted series + the pca_vector
def measure_dispersion_based_on_PCA(df, components=2, excluded_test_set=0, use_changes=True):
    components = min(components, len(df.columns))
    df = df.dropna().copy()
    if use_changes:
        df = df.diff().dropna()
    if excluded_test_set:
        df = df.iloc[:-excluded_test_set]
    from sklearn.decomposition import PCA

    pca = PCA(n_components=components)
    pca.fit(df)
    print("Explained variance: ", np.round(pca.explained_variance_ratio_, 2))
    return pca


def construct_scatter_of_beta_by_level(
    df, target, explanatory, x_axis_explanatory=True, remove_outliers_flag=True
):
    from ..data.processing import remove_outliers  # Inferred: relative import dots are slightly blurred.
    from ..visualization.matplotlib_plots import scatter_plot  # Inferred: relative import dots are slightly blurred.

    df = df.dropna()
    y = df[[target]]
    X = df[explanatory]
    beta = (y - y.shift(1)) / (X - X.shift(1)).values
    x_axis = (X + X.shift(1)) / 2 if x_axis_explanatory else (y + y.shift(1)) / 2
    beta = beta.dropna()
    x_axis = x_axis.dropna()
    # remove outliers
    if remove_outliers_flag:
        beta[target] = remove_outliers(beta.values, low=5, high=95)
    scatter_plot(x_axis.values, beta.values, title="Scatter of betas by level of variable")
