"""
Key analytics and processing functions"""

import numpy as np
import pandas as pd
from ..data.processing import (
    extractDataInFocus,
    adjustSeries,
    construct_series,
    standardise_dataframe,
)
from ..data.helpers import datetime_to_str
from ..visualization.matplotlib_plots import (
    plot_from_df,
    scatter_plot,
    plot_correlation_heatmap,
)
from . import sensitivity

pd.set_option("display.max_rows", 50)  # consider 'None'
pd.set_option(
    "display.max_columns", 50
)  # consider 'None'; pd.set_option('display.width', 550)
colours = [
    "#000000",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#808080",  # Inferred: this grey colour code is blurred.
    "#d62728",
]


# define the model
def generate_time_series(
    length=100, alpha=0.1, beta=0.1, sigma=1, rolling_momentum=120, rolling_meanreversion=20
):
    # alpha is the momentum coefficient, it needs to be positive
    # beta is the mean reversion coefficient, it needs to be positive
    # sigma is the noise coefficient, keep it equal to 1
    # rules are:
    # beta << sigma [otherwise the mean reversion effect is too strong]
    # import time
    # np.random.seed(int(time.time()))
    time_series = [0]
    for _ in range(1, length):
        # add elements, in sequence: momentum [start of day], then noise [random flows], then
        # mean reversion [people fading]
        next_step = 0
        next_step += np.random.normal(0, sigma)
        next_step += 0  # JUMP
        next_step += alpha * np.sign(
            time_series[-1] - np.mean(time_series[-rolling_momentum:])
        )
        next_step += -beta * (
            time_series[-1] + next_step - np.mean(time_series[-rolling_meanreversion:])
        )
        # save values
        time_series.append(time_series[-1] + next_step)
    return time_series


# inspect sensitivity vs a specific series
def inspect_sensitivity(
    allData, target, explanatory, beta_override=False, sampling_freq="W", nObs_rolling_beta=66
):
    if sampling_freq == 5:
        print(
            "sampling_freq is being passed as integer, this syntax is deprecated, we move to"
            " W or M "
        )
        sampling_freq = "W"
    extracted_data = extractDataInFocus(
        allData, [target] + explanatory, sampling_freq=sampling_freq
    )
    # measure LR sensitivity
    betas, rsq, resid, resid_x = sensitivity.measure_rolling_LR_betas_from_DF(
        extracted_data.diff(), target, explanatory, nObs=nObs_rolling_beta
    )
    # plot betas and RSQ
    betas["zero"] = 0
    plot_from_df(betas, title="Rolling betas", figsize=(12, 6))
    plot_from_df(rsq, title="RSQ", figsize=(12, 6))
    rsq = np.round(rsq.iloc[-1].values, 2) * 100
    beta = betas.iloc[-1].values
    beta = np.round(beta, 5)
    beta = np.round(betas.iloc[-1].values * 20, 2) / 20
    if len(explanatory) == 1:
        single_explanatory_variable = explanatory[0]
        # plot scatter on levels ...
        title = "Scatter of levels"
        X = extracted_data[single_explanatory_variable].values
        y = extracted_data[target].values
        scatter_plot(
            X,
            y,
            title=title,
            x_label=single_explanatory_variable,
            y_label=target,
            cmap="Blues",
        )
        # ... and on residuals
        title = "Scatter of residuals"
        X = resid_x.values[:, 0]
        y = resid.values[:, 0]
        scatter_plot(
            X,
            y,
            title=title,
            x_label="Chg of " + single_explanatory_variable,
            y_label="Residual",
            cmap="Oranges",
        )
    # we construct a beta-adjusted series
    if beta_override:
        if type(beta_override) == float:
            beta_override = [beta_override]
        beta = beta_override
    print("We are using a beta of", beta, "and constructing a new series")
    adj_series = adjustSeries(allData, target, explanatory, beta)
    adj_series = adj_series.dropna()
    plot_from_df(
        adj_series,
        title="History of beta-adjusted series with beta " + str(beta),
        figsize=(9, 4),
    )
    print("Results are: beta", beta, "; RSQ", rsq, "%")
    return adj_series, beta


# now inspect sensitivities against macro variables
def sensitivity_to_macro_variables(
    data_df, target, explanatory_variables=None, pvalue_threshold=0.15
):
    # define explanatory variables
    if explanatory_variables is None:
        explanatory_variables = ["1y1y", "10y", "HVol", "5dbf1w"]  # Inferred: the last two series labels are blurred.
    # construct the df to analyse
    data_df = data_df[[target] + explanatory_variables]
    # resample the data
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.resample("W", label="right", closed="right").last()
    data_df = data_df.dropna()
    # create the df needed
    raw_changes = data_df.diff().dropna()
    # run the analysis
    import statsmodels.api as sm

    raw_changes = sm.add_constant(raw_changes)
    explanatory_variables.append("const")
    sensitivities = pd.DataFrame(data=0, index=explanatory_variables, columns=["Sensitivity"])
    model = sm.OLS(raw_changes[target], raw_changes[explanatory_variables])
    analysis = model.fit()
    print(analysis.summary(), "\n\n")
    # clean up the series in a loop
    while (analysis.pvalues < pvalue_threshold).any():
        # identify the most meaningful variable
        explanatory = analysis.pvalues.idxmin()
        sensi = analysis.params[explanatory]
        conf_int = analysis.conf_int().loc[explanatory].values
        # record the sensitivity
        # sensi = np.round(sensi * 40, 0) / 40
        # assert min(conf_int) <= sensi <= max(conf_int), 'Watch out: you are approximating
        # the beta but it is outside of the uncertainty range'
        sensitivities.loc[explanatory] += sensi
        # adjust the series
        print(
            "The series is being adjusted by",
            explanatory,
            "with beta of",
            np.round(sensi, 2),
            "and pvalue of",
            np.round(analysis.pvalues.min(), 2),
        )
        if explanatory == "const":
            break
        data_df[target] -= sensi * data_df[explanatory]
        # run the analysis again
        raw_changes = data_df.diff().dropna()
        raw_changes = sm.add_constant(raw_changes)
        model = sm.OLS(raw_changes[target], raw_changes[explanatory_variables])
        analysis = model.fit()
    print("\n\nHere are the detected sensitivities:\n", sensitivities)
    # now plot the history of the macro adjusted series
    plot_from_df(data_df[[target]])
    return data_df[[target]]


# estimate PCA weights
def estimate_PCA_weights(
    allData, variables, PCA_component=-1, nObs=50, sampling_freq="W", VERBOSE=True
):
    if sampling_freq == 5:
        print(
            "sampling_freq is being passed as integer, this syntax is deprecated, we move to"
            " W or M "
        )
        sampling_freq = "W"
    # measure PCA sensitivity
    df = extractDataInFocus(
        allData.ffill(), securities=variables, sampling_freq=sampling_freq
    )
    df = df.diff().dropna()
    eigenvecs, explained_variances, all_components = sensitivity.measure_rolling_PCA_from_DF(
        df, nObs=nObs, component=PCA_component
    )
    # construct PCA-adjusted series with the most recent vector
    eigenvecs = eigenvecs.dropna()
    # make the belly unit value and automatically makes it positive
    eigenvecs = eigenvecs * 2 / eigenvecs[variables[1]].values.reshape(-1, 1)
    last_pca_vector = eigenvecs.iloc[-1].values
    series = construct_series(allData, variables, last_pca_vector).dropna()
    if VERBOSE:
        eigenvecs_positive = eigenvecs.copy()
        # move all coeff to positive sign
        for variable in variables:
            eigenvecs_positive[variable] *= np.sign(eigenvecs[variable].iloc[-1])
        plot_from_df(
            eigenvecs_positive,
            title="History of eigenvecs (coeffs are forced to be positive)",
        )
        print(
            "Note the below series is constructed with the right signs for all coefficients"
        )
        plot_from_df(series, title="PCA-adj series", figsize=(6, 4))
        print("Last eigenvector is ", np.round(last_pca_vector, 2))
    return last_pca_vector, series, eigenvecs


# ==============================================================================
# analyse returns
# ==============================================================================
# given a series it calculates what the move was for each data point over a certain horizon
def calculate_target_changes(
    series, max_window=22, sharpe_drift_for_significant_move=0.3, onehot=False
):
    std_lookback_period = 66
    target_std_move = sharpe_drift_for_significant_move * np.sqrt(max_window)
    # calculate rolling std
    std = series.diff().rolling(std_lookback_period).std()
    std = std.fillna(method="backfill")
    # calculate changes and divide by rolling std
    digital_moves = pd.DataFrame(index=series.index, columns=["move"])
    changes = series.diff() / std
    # run a loop to calculate the direction of the move over a max window and with some
    # barriers
    sum_changes = day_count = 0
    day_starting_window = changes.index[0]
    for day in changes.index[1:]:
        sum_changes += changes.loc[day].values
        day_count += 1
        if sum_changes >= target_std_move or day_count >= max_window:
            digital_moves.loc[day_starting_window] = np.sign(sum_changes)
            sum_changes = day_count = 0
            day_starting_window = day
    # remove NaN
    digital_moves = digital_moves.dropna()
    return digital_moves


def calculate_features(array, ZS=[20, 60, 250], RANGE=[20, 60, 250]):
    last = array[-1]
    features = [last]
    # add the z-scores per lookback
    for lookback in ZS:
        if len(array) > lookback:
            mov_avg = array[-lookback:-1].mean()
            std = np.diff(array[-lookback - 1 : -1].reshape(-1)).std()
            features.append((last - mov_avg) / std)
        else:
            features.append(np.nan)
    # add the range per lookback
    up_to_numb_days = 5
    for lookback in RANGE:
        if len(array) > lookback:
            MAX = array[-lookback:-up_to_numb_days].max()
            MIN = array[-lookback:-up_to_numb_days].min()
            features.append(((last - MIN) / (MAX - MIN) * 2 - 1) * 1)
        else:
            features.append(np.nan)
    features = np.array(features)
    features = np.round(features, 1)
    columns = ["last"]
    for lookback in ZS:
        columns.append("ZS_" + str(lookback))
    for lookback in RANGE:
        columns.append("Rg_" + str(lookback))
    return features, columns


def calculate_same_direction_moves(series, features, window_for_same_direction_moves=[10]):
    for window in window_for_same_direction_moves:
        perc_up_moves = u.count_same_direction_moves(
            series, rolling_window=window, SHOW_PLOT=False
        )
        features["% moves " + str(window)] = round(perc_up_moves, 0)
    return features


# ==============================================================================
# animate a scatter plot
# ==============================================================================
# watch out you need to input a df, then the rules for x_data, and the names of labels
def animate_scatter_plot(df, x_data, labels):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    import pandas as pd

    date = None
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        date = df.iloc[frame : frame + 1].index[0]
        date = str(date)
        date = datetime_to_str(date)
        y_data = df.iloc[frame : frame + 1].values.flatten()
        ax.scatter(x_data, y_data)
        scatter = ax.scatter(x_data, y_data)
        ax.set_xlim(x_data.min() - 1, x_data.max() + 1)
        ax.set_ylim(df.min().min() - 1, df.max().max() + 1)
        for i in range(len(labels)):
            ax.text(
                x=x_data[i] + 0.3,
                y=y_data[i] + 0.5,
                s=labels[i],
                ha="center",
                va="bottom",
                size="x-large",
            )
        ax.text(x=x_data.min() - 0.5, y=df.min().min() + 0.5, s=date, size="xx-large")
        plt.title("Animation")
        plt.draw()
        return (scatter,)

    animation = FuncAnimation(
        fig, update, frames=len(df), interval=1, blit=True, repeat=False
    )
    return animation, date


# ==============================================================================
# animated scatter plot by Jovi
# ==============================================================================
def animate_scatter_plot_with_moving_xaxis(
    y_df,
    x_df,
    color_vector=None,
    size_vector=None,
    shadow_frames=20,
    normalise=False,
    x_label="x",
    y_label="y",
):
    """
    Plots an animated chart of y_df (y-axis) against x_df (x-axis) Parameters - y_df :
    pd.DataFrame - x_df : pd.DataFrame - color_vector : list(str)..."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    assert list[Any](y_df.columns) == list[Any](x_df.columns)
    if color_vector:
        assert len(y_df.columns) == len(color_vector)
    if size_vector:
        assert len(y_df.columns) == len(size_vector)
    cols = y_df.columns
    x_data = np.arange(len(cols))
    if normalise:
        y_df = (y_df - y_df.mean(axis=0)) / y_df.std(axis=0)
        x_df = (x_df - x_df.mean(axis=0)) / x_df.std(axis=0)
    ## size and color settings
    fig, ax = plt.subplots(figsize=(14, 10))
    if not color_vector:
        if len(y_df.columns) % 2:  # odd no. of columns
            color_vector = (
                ["blue"] * (len(cols) // 2) + ["black"] + ["red"] * (len(cols) // 2)
            )
        else:
            color_vector = ["black"] * len(cols)
    if not size_vector:
        # if len(y_df.columns) % 2:
        #     size_vector = [100] * (len(cols)//2) + [200] + [100] * (len(cols)//2)
        # else:
        size_vector = [100] * len(cols)
    x_min, x_max = x_df.min().min(), x_df.max().max()
    y_min, y_max = y_df.min().min(), y_df.max().max()

    def update(frame):
        ax.clear()
        date = y_df.iloc[frame : frame + 1].index[0]
        date = datetime_to_str(date)
        x_data = x_df.iloc[frame : frame + 1].values.flatten()
        y_data = y_df.iloc[frame : frame + 1].values.flatten()
        ## main curve
        ax.scatter(x_data, y_data, color=color_vector, s=size_vector)
        scatter = ax.scatter(x_data, y_data)
        ## shadow curves
        for i in range(max(0, frame - shadow_frames), frame):
            x_shadow = x_df.iloc[i : i + 1].values.flatten()
            y_shadow = y_df.iloc[i : i + 1].values.flatten()
            alpha = 0.3 * (1 - (frame - i) / shadow_frames)  # Decreasing opacity
            ax.scatter(x_shadow, y_shadow, color="grey", s=100, alpha=alpha, edgecolor="none")
        ## annotations
        for i in range(len(cols)):
            ax.text(
                x=x_data[i] + 0.3,
                y=y_data[i] + 0.3,
                s=cols[i],
                ha="center",
                va="bottom",
                fontsize=12,
            )
        # ax.text(x=x_min+9, y=y_max+0.6, s=date, fontsize=16, color='red', fontweight='bold')
        ax.text(
            x=x_min + 0.02 * (x_max - x_min),
            y=y_max - 0.05 * (y_max - y_min),
            s=date,
            fontsize=16,
            color="red",
            fontweight="bold",
        )
        ax.set_title("Animated Scatter plot", fontsize=18, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=16, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=16, fontweight="bold", rotation=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.draw()
        return (scatter,)

    animation = FuncAnimation(fig, update, frames=len(y_df), blit=True, repeat=False)
    return animation


# ==============================================================================
# extended correlation and PCA analysis of a group of securities
# ==============================================================================
def extended_correlation_and_PCA_analysis_for_a_group_of_securities(
    data, short_history_months=3, long_history_months=12, sampling_freq="W", n_components=3
):
    """
    sample_freq = 'B', 'D', 'W', 'M', 'Q', 'A', 'H', 'T', 'S'"""

    # ensure we dont modify the original dataframe
    data = data.copy()
    saved_data = data.copy()
    # Observe trend
    colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    plot_from_df(data, title="History", colours=colours, figsize=(12, 6))
    normalised_data = standardise_dataframe(data)
    plot_from_df(
        normalised_data, title="NORMALISED history", colours=colours, figsize=(12, 6)
    )
    # ==============================================================================
    # Correlation of weekly changes
    # ==============================================================================
    data = saved_data.dropna()
    data_resampled = data.resample(sampling_freq).last()
    data_resampled_changes = data_resampled.diff().dropna()
    history_short = pd.Timestamp.today() - pd.DateOffset(months=short_history_months)
    history_long = pd.Timestamp.today() - pd.DateOffset(months=long_history_months)
    data_resampled_changes_short = data_resampled_changes.loc[history_short:]
    data_resampled_changes_long = data_resampled_changes.loc[history_long:]
    # plot the correlations - no need to standardise for correlation measurements
    plot_correlation_heatmap(
        data_resampled_changes_short,
        data_resampled_changes_long,
        title="Short-term (upper) and long-term (lower) Correlation Matrix",
    )
    # ==============================================================================
    # rough PCA analysis, i dont like it
    # ==============================================================================
    ## estimate diversion on PCA relationships
    should_normalise_mean = True
    should_normalise_std = True
    saved_data = data.copy()
    ## process data
    data = saved_data.dropna().copy()
    # data = data.fillna(method='ffill').dropna()
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    ## resample and calibrate the PCA s
    data_norm_and_resampled = data_norm.resample(sampling_freq).last()
    # note this function below already calculates the CHANGES
    print("\nFull PCA analysis and explanation by component")
    pca = sensitivity.measure_dispersion_based_on_PCA(
        data_norm_and_resampled,
        components=len(data_norm_and_resampled.columns),
        use_changes=True,
    )
    print("\nPCA with chosen # of components")
    pca = sensitivity.measure_dispersion_based_on_PCA(
        data_norm_and_resampled, components=n_components, use_changes=True
    )
    ## transform and inverse transform the data - mean to remove unnecessary noise
    data_reconstructed = pd.DataFrame(
        data=pca.inverse_transform(pca.transform(data_norm)),
        index=data_norm.index,
        columns=data_norm.columns,
    )
    ## un-normalise and plot
    data_reconstructed_unnormalised = data_reconstructed * std + mean
    data = saved_data.copy()
    distance = data - data_reconstructed_unnormalised
    distance = distance.dropna()
    ## normalise the distance and select the history
    if should_normalise_mean:
        distance -= distance.mean()
    if should_normalise_std:
        distance /= distance.std()
    # # EWM rolling normalisation
    # ewm_span = 250
    # if should_normalise_mean: distance -= distance.ewm(span=ewm_span).mean()
    # if should_normalise_std: distance /= (distance.ewm(span=ewm_span).std() *
    # np.sqrt(ewm_span))
    ## calculate eigenvectors
    eigenvectors = pd.DataFrame(
        data=np.round(pca.components_.T, 2),
        index=distance.columns,
        columns=np.arange(1, n_components + 1),
    )
    print(eigenvectors)
    ## select a subset of tickers to view and print
    plot_from_df(distance, figsize=(15, 10), colours=colours, title="Reconstruction error")
    print(
        "Variables sorted by how far from mean and adj by std\n",
        distance.iloc[-1].sort_values(ascending=False),
    )


def calculate_persistency(series, n=3, lookback=60, threshold=1):
    """
    Calculate the average n-day momentum over specified lookback days for a given time series.
    Parameters: - series : pd.Series, containing the time..."""

    assert lookback < len(series), "Lookback window must be smaller than length of series."
    if isinstance(series, pd.Series):
        df_mm = series.to_frame("value").dropna()  # drop first row of NaN
    elif isinstance(series, pd.DataFrame):
        df_mm = series.rename(columns={series.columns[0]: "value"}).dropna()
    else:
        raise TypeError("Input must be a pandas Series or single-column DataFrame.")
    # direction of change
    df_mm["change"] = np.sign(df_mm["value"].diff())
    for i in range(1, n + 1):
        df_mm[f"change_{i}"] = df_mm["change"].shift(i)
    # Calculate 1-day to n-day momentum
    df_mm["1d_momentum"] = df_mm.apply(lambda row: row["change"] == row["change_1"], axis=1)
    for num_days in range(2, n + 1):  # Loop from 2 to max_n
        momentum_col = f"{num_days}d_momentum"

        def check_n_day_momentum(row, n_days=num_days):
            # Get the prior n days' changes
            prior_changes = [row[f"change_{i}"] for i in range(1, n_days + 1)]
            unique_prior_directions = set[Any](prior_changes)
            # If prior days didn't all move in same direction, return NaN (no trend to test)
            if len(unique_prior_directions) > 1:
                return np.nan
            # If prior days were consistent, check if today continues the trend
            today_matches_yesterday = row["change"] == row["change_1"]
            return today_matches_yesterday

        df_mm[momentum_col] = df_mm.apply(check_n_day_momentum, axis=1)
    df_pers = pd.DataFrame(index=df_mm.index)
    # Calculate rolling averages and trade counts
    for n in range(1, n + 1):
        momentum_col = f"{n}d_momentum"
        df_pers[f"{n}d"] = df_mm[momentum_col].rolling(window=lookback, min_periods=1).mean()
        df_pers[f"{n}d_numtrades"] = (
            df_mm[momentum_col].rolling(window=lookback, min_periods=1).count()
        )
    df_pers = df_pers.iloc[lookback:]  # Drop the first `lookback` rows
    return df_pers
