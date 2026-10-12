"""
Rudimentary analysis of seasonal patterns in indices/time series."""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_1samp, wilcoxon
import plotly.graph_objects as go

FIGSIZE = (9, 6)


# ==============================================================================
# SUPPORTING FUNCTIONS
# ==============================================================================
def _calculate_changes(data, window_days=252, min_periods=126, normalise=True):
    """
    data: price series as DataFrame (1 col), indexed by date. Returns: normalised daily "changes"
    (residuals) as DataFrame."""

    # process data
    data = data.copy()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    # calculate daily changes
    changes = data.diff()
    if normalise:
        # trailing mean of daily changes (no look-ahead)
        mu = (
            changes.rolling(window=window_days, min_periods=min_periods).mean().shift(1)
        )  # shift to ensure only info up to t-1 is used
        std = (
            changes.rolling(window=window_days, min_periods=min_periods).std().shift(1)
        )  # shift to ensure only info up to t-1 is used
        changes = (changes - mu) / std
    # drop na
    changes = changes.dropna()
    return changes


def _sum_up_changes_within_ranges(data: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """
    Given a series of "changes", aggregate by year and period (month/week/day/...). For
    period='half_month', splits each month into 1-/1+, 2-/2+, etc."""

    data = data.copy()
    date_index = pd.to_datetime(data.index)
    data.index = date_index
    col = data.columns[0]
    if period == "half_month":
        half = np.where(date_index.day <= 15, "-", "+")
        labels = date_index.month.astype(str) + half
        grouped = data.groupby([date_index.year, labels])[col].sum()
        changes = grouped.unstack(0).T  # years as rows
        half_months_order = [f"{m}{s}" for m in range(1, 13) for s in ("-", "+")]
        return changes.reindex(columns=half_months_order)
    else:
        # e.g. 'month', 'week', 'day'
        grp_key = getattr(date_index, period)
        grouped = data.groupby([date_index.year, grp_key])[col].sum()
        return grouped.unstack(0).T


def _calculate_average_changes(data: pd.DataFrame, trimmed_thresholds) -> pd.DataFrame:
    """
    Compute trimmed means across rows for each column. thresholds: e.g. [0, 15, 35, 50] 0 -> mean
    50 -> median else trimmed mean using quantiles per column."""

    # data = data.copy().dropna(axis=0)       # not needed, as done before in the code
    result = pd.DataFrame(index=data.columns, dtype=float)
    for t in trimmed_thresholds:
        label = f"{t}% Trimmed Mean"
        if t == 0:
            result[label] = data.mean(axis=0)
        elif t == 50:
            result[label] = data.median(axis=0)
        else:
            q_low = data.quantile(t / 100.0, axis=0)
            q_high = data.quantile(1 - t / 100.0, axis=0)
            trimmed = data.where((data >= q_low) & (data <= q_high))
            result[label] = trimmed.mean(axis=0)
    return result.round(3)


def _calculate_average_changes_with_lookback(
    data: pd.DataFrame, lookback_years=(7, 3)
) -> pd.DataFrame:
    """
    Mean by column for last full N years (rows)."""

    # data = data.copy().dropna(axis=0)       # not needed, as done before in the code
    out = {}
    for lb in lookback_years:
        out[f"Mean {lb} Years"] = data.iloc[-lb:].mean(axis=0)
    return pd.DataFrame(out, index=data.columns).round(3)


def _global_seasonality_anova(changes_per_year_period):
    """
    One-way ANOVA across periods: H0: all period means are equal H1: at least one period mean
    differs Parameters ---------- changes_per_year_period : DataFrame rows = years, columns =
    periods Returns ------- f_stat : float p_value : float"""

    # each column is a group: all years for that period
    groups = [
        col.dropna().values
        for _, col in changes_per_year_period.items()
        if col.dropna().size > 1  # need at least 2 obs for ANOVA
    ]
    if len(groups) < 2:
        return np.nan, np.nan  # not enough data
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def _calculate_p_values(changes: pd.DataFrame):
    """
    Compute p-values for tests vs overall mean/median across all cells. Returns (p_values_df,
    has_seasonality_detected: bool)"""

    values = changes.values
    valid = ~np.isnan(values)
    overall_mean = values[valid].mean()
    overall_median = np.median(values[valid])
    # mean: one-sample t-test vs overall mean
    t_stat, p_mean = ttest_1samp(values, popmean=overall_mean, nan_policy="omit", axis=0)
    # median: Wilcoxon vs overall median, per column
    p_median = changes.apply(
        lambda col: wilcoxon(col.dropna() - overall_median).pvalue, axis=0
    )
    p_values = pd.DataFrame(
        {
            "Mean p-value": p_mean,  # t-test
            "Median p-value": p_median,  # Wilcoxon
        },
        index=changes.columns,
    )
    return t_stat, p_values


def _suggest_period_labels(n_periods: int):
    """
    Suggest labels for common period counts."""

    if n_periods == 12:
        return [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    if n_periods == 24:
        return [
            f"{m}{s}"
            for m in [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            for s in ("-", "+")
        ]
    if n_periods == 31:
        return [str(i) for i in range(1, 32)]
    # fallback: 1..N
    return [str(i) for i in range(1, n_periods + 1)]


def _plot_results(df: pd.DataFrame, title: str = ""):
    """
    Plot each column as a line, with overall median line (Plotly interactive)."""

    if df.empty:
        return
    labels = _suggest_period_labels(len(df.index))
    med = np.median(df.values)
    max_y = np.nanmax(np.abs(df.values)) if not np.isnan(np.nanmax(np.abs(df.values))) else 1
    fig = go.Figure()
    # Add zero line (black dashed)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7, layer="below")
    # Add median line (red dashed)
    fig.add_hline(
        y=med,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text=f"Median: {med:.3f}",
        layer="below",
    )
    # Add data lines with varying opacity
    n_cols = len(df.columns)
    for i, col in enumerate[Any](df.columns):
        opacity = (i + 1) / n_cols
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col].values,
                mode="lines",
                name=str(col),
                line=dict[str, str | float](color="rgb(0, 0, 255)", width=1.2),
                opacity=opacity,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Period: %{x}<br>Value:"
                    " %{y:.4g}<extra></extra>"
                ),
            )
        )
    # Configure layout
    width = int(FIGSIZE[0] * 100)
    height = int(FIGSIZE[1] * 100)
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict[str, list[Any | float] | bool](range=[-max_y * 1.1, max_y * 1.1], showgrid=True),
        xaxis=dict[str, bool | str | int](showgrid=True, tickmode="linear", tick0=0),
        legend=dict[str, float | str](
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )
    # Set x-axis labels
    if len(labels) > 0:
        fig.update_xaxes(tickvals=df.index.tolist(), ticktext=labels)
    fig.show()


def _print_p_values_pretty(p_values: pd.DataFrame, significance: float):
    labels = _suggest_period_labels(len(p_values.index))
    p = p_values.round(2)
    p.index = labels
    mask = p <= significance
    styled = pd.DataFrame(" ", index=p.index, columns=p.columns)
    styled[mask] = "***"  # Inferred: the exact count of asterisks is blurred.
    styled += p.astype(str)
    print("\n", styled)


def _count_percentage_of_signals(p_values: pd.DataFrame, significance: float):
    signals = (p_values < significance).sum().sum()
    total = p_values.count().sum()
    perc = np.round(signals / total, 3) if total > 0 else 0.0
    print(
        f"Seasonalities at {significance * 100:.0f}% "
        f"significance level was detected {perc * 100:.2f}% times"
    )


def _calculate_confidence_intervals(changes: pd.DataFrame) -> pd.DataFrame:
    """
    Min, mean±std, max per column."""

    std_coeff = 0.5
    desc = changes.agg(["min", "mean", "std", "max"])
    ci = pd.DataFrame(index=changes.columns)
    ci["Min"] = desc.loc["min"]
    ci[f"Mu-{std_coeff}std"] = desc.loc["mean"] - std_coeff * desc.loc["std"]
    ci["mean"] = desc.loc["mean"]
    ci[f"Mu+{std_coeff}std"] = desc.loc["mean"] + std_coeff * desc.loc["std"]
    ci["Max"] = desc.loc["max"]
    ci["bounds_same_sign"] = (
        np.sign(ci[f"Mu+{std_coeff}std"]) * np.sign(ci[f"Mu-{std_coeff}std"]) > 0
    )
    ci["direction"] = np.where(
        ci["bounds_same_sign"], np.where(ci["mean"] > 0, "+", "-"), " "
    )
    return ci


def _build_period_diagnostics(changes_per_year_period, p_values):
    """
    Build an explanatory table of per-period stats: - mean, median, std - effect vs overall mean -
    z-score-like metric - existing p-values (mean and median tests)"""

    chgs = changes_per_year_period.copy()
    overall_mean = chgs.values[~np.isnan(chgs.values)].mean()
    # basic stats
    per_mean = chgs.mean(axis=0)
    per_median = chgs.median(axis=0)
    per_std = chgs.std(axis=0)
    # effect and "z"
    effect = per_mean - overall_mean
    z_score = effect / per_std.replace(0, np.nan)
    # confidence intervals
    ci = _calculate_confidence_intervals(chgs)
    direction = ci["direction"]
    diag = pd.DataFrame(
        {
            "z_score": z_score,
            "dir": direction,
            "period-mean": effect,
            "mean": per_mean,
            "median": per_median,
            "std": per_std,
        }
    )
    return diag


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def inspect_seasonality(
    data: pd.DataFrame,
    period: str = "month",
    period_in_focus=False,
    significance: float = 0.2,
    VERBOSE: bool = True,
    FORCE_QUIET: bool = False,
):
    """
    Core non-robust seasonality inspection for a single series."""

    if not FORCE_QUIET:
        name = data.columns[0]
        print(f"\n\nResults for series {name} since year {data.index.min().year}\n")
        print("DISPLAY OF MONTHLY CHANGES (rec low and pay high)")
    # define trimmed means
    trimmed = [0, 15, 50]
    # define the changes
    changes_raw = _calculate_changes(data, window_days=252, min_periods=60, normalise=True)
    # calculate changes per year per period
    changes_per_year_period_no_drop = _sum_up_changes_within_ranges(
        changes_raw, period=period
    )
    changes_per_year_period = changes_per_year_period_no_drop.dropna(axis=0)
    if VERBOSE:
        print(changes_per_year_period_no_drop)
    # global test
    f_stat, p_global = _global_seasonality_anova(changes_per_year_period)
    global_seasonality_detected = p_global < significance
    # test p_values
    t_stat, p_values = _calculate_p_values(changes_per_year_period)
    if not FORCE_QUIET:
        if global_seasonality_detected:
            print(
                f"Seasonality present with significance {significance}, f_stat: {f_stat:.3f},"
                f" p-value: {p_global:.3f}."
            )  # \nGlobal test suggests some period means differ. ")
        else:
            print(
                f"Seasonality **NOT** present with significance {significance}, f_stat:"
                f" {f_stat:.3f}, p-value: {p_global:.3f}."
            )
            # \nGlobal test does not reject hypothesis of equal means across periods.
            # Per-period signals may be noise. ")
    if period_in_focus:
        pvalue_specific_period = p_values.iloc[[period_in_focus - 1]]
        period_seasonality_detected = (pvalue_specific_period < significance).values.all()
    else:
        period_seasonality_detected = False
    # Seasonality is detected if:
    # 1. Global test shows significance across all periods, AND
    # 2. Either no specific period is in focus (testing all periods), OR the specific period
    # is significant
    detected = global_seasonality_detected and (
        period_seasonality_detected or not period_in_focus
    )
    if (VERBOSE or detected) and not FORCE_QUIET:
        if period_seasonality_detected:
            print(f"Seasonality detected for {period_in_focus}")
        avg = _calculate_average_changes(changes_per_year_period, trimmed)
        avg_lb = _calculate_average_changes_with_lookback(
            changes_per_year_period, lookback_years=(5, 2)
        )
        _plot_results(avg, f"{name}: different trimmed means")
        _plot_results(avg_lb, f"{name}: different lookbacks")
        # _plot_results(changes.iloc[-3:].T, f"{name}: recent years")
        _print_p_values_pretty(p_values, significance)
        _count_percentage_of_signals(p_values, significance)
        # diagnostics
        diagnostics = _build_period_diagnostics(changes_per_year_period, p_values)
        print("\n", diagnostics)
    return p_values, detected


def run_seasonality_test(
    data: pd.DataFrame,
    period: str = "month",
    period_in_focus=False,
    significance: float = 0.20,
    VERBOSE: bool = True,
    ROBUST: bool = False,
    consistency_threshold: float = 0.8,
):
    """
    High-level entry point for seasonality testing. If ROBUST=True, tests consistency of
    seasonality across shifted and perturbed versions of the series. A period is considered
    robustly seasonal if it shows significance in >= consisten..."""

    if not ROBUST:
        return inspect_seasonality(
            data,
            period=period,
            period_in_focus=period_in_focus,
            significance=significance,
            VERBOSE=VERBOSE,
            FORCE_QUIET=False,
        )
    shift_range = [-4, -2, 0, 2, 4]
    if VERBOSE:
        print("Running a robust seasonality test using shifts", shift_range)
        print(f"Requiring consistency across >= {consistency_threshold * 100:.0f}% of shifts")
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    # Track significance across shifts for each period
    consistency_rate = 0
    for shift in shift_range:
        shifted = data.shift(shift)
        p_v, is_significant = inspect_seasonality(
            shifted,
            period=period,
            period_in_focus=period_in_focus,
            significance=significance,
            VERBOSE=False,
            FORCE_QUIET=True,
        )
        # Calculate consistency: what fraction of shifts showed significance for each period
        consistency_rate += is_significant / len(shift_range)
    # A period is robustly seasonal if it's significant in >= consistency_threshold of shifts
    robust_significance = consistency_rate >= consistency_threshold
    if robust_significance:
        print("-- Robust seasonality detected --")
    return consistency_rate, robust_significance


# ==============================================================================
# advanced seasonality functions
# ==============================================================================
# different forms of seasonality plotting
# day by day changes with heatmap
def seasonality_day_by_day(
    series,
    start_year=False,
    date_adjustment=0,
    cmap="RdYlBu_r",
    remove_median=False,
    remove_first_of_month=False,
):
    # select the year subset
    data = series.copy()
    data.index = pd.to_datetime(data.index)
    if start_year:
        data = data.loc[str(start_year) :]
    data = data.astype(np.float32)
    # shift the data if needed
    data = data.shift(date_adjustment)
    data = data.dropna()
    # group the data
    grouped = data.groupby([data.index.year, data.index.month, data.index.day]).mean()
    grouped.index.names = ["year", "month", "day"]
    data_label = data.columns[0]
    for year in np.unique(data.index.year):
        # Pivot the grouped data to create the changes_per_year_period dataframe
        changes_per_day = grouped.loc[year].pivot_table(
            index="day", columns="month", values=data_label
        )
        # subtract the first day
        if remove_median:
            reshaped_data = changes_per_day.values.reshape(-1)
            mask = ~np.isnan(reshaped_data)
            changes_per_day -= np.median(reshaped_data[mask])
        if remove_first_of_month:
            first_non_nan = changes_per_day.apply(lambda col: col[col.first_valid_index()])
            changes_per_day = changes_per_day.subtract(first_non_nan, axis=1)
        changes_per_day = np.round(changes_per_day, 1)
        # skip plotting if no data or insufficient valid data
        if changes_per_day.empty or len(changes_per_day.columns) == 0:
            continue
        # skip if too many NaN values (heatmap won't render properly)
        valid_cells = (~changes_per_day.isna()).sum().sum()
        total_cells = changes_per_day.size
        # Ensure all days 1-31 are present in the heatmap
        all_days = list[int](range(1, 32))
        changes_per_day_full = pd.DataFrame(index=all_days, columns=changes_per_day.columns)
        changes_per_day_full.loc[changes_per_day.index] = changes_per_day
        changes_per_day = changes_per_day_full
        # Sort days in ascending order (1-31)
        changes_per_day = changes_per_day.sort_index()
        # plot heatmap with Plotly
        fig_width = max(int(len(changes_per_day.columns) * 60), 300)
        fig_height = max(int(len(changes_per_day.index) * 20), 300)
        try:
            # Create custom text array for display (empty for NaN)
            text_array = []
            hover_text = []
            for day in changes_per_day.index:
                row_text = []
                row_hover = []
                for month in changes_per_day.columns:
                    val = changes_per_day.loc[day, month]
                    if pd.isna(val):
                        row_text.append("")  # Empty string for NaN cells
                        row_hover.append(f"Day {day}, Month {month}: No data")
                    else:
                        row_text.append(f"{val:.1f}")
                        row_hover.append(f"Day {day}, Month {month}: {val:.1f}")
                text_array.append(row_text)
                hover_text.append(row_hover)
            # Create Plotly heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=changes_per_day.values,
                    x=changes_per_day.columns,
                    y=changes_per_day.index,
                    colorscale=cmap if cmap else "RdBu",
                    text=text_array,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict[str, int](thickness=15),
                    hovertext=hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                    showscale=True,
                )
            )
            fig.update_layout(
                title=str(year),
                width=fig_width,
                height=fig_height,
                xaxis_title="Month",
                yaxis_title="Day",
                template="plotly_white",
                yaxis=dict[str, str | int](
                    autorange="reversed",  # Display days 1 at top, 31 at bottom
                    tickmode="linear",  # Use linear tick spacing
                    tick0=1,  # Start at day 1
                    dtick=1,  # Tick every 1 day
                    showgrid=False,
                ),
                xaxis=dict[str, bool | list[Any]](
                    showgrid=False,
                    tickvals=list[Any](changes_per_day.columns),
                    ticktext=[str(m) for m in changes_per_day.columns],
                ),
            )
            fig.show()
        except Exception as e:
            print(f"Could not plot heatmap for {year}: {e}")


def seasonality_around_specific_set_of_dates(
    series,
    dates,
    window_of_days_around_a_date=15,
    print_moves_per_date=False,
    print_recent_performance=True,
):
    from .. import construct_series
    from ..visualization.plotly_plots import plot_from_df

    # process data
    custom_series = series.copy()
    custom_series.index = pd.to_datetime(custom_series.index)
    selected_dates = pd.to_datetime(dates)
    # settings
    colours = ["#87CEFA", "#1E90FF", "#003366", "#FFD700", "#DAA520"]
    grey_colour = "#D3D3D3"
    if len(dates) > 3:
        colours = (len(dates) - 3) * [grey_colour] + colours

    # rebase function around a date
    def rebase_around_a_date(data, window_of_days_around_a_date, events_around_a_date):
        # Handle both integer and tuple inputs for the window
        if isinstance(window_of_days_around_a_date, int):
            window_start, window_end = (
                -window_of_days_around_a_date,
                window_of_days_around_a_date,
            )
        elif (
            isinstance(window_of_days_around_a_date, tuple)
            and len(window_of_days_around_a_date) == 2
        ):
            window_start, window_end = window_of_days_around_a_date
        else:
            raise ValueError(
                "window_of_days_around_a_date must be an integer or a tuple of two integers"
                " (a, b)."
            )
        # construct empty df
        margin = 5
        idx_extended = [i for i in range(window_start - margin, window_end + margin + 1)]
        idx_true = [i for i in range(window_start, window_end + 1)]
        rebased_data = pd.DataFrame(index=idx_extended)
        # populate the rebased df
        for i, event in enumerate[Any](events_around_a_date):
            # skip the event if the history is not available
            if data.index[0] > event.start_date or data.index[-1] < event.start_date:
                continue
            # identify the event date
            event_date = data.loc[event.start_date.date() :].index[0]
            # construct the time series around the event
            from pandas.tseries.offsets import BDay

            start_date = pd.to_datetime(event_date - BDay(-window_start)).date()
            end_date = pd.to_datetime(event_date + BDay(window_end + 1)).date()
            extracted_data = data.loc[start_date:end_date]
            extracted_data -= data.loc[event_date]
            idx = [
                list[Any](extracted_data.index).index(day)
                - list[Any](extracted_data.index).index(event_date)
                for day in extracted_data.index
            ]
            assert len(idx) == len(extracted_data), "Issue on function REBASE AROUND A DATE"
            extracted_data = pd.DataFrame(
                index=idx, data=extracted_data.values, columns=[event.name]
            )
            rebased_data = pd.concat([rebased_data, extracted_data], axis=1)
            rebased_data = rebased_data.loc[idx_true]
        return rebased_data

    # construct events
    from collections import namedtuple

    event_around_a_date = namedtuple("Event", ["name", "start_date", "end_date"])
    events_around_a_date = []
    for i in np.arange(len(selected_dates)):
        date = pd.to_datetime(selected_dates[i])
        events_around_a_date.append(
            event_around_a_date(str(selected_dates[i].date()), date, date)
        )
    # plot performance
    performance_daily_closes = rebase_around_a_date(
        custom_series, window_of_days_around_a_date, events_around_a_date
    )
    performance_to_plot = performance_daily_closes.copy()
    # prepare the subsection of columns to plot
    performance_to_plot = performance_daily_closes.copy()
    performance_to_plot["Mean"] = performance_to_plot.mean(axis=1)
    performance_to_plot["Median"] = performance_to_plot.median(axis=1)
    performance_to_plot = performance_to_plot[
        list[Any](performance_to_plot.columns[-2:]) + list[Any](performance_to_plot.columns[:-2])
    ]
    plot_from_df(performance_to_plot, title="Daily Closes", figsize=FIGSIZE, colours=colours)
    # more plots
    if print_recent_performance:
        plot_from_df(
            custom_series.iloc[-100:],
            title="Recent performance, last 100 days",
            axis_format="%b-%y",
            figsize=FIGSIZE,
            colours=colours,
        )
    if print_moves_per_date:
        print(performance_to_plot)


def inspect_how_the_series_evolves_from_today(
    series, window_before=15, window_after=30, start_year=2019
):
    from ..visualization.plotly_plots import plot_from_df
    import datetime as dt

    today = dt.datetime.today().date()
    year = today.year
    month = today.month
    day = today.day
    start_dates = [dt.date(year_roll, month, day) for year_roll in range(start_year, year)]
    # construct the df with dates
    df = pd.DataFrame(index=range(-window_before, window_after, 1))
    # add all other years
    for start in start_dates:
        idx = series.loc[start:].index[0]
        idx_num = series.index.get_loc(idx)
        if idx_num - window_before >= 0:
            df[start.year] = series.iloc[
                idx_num - window_before : idx_num + window_after
            ].values
    # add the history for the current year. admittedly it uses a bit of weird coding but hey
    # it's sat night and I cant be bothered to improve it
    df[year] = list[Any](series.iloc[-window_before:].values.reshape(-1)) + [
        np.nan for _ in range(window_after)
    ]
    # normalise
    df -= df.loc[-1]
    # calculate the mean
    df["mean"] = df.mean(axis=1)
    # print
    colours = [
        "lavender",
        "mediumslateblue",
        "slateblue",
        "royalblue",
        "blue",
        "lightcoral",
        "red",
    ]
    colours = colours[-len(df.columns) :]
    colours.reverse()
    plot_from_df(df, title="History starting today", colours=colours, figsize=FIGSIZE)


def display_seasonality_of_the_moving_average(data: pd.DataFrame, period: str = "month"):
    print("\n\nDISPLAY OF ANNUAL SEASONALITY (rec high and pay low)")
    trimmed = [0, 15, 50]
    # define the changes
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data.ffill()
    # we attempt a different approach: we measure the level vs the moving average,
    # the MA has to be long enough that we average across 2 years
    # note that the statistical tests will mean nothing!
    rolling_window = 22 * 24
    changes_raw = (data - data.rolling(rolling_window).mean().shift(1)) / data.rolling(
        rolling_window
    ).std().shift(1)
    changes_raw = _sum_up_changes_within_ranges(changes_raw, period=period)
    changes = changes_raw.dropna(axis=0)
    avg = _calculate_average_changes(changes, trimmed)
    avg_lb = _calculate_average_changes_with_lookback(changes, lookback_years=(15, 7, 3))
    name = data.columns[0]
    # _plot_results(avg, f"{name}: different trimmed means")
    _plot_results(avg_lb, f"{name}: different lookbacks")
    # plot_results(changes.iloc[-3:].T, f"{name}: recent years")


# ==============================================================================
# loops
# ==============================================================================
# inspect seasonality
def search_for_seasonalities(
    df,
    universe,
    period_in_focus=None,
    period="month",
    ROBUST=False,
    significance=0.20,
    START_YEAR=2015,
):
    from ..data.processing import extractDataInFocus

    list_of_seasonality_patterns = []
    for target in universe:
        print("\n\nInspecting", target)
        series = extractDataInFocus(df, [target], START_YEAR=START_YEAR)
        was_seasonality_detected = False
        try:
            p_values, was_seasonality_detected = run_seasonality_test(
                series,
                period=period,
                period_in_focus=period_in_focus,
                significance=significance,
                ROBUST=ROBUST,
                VERBOSE=False,
            )
        except:
            print(
                "I couldnt inspect",
                target,
                ". The function search_for_seasonalities failed, which means that the"
                " invocation of run_seasonality_test failed inside of it",
            )
        if was_seasonality_detected:
            list_of_seasonality_patterns.append(target)
    return list_of_seasonality_patterns
