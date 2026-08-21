```python
"""
Calculate condor weights and price with PCA-based weight methods. Purpose: find curve-neutral fly
structures that are mean-reverting or low volatility. - Method 1: CURVATURE - PCA on raw
levels/changes - Method 2: CONDOR_AL - PCA
"""

import numpy as np
import pandas as pd
import datetime as dt
from functools import lru_cache
from statsmodels.tsa.stattools import adfuller
import pl.finance as pl
from src.utils import project_utils as u
from src import globalVariables as gv

# ==============================================================================
# PARAMETERS
# ==============================================================================

# Starting Parameters - 50bps starting / 10 dv01 per side basis points per shock step for convexity
# calculation - CURVE_SHOCKS_PER_SIDE: number of shock scenarios each direction (total = 2n+1) -
# TARGET_DV01: Target dollar value of the f...
# ...

STARTING_GRANULARITY = 50
CURVE_SHOCKS_PER_SIDE = 3
TARGET_DV01 = 100e3
NOTIONAL = 100e6
MODEL_PCA = 100
MODEL_DVA = 50
SHORT_ROLLING_WINDOW_DAYS = 3 * 22
LONG_ROLLING_WINDOW_DAYS = 12 * 22
SAMPLING_FREQ = "W-FRI"
W_PERIODS_PER_PERIOD = 52
N_QUARTERS_PER_PERIOD = 4
ZSCORE_FUND = 1.5
ZSCORE_FLAW = 0.5
WING_IMBALANCE_THRESHOLD = 0.1
MEAN_REVERSION_CONFIDENCE = 0.90
START_YEAR = "2023"
START_YEAR_PLOT = "2024"
MIN_MANUAL_MR_PAST_CRASH_BUILD = 0
CALENDAR = "TGT"
CURRENCIES = ["USD", "AUD", "NZD", "JPY", "EUR", "GBP", "CAD"]
VERBOSE = False
FIGSIZE = (8, 4)
CHOSEN_CURR_MAP = {
    "USD": "USD SOFR",
    "AUD": "AUD BBR3 6 6M",
    "EUR": "EUR EURIBOR 6 6M",
    "NZD": "NZD BKBBM31 3 3M",
    "CAD": "CAD CDOR 3 3M",
    "JPY": "JPY OIS",
    "GBP": "GBP SONIA",
}

UNIVERSE = [
    "1y1y",
    "1y2y",
    "1y3y",
    "1y4y",
    "1y5y",
    "2y1y",
    "2y2y",
    "2y3y",
    "2y4y",
    "2y5y",
    "3y1y",
    "3y2y",
    "3y3y",
    "3y4y",
    "3y5y",
    "4y1y",
    "4y2y",
    "4y3y",
    "4y4y",
    "4y5y",
    "5y1y",
    "5y2y",
    "5y3y",
    "5y4y",
    "5y5y",
    "7y1y",
    "7y2y",
    "7y3y",
    "7y4y",
    "7y5y",
    "10y1y",
    "10y2y",
    "10y3y",
    "10y4y",
    "10y5y",
]

CONDOR_SWAPS_SET = [
    (1, 2, 3), (1, 3, 5),
    (2, 3, 5), (2, 5, 7), (2, 5, 10),
    (3, 5, 7), (3, 7, 10), (3, 10, 20),
    (5, 7, 10), (5, 10, 15), (5, 10, 20),
    (7, 10, 15), (7, 10, 20),
    (10, 15, 20), (10, 15, 30), (10, 20, 30),
    (15, 20, 30),
]

FOUR_LEG_STRUCTURES = [
    [(1, 2), (3, 5)], [(1, 3), (2, 5)], [(1, 3), (5, 7)],
    [(2, 3), (5, 7)], [(2, 5), (7, 10)], [(2, 5), (10, 15)],
    [(3, 5), (7, 10)], [(3, 7), (10, 15)], [(3, 10), (15, 20)],
    [(5, 7), (10, 15)], [(5, 10), (15, 20)], [(5, 10), (20, 30)],
    [(7, 10), (15, 20)], [(7, 10), (20, 30)],
    [(10, 15), (20, 30)],
]

BLUE_COLORS = ['skyblue', 'darkblue']
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 550)
pd.options.display.float_format = "{:,.2f}".format

# ==============================================================================
# MARKET SETUP
# ==============================================================================
CONV_HOLDER = pl.conventionHolderCreate()

def create_market(run_date):
    """
    Create base market for pricing. """
    mkt_build = pl.marketBuild(
        run_date,
        marketNames=["US:USD", "US:EUR", "US:AUD", "US:NZD", "US:GBP", "US:CAD", "US:JPY"],
        convHolder=CONV_HOLDER,
    )
    return pl.dataBuild(dataBaseFolder=mkt_build)["Market"][0]

@lru_cache(maxsize=128)
def create_framework_with_std_pillars(mkt):
    """
    Create pricing framework with standard pillars. """
    std_pillar = pl.frameworkStandardPillars("STD", CONV_HOLDER)
    pillar_provider = pl.flireStdPillarProvider(
        std_pillar, mkt, CONV_HOLDER, None, None, True
    )
    return pl.flireFramework(
        mkt, pillar_provider, False, True, None, "exact", CONV_HOLDER, None
    )

def shift_market_parallel(base_mkt, rate_shift):
    """
    Parallel shift the market by rate_shift (in decimal). """
    return pl.flireMarketParallelShift(
        base_mkt, CONV_HOLDER, None, rateShift=rate_shift, deltaDynamics="StrictlyOnOis"
    )

def price_swap(
    market,
    currency,
    start,
    end,
    notional=1e6,
    rate=0.0,
    direction="rec",
    option_type="rec",
):
    """
    Price a swap with given parameters."""
    flfw = create_framework_with_std_pillars(market)
    swap = pl.flireSwapCreate(
        tradeType="swap",
        currency=gv.CURRENCY_TO_INDEX[currency],
        expiry=start,
        tenor=end,
        strike=rate,
        direction=direction,
        notional=notional,
        outputType=option_type,
    )
    return flfw.price(swap)

# ==============================================================================
# PRE-CALCULATION
# ==============================================================================
def calculate_pv_and_deltas_for_universe(base_market, currency, notional, direction):
    """
    Pre-calculate PVs and deltas for entire universe under all shock scenarios."""
    swaps = [(f"{a}y", f"{b}y") for a, b in [elem.split('y')[:2] for elem in UNIVERSE]]
    shocks = []
    # STARTING_GRANULARITY
    for i in range(-CURVE_SHOCKS_PER_SIDE, CURVE_SHOCKS_PER_SIDE + 1):
        shocks.append(i * STARTING_GRANULARITY / 10000)

    swap_names = [start + end for start, end in swaps]
    pvs = pd.DataFrame(index=swap_names, columns=[f"{s}" for s in shocks])
    mid_rates = pd.DataFrame(index=swap_names, columns=["mid"])
    deltas = pd.DataFrame(index=swap_names, columns=shocks)
    
    for shock in shocks:
        if shock == 0:
            bumped_market = base_market
        else:
            bumped_market = shift_market_parallel(base_market, shock / 1e4)

        for start, end in swaps:
            swap_name = start + end
            if shock == 0:
                mid_rate = price_swap(
                    base_market, currency, start, end, notional, 0.0, direction, "rate"
                )
                mid_rates.loc[swap_name, "mid"] = mid_rate

            pv = price_swap(
                bumped_market, currency, start, end, notional, mid_rate, direction, "pv"
            )
            pvs.loc[swap_name, f"{shock}"] = pv
            
            delta = price_swap(
                bumped_market, currency, start, end, notional, mid_rate, direction, "delta"
            )
            deltas.loc[swap_name, shock] = delta

    return pvs, deltas, mid_rates

# ==============================================================================
# WEIGHT CALCULATION METHODS
# ==============================================================================
def make_dv01_neutral(weights, dv01s):
    """
    Project weights onto DV01-neutral subspace while preserving fly direction. For a assets,
    DV01-neutrality requires: Sum(w_i * dv01_i) = 0 so adjust weights minimally to satisfy this
    constraint: w_neutral = w - (w.T*d / d.T*d) * d."""

    w = np.array(weights, dtype=float)
    d = np.array(dv01s, dtype=float)

    # Project out the DV01 component
    dot_wd = np.dot(w, d)
    dot_dd = np.dot(d, d)
    if dot_dd > 1e-10:
        w_neutral = w - (dot_wd / dot_dd) * d
    else:
        w_neutral = w # cannot neutralize if DV01s are all zero

    # Re-normalize: preserves belly sum for flies (middle elements)
    # A B C structure => w_original = [w1, w2, w3]
    if len(w) == 3:
        # 3-leg fly: normalize so |w2| is same as original belly weight magnitude
        original_belly = abs(w[1])
        if abs(w_neutral[1]) > 1e-8 and original_belly > 0:
            scale = original_belly / abs(w_neutral[1])
            w_neutral *= scale
    elif len(w) == 4:
        # 4-leg: preserve belly sum magnitude
        original_belly_sum = abs(w[1]) + abs(w[2])
        new_belly_sum = abs(w_neutral[1]) + abs(w_neutral[2])
        if new_belly_sum > 1e-8 and original_belly_sum > 0:
            scale = original_belly_sum / new_belly_sum
            w_neutral *= scale
    return w_neutral

def extract_condor_slope_weights_d1ag(data_matrix, lookback=MODEL_PCA):
    """
    Method 2 (CONDOR_AL): PCA on weekly changes of (D-A) and (C-B). """
    
    if data_matrix.shape[1] != 4:
        return None
        
    weekly_levels = data_matrix.resample(SAMPLING_FREQ).last().dropna()
    level_changes = weekly_levels.diff(1).iloc[1:]  # 1w for range(4)
    slope_changes = pd.DataFrame(index=level_changes.index)
    
    slope_changes['A'] = level_changes.iloc[:, 3] - level_changes.iloc[:, 0]
    slope_changes['B'] = level_changes.iloc[:, 2] - level_changes.iloc[:, 1]

    pca_vector, _, _ = u.sensitivity_measure.pca(slope_changes.iloc[-lookback:], component=1)
    a, b = pca_vector
    a, b = float(a), float(b)

    if abs(a) < 1e-10:
        a = 1.0
    if a < 0:
        a, b = -a, -b

    weights = np.array([-a, -b, b, a], dtype=float)

    if weights[1] > 0:
        weights *= -1

    return weights


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================

def calculate_std_all_data_weights(weights, names, mkt_data=None):
    """
    Calculate weekly standard deviation of the weighted structure."""
    df = mkt_data.resample(SAMPLING_FREQ).last()
    df = df[names].dropna()
    series = (df * weights).sum(axis=1)
    series_diff = series.diff().dropna()
    std = series_diff.std()
    return std, series

def check_mean_reverting(series, confidence_level=MEAN_REVERSION_CONFIDENCE):
    """
    Check if series is mean-reverting using ADF test. """
    try:
        critical_values = {0.90: -2.57, 0.95: -2.86, 0.99: -3.43}
        
        adf_result = adfuller(series.dropna(), autolag="AIC")
        adf_stat = adf_result[0]
        
        is_mr = adf_stat < critical_values.get(confidence_level, -2.57)
        return is_mr, adf_stat
    except Exception:
        return False, np.nan

def calculate_carry(
    base_market, currency, swaps, weights, notional, direction, verbose=False
):
    """
    Calculate carry/roll for the structure."""
    
    live_rates = [
        price_swap(base_market, currency, f"{s.split('y')[0]}y", f"{s.split('y')[1]}y", notional, 'x', direction, "rate")
        for s in swaps
    ]

    rolled_rates = [
        price_swap(
            base_market, currency, f"{int(s.split('y')[0])-1}y", f"{int(s.split('y')[1])-1}y", notional, 'x', direction, "rate"
        )
        for s in swaps
    ]

    live_swaps = np.sum(np.array(live_rates) * weights)
    rolled_swaps = np.sum(np.array(rolled_rates) * weights)

    if verbose:
        print(
            f"flies live and rolled are: {live_swaps} and {rolled_swaps}, hence the carry is {rolled_swaps - live_swaps}"
        )

    return live_swaps - rolled_swaps


def calculate_fly_metrics(
    base_market,
    currency,
    chosen_swaps,
    weights,
    raw_data,
    cr_data,
    pvs,
    deltas,
    mid_rates,
    direction="rec",
    verbose=VERBOSE,
):
    """
    Calculate all metrics for a fly structure using pre-calculated PVs/deltas."""

    chosen_swaps_with_y = [(f"{s.split('y')[0]}y", f"{s.split('y')[1]}y") for s in chosen_swaps]
    chosen_swap_names = [start + end for start, end in chosen_swaps_with_y]
    short_currency = currency if currency != "USD" else ""
    chosen_swap_names_with_curr = [short_currency + name for name in chosen_swap_names]
    structure_name = "_".join(chosen_swap_names_with_curr)
    
    if verbose:
        print(f"
=======================================================")
        print(f"Inspecting {chosen_swap_names_with_curr} with {weights} weights and ")
    
    series = u.construct_series(
        raw_data, chosen_swap_names_with_curr, weights, name=structure_name
    ).dropna()
    if verbose:
        print(f"The chosen weights are: {', '.join([str(f'{w:.2f}') for w in weights])}")

    is_mean_reverting, _ = check_mean_reverting(series)
    
    short_zscore = (
        series - series.rolling(SHORT_ROLLING_WINDOW_DAYS).mean()
    ) / series.rolling(SHORT_ROLLING_WINDOW_DAYS).std()
    long_zscore = (
        series - series.rolling(LONG_ROLLING_WINDOW_DAYS).mean()
    ) / series.rolling(LONG_ROLLING_WINDOW_DAYS).std()

    carry_roll = calculate_carry(
        base_market, currency, chosen_swaps, weights, NOTIONAL, direction, verbose
    )
    cr_series = u.construct_series(
        cr_data, chosen_swap_names_with_curr, weights, name=structure_name
    ).dropna()

    cr_from_ficdb = (
        cr_series.iloc[-1] * N_QUARTERS_PER_PERIOD if len(cr_series) > 0 else 0.0
    )

    if verbose:
        print(f"3m 6m spread carry is {carry_roll:.2f} from ISG, and {cr_from_ficdb:.2f} from FICDB")

    std_adjusted, _ = calculate_std_all_data_weights(weights, chosen_swap_names_with_curr, mkt_data=raw_data)
    std_adjusted = std_adjusted * np.sqrt(W_PERIODS_PER_PERIOD)
    sharpe = cr_from_ficdb / std_adjusted if std_adjusted > 0 else 0.0
    if verbose:
        print(f"Annualized Standard Deviation: {std_adjusted:.2f}")
        print(f"Annualized sharpe: {sharpe:.2f}")

    notional_weights = (
        TARGET_DV01 * np.array(weights) / deltas.loc[chosen_swap_names, 0].values
    )
    pv_for_structure = pvs.loc[chosen_swap_names].values
    pv_for_structure_total = (pv_for_structure * notional_weights.reshape(-1, 1))
    pv_for_structure_total = pd.DataFrame(pv_for_structure_total, columns=pvs.columns)
    pv_for_structure_total_loc = pv_for_structure_total.sum(axis=0) - TARGET_DV01

    symmetric_shocks = sorted(
        [
            float(s) for s in pv_for_structure_total_loc.index
            if float(s) > 0 and str(-float(s)) in pv_for_structure_total_loc.index
        ]
    )

    conv_pv = (
        float(
            np.sum(
                [
                    pv_for_structure_total_loc[str(s)] + pv_for_structure_total_loc[str(-s)]
                    for s in symmetric_shocks
                ]
            )
        )
        if symmetric_shocks
        else 0.0
    )

    conv_carry_ratio = conv_pv / cr_from_ficdb if abs(cr_from_ficdb) > 0 else 0.0
    series_dropna = series.dropna()

    return {
        "structure": structure_name,
        "z_3m": short_zscore.iloc[-1] if len(short_zscore.dropna()) > 0 else np.nan,
        "z_12m": long_zscore.iloc[-1] if len(long_zscore.dropna()) > 0 else np.nan,
        "is_mean_reverting": is_mean_reverting,
        "carry": carry_roll,
        "sharpe": float(sharpe),
        "conv": float(conv_pv),
        "conv/carry": float(conv_carry_ratio),
        "chg_3m": series_dropna.diff().iloc[-1] if len(series_dropna) > 0 else np.nan,
        "chg_6m": series_dropna.diff(12).iloc[-1] if len(series_dropna) > 12 else np.nan,
        "chg_1y": series_dropna.diff(52).iloc[-1] if len(series_dropna) > 52 else np.nan,
        "ficdb_carry": float(cr_from_ficdb),
        "std": float(std_adjusted),
        "series": series
    }

def flag_fly_structure(sharpe, z_short, z_long, conv_carry, chg_3m_6m):
    """
    Flag structures based on sharpe and z-score alignment. """
    
    flag = ""
    if abs(sharpe) > ZSCORE_FUND        and np.sign(sharpe) == np.sign(z_short)        and np.sign(z_short) == np.sign(z_long):
        flag = "S"

    if abs(z_long) > ZSCORE_FLAW        and np.sign(z_short) == np.sign(z_long)        and np.sign(z_long) == np.sign(chg_3m_6m):
        flag += "T"

    return flag

# ==============================================================================
# HELPERS
# ==============================================================================
def build_result_row(metrics, weights, n_legs):
    """
    Build a single result row dict from metrics."""
    
    row = {
        "flag": flag_fly_structure(
            metrics["sharpe"],
            metrics["z_3m"],
            metrics["z_12m"],
            metrics["conv/carry"],
            metrics["chg_3m"],
        ),
        "z_3m": metrics["z_3m"],
        "z_12m": metrics["z_12m"],
        "sharpe": metrics["sharpe"],
        "conv": metrics["conv"],
        "carry": metrics["carry"],
        "conv/carry": metrics["conv/carry"],
        "w1": weights[0],
        "w2": weights[1],
        "w3": weights[2],
        "w4": weights[3] if n_legs == 4 else np.nan,
        "chg_3m": metrics["chg_3m"],
        "chg_6m": metrics["chg_6m"],
        "chg_1y": metrics["chg_1y"],
        "ficdb_carry": metrics["ficdb_carry"],
        "mean_rev": "Y" if metrics["is_mean_reverting"] else "N",
        "std": metrics["std"],
    }
    return row

def parse_structure_name(structure_name):
    """
    Parse structure name into currency and leg names."""
    parts = structure_name.split("_")
    
    first_part = parts[0]
    curr = first_part[:3] if not first_part[0].isdigit() else "USD"
    
    legs = [first_part.replace(curr, "")] + parts[1:] if not first_part[0].isdigit() else parts
    
    return curr, [curr + "_" + leg for leg in legs] if curr != "USD" else legs

def process_structures(
    structures,
    base_market,
    currency,
    cr_data,
    weights_fn,
    raw_data,
    notional,
    pvs,
    deltas,
    mid_rates,
    direction="rec",
    verbose=VERBOSE,
    dv01_neutral=False,
):
    """
    Process a set of structures with a given weight function."""
    
    short_currency = currency if currency != "USD" else ""
    results = []
    
    for legs in structures:
        chosen_swaps = [f"{x}y{y}y" for x, y in legs] if type(legs[0]) is tuple else [f"{l}y" for l in legs]
        if type(legs[0]) is not tuple: 
             # 3-leg tuple (1, 2, 3) mapped to ['1y2y', '2y3y', '1y3y' etc] -> needs correct mapping
             chosen_swaps = [f"{legs[0]}y{legs[1]}y", f"{legs[1]}y{legs[2]}y", f"{legs[0]}y{legs[2]}y"]
             
        chosen_swap_names = chosen_swaps
        chosen_swap_names_with_curr = [short_currency + name for name in chosen_swap_names]
        structure_name = "_".join(chosen_swap_names_with_curr)
        
        if verbose:
            print(f"Inspecting {chosen_swap_names_with_curr}...")
            
        if all(col in raw_data.columns for col in chosen_swap_names_with_curr):
            try:
                data_subset = raw_data.loc[:, chosen_swap_names_with_curr]
                weights = weights_fn(data_subset, lookback=MODEL_PCA)
                
                if weights is None:
                    continue
                
                # Apply DV01 Neutralization if requested
                if dv01_neutral:
                    try:
                        dv01s = deltas.loc[chosen_swap_names, 0].values
                        weights = make_dv01_neutral(weights, dv01s)
                    except Exception:
                        pass # keep original weights if DV01 lookup fails
                        
                weights = np.round(weights, 4)
                metrics = calculate_fly_metrics(
                    base_market,
                    currency,
                    chosen_swaps,
                    weights,
                    raw_data,
                    cr_data,
                    pvs,
                    deltas,
                    mid_rates,
                    direction=direction,
                    verbose=verbose,
                )
                
                results.append(
                    {
                        "structure": metrics["structure"],
                        "series": metrics["series"] if "series" in metrics else None,
                        "row": build_result_row(metrics, weights, len(legs)),
                    }
                )
            except Exception as e:
                if verbose:
                    print(f"Error processing {structure_name}: {e}")
                pass
                
    return results

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def analyze_swap_flies(raw_data, cr_data, verbose=VERBOSE):
    """
    Main analysis for 3-leg flies and 4-leg condors."""
    
    today_date = dt.date.today()
    curve_valuation_day = pl.dateAddByCalendar(
        today_date, f"{MIN_MANUAL_MR_PAST_CRASH_BUILD}b", calendar=CALENDAR
    )
    
    base_market = create_market(curve_valuation_day)
    
    curvature_rows, condor_4leg_rows = [], []
    curvature_df_list, condor_4leg_df_list = [], []
    
    if verbose:
        print("Valuing fly is assumed to be 'rec'")
        
    for currency in CURRENCIES:
        print(f"\n==============================================")
        print(f"--> Processing {currency} (cal {CALENDAR})")
        
        # 0) SETUP: Data, calculate pv_and_deltas for universe
        short_currency = currency if currency != "USD" else ""
        direction = "rec"
        pvs, deltas, mid_rates = calculate_pv_and_deltas_for_universe(
            base_market, currency, NOTIONAL, direction
        )
        
        # METHOD 1: CURVATURE (3-leg) - NO DV01 neutralization (matches original behavior)
        if verbose: 
            print(f"-> METHOD 1: CURVATURE (3-leg)")
            
        result1 = process_structures(
            CONDOR_SWAPS_SET,
            base_market,
            currency,
            cr_data,
            lambda df, l: pl.extractWeights_Method1(df, l),
            raw_data,
            NOTIONAL,
            pvs,
            deltas,
            mid_rates,
            direction=direction,
            verbose=verbose,
            dv01_neutral=False,
        )
        
        for res in result1:
            curvature_rows.append({"structure": res["structure"], **res["row"]})
            if "series" in res and res["series"] is not None:
                curvature_df_list.append(res["series"].to_frame(name=res["structure"]).squeeze())
                
        # METHOD 2: CONDOR_AL (4-leg) - PCA on (D-A vs C-B)
        if verbose:
            print(f"-> METHOD 2: CONDOR_AL (4-leg)")
            
        result2 = process_structures(
            FOUR_LEG_STRUCTURES,
            base_market,
            currency,
            cr_data,
            extract_condor_slope_weights_d1ag,
            raw_data,
            NOTIONAL,
            pvs,
            deltas,
            mid_rates,
            direction=direction,
            verbose=verbose,
            dv01_neutral=True,
        )
        
        for res in result2:
            condor_4leg_rows.append({"structure": res["structure"], **res["row"]})
            if "series" in res and res["series"] is not None:
                condor_4leg_df_list.append(res["series"].to_frame(name=res["structure"]).squeeze())

    # Build DataFrames
    curvature_df, condor_4leg_df = pd.DataFrame(), pd.DataFrame()
    if curvature_rows:
        curvature_df = pd.DataFrame.from_dict(
            {idx: row for idx, row in enumerate(curvature_rows)}, orient="index"
        )
    if condor_4leg_rows:
        condor_4leg_df = pd.DataFrame.from_dict(
            {idx: row for idx, row in enumerate(condor_4leg_rows)}, orient="index"
        )

    if not curvature_df.empty:
        curvature_df["abs_z_3m"] = curvature_df["z_3m"].abs()
        curvature_df["abs_z_12m"] = curvature_df["z_12m"].abs()
        curvature_df["z_diff"] = (curvature_df["z_12m"] - curvature_df["z_3m"]).abs()
        wing_imbalance = (curvature_df["w3"] - curvature_df["w1"]).abs()
        curvature_df["wing_imbalance"] = wing_imbalance
        curvature_df = curvature_df[
            wing_imbalance < WING_IMBALANCE_THRESHOLD
        ]
        curvature_df = curvature_df.sort_values(by="sharpe", ascending=False)
        
    if not condor_4leg_df.empty:
        condor_4leg_df["abs_z_3m"] = condor_4leg_df["z_3m"].abs()
        condor_4leg_df["abs_z_12m"] = condor_4leg_df["z_12m"].abs()
        condor_4leg_df["z_diff"] = (condor_4leg_df["z_12m"] - condor_4leg_df["z_3m"]).abs()
        wing_imbalance = (condor_4leg_df["w4"] - condor_4leg_df["w1"]).abs() # difference between |w1| and |w4|
        condor_4leg_df["wing_imbalance"] = wing_imbalance
        condor_4leg_df = condor_4leg_df[
            wing_imbalance < WING_IMBALANCE_THRESHOLD
        ]
        condor_4leg_df = condor_4leg_df.sort_values(by="sharpe", ascending=False)

    col_order_3leg = [
        "flag",
        "structure",
        "z_3m",
        "z_12m",
        "sharpe",
        "carry",
        "conv",
        "conv/carry",
        "w1",
        "w2",
        "w3",
        "chg_3m",
        "chg_6m",
        "chg_1y",
        "ficdb_carry",
        "mean_rev",
    ]
    col_order_4leg = [
        "flag",
        "structure",
        "z_3m",
        "z_12m",
        "sharpe",
        "carry",
        "conv",
        "conv/carry",
        "w1",
        "w2",
        "w3",
        "w4",
        "chg_3m",
        "chg_6m",
        "chg_1y",
        "ficdb_carry",
        "mean_rev",
    ]

    if not curvature_df.empty:
        curvature_df = curvature_df[[c for c in col_order_3leg if c in curvature_df.columns]]
    if not condor_4leg_df.empty:
        condor_4leg_df = condor_4leg_df[[c for c in col_order_4leg if c in condor_4leg_df.columns]]

    curvature_series = pd.DataFrame(curvature_df_list).T if curvature_df_list else pd.DataFrame()
    condor_4leg_series = pd.DataFrame(condor_4leg_df_list).T if condor_4leg_df_list else pd.DataFrame()

    return curvature_df, curvature_series, condor_4leg_df, condor_4leg_series

# ==============================================================================
# OUTPUT
# ==============================================================================
def print_and_plot_results(results_df, data, adj_data, method_name, n_legs=3):
    """
    Print results and plot flagged series for a method."""
    
    if results_df.empty:
        print(f"\n{'=' * 100}")
        print(f"METHOD: {method_name}")
        print("0 results found.")
        print(f"{'=' * 100}")
        return

    print(f"\n{'=' * 100}")
    print(f"METHOD: {method_name}")
    print(f"{len(results_df)} results found.")
    print(f"{'=' * 100}")
    print("If annualized Sharpe is positive, it's positive carry to receive.")
    print("This is defined as annual carry / (weekly std * sqrt(N_PERIODS_STD_ADJUSTMENT))")
    print("\n-------------------------------------------------------------")
    print("Conv/carry compares symmetric shocked PV (approx convexity) vs annual carry")
    print("-------------------------------------------------------------")
    print("Positive and larger means convexity helps relative to carry; negative means you are")
    print("short convexity.")
    print("\n-------------------------------------------------------------")
    print(f"Note that the wing imbalance threshold is: {WING_IMBALANCE_THRESHOLD}")
    print("-------------------------------------------------------------")
    
    non_empty_flags_num = len(results_df[results_df["flag"] != ""])
    print(f"Flagged series: {non_empty_flags_num} out of {len(results_df)}")
    
    sorted_results = results_df.sort_values(by="sharpe", ascending=False)
    print("\nTop 5 results by sharpe:")
    print(sorted_results.head(5).to_string(index=False))

    non_empty_flags = sorted_results[sorted_results["flag"] != ""]
    if not non_empty_flags.empty:
        for index, row in non_empty_flags.iterrows():
            structure_name = row["structure"]
            direction = "Rec" if row["sharpe"] > 0 else "Pay"
            
            print("\n-------------------------------------------------------------")
            print(f"{direction} {structure_name} with sharpe: {row['sharpe']:.2f} & carry & roll: {row['carry']:.2f}")
            print("-------------------------------------------------------------")
            
            u.plot_from_df(
                data,
                [structure_name],
                loc=[START_YEAR_PLOT, dt.date.today().strftime("%Y-%m-%d")],
                title=structure_name,
                figsize=FIGSIZE
            )

            curr, legs = parse_structure_name(structure_name)
            weights = [
                row["w1"],
                row["w2"],
                row["w3"],
            ]
            if n_legs == 4:
                weights.append(row["w4"])
                
            series = u.construct_series(adj_data, legs, weights, name=structure_name)
            
            title_adj = (
                f"{structure_name} (Carry Adjusted) with sharpe: \n"
                f"{row['sharpe']:.2f} carry & roll {row['carry']:.2f}"
            )
            
            u.plot_from_df(
                series.to_frame(),
                [structure_name],
                title=title_adj,
                colors=BLUE_COLORS,
                figsize=FIGSIZE
            )
            print("\n")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    raw_data = pd.read_parquet(gv.AllData_file_Path)
    cr_data = pd.read_parquet(gv.CarryRoll_Data_file_Path)
    adj_data = pd.read_parquet(gv.AllData_Adj_file_Path)
    
    curvature_df, curvature_data, condor_4leg_df, condor_4leg_data = analyze_swap_flies(
        raw_data, cr_data, verbose=VERBOSE
    )
    
    print_and_plot_results(
        curvature_df,
        curvature_data,
        adj_data,
        "METHOD 1: CURVATURE (PCA on raw levels)",
        n_legs=3,
    )
    
    print_and_plot_results(
        condor_4leg_df,
        condor_4leg_data,
        adj_data,
        "METHOD 2: CONDOR_AL (PCA on D-A vs C-B)",
        n_legs=4,
    )
```