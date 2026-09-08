jordan:
"""Syntax-aware reconstruction of 3_visualization.py from the supplied video."""
import base64 from datetime import date, datetime
import numpy as np import pandas as pd import plotly.graph_objects as go from IPython.display import HTML, display from plotly.subplots import make_subplots
pd.set_option("display.max_columns", 50) pd.set_option("display.max_rows", 100)
PARQUET_PATH = "processed.parquet" OUTPUT_FILENAME = "file.parquet" PLOTLY_CONFIG = {"modeBarButtonsToRemove": ["select2d", "lasso2d"]} DPI = 100 YEAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"] INDUSTRY_COLORS = [ "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", ] MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SETTINGS
ANALYSIS = "GB" FIGSIZE = (12, 6) FIGSIZE_LARGER = (16, 8) N_GROUPS = 5 FOCUS_MONTH = 1 GROUP_BY = "INDUSTRY_GROUP" # or INDUSTRY_SUBGROUP / ISSUER_INDUSTRY N_TOP_INDUSTRIES = 5 N_TOP_ISSUERS = 5 RUN_ANALYSIS = False
def apply_figsize(figsize, height_default, width_default=None): """Convert a Matplotlib-style figsize to Plotly pixel dimensions.""" return ((int(figsize[0] * DPI), int(figsize[1] * DPI)) if figsize else (width_default, height_default))
def get_colorscale_params(values, colorscale="RdYlGn", reversescale=True): """Set the colour range, centring diverging values around zero.""" min_val = min(values) if values else 0 max_val = max(values) if values else 0 if min_val < 0: abs_max = max(abs(min_val), abs(max_val)) return colorscale, reversescale, -abs_max, abs_max return colorscale, reversescale, min_val, max_val
def create_pivots(df, scale=1e9): """Create standard month-by-year pivot tables for amount and DV01.""" pivot_amt = ( df.pivot_table(values="AMT_ISSUED_USD", index="MONTH", columns="YEAR", aggfunc="sum") .fillna(0).reindex(range(1, 13)).fillna(0) / scale ) pivot_dv01 = ( df.pivot_table(values="DV01_MM", index="MONTH", columns="YEAR", aggfunc="sum") .fillna(0).reindex(range(1, 13)).fillna(0) ) return pivot_amt, pivot_dv01
def plot_heatmap( pivot, title, x_label="X", y_label="Y", reverse_x=True, reverse_y=False, colorscale="RdYlGn", reversescale=True, decimals=2, figsize=None, height=600, width=None, show_text=True, font_size=10, reverse_yaxis=False, ): x_vals = sorted(pivot.columns, reverse=reverse_x) y_vals = sorted(pivot.index, reverse=reverse_y) ordered = pivot.loc[y_vals, x_vals] cs, rs, zmin, zmax = get_colorscale_params( ordered.values.flatten().tolist(), colorscale, reversescale ) params = { "z": ordered.values, "x": [str(x) for x in x_vals], "y": [str(y) for y in y_vals], "colorscale": cs, "reversescale": rs, "zmin": zmin, "zmax": zmax, "showscale": True, "hovertemplate": ( f"{y_label}: %{{y}}
{x_label}: %{{x}}
" f"Value: %{{z:.{decimals}f}}<extra></extra>" ), } if show_text: params.update( text=ordered.round(decimals).values, texttemplate="%{text}", textfont={"size": font_size}, ) fig = go.Figure(data=go.Heatmap(**params)) width_px, height_px = apply_figsize(figsize, height, width) layout = { "title": {"text": title, "x": 0.5, "xanchor": "center"}, "xaxis_title": x_label, "yaxis_title": y_label, "height": height_px, "template": "plotly_white", "font": {"size": 11}, } if width_px: layout["width"] = width_px if reverse_yaxis: layout["yaxis"] = {"autorange": "reversed"} fig.update_layout(**layout) fig.show(config=PLOTLY_CONFIG)
def plot_bar( pivot, title, ylabel, n_groups=5, reverse_groups=True, colors=None, x_label="Category", figsize=None, height=500, width=None, barmode="group", show_legend=True, orientation="v", ): groups = sorted(pivot.columns, reverse=reverse_groups)[:n_groups] categories = sorted(pivot.index) colors = colors or YEAR_COLORS fig = go.Figure() for i, group in enumerate(groups): values = [ pivot.loc[category, group] if category in pivot.index and group in pivot.

jordan:
columns else 0 for category in categories ] params = { "name": str(group), "marker_color": colors[i % len(colors)], "hovertemplate": ( f"Group: {group}
{x_label}: %{{x}}
" f"{ylabel}: %{{y:.2f}}<extra></extra>" ), } if orientation == "v": params.update(x=categories, y=values) else: params.update(x=values, y=categories) fig.add_trace(go.Bar(**params)) width_px, height_px = apply_figsize(figsize, height, width) axis = {"tickmode": "array", "tickvals": categories, "ticktext": [str(c) for c in categories]} layout = { "title": {"text": title, "x": 0.5, "xanchor": "center"}, "barmode": barmode, "template": "plotly_white", "height": height_px, "showlegend": show_legend, "legend": {"title": {"text": "Group"}, "traceorder": "normal"}, "font": {"size": 11}, } if width_px: layout["width"] = width_px if orientation == "v": layout.update(xaxis_title=x_label, yaxis_title=ylabel, xaxis=axis) else: layout.update(xaxis_title=ylabel, yaxis_title=x_label, yaxis=axis) fig.update_layout(**layout) fig.show(config=PLOTLY_CONFIG)
def plot_standard_charts(pivot_amt, pivot_dv01, label, figsize=(12, 6), n_groups=5): """Plot heatmaps and bar charts for amount and DV01.""" for pivot, metric, ylabel in [ (pivot_amt, "Notional", "Notional (bln USD)"), (pivot_dv01, "DV01", "DV01 (mm)"), ]: print(f"\n{'=' * 60}\n{label} - {ylabel}\n{'=' * 60}") plot_heatmap( pivot, f"{label} - {ylabel}", x_label="Year", y_label="Month", figsize=figsize, reverse_yaxis=True, ) plot_bar( pivot, f"{label} - {metric} by Month", ylabel, n_groups=n_groups, x_label="Month", figsize=figsize, )
def plot_industry_charts( df, label, figsize=(12, 6), n_years=5, group_col="ISSUER_INDUSTRY", ): """Plot stacked monthly industry bars for each recent year.""" years = sorted(df["YEAR"].dropna().unique(), reverse=True)[:n_years] recent = df[df["YEAR"].isin(years)] industries = ( recent.groupby(group_col)["AMT_ISSUED_USD"].sum() .sort_values(ascending=False).index.tolist() ) months = list(range(1, 13)) bar_width = 0.15 offsets = np.linspace( -bar_width * (len(years) - 1) / 2, bar_width * (len(years) - 1) / 2, len(years), ) for metric, metric_col, scale, ylabel in [ ("Notional", "AMT_ISSUED_USD", 1e9, "Notional (bln USD)"), ("DV01", "DV01_MM", 1, "DV01 (mm)"), ]: fig = go.Figure() pivots, cumulative_bases, bar_tops = {}, {}, {} for year in years: pivot = ( recent[recent["YEAR"] == year].pivot_table( values=metric_col, index="MONTH", columns=group_col, aggfunc="sum", ).fillna(0).reindex(range(1, 13)).fillna(0) / scale ) pivots[year] = pivot cumulative = np.zeros(12) cumulative_bases[year] = {} for industry in industries: cumulative_bases[year][industry] = cumulative.copy() if industry in pivot.columns: cumulative += pivot[industry].values bar_tops[year] = cumulative for industry_index, industry in enumerate(industries): first_trace = True for year_index, year in enumerate(years): if industry not in pivots[year].columns: continue fig.add_trace(go.Bar( name=industry if first_trace else None, x=[month + offsets[year_index] for month in months], y=pivots[year][industry].values, base=cumulative_bases[year][industry], width=bar_width, marker_color=INDUSTRY_COLORS[industry_index % len(INDUSTRY_COLORS)], legendgroup=industry, showlegend=first_trace, hovertemplate=( f"<b>{year}</b><br>Month: %{{x:.0f}}
{industry}
" f"{ylabel}: %{{y:.2f}}<extra></extra>" ), )) first_trace = False annotations = [ dict( x=month + offsets[i], y=bar_tops[year][month - 1], text=f"<b>{year}</b>", showarrow=False, yshift=10, font={"size": 10}, ) for i, year in enumerate(years) for month in months ] width_px, height_px = apply_figsize(figsize, 600) fig.update_layout( title={"text": f"{label} - {metric} by Industry", "x": 0.5, "xanchor": "center"}, xaxis_title="Month", yaxis_title=ylabel, barmode="overlay", template="plotly_white", height=height_px, width=width_px, annotations=annotations, xaxis={"tickmode": "array", "tickvals": months, "ticktext": [str(m) for m in months]}, legend={"title": {"text": group_col.replace("_", " ").

jordan:
title()}, "traceorder": "normal"}, font={"size": 11}, updatemenus=[dict( type="buttons", direction="left", x=0, y=1.15, buttons=[ dict(label="Clear All", method="restyle", args=[{"visible": "legendonly"}]), dict(label="Show All", method="restyle", args=[{"visible": True}]), ], )], ) fig.show(config=PLOTLY_CONFIG)
def plot_net_by_industry( offshore_df, kanga_df, metric_col, scale, ylabel, title, analysis="AU", figsize=(14, 8), n_years=5, group_col="ISSUER_INDUSTRY", ): """Plot offshore above zero, Kanga below zero, and the net as a line.""" all_data = pd.concat([offshore_df, kanga_df], ignore_index=True) years = sorted(all_data["YEAR"].dropna().unique(), reverse=True)[:n_years] industries = ( all_data.groupby(group_col)[metric_col].sum() .sort_values(ascending=False).index.tolist() ) months = list(range(1, 13)) bar_width = 0.08 pair_width = bar_width * 2.2 total_width = pair_width * len(years) pivots = {}

for year in years:
    pivots[year] = {
        "offshore": (
            offshore_df[offshore_df["YEAR"] == year].pivot_table(
                values=metric_col, index="MONTH", columns=group_col,
                aggfunc="sum",
            ).fillna(0).reindex(range(1, 13)).fillna(0) / scale
        ),
        "kanga": (
            kanga_df[kanga_df["YEAR"] == year].pivot_table(
                values=metric_col, index="MONTH", columns=group_col,
                aggfunc="sum",
            ).fillna(0).reindex(range(1, 13)).fillna(0) / scale
        ),
    }

fig = go.Figure()
bar_trace_indices = []
positive_tops = {}

for industry_index, industry in enumerate(industries):
    first_trace = True
    for year_index, year in enumerate(years):
        offshore_pivot = pivots[year]["offshore"]
        kanga_pivot = pivots[year]["kanga"]
        year_offset = (
            -total_width / 2 + pair_width / 2 + year_index * pair_width
        )
        offshore_base = sum(
            (offshore_pivot[item].values for item in industries[:industry_index]
             if item in offshore_pivot.columns),
            np.zeros(12),
        )
        kanga_base = -sum(
            (kanga_pivot[item].values for item in industries[:industry_index]
             if item in kanga_pivot.columns),
            np.zeros(12),
        )
        for pivot, base, x_offset, side in [
            (offshore_pivot, offshore_base, -bar_width / 2, "Offshore"),
            (kanga_pivot, kanga_base, bar_width / 2, "Kanga"),
        ]:
            if industry not in pivot.columns:
                continue
            values = pivot[industry].values
            if side == "Kanga":
                values = -values
            bar_trace_indices.append(len(fig.data))
            fig.add_trace(go.Bar(
                name=industry if first_trace else None,
                x=[m + year_offset + x_offset for m in months],
                y=values,
                base=base,
                width=bar_width,
                marker_color=INDUSTRY_COLORS[
                    industry_index % len(INDUSTRY_COLORS)
                ],
                legendgroup=industry,
                showlegend=first_trace,
                hovertemplate=(
                    f"<b>{year} {side}</b><br>Month: %{{x:.0f}}<br>"
                    f"{industry}<br>{ylabel}: %{{y:.2f}}<extra></extra>"
                ),
            ))
            first_trace = False

for year_index, year in enumerate(years):
    offshore_pivot = pivots[year]["offshore"]
    kanga_pivot = pivots[year]["kanga"]
    offshore_total = offshore_pivot.sum(axis=1).values
    kanga_total = kanga_pivot.sum(axis=1).values
    net_values = offshore_total - kanga_total
    positive_tops[year] = offshore_total
    year_offset = -total_width / 2 + pair_width / 2 + year_index * pair_width
    fig.add_trace(go.Scatter(
        name=f"Net ({year})",
        x=[m + year_offset for m in months],
        y=net_values,
        mode="lines+markers+text",
        line={"color": YEAR_COLORS[year_index % len(YEAR_COLORS)],
              "width": 3},
        marker={"size": 7},
        text=[f"{value:.1f}" for value in net_values],

jordan:
textposition="middle right",
        legendgroup=f"net-{year}",
        hovertemplate=(
            f"<b>Net {year}</b><br>Month: %{{x}}<br>"
            f"{ylabel}: %{{y:.2f}}<extra></extra>"
        ),
    ))

annotations = [
    dict(
        x=month + (-total_width / 2 + pair_width / 2 + i * pair_width),
        y=positive_tops[year][month - 1], text=f"<b>{year}</b>",
        showarrow=False, yshift=20, font={"size": 9},
    )
    for i, year in enumerate(years) for month in months
]
width_px, height_px = apply_figsize(figsize, 720)
all_visible = [True] * len(fig.data)
bars_hidden = [i not in bar_trace_indices for i in range(len(fig.data))]
fig.update_layout(
    title={"text": title, "x": 0.5, "xanchor": "center"},
    xaxis_title="Month", yaxis_title=ylabel, barmode="overlay",
    template="plotly_white", height=height_px, width=width_px,
    annotations=annotations,
    xaxis={"tickmode": "array", "tickvals": months,
           "ticktext": [str(m) for m in months], "range": [0.5, 12.5]},
    legend={"title": {"text": "Legend"}, "traceorder": "grouped",
            "font": {"size": 9}},
    font={"size": 11},
    updatemenus=[dict(
        type="buttons", direction="left", x=0, y=1.15,
        xanchor="left", yanchor="top",
        buttons=[
            dict(label="Show Bars", method="update",
                 args=[{"visible": all_visible},
                       {"annotations": annotations}]),
            dict(label="Hide Bars", method="update",
                 args=[{"visible": bars_hidden}, {"annotations": []}]),
        ],
    )],
)
fig.show(config=PLOTLY_CONFIG)
def plot_tenor_bucket_analysis( df, label, metric_col, scale, ylabel, title, figsize=(14, 6), n_years=5, ): """Plot tenor-bucket heatmaps by year.""" years = sorted(df["YEAR"].dropna().unique(), reverse=True)[:n_years] buckets = sorted(df["TENOR_BUCKET"].dropna().astype(int).unique()) months = list(range(1, 13)) fig = make_subplots( rows=1, cols=len(years), subplot_titles=[str(y) for y in years], horizontal_spacing=0.02, ) matrices, all_values = {}, [] for year in years: year_data = df[df["YEAR"] == year] matrix = [[ year_data[ (year_data["MONTH"] == month) & (year_data["TENOR_BUCKET"] == bucket) ][metric_col].sum() / scale for bucket in buckets ] for month in months] matrices[year] = matrix all_values.extend(value for row in matrix for value in row) colorscale, reversescale, zmin, zmax = get_colorscale_params(all_values) for index, year in enumerate(years): fig.add_trace(go.Heatmap( z=matrices[year], x=[str(b) for b in buckets], y=[str(m) for m in months], colorscale=colorscale, reversescale=reversescale, zmin=zmin, zmax=zmax, showscale=(index == len(years) - 1), text=[[f"{v:.1f}" for v in row] for row in matrices[year]], texttemplate="%{text}", textfont={"size": 7}, hovertemplate=( f"{label} {year}
Month: %{{y}}
Tenor: %{{x}}
" f"{ylabel}: %{{z:.2f}}<extra></extra>" ), ), row=1, col=index + 1) width_px, height_px = apply_figsize(figsize, 600) fig.update_layout( title={"text": title, "x": 0.5, "xanchor": "center"}, template="plotly_white", height=height_px, width=width_px, font={"size": 9}, ) for col in range(1, len(years) + 1): fig.update_yaxes( autorange="reversed", title_text="Month" if col == 1 else None, row=1, col=col, ) fig.update_xaxes(title_text="Tenor Bucket", row=1, col=col) fig.show(config=PLOTLY_CONFIG)
def _parse_date(value, default): """Parse a date in any of the formats visible in the source.""" if value is None: return default for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"): try: return datetime.strptime(value, fmt).date() except ValueError: continue raise ValueError(f"Cannot parse date: {value}")
def plot_tenor_currency_heatmap( df, label, start_date=None, end_date=None, currencies=None, metric_col="AMT_ISSUED_USD", scale=1e9, ylabel="Notional (bln USD)", figsize=(14, 6), ): """Plot tenor bucket by currency and year for a chosen date range.""" today = date.today() start_dt = _parse_date(start_date, date(today.year - 1, 1, 1)) end_dt = _parse_date(end_date, today) pricing_dates = pd.to_datetime(df["SECURITY_PRICING_DATE"]).

jordan:
dt.date filtered = df[(pricing_dates >= start_dt) & (pricing_dates <= end_dt)].copy() if currencies is not None: filtered = filtered[filtered["CRNCY"].isin(currencies)] if filtered.empty: print(f"No data found between {start_dt} and {end_dt}") return

if currencies is not None:
    currency_list = [
        currency for currency in currencies
        if currency in filtered["CRNCY"].unique()
    ]
else:
    currency_list = sorted(filtered["CRNCY"].dropna().unique())
years = sorted(filtered["YEAR"].dropna().unique())
buckets = sorted(filtered["TENOR_BUCKET"].dropna().astype(int).unique())
fig = make_subplots(
    rows=1, cols=len(years), subplot_titles=[str(y) for y in years],
    horizontal_spacing=0.02,
)
matrices, all_values = {}, []
for year in years:
    year_data = filtered[filtered["YEAR"] == year]
    matrix = [[
        year_data[
            (year_data["TENOR_BUCKET"] == bucket)
            & (year_data["CRNCY"] == currency)
        ][metric_col].sum() / scale
        for currency in currency_list
    ] for bucket in buckets]
    matrices[year] = matrix
    all_values.extend(value for row in matrix for value in row)
colorscale, reversescale, zmin, zmax = get_colorscale_params(all_values)
for index, year in enumerate(years):
    fig.add_trace(go.Heatmap(
        z=matrices[year], x=currency_list,
        y=[f"{bucket}Y" for bucket in buckets],
        colorscale=colorscale, reversescale=reversescale,
        zmin=zmin, zmax=zmax, showscale=(index == len(years) - 1),
        text=[[f"{value:.1f}" for value in row]
              for row in matrices[year]],
        texttemplate="%{text}", textfont={"size": 8},
        hovertemplate=(
            f"{label} {year}<br>Tenor: %{{y}}<br>Currency: %{{x}}<br>"
            f"{ylabel}: %{{z:.2f}}<extra></extra>"
        ),
    ), row=1, col=index + 1)
width_px, height_px = apply_figsize(figsize, 600)
date_range = f"{start_dt:%d/%m/%Y} - {end_dt:%d/%m/%Y}"
fig.update_layout(
    title={
        "text": (f"{label} - {ylabel} by Tenor Bucket & Currency"
                 f"<br><sup>{date_range}</sup>"),
        "x": 0.5, "xanchor": "center",
    },
    template="plotly_white", height=height_px, width=width_px,
    font={"size": 10},
)
for col in range(1, len(years) + 1):
    fig.update_yaxes(
        autorange="reversed",
        title_text="Tenor Bucket" if col == 1 else None,
        row=1, col=col,
    )
    fig.update_xaxes(title_text="Currency", row=1, col=col)
fig.show(config=PLOTLY_CONFIG)

summary = filtered.pivot_table(
    values=metric_col, index="YEAR", columns="CRNCY", aggfunc="sum",
).fillna(0) / scale
print(f"\n{'=' * 60}\n{label} - SUMMARY BY CURRENCY ({date_range})")
print(summary.round(2).to_string())
print("\nTotal by Currency:")
print(summary.sum().sort_values(ascending=False).round(2).to_string())
def run_analysis(df, label, focus_month=1, group_col="ISSUER_SUBGROUP"): """Print seasonality statistics and graph a selected month by year.""" current_year = date.today().year complete_years = sorted( y for y in df["YEAR"].dropna().unique() if y < current_year ) history = df[df["YEAR"].isin(complete_years)] yearly_totals = history.groupby("YEAR")["AMT_ISSUED_USD"].sum() / 1e9 monthly_average_per_year = yearly_totals.mean() / 12 monthly_totals = history.groupby("MONTH")["AMT_ISSUED_USD"].sum() / 1e9 monthly_average = monthly_totals / max(len(complete_years), 1) seasonal_index = monthly_average / monthly_average_per_year print(f"\n{'=' * 70}\n{label} - SEASONALITY ANALYSIS OVER ALL PERIODS") print(f"{'Month':<10} {'Avg':>12} {'Diff':>12} {'Index':>8} {'Pattern':<15}") for month in range(1, 13): avg = monthly_average.get(month, 0) diff = avg - monthly_average_per_year index = seasonal_index.get(month, 1) if index >= 1.3: pattern = "*** High avg" elif index >= 1.1: pattern = "** Above avg" elif index <= 0.7: pattern = "*** Low avg" elif index <= 0.9: pattern = "** Below avg" else: pattern = "Normal" print(f"{MONTH_NAMES[month - 1]:<10} {avg:>12.2f} {diff:>+12.2f} " f"{index:>8.2f} {pattern:<15}")

monthly_pivot = history.pivot_table(

jordan:
values="AMT_ISSUED_USD", index="MONTH", columns="YEAR",
    aggfunc="sum",
).fillna(0) / 1e9
print(
    f"\n{'Month':<10} {'Above Avg':>12} {'Below Avg':>12} "
    f"{'Consistency':>12}"
)
print("-" * 50)
for month in range(1, 13):
    if month not in monthly_pivot.index:
        continue
    monthly_values = monthly_pivot.loc[month]
    yearly_averages = yearly_totals / 12
    above = sum(
        monthly_values.get(year, 0) > yearly_averages.get(year, 0)
        for year in complete_years
    )
    below = len(complete_years) - above
    consistency = max(above, below) / max(len(complete_years), 1) * 100
    print(
        f"{MONTH_NAMES[month - 1]:<10} {above:>12} {below:>12} "
        f"{consistency:>11.0f}%"
    )

focus = (
    df[df["MONTH"] == focus_month].groupby("YEAR")["AMT_ISSUED_USD"].sum()
    / 1e9
)
historical_average = focus[focus.index < current_year].mean()
print(
    f"\nFOCUS: {MONTH_NAMES[focus_month - 1].upper()} "
    "ISSUANCE BY YEARS"
)
print(f"{'Year':<6} {'Amount':>20} {'vs Avg':>20}")
print("-" * 55)
for year in focus.index:
    value = focus[year]
    difference = value - historical_average
    marker = " <--" if year == current_year else ""
    print(f"{year:<6} {value:>20.1f} {difference:>+20.1f}{marker}")
print("-" * 35)
print(f"{'Avg':<6} {historical_average:>20.1f}")
years = sorted(focus.index, reverse=True)
values = [focus[y] for y in years]
fig = go.Figure(go.Bar(
    x=[str(y) for y in years], y=values,
    marker_color=["#d62728" if y == current_year else "#1f77b4" for y in years],
    text=[f"{v:.1f}" for v in values], textposition="outside",
    hovertemplate="Year: %{x}<br>Amount: %{y:.1f} bln<extra></extra>",
))
fig.add_hline(
    y=historical_average, line_dash="dash", line_color="gray",
    annotation_text=f"Avg: {historical_average:.1f}",
    annotation_position="right",
)
fig.update_layout(
    title={
        "text": f"{label} - {MONTH_NAMES[focus_month - 1]} Issuance by Year (bln USD)",
        "x": 0.5, "xanchor": "center",
    },
    xaxis_title="Year", yaxis_title="Amount", template="plotly_white",
    height=500,
)
fig.show(config=PLOTLY_CONFIG)

totals = history.groupby(group_col)["AMT_ISSUED_USD"].sum()
group_label = group_col.replace("_", " ").title()
print(f"\n{group_label.upper()} SEASONALITY")
for industry in totals.sort_values(ascending=False).head(15).index:
    monthly = (
        history[history[group_col] == industry]
        .groupby("MONTH")["AMT_ISSUED_USD"].sum()
        / 1e9 / max(len(complete_years), 1)
    )
    peak, trough = monthly.idxmax(), monthly.idxmin()
    print(
        f"{industry:<35} Peak: {MONTH_NAMES[peak - 1]} ({monthly[peak]:.2f}), "
        f"Trough: {MONTH_NAMES[trough - 1]} ({monthly[trough]:.2f})"
    )
def get_offshore(df, analysis): """Return offshore issuance for the selected analysis market.""" non_financial = ~df["ISSUER_INDUSTRY"].isin(["BANK", "FINANCIAL"]) not_subordinated = df["IS_SUBORDINATED"].eq(False) masks = { "JP": ( (df["DOMICILE"].eq("JP") & df["CRNCY"].ne("JPY") & non_financial & ~df["INDUSTRY_GROUP"].isin(["Banks", "Investment Companies"]) & not_subordinated) | (df["TICKER"].eq("JBIC") & df["CRNCY"].ne("JPY")) ), "AU": (df["REGION"].eq("AUNZ") & df["CRNCY"].ne("AUD") & non_financial & ~df["INDUSTRY_GROUP"].isin(["Software"]) & not_subordinated), "NZ": (df["REGION"].eq("AUNZ") & df["CRNCY"].ne("NZD") & non_financial & ~df["INDUSTRY_GROUP"].isin(["Software"]) & not_subordinated), "GB": (df["DOMICILE"].eq("GB") & df["CRNCY"].ne("GBP") & df["CRNCY"].ne("USD") & non_financial & df["IS_COVERED"].eq(False) & ~df["INDUSTRY_GROUP"].isin(["Banks"]) & not_subordinated), "EU": (df["REGION"].eq("EU") & df["CRNCY"].ne("EUR") & df["CRNCY"].ne("USD") & non_financial & df["IS_COVERED"].eq(False) & ~df["INDUSTRY_GROUP"].isin(["Banks"]) & not_subordinated), "CA": (df["DOMICILE"].eq("CA") & df["CRNCY"].ne("CAD") & non_financial & df["IS_COVERED"].eq(False) & ~df["INDUSTRY_GROUP"].isin(["Banks"]) & not_subordinated), "SG": (df["DOMICILE"].eq("SG") & df["CRNCY"].ne("SGD") & non_financial & ~df["INDUSTRY_GROUP"].

jordan:
isin(["Banks", "Investment Companies"]) & not_subordinated), } if analysis not in masks: raise ValueError(f"Unsupported analysis: {analysis}") return df[masks[analysis]].copy()
def get_kanga(df, analysis): """Return foreign issuers borrowing in the selected domestic currency.""" non_financial = ~df["ISSUER_INDUSTRY"].isin(["BANK", "FINANCIAL"]) not_subordinated = df["IS_SUBORDINATED"].eq(False) masks = { "JP": (df["DOMICILE"].ne("JP") & df["CRNCY"].eq("JPY") & non_financial & not_subordinated & df["TICKER"].ne("JBIC")), "AU": (df["REGION"].ne("AUNZ") & df["CRNCY"].eq("AUD") & non_financial & not_subordinated), "NZ": (df["REGION"].ne("AUNZ") & df["CRNCY"].eq("NZD") & non_financial & not_subordinated), "GB": (df["DOMICILE"].ne("GB") & df["DOMICILE"].ne("US") & df["CRNCY"].eq("GBP") & non_financial & df["IS_COVERED"].eq(False) & not_subordinated), "EU": (df["REGION"].ne("EU") & df["DOMICILE"].ne("US") & df["CRNCY"].eq("EUR") & non_financial & df["IS_COVERED"].eq(False) & not_subordinated), "CA": (df["DOMICILE"].ne("CA") & df["CRNCY"].eq("CAD") & non_financial & df["IS_COVERED"].eq(False) & not_subordinated), "SG": (df["DOMICILE"].ne("SG") & df["CRNCY"].eq("SGD") & non_financial & not_subordinated), } if analysis not in masks: raise ValueError(f"Unsupported analysis: {analysis}") return df[masks[analysis]].copy()
def print_top_issuers(df, group_col, n_industries=5, n_issuers=5): """Print the leading issuers and their most common tenor buckets.""" recent_years = sorted(df["YEAR"].dropna().unique(), reverse=True)[:5] recent = df[df["YEAR"].isin(recent_years)] industries = ( recent.groupby(group_col)["AMT_ISSUED_USD"].sum() .sort_values(ascending=False).head(n_industries).index ) for industry in industries: industry_data = recent[recent[group_col] == industry] rows = [] for issuer, issuer_data in industry_data.groupby("ISSUER"): buckets = ", ".join( f"{int(bucket)}Y ({count})" for bucket, count in issuer_data["TENOR_BUCKET"] .value_counts().head(3).items() ) rows.append((issuer, issuer_data["AMT_ISSUED_USD"].sum(), buckets)) rows.sort(key=lambda row: row[1], reverse=True) print(f"\n{industry}") for i, (issuer, amount, buckets) in enumerate(rows[:n_issuers], 1): print(f" {i}. {issuer:<35} {amount / 1e9:>8.2f} bln USD [{buckets}]")
def main(): data = pd.read_parquet(PARQUET_PATH) offshore = get_offshore(data, ANALYSIS) kanga = get_kanga(data, ANALYSIS)

offshore_amt, offshore_dv01 = create_pivots(offshore)
plot_standard_charts(
    offshore_amt, offshore_dv01, f"OFFSHORE ({ANALYSIS})", FIGSIZE,
)
plot_industry_charts(
    offshore, f"OFFSHORE ({ANALYSIS})", FIGSIZE, N_GROUPS, GROUP_BY,
)
if RUN_ANALYSIS:
    run_analysis(offshore, "OFFSHORE", FOCUS_MONTH, GROUP_BY)
print_top_issuers(offshore, GROUP_BY, N_TOP_INDUSTRIES, N_TOP_ISSUERS)

kanga_amt, kanga_dv01 = create_pivots(kanga)
plot_standard_charts(kanga_amt, kanga_dv01, f"KANGA ({ANALYSIS})", FIGSIZE)
plot_industry_charts(
    kanga, f"KANGA ({ANALYSIS})", FIGSIZE, N_GROUPS, GROUP_BY,
)
if RUN_ANALYSIS:
    run_analysis(kanga, "KANGA", FOCUS_MONTH, GROUP_BY)
print_top_issuers(kanga, GROUP_BY, N_TOP_INDUSTRIES, N_TOP_ISSUERS)

all_years = sorted(set(offshore_amt.columns) | set(kanga_amt.columns))
offshore_amt = offshore_amt.reindex(columns=all_years, fill_value=0)
offshore_dv01 = offshore_dv01.reindex(columns=all_years, fill_value=0)
kanga_amt = kanga_amt.reindex(columns=all_years, fill_value=0)
kanga_dv01 = kanga_dv01.reindex(columns=all_years, fill_value=0)
net_amt = offshore_amt - kanga_amt
net_dv01 = offshore_dv01 - kanga_dv01

plot_heatmap(
    net_amt, f"NET ({ANALYSIS}) OFFSHORE - KANGA - Notional (bln USD)",
    x_label="Year", y_label="Month", figsize=FIGSIZE, reverse_yaxis=True,
)
plot_bar(
    net_amt, "NET - Notional by Month", "Notional (bln USD)",
    n_groups=N_GROUPS, x_label="Month", figsize=FIGSIZE,
)
plot_heatmap(
    net_dv01, f"NET ({ANALYSIS}) OFFSHORE - KANGA - DV01 (mm)",
    x_label="Year", y_label="Month", figsize=FIGSIZE, reverse_yaxis=True,
)
plot_bar(
    net_dv01, "NET - DV01 by Month", "DV01 (mm)",

jordan:
n_groups=N_GROUPS, x_label="Month", figsize=FIGSIZE,
)

group_label = GROUP_BY.replace("_", " ").title()
plot_net_by_industry(
    offshore, kanga, "AMT_ISSUED_USD", 1e9,
    "Notional (bln USD)",
    f"NET by {group_label} ({ANALYSIS}) - Notional (bln USD)",
    ANALYSIS, FIGSIZE_LARGER, N_GROUPS, GROUP_BY,
)
plot_net_by_industry(
    offshore, kanga, "DV01_MM", 1, "DV01 (mm)",
    f"NET by {group_label} ({ANALYSIS}) - DV01 (mm)",
    ANALYSIS, FIGSIZE_LARGER, N_GROUPS, GROUP_BY,
)

net_tenor_data = pd.concat([
    offshore.assign(
        NET_AMT=offshore["AMT_ISSUED_USD"],
        NET_DV01=offshore["DV01_MM"],
    ),
    kanga.assign(
        NET_AMT=-kanga["AMT_ISSUED_USD"],
        NET_DV01=-kanga["DV01_MM"],
    ),
], ignore_index=True)
tenor_configs = [
    ("OFFSHORE", offshore, "AMT_ISSUED_USD", 1e9, "Notional (bln USD)"),
    ("OFFSHORE", offshore, "DV01_MM", 1, "DV01 (mm)"),
    ("KANGA", kanga, "AMT_ISSUED_USD", 1e9, "Notional (bln USD)"),
    ("KANGA", kanga, "DV01_MM", 1, "DV01 (mm)"),
    ("NET", net_tenor_data, "NET_AMT", 1e9, "Notional (bln USD)"),
    ("NET", net_tenor_data, "NET_DV01", 1, "DV01 (mm)"),
]
for label, frame, metric_col, scale, ylabel in tenor_configs:
    title_label = (
        f"{label} ({ANALYSIS})" if label != "NET"
        else f"NET ({ANALYSIS}) OFFSHORE - KANGA"
    )
    plot_tenor_bucket_analysis(
        frame, title_label, metric_col, scale, ylabel,
        f"{title_label} - {ylabel} by Tenor Bucket",
        FIGSIZE_LARGER, N_GROUPS,
    )

# Export the source parquet through a notebook download link.
with open(PARQUET_PATH, "rb") as file:
    parquet_bytes = file.read()
encoded = base64.b64encode(parquet_bytes).decode("utf-8")
display(HTML(f"""
    <a id="dl_parquet"
       href="data:application/octet-stream;base64,{encoded}"
       download="{OUTPUT_FILENAME}"
       style="font-size:1.2em;font-weight:bold;padding:8px 16px;
              background:#007BFF;color:white;border-radius:5px;
              text-decoration:none;">
        Download {OUTPUT_FILENAME}
    </a>
    <script>document.getElementById('dl_parquet').click();</script>
"""))
if name == "main": main()
