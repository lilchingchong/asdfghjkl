import pandas as pd
import numpy as np
import bql
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def process_data(file_path="data/combined.parquet"):
    # data
    data = pd.read_parquet("combined.parquet")
    
    # 1. Bucket definition
    # Tenor bucket definition
    TENOR_BUCKETS = [
        0, .25, .5, .75,
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 17, 20, 22, 25, 30, 35, 40,
        50] + [k for k in range(60, 1200)]
        
    # Tenor in months - mapping
    TENOR_MAP = {i: bucket for bucket, tenors in TENOR_BUCKETS.items() for i in tenors}
    # Apply
    # data['YRS_TO_MTY'] = data.apply(lambda row: TENOR_MAP.get(row['YRS_TO_MTY'], row['YRS_TO_MTY']), axis=1)

    # MATURITY_DATE / CALL / PUT
    FIRST_CALL_DATE = pd.to_datetime(data.get('FIRST_CALL_DATE', pd.Series(dtype='datetime64[ns]')), errors='coerce')
    # FIRST_CALL_DATE = pd.to_datetime(data['FIRST_CALL_DATE'], errors='coerce')
    # If FIRST_CALL_DATE is not null and is > pricing date (and < maturity?), also use maturity
    has_call = FIRST_CALL_DATE.notnull()
    # MATURITY_DATE = np.where(has_call, FIRST_CALL_DATE, data['MATURITY'])
    # data['MATURITY_DATE'] = MATURITY_DATE
    data['TERM'] = np.where(has_call, data['YRS_TO_CALL'], data['YRS_TO_MTY'])
    data['TENOR_SOURCE'] = np.where(has_call & (data['TERM'] > 0), 'FIRST_CALL', 'MATURITY')
    # data['YRS_TO_MTY'] = np.where(has_call & (data['TERM'] > 0), data['YRS_TO_CALL'], data['YRS_TO_MTY'])
    # data['MATURITY_DATE'] = pd.to_datetime(data['MATURITY_DATE'], errors='coerce')
    data['YEAR'] = pd.to_datetime(data['SECURITY_PRICING_DATE']).dt.year
    data['MONTH'] = pd.to_datetime(data['SECURITY_PRICING_DATE']).dt.month
    data['MY'] = pd.to_datetime(data['SECURITY_PRICING_DATE']).dt.to_period('M')
    
    # # # GET FX SPOT DATA ON SECURITY PX DATE IF IS NON-USD
    def fetch_fx_rates(data):
        """Fetches FX rates for non-USD bonds. BQL query is: bq.data.px_last(dates=..., currency='USD')"""
        data = data.copy()
        data['FX_DATE'] = pd.to_datetime(data['SECURITY_PRICING_DATE'])
        currencies = data[data['CRNCY'] != 'USD']['CRNCY'].dropna().unique()
        if len(currencies) == 0:
            data['FX_SPOT'] = 1.0
            data['IS_USD'] = True
            return data
        data['FX_DATE'] = data['SECURITY_PRICING_DATE'].dt.date
        min_date = data['FX_DATE'].min()
        max_date = data['FX_DATE'].max()

        # Build queries
        direct_tickers = [f"{ccy} Curncy" for ccy in currencies]
        cross_tickers = [f"{ccy}USD Curncy" for ccy in currencies]
        req = bq.Request(direct_tickers + cross_tickers,
                         bq.data.px_last(dates=bq.func.range(min_date, max_date)))

        resp = bq.execute(req)
        # resp is a list of dataframes, one for each field. We only have one field.
        fx_raw = resp[0].df().reset_index().rename(columns={'ID': 'TICKER', 'DATE': 'DATE_FX'})
        fx_raw['DATE_FX'] = pd.to_datetime(fx_raw['DATE_FX'])

        # Parse direct quotes (USD Terms)
        direct = fx_raw[fx_raw['TICKER'].str.endswith('USD Curncy')]
        # direct['CRNCY'] = direct['TICKER'].str.replace('USD Curncy', '')
        direct = direct.assign(CRNCY=direct['TICKER'].str.replace('USD Curncy', ''))
        direct = direct.rename(columns={'PX_LAST': 'FX_DIRECT'})
        # Parse cross quotes (Non-USD Terms)
        cross = fx_raw[~fx_raw['TICKER'].str.endswith('USD Curncy')]
        # cross['CRNCY'] = cross['TICKER'].str.replace(' Curncy', '')
        cross = cross.assign(CRNCY=cross['TICKER'].str.replace(' Curncy', ''))
        cross = cross.rename(columns={'PX_LAST': 'FX_CROSS'})

        # Pivot and fill
        # direct_wide = direct.pivot(index='DATE_FX', columns='CRNCY', values='FX_DIRECT').ffill().bfill()
        # cross_wide = cross.pivot(index='DATE_FX', columns='CRNCY', values='FX_CROSS').ffill().bfill()
        # back to long
        # direct_long = direct_wide.stack().reset_index()
        # direct_long.columns = ['SECURITY_PRICING_DATE', 'CRNCY', 'FX_DIRECT']
        # cross_long = cross_wide.stack().reset_index()
        # cross_long.columns = ['SECURITY_PRICING_DATE', 'CRNCY', 'FX_CROSS']

        # Merge back to data
        data = data.merge(direct[['DATE_FX', 'CRNCY', 'FX_DIRECT']], left_on=['FX_DATE', 'CRNCY'], right_on=['DATE_FX', 'CRNCY'], how='left')
        data = data.merge(cross[['DATE_FX', 'CRNCY', 'FX_CROSS']], left_on=['FX_DATE', 'CRNCY'], right_on=['DATE_FX', 'CRNCY'], how='left')

        # USD gets 1.0
        data.loc[data['CRNCY'] == 'USD', 'FX_DIRECT'] = 1.0
        data.loc[data['CRNCY'] == 'USD', 'FX_CROSS'] = 1.0
        return data

    data = fetch_fx_rates(data)

    def resolve_fx_and_convert(data, tolerance=0.05):
        data = data.copy()
        # Treat 0 as missing to avoid division-by-zero
        direct_inv = np.where(data['FX_DIRECT'] != 0, 1 / data['FX_DIRECT'].replace(0, np.nan), np.nan)
        cross_matches_inv = np.isclose(data['FX_CROSS'], direct_inv, rtol=tolerance)

        # Find valid conversions
        both_ok = data['FX_DIRECT'].notna() & data['FX_CROSS'].notna()
        match = both_ok & cross_matches_inv

        # Create cross_spot as a pd.Series to maintain alignment and correct assignment
        cross_spot = pd.Series(index=data.index, dtype=float)

        # Scenario 1: Both exist and match -> direct is preferred
        cross_spot[match] = data.loc[match, 'FX_DIRECT']

        # Scenario 2: Only direct exists
        only_direct = data['FX_DIRECT'].notna() & data['FX_CROSS'].isna()
        cross_spot[only_direct] = data.loc[only_direct, 'FX_DIRECT']

        # Scenario 3: Only cross exists
        only_cross = data['FX_CROSS'].notna() & data['FX_DIRECT'].isna()
        cross_spot[only_cross] = 1 / data.loc[only_cross, 'FX_CROSS']

        # Scenario 4: Both exist but DO NOT match -> direct is preferred
        mismatch = both_ok & ~cross_matches_inv
        cross_spot[mismatch] = data.loc[mismatch, 'FX_DIRECT']

        # Assign back to data
        data['FX_SPOT'] = cross_spot

        data['IS_USD'] = data['CRNCY'] == 'USD'
        
        # Check for completely unresolvable fx
        data['IS_MISSING_FX'] = np.where(~data['IS_USD'] & data['FX_SPOT'].isna(), True, False)

        # Default missing FX to 1.0 to avoid breaking later steps, but flag them
        data['FX_SPOT'] = data['FX_SPOT'].fillna(1.0)

        return data

    data = resolve_fx_and_convert(data, tolerance=0.05)

    # data['AMT_OUT_ISSUED_USD'] = data['AMT_OUT_ISSUED'] * data['FX_SPOT']
    # data['ISSUE_PX_USD'] = data['ISSUE_PX'] * data['FX_SPOT']

    # USD special case
    data.loc[data['CRNCY'] == 'USD', 'FX_SPOT'] = 1.0
    data.loc[data['CRNCY'] == 'USD', 'IS_USD'] = True

    data['AMT_OUT_ISSUED_USD'] = data['AMT_OUT_ISSUED'] * data['FX_SPOT']

    # # ADDING REGION MAP & CONVERTING AMOUNT ISSUED TO USD
    # Region mapping
    REGION_MAP = {
        'NAMER': ['US', 'CA', 'BM', 'KY', 'VG', 'BS', 'PR'], # North America + Caribbean
        'EMEA': ['GB', 'FR', 'DE', 'NL', 'CH', 'IE', 'IT', 'ES', 'SE', 'BE', 'DK', 'FI', 'NO', 'AT', 'LU',
                 'AE', 'SA', 'IL', 'ZA', 'TR', 'PL', 'CZ', 'HU', 'RU', 'QA', 'KW', 'OM', 'BH'],
        'LATAM': ['MX', 'BR', 'CL', 'CO', 'PE', 'AR', 'PA', 'UY'],
        'APAC': ['JP', 'AU', 'CN', 'HK', 'SG', 'KR', 'IN', 'ID', 'MY', 'TH', 'TW', 'PH', 'NZ', 'MO']
    }

    # DO not roll up Eurozone (let BQL pull the actual country), but map to EM/DM
    # EM/DM mapping
    # EM_DM_MAP = {
    #     'DM': ['US', 'JP', 'GB', 'CA', 'FR', 'CH', 'DE', 'AU', 'NL', 'SE', 'HK', 'SG', 'IT', 'ES', 'BE', 'DK', 'FI', 'NO', 'AT', 'IE', 'NZ', 'IL', 'LU', 'PT'],
    #     'EM': ['CN', 'MX', 'BR', 'KR', 'ID', 'IN', 'SA', 'AE', 'TR', 'CL', 'CO', 'PE', 'ZA', 'PH', 'MY', 'TH', 'PL', 'RU', 'AR', 'QA', 'KW', 'OM', 'BH']
    # }
    # Actually use bql for this: bq.data.is_emerging_market() -> mapped to is_em
    
    # def map_region(country):
    #     for region, countries in REGION_MAP.items():
    #         if country in countries:
    #             return region
    #     return 'OTHER'
    # data['REGION'] = data['COUNTRY'].map(map_region)
    # data['IS_EM'] = data['COUNTRY'].isin(EM_DM_MAP['EM'])
    
    # # OVERRIDE BQL MATURITY TYP WITH MAPPED VALUES (CONVERTIBLE)
    
    # # OAD CALCULATION
    # # OAD = OAS / 10000 
    # # modified duration calc approx:
    def calculate_macaulay_duration(row, max_years=100):
        """Calculates Macaulay duration. Duration capped at max_dur_years."""
        # T = min(row['MATURITY'], max_years)
        # T = min(row['TERM'], max_years)
        T = row['YRS_TO_MTY']
        if pd.isna(T) or T <= 0:
            return 0.0

        cpn_type = row.get('CPN_TYP', '')
        cpn = row.get('CPN', 0.0)
        cpn_freq = row.get('CPN_FREQ', 0)

        # Treat 0 freq as 2 for semi-annual standard
        if pd.isna(cpn_freq) or cpn_freq == 0:
            freq = 2.0
        elif isinstance(cpn_freq, str):
            if cpn_freq.upper() in ['VARIABLE', 'DEFAULTED', 'EXCHANGED', 'FLAT TRADING', 'SPARE - N/A']:
                freq = 0
            else:
                try:
                    freq = float(cpn_freq)
                except (ValueError, TypeError):
                    freq = 2.0
        else:
            freq = float(cpn_freq)
            
        if cpn_type == 'VARIABLE' and cpn == 0:
            # Treat as zero-coupon, MacDur = maturity
            return min(T, max_years)
            
        # If yield is not available, we can't accurately calculate, return T
        y = row.get('YLD_YTM_MID', np.nan)
        if pd.isna(y) or y <= 0:
            return min(T, max_years)
            
        y = y / 100.0  # Decimal yield
        c = cpn / 100.0 # Decimal coupon
        
        if freq == 0:
            return min(T, max_years)
            
        m = freq
        n = T * m # Number of periods
        
        # If coupon is zero, MacDur = Maturity
        if c == 0:
            return min(T, max_years)
            
        # Calculate Macaulay duration in periods
        y_m = y / m
        c_m = c / m
        
        # Handle edge cases for numerical stability
        if y_m == 0:
            return min(T, max_years)
            
        term1 = (1 + y_m) / y_m
        
        num = 1 + y_m + n * (c_m - y_m)
        # Avoid overflow in power calculation for very long dates
        try:
            den_term = (1 + y_m)**n - 1
            den = c_m * den_term + y_m
            if den == 0:
                return min(T, max_years)
            mac_dur_periods = term1 - (num / den)
            mac_dur_years = mac_dur_periods / m
            return min(max(0, mac_dur_years), max_years)
        except OverflowError:
            # If n is too large, it approximates to a perpetual bond
            mac_dur_years = (1 + y_m) / (m * y_m)
            return min(mac_dur_years, max_years)


    def calculate_mod_duration(row, max_years=100):
        mac_dur = calculate_macaulay_duration(row, max_years)
        
        y = row.get('YLD_YTM_MID', np.nan)
        if pd.isna(y) or y <= 0:
            return mac_dur
        
        y = y / 100.0
        cpn_freq = row.get('CPN_FREQ', 2.0)
        try:
            m = float(cpn_freq) if float(cpn_freq) > 0 else 2.0
        except (ValueError, TypeError):
            m = 2.0
            
        mod_dur = mac_dur / (1 + y/m)
        return mod_dur
        
    def calculate_oad(row, max_years=100):
        mod_dur = calculate_mod_duration(row, max_years)
        return mod_dur

    def calculate_spread_dur_approx(row):
        return calculate_oad(row) # For bonds without embedded options, OAD ~ spread duration

    def calculate_duration_data(data):
        data['OAD_CALC'] = data.apply(calculate_oad, axis=1)
        data['SPREAD_DUR'] = data.apply(calculate_spread_dur_approx, axis=1)
        
        # OAS = Spread Duration * Spread (approx), to get OAS in bps:
        # But here we already have OAS, we just use it.
        # If we need DTS (Duration Times Spread)
        data['DTS'] = data['SPREAD_DUR'] * data['OAS_YAS_DEST']
        return data

    # data = calculate_duration_data(data)

    # data['OAD'] = data.apply(calculate_macaulay_duration, axis=1)
    # # DTS = Duration Times Spread
    # data['DTS'] = data['OAD'] * data['OAS_YAS_DEST']
    
    # # Convert OAS to decimal
    # data['OAS_DECIMAL'] = data['OAS_YAS_DEST'] / 10000

    # MARKET CAP CALCULATION
    # data['MARKET_CAP_USD'] = (data['PX_LAST'] / 100) * data['AMT_OUT_ISSUED_USD']

    data = calculate_duration_data(data)

    # # deduplicate duplicated ids. (BQL returns ID as issued on REGS and 144A and this can cause double count ...)
    data = data.sort_values(by=['ID', 'SECURITY_PRICING_DATE'])
    data = data.drop_duplicates(subset=['ID'])
    
    # deduplicate rows that have same AMT OUT ISSUED AND same OAS
    # # first isolate the ones which are exactly the same
    dups = data.duplicated(subset=['ISSUER', 'SECURITY_PRICING_DATE', 'AMT_OUT_ISSUED', 'OAS_YAS_DEST'], keep=False)
    
    # # Create mask to KEEP all rows EXCEPT ANY duplicate AMT_OUT_ISSUED AND OAS
    mask_to_keep = ~dups
    data = data[mask_to_keep]

    # data.to_parquet('processed.parquet')
    return data

if __name__ == '__main__':
    df = process_data("data/bql_data/usd_corp_combined.parquet")
    print(df.head())
