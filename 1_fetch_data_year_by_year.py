import os
import pandas as pd
from datetime import date, timedelta
import bql
bq = bql.Service()

def fetch_usd_corp_data_yr(start_date, end_date):
    """Fetches BQL data needed to calculate IG/HY(or start_date,end_date]. SECURITY_PRICING_DATE in (or start_date,end_date]. Returns DataFrame."""
    base_filter = (
        bq.univ.bond()
        .and_(bq.univ.asset_class() == "Corporates")
        # yrs_to_mty_issue is the years to maturity AT THE TIME OF ISSUE, so we're selecting currently
        # .and_(bq.data.yrs_to_mty_issue() > 1.5)
        
        .and_(bq.data.security_pricing_date() >= start_date)
        .and_(bq.data.security_pricing_date() <= end_date)
    )
    
    universe = bq.univ.bond(base_filter, consolidateduplicates=bq.filter(base_filter))
    data_items = {
        "ID": bq.data.id(),
        "TICKER": bq.data.ticker(),
        "CRNCY": bq.data.crncy(),
        "COUNTRY": bq.data.country(),
        "EXCEED15": bq.data.BVAL_MARKET_ENTRY_IMPBC(15),
        "BICS_LEVEL_2": bq.data.bics_level_2(),
        "BCLASS3": bq.data.bclass3(),
        "MATURITY": bq.data.maturity(),
        "PRICING_SOURCE": bq.data.pricing_source(),
        "SECURITY_PRICING_DATE": bq.data.security_pricing_date(),
        "OAS_YAS_DEST": bq.data.oas_yas_dest(),
        "CPN_FREQ": bq.data.cpn_freq(),
        "MATURITY_TYP": bq.data.maturity_typ(),
        "IS_SUBORDINATED": bq.data.is_subordinated(),
        "IS_SECURED": bq.data.is_secured(),
        "IS_CONVERTIBLE": bq.data.is_convertible(),
        "MANAGEMENT_EXPECTS": bq.data.entry_of_default(),
        "IS_PERPETUAL": bq.data.is_perpetual(),
        "YRS_TO_MTY_ISSUE": bq.data.yrs_to_mty_issue(),
        "IS_DEFAULTED": bq.data.is_defaulted(),
        "IS_DISTRESSED": bq.data.is_distressed(),
        "IS_COCO": bq.data.is_coco(),
        "IS_EMERGING_MARKET": bq.data.is_emerging_market(),
        "AMT_OUT_ISSUED": bq.data.amt_out_issued(),
        "YRS_TO_MTY": bq.data.yrs_to_mty(),
    }
    request = bql.Request(universe, data_items)
    response = bq.execute(request)
    # response is multiple DataFrames horizontally, then we have duplicate columns
    result = response[0].df()
    for idx in range(1, len(response)):
        result = pd.concat([result, response[idx].df()], axis=1)
    result = result.reset_index()
    result = result.loc[:, ~result.columns.duplicated()] # dedups columns
    # Bloomberg tends to set string na values to '<NA>'
    for col in ["SECURITY_PRICING_DATE", "MATURITY"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors='coerce')
    return result

def fetch_usd_provincial_bond_data(start_date, end_date):
    base_filter = (
        bq.univ.bond()
        .and_(
            (bq.data.univ_asset_class() == "Governments") | 
            (bq.data.univ_asset_class() == "Municipals")
        )
        .and_(bq.data.security_pricing_date() >= start_date)
        .and_(bq.data.security_pricing_date() <= end_date)
        .and_(bq.data.crncy() == "USD")
    )
    universe = bq.univ.bond(base_filter, consolidateduplicates=bq.filter(base_filter))
    data_items = {
        "ID": bq.data.id(),
        "CRNCY": bq.data.crncy(),
        "TICKER": bq.data.ticker(),
        "COUNTRY": bq.data.country(),
        "EXCEED15": bq.data.BVAL_MARKET_ENTRY_IMPBC(15),
        "BICS_LEVEL_2": bq.data.bics_level_2(),
        "BCLASS3": bq.data.bclass3(),
        "MATURITY": bq.data.maturity(),
        "PRICING_SOURCE": bq.data.pricing_source(),
        "SECURITY_PRICING_DATE": bq.data.security_pricing_date(),
        "OAS_YAS_DEST": bq.data.oas_yas_dest(),
        "CPN_FREQ": bq.data.cpn_freq(),
        "MATURITY_TYP": bq.data.maturity_typ(),
        "IS_SUBORDINATED": bq.data.is_subordinated(),
        "IS_SECURED": bq.data.is_secured(),
        "IS_CONVERTIBLE": bq.data.is_convertible(),
        "IS_PERPETUAL": bq.data.is_perpetual(),
        "YRS_TO_MTY_ISSUE": bq.data.yrs_to_mty_issue(),
        "IS_DEFAULTED": bq.data.is_defaulted(),
        "IS_DISTRESSED": bq.data.is_distressed(),
        "IS_COCO": bq.data.is_coco(),
        "IS_EMERGING_MARKET": bq.data.is_emerging_market(),
        "AMT_OUT_ISSUED": bq.data.amt_out_issued(),
        "YRS_TO_MTY": bq.data.yrs_to_mty(),
    }
    request = bql.Request(universe, data_items)
    response = bq.execute(request)
    result = response[0].df()
    for idx in range(1, len(response)):
        result = pd.concat([result, response[idx].df()], axis=1)
    result = result.reset_index()
    result = result.loc[:, ~result.columns.duplicated()]
    for col in ["SECURITY_PRICING_DATE", "MATURITY"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors='coerce')
    return result

def process_year(year):
    """Process data for a specific year, load from disk if file exists and year < current year."""
    # 1/1 to 12/31
    pq_path = f"data/bql_data/usd_corp_data_{year}.parquet"
    combined_pq_path = f"data/bql_data/usd_corp_combined_{year}.parquet"

    # If file exists and it's a completed year, just load it
    today = date.today()
    current_year = today.year
    if os.path.exists(combined_pq_path) and year < current_year:
        print(f"Year {year}: pq data returned.")
        return pd.read_parquet(combined_pq_path)

    # Fetch from bql
    dates = [date(year, m, 1) for m in range(1, 13)] + [date(year + 1, 1, 1)] if year < today.year else [date(year, m, 1) for m in range(1, 13) if date(year, m, 1) <= today] + [today + timedelta(days=1)]
    res = []
    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1] - timedelta(days=1)
        print(f"Fetching data from {start_date} to {end_date}...")
        corporates = fetch_usd_corp_data_yr(start_date, end_date)
        provincials = fetch_usd_provincial_bond_data(start_date, end_date)
        
        res.append(pd.concat([corporates, provincials]))
        
    print(f"Writing {sum([len(r) for r in res])} rows to {combined_pq_path}...")
    combined = pd.concat(res, ignore_index=True)
    combined.to_parquet(combined_pq_path)
    
    return combined

def process_years(start_year=2021, end_year=date.today().year):
    # Combine and save
    COMBINED_PATH = "data/bql_data/usd_corp_combined.parquet"
    old_combined = pd.read_parquet(COMBINED_PATH) if os.path.exists(COMBINED_PATH) else pd.DataFrame()
    if not old_combined.empty:
        print(f"Loaded {len(old_combined)} rows, saved to {COMBINED_PATH}")

    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        combined = process_year(year)

        # Find new rows
        if not old_combined.empty:
            new_rows = combined[~combined.set_index(['ID', 'SECURITY_PRICING_DATE']).index.isin(old_combined.set_index(['ID', 'SECURITY_PRICING_DATE']).index)]
            if len(new_rows) > 0:
                print(f"Found {len(new_rows)} new rows.")
                old_combined = pd.concat([old_combined, new_rows], ignore_index=True)
        else:
            old_combined = combined

    print(f"Writing {len(old_combined)} rows to {COMBINED_PATH}...")
    old_combined.to_parquet(COMBINED_PATH)

    # Verify correct
    print(f"Verification: {len(old_combined[old_combined['SECURITY_PRICING_DATE'].dt.date == pd.to_datetime('today').date()])} rows today.")

    # print count summary for each year
    today_year = date.today().year
    for year in range(start_year, today_year + 1):
        print(f"{year}: {len(old_combined[old_combined['SECURITY_PRICING_DATE'].dt.year == year])} rows")

    print(old_combined.head())
    print(old_combined[['TICKER', 'ISSUER', 'CRNCY', 'SECURITY_PRICING_DATE']].head(10).to_string())

    print(f"Sorted df: {old_combined.sort_values(by=['SECURITY_PRICING_DATE', 'TICKER'], ascending=[False, True]).head(10).to_string()}")

    # PULL ADDITIONAL FIELDS FROM BQL USING CACHE (ISIN - FORCE ALL -> True = legacy field)
    return old_combined

def fetch_additional_fields(
    combined_path="data/bql_data/usd_corp_combined.parquet",
    cache_path="processed.parquet",
    fields_map={
        "FIRST_CALL_DT": bq.data.FIRST_CALL_DT_ISSUANCE(),
        "INDUSTRY_GROUP": bq.data.INDUSTRY_GROUP(),
        "INDUSTRY_SUBGROUP": bq.data.INDUSTRY_SUBGROUP(),
        "PRIV_PLACE": bq.data.PRIV_PLACE()
    },
    batch_size=10000,
    force_all=False
):
    try:
        combined = pd.read_parquet(combined_path)
        original_count = len(combined)
        print(f"Loaded combined: {original_count} rows, {combined['ID'].nunique()} unique IDs")
    except Exception as e:
        print(f"Could not load {combined_path}: {e}")
        return

    try:
        cache = pd.read_parquet(cache_path)
        has_cache = True
        print(f"Loaded cache: {len(cache)} rows, cache IDs: {cache['ID'].nunique()} unique IDs")
    except Exception as e:
        has_cache = False
        cache = pd.DataFrame(columns=['ID'] + list(fields_map.keys()))
        print(f"No cache found at {cache_path}, creating new")

    missing_fields_ids = {} # field_name: [id1, id2...]
    # DETERMINE WHAT TO FETCH
    for field in fields_map.keys():
        if force_all or field not in combined.columns:
            if has_cache and field in cache.columns:
                # IDs in combined but not in cache for this field
                # We need to fetch if it is missing in cache (is null in cache)
                # or if it's not in cache at all
                ids_in_cache = cache[cache[field].notna()]['ID'].unique()
                to_fetch = set(combined['ID']) - set(ids_in_cache)
                missing_fields_ids[field] = list(to_fetch)
                print(f"Field {field}: in cache, missing for {len(to_fetch)} IDs (e.g. {list(to_fetch)[:3] if to_fetch else 'None'})")
            else:
                # Field completely missing from cache, fetch for all unique IDs
                missing_fields_ids[field] = list(combined['ID'].unique())
                print(f"Field {field}: not in cache, fetching all {len(missing_fields_ids[field])}")

    # FETCH IN BATCHES
    for field, ids in missing_fields_ids.items():
        if len(ids) == 0:
            continue
        print(f"\nFetching {field} for {len(ids)} missing IDs, batching by {batch_size}...")
        fetched = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            universe = bq.univ.list(batch_ids)
            req = bq.Request(universe, {field: fields_map[field]})
            print(f"  Batch {i//batch_size + 1}/{len(ids)//batch_size + 1} ({len(batch_ids)} items)")
            try:
                res = bq.execute(req)
                batch_result = res[0].df()
                
                # Account for potential multi-DataFrame responses in additional fields
                for idx in range(1, len(res)):
                    batch_result = pd.concat([batch_result, res[idx].df()], axis=1)
                
                batch_result = batch_result.reset_index()
                batch_result = batch_result.loc[:, ~batch_result.columns.duplicated()]
                fetched.append(batch_result)
            except Exception as e:
                print(f"  Error fetching batch: {e}")

        if fetched:
            fetched_df = pd.concat(fetched, ignore_index=True)
            # drop duplicates that might come from Bloomberg
            fetched_df = fetched_df.drop_duplicates(subset=['ID'])

            # -> UPDATE CACHE <-
            if field not in cache.columns:
                cache[field] = pd.NA

            if 'ID' in cache.columns:
                # Set index to update existing records without wiping out other columns
                cache = cache.set_index('ID')
                fetched_idx = fetched_df.set_index('ID')
                
                # Update existing IDs in the cache for this specific field
                if field in fetched_idx.columns:
                    cache[field].update(fetched_idx[field])
                
                # Append completely new IDs to the cache
                new_records = fetched_idx[~fetched_idx.index.isin(cache.index)]
                if not new_records.empty:
                    cache = pd.concat([cache, new_records])
                
                cache = cache.reset_index()
                
            print(f"  Updated cache with {len(fetched_df)} rows for {field}")
        else:
            print(f"No new data to fetch - using cache only")

    print(f"Saving cache to {cache_path}...")
    cache.to_parquet(cache_path, index=False)

    # -> APPLY CACHE TO COMBINED <-
    for field in fields_map.keys():
        if field in cache.columns:
            # Drop existing field columns from combined to avoid conflicts
            if field in combined.columns:
                combined = combined.drop(columns=[field])
            # drop any duplicates in cache before merge if there are any (there shouldn't be)
            cache_to_merge = cache.drop_duplicates(subset=['ID'])
            # Left join: preserve all rows in combined, matching duplicate IDs
            combined = combined.merge(cache_to_merge[['ID', field]], on='ID', how='left')
        else:
            print(f"Warning: Cache is empty, setting null columns for requested fields")
            combined[field] = pd.NA

    result = combined

    # Verify row count preserved
    if len(result) != original_count:
        print(f"WARNING: Row count changed from {original_count} to {len(result)}!")
        raise AssertionError(f"Row count mismatch (original: {original_count} != new: {len(result)})")

    # Save back to combined
    result.to_parquet(combined_path, index=False)
    print(f"Saved combined with additional fields: {list(fields_map.keys())} rows to {combined_path}")
    return result
 
# Run it
combined = fetch_additional_fields(
    combined_path="data/bql_data/usd_corp_combined.parquet",
    cache_path="processed.parquet", 
    fields_map={
        "FIRST_CALL_DT": bq.data.FIRST_CALL_DT_ISSUANCE(),
        "INDUSTRY_GROUP": bq.data.INDUSTRY_GROUP(),
        "INDUSTRY_SUBGROUP": bq.data.INDUSTRY_SUBGROUP(),
        "PRIV_PLACE": bq.data.PRIV_PLACE()
    },
    batch_size=10000,
    force_all=False
)
