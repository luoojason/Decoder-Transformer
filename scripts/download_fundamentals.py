# download_funda_fixed.py
import simfin as sf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import datetime as dt

sf.set_data_dir(os.path.expanduser('~/simfin_cache'))
sf.set_api_key(os.environ.get('SIMFIN_API_KEY', ''))
os.makedirs('real_data/funda', exist_ok=True)

print('Loading SimFin data...')
inc = sf.load_income(variant='quarterly', market='us')
bal = sf.load_balance(variant='quarterly', market='us')
cf  = sf.load_cashflow(variant='quarterly', market='us')

# Reset index to make symbol and date regular columns
inc_reset = inc.reset_index()
bal_reset = bal.reset_index()
cf_reset = cf.reset_index()

# Merge on Ticker and Publish Date
fund = inc_reset[['Ticker', 'Publish Date', 'Net Income']].merge(
    bal_reset[['Ticker', 'Publish Date', 'Total Assets', 'Total Equity']], 
    on=['Ticker', 'Publish Date'], 
    how='outer'
).merge(
    cf_reset[['Ticker', 'Publish Date', 'Net Cash from Operating Activities']], 
    on=['Ticker', 'Publish Date'], 
    how='outer'
)

# Rename columns to lowercase
fund.columns = [c.lower().replace(' ', '_') for c in fund.columns]
fund = fund.rename(columns={'ticker': 'symbol'})

# Convert publish_date to datetime
fund['publish_date'] = pd.to_datetime(fund['publish_date'])

# Sort by symbol and publish_date
fund = fund.sort_values(['symbol', 'publish_date'])

print(f'Loaded {len(fund)} fundamental records for {fund["symbol"].nunique()} symbols')

# Generate daily files for 2018-2025 business days
cal = pd.bdate_range('2018-01-01', '2024-12-31')
print(f'Generating {len(cal)} daily parquet files...')

for i, day in enumerate(cal):
    # For each day, get the most recent fundamental data published on or before that day
    # Filter to data published on or before this day
    available = fund[fund['publish_date'] <= pd.Timestamp(day)]
    
    # Get the most recent record per symbol
    if len(available) > 0:
        latest = available.groupby('symbol').tail(1).reset_index(drop=True)
        latest['date'] = day.date()
        
        # Write to parquet
        table = pa.Table.from_pandas(latest)
        pq.write_table(table, f'real_data/funda/date={day.date()}.parquet')
    
    if (i + 1) % 50 == 0:
        print(f'  Progress: {i + 1}/{len(cal)} files written')

print(f'Done! Generated {len(cal)} files in real_data/funda/')