#!/usr/bin/env python3
"""
Download aligned price data from Yahoo Finance for symbols that have SimFin fundamental data.
Uses a curated list of liquid US stocks to avoid rate limiting.
"""

import yfinance as yf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time

# Curated list of 100 liquid US stocks that likely have fundamental data
LIQUID_SYMBOLS = [
    # Mega cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    # Large cap tech
    'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'COST',
    # Industrial
    'CAT', 'GE', 'HON', 'UNP', 'BA', 'RTX', 'LMT', 'DE', 'MMM', 'UPS',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    # Telecom/Media
    'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'TMUS', 'CHTR', 'PARA', 'WBD', 'FOXA',
    # Retail/E-commerce
    'AMZN', 'BABA', 'JD', 'PDD', 'MELI', 'EBAY', 'ETSY', 'W', 'SHOP', 'SE',
    # Other sectors
    'BRK.B', 'V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI', 'AFRM', 'UPST', 'LC'
]

START_DATE = '2018-01-01'
END_DATE = '2025-01-01'
OUTPUT_DIR = 'real_data/prices'

def download_and_save_prices(symbols):
    """Download price data and save as daily parquet files."""
    print(f"Downloading price data for {len(symbols)} symbols from {START_DATE} to {END_DATE}...")
    
    # Clear existing price files
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download in batches to avoid rate limiting
    batch_size = 20
    all_data = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {len(batch)} symbols")
        
        try:
            data = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                interval='1d',
                group_by='ticker',
                progress=False,
                auto_adjust=False
            )
            
            # Process each symbol in the batch
            for symbol in batch:
                try:
                    if len(batch) == 1:
                        sym_data = data
                    else:
                        sym_data = data[symbol]
                    
                    if sym_data.empty:
                        print(f"    Warning: No data for {symbol}")
                        continue
                    
                    df = sym_data.reset_index()
                    df['symbol'] = symbol
                    df['date'] = pd.to_datetime(df['Date']).dt.date
                    
                    # Rename columns
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume',
                        'Adj Close': 'adj_close'
                    })
                    
                    # Select columns
                    df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"    Warning: Failed to process {symbol}: {e}")
                    continue
            
            # Rate limiting delay
            if i + batch_size < len(symbols):
                time.sleep(2)
                
        except Exception as e:
            print(f"    Error downloading batch: {e}")
            time.sleep(5)
            continue
    
    if not all_data:
        print("ERROR: No data downloaded!")
        return 0, 0
    
    # Combine all data
    print("\nCombining and saving daily files...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Group by date and save
    file_count = 0
    for date, group in combined.groupby('date'):
        table = pa.Table.from_pandas(group)
        filename = f'{OUTPUT_DIR}/date={date}.parquet'
        pq.write_table(table, filename)
        file_count += 1
        
        if file_count % 50 == 0:
            print(f"  Progress: {file_count} files written")
    
    unique_symbols = combined['symbol'].nunique()
    print(f"\nDone! Generated {file_count} daily price files")
    print(f"Symbols: {unique_symbols}")
    return file_count, unique_symbols

def verify_alignment():
    """Verify that price and fundamental data are aligned."""
    print("\n" + "="*60)
    print("VERIFICATION: Checking price/fundamental alignment")
    print("="*60)
    
    test_date = '2023-01-03'
    price_file = f'real_data/prices/date={test_date}.parquet'
    funda_file = f'real_data/funda/date={test_date}.parquet'
    
    if os.path.exists(price_file) and os.path.exists(funda_file):
        price_df = pq.read_table(price_file).to_pandas()
        funda_df = pq.read_table(funda_file).to_pandas()
        
        price_symbols = set(price_df['symbol'].unique())
        funda_symbols = set(funda_df['symbol'].unique())
        
        overlap = price_symbols.intersection(funda_symbols)
        
        print(f"\nDate: {test_date}")
        print(f"Price symbols: {len(price_symbols)}")
        print(f"Funda symbols: {len(funda_symbols)}")
        print(f"Overlap: {len(overlap)} symbols")
        print(f"Coverage: {len(overlap)/len(price_symbols)*100:.1f}%")
        
        if len(overlap) >= len(price_symbols) * 0.9:
            print(f"\n✅ GOOD ALIGNMENT: {len(overlap)}/{len(price_symbols)} price symbols have fundamental data")
            print(f"Sample overlapping symbols: {sorted(list(overlap))[:10]}")
        else:
            missing = price_symbols - funda_symbols
            print(f"\n⚠️  {len(missing)} symbols in prices but not in funda: {sorted(list(missing))}")
    else:
        print(f"Could not verify - files not found")

def main():
    print("="*60)
    print("ALIGNED PRICE DATA DOWNLOAD (CURATED UNIVERSE)")
    print("="*60)
    
    # Use curated list
    print(f"\nUsing curated universe of {len(LIQUID_SYMBOLS)} liquid US stocks")
    print(f"Sample: {LIQUID_SYMBOLS[:10]}")
    
    # Download and save
    file_count, symbol_count = download_and_save_prices(LIQUID_SYMBOLS)
    
    # Verify alignment
    if file_count > 0:
        verify_alignment()
        
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        print(f"Generated {file_count} daily files with {symbol_count} symbols")
        print(f"Ready for: python dry_fit.py --data real_data --out_dir ./validation")
    else:
        print("\nFAILED: No data downloaded")

if __name__ == "__main__":
    main()
