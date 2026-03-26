#!/usr/bin/env python3
"""
Generate synthetic orderbook data for transaction cost modeling.
Creates realistic spread and borrow cost estimates based on stock characteristics.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

def get_price_data():
    """Load price data to extract volume and volatility characteristics."""
    print("Loading price data to calibrate orderbook costs...")
    
    # Load all price files to calculate statistics
    price_files = sorted([f for f in os.listdir('real_data/prices') if f.endswith('.parquet')])
    
    if not price_files:
        raise FileNotFoundError("No price data found. Run download_prices_aligned.py first.")
    
    # Load first file to get symbols
    first_file = f'real_data/prices/{price_files[0]}'
    df_sample = pq.read_table(first_file).to_pandas()
    symbols = sorted(df_sample['symbol'].unique().tolist())
    
    print(f"Found {len(symbols)} symbols")
    
    # Calculate average volume and volatility per symbol
    all_data = []
    for file in price_files[:50]:  # Sample first 50 days for speed
        df = pq.read_table(f'real_data/prices/{file}').to_pandas()
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate statistics per symbol
    stats = combined.groupby('symbol').agg({
        'volume': 'mean',
        'close': ['mean', 'std']
    }).reset_index()
    
    stats.columns = ['symbol', 'avg_volume', 'avg_price', 'price_std']
    stats['volatility'] = stats['price_std'] / stats['avg_price']
    
    return symbols, stats

def generate_orderbook_data(symbols, stats):
    """
    Generate synthetic orderbook data with realistic cost estimates.
    
    Cost Model:
    - Spread: Inversely proportional to volume, proportional to volatility
    - Borrow cost: Based on volatility (higher vol = higher borrow cost)
    """
    os.makedirs('real_data/orderbook', exist_ok=True)
    
    # Generate for 2018-2025 business days
    dates = pd.bdate_range('2018-01-01', '2024-12-31')
    print(f"Generating orderbook data for {len(dates)} days...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Merge symbols with stats
    symbol_stats = stats.set_index('symbol')
    
    for day_idx, day in enumerate(dates):
        orderbook_data = []
        
        for symbol in symbols:
            # Get symbol statistics
            if symbol in symbol_stats.index:
                avg_vol = symbol_stats.loc[symbol, 'avg_volume']
                volatility = symbol_stats.loc[symbol, 'volatility']
            else:
                # Fallback for symbols without stats
                avg_vol = 1_000_000
                volatility = 0.02
            
            # SPREAD CALCULATION
            # Base spread: 0.01% for liquid stocks
            # Adjusted by volume (lower volume = wider spread)
            # Adjusted by volatility (higher vol = wider spread)
            base_spread_bps = 1.0  # 1 basis point = 0.01%
            
            # Volume factor: normalize to typical 1M volume
            volume_factor = max(0.5, min(3.0, 1_000_000 / max(avg_vol, 100_000)))
            
            # Volatility factor: 2% vol = 1x, scales linearly
            vol_factor = max(0.5, min(3.0, volatility / 0.02))
            
            # Final spread in decimal (e.g., 0.0001 = 0.01%)
            spread = (base_spread_bps / 10000) * volume_factor * vol_factor
            
            # Add small daily noise
            spread *= (1 + np.random.randn() * 0.1)
            spread = max(0.00005, min(0.001, spread))  # Clamp to 0.5-10 bps
            
            # BORROW COST CALCULATION
            # Base borrow rate: 0.5% annual
            # Adjusted by volatility (higher vol = higher borrow cost)
            # Hard-to-borrow stocks can have 5-20% annual rates
            base_borrow_annual = 0.005  # 0.5% annual
            
            # Volatility premium: high vol stocks are harder to borrow
            vol_premium = volatility * 10  # 2% vol -> 20% premium
            
            # Annual borrow rate
            borrow_annual = base_borrow_annual + vol_premium
            borrow_annual = max(0.001, min(0.20, borrow_annual))  # Clamp to 0.1-20%
            
            # Convert to daily rate
            borrow_daily = borrow_annual / 252
            
            # Add small daily noise
            borrow_daily *= (1 + np.random.randn() * 0.05)
            borrow_daily = max(0.00001, borrow_daily)
            
            # BID-ASK PRICES (for reference, not used by model)
            # Assume mid price from close, spread around it
            # We don't have actual close prices here, so we'll use placeholders
            mid_price = 100.0  # Placeholder
            bid = mid_price * (1 - spread / 2)
            ask = mid_price * (1 + spread / 2)
            
            orderbook_data.append({
                'symbol': symbol,
                'date': day.date(),
                'spread': spread,
                'borrow_cost': borrow_daily,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })
        
        # Save to parquet
        df = pd.DataFrame(orderbook_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'real_data/orderbook/date={day.date()}.parquet')
        
        if (day_idx + 1) % 50 == 0:
            print(f"  Progress: {day_idx + 1}/{len(dates)} files written")
    
    print(f"\nDone! Generated {len(dates)} orderbook files with {len(symbols)} symbols each")
    return len(dates)

def verify_orderbook_data():
    """Verify orderbook data was created correctly."""
    print("\n" + "="*60)
    print("VERIFICATION: Checking orderbook data")
    print("="*60)
    
    test_date = '2023-01-03'
    orderbook_file = f'real_data/orderbook/date={test_date}.parquet'
    
    if os.path.exists(orderbook_file):
        df = pq.read_table(orderbook_file).to_pandas()
        print(f"\nDate: {test_date}")
        print(f"Symbols: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Statistics
        print(f"\nSpread statistics (bps):")
        print(f"  Mean: {df['spread'].mean() * 10000:.2f} bps")
        print(f"  Median: {df['spread'].median() * 10000:.2f} bps")
        print(f"  Min: {df['spread'].min() * 10000:.2f} bps")
        print(f"  Max: {df['spread'].max() * 10000:.2f} bps")
        
        print(f"\nBorrow cost statistics (annual %):")
        print(f"  Mean: {df['borrow_cost'].mean() * 252 * 100:.2f}%")
        print(f"  Median: {df['borrow_cost'].median() * 252 * 100:.2f}%")
        print(f"  Min: {df['borrow_cost'].min() * 252 * 100:.2f}%")
        print(f"  Max: {df['borrow_cost'].max() * 252 * 100:.2f}%")
        
        # Check alignment with price data
        price_file = f'real_data/prices/date={test_date}.parquet'
        if os.path.exists(price_file):
            price_df = pq.read_table(price_file).to_pandas()
            price_symbols = set(price_df['symbol'].unique())
            orderbook_symbols = set(df['symbol'].unique())
            
            if price_symbols == orderbook_symbols:
                print(f"\n✅ PERFECT ALIGNMENT: Orderbook and price data have identical symbols")
            else:
                print(f"\n⚠️  Mismatch detected")
    else:
        print("Orderbook file not found")

def main():
    print("="*60)
    print("SYNTHETIC ORDERBOOK DATA GENERATION")
    print("="*60)
    print("\nGenerating realistic transaction costs based on:")
    print("- Volume (lower volume → wider spreads)")
    print("- Volatility (higher vol → wider spreads + higher borrow costs)")
    print()
    
    # Get symbols and statistics from price data
    symbols, stats = get_price_data()
    
    # Generate orderbook data
    file_count = generate_orderbook_data(symbols, stats)
    
    # Verify
    verify_orderbook_data()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Generated {file_count} orderbook files")
    print(f"All data sources ready: prices, funda, news, orderbook")
    print(f"\n✅ Ready for: python dry_fit.py --data real_data --out_dir ./validation")

if __name__ == "__main__":
    main()
