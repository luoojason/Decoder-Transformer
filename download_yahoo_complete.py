# download_yahoo_complete.py

import yfinance as yf, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, os, datetime as dt

os.makedirs('real_data/prices', exist_ok=True)
tickers = ['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','JPM','JNJ','V']
print('Downloading 2023 daily bars from Yahoo Finance …')

data = yf.download(tickers, start='2023-01-01', end='2024-01-01', interval='1d', group_by='ticker')

for t in tickers:
    df = data[t].dropna().reset_index()
    df['symbol'] = t
    df['date']   = df['Date'].dt.date
    df = df[['symbol','Open','High','Low','Close','Volume','date']].rename(columns=str.lower)
    for day, sub in df.groupby('date'):
        table = pa.Table.from_pandas(sub)
        pq.write_table(table, f'real_data/prices/date={day}.parquet')

print('Done – files written to real_data/prices/')