#!/usr/bin/env python3
"""
Download real news using NewsAPI and encode with FinBERT.
Uses the same symbols as price data for perfect alignment.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import numpy as np
from datetime import datetime, timedelta
import time

# NewsAPI key
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY', '')

def get_price_symbols():
    """Get list of symbols from price data."""
    price_file = 'real_data/prices/date=2023-01-03.parquet'
    if not os.path.exists(price_file):
        raise FileNotFoundError("Price data not found. Run download_prices_aligned.py first.")
    
    df = pq.read_table(price_file).to_pandas()
    symbols = sorted(df['symbol'].unique().tolist())
    print(f"Found {len(symbols)} symbols in price data")
    return symbols

def get_company_name(symbol):
    """Map ticker to company name for better news search."""
    # Common mappings
    mapping = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google Alphabet',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta Facebook',
        'NVDA': 'NVIDIA',
        'JPM': 'JPMorgan',
        'JNJ': 'Johnson Johnson',
        'V': 'Visa',
        'WMT': 'Walmart',
        'HD': 'Home Depot',
        'BAC': 'Bank of America',
        'XOM': 'Exxon Mobil',
        'CVX': 'Chevron',
        'ABBV': 'AbbVie',
        'PFE': 'Pfizer',
        'MRK': 'Merck',
        'KO': 'Coca Cola',
        'PEP': 'Pepsi',
        'COST': 'Costco',
        'ABT': 'Abbott',
        'TMO': 'Thermo Fisher',
        'DHR': 'Danaher',
        'UNH': 'UnitedHealth',
        'LLY': 'Eli Lilly',
        'NKE': 'Nike',
        'DIS': 'Disney',
        'CMCSA': 'Comcast',
        'NFLX': 'Netflix',
        'INTC': 'Intel',
        'CSCO': 'Cisco',
        'ORCL': 'Oracle',
        'IBM': 'IBM',
        'QCOM': 'Qualcomm',
        'TXN': 'Texas Instruments',
        'AMD': 'AMD',
        'CRM': 'Salesforce',
        'ADBE': 'Adobe',
        'ACN': 'Accenture',
        'BA': 'Boeing',
        'CAT': 'Caterpillar',
        'GE': 'General Electric',
        'HON': 'Honeywell',
        'UNP': 'Union Pacific',
        'DE': 'Deere',
        'MMM': '3M',
        'RTX': 'Raytheon',
        'LMT': 'Lockheed Martin',
    }
    return mapping.get(symbol, symbol)

def download_news_with_newsapi(symbols, start_date, end_date):
    """
    Download news using NewsAPI and encode with FinBERT.
    """
    try:
        from newsapi import NewsApiClient
    except ImportError:
        print("Installing newsapi-python...")
        os.system("pip install newsapi-python")
        from newsapi import NewsApiClient
    
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers")
        from transformers import AutoTokenizer, AutoModel
    
    # Initialize NewsAPI
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    
    # Load FinBERT
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert')
    model.eval()
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    os.makedirs('real_data/news', exist_ok=True)
    
    # Process each day
    dates = pd.bdate_range(start_date, end_date)
    print(f"\nProcessing {len(dates)} business days...")
    
    for day_idx, day in enumerate(dates):
        day_start = day.strftime('%Y-%m-%d')
        day_end = (day + timedelta(days=1)).strftime('%Y-%m-%d')
        
        embeddings = []
        
        for sym_idx, symbol in enumerate(symbols):
            try:
                # Search query: ticker OR company name
                query = f'{symbol} OR {get_company_name(symbol)}'
                
                # Fetch news for this symbol on this day
                articles = newsapi.get_everything(
                    q=query,
                    from_param=day_start,
                    to=day_end,
                    language='en',
                    sort_by='relevancy',
                    page_size=5  # Top 5 most relevant articles
                )
                
                # Combine headlines and descriptions
                texts = []
                if articles.get('articles'):
                    for article in articles['articles']:
                        title = article.get('title', '')
                        desc = article.get('description', '')
                        if title or desc:
                            texts.append(f"{title}. {desc}")
                
                # If no news, use neutral text
                text = ' '.join(texts) if texts else 'No significant news today.'
                
                # Encode with FinBERT
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling of last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                embeddings.append(embedding.tolist())
                
                # Rate limiting: NewsAPI free tier allows ~2 requests/second
                time.sleep(0.6)
                
            except Exception as e:
                print(f"  Warning: Failed to fetch news for {symbol} on {day_start}: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(768).tolist())
        
        # Save to parquet
        df = pd.DataFrame({
            'symbol': symbols,
            'news_vec_768': embeddings,
            'date': day.date()
        })
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'real_data/news/date={day.date()}.parquet')
        
        if (day_idx + 1) % 10 == 0:
            print(f"  Progress: {day_idx + 1}/{len(dates)} days processed")
    
    print(f"\nDone! Generated {len(dates)} news files with real FinBERT embeddings")

def verify_news_data():
    """Verify news data was created correctly."""
    print("\n" + "="*60)
    print("VERIFICATION: Checking news data")
    print("="*60)
    
    test_date = '2023-01-03'
    news_file = f'real_data/news/date={test_date}.parquet'
    
    if os.path.exists(news_file):
        df = pq.read_table(news_file).to_pandas()
        print(f"\nDate: {test_date}")
        print(f"Symbols: {len(df)}")
        print(f"Embedding dimension: {len(df['news_vec_768'].iloc[0])}")
        print(f"Sample symbols: {df['symbol'].head(10).tolist()}")
        
        # Check embedding statistics
        sample_vec = np.array(df['news_vec_768'].iloc[0])
        print(f"Sample embedding mean: {sample_vec.mean():.4f}")
        print(f"Sample embedding std: {sample_vec.std():.4f}")
        
        # Check alignment with price data
        price_file = f'real_data/prices/date={test_date}.parquet'
        if os.path.exists(price_file):
            price_df = pq.read_table(price_file).to_pandas()
            price_symbols = set(price_df['symbol'].unique())
            news_symbols = set(df['symbol'].unique())
            
            if price_symbols == news_symbols:
                print(f"\n✅ PERFECT ALIGNMENT: News and price data have identical symbols")
            else:
                print(f"\n⚠️  Mismatch detected")
    else:
        print("News file not found")

def main():
    print("="*60)
    print("REAL NEWS DOWNLOAD WITH NEWSAPI + FinBERT")
    print("="*60)
    
    # Get symbols from price data
    symbols = get_price_symbols()
    print(f"Sample symbols: {symbols[:10]}")
    
    # Download news for 2023
    # Note: NewsAPI free tier has historical limit of ~1 month
    # For full 2023 data, you'd need a paid plan or process in batches
    print("\nNote: NewsAPI free tier limits historical data to ~1 month")
    print("Processing most recent available period...")
    
    # Process last 30 days of 2023 as example
    # For full year, you'd need to run this incrementally or upgrade to paid tier
    start_date = '2023-12-01'
    end_date = '2023-12-29'
    
    print(f"\nDate range: {start_date} to {end_date}")
    
    download_news_with_newsapi(symbols, start_date, end_date)
    
    # Verify
    verify_news_data()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print("Real news embeddings generated with FinBERT")
    print("\nTo process full 2023:")
    print("1. Upgrade to NewsAPI paid plan for historical access")
    print("2. OR run this script monthly throughout the year")
    print("3. OR use synthetic embeddings for older dates")

if __name__ == "__main__":
    main()
