#!/usr/bin/env python3
"""
REAL NEWS + FinBERT ENCODING IMPLEMENTATION GUIDE

This script demonstrates how to fetch real news and encode with FinBERT.
Requires: pip install transformers torch newsapi-python yfinance

OPTION 1: NewsAPI (Recommended for production)
- Sign up at https://newsapi.org (free tier: 100 requests/day)
- Get API key
- Access 80,000+ sources, historical data up to 1 month

OPTION 2: Yahoo Finance News (Free, limited)
- No API key needed
- Limited to recent news only
- Rate limited

OPTION 3: Alpha Vantage News Sentiment (Free tier available)
- Sign up at https://www.alphavantage.co
- News + sentiment scores
- 25 requests/day free tier
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel

# ============================================================================
# OPTION 1: NewsAPI Implementation (RECOMMENDED)
# ============================================================================

def download_news_newsapi(symbols, start_date, end_date):
    """
    Download news using NewsAPI.
    
    Setup:
        1. Sign up at https://newsapi.org
        2. Get your API key
        3. pip install newsapi-python
        4. Set environment variable: export NEWSAPI_KEY='your_key_here'
    """
    try:
        from newsapi import NewsApiClient
    except ImportError:
        print("Install newsapi: pip install newsapi-python")
        return None
    
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        print("ERROR: Set NEWSAPI_KEY environment variable")
        print("Get key from: https://newsapi.org")
        return None
    
    newsapi = NewsApiClient(api_key=api_key)
    
    # Load FinBERT
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert')
    model.eval()
    
    os.makedirs('real_data/news', exist_ok=True)
    
    # Process each day
    dates = pd.bdate_range(start_date, end_date)
    
    for day in dates:
        day_start = day.strftime('%Y-%m-%d')
        day_end = (day + timedelta(days=1)).strftime('%Y-%m-%d')
        
        embeddings = []
        
        for symbol in symbols:
            # Fetch news for this symbol on this day
            try:
                articles = newsapi.get_everything(
                    q=f'{symbol} OR {get_company_name(symbol)}',
                    from_param=day_start,
                    to=day_end,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )
                
                # Combine headlines and descriptions
                texts = []
                if articles['articles']:
                    for article in articles['articles']:
                        title = article.get('title', '')
                        desc = article.get('description', '')
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
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling of last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                embeddings.append(embedding.tolist())
                
            except Exception as e:
                print(f"Warning: Failed to fetch news for {symbol} on {day_start}: {e}")
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
        
        print(f"Processed {day_start}")
    
    print(f"Done! Generated {len(dates)} news files")

# ============================================================================
# OPTION 2: Yahoo Finance News (FREE, LIMITED)
# ============================================================================

def download_news_yfinance(symbols, start_date, end_date):
    """
    Download news using yfinance.
    
    Limitations:
    - Only recent news (last few weeks)
    - No historical date filtering
    - Rate limited
    
    Setup:
        pip install yfinance transformers torch
    """
    import yfinance as yf
    
    # Load FinBERT
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert')
    model.eval()
    
    os.makedirs('real_data/news', exist_ok=True)
    
    dates = pd.bdate_range(start_date, end_date)
    
    for day in dates:
        embeddings = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                # Filter news by date (approximate)
                day_timestamp = day.timestamp()
                relevant_news = [
                    n for n in news
                    if abs(n.get('providerPublishTime', 0) - day_timestamp) < 86400
                ]
                
                # Combine headlines
                headlines = [n.get('title', '') for n in relevant_news]
                text = ' '.join(headlines) if headlines else 'No news available.'
                
                # Encode with FinBERT
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                embeddings.append(embedding.tolist())
                
            except Exception as e:
                print(f"Warning: {symbol} - {e}")
                embeddings.append(np.zeros(768).tolist())
        
        # Save
        df = pd.DataFrame({
            'symbol': symbols,
            'news_vec_768': embeddings,
            'date': day.date()
        })
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'real_data/news/date={day.date()}.parquet')
        
        print(f"Processed {day.date()}")

# ============================================================================
# OPTION 3: Alpha Vantage News Sentiment
# ============================================================================

def download_news_alphavantage(symbols, start_date, end_date):
    """
    Download news using Alpha Vantage News Sentiment API.
    
    Setup:
        1. Sign up at https://www.alphavantage.co
        2. Get free API key (25 requests/day)
        3. pip install requests transformers torch
        4. Set: export ALPHAVANTAGE_KEY='your_key_here'
    """
    import requests
    
    api_key = os.getenv('ALPHAVANTAGE_KEY')
    if not api_key:
        print("ERROR: Set ALPHAVANTAGE_KEY environment variable")
        print("Get key from: https://www.alphavantage.co/support/#api-key")
        return None
    
    # Load FinBERT
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert')
    model.eval()
    
    os.makedirs('real_data/news', exist_ok=True)
    
    dates = pd.bdate_range(start_date, end_date)
    
    for day in dates:
        embeddings = []
        
        for symbol in symbols:
            try:
                # Alpha Vantage News Sentiment endpoint
                url = f'https://www.alphavantage.co/query'
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'time_from': day.strftime('%Y%m%dT0000'),
                    'time_to': day.strftime('%Y%m%dT2359'),
                    'apikey': api_key
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                # Extract news
                articles = data.get('feed', [])
                texts = [a.get('title', '') + ' ' + a.get('summary', '') for a in articles]
                text = ' '.join(texts) if texts else 'No news.'
                
                # Encode
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                embeddings.append(embedding.tolist())
                
            except Exception as e:
                print(f"Warning: {symbol} - {e}")
                embeddings.append(np.zeros(768).tolist())
        
        # Save
        df = pd.DataFrame({
            'symbol': symbols,
            'news_vec_768': embeddings,
            'date': day.date()
        })
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'real_data/news/date={day.date()}.parquet')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_company_name(symbol):
    """Map ticker to company name for better news search."""
    mapping = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google Alphabet',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta Facebook',
        'NVDA': 'NVIDIA',
        # Add more as needed
    }
    return mapping.get(symbol, symbol)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
    REAL NEWS + FinBERT ENCODING GUIDE
    ===================================
    
    RECOMMENDED APPROACH:
    ---------------------
    1. Sign up for NewsAPI (https://newsapi.org)
       - Free tier: 100 requests/day
       - Historical data up to 1 month
       - 80,000+ sources
    
    2. Get your API key
    
    3. Set environment variable:
       export NEWSAPI_KEY='your_api_key_here'
    
    4. Install dependencies:
       pip install newsapi-python transformers torch
    
    5. Run:
       python download_news_real.py
    
    ALTERNATIVE (FREE BUT LIMITED):
    --------------------------------
    Use yfinance (Option 2) - no API key needed but:
    - Only recent news
    - No historical filtering
    - Rate limited
    
    PRODUCTION TIPS:
    ----------------
    - Cache FinBERT model locally to avoid re-downloading
    - Use GPU for faster encoding: model.to('cuda')
    - Batch process multiple texts together
    - Add retry logic for API failures
    - Store raw news text alongside embeddings for debugging
    """)
    
    # Example usage (uncomment to run):
    # symbols = ['AAPL', 'MSFT', 'GOOGL']
    # download_news_newsapi(symbols, '2023-01-01', '2023-01-31')
