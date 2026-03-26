# Decoder Transformer

A transformer-based portfolio optimization model that combines multi-modal financial data (prices, fundamentals, news sentiment, orderbook microstructure) to generate optimal portfolio weights.

## Architecture

The core model (`SafeTopModel`) is built on PyTorch Lightning and features:

- **Microstructure Encoder** — CNN with gating for orderbook features (spread, borrow cost)
- **Multi-Modal Fusion** — Combines price, fundamental, news, and microstructure signals into a unified 256-d representation
- **Johnson SU Distribution Head** — Models non-Gaussian return distributions with skewness and kurtosis
- **CVXPY Optimization Layer** — Differentiable convex optimizer that outputs portfolio weights subject to sector neutrality and risk constraints
- **Expert Clustering** — Learns market regime clusters for conditional prediction

## Project Structure

```
├── SafeTopModel.py           # Core model architecture
├── feature_engineering.py    # Technical indicator computation (RSI, MACD, Bollinger, etc.)
├── train.py                  # Training script with PyTorch Lightning
├── roll.py                   # Rolling inference for production deployment
├── dry_fit.py                # Backtesting and validation framework
├── run_smoke.sh              # Integration test suite
├── scripts/
│   ├── download_prices.py          # Market price data (Yahoo Finance, 100 symbols)
│   ├── download_fundamentals.py    # Fundamental data (SimFin)
│   ├── download_news.py            # News sentiment embeddings (NewsAPI + FinBERT)
│   ├── download_news_synthetic.py  # Synthetic news embeddings for backtesting
│   └── download_orderbook.py       # Synthetic orderbook/transaction cost data
├── tests/
│   ├── test_model.py         # Model component unit tests
│   ├── test_cvx.py           # CVXPY optimization layer tests
│   ├── test_train.py         # Training loop integration tests
│   └── test_roll.py          # Rolling inference tests
└── PROJECT_MANUAL.docx       # Detailed documentation
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

Set these before running download scripts:

```bash
export SIMFIN_API_KEY="your_key"     # For fundamental data (simfin.com)
export NEWSAPI_KEY="your_key"        # For news data (newsapi.org)
```

## Usage

### 1. Download Data

```bash
python scripts/download_prices.py
python scripts/download_fundamentals.py
python scripts/download_news.py            # Real news (requires API key)
python scripts/download_news_synthetic.py   # Or synthetic for backtesting
python scripts/download_orderbook.py
```

Data is stored as Hive-partitioned Parquet files in `real_data/` (year=YYYY/month=MM/day=DD.parquet).

### 2. Train

```bash
python train.py --data real_data/ --max_epochs 50
```

### 3. Validate

```bash
python dry_fit.py --data real_data/ --out_dir ./validation
```

### 4. Rolling Inference

```bash
python roll.py --ckpt checkpoints/best.ckpt --data real_data/ --out weights.csv --date 2024-01-15
```

### 5. Run Tests

```bash
chmod +x run_smoke.sh
./run_smoke.sh
```

## Data Flow

```
Vendor Data → real_data/ (Parquet) → train.py → Checkpoint (.ckpt)
                                   → roll.py  → target_weights.csv
```

## License

MIT
