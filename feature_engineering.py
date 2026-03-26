
import polars as pl
import numpy as np

class FeatureEngineer:
    """
    Feature Engineering module using Polars for high-performance calculation
    of technical indicators and other features.
    """
    
    def __init__(self):
        pass

    def add_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all feature engineering steps."""
        df = self.add_technical_indicators(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        return df

    def add_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add basic technical indicators like RSI, MACD, Bollinger Bands."""
        
        # Ensure we sort by date per symbol for correct window functions
        # df = df.sort(["symbol", "date"]) # Assuming caller handles sorting or we do it here
        
        # 1. RSI (Relative Strength Index) - 14 period
        # RSI = 100 - 100 / (1 + RS)
        # RS = Average Gain / Average Loss
        # We'll use EWM for "Average" to approximate Wilder's smoothing
        
        def calculate_rsi(df_lazy):
            delta = pl.col("close").diff()
            gain = delta.clip(lower_bound=0)
            loss = -delta.clip(upper_bound=0)
            
            avg_gain = gain.ewm_mean(com=13, min_periods=14)
            avg_loss = loss.ewm_mean(com=13, min_periods=14)
            
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # 2. MACD (Moving Average Convergence Divergence)
        # MACD Line: 12-day EMA - 26-day EMA
        # Signal Line: 9-day EMA of MACD Line
        # Histogram: MACD Line - Signal Line
        
        def calculate_macd(df_lazy):
            ema12 = pl.col("close").ewm_mean(span=12, adjust=False)
            ema26 = pl.col("close").ewm_mean(span=26, adjust=False)
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm_mean(span=9, adjust=False)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        # 3. Bollinger Bands (20, 2)
        # Middle Band = 20-day SMA
        # Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
        # Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
        
        def calculate_bb(df_lazy):
            middle = pl.col("close").rolling_mean(window_size=20)
            std = pl.col("close").rolling_std(window_size=20)
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            # Normalized bandwidth: (Upper - Lower) / Middle
            bandwidth = (upper - lower) / (middle + 1e-9)
            # %B: (Price - Lower) / (Upper - Lower)
            pct_b = (pl.col("close") - lower) / (upper - lower + 1e-9)
            return bandwidth, pct_b

        # Apply using `over("symbol")`
        df = df.with_columns([
            calculate_rsi(df).over("symbol").alias("ti_rsi_14"),
            
            calculate_macd(df)[0].over("symbol").alias("ti_macd_line"),
            calculate_macd(df)[1].over("symbol").alias("ti_macd_signal"),
            calculate_macd(df)[2].over("symbol").alias("ti_macd_hist"),
            
            calculate_bb(df)[0].over("symbol").alias("ti_bb_bandwidth"),
            calculate_bb(df)[1].over("symbol").alias("ti_bb_pct_b"),
        ])
        
        return df

    def add_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum-based features."""
        
        # ROC (Rate of Change) for different horizons
        # ROC = (Close_t / Close_{t-n}) - 1
        
        horizons = [1, 3, 5, 10, 20]
        exprs = []
        for h in horizons:
            exprs.append(
                ((pl.col("close") / pl.col("close").shift(h) - 1).over("symbol").alias(f"mom_roc_{h}d"))
            )
            
        df = df.with_columns(exprs)
        return df

    def add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility indicators like ATR and historical volatility."""
        
        # ATR (Average True Range)
        # TR = Max(High - Low, Abs(High - Close_prev), Abs(Low - Close_prev))
        # ATR = SMA(TR, 14) 
        
        # We need to compute TR first
        
        tr_calc = pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ])
        
        df = df.with_columns(
            tr_calc.over("symbol").alias("temp_tr")
        )
        
        df = df.with_columns(
            pl.col("temp_tr").rolling_mean(window_size=14).over("symbol").alias("vol_atr_14")
        ).drop("temp_tr")
        
        # Normalized ATR (ATR / Close)
        df = df.with_columns(
            (pl.col("vol_atr_14") / pl.col("close")).alias("vol_natr_14")
        )
        
        return df

    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volume-based indicators."""
        
        # OBV (On-Balance Volume)
        # If Close > Close_prev, OBV += Vol
        # If Close < Close_prev, OBV -= Vol
        # Then we often look at OBV trend (e.g. slope) or divergence. 
        # Here we'll compute simple OBV and maybe a ROC of OBV.
        
        close_diff = pl.col("close").diff()
        direction = pl.when(close_diff > 0).then(1).when(close_diff < 0).then(-1).otherwise(0)
        vol_flow = direction * pl.col("volume")
        
        # Cumulative sum over symbol
        obv = vol_flow.cum_sum()
        
        df = df.with_columns(
            obv.over("symbol").alias("vol_obv")
        )
        
        # OBV Slope (5-day change of OBV)
        df = df.with_columns(
            (pl.col("vol_obv") - pl.col("vol_obv").shift(5)).over("symbol").alias("vol_obv_slope_5d")
        )
        
        # Volume Ratio (Vol / MA(Vol, 20))
        vol_ma = pl.col("volume").rolling_mean(window_size=20)
        df = df.with_columns(
            (pl.col("volume") / (vol_ma + 1)).over("symbol").alias("vol_ratio_20d")
        )
        
        return df

    def add_fundamental_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add fundamental ratios assuming columns exist."""
        # Check if fundamental columns exist
        required = ["net_income", "total_assets", "total_equity", "net_cash_from_operating_activities"]
        if not all([c in df.columns for c in required]):
            # If missing, return as is (or handle gracefully)
            return df
            
        # 1. ROA (Return on Assets)
        # Net Income / Total Assets
        df = df.with_columns(
            (pl.col("net_income") / (pl.col("total_assets") + 1)).fill_null(0).alias("funda_roa")
        )
        
        # 2. ROE (Return on Equity)
        # Net Income / Total Equity
        df = df.with_columns(
            (pl.col("net_income") / (pl.col("total_equity") + 1)).fill_null(0).alias("funda_roe")
        )
        
        # 3. Solvency (Equity / Assets)
        df = df.with_columns(
            (pl.col("total_equity") / (pl.col("total_assets") + 1)).fill_null(0).alias("funda_solvency")
        )
        
        # 4. Earnings Quality (Cash / Net Income)
        # Use abs denominator to handle signs? Usually simple ratio. 
        # Clip outliers.
        df = df.with_columns(
            (pl.col("net_cash_from_operating_activities") / (pl.col("net_income") + 1)).clip(-10, 10).fill_null(0).alias("funda_quality")
        )
        
        # 5. Asset Turnover Proxy (Sales not avail, maybe Net Income / Assets * Margin? No.)
        # Just use what we have.
        
        # 6. Log Total Assets (Size)
        df = df.with_columns(
            pl.col("total_assets").log().fill_null(0).alias("funda_log_assets")
        )
        
        return df

if __name__ == "__main__":
    # Simple test
    print("Feature Engineering Module")
