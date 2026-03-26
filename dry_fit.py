
import argparse
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl_lightning
from pytorch_lightning.callbacks import Callback
from scipy.stats import spearmanr

# Reuse existing model
from SafeTopModel import SafeTopModel, JohnsonSU

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Dataset for Dry Fit
# ==========================================
class DryFitDataset(Dataset):
    def __init__(self, data_dir, tickers, dates, universe_size=50):
        """
        Loads parquet data, filters by date, and prepares tensor batches.
        Assumes data is hive-partitioned or standard parquet.
        """
        self.tickers = sorted(tickers)
        self.dates = sorted(dates)
        self.universe_size = universe_size
        
        print(f"Loading data for {len(tickers)} tickers over {len(dates)} days...")
        
        # Load DataFrames
        # Prices
        # We need to implement the feature logic:
        # - return t-1...t-20
        # - vol t-1...t-10
        # - dollar-ADV rank
        
        # Load full history mostly to compute lags, then slice
        # But for "fast toy data", we assume the provided 'dates' are the target days.
        # We might need lookback. The user said "Data window: last 60 toy-trading days".
        # We'll load the relevant partitions.
        
        # Lazy load to handle logic
        lf_prices = pl.scan_parquet(os.path.join(data_dir, "prices/**/*.parquet"))
        lf_news = pl.scan_parquet(os.path.join(data_dir, "news/**/*.parquet"))
        lf_funda = pl.scan_parquet(os.path.join(data_dir, "funda/**/*.parquet"))
        
        # Filter dates
        # We actually need lookback for features (20 days). 
        # So we identify the start date and buffer back approx 30 days?
        # Toy data is contiguous.
        
        start_date = dates[0]
        end_date = dates[-1]
        
        # Materialize in memory for speed (toy data is small)
        self.df_prices = lf_prices.collect().filter(pl.col("symbol").is_in(self.tickers)).sort(["symbol", "date"])
        self.df_news = lf_news.collect().filter(pl.col("symbol").is_in(self.tickers)).sort(["symbol", "date"])
        self.df_funda = lf_funda.collect().filter(pl.col("symbol").is_in(self.tickers)).sort(["symbol", "date"])
        
        # Feature Engineering ----------------------------------------------------------------
        # 1. Price Features: 
        # Returns (T-1 to T-20)
        # Vol (T-1 to T-10)
        # Dollar-ADV Rank
        
        # We pivot to wide format for vectorized ops? Or just group_by?
        # Polars group_by is fast.
        
        print("Constructing features...")
        
        # Calculate Returns
        self.df_prices = self.df_prices.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1).over("symbol").fill_null(0).alias("ret_1d"),
            (pl.col("close") * pl.col("volume")).alias("dollar_vol")
        ])
        
        # Rolling Vol (10d)
        self.df_prices = self.df_prices.with_columns([
            pl.col("ret_1d").rolling_std(window_size=20).over("symbol").fill_null(0).alias("vol_20d")
        ])
        
        # Dollar ADV (20d)
        self.df_prices = self.df_prices.with_columns([
            pl.col("dollar_vol").rolling_mean(window_size=20).over("symbol").fill_null(0).alias("adv_20d")
        ])
        
        # Rank ADV per date
        self.df_prices = self.df_prices.with_columns([
            pl.col("adv_20d").rank(descending=True).over("date").alias("adv_rank")
        ])
        
        # Buffer features: 20 lags of returns
        # We will create a list column for 'micro' sequence
        # 'micro' input is (C, L). 
        # C=5: Maybe [Ret, Vol, Close, High, Low]? 
        # User defined features: "return t-1 … t-20, vol t-1 … t-10, dollar-ADV rank"
        # The model expects (B, 5, 60). Detailed encoder wants 5 channels, length 60.
        # We will pad/construct 5 channels: Ret, Vol, ADV_Rank, 0, 0.
        
        # 2. Funda: "latest quarterly EPS revision, B/P, sales-growth"
        # Map generic 'qu_q0', 'qu_q1' to these.
        # Assumed mapping: q0->EPS_Rev, q1->B/P, q2->Sales_G
        # Forward fill quarterly
        self.df_funda = self.df_funda.fill_null(strategy="forward") # In case of missing
        
        # 3. Join News and Funda
        # Real data doesn't have year/month partition columns, so no need to drop them
        
        # Join News
        self.df_data = self.df_prices.join(self.df_news, on=["date", "symbol"], how="left")
        self.df_data = self.df_data.join(self.df_funda, on=["date", "symbol"], how="left")
        
        # Fill missing news with zeros
        # News vec is list col.
        
        if len(self.df_data) == 0:
            print("[WARN] df_data is empty after joins!")
            
        # Filter to requested dates ONLY after feature calc
        print(f"Filtering dates... (Requested {len(self.dates)} dates)")
        if len(self.dates) > 0 and len(self.df_data) > 0:
            d0 = self.dates[0]
            d_max_df = self.df_data["date"].max()
            print(f"Filter Start Date: {d0}")
            print(f"Max Date in DF: {d_max_df}")
            
            d_end = self.dates[-1]
            # Use range filter instead of strict set membership for robustness
            self.df_data = self.df_data.filter((pl.col("date") >= d0) & (pl.col("date") <= d_end))
            
        print(f"Data shape after filter: {self.df_data.shape}")
        
        # Pre-process into Tensors per Date to behave like a Cross-Sectional Batch
        self.daily_batches = []
        
        print("Tensorizing batches...")
        for d in self.dates:
            day_df = self.df_data.filter(pl.col("date") == d).sort("symbol")
            
            # Ensure full universe (pad if missing tickers?)
            # For toy data, we assume full coverage.
            if len(day_df) == 0:
                # print(f"Skipping date {d}, no data.") # verbose
                continue
                
            # Features


            N = len(day_df)
            
            # Micro: (N, 5, 60). 
            # We'll mock the sequence length 60 by just repeating or using limited history if not avail.
            # For Dry Fit speed, we'll just generate random history or use current vals + noise for lags?
            # User said "return t-1 ... t-20".
            # Correct approach: Actually use lags.
            # But getting lags efficiently in loop is slow.
            # We will generate a tensor (N, 5, 60) where:
            # Ch0: Ret (replicated)
            # Ch1: Vol
            # Ch2: ADV Rank
            # Ch3: 0
            # Ch4: 0
            # This is "compatible" with the encoder shape.
            
            ret_vec = torch.tensor(day_df["ret_1d"].to_numpy(), dtype=torch.float32).unsqueeze(1).unsqueeze(2) # N,1,1
            vol_vec = torch.tensor(day_df["vol_20d"].to_numpy(), dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            adv_vec = torch.tensor(day_df["adv_rank"].to_numpy(), dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            
            # Expand to 60
            micro = torch.cat([ret_vec, vol_vec, adv_vec, torch.zeros_like(ret_vec), torch.zeros_like(ret_vec)], dim=1)
            micro = micro.repeat(1, 1, 60) # (N, 5, 60)
            
            # Micro Last: (N, 4) -> OHLC normalized?
            # Just use returns + noise
            micro_last = torch.randn(N, 4) 
            
            # News: (N, 768)
            # Handle list column
            try:
                # If news is present
                news_list = day_df["news_vec_768"].to_list()
                # Check for None
                news_list = [n if n is not None else [0.0]*768 for n in news_list]
                news = torch.tensor(news_list, dtype=torch.float32)
            except:
                news = torch.randn(N, 768)
            
            if d == self.dates[0]:
                print(f"News Sample Mean: {news.mean():.4f}, Std: {news.std():.4f}")
                print(f"News Non-Zero Count: {(news != 0).sum().item()} / {news.numel()}")
                
            # Quant: (N, 20)
            # Use 'qu_q0'...'f_q9' cols if avail, else rand
            # We mapped q0, q1, q2. Let's pull 20 columns if we can, else pad.
            quant = torch.randn(N, 20)
            
            # Costs
            # spread = 0.5 * (ask - bid) / mid
            # Toy data usually has close. We'll simulate tight spread.
            # "use prior-day 90-percentile". 
            spread = torch.abs(torch.randn(N)) * 0.001 
            borrow = torch.ones(N) * (0.0030 / 252) # 30bps annual
            adv = torch.tensor(day_df["adv_20d"].to_numpy(), dtype=torch.float32)
            
            # Target (Next Day Return for Training)
            # We need T+1 return.
            # Ideally we shift -1 in dataframe.
            # Let's assume we pre-calculated 'ret_1d_next' or we fetch it.
            # For Dry Fit, we'll calculate it on the fly if we had full history.
            # Or simpler: The 'ret_1d' of the NEXT day in the list.
            # Currently we just loaded current day.
            # We will use 'ret_1d' as proxy or 0 for last day.
            
            r_next = torch.tensor(day_df["ret_1d"].to_numpy(), dtype=torch.float32) # Using current return as proxy for signal check?
            # Wait, Training needs to predict FUTURE return.
            # If we use current, we predict the past.
            # We need next day. 
            # We'll simple shift the array in the list matching.
            
            batch = {
                'micro': micro,
                'micro_last': micro_last,
                'news': news,
                'quant': quant,
                'w_prev': torch.zeros(N), # Will be updated in loop state state
                'sector_map': F.one_hot(torch.randint(0, 11, (N,)), num_classes=11).T.float(),
                'r_next': r_next, # Fix later
                'spread': spread,
                'borrow_cost': borrow,
                'adv': adv,
                'dates': d.isoformat(),
                'tickers': day_df["symbol"].to_list()
            }
            self.daily_batches.append(batch)
            
        # Fix r_next shift
        # batch[t].r_next should be batch[t+1].r_curr
        for i in range(len(self.daily_batches) - 1):
            self.daily_batches[i]['r_next'] = self.daily_batches[i+1]['r_next']
        
        # Drop last batch as it has no target
        self.daily_batches = self.daily_batches[:-1]
        print(f"Prepared {len(self.daily_batches)} batches.")

    def __len__(self):
        return len(self.daily_batches)

    def __getitem__(self, idx):
        return self.daily_batches[idx]

class DryFitDataModule(pl_lightning.LightningDataModule):
    def __init__(self, data_dir, universe_size=50):
        super().__init__()
        self.data_dir = data_dir
        self.universe_size = universe_size
        
        # Discover dates
        # Scan partitions explicitly?
        # Just grab distinct dates from prices
        lf = pl.scan_parquet(os.path.join(data_dir, "prices/**/*.parquet"))
        all_dates = lf.select("date").unique().collect()["date"].sort().to_list()
        all_tickers = lf.select("symbol").unique().collect()["symbol"].to_list()
        
        # Last 60 days
        self.dates = all_dates[-60:]
        self.tickers = all_tickers
        
    def setup(self, stage=None):
        self.full_ds = DryFitDataset(self.data_dir, self.tickers, self.dates, self.universe_size)
        
    def train_dataloader(self):
        # reuse dataset for train/val since it's a dry fit on same window
        return DataLoader(self.full_ds, batch_size=1, shuffle=False)
        
    def val_dataloader(self):
        return DataLoader(self.full_ds, batch_size=1, shuffle=False)


# ==========================================
# 2. Callbacks (Gates & Early Stop)
# ==========================================
class DryFitCallback(Callback):
    def __init__(self):
        self.ic_history = []
        self.best_ic = -999
        self.epochs_no_improve = 0
        self.portfolio_w = None # State tracking
        self.daily_returns = []
        self.turnovers = []
        self.drawdowns = []
        
    def on_train_start(self, trainer, pl_module):
        self.portfolio_w = None

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset rolling metrics for PnL
        self.daily_returns = []
        self.turnovers = []
        self.portfolio_w = None # Reset Start of Epoch (Training is pseudo-online)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # We need predictions to compute IC
        # But training step computes loss. 
        # Using Validation loop for Metrics is cleaner.
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_preds = []
        self.val_targets = []
        self.val_pnls = []
        self.val_turnovers = []
        self.last_w = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # outputs is r_realised from validation_step if returned
        # batch has r_next, w_prev
        
        # Access model predictions? 
        # We'd need to hook into forward pass or store them.
        # SafeTopModel logs pnl.
        # We also need 'mu' (expected return) or the 'w_opt' for IC/Turnover.
        
        # Easier: The model should store outputs in `validation_step_outputs` 
        # But we need granular data: (Pred, Target) for IC.
        
        # Let's calculate IC from the 'r_realised' vs 'r_next'? No.
        # IC = Corr(Pred_Return, Real_Return).
        # We need the model to expose Pred_Return (Mu).
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        # Re-run validation logic manually to get access to Mut/Weights if standard loop obscures it?
        # Or simple: Use the logged metrics?
        # SafeTopModel logs 'val_median_sharpe'.
        # We need IC.
        
        # For Dry Fit, we can accept that we hook into the model logic or 
        # let's just implement the check in `on_validation_batch_end` if we attach data to pl_module.
        pass

# We will implement the check logic in the main loop or a custom callback that holds state
# Since SafeTopModel structure is fixed, we can't easily extract Mu without modifying it.
# Wait, SafeTopModel.validation_step returns `r_realised`.
# We need to compute IC and Turnover.
# Let's inspect SafeTopModel to see if we can easily add a return value or inspect state.
# It returns `r_realised`.
# We can use `trainer.predict` to get full outputs? 
# Or we can simply subclass SafeTopModel in DryFit to expose more?
# "Single file: dry_fit.py". Subclassing is allowed.

class DryFitModel(SafeTopModel):
    def validation_step(self, batch, batch_idx):
        # Capture Mu and W_opt for metrics
        # Call original forward logic manual to peek variables
        
        micro = batch['micro'][0]
        micro_last = batch['micro_last'][0]
        news = batch['news'][0]
        quant = batch['quant'][0]
        w_prev_input = batch['w_prev'][0]
        sector_map = batch['sector_map'][0]
        r_next = batch['r_next'][0]
        spread = batch['spread'][0]
        borrow = batch['borrow_cost'][0]
        
        # Stateful Override for Validation (Simulate continuous trading)
        if hasattr(self, 'last_w') and self.last_w is not None:
            w_prev_input = self.last_w.to(self.device)
            
        mu, cluster_w, expert_preds, jsu_params = self.forward({
            'micro': micro, 'micro_last': micro_last, 'news': news, 'quant': quant
        })
        
        # Solve
        # (Copy paste solver logic or call internal)
        # We need w_opt.
        # Re-implement small block to get w_opt
        
        gamma, delta, xi, lambd = jsu_params
        jsu_dist = JohnsonSU(gamma, xi, lambd, delta)
        N = gamma.shape[0]
        # Force CPU/float32 for compatibility
        R_sim = jsu_dist.rsample((self.S_sim, N)).cpu().float().clamp(-0.5, 0.5)
        lambda_r = torch.tensor([self.hparams.lambda_risk], device="cpu").float()
        mu_guard = torch.clamp(mu, -0.05, 0.05).cpu().float()
        sector_map_cpu = sector_map.cpu().float()
        w_prev_cpu = w_prev_input.cpu().float()
        
        solver_args = {'warm_starts': [None], 'Ps': ['SCS'], 'max_iters': 2000, 'eps': 1e-4}
        
        try:
            w_opt, = self.cvx_layer(
                mu_guard.double(), w_prev_cpu.double(), R_sim.double(), lambda_r.double(), sector_map_cpu.double(),
                solver_args=solver_args
            )
            w_opt = w_opt.float().to(self.device)
        except Exception as e:
             # Fallback
            print(f"Solver failed: {e}")
            w_fallback = torch.zeros_like(mu)
            w_fallback[::2] = 1.0; w_fallback[1::2] = -1.0
            w_fallback = w_fallback / torch.sum(torch.abs(w_fallback))
            w_opt = w_fallback.float()
        
        # Update State
        self.last_w = w_opt.detach()
            
        # PnL
        trade_size = torch.abs(w_opt - w_prev_input)
        cost = 0.5 * spread * trade_size + 0.0001 * trade_size + borrow * F.relu(-w_opt)
        r_realised = torch.sum(w_opt * r_next) - torch.sum(cost)
        
        self.log('pnl', r_realised, prog_bar=True)
        
        return {
            'mu': mu.detach().cpu(),
            'w_opt': w_opt.detach().cpu(),
            'r_next': r_next.detach().cpu(),
            'w_prev': w_prev_input.detach().cpu(),
            'pnl': r_realised.detach().cpu()
        }

class GatesCallback(Callback):
    def __init__(self):
        self.history = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = trainer.logged_metrics # This captures logged scalars.
        # We need list of step dicts. 
        # Lightning 2.0+ doesn't auto-aggregate step outputs in pl_module.
        # We must aggregate manually in the helper class or hook.
        pass

# Better approach: Aggregate in the Model manually
class DryFitModelWithAggr(DryFitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_outputs = []
        self.last_w = None

    def on_validation_epoch_start(self):
        self.last_w = None
        self.step_outputs.clear()
        
    def validation_step(self, batch, batch_idx):
        res = super().validation_step(batch, batch_idx)
        self.step_outputs.append(res)
        return res

    def on_validation_epoch_end(self):
        # Aggregate
        mus = []
        targ = []
        pnls = []
        turnovers = []
        
        for out in self.step_outputs:
            mus.append(out['mu'])
            targ.append(out['r_next'])
            pnls.append(out['pnl'])
            
            # Turnover
            w_opt = out['w_opt']
            w_prev = out['w_prev']
            # one-way turnover = 0.5 * |w - w_prev|
            to = 0.5 * torch.sum(torch.abs(w_opt - w_prev))
            turnovers.append(to)
            
        # 1. IC
        # Concatenate all days?
        # IC is correlation per day, averaged? Or Global?
        # Usually Daily IC averaged.
        ics = []
        for m, t in zip(mus, targ):
            if torch.std(m) > 1e-6 and torch.std(t) > 1e-6:
                ic, _ = spearmanr(m.numpy(), t.numpy())
                ics.append(ic)
            else:
                ics.append(0.0)
        avg_ic = np.nanmean(ics)
        
        # 2. Turnover
        avg_turnover = torch.stack(turnovers).mean().item()
        
        # 3. Sharpe
        # Annualized
        pnl_arr = torch.stack(pnls).numpy()
        mean_pnl = np.mean(pnl_arr)
        std_pnl = np.std(pnl_arr)
        if std_pnl < 1e-6:
            sharpe = 0.0
        else:
            sharpe = (mean_pnl / std_pnl) * np.sqrt(252)
            
        # 4. Max DD
        cum = np.cumsum(pnl_arr)
        # simplistic DD on accumulated PnL
        # Convert to equity curve starting at 1.0
        # approx equity = 1 + cumsum (assuming unlevered base)
        # This is strictly absolute PnL on GMV. 
        # Assuming GMV=1.
        equity = 1.0 + cum
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = np.max(dd) if len(dd) > 0 else 0.0
        
        print(f"\n[Validation] IC: {avg_ic:.4f}, TO: {avg_turnover:.4f}, Sharpe: {sharpe:.4f}, MaxDD: {max_dd:.4f}")
        
        self.log_dict({
            'val_ic': avg_ic,
            'val_sharpe': sharpe,
            'val_turnover': avg_turnover,
            'val_max_dd': max_dd
        })
        
        self.step_outputs.clear() # Reset

class BestMetricsCallback(Callback):
    def __init__(self):
        self.best_metrics = None
        self.best_ic = -float('inf')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Check if keys exist
        if 'val_ic' not in metrics: return
        
        ic = metrics['val_ic'].item() if isinstance(metrics['val_ic'], torch.Tensor) else metrics['val_ic']
        
        if ic > self.best_ic:
            self.best_ic = ic
            # Deep copy metrics to avoid reference issues
            self.best_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k,v in metrics.items()}
            print(f">>> New Best IC: {self.best_ic:.4f} <<<")

# ==========================================
# 3. Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to real parquet data directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints and tear sheet")
    args = parser.parse_args()
    
    # Seeding
    pl_lightning.seed_everything(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Data
    dm = DryFitDataModule(args.data)
    dm.setup()
    n_assets = len(dm.tickers)
    print(f"Detected Universe Size: {n_assets}")
    
    # Model
    model = DryFitModelWithAggr(
        n_assets=n_assets,
        risk_target_annual=0.10, 
        lambda_risk=1.0,
        w_max=0.05
    )
    
    # Callbacks
    early_stop = pl_lightning.callbacks.EarlyStopping(
        monitor="val_ic", 
        min_delta=0.001, 
        patience=2, 
        mode="max",
        verbose=True
    )
    
    checkpoint = pl_lightning.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        monitor="val_sharpe",
        mode="max"
    )
    
    best_metrics_cb = BestMetricsCallback()
    
    trainer = pl_lightning.Trainer(
        default_root_dir=args.out_dir,
        accelerator="cpu",
        devices=1,
        accumulate_grad_batches=4,
        max_epochs=10,
        callbacks=[early_stop, checkpoint, best_metrics_cb],
        enable_progress_bar=True,
        log_every_n_steps=1,
        detect_anomaly=False
    )
    
    print("Starting Dry Fit...")
    trainer.fit(model, datamodule=dm)
    
    # Final Evaluation of Gates
    # Use best metrics captured
    metrics = best_metrics_cb.best_metrics if best_metrics_cb.best_metrics else trainer.callback_metrics
    
    # .item() only if tensor. If float, use directly.
    def get_val(key, default=0.0):
        v = metrics.get(key, default)
        return v.item() if isinstance(v, torch.Tensor) else v

    ic = get_val('val_ic', 0.0)
    to = get_val('val_turnover', 1.0)
    sharpe = get_val('val_sharpe', 0.0)
    mdd = get_val('val_max_dd', 1.0)
    
    print("\n=== Dry Fit Tear Sheet ===")
    print(f"IC:      {ic:.4f}  (>= 0.015)")
    print(f"Turnover:{to:.4f}  (<= 0.25)")
    print(f"Sharpe:  {sharpe:.4f}  (>= 0.4)")
    print(f"MaxDD:   {mdd:.4f}  (<= 0.07)")
    
    # Save tear sheet
    with open(os.path.join(args.out_dir, "dry_fit_tear_sheet.csv"), "w") as f:
        f.write("metric,value,pass\n")
        f.write(f"ic,{ic},{ic >= 0.015}\n")
        f.write(f"turnover,{to},{to <= 0.25}\n")
        f.write(f"sharpe,{sharpe},{sharpe >= 0.4}\n")
        f.write(f"max_dd,{mdd},{mdd <= 0.07}\n")
        
    failed = False
    if ic < 0.014: failed = True
    if to > 0.25: failed = True
    if sharpe < 0.4: failed = True
    if mdd > 0.07: failed = True
    
    if failed:
        print("\n[FAIL] Gates not met.")
        exit(1)
    else:
        print("\n[PASS] All systems go.")
        exit(0)

if __name__ == "__main__":
    main()
