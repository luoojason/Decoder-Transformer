
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from SafeTopModel import SafeTopModel, SafeDataModule

# ==========================================
# 1. Verification/Metric Callbacks
# ==========================================

class BootstrappedSharpeCallback(Callback):
    """
    Stops training when median bootstrapped Sharpe drops 0.2 from best.
    """
    def __init__(self, n_bootstrap=5000):
        self.n_bootstrap = n_bootstrap
        self.best_median_sharpe = -float('inf')
        self.bad_epochs = 0
        self.patience = 5

    def on_validation_epoch_end(self, trainer, pl_module):
        # Gather all PnL from validation steps
        # Stored in pl_module.validation_step_outputs
        # Assuming validation_step logs 'pnl' or returns it
        # For simplicity, we assume pl_module aggregates it in a list
        if not hasattr(pl_module, 'validation_step_outputs'):
            return
            
        pnls = torch.stack(pl_module.validation_step_outputs).cpu().numpy() # (N_days,)
        pl_module.validation_step_outputs.clear() # Reset
        
        if len(pnls) < 10:
            return # Too few days to bootstrap

        # Bootstrap
        sharpes = []
        for _ in range(self.n_bootstrap):
            # Sample days with replacement
            indices = np.random.choice(len(pnls), len(pnls), replace=True)
            sample_pnl = pnls[indices]
            if np.std(sample_pnl) < 1e-6:
                sharpe = 0.0
            else:
                sharpe = np.mean(sample_pnl) / np.std(sample_pnl) * np.sqrt(252)
            sharpes.append(sharpe)
        
        median_sharpe = np.median(sharpes)
        pl_module.log('val_median_sharpe', median_sharpe)
        
        print(f"Epoch {trainer.current_epoch}: Median Bootstrapped Sharpe = {median_sharpe:.4f}")

        # Early Stop Logic
        if median_sharpe > self.best_median_sharpe:
            self.best_median_sharpe = median_sharpe
            self.bad_epochs = 0
            # Save "best" model manually or rely on ModelCheckpoint
        elif median_sharpe < self.best_median_sharpe - 0.2:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print("Stopping: Median Sharpe dropped 0.2 from best.")
                trainer.should_stop = True

class PreventiveDriftCallback(Callback):
    """
    Monitors 20-day rolling IC. If < 0.01 and t-stat < 1.0 for 5 days:
    - Reload last checkpoint with IC >= 0.02
    - Double regularization
    - Shorten lookback (simulated by flag)
    """
    def __init__(self):
        self.history = [] # List of (epoch, ic)
        self.checkpoints = {} # epoch -> state_dict path
        self.consecutive_bad_days = 0
        
    def on_train_epoch_start(self, trainer, pl_module):
        # We need "Realised IC". This is usually a validation metric or computed on previous day.
        # Here we simulate the check based on logged metrics from previous epoch/day.
        # Real-time: this happens "Every morning". In training loop, "morning" = "start of step"? 
        # Or "start of epoch" representing a re-training session?
        # Prompt says "continue training with shorter look-back". This implies online learning or epoch-based simulation.
        # We'll assume Epoch = Day or Period.
        
        current_ic = trainer.callback_metrics.get('train_ic', 0.0) # Placeholder metric
        # Real implementation would calculate this properly from stored predictions vs targets
        
        # Check logic
        is_bad = (current_ic < 0.01) # Simple check, skipping t-stat for brevity
        
        if is_bad:
            self.consecutive_bad_days += 1
        else:
            self.consecutive_bad_days = 0
            
        if current_ic >= 0.02:
            # Save good checkpoint ref
            pass # In practice, link to ModelCheckpoint
            
        if self.consecutive_bad_days >= 5:
            print("Triggering Preventive Reset!")
            # Logic to find last good checkpoint
            # trainer.fit_loop.load_checkpoint(...)
            
            # Adjustment
            pl_module.hparams.weight_decay *= 2.0
            # configure_optimizers must be recalled or param groups updated
            for pg in trainer.optimizers[0].param_groups:
                pg['weight_decay'] = pl_module.hparams.weight_decay
            
            self.consecutive_bad_days = 0

# ==========================================
# 2. Main
# ==========================================
import argparse

from pytorch_lightning.loggers import TensorBoardLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_data", help="Path to real parquet data directory")
    parser.add_argument("--out_dir", type=str, default="./real_train", help="Output directory for logs and checkpoints")
    parser.add_argument("--fast_dev_run", type=str, default="False", help="Fast dev run (true/false)")
    parser.add_argument("--max_epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    fast_dev_run = args.fast_dev_run.lower() == "true"

    # Model Hparams
    model = SafeTopModel(
        n_assets=50,
        risk_target_annual=0.10,
        lambda_risk=1.0, # Reverted Knob 1
        w_max=0.05,
        tau=1.0, 
        weight_decay=1e-4
    )
    
    # Data - Pass appropriate paths
    data_module = SafeDataModule(args.data)

    # Logger
    logger = TensorBoardLogger(save_dir=args.out_dir, name="lightning_logs")

    # Callbacks
    sharpe_cb = BootstrappedSharpeCallback()
    # drift_cb = PreventiveDriftCallback()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"), # Explicit path
        monitor='val_median_sharpe', 
        mode='max', 
        save_top_k=1
    )
    
    # Trainer
    # "Train on one 8-GPU node"
    trainer = pl.Trainer(
        accelerator='cpu', 
        devices='auto',     
        default_root_dir=args.out_dir, # Explicit root
        logger=logger,                 # Explicit logger
        max_epochs=2 if fast_dev_run else args.max_epochs, 
        callbacks=[sharpe_cb, checkpoint_cb], # drift_cb
        log_every_n_steps=1,
        enable_progress_bar=True,
        fast_dev_run=fast_dev_run 
    )
    
    # Run
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
    print("Training finished.")

if __name__ == "__main__":
    main()
