
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from SafeTopModel import SafeTopModel, SafeDataModule

def main():
    print("Starting Training with TensorBoard Logging (real_train_m4)...")
    
    # 1. Logger
    # User requested 'real_train_m4'
    logger = TensorBoardLogger(
        save_dir=".", 
        name="real_train_m4",
        default_hp_metric=False 
    )
    
    # 2. Model
    # Uses the updated SafeTopModel with Feature Engineering & Reduced Costs
    model = SafeTopModel(
        n_assets=50,
        risk_target_annual=0.10,
        lambda_risk=1.0,
        w_max=0.05,
        weight_decay=1e-3
    )
    
    # Optimizer Config (Best from tuning)
    def configure_optimizers():
        return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.configure_optimizers = configure_optimizers
    
    # 3. Data
    dm = SafeDataModule(data_dir="real_data", batch_size=1)
    
    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="real_train_m4/checkpoints",
        filename="model-{epoch:02d}-{train_loss:.4f}-{pnl:.6f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 5. Trainer
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=50, # Extended training "until fit"
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10, # Frequent logging for TensorBoard
        limit_train_batches=200, # Same batch size as before
        limit_val_batches=50
    )
    
    print(f"Logging to: {logger.log_dir}")
    trainer.fit(model, datamodule=dm)
    
    print("Training Complete.")

if __name__ == "__main__":
    main()
