
import os
import argparse
import torch
import polars as pl
import numpy as np
from SafeTopModel import SafeTopModel, JohnsonSU

def load_latest_checkpoint(ckpt_dir):
    # Find .ckpt file with highest epoch/step or matching best metric
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found.")
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not files:
        raise FileNotFoundError("No checkpoints found.")
    # Sort by modification time or name
    latest = sorted(files, key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))[-1]
    return os.path.join(ckpt_dir, latest)

def get_latest_data(data_path):
    # Load polars DF, filter to last available date
    # In production, this would connect to a DB or API
    # Return a Batch dictionary compatible with SafeTopModel.forward
    print(f"Loading data from {data_path}...")
    # Dummy creation for demonstration
    # Assume 50 assets
    N = 50
    return {
        'micro': torch.randn(1, N, 5, 60),
        'micro_last': torch.randn(1, N, 4),
        'news': torch.randn(1, N, 768),
        'quant': torch.randn(1, N, 20),
        'w_prev': torch.zeros(1, N), # Need actual previous weights
        'sector_map': torch.randn(1, 11, N), # Dummy sector map
        'metadata': {'tickers': [f'TICK_{i}' for i in range(N)]}
    }

def run_roll(ckpt_path, data_path, out_path):
    # 1. Load Model
    print(f"Loading model from {ckpt_path}...")
    model = SafeTopModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()
    
    # 2. Get Data
    batch = get_latest_data(data_path)
    
    # Unbatch if necessary (remove batch dim 1)
    # SafeTopModel.forward expects inputs for N assets (unbatched or batch-handled).
    # Since batch creation added dim 0 (1, N...), and SafeTopModel training unbatches manually,
    # we should unbatch here too to align with forward logic (which usually takes batched but we want 1 day).
    # Actually forward handles standard batch. 
    # But batch['micro'] is (1, N, C, L). Forward passes this to micro_encoder.
    # MicroEncoder expects (B, C, L). 
    # If we pass (1, N, C, L), it fails or treats 1 as B?
    # Actually training_step does: micro = batch['micro'][0].
    # So forward logic assumes input is (N, C, L).
    # So we MUST unbatch here.
    
    unbatched = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            unbatched[k] = v[0]
        else:
            unbatched[k] = v
    batch = unbatched
    
    # 3. Model Forward (C, D)
    with torch.no_grad():
        mu, cluster_w, expert_preds, jsu_params = model(batch)
    
    # 4. Simulation (E)
    # Simulate 5000 returns
    gamma, delta, xi, lambd = jsu_params
    jsu_dist = JohnsonSU(gamma, xi, lambd, delta)
    
    # 5000 samples
    S = 5000
    # Independent simulation for now (or load covariance from data bundle if available)
    # gamma is (1, N). We want R_sim (S, N).
    # Pass (S, N) to rsample? 
    # z will be (S, N). gamma (1, N) broadcasts.
    N = model.hparams.n_assets
    R_sim = jsu_dist.rsample((S, N)) # (S, N)
    # If returned shape is (S, N, 1, N) or something due to broadcasting, check JSU implementation.
    # JSU rsample uses z shape as base. (S, N). 
    # (S, N) - (1, N) -> (S, N). Correct.
    # Note: previous impl R_sim was (S, B, N) -> (S, 1, 50).
    # If we pass (S, N), we get (S, N).
    
    # R_sim = R_sim.squeeze(1) # No need if we generated (S, N)

    
    # 5. Optimization (E, F)
    # Using the CVXPY Layer (or raw solver). Layer is already in model.
    # Note: Layer expects fixed S. If training S != Inference S, we might need to separate the solver logic
    # or rebuild the layer.
    # SafeTopModel.cvx_layer was built with S_sim (e.g. 500 or 1000).
    # If we want S=5000, we must rebuild the solver or loop?
    # Actually, CVXPY parameters (R_sim) size must match compiled layer.
    # Easier to just solve raw cvxpy problem here since we don't need gradients.
    
    print("Solving optimization problem...")
    import cvxpy as cp
    w = cp.Variable(model.hparams.n_assets)
    
    # Extract params
    mu_np = mu.squeeze(0).numpy()
    w_prev_np = batch['w_prev'].squeeze(0).numpy()
    R_sim_np = R_sim.numpy()
    sector_map_np = batch['sector_map'].squeeze(0).numpy()
    
    lambda_risk = model.hparams.lambda_risk
    risk_target = model.hparams.risk_target_annual / np.sqrt(252)
    w_max = model.hparams.w_max
    
    # Formulation
    alpha = 0.95
    z = cp.Variable(1)
    u = cp.Variable(S, nonneg=True)
    
    losses = - R_sim_np @ w
    cvar = z + (1.0 / (S * (1.0 - alpha))) * cp.sum(u)
    
    constraints = [
        u >= losses - z,
        cvar <= risk_target,
        cp.sum(cp.abs(w)) <= 2.0,
        cp.abs(w) <= w_max,
        cp.sum(cp.abs(w - w_prev_np)) <= 0.20,
        sector_map_np @ w == 0
    ]
    
    objective = cp.Maximize(mu_np @ w - lambda_risk * cvar)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS) # Robust solver
    except Exception:
        print("ECOS failed, trying SCS...")
        prob.solve(solver=cp.SCS)
    
    if w.value is None:
        print("Optimization failed/infeasible. Fallback to w_prev or zero.")
        w_tgt = w_prev_np
    else:
        w_tgt = w.value
        
    # 6. Output
    tickers = batch['metadata']['tickers']
    # Calculate sigma (std of returns) for the tear sheet/output
    # R_sim is (S, N)
    sigma_vec = R_sim.std(dim=0).numpy() # (N,)
    
    # Cluster ID: max weight in cluster_w (N, K)
    cluster_ids = cluster_w.argmax(dim=1).numpy() # (N,)
    
    df_out = pl.DataFrame({
        "ticker": tickers,
        "weight": w_tgt,
        "mu": mu_np,
        "sigma": sigma_vec,
        "cluster_id": cluster_ids
    })
    
    df_out.write_csv(out_path)
    print(f"Target weights written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or dir")
    parser.add_argument("--data", type=str, required=True, help="Path to real parquet data directory")
    parser.add_argument("--out", type=str, default="target_weights.csv")
    parser.add_argument("--date", type=str, help="Date to roll from (YYYY-MM-DD)")
    args = parser.parse_args()
    
    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        ckpt = load_latest_checkpoint(ckpt)
        
    run_roll(ckpt, args.data, args.out)
