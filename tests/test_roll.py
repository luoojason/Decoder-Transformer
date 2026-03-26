
import pytest
import subprocess
import os
import polars as pl
import sys

def test_roll_execution():
    # Helper to create a dummy checkpoint if none exists?
    # We rely on test_train to have created one, or we mock it.
    # But integration tests inside unit tests are tricky.
    # We'll use a mocked run of roll.py main logic if possible, OR
    # just assume `lightning_logs` has something if we run sequentially.
    
    # Better: Use the `run_smoke.sh` workflow which guarantees train runs first.
    # But here we need to test `roll.py` logic.
    # We can import `run_roll` and mock the checkpoint loading.
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from roll import run_roll
    from unittest.mock import MagicMock, patch
    
    # Mock model loading
    with patch('SafeTopModel.SafeTopModel.load_from_checkpoint') as mock_load:
        # Mock model instance
        mock_model = MagicMock()
        mock_model.hparams.n_assets = 5
        mock_model.hparams.lambda_risk = 1.0
        mock_model.hparams.risk_target_annual = 0.10
        mock_model.hparams.w_max = 0.05
        
        # Mock forward return
        # mu (1, N), cluster_w (1, N, K), experts (1, N, K), jsu_params (4 params)
        N = 5
        mock_model.return_value = (
            torch.randn(N), # mu
            torch.randn(N, 4), # cluster_w
            torch.randn(N, 4), # experts
            (torch.randn(N), torch.ones(N), torch.randn(N), torch.ones(N)) # gamma, delta, xi, lambd
        )
        mock_load.return_value = mock_model
        
        # Mock data loading
        with patch('roll.get_latest_data') as mock_data:
            mock_data.return_value = {
                'micro': torch.randn(1, N, 5, 60),
                'micro_last': torch.randn(1, N, 4),
                'news': torch.randn(1, N, 768),
                'quant': torch.randn(1, N, 20),
                'w_prev': torch.zeros(1, N),
                'sector_map': torch.randn(1, 11, N),
                'metadata': {'tickers': [f'TICK_{i}' for i in range(N)]}
            }
            
            # Run
            out_csv = "tests/test_weights.csv"
            run_roll("dummy.ckpt", "dummy_data", out_csv)
            
            # Assert
            assert os.path.exists(out_csv)
            df = pl.read_csv(out_csv)
            cols = df.columns
            expected = ["ticker", "weight", "mu", "sigma", "cluster_id"]
            for c in expected:
                assert c in cols
            
            # Clean up
            os.remove(out_csv)

import torch
