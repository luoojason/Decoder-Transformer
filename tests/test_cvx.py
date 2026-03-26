
import torch
import cvxpy as cp
import numpy as np
import pytest
from SafeTopModel import build_cvxpy_layer

def test_cvx_layer_isolation():
    """
    Test the CvxpyLayer in isolation with known, well-behaved inputs.
    """
    torch.manual_seed(42)
    
    # 1. Build Layer
    N = 20
    # S should match what build_cvxpy_layer uses internally (currently 1000 in code, let's check)
    # We will import the function and see.
    # Actually, build_cvxpy_layer hardcodes S=1000 inside.
    
    layer, S = build_cvxpy_layer(n_assets=N, w_max=0.10)
    print(f"Layer built with S={S}")
    
    # 2. Create Inputs
    # mu: (N) - expects float64 usually for diffcp
    mu = torch.randn(N, dtype=torch.float64) * 0.01
    w_prev = torch.zeros(N, dtype=torch.float64)
    w_prev[0] = 1.0 # Fully invested in asset 0
    
    # R_sim: (S, N)
    # create reasonable returns (mean 0, std 0.01)
    R_sim = torch.randn(S, N, dtype=torch.float64) * 0.01
    
    # Params
    lambda_risk = torch.tensor([1.0], dtype=torch.float64)
    risk_target = torch.tensor([0.05], dtype=torch.float64) # 5% daily? No, annual/sqrt(252).
    # Say 0.10/16 ~ 0.006
    risk_target = torch.tensor([0.01], dtype=torch.float64)
    
    # Sector Map
    # Needs to be feasible. sum(w * sector) = 0.
    # If all sectors are 0, constraint is 0=0.
    # Let's make a simple map. 2 sectors. First 10 -> sec 0 (1), Next 10 -> sec 1 (0). 
    # Actually sector_map is (11, N).
    # Let's just use zeros for now to ensure feasibility of "neutrality" for trivial case
    # OR better: Sector 0 has assets [0,1], Sector 1 has [2,3].
    # But w_prev has w[0]=1. If sector 0 must be neutral, w[0] + w[1] = 0? 
    # Usually sector constraint is: w_sector_i - w_sector_j = 0? Or Neutral per sector?
    # Code says: "Sector_Map @ w == 0" with shapes (n_sectors, N).
    # This implies sum_{j in sector_i} w_j = 0.
    # This means Long/Short neutrality PER SECTOR.
    # If w_prev is Long-only (w[0]=1), then w_prev is INFEASIBLE wrt sector neutrality (unless sector 0 has matching short).
    # Turnover constraint |w - w_prev| <= limit might conflict if w_prev is far from feasible.
    # So we should make w_prev feasible or relax constraints.
    # Let's make w_prev zeros to be safe.
    w_prev = torch.zeros(N, dtype=torch.float64)
    
    sector_map = torch.zeros((11, N), dtype=torch.float64)
    # Assign asset 0 and 1 to sector 0.
    sector_map[0, 0] = 1.0
    sector_map[0, 1] = 1.0 
    # This requires w[0] + w[1] = 0.
    
    # 3. Solver Args
    # Try OSQP (QP/LP solver)
    solver_args = {'warm_starts': [None], 'Ps': ['OSQP'], 'max_iters': 500}
    
    try:
        w_opt, = layer(
            mu, 
            w_prev, 
            R_sim, 
            lambda_risk, 
            risk_target, 
            sector_map, 
            solver_args=solver_args
        )
        print("Solver success!")
        print("w_opt:", w_opt)
        
        # Check constraints
        # 1. Sector neutrality
        sec_exposure = sector_map @ w_opt
        assert torch.allclose(sec_exposure, torch.zeros_like(sec_exposure), atol=1e-3)
        
        # 2. Sum abs w <= k_gross (2.0)
        gloss_exp = torch.sum(torch.abs(w_opt))
        assert gloss_exp <= 2.0 + 1e-3
        
    except Exception as e:
        pytest.fail(f"Solver failed: {e}")

if __name__ == "__main__":
    test_cvx_layer_isolation()
