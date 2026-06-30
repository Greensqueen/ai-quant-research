from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from portfolio.risk_metrics import RiskMetrics
from portfolio.efficient_frontier import EfficientFrontier

router = APIRouter()

class PortfolioRequest(BaseModel):
    returns_data: list
    weights: list
    symbols: list

class EfficientFrontierRequest(BaseModel):
    returns_data: list
    symbols: list
    n_points: int = 50

@router.post("/metrics")
def get_portfolio_metrics(request: PortfolioRequest):
    try:
        returns = pd.DataFrame(request.returns_data)
        returns.columns = request.symbols
        weights = np.array(request.weights)
        
        rm = RiskMetrics()
        metrics = rm.get_portfolio_metrics(returns, weights)
        
        return {
            "symbols": request.symbols,
            "weights": weights.tolist(),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/efficient_frontier")
def calculate_efficient_frontier(request: EfficientFrontierRequest):
    try:
        returns = pd.DataFrame(request.returns_data)
        returns.columns = request.symbols
        
        ef = EfficientFrontier(returns)
        frontier = ef.generate_efficient_frontier(n_points=request.n_points)
        optimal = ef.get_optimal_portfolios()
        
        frontier_data = []
        for _, row in frontier.iterrows():
            frontier_data.append({
                "return": row["return"],
                "std": row["std"],
                "sharpe_ratio": row["sharpe_ratio"],
                "weights": row["weights"].tolist()
            })
        
        return {
            "symbols": request.symbols,
            "efficient_frontier": frontier_data,
            "minimum_variance": {
                "return": optimal["minimum_variance"]["return"],
                "std": optimal["minimum_variance"]["std"],
                "weights": optimal["minimum_variance"]["weights"].tolist()
            },
            "maximum_sharpe": {
                "return": optimal["maximum_sharpe"]["return"],
                "std": optimal["maximum_sharpe"]["std"],
                "sharpe_ratio": optimal["maximum_sharpe"]["sharpe_ratio"],
                "weights": optimal["maximum_sharpe"]["weights"].tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize")
def optimize_portfolio(request: EfficientFrontierRequest):
    try:
        returns = pd.DataFrame(request.returns_data)
        returns.columns = request.symbols
        
        ef = EfficientFrontier(returns)
        optimal = ef.get_optimal_portfolios()
        
        return {
            "symbols": request.symbols,
            "minimum_variance": {
                "return": optimal["minimum_variance"]["return"],
                "std": optimal["minimum_variance"]["std"],
                "weights": dict(zip(request.symbols, optimal["minimum_variance"]["weights"].tolist()))
            },
            "maximum_sharpe": {
                "return": optimal["maximum_sharpe"]["return"],
                "std": optimal["maximum_sharpe"]["std"],
                "sharpe_ratio": optimal["maximum_sharpe"]["sharpe_ratio"],
                "weights": dict(zip(request.symbols, optimal["maximum_sharpe"]["weights"].tolist()))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))