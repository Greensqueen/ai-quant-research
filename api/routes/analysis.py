from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from analysis.technical_indicators import TechnicalIndicators
from analysis.statistical_analysis import StatisticalAnalysis

router = APIRouter()

class AnalysisRequest(BaseModel):
    data: list
    indicators: list = ["sma", "rsi", "macd", "bollinger"]

class ReturnsRequest(BaseModel):
    data: list
    column: str = "Close"
    method: str = "log"

class BetaRequest(BaseModel):
    stock_data: list
    market_data: list
    stock_column: str = "Returns"
    market_column: str = "Returns"

@router.post("/technical")
def calculate_technical_indicators(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.data)
        ti = TechnicalIndicators()
        
        if "sma" in request.indicators:
            df = ti.add_sma(df, window=20)
            df = ti.add_sma(df, window=50)
        if "ema" in request.indicators:
            df = ti.add_ema(df, window=12)
            df = ti.add_ema(df, window=26)
        if "rsi" in request.indicators:
            df = ti.add_rsi(df)
        if "macd" in request.indicators:
            df = ti.add_macd(df)
        if "bollinger" in request.indicators:
            df = ti.add_bollinger_bands(df)
        if "atr" in request.indicators:
            df = ti.add_atr(df)
        if "obv" in request.indicators:
            df = ti.add_obv(df)
        if "momentum" in request.indicators:
            df = ti.add_momentum(df)
        
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/returns")
def calculate_returns(request: ReturnsRequest):
    try:
        df = pd.DataFrame(request.data)
        sa = StatisticalAnalysis()
        df = sa.calculate_returns(df, column=request.column, method=request.method)
        df = sa.calculate_cumulative_returns(df)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/stats")
def get_statistics(request: ReturnsRequest):
    try:
        df = pd.DataFrame(request.data)
        sa = StatisticalAnalysis()
        df = sa.calculate_returns(df, column=request.column, method=request.method)
        stats = sa.get_descriptive_stats(df)
        stats['max_drawdown'] = sa.calculate_max_drawdown(df, column=request.column)
        stats['win_rate'] = sa.calculate_win_rate(df)
        stats['profit_factor'] = sa.calculate_profit_factor(df)
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/beta")
def calculate_beta(request: BetaRequest):
    try:
        stock_df = pd.DataFrame(request.stock_data)
        market_df = pd.DataFrame(request.market_data)
        sa = StatisticalAnalysis()
        stock_df = sa.calculate_returns(stock_df, column=request.stock_column)
        market_df = sa.calculate_returns(market_df, column=request.market_column)
        beta = sa.calculate_beta(stock_df, market_df)
        alpha = sa.calculate_alpha(stock_df, market_df)
        return {"beta": beta, "alpha": alpha}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/var")
def calculate_var(request: ReturnsRequest):
    try:
        df = pd.DataFrame(request.data)
        sa = StatisticalAnalysis()
        df = sa.calculate_returns(df, column=request.column, method=request.method)
        var_historical = sa.calculate_var(df, method="historical")
        var_parametric = sa.calculate_var(df, method="parametric")
        var_monte_carlo = sa.calculate_var(df, method="monte_carlo")
        return {
            "historical_var": var_historical,
            "parametric_var": var_parametric,
            "monte_carlo_var": var_monte_carlo
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))