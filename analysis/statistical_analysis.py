import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

class StatisticalAnalysis:
    def __init__(self):
        pass
    
    def calculate_returns(self, df: pd.DataFrame, column: str = 'Close', method: str = 'log') -> pd.DataFrame:
        if method == 'log':
            df['Returns'] = np.log(df[column] / df[column].shift(1))
        else:
            df['Returns'] = df[column].pct_change()
        return df
    
    def calculate_cumulative_returns(self, df: pd.DataFrame, column: str = 'Returns') -> pd.DataFrame:
        df['Cumulative_Returns'] = (1 + df[column]).cumprod() - 1
        return df
    
    def get_descriptive_stats(self, df: pd.DataFrame, column: str = 'Returns') -> dict:
        returns = df[column].dropna()
        return {
            'mean': returns.mean(),
            'median': returns.median(),
            'std': returns.std(),
            'var': returns.var(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'count': returns.count(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252)
        }
    
    def calculate_beta(self, df_stock: pd.DataFrame, df_market: pd.DataFrame, stock_col: str = 'Returns', market_col: str = 'Returns') -> float:
        stock_returns = df_stock[stock_col].dropna()
        market_returns = df_market[market_col].dropna()
        
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        X = sm.add_constant(market_returns)
        model = sm.OLS(stock_returns, X).fit()
        return model.params[1]
    
    def calculate_alpha(self, df_stock: pd.DataFrame, df_market: pd.DataFrame, risk_free_rate: float = 0.02, stock_col: str = 'Returns', market_col: str = 'Returns') -> float:
        stock_returns = df_stock[stock_col].dropna()
        market_returns = df_market[market_col].dropna()
        
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        excess_stock_returns = stock_returns - risk_free_rate / 252
        excess_market_returns = market_returns - risk_free_rate / 252
        
        X = sm.add_constant(excess_market_returns)
        model = sm.OLS(excess_stock_returns, X).fit()
        return model.params[0] * 252
    
    def perform_normality_test(self, df: pd.DataFrame, column: str = 'Returns') -> dict:
        returns = df[column].dropna()
        stat, p_value = stats.normaltest(returns)
        return {
            'test_statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def calculate_var(self, df: pd.DataFrame, column: str = 'Returns', confidence_level: float = 0.95, method: str = 'historical') -> float:
        returns = df[column].dropna()
        
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            var = mean + std * stats.norm.ppf(1 - confidence_level)
        elif method == 'monte_carlo':
            simulations = np.random.normal(returns.mean(), returns.std(), 10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
        else:
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return abs(var)
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame, column: str = 'Returns', risk_free_rate: float = 0.02) -> float:
        returns = df[column].dropna()
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, df: pd.DataFrame, column: str = 'Returns', risk_free_rate: float = 0.02) -> float:
        returns = df[column].dropna()
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return excess_returns.mean() / downside_std * np.sqrt(252)
    
    def calculate_max_drawdown(self, df: pd.DataFrame, column: str = 'Close') -> float:
        rolling_max = df[column].cummax()
        drawdown = (df[column] - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_win_rate(self, df: pd.DataFrame, column: str = 'Returns') -> float:
        returns = df[column].dropna()
        winning_days = returns[returns > 0].count()
        total_days = returns.count()
        return winning_days / total_days if total_days > 0 else 0
    
    def calculate_profit_factor(self, df: pd.DataFrame, column: str = 'Returns') -> float:
        returns = df[column].dropna()
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else float('inf')