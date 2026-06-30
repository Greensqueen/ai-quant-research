import pandas as pd
import numpy as np
from scipy import stats

class RiskMetrics:
    def __init__(self):
        pass
    
    def calculate_portfolio_return(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        return np.dot(returns.mean(), weights) * 252
    
    def calculate_portfolio_std(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        cov_matrix = returns.cov()
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_var) * np.sqrt(252)
    
    def calculate_sharpe_ratio(self, returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        portfolio_return = self.calculate_portfolio_return(returns, weights)
        portfolio_std = self.calculate_portfolio_std(returns, weights)
        return (portfolio_return - risk_free_rate) / portfolio_std
    
    def calculate_sortino_ratio(self, returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        portfolio_returns = returns @ weights
        excess_returns = portfolio_returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_std.std() * np.sqrt(252)
        return (excess_returns.mean() * 252) / downside_std
    
    def calculate_var(self, returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95, method: str = 'historical') -> float:
        portfolio_returns = returns @ weights
        
        if method == 'historical':
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            var = mean + std * stats.norm.ppf(1 - confidence_level)
        elif method == 'monte_carlo':
            cov_matrix = returns.cov()
            simulations = np.random.multivariate_normal(returns.mean(), cov_matrix, 10000) @ weights
            var = np.percentile(simulations, (1 - confidence_level) * 100)
        else:
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        return abs(var) * np.sqrt(252)
    
    def calculate_cvar(self, returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return abs(cvar) * np.sqrt(252)
    
    def calculate_max_drawdown(self, prices: pd.DataFrame, weights: np.ndarray) -> float:
        portfolio_values = (prices * weights).sum(axis=1)
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_beta(self, returns: pd.DataFrame, weights: np.ndarray, market_returns: pd.Series) -> float:
        portfolio_returns = returns @ weights
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        return covariance / market_variance
    
    def calculate_omega_ratio(self, returns: pd.DataFrame, weights: np.ndarray, threshold: float = 0.0) -> float:
        portfolio_returns = returns @ weights
        excess_returns = portfolio_returns - threshold / 252
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def calculate_information_ratio(self, returns: pd.DataFrame, weights: np.ndarray, benchmark_returns: pd.Series) -> float:
        portfolio_returns = returns @ weights
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        return active_returns.mean() * 252 / tracking_error
    
    def get_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray, market_returns: pd.Series = None, risk_free_rate: float = 0.02) -> dict:
        metrics = {
            'return': self.calculate_portfolio_return(returns, weights),
            'std': self.calculate_portfolio_std(returns, weights),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, weights, risk_free_rate),
            'var_95': self.calculate_var(returns, weights, 0.95),
            'cvar_95': self.calculate_cvar(returns, weights, 0.95),
            'max_drawdown': self.calculate_max_drawdown(returns.cumsum().add(1), weights)
        }
        
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, weights, market_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, weights, market_returns)
        
        return metrics