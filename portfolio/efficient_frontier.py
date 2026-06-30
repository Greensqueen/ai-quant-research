import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class EfficientFrontier:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.cov_matrix = returns.cov()
        self.mean_returns = returns.mean()
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        return np.dot(self.mean_returns, weights) * 252
    
    def portfolio_std(self, weights: np.ndarray) -> float:
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
    
    def neg_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        ret = self.portfolio_return(weights)
        std = self.portfolio_std(weights)
        return -(ret - risk_free_rate) / std
    
    def minimize_volatility(self, target_return: float) -> dict:
        constraints = [
            {'type': 'eq', 'fun': lambda w: self.portfolio_return(w) - target_return},
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self.portfolio_std,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': target_return,
            'std': result.fun,
            'success': result.success
        }
    
    def maximize_sharpe_ratio(self, risk_free_rate: float = 0.02) -> dict:
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self.neg_sharpe_ratio,
            initial_weights,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': self.portfolio_return(result.x),
            'std': self.portfolio_std(result.x),
            'sharpe_ratio': (self.portfolio_return(result.x) - risk_free_rate) / self.portfolio_std(result.x),
            'success': result.success
        }
    
    def minimize_variance(self) -> dict:
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self.portfolio_std,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': self.portfolio_return(result.x),
            'std': result.fun,
            'success': result.success
        }
    
    def generate_efficient_frontier(self, n_points: int = 50, risk_free_rate: float = 0.02) -> pd.DataFrame:
        min_return = self.portfolio_return(np.eye(self.n_assets)).min()
        max_return = self.portfolio_return(np.eye(self.n_assets)).max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        for target_return in target_returns:
            try:
                opt_result = self.minimize_volatility(target_return)
                if opt_result['success']:
                    frontier.append({
                        'return': opt_result['return'],
                        'std': opt_result['std'],
                        'sharpe_ratio': (opt_result['return'] - risk_free_rate) / opt_result['std'],
                        'weights': opt_result['weights']
                    })
            except:
                continue
        
        return pd.DataFrame(frontier)
    
    def get_optimal_portfolios(self, risk_free_rate: float = 0.02) -> dict:
        return {
            'minimum_variance': self.minimize_variance(),
            'maximum_sharpe': self.maximize_sharpe_ratio(risk_free_rate)
        }
    
    def plot_efficient_frontier(self, risk_free_rate: float = 0.02) -> plt.Figure:
        frontier_df = self.generate_efficient_frontier()
        max_sharpe = self.maximize_sharpe_ratio(risk_free_rate)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(frontier_df['std'], frontier_df['return'], c=frontier_df['sharpe_ratio'], cmap='viridis', label='Efficient Frontier')
        ax.scatter(max_sharpe['std'], max_sharpe['return'], color='red', marker='*', s=200, label='Max Sharpe Ratio')
        
        x_vals = np.linspace(0, max(frontier_df['std']) * 1.1, 100)
        y_vals = risk_free_rate + (max_sharpe['sharpe_ratio'] * x_vals)
        ax.plot(x_vals, y_vals, 'r--', label='Capital Market Line')
        
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True)
        
        return fig