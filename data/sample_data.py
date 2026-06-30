import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_stock_data(symbol: str = "AAPL", days: int = 365) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))
    prices = 150 * (1 + returns).cumprod()
    
    high = prices * (1 + np.random.uniform(0, 0.02, len(dates)))
    low = prices * (1 - np.random.uniform(0, 0.02, len(dates)))
    open_price = low + np.random.uniform(0, high - low, len(dates))
    volume = np.random.randint(1000000, 10000000, len(dates))
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': prices,
        'Adj Close': prices * 0.98,
        'Volume': volume
    })
    
    return df

def generate_multiple_stocks(symbols: list, days: int = 365) -> dict:
    result = {}
    for symbol in symbols:
        result[symbol] = generate_sample_stock_data(symbol, days)
    return result

def get_sample_stock_info(symbol: str) -> dict:
    info = {
        'AAPL': {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 2700000000000,
            'pe_ratio': 28.5,
            'dividend_yield': 0.006,
            'beta': 1.28,
            '52_week_high': 196.74,
            '52_week_low': 164.08
        },
        'MSFT': {
            'symbol': 'MSFT',
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': 2500000000000,
            'pe_ratio': 32.4,
            'dividend_yield': 0.009,
            'beta': 0.95,
            '52_week_high': 378.88,
            '52_week_low': 305.75
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'industry': 'Internet Content & Information',
            'market_cap': 1800000000000,
            'pe_ratio': 21.8,
            'dividend_yield': 0,
            'beta': 1.05,
            '52_week_high': 141.96,
            '52_week_low': 101.88
        },
        'META': {
            'symbol': 'META',
            'name': 'Meta Platforms Inc.',
            'sector': 'Communication Services',
            'industry': 'Interactive Media',
            'market_cap': 800000000000,
            'pe_ratio': 15.2,
            'dividend_yield': 0.012,
            'beta': 1.35,
            '52_week_high': 505.66,
            '52_week_low': 350.66
        }
    }
    return info.get(symbol, info['AAPL'])