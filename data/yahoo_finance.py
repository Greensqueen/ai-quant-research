import yfinance as yf
import pandas as pd
import time
from datetime import datetime

class YahooFinance:
    def __init__(self):
        self.data = None
        self.retry_delay = 5
        self.max_retries = 3
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        for attempt in range(self.max_retries):
            try:
                self.data = yf.download(symbol, start=start_date, end=end_date)
                if self.data.empty:
                    raise ValueError("No data returned from Yahoo Finance")
                self.data.reset_index(inplace=True)
                self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
                return self.data
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to fetch data from Yahoo Finance after {self.max_retries} attempts: {str(e)}")
    
    def get_multiple_stocks(self, symbols: list, start_date: str, end_date: str) -> dict:
        result = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date)
                result[symbol] = data
            except Exception as e:
                print(f"Failed to fetch data for {symbol}: {str(e)}")
                result[symbol] = None
            time.sleep(1)
        return result
    
    def get_stock_info(self, symbol: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if not info:
                    raise ValueError("No info returned")
                return {
                    'symbol': symbol,
                    'name': info.get('longName', '') or info.get('shortName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0)
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to get stock info after {self.max_retries} attempts: {str(e)}")
    
    def get_dividends(self, symbol: str) -> pd.DataFrame:
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                dividends = ticker.dividends
                if not dividends.empty:
                    dividends = dividends.reset_index()
                    dividends['Date'] = dividends['Date'].dt.strftime('%Y-%m-%d')
                return dividends
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to get dividends after {self.max_retries} attempts: {str(e)}")
    
    def get_splits(self, symbol: str) -> pd.DataFrame:
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                splits = ticker.splits
                if not splits.empty:
                    splits = splits.reset_index()
                    splits['Date'] = splits['Date'].dt.strftime('%Y-%m-%d')
                return splits
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to get splits after {self.max_retries} attempts: {str(e)}")