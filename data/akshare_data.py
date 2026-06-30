import akshare as ak
import pandas as pd
from datetime import datetime

class AKShareData:
    def __init__(self):
        self.data = None
    
    def get_stock_list(self) -> pd.DataFrame:
        try:
            return ak.stock_zh_a_spot()
        except Exception as e:
            raise ValueError(f"Failed to get stock list: {str(e)}")
    
    def get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            df.reset_index(inplace=True)
            df['日期'] = df['日期'].astype(str)
            return df
        except Exception as e:
            raise ValueError(f"Failed to get daily data: {str(e)}")
    
    def get_index_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = ak.stock_zh_index_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            df.reset_index(inplace=True)
            df['日期'] = df['日期'].astype(str)
            return df
        except Exception as e:
            raise ValueError(f"Failed to get index data: {str(e)}")
    
    def get_fundamental_data(self, symbol: str) -> pd.DataFrame:
        try:
            return ak.stock_financial_report_sina(symbol=symbol)
        except Exception as e:
            raise ValueError(f"Failed to get fundamental data: {str(e)}")
    
    def get_index_list(self) -> pd.DataFrame:
        try:
            return ak.stock_zh_index_spot()
        except Exception as e:
            raise ValueError(f"Failed to get index list: {str(e)}")
    
    def get_hk_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = ak.stock_hk_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            df.reset_index(inplace=True)
            df['日期'] = df['日期'].astype(str)
            return df
        except Exception as e:
            raise ValueError(f"Failed to get HK stock data: {str(e)}")
    
    def get_us_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = ak.stock_us_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            df.reset_index(inplace=True)
            df['日期'] = df['日期'].astype(str)
            return df
        except Exception as e:
            raise ValueError(f"Failed to get US stock data: {str(e)}")