import os
import tushare as ts
import pandas as pd
from datetime import datetime

class TuShareData:
    def __init__(self, token: str = None):
        self.data = None
        self.token = token or os.getenv('TUSHARE_TOKEN')
        if self.token:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
    
    def _check_token(self):
        if self.pro is None:
            raise ValueError("TuShare token not configured. Please set TUSHARE_TOKEN environment variable.")
    
    def get_stock_basic(self) -> pd.DataFrame:
        try:
            self._check_token()
            return self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get stock basic info: {str(e)}")
    
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            self._check_token()
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df['trade_date'] = df['trade_date'].astype(str)
            return df
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get daily data: {str(e)}")
    
    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            self._check_token()
            df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df['trade_date'] = df['trade_date'].astype(str)
            return df
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get index daily data: {str(e)}")
    
    def get_financial_report(self, ts_code: str) -> pd.DataFrame:
        try:
            self._check_token()
            return self.pro.fina_mainbz(ts_code=ts_code, start_date='', end_date='')
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get financial report: {str(e)}")
    
    def get_dividend(self, ts_code: str) -> pd.DataFrame:
        try:
            self._check_token()
            return self.pro.dividend(ts_code=ts_code)
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get dividend data: {str(e)}")
    
    def get_index_weight(self, index_code: str) -> pd.DataFrame:
        try:
            self._check_token()
            return self.pro.index_weight(index_code=index_code)
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get index weight: {str(e)}")
    
    def get_concept_stocks(self) -> pd.DataFrame:
        try:
            self._check_token()
            return self.pro.concept()
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to get concept stocks: {str(e)}")