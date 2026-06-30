from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from data.yahoo_finance import YahooFinance
from data.akshare_data import AKShareData

router = APIRouter()

class StockDataRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str

class MultipleStocksRequest(BaseModel):
    symbols: list
    start_date: str
    end_date: str

@router.get("/yahoo/{symbol}")
def get_yahoo_stock_data(symbol: str, start_date: str, end_date: str):
    try:
        yf = YahooFinance()
        data = yf.get_stock_data(symbol, start_date, end_date)
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/yahoo/info/{symbol}")
def get_yahoo_stock_info(symbol: str):
    try:
        yf = YahooFinance()
        info = yf.get_stock_info(symbol)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/yahoo/multiple")
def get_multiple_stocks(request: MultipleStocksRequest):
    try:
        yf = YahooFinance()
        data = yf.get_multiple_stocks(request.symbols, request.start_date, request.end_date)
        result = {}
        for symbol, df in data.items():
            if df is not None:
                result[symbol] = df.to_dict(orient="records")
            else:
                result[symbol] = None
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/akshare/stock_list")
def get_akshare_stock_list():
    try:
        ak = AKShareData()
        data = ak.get_stock_list()
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/akshare/daily/{symbol}")
def get_akshare_daily_data(symbol: str, start_date: str, end_date: str):
    try:
        ak = AKShareData()
        data = ak.get_daily_data(symbol, start_date, end_date)
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/akshare/index_list")
def get_akshare_index_list():
    try:
        ak = AKShareData()
        data = ak.get_index_list()
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/akshare/index/{symbol}")
def get_akshare_index_data(symbol: str, start_date: str, end_date: str):
    try:
        ak = AKShareData()
        data = ak.get_index_data(symbol, start_date, end_date)
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))