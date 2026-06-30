from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm.news_analyzer import NewsAnalyzer
from llm.question_answering import StockQA

router = APIRouter()

class NewsAnalysisRequest(BaseModel):
    symbol: str
    days: int = 7

class QuestionRequest(BaseModel):
    question: str
    stock_data: dict = None
    news_analysis: dict = None

class PriceMovementRequest(BaseModel):
    symbol: str
    price_change: float
    news_analysis: dict = None

class ReportRequest(BaseModel):
    symbol: str
    technical_analysis: dict = None
    fundamental_analysis: dict = None
    news_analysis: dict = None

@router.post("/news_analysis")
def analyze_news(request: NewsAnalysisRequest):
    try:
        analyzer = NewsAnalyzer()
        news = analyzer.get_stock_news(request.symbol, days=request.days)
        analysis = analyzer.analyze_news(news, request.symbol)
        
        return {
            "symbol": request.symbol,
            "news_count": len(news),
            "analysis": analysis,
            "news": news[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/question")
def answer_question(request: QuestionRequest):
    try:
        qa = StockQA()
        answer = qa.answer_question(
            question=request.question,
            stock_data=request.stock_data,
            news_analysis=request.news_analysis
        )
        
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/explain_price")
def explain_price_movement(request: PriceMovementRequest):
    try:
        qa = StockQA()
        explanation = qa.explain_price_movement(
            symbol=request.symbol,
            price_change=request.price_change,
            news_analysis=request.news_analysis
        )
        
        return {
            "symbol": request.symbol,
            "price_change": request.price_change,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/report")
def generate_report(request: ReportRequest):
    try:
        qa = StockQA()
        report = qa.generate_report(
            symbol=request.symbol,
            technical_analysis=request.technical_analysis,
            fundamental_analysis=request.fundamental_analysis,
            news_analysis=request.news_analysis
        )
        
        return {
            "symbol": request.symbol,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))