from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import data, analysis, ml, portfolio, llm

app = FastAPI(
    title="AI Quant Research Platform API",
    description="专业的AI量化研究平台API，提供股票数据获取、数据分析、机器学习预测、投资组合分析和LLM解释等功能",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(ml.router, prefix="/api/ml", tags=["Machine Learning"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])

@app.get("/")
def root():
    return {"message": "AI Quant Research Platform API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}