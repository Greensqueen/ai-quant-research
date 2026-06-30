import os
import pandas as pd
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

class StockQA:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if HAS_LANGCHAIN and self.openai_api_key:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=self.openai_api_key)
        else:
            self.llm = None
    
    def answer_question(self, question: str, stock_data: Dict[str, Any] = None, news_analysis: Dict[str, Any] = None) -> str:
        if not HAS_LANGCHAIN or self.llm is None:
            return "LLM模块未安装或配置，请安装 langchain-openai 并配置 OpenAI API 密钥。"
        
        context = ""
        
        if stock_data:
            context += "股票基本信息：\n"
            for key, value in stock_data.items():
                context += f"- {key}: {value}\n"
        
        if news_analysis:
            context += "\n最新新闻分析：\n"
            context += f"市场情绪: {news_analysis.get('sentiment', '未知')}\n"
            context += f"关键因素: {', '.join(news_analysis.get('key_factors', []))}\n"
            context += f"潜在影响: {news_analysis.get('impact', '未知')}\n"
        
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            你是一名专业的AI量化分析师，擅长分析股票市场和回答投资相关问题。

            参考信息：
            {context}

            用户问题：
            {question}

            请根据提供的参考信息（如果有的话），用中文给出专业、准确、易懂的回答。
            如果参考信息不足以回答问题，请基于你的专业知识回答，但要说明信息来源有限。

            回答要求：
            1. 直接回答问题，不要说多余的话
            2. 如果是预测类问题，要明确说明这是预测，不能保证准确性
            3. 如果是分析类问题，要提供具体的分析依据
            4. 避免使用过于专业的术语，让普通投资者也能理解
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(question=question, context=context)
        
        return answer.strip()
    
    def explain_price_movement(self, symbol: str, price_change: float, news_analysis: Dict[str, Any] = None) -> str:
        if not HAS_LANGCHAIN or self.llm is None:
            return "LLM模块未安装或配置，请安装 langchain-openai 并配置 OpenAI API 密钥。"
        
        context = ""
        if news_analysis:
            context = f"""
            最新新闻分析：
            - 市场情绪: {news_analysis.get('sentiment', '未知')}
            - 关键因素: {', '.join(news_analysis.get('key_factors', []))}
            - 潜在影响: {news_analysis.get('impact', '未知')}
            """
        
        prompt = PromptTemplate(
            input_variables=["symbol", "price_change", "context"],
            template="""
            作为专业的金融分析师，请解释股票 {symbol} 价格变动 {price_change}% 的可能原因。

            参考信息：
            {context}

            请从以下几个方面进行分析：
            1. 可能的驱动因素
            2. 市场情绪的影响
            3. 技术面因素（如果适用）
            4. 基本面因素（如果适用）

            请用中文提供专业但易懂的分析，明确说明这是基于现有信息的分析，不构成投资建议。
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        explanation = chain.run(symbol=symbol, price_change=price_change, context=context)
        
        return explanation.strip()
    
    def generate_report(self, symbol: str, technical_analysis: Dict[str, Any] = None, fundamental_analysis: Dict[str, Any] = None, news_analysis: Dict[str, Any] = None) -> str:
        if not HAS_LANGCHAIN or self.llm is None:
            return "LLM模块未安装或配置，请安装 langchain-openai 并配置 OpenAI API 密钥。"
        
        context = ""
        
        if technical_analysis:
            context += "技术分析：\n"
            for key, value in technical_analysis.items():
                context += f"- {key}: {value}\n"
        
        if fundamental_analysis:
            context += "\n基本面分析：\n"
            for key, value in fundamental_analysis.items():
                context += f"- {key}: {value}\n"
        
        if news_analysis:
            context += "\n新闻分析：\n"
            context += f"- 市场情绪: {news_analysis.get('sentiment', '未知')}\n"
            context += f"- 关键因素: {', '.join(news_analysis.get('key_factors', []))}\n"
        
        prompt = PromptTemplate(
            input_variables=["symbol", "context"],
            template="""
            请为股票 {symbol} 生成一份专业的投资分析报告。

            分析数据：
            {context}

            报告结构：
            1. 执行摘要：简要总结分析结果
            2. 技术面分析：分析技术指标和图表形态
            3. 基本面分析：分析公司财务状况和行业地位
            4. 市场情绪：分析近期新闻和市场情绪
            5. 风险提示：列出主要风险因素
            6. 投资建议：给出明确的投资建议（买入/持有/卖出）及理由

            请用中文撰写，语言专业但易懂，适合专业投资者阅读。
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        report = chain.run(symbol=symbol, context=context)
        
        return report.strip()