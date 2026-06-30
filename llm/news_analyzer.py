import os
import pandas as pd
import requests
from datetime import datetime, timedelta

try:
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

class NewsAnalyzer:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if HAS_LANGCHAIN and self.openai_api_key:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=self.openai_api_key)
        else:
            self.llm = None
    
    def get_stock_news(self, symbol: str, days: int = 7) -> list:
        try:
            news_api_key = os.getenv('NEWS_API_KEY')
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            news = []
            
            if news_api_key:
                base_url = "https://newsapi.org/v2/everything"
                
                params = {
                    'q': symbol,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'apiKey': news_api_key
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        news.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'published_at': article.get('publishedAt', '')
                        })
            else:
                news.append({
                    'title': f"关于 {symbol} 的市场动态",
                    'description': "由于未配置新闻API密钥，无法获取实时新闻数据。请在.env文件中配置NEWS_API_KEY。",
                    'source': '系统提示',
                    'published_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                })
            
            return news[:10]
        except Exception as e:
            print(f"Failed to fetch news: {str(e)}")
            return []
    
    def analyze_news(self, news_list: list, symbol: str) -> dict:
        if not news_list:
            return {
                'summary': "暂无新闻数据",
                'sentiment': 'neutral',
                'key_factors': [],
                'impact': 'neutral'
            }
        
        if not HAS_LANGCHAIN or self.llm is None:
            return {
                'summary': "LLM模块未安装或配置",
                'sentiment': 'neutral',
                'key_factors': [],
                'impact': 'neutral'
            }
        
        news_text = "\n".join([f"{n['title']}: {n['description']}" for n in news_list])
        
        prompt = PromptTemplate(
            input_variables=["news", "symbol"],
            template="""
            作为一名专业的金融分析师，请分析以下关于股票 {symbol} 的新闻，并提供详细的分析报告：

            新闻内容：
            {news}

            请从以下几个方面进行分析：
            1. 新闻摘要：简要总结所有新闻的核心内容
            2. 市场情绪：判断整体市场情绪是正面、负面还是中性
            3. 关键影响因素：列出影响该股票的主要因素
            4. 潜在影响：分析这些新闻可能对股价产生的影响

            请用中文提供专业但易懂的分析报告。
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(news=news_text, symbol=symbol)
        
        return self._parse_analysis(result)
    
    def _parse_analysis(self, result: str) -> dict:
        lines = result.strip().split('\n')
        analysis = {
            'summary': "",
            'sentiment': 'neutral',
            'key_factors': [],
            'impact': 'neutral'
        }
        
        current_section = None
        for line in lines:
            if '新闻摘要' in line or '摘要' in line:
                current_section = 'summary'
            elif '市场情绪' in line or '情绪' in line:
                current_section = 'sentiment'
            elif '关键影响因素' in line or '关键因素' in line:
                current_section = 'key_factors'
            elif '潜在影响' in line or '影响' in line:
                current_section = 'impact'
            elif current_section == 'summary':
                analysis['summary'] += line.strip() + " "
            elif current_section == 'sentiment':
                if '正面' in line:
                    analysis['sentiment'] = 'positive'
                elif '负面' in line:
                    analysis['sentiment'] = 'negative'
            elif current_section == 'key_factors':
                if line.strip() and line.strip()[0].isdigit():
                    analysis['key_factors'].append(line.strip())
            elif current_section == 'impact':
                analysis['impact'] = line.strip()
        
        return analysis