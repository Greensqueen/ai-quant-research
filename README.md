# AI Quant Research Platform

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

一个专业的AI量化研究平台，整合股票数据获取、数据分析、机器学习预测、LLM解释和投资组合分析等功能。

---

## ✨ 功能特点

### 1. 股票数据获取
- **Yahoo Finance** - 全球股票数据
- **AKShare** - 中国A股数据（推荐使用）
- **TuShare** - 沪深股票市场数据

### 2. 数据分析
- Pandas数据处理
- NumPy数值计算
- 技术指标计算（MA、RSI、MACD等）

### 3. 机器学习预测
- Linear Regression
- Random Forest
- XGBoost
- LightGBM

### 4. LLM解释
- 自然语言问答（如"为什么今天腾讯跌了？"）
- 新闻分析
- 智能报告生成

### 5. 投资组合分析
- 风险评估
- 收益分析
- Sharpe Ratio
- VaR（风险价值）
- Efficient Frontier（有效前沿）

### 6. Dashboard
- Streamlit可视化界面
- FastAPI后端服务

---

## 🛠 技术栈

| 分类 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 后端 | FastAPI 0.104+ |
| 可视化 | Streamlit 1.30+, Plotly |
| 数据分析 | Pandas, NumPy, SciPy |
| 机器学习 | scikit-learn, XGBoost, LightGBM |
| LLM | LangChain, OpenAI API |
| 部署 | Docker, Docker Compose |

---

## 📁 项目结构

```
ai-quant-research/
├── api/                    # FastAPI后端服务
│   ├── main.py             # API入口
│   └── routes/             # 各模块路由
│       ├── data.py         # 数据获取API
│       ├── analysis.py     # 数据分析API
│       ├── ml.py           # 机器学习API
│       ├── portfolio.py    # 投资组合API
│       └── llm.py          # LLM解释API
├── data/                   # 数据模块
│   ├── yahoo_finance.py    # Yahoo Finance数据
│   ├── akshare_data.py     # AKShare数据（推荐）
│   ├── tushare_data.py     # TuShare数据
│   └── sample_data.py      # 示例数据（无网络时自动使用）
├── analysis/               # 数据分析模块
│   ├── technical_indicators.py  # 技术指标计算
│   └── statistical_analysis.py  # 统计分析
├── ml/                     # 机器学习模块
│   ├── base_model.py       # 基础模型抽象类
│   ├── linear_regression.py    # 线性回归
│   ├── random_forest.py        # 随机森林
│   ├── xgboost_model.py        # XGBoost
│   └── lightgbm_model.py       # LightGBM
├── portfolio/              # 投资组合分析
│   ├── risk_metrics.py         # 风险指标
│   └── efficient_frontier.py   # 有效前沿
├── llm/                    # LLM解释模块
│   ├── news_analyzer.py    # 新闻分析器
│   └── question_answering.py   # 问答系统
├── dashboard/              # Streamlit Dashboard
│   └── main.py             # Dashboard入口
├── scripts/                # 启动脚本
├── .env.example            # 环境变量示例
├── .env                    # 环境变量配置（需要创建）
├── Dockerfile              # Docker配置
├── docker-compose.yml      # Docker Compose配置
├── requirements.txt        # Python依赖
└── README.md               # 项目说明
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/ai-quant-research.git
cd ai-quant-research
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（可选）

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件（详细说明见下方）：

```env
# OpenAI API Key (用于LLM功能，可选)
OPENAI_API_KEY=your_openai_api_key_here

# NewsAPI Key (用于新闻分析，可选)
NEWS_API_KEY=your_news_api_key_here

# TuShare Token (用于中国A股数据，可选)
TUSHARE_TOKEN=your_tushare_token_here
```

### 4. 启动服务

#### 方式一：使用命令行

**启动Streamlit Dashboard（推荐）**
```bash
streamlit run dashboard/main.py
```

**启动FastAPI后端**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 方式二：使用Docker

```bash
docker-compose up
```

### 5. 访问服务

| 服务 | 地址 |
|------|------|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI文档 | http://localhost:8000/docs |

---

## ⚠️ 关于数据获取的重要说明

### Yahoo Finance 数据获取问题

**问题描述**：
Yahoo Finance 近期对API进行了限制，可能会遇到以下错误：
- `YFRateLimitError` - 速率限制
- `DNSError` - DNS解析失败
- 数据返回为空

**解决方案**：
1. **使用AKShare（推荐）**：AKShare是国内数据源，更稳定，无需API密钥
2. **使用示例数据**：项目内置了示例数据生成器，当无法获取真实数据时会自动使用
3. **等待重试**：Yahoo Finance限制通常是临时的，项目已实现自动重试机制

### 推荐数据源优先级

| 优先级 | 数据源 | 特点 |
|--------|--------|------|
| 1 | AKShare | 稳定、国内数据、无需密钥 |
| 2 | TuShare | 国内数据、需要Token |
| 3 | Yahoo Finance | 全球数据、有时限限制 |

### 测试数据获取

首次打开Dashboard时，如果网络环境限制导致无法获取真实数据，系统会自动使用示例数据。你可以：
1. 查看警告提示（橙色消息）
2. 正常使用所有分析功能
3. 数据为模拟生成，但分析逻辑完全真实

---

## 🔑 API密钥获取指南

### 1. OpenAI API Key（可选）

**用途**: LLM问答、新闻分析、智能报告生成

**获取步骤**:
1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 注册/登录账号
3. 点击右上角头像 -> `View API keys`
4. 点击 `Create new secret key`
5. 复制生成的API密钥

**费用**: 按使用量计费，新用户有免费额度（约$5）

### 2. NewsAPI Key（可选）

**用途**: 获取股票相关新闻

**获取步骤**:
1. 访问 [NewsAPI](https://newsapi.org/)
2. 注册免费账号
3. 在 `API Keys` 页面获取密钥

**费用**: 免费版每月100次请求，付费版提供更多请求

### 3. TuShare Token（可选）

**用途**: 获取中国A股数据

**获取步骤**:
1. 访问 [TuShare](https://tushare.pro/)
2. 注册账号并完成实名认证
3. 在个人中心获取Token

**费用**: 基础数据免费，高级数据需要积分

---

## 📊 使用示例

### Python API

```python
# 获取股票数据（自动使用示例数据或真实数据）
from data.yahoo_finance import YahooFinance
yf = YahooFinance()
data = yf.get_stock_data("AAPL", "2023-01-01", "2024-01-01")

# 使用AKShare获取A股数据（推荐）
from data.akshare_data import AKShareData
ak = AKShareData()
data = ak.get_daily_data("000001", "20230101", "20240101")

# 计算技术指标
from analysis.technical_indicators import TechnicalIndicators
ti = TechnicalIndicators()
data = ti.add_sma(data, window=20)
data = ti.add_rsi(data)

# 机器学习预测
from ml.xgboost_model import XGBoostModel
model = XGBoostModel()
predictions = model.predict(data)
```

### REST API

```bash
# 获取股票数据
curl "http://localhost:8000/api/data/yahoo/AAPL?start_date=2023-01-01&end_date=2024-01-01"

# 获取技术指标
curl -X POST "http://localhost:8000/api/analysis/technical" \
  -H "Content-Type: application/json" \
  -d '{"data": [...], "indicators": ["sma", "rsi", "macd"]}'

# 训练机器学习模型
curl -X POST "http://localhost:8000/api/ml/train" \
  -H "Content-Type: application/json" \
  -d '{"data": [...], "model_type": "xgboost"}'
```

---

## 🔧 模块说明

### data/ - 数据获取
- **YahooFinance**: 全球股票数据，可能有限制
- **AKShareData**: 中国A股数据，无需API密钥（推荐）
- **TuShareData**: 中国A股数据，需要Token
- **sample_data**: 示例数据生成器（自动备用）

### analysis/ - 数据分析
- **TechnicalIndicators**: 技术指标计算（SMA、EMA、RSI、MACD等）
- **StatisticalAnalysis**: 统计分析（收益率、夏普比率、VaR等）

### ml/ - 机器学习
- **LinearRegressionModel**: 线性回归
- **RandomForestModel**: 随机森林
- **XGBoostModel**: XGBoost
- **LightGBMModel**: LightGBM

### portfolio/ - 投资组合分析
- **RiskMetrics**: 风险指标计算
- **EfficientFrontier**: 有效前沿和最优投资组合

### llm/ - LLM解释
- **NewsAnalyzer**: 新闻获取和分析
- **StockQA**: 股票问答系统

---

## 🐳 Docker部署

### 构建镜像

```bash
docker build -t ai-quant-research .
```

### 运行容器

```bash
docker run -p 8000:8000 -p 8501:8501 ai-quant-research
```

### 使用Docker Compose

```bash
docker-compose up -d
```

---

## 📝 许可证

MIT License

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📮 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: https://github.com/Greensqueen/ai-quant-research/issues

---

## ⚠️ 风险提示

本项目仅供研究和学习使用，不构成投资建议。使用前请充分了解相关风险。投资有风险，入市需谨慎。

---

## 📌 常见问题

### Q: 为什么数据获取失败？
A: Yahoo Finance有速率限制，系统会自动使用示例数据。推荐使用AKShare获取A股数据。

### Q: LLM功能无法使用？
A: 需要在.env文件中配置OPENAI_API_KEY，或者查看是否有网络访问限制。

### Q: 如何添加更多数据源？
A: 在data/目录下创建新的数据源模块，并在api/routes/data.py中添加对应的API接口。

### Q: 如何部署到生产环境？
A: 使用Docker Compose部署，建议配合Nginx反向代理和SSL证书。
