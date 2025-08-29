import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = "deepseek-chat"

# 设置LLM
llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                 temperature=0., max_tokens=100)

# 工具1: 搜索苹果公司最新年报中的营收数据
def search_apple_revenue(query):
    """Searches for Apple Inc.'s annual revenue data from a financial news API. Input should be a search query."""
    # 模拟一个金融API的调用（这里用模拟数据代替）
    # 在实际应用中，这里可能是调用Alpha Vantage, Yahoo Finance, 或其他金融数据API
    print(f"[搜索工具被调用: {query}]")

    # 模拟返回的数据结构
    mock_response = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "latest_annual_revenue": 383285000000,  # 2023财年营收：3832.85亿美元
        "previous_annual_revenue": 365817000000,  # 2022财年营收：3658.17亿美元
        "fiscal_year": "2023",
        "currency": "USD"
    }
    return json.dumps(mock_response, ensure_ascii=False)


# 工具2: 计算增长率
def calculate_growth_rate(input_str):
    """Calculates the growth rate between two numbers. Input should be a string 'current, previous'."""
    print(f"[计算器工具被调用: {input_str}]")
    try:
        current, previous = map(float, input_str.split(','))
        growth_rate = ((current - previous) / previous) * 100
        return f"增长率为: {growth_rate:.2f}%"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具3: 获取当前股价
def get_current_stock_price(query):
    """Fetches the current stock price for a given ticker symbol. Input should be a stock symbol like 'AAPL'."""
    print(f"[股价查询工具被调用: {query}]")
    # 模拟一个股价API的调用
    mock_response = {
        "symbol": "AAPL",
        "price": 172.35,
        "currency": "USD",
        "last_updated": datetime.now().isoformat(),
        "change": +1.25
    }
    return json.dumps(mock_response, ensure_ascii=False)


# 将函数封装成Tool对象
tools = [
    Tool(
        name="Search_Apple_Financials",
        func=search_apple_revenue,
        description="Useful for searching Apple Inc.'s latest annual financial revenue data. Input should be a search query about Apple's revenue."
    ),
    Tool(
        name="Calculate_Growth_Rate",
        func=calculate_growth_rate,
        description="Useful for calculating percentage growth rate between two numbers. Input should be two numbers separated by a comma, e.g., 'current, previous'."
    ),
    Tool(
        name="Get_Stock_Price",
        func=get_current_stock_price,
        description="Useful for fetching the current stock price of a company. Input should be a stock ticker symbol like 'AAPL'."
    )
]

# 从LangChain Hub拉取ReAct提示模板
# 这是一个关键部分，模板指导代理如何思考
react_prompt = hub.pull("hwchase17/react")
print(f'{print(react_prompt.template)}=')
quit()

# 创建ReAct代理
agent = create_react_agent(llm, tools, react_prompt)

# 创建代理执行器
agent_executor = AgentExecutor(agent=agent,
                              tools=tools,
                              verbose=True,  # 重要：开启详细日志以查看思考过程
                              handle_parsing_errors=True)

# 定义复杂查询
complex_query = """
请帮我分析一下苹果公司（Apple Inc.）最近的财务状况。
我需要知道他们最近一个财年的营收增长率是多少，
并用这个增长率估算他们下一财年的预期营收。
最后，查一下他们当前的股价是多少。
"""

# 执行任务
try:
    result = agent_executor.invoke({"input": complex_query})
    print("\n" + "="*50)
    print("最终结果:")
    print(result['output'])
except Exception as e:
    print(f"执行过程中出现错误: {e}")
