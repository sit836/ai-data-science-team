import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent, DataCleaningAgent

load_dotenv()

MODEL = "deepseek-chat"
LOG = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                 temperature=0., max_tokens=100)

df = pd.read_csv("data/bike_sales_data.csv")

pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
    data_cleaning_agent=DataCleaningAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# user_instructions="用列平均补缺失"
user_instructions = "删除包含缺失值的行"
pandas_data_analyst.invoke_agent(
    user_instructions=user_instructions,
    data_raw=df,
)

df_cleaned = pd.DataFrame(pandas_data_analyst.response.get("data_cleaned"))
print(df_cleaned)
