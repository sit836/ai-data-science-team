import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent, DataCleaningAgent

load_dotenv()

MODEL = "deepseek-v3"
LOG = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                 temperature=0.)

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

pandas_data_analyst.invoke_agent(
    user_instructions="用列平均补缺失",
    data_raw=df,
)
print(pandas_data_analyst.get_state_keys())

df_cleaned = pd.DataFrame(pandas_data_analyst.response.get("data_cleaned"))
print(df_cleaned)
