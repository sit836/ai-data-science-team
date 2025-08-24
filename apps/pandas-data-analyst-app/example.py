from langchain_openai import ChatOpenAI
import pandas as pd
import os
import yaml
from pprint import pprint
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent, DataCleaningAgent

load_dotenv()

MODEL = "deepseek-v3"
LOG = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"),    base_url=os.getenv("OPENAI_API_BASE"))

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

# plt.plot(pandas_data_analyst.show(xray=1))
# plt.show()

pandas_data_analyst.invoke_agent(
    user_instructions = "clean the data?",
    data_raw=df,
)

print(pandas_data_analyst.get_data_wrangled())
