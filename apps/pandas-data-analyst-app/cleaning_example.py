import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ai_data_science_team import DataCleaningAgent

load_dotenv()

MODEL = "deepseek-chat"
LOG = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                 temperature=0., max_tokens=100)

df = pd.read_csv("data/bike_sales_data.csv")

data_cleaning_agent = DataCleaningAgent(
    model=llm,
    log=LOG,
    log_path=LOG_PATH,
    human_in_the_loop=True,
)

user_instructions = "删除包含缺失值的行"
config = {"configurable": {"thread_id": "1"}}

data_cleaning_agent.invoke_agent(
    user_instructions=user_instructions,
    data_raw=df,
    config=config
)
# print(data_cleaning_agent.get_recommended_cleaning_steps())
#
# df_cleaned = pd.DataFrame(data_cleaning_agent.response.get("data_cleaned"))
# print(df_cleaned)

# Human Review
state = data_cleaning_agent._compiled_graph.get_state(config=config)

print(state.tasks[-1].interrupts[-1].value)

data_cleaning_agent.invoke(Command(
    resume="缺失补0"),
                           config=config)
state = data_cleaning_agent._compiled_graph.get_state(config=config)
print(state.tasks[-1].interrupts[-1].value)
