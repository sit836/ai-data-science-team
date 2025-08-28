import json
import operator
import os
from typing import Any, Optional, Annotated, Sequence, List, Dict, Tuple

import pandas as pd
import numpy as np
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Checkpointer
from scipy.stats import norm

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.messages import get_tool_call_names
from ai_data_science_team.utils.regex import format_agent_name

# TODO: tools folder, add get_recommendation.py

# Setup
AGENT_NAME = "bayesian_optimization_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class BayesianOptimizationAgent(BaseAgent):
    def __init__(
            self,
            model: Any,
            n_samples=30,
            log=False,
            log_path=None,
            file_name="data_cleaner.py",
            function_name="data_cleaner",
            overwrite=True,
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False,
            checkpointer: Checkpointer = None
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

        def _make_compiled_graph(self):
            """
            Create the compiled graph for the Bayesian optimization agent. Running this method will reset the response to None.
            """
            self.response = None
            return make_bo_agent(**self._params)

    async def ainvoke_agent(self, data_raw: pd.DataFrame, user_instructions: str = None, max_retries: int = 3,
                            retry_count: int = 0, **kwargs):
        """
        Asynchronously invokes the agent. The response is stored in the response attribute.

        Parameters:
        ----------
            data_raw (pd.DataFrame):
                The raw dataset to be cleaned.
            user_instructions (str):
                Instructions for data cleaning agent.
            max_retries (int):
                Maximum retry attempts for cleaning.
            retry_count (int):
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to ainvoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str = None, max_retries: int = 3,
                     retry_count: int = 0, **kwargs):
        """
        Invokes the agent. The response is stored in the response attribute.

        Parameters:
        ----------
            data_raw (pd.DataFrame):
                The raw dataset to be cleaned.
            user_instructions (str):
                Instructions for data cleaning agent.
            max_retries (int):
                Maximum retry attempts for cleaning.
            retry_count (int):
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to invoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None


def make_bo_agent(model,
                  n_samples=30,
                  log=False,
                  log_path=None,
                  file_name="data_cleaner.py",
                  function_name="data_cleaner",
                  overwrite=True,
                  human_in_the_loop=False,
                  bypass_recommended_steps=False,
                  bypass_explain_code=False,
                  checkpointer: Checkpointer = None
                  ):
    pass
