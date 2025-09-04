import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from dotenv import load_dotenv
from gpytorch.mlls import ExactMarginalLogLikelihood
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.types import Checkpointer

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name

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


@tool
def get_recommendations(X, Y, num_recommendations=1, optimize_direction="maximize"):
    """Find optimal recommendations with Bayesian optimization.

    Args:
        X (torch.Tensor): (n, d) input parameters
        Y (torch.Tensor): (n,) or (n, 1) target values
        num_recommendations (int): number of recommendations to return
        optimize_direction (str): "maximize" or "minimize"

    Returns:
        torch.Tensor: (1, d) recommended parameter configuration
    """

    def _build_bounds_from_data(X: torch.Tensor) -> torch.Tensor:
        """Build [2, d] bounds tensor from observed data X (min/max per column)."""
        col_min = X.min(dim=0).values
        col_max = X.max(dim=0).values
        return torch.stack([col_min, col_max]).to(dtype=torch.double)

    X = torch.tensor(X, dtype=torch.double)
    Y = torch.tensor(Y, dtype=torch.double)
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    if optimize_direction == "minimize":
        Y = -Y

    dim_inputs = X.shape[1]

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        input_transform=Normalize(d=dim_inputs),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = LogExpectedImprovement(model=model, best_f=Y.max())
    bounds = _build_bounds_from_data(X)

    candidate, _ = optimize_acqf(
        acqf,
        bounds=bounds,
        q=num_recommendations,
        num_restarts=1,
        raw_samples=64,
    )
    return candidate


def make_bo_agent(model,
                  n_samples=30,
                  log=False,
                  log_path=None,
                  # file_name="data_cleaner.py",
                  # function_name="data_cleaner",
                  overwrite=True,
                  human_in_the_loop=False,
                  bypass_recommended_steps=False,
                  bypass_explain_code=False,
                  checkpointer: Checkpointer = None
                  ):
    print(format_agent_name(AGENT_NAME))

    tools = [get_recommendations]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke(
        {
            "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
        }
    )
    print("Agent response:", result)


if __name__ == '__main__':
    X, Y = np.array([[1, 2, 3], [1, 2, 2], [4, 3, 4]]), np.array([[1, 12, 3]])

    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                     temperature=0., max_tokens=100)
    agent_executor = make_bo_agent(llm)

    test_query = "I have input parameters [[1,2,3],[2,3,1],[3,1,2],[4,5,6],[5,6,4]] and target values [10,15,12,25,30]. Can you get recommendations to maximize target"

    result = agent_executor.invoke({"input": test_query})
    print("Agent test results:", result)
