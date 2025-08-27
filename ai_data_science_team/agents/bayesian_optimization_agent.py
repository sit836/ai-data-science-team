import json
import operator
import os
from typing import Any, Optional, Annotated, Sequence, List, Dict, Tuple

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
    """
    A Bayesian Optimization Agent that can optimize black-box functions using Bayesian methods.

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    bounds : Dict[str, Tuple[float, float]]
        The bounds for each parameter to optimize.
    create_react_agent_kwargs : dict
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent's state.

    Methods:
    --------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled graph.
    ainvoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions asynchronously.
    invoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions.
    get_internal_messages(markdown: bool=False)
        Returns the internal messages from the agent's response.
    get_optimization_results()
        Returns the optimization results.
    get_ai_message(markdown: bool=False)
        Returns the AI message from the agent's response.
    """

    def __init__(
            self,
            model: Any,
            bounds: Dict[str, Tuple[float, float]],
            create_react_agent_kwargs: Optional[Dict] = {},
            invoke_react_agent_kwargs: Optional[Dict] = {},
            checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "bounds": bounds,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled graph for the agent."""
        self.response = None
        return make_bayesian_optimization_agent(**self._params)

    def update_params(self, **kwargs):
        """Updates the agent's parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(self, user_instructions: str = None, **kwargs):
        """Runs the agent with the given user instructions asynchronously."""
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
            },
            **kwargs
        )
        self.response = response
        return None

    def invoke_agent(self, user_instructions: str = None, **kwargs):
        """Runs the agent with the given user instructions."""
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
            },
            **kwargs
        )
        self.response = response
        return None

    def get_internal_messages(self, markdown: bool = False):
        """Returns the internal messages from the agent's response."""
        if self.response is None:
            return []
        return self.response.get("internal_messages", [])

    def get_optimization_results(self):
        """Returns the optimization results."""
        if self.response is None:
            return {}
        return self.response.get("optimization_results", {})

    def get_ai_message(self, markdown: bool = False):
        """Returns the AI message from the agent's response."""
        if self.response is None or not self.response.get("messages"):
            return ""
        return self.response["messages"][0].content

    def get_tool_calls(self):
        """Returns the tool calls made by the agent."""
        if self.response is None:
            return []
        return self.response.get("tool_calls", [])


def get_recommendation(
        observed_points: List[Dict[str, Any]],
        bounds: Dict[str, Tuple[float, float]],
        acquisition_function: str = "ei"
) -> Dict[str, float]:
    """
    Bayesian optimization recommendation tool.

    Parameters:
    ----------
    observed_points : List[Dict[str, Any]]
        List of observed points with parameters and objective values.
    bounds : Dict[str, Tuple[float, float]]
        Bounds for each parameter.
    acquisition_function : str
        Acquisition function to use ('ei', 'ucb', 'pi')

    Returns:
    --------
    Dict[str, float]
        Recommended parameter values for next evaluation.
    """
    if not observed_points:
        # Return random initial point if no observations
        return {param: np.random.uniform(low, high) for param, (low, high) in bounds.items()}

    # Extract parameters and objective values
    X = []
    y = []
    for point in observed_points:
        params = [point[param] for param in bounds.keys()]
        X.append(params)
        y.append(point['objective'])

    X = np.array(X)
    y = np.array(y)

    # Simple surrogate model (Gaussian process emulation)
    def surrogate_model(x_test):
        # Simple distance-based prediction (replace with actual GP in production)
        distances = np.linalg.norm(X - x_test, axis=1)
        weights = np.exp(-distances / 0.1)
        weights = weights / np.sum(weights)
        return np.sum(weights * y), np.std(weights * y)  # Mean and uncertainty

    # Generate candidate points
    n_candidates = 1
    candidates = []
    for _ in range(n_candidates):
        candidate = {param: np.random.uniform(low, high) for param, (low, high) in bounds.items()}
        candidates.append(candidate)

    # Evaluate acquisition function for each candidate
    best_value = np.max(y)
    acquisition_values = []

    for candidate in candidates:
        x_test = np.array([candidate[param] for param in bounds.keys()])
        mu, sigma = surrogate_model(x_test)

        if acquisition_function == "ei":  # Expected Improvement
            if sigma > 0:
                z = (mu - best_value) / sigma
                ei = (mu - best_value) * norm.cdf(z) + sigma * norm.pdf(z)
            else:
                ei = 0
            acquisition_values.append(ei)

        elif acquisition_function == "ucb":  # Upper Confidence Bound
            acquisition_values.append(mu + 2.0 * sigma)

        elif acquisition_function == "pi":  # Probability of Improvement
            if sigma > 0:
                z = (mu - best_value) / sigma
                pi = norm.cdf(z)
            else:
                pi = 0
            acquisition_values.append(pi)

    # Select candidate with highest acquisition value
    best_candidate_idx = np.argmax(acquisition_values)
    return candidates[best_candidate_idx]


# Create the tool for LangChain
get_recommendation_tool = {
    "name": "get_recommendation",
    "description": "Get the next parameter recommendation for Bayesian optimization based on observed results",
    "parameters": {
        "type": "object",
        "properties": {
            "observed_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "number"},
                        **{param: {"type": "number"} for param in ["param1", "param2"]}  # Will be dynamically updated
                    }
                },
                "description": "List of observed points with parameters and objective values"
            },
            "acquisition_function": {
                "type": "string",
                "enum": ["ei", "ucb", "pi"],
                "description": "Acquisition function to use for Bayesian optimization"
            }
        },
        "required": ["observed_points"]
    },
    "function": lambda observed_points, acquisition_function="ei": get_recommendation(
        observed_points, bounds, acquisition_function
    )
}

tools = [get_recommendation_tool]


def make_bayesian_optimization_agent(
        model: Any,
        bounds: Dict[str, Tuple[float, float]],
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates a Bayesian Optimization Agent.

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    bounds : Dict[str, Tuple[float, float]]
        The bounds for each parameter to optimize.
    create_react_agent_kwargs : dict
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent's state.

    Returns:
    --------
    app : langchain.graphs.CompiledStateGraph
        An agent that can perform Bayesian optimization.
    """

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        optimization_results: dict
        tool_calls: List[str]
        observed_points: List[Dict[str, Any]]

    # Update tool parameters dynamically based on bounds
    param_properties = {param: {"type": "number"} for param in bounds.keys()}
    get_recommendation_tool["parameters"]["properties"]["observed_points"]["items"]["properties"] = {
        "objective": {"type": "number"},
        **param_properties
    }

    def bayesian_optimization_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    ")

        print("    * RUN BAYESIAN OPTIMIZATION AGENT")

        tool_node = ToolNode(tools=tools)

        optimization_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **create_react_agent_kwargs,
        )

        response = optimization_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
                "observed_points": state.get("observed_points", []),
            },
            invoke_react_agent_kwargs,
        )

        print("    * POST-PROCESS OPTIMIZATION RESULTS")

        internal_messages = response['messages']
        observed_points = state.get("observed_points", [])

        # Extract tool calls and update observed points
        tool_calls = get_tool_call_names(internal_messages)

        # Check if we have a new observation to add
        new_observation = None
        for msg in internal_messages:
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('function_call'):
                func_call = msg.additional_kwargs['function_call']
                if func_call.get('name') == 'get_recommendation':
                    # This is a recommendation call, not an observation
                    continue
                # Here you would parse actual observation results

        optimization_results = {
            "best_parameters": None,
            "best_score": None,
            "history": observed_points,
            "recommendation": None
        }

        if observed_points:
            best_idx = np.argmax([point['objective'] for point in observed_points])
            optimization_results["best_parameters"] = {
                k: v for k, v in observed_points[best_idx].items() if k != 'objective'
            }
            optimization_results["best_score"] = observed_points[best_idx]['objective']

        # Get the last AI message
        last_ai_message = AIMessage(
            f"Bayesian Optimization Results:\n{json.dumps(optimization_results, indent=2)}",
            role=AGENT_NAME
        )

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "optimization_results": optimization_results,
            "tool_calls": tool_calls,
            "observed_points": observed_points,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("bayesian_optimization_agent", bayesian_optimization_agent)
    workflow.add_edge(START, "bayesian_optimization_agent")
    workflow.add_edge("bayesian_optimization_agent", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app
