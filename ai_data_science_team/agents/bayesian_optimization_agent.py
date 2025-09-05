import operator
import os
from typing import Any, Optional, Annotated, Sequence, List, Dict

import numpy as np
import pandas as pd
from IPython.display import Markdown
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Checkpointer
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.messages import get_tool_call_names
from ai_data_science_team.utils.regex import format_agent_name

AGENT_NAME = "bayesian_optimization_agent"


class BayesianOptimizer:
    """贝叶斯优化器实现"""

    def __init__(self, bounds, n_initial_points=5):
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.X = []
        self.y = []
        self.gp = None

    def initialize(self, X_init, y_init):
        """初始化优化器"""
        self.X = X_init
        self.y = y_init

    def fit_gp(self):
        """拟合高斯过程"""
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(np.array(self.X), np.array(self.y))

    def acquisition_function(self, x, xi=0.01):
        """获取函数（预期改进）"""
        if len(self.y) == 0:
            return 0

        x = np.array(x).reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)

        if sigma == 0:
            return 0

        best_y = max(self.y)
        z = (mu - best_y - xi) / sigma
        return (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def suggest_next_point(self):
        """建议下一个采样点"""
        if len(self.X) < self.n_initial_points:
            # 初始阶段随机采样
            return [np.random.uniform(b[0], b[1]) for b in self.bounds]

        self.fit_gp()

        # 优化获取函数
        def neg_acquisition(x):
            return -self.acquisition_function(x)

        result = minimize(neg_acquisition,
                          x0=[np.random.uniform(b[0], b[1]) for b in self.bounds],
                          bounds=self.bounds,
                          method='L-BFGS-B')

        return result.x.tolist()

    def update(self, x, y):
        """更新优化器"""
        self.X.append(x)
        self.y.append(y)


class BayesianOptimizationAgent(BaseAgent):
    """
    贝叶斯优化智能体，用于超参数优化和实验设计。

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        用于生成工具调用代理的语言模型。
    create_react_agent_kwargs : dict
        传递给create_react_agent函数的额外关键字参数。
    invoke_react_agent_kwargs : dict
        传递给react代理invoke方法的额外关键字参数。
    checkpointer : langgraph.types.Checkpointer
        用于保存和加载代理状态的检查点。

    Methods:
    --------
    update_params(**kwargs)
        更新代理参数并重新编译图。
    ainvoke_agent(user_instructions: str=None, **kwargs)
        异步运行代理。
    invoke_agent(user_instructions: str=None, **kwargs)
        运行代理。
    get_internal_messages(markdown: bool=False)
        返回代理响应的内部消息。
    get_optimization_results(as_dataframe: bool=False)
        返回优化结果。
    get_ai_message(markdown: bool=False)
        返回AI消息。
    get_tool_calls()
        返回工具调用。

    """

    def __init__(
            self,
            model: Any,
            create_react_agent_kwargs: Optional[Dict] = {},
            invoke_react_agent_kwargs: Optional[Dict] = {},
            checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.optimizer = None
        self.current_suggestion = None

    def _make_compiled_graph(self):
        """创建编译图"""
        self.response = None
        return make_bayesian_optimization_agent(**self._params)

    def update_params(self, **kwargs):
        """更新参数并重新编译图"""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(self, user_instructions: str = None, **kwargs):
        """异步运行代理"""
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "optimizer_state": self.optimizer.__dict__ if self.optimizer else {},
                "current_suggestion": self.current_suggestion,
            },
            **kwargs
        )
        self.response = response
        return None

    def invoke_agent(self, user_instructions: str = None, **kwargs):
        """运行代理"""
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "optimizer_state": self.optimizer.__dict__ if self.optimizer else {},
                "current_suggestion": self.current_suggestion,
            },
            **kwargs
        )
        self.response = response
        return None

    def get_internal_messages(self, markdown: bool = False):
        """返回内部消息"""
        pretty_print = "\n\n".join([f"### {msg.type.upper()}\n\nID: {msg.id}\n\nContent:\n\n{msg.content}" for msg in
                                    self.response["internal_messages"]])
        if markdown:
            return Markdown(pretty_print)
        else:
            return self.response["internal_messages"]

    def get_optimization_results(self, as_dataframe: bool = False):
        """返回优化结果"""
        if as_dataframe:
            return pd.DataFrame(self.response["optimization_results"])
        else:
            return self.response["optimization_results"]

    def get_ai_message(self, markdown: bool = False):
        """返回AI消息"""
        if markdown:
            return Markdown(self.response["messages"][0].content)
        else:
            return self.response["messages"][0].content

    def get_tool_calls(self):
        """返回工具调用"""
        return self.response["tool_calls"]


def make_bayesian_optimization_agent(
        model: Any,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
):
    """
    创建贝叶斯优化代理

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        语言模型
    create_react_agent_kwargs : dict
        create_react_agent的额外参数
    invoke_react_agent_kwargs : dict
        invoke方法的额外参数
    checkpointer : langgraph.types.Checkpointer
        检查点

    Returns:
    --------
    app : CompiledStateGraph
        贝叶斯优化代理
    """

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        optimization_results: dict
        tool_calls: List[str]
        optimizer_state: dict
        current_suggestion: list
        confirmed_parameters: bool
        optimization_goal: str
        input_variables: list
        output_variable: str
        variable_bounds: dict

    def bayesian_optimization_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    ")

        print("    * RUNNING BAYESIAN OPTIMIZATION AGENT")

        # 定义工具
        def identify_parameters(data_info: str, objective: str):
            """识别优化参数和目标"""
            # 与用户交互确定优化目标
            print("Please specify your optimization goal (maximize/minimize):")
            goal = input().strip().lower()
            while goal not in ["maximize", "minimize"]:
                print("Please enter either 'maximize' or 'minimize':")
                goal = input().strip().lower()

            # 与用户交互确定输入变量
            print("Please enter the names of input variables (comma-separated):")
            input_vars = [v.strip() for v in input().split(",")]

            # 与用户交互确定输出变量
            print("Please enter the name of the output variable:")
            output_var = input().strip()

            # 与用户交互确定变量范围
            bounds = {}
            for var in input_vars:
                print(f"Please enter the range for {var} (min,max):")
                min_val, max_val = map(float, input().split(","))
                bounds[var] = (min_val, max_val)

            return {
                "status": "parameters_identified",
                "optimization_goal": goal,
                "input_variables": input_vars,
                "output_variable": output_var,
                "variable_bounds": bounds
            }

        def suggest_next_parameters(optimizer_state: dict):
            """建议下一组参数"""
            # 从状态中获取边界信息
            bounds_list = []
            input_vars = state.get("input_variables", [])
            variable_bounds = state.get("variable_bounds", {})

            for var in input_vars:
                if var in variable_bounds:
                    bounds_list.append(variable_bounds[var])

            # 创建或恢复优化器
            if not state.get("optimizer_state") or not state["optimizer_state"].get("bounds"):
                optimizer = BayesianOptimizer(bounds_list)
            else:
                optimizer = BayesianOptimizer(bounds_list)
                optimizer.__dict__.update(state["optimizer_state"])

            # 获取建议
            suggestion = optimizer.suggest_next_point()

            # 格式化建议以便用户理解
            formatted_suggestion = {}
            for i, var in enumerate(input_vars):
                formatted_suggestion[var] = suggestion[i]

            # 保存优化器状态
            state["optimizer_state"] = optimizer.__dict__
            state["current_suggestion"] = suggestion

            return {"suggestion": formatted_suggestion}

        def confirm_parameters(parameters: list):
            """确认参数设置"""
            print("The suggested parameters are:")
            for var, value in parameters.items():
                print(f"{var}: {value}")

            print("Do you want to use these parameters? (yes/no)")
            response = input().strip().lower()

            if response in ["yes", "y"]:
                return {"confirmed": True}
            else:
                print("Please provide your preferred parameter values:")
                custom_params = {}
                for var in parameters.keys():
                    print(f"{var}:")
                    value = float(input().strip())
                    custom_params[var] = value

                state["current_suggestion"] = [custom_params[var] for var in state.get("input_variables", [])]
                return {"confirmed": True, "custom_parameters": custom_params}

        def update_optimization_result(parameters: list, result: float):
            """更新优化结果"""
            # 获取当前优化器状态或创建新优化器
            bounds_list = []
            input_vars = state.get("input_variables", [])
            variable_bounds = state.get("variable_bounds", {})

            for var in input_vars:
                if var in variable_bounds:
                    bounds_list.append(variable_bounds[var])

            if not state.get("optimizer_state") or not state["optimizer_state"].get("bounds"):
                optimizer = BayesianOptimizer(bounds_list)
            else:
                optimizer = BayesianOptimizer(bounds_list)
                optimizer.__dict__.update(state["optimizer_state"])

            # 更新优化器
            optimizer.update(parameters, result)

            # 保存优化器状态
            state["optimizer_state"] = optimizer.__dict__

            # 更新优化结果
            if "optimization_results" not in state:
                state["optimization_results"] = []

            result_entry = {
                "parameters": dict(zip(input_vars, parameters)),
                "result": result
            }
            state["optimization_results"].append(result_entry)

            return {"status": "updated",
                    "best_result": max(optimizer.y) if state.get("optimization_goal") == "maximize" else min(
                        optimizer.y)}

        tools = [
            identify_parameters,
            suggest_next_parameters,
            confirm_parameters,
            update_optimization_result,
        ]

        tool_node = ToolNode(tools=tools)

        agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **create_react_agent_kwargs,
        )

        response = agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
                "optimizer_state": state.get("optimizer_state", {}),
                "current_suggestion": state.get("current_suggestion", []),
                "confirmed_parameters": state.get("confirmed_parameters", False),
                "optimization_goal": state.get("optimization_goal", ""),
                "input_variables": state.get("input_variables", []),
                "output_variable": state.get("output_variable", ""),
                "variable_bounds": state.get("variable_bounds", {}),
            },
            invoke_react_agent_kwargs,
        )

        print("    * POST-PROCESSING OPTIMIZATION RESULTS")

        internal_messages = response['messages']

        # 处理结果
        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        # 提取优化结果
        optimization_results = response.get("optimization_results", {})
        tool_calls = get_tool_call_names(internal_messages)

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "optimization_results": optimization_results,
            "tool_calls": tool_calls,
            "optimizer_state": response.get("optimizer_state", {}),
            "current_suggestion": response.get("current_suggestion", []),
            "confirmed_parameters": response.get("confirmed_parameters", False),
            "optimization_goal": response.get("optimization_goal", ""),
            "input_variables": response.get("input_variables", []),
            "output_variable": response.get("output_variable", ""),
            "variable_bounds": response.get("variable_bounds", {}),
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


if __name__ == '__main__':
    X, Y = np.array([[1, 2, 3], [1, 2, 2], [4, 3, 4]]), np.array([[1, 12, 3]])

    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                     temperature=0., max_tokens=100)
    agent_executor = make_bayesian_optimization_agent(llm)

    test_query = "I want to optimize a process with parameters temperature, pressure, and time. The output is yield."

    result = agent_executor.invoke({"user_instructions": test_query})
    print("Agent test results:", result)
