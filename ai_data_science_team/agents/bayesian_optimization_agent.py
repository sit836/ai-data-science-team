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
        pretty_print = "\n\n".join([f"### {msg.type.upper()}\n\nID: {msg.id}\n\n内容:\n\n{msg.content}" for msg in
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
        input_data: dict
        auto_mode: bool  # 新增：自动模式标志
        user_confirmed_settings: bool  # 新增：用户确认设置标志

    def bayesian_optimization_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    ")

        print("    * 正在运行贝叶斯优化智能体")

        # 定义工具
        def identify_parameters(data_info: str, objective: str):
            """识别优化参数和目标"""
            # 检查是否有输入数据
            if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                X_data = state["input_data"]["X"]
                Y_data = state["input_data"]["Y"]

                print(f"检测到输入数据: X形状={X_data.shape}, Y形状={Y_data.shape}")

                # 自动识别变量
                if len(X_data.shape) == 2:
                    n_features = X_data.shape[1]
                    input_vars = [f"参数_{i + 1}" for i in range(n_features)]
                    print(f"自动识别到 {n_features} 个输入参数")
                else:
                    input_vars = ["输入参数"]

                output_var = "输出结果"

                # 自动确定边界
                bounds = {}
                for i, var in enumerate(input_vars):
                    if len(X_data.shape) == 2:
                        min_val = float(np.min(X_data[:, i]))
                        max_val = float(np.max(X_data[:, i]))
                    else:
                        min_val = float(np.min(X_data))
                        max_val = float(np.max(X_data))
                    bounds[var] = (min_val, max_val)

                # 自动确定优化目标（基于数据趋势）
                if np.mean(Y_data) > np.median(Y_data):
                    goal = "maximize"
                else:
                    goal = "minimize"

                print(f"自动设置优化目标: {'最大化' if goal == 'maximize' else '最小化'}")

                return {
                    "status": "参数已自动识别",
                    "optimization_goal": goal,
                    "input_variables": input_vars,
                    "output_variable": output_var,
                    "variable_bounds": bounds,
                    "auto_mode": True
                }

            return {
                "status": "需要手动输入参数",
                "auto_mode": False
            }

        def get_user_settings():
            """获取用户设置"""
            print("请提供以下信息:")

            # 获取输入变量
            input_vars = input("请输入输入变量名称（用逗号分隔）: ").split(',')
            input_vars = [var.strip() for var in input_vars]

            # 获取输出变量
            output_var = input("请输入输出变量名称: ").strip()

            # 获取优化目标
            goal = input("请输入优化目标（maximize/minimize）: ").strip().lower()
            while goal not in ["maximize", "minimize"]:
                print("请输入有效的优化目标（maximize/minimize）")
                goal = input("请输入优化目标（maximize/minimize）: ").strip().lower()

            # 获取变量边界
            bounds = {}
            for var in input_vars:
                min_val = float(input(f"请输入变量 {var} 的最小值: "))
                max_val = float(input(f"请输入变量 {var} 的最大值: "))
                bounds[var] = (min_val, max_val)

            return {
                "input_variables": input_vars,
                "output_variable": output_var,
                "optimization_goal": goal,
                "variable_bounds": bounds
            }

        def confirm_settings(settings: dict):
            """确认设置"""
            print("\n请确认以下设置:")
            print(f"输入变量: {settings['input_variables']}")
            print(f"输出变量: {settings['output_variable']}")
            print(f"优化目标: {settings['optimization_goal']}")
            print("变量边界:")
            for var, bound in settings['variable_bounds'].items():
                print(f"  {var}: {bound}")

            response = input("\n确认这些设置吗？（是/否）: ").strip().lower()
            if response in ["是", "yes", "y"]:
                return {"confirmed": True}
            else:
                return {"confirmed": False}

        def suggest_next_parameters(optimizer_state: dict):
            """建议下一组参数"""
            bounds_list = []
            input_vars = state.get("input_variables", [])
            variable_bounds = state.get("variable_bounds", {})

            for var in input_vars:
                if var in variable_bounds:
                    bounds_list.append(variable_bounds[var])

            # 创建或恢复优化器
            if not state.get("optimizer_state") or not state["optimizer_state"].get("bounds"):
                optimizer = BayesianOptimizer(bounds_list)
                # 如果有初始数据，初始化优化器
                if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                    X_data = state["input_data"]["X"]
                    Y_data = state["input_data"]["Y"]
                    if len(X_data) > 0 and len(Y_data) > 0:
                        # 确保数据格式正确
                        if len(Y_data.shape) > 1:
                            Y_data = Y_data.flatten()
                        optimizer.initialize(X_data.tolist(), Y_data.tolist())
                        print(f"使用 {len(X_data)} 个初始数据点初始化优化器")
            else:
                optimizer = BayesianOptimizer(bounds_list)
                optimizer.__dict__.update(state["optimizer_state"])

            # 获取建议
            suggestion = optimizer.suggest_next_point()
            print(f"建议的下一个参数点: {suggestion}")

            # 格式化建议
            formatted_suggestion = {}
            for i, var in enumerate(input_vars):
                formatted_suggestion[var] = suggestion[i]

            # 保存状态
            state["optimizer_state"] = optimizer.__dict__
            state["current_suggestion"] = suggestion

            return {"suggestion": formatted_suggestion}

        def confirm_parameters(parameters: dict):
            """确认参数设置"""
            print("建议的参数如下:")
            for var, value in parameters.items():
                print(f"{var}: {value}")

            # 在自动模式下自动确认
            if state.get("auto_mode", False):
                print("自动模式：使用建议参数")
                return {"confirmed": True}

            print("您想要使用这些参数吗？（是/否）")
            response = input().strip().lower()

            if response in ["是", "yes", "y"]:
                return {"confirmed": True}
            else:
                print("请提供您偏好的参数值:")
                custom_params = {}
                for var in parameters.keys():
                    print(f"{var}:")
                    value = float(input().strip())
                    custom_params[var] = value

                state["current_suggestion"] = [custom_params[var] for var in state.get("input_variables", [])]
                return {"confirmed": True, "custom_parameters": custom_params}

        def update_optimization_result(parameters: list, result: float):
            """更新优化结果"""
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

            # 保存状态
            state["optimizer_state"] = optimizer.__dict__

            # 更新优化结果
            if "optimization_results" not in state:
                state["optimization_results"] = []

            result_entry = {
                "parameters": dict(zip(input_vars, parameters)),
                "result": result
            }
            state["optimization_results"].append(result_entry)

            best_result = max(optimizer.y) if state.get("optimization_goal") == "maximize" else min(optimizer.y)
            print(f"更新完成！当前最佳结果: {best_result}")

            return {
                "status": "已更新",
                "best_result": best_result,
                "total_evaluations": len(optimizer.y)
            }

        tools = [
            identify_parameters,
            get_user_settings,
            confirm_settings,
            suggest_next_parameters,
            confirm_parameters,
            update_optimization_result,
        ]

        tool_node = ToolNode(tools=tools)

        # 创建提示模板，指导模型使用工具
        system_prompt = """你是一个贝叶斯优化专家。请按照以下步骤执行优化：
        1. 首先询问用户输入变量、输出变量和优化目标
        2. 确认用户设置
        3. 调用suggest_next_parameters获取建议参数
        4. 调用confirm_parameters确认参数
        5. 获取实验结果后调用update_optimization_result更新优化器

        对于用户提供的初始数据，自动进行参数识别和优化。"""

        agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **create_react_agent_kwargs,
        )

        # 准备初始消息
        messages = [("system", system_prompt), ("user", state["user_instructions"])]

        response = agent.invoke(
            {
                "messages": messages,
                "optimizer_state": state.get("optimizer_state", {}),
                "current_suggestion": state.get("current_suggestion", []),
                "confirmed_parameters": state.get("confirmed_parameters", False),
                "optimization_goal": state.get("optimization_goal", ""),
                "input_variables": state.get("input_variables", []),
                "output_variable": state.get("output_variable", ""),
                "variable_bounds": state.get("variable_bounds", {}),
                "input_data": state.get("input_data", {}),
                "auto_mode": state.get("auto_mode", False),
                "user_confirmed_settings": state.get("user_confirmed_settings", False),
            },
            invoke_react_agent_kwargs,
        )

        print("    * 正在后处理优化结果")

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
            "input_data": response.get("input_data", {}),
            "auto_mode": response.get("auto_mode", False),
            "user_confirmed_settings": response.get("user_confirmed_settings", False),
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


# 测试函数
def test_bayesian_optimization():
    """测试贝叶斯优化功能"""
    # 创建测试数据
    np.random.seed(42)
    X = np.random.rand(10, 2)  # 10个样本，2个特征
    Y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 10)  # 简单的目标函数

    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.0,
        max_tokens=500
    )

    # 创建优化代理
    agent = BayesianOptimizationAgent(llm)

    # 运行优化
    test_query = "使用提供的数据进行贝叶斯优化，目标是找到最佳参数组合"

    print("\n开始优化过程...")
    agent.invoke_agent(
        user_instructions=test_query,
        input_data={"X": X, "Y": Y}
    )

    # 获取结果
    print("\n优化结果:")
    results = agent.get_optimization_results()
    print("优化记录:", results)

    # 显示最佳结果
    if results and len(results) > 0:
        best_result = max(results, key=lambda x: x['result']) if len(results) > 0 else None
        print(f"\n最佳参数组合: {best_result['parameters']}")
        print(f"最佳结果值: {best_result['result']}")

    return agent


if __name__ == '__main__':
    # 运行测试
    test_bayesian_optimization()
