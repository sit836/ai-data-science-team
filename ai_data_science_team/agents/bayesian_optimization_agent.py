import operator
import os
from typing import Any, Optional, Annotated, Sequence, List, Dict, TypedDict
import numpy as np
import pandas as pd
from IPython.display import Markdown
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Checkpointer
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# 加载环境变量
load_dotenv()

AGENT_NAME = "bayesian_optimization_agent"

# 定义可用列名
AVAILABLE_COLUMNS = ['特征1', '特征2', '特征3', '特征4', '特征5', '目标值']


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


class AgentState(TypedDict):
    """代理状态定义"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instructions: str
    input_variables: List[str]
    output_variable: str
    optimization_goal: str
    variable_bounds: Dict[str, tuple]
    input_data: Optional[Dict[str, Any]]
    optimizer_state: Optional[Dict[str, Any]]
    current_suggestion: Optional[List[float]]
    optimization_results: List[Dict[str, Any]]
    confirmed_settings: bool
    step: str  # 当前步骤: setup, confirm, optimize, complete
    need_human_approval: bool


def validate_variable_name(var_name: str) -> bool:
    """验证变量名是否存在于可用列中"""
    return var_name in AVAILABLE_COLUMNS


def validate_bounds(min_val: float, max_val: float) -> bool:
    """验证最小值不大于最大值"""
    return min_val <= max_val


def get_validated_input_variables():
    """获取并验证输入变量"""
    while True:
        print("\n请选择输入变量（特征列）:")
        print(f"可用列: {AVAILABLE_COLUMNS}")
        input_cols = input("请输入用逗号分隔的列名（例如: 特征1,特征2）: ").split(',')
        input_cols = [col.strip() for col in input_cols]

        # 验证变量名
        invalid_vars = [var for var in input_cols if not validate_variable_name(var)]

        if invalid_vars:
            print(f"错误: 以下变量名不存在: {invalid_vars}")
            print("请使用以下可用列:", AVAILABLE_COLUMNS)
            continue

        if not input_cols:
            print("错误: 至少需要一个输入变量")
            continue

        return input_cols


def get_validated_output_variable():
    """获取并验证输出变量"""
    while True:
        print("\n请选择输出变量（目标列）:")
        output_col = input("请输入列名: ").strip()

        # 验证变量名
        if not validate_variable_name(output_col):
            print(f"错误: 变量名 '{output_col}' 不存在")
            print("请使用以下可用列:", AVAILABLE_COLUMNS)
            continue

        return output_col


def get_validated_bounds(var_name: str):
    """获取并验证变量边界"""
    while True:
        try:
            min_val = float(input(f"请输入变量 '{var_name}' 的最小值: "))
            max_val = float(input(f"请输入变量 '{var_name}' 的最大值: "))

            # 验证边界
            if not validate_bounds(min_val, max_val):
                print("错误: 最小值不能大于最大值，请重新输入")
                continue

            return min_val, max_val
        except ValueError:
            print("错误: 请输入有效的数字")


def create_bayesian_optimization_agent(model: Any):
    """创建贝叶斯优化代理"""

    def setup_node(state: AgentState):
        """设置节点 - 获取用户输入、输出和目标"""
        print("\n=== 设置优化参数 ===")

        # 获取并验证输入变量
        input_cols = get_validated_input_variables()

        # 获取并验证输出变量
        output_col = get_validated_output_variable()

        # 获取优化目标
        print("\n请选择优化目标:")
        goal = input("最大化还是最小化？(maximize/minimize): ").strip().lower()
        while goal not in ["maximize", "minimize"]:
            print("请输入有效的优化目标（maximize/minimize）")
            goal = input("最大化还是最小化？(maximize/minimize): ").strip().lower()

        # 获取并验证变量边界
        print("\n请设置变量边界:")
        variable_bounds = {}
        for var in input_cols:
            min_val, max_val = get_validated_bounds(var)
            variable_bounds[var] = (min_val, max_val)

        # 更新状态
        state["input_variables"] = input_cols
        state["output_variable"] = output_col
        state["optimization_goal"] = goal
        state["variable_bounds"] = variable_bounds
        state["step"] = "confirm"
        state["need_human_approval"] = True  # 需要人工审批

        print("\n设置完成，等待确认...")
        return state

    def approval_node(state: AgentState):
        """审批节点 - 用户确认或修改设置"""
        print("\n=== 请确认优化设置 ===")
        print(f"输入变量: {state['input_variables']}")
        print(f"输出变量: {state['output_variable']}")
        print(f"优化目标: {state['optimization_goal']}")
        print("变量边界:")
        for var, bounds in state['variable_bounds'].items():
            print(f"  {var}: {bounds[0]} ~ {bounds[1]}")

        print("\n选项:")
        print("1. 确认设置并开始优化")
        print("2. 修改输入变量")
        print("3. 修改输出变量")
        print("4. 修改优化目标")
        print("5. 修改变量边界")

        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == "1":
            # 确认设置
            state["confirmed_settings"] = True
            state["step"] = "optimize"
            state["need_human_approval"] = False
            print("设置已确认，开始优化...")
        elif choice == "2":
            # 修改输入变量
            print("\n当前输入变量:", state["input_variables"])
            new_input_cols = get_validated_input_variables()
            state["input_variables"] = new_input_cols

            # 确保变量边界与新输入变量一致
            new_bounds = {}
            for var in new_input_cols:
                if var in state["variable_bounds"]:
                    new_bounds[var] = state["variable_bounds"][var]
                else:
                    print(f"\n需要为新增变量 '{var}' 设置边界:")
                    min_val, max_val = get_validated_bounds(var)
                    new_bounds[var] = (min_val, max_val)

            state["variable_bounds"] = new_bounds
            print("输入变量已更新")

        elif choice == "3":
            # 修改输出变量
            print("\n当前输出变量:", state["output_variable"])
            new_output_col = get_validated_output_variable()
            state["output_variable"] = new_output_col
            print("输出变量已更新")

        elif choice == "4":
            # 修改优化目标
            print("\n当前优化目标:", state["optimization_goal"])
            new_goal = input("请输入新的优化目标 (maximize/minimize): ").strip().lower()
            while new_goal not in ["maximize", "minimize"]:
                print("请输入有效的优化目标（maximize/minimize）")
                new_goal = input("请输入新的优化目标 (maximize/minimize): ").strip().lower()
            state["optimization_goal"] = new_goal
            print("优化目标已更新")

        elif choice == "5":
            # 修改变量边界
            print("\n当前变量边界:")
            for var, bounds in state['variable_bounds'].items():
                print(f"  {var}: {bounds[0]} ~ {bounds[1]}")

            var_to_modify = input("请输入要修改的变量名: ").strip()
            if var_to_modify in state["variable_bounds"]:
                min_val, max_val = get_validated_bounds(var_to_modify)
                state["variable_bounds"][var_to_modify] = (min_val, max_val)
                print("变量边界已更新")
            else:
                print(f"变量 '{var_to_modify}' 不存在，请先添加到输入变量中")

        # 无论修改什么，都需要重新确认
        state["need_human_approval"] = True

        return state

    def validate_state(state: AgentState):
        """验证状态是否一致"""
        # 确保所有输入变量都有对应的边界
        missing_bounds = []
        for var in state["input_variables"]:
            if var not in state["variable_bounds"]:
                missing_bounds.append(var)

        if missing_bounds:
            print(f"\n警告: 以下变量缺少边界设置: {missing_bounds}")
            print("请为这些变量设置边界:")
            for var in missing_bounds:
                min_val, max_val = get_validated_bounds(var)
                state["variable_bounds"][var] = (min_val, max_val)

        return state

    def optimization_node(state: AgentState):
        """优化节点 - 执行贝叶斯优化"""
        print(f"\n--- 执行贝叶斯优化 ---")

        # 首先验证状态一致性
        state = validate_state(state)

        # 获取或创建优化器
        bounds_list = []
        for var in state["input_variables"]:
            if var in state["variable_bounds"]:
                bounds_list.append(state["variable_bounds"][var])
            else:
                # 如果变量没有边界，使用默认边界
                print(f"警告: 变量 '{var}' 没有边界设置，使用默认边界 [0, 1]")
                bounds_list.append((0.0, 1.0))
                state["variable_bounds"][var] = (0.0, 1.0)

        if state.get("optimizer_state") is None:
            optimizer = BayesianOptimizer(bounds_list)
            # 如果有初始数据，初始化优化器
            if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                X_data = state["input_data"]["X"]
                Y_data = state["input_data"]["Y"]
                if len(X_data) > 0 and len(Y_data) > 0:
                    if len(Y_data.shape) > 1:
                        Y_data = Y_data.flatten()
                    optimizer.initialize(X_data.tolist(), Y_data.tolist())
                    print(f"使用 {len(X_data)} 个初始数据点初始化优化器")
        else:
            optimizer = BayesianOptimizer(bounds_list)
            optimizer.__dict__.update(state["optimizer_state"])

        # 获取建议参数
        suggestion = optimizer.suggest_next_point()
        state["current_suggestion"] = suggestion

        # 格式化建议
        formatted_suggestion = {}
        for i, var in enumerate(state["input_variables"]):
            formatted_suggestion[var] = suggestion[i]

        # 显示建议参数
        print("\n建议的参数:")
        for var, value in formatted_suggestion.items():
            print(f"  {var}: {value:.4f}")

        # 保存优化器状态
        state["optimizer_state"] = optimizer.__dict__

        # 请求用户输入实验结果
        print("\n请输入实验结果:")
        result = float(input("结果值: "))

        # 更新优化器
        optimizer.update(suggestion, result)

        # 记录结果
        result_entry = {
            "parameters": formatted_suggestion,
            "result": result
        }
        state["optimization_results"].append(result_entry)

        # 显示当前最佳结果
        if optimizer.y:
            if state["optimization_goal"] == "maximize":
                best_result = max(optimizer.y)
            else:
                best_result = min(optimizer.y)
            print(f"更新完成！当前最佳结果: {best_result:.4f}")
            print(f"总评估次数: {len(optimizer.y)}")
        else:
            print("更新完成！这是第一次评估。")

        # 保存更新后的优化器状态
        state["optimizer_state"] = optimizer.__dict__

        # 询问是否继续优化
        continue_opt = input("\n是否继续优化？(y/n): ").strip().lower()
        if continue_opt in ["y", "yes", "是"]:
            state["step"] = "optimize"
        else:
            state["step"] = "complete"

        return state

    def results_node(state: AgentState):
        """结果节点 - 显示最终优化结果"""
        print("\n=== 优化完成 ===")
        results = state["optimization_results"]

        if results:
            print(f"完成了 {len(results)} 次优化迭代")

            # 找到最佳结果
            if state["optimization_goal"] == "maximize":
                best_result = max(results, key=lambda x: x["result"])
            else:
                best_result = min(results, key=lambda x: x["result"])

            print(f"\n最佳参数组合:")
            for param, value in best_result["parameters"].items():
                print(f"  {param}: {value:.4f}")
            print(f"最佳结果值: {best_result['result']:.4f}")

            # 显示所有优化历史
            print(f"\n优化历史:")
            for i, result in enumerate(results, 1):
                print(f"迭代 {i}: 结果 = {result['result']:.4f}")
        else:
            print("没有优化结果记录")

        state["step"] = "end"
        return state

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("setup", setup_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("optimize", optimization_node)
    workflow.add_node("results", results_node)

    # 定义流程
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "approval")

    # 使用条件边处理审批结果
    def after_approval(state: AgentState):
        if state.get("confirmed_settings", False):
            return "optimize"
        else:
            return "approval"  # 需要重新审批

    workflow.add_conditional_edges("approval", after_approval)

    # 优化节点的后续流程
    def after_optimization(state: AgentState):
        if state.get("step") == "complete":
            return "results"
        else:
            return "optimize"  # 继续优化

    workflow.add_conditional_edges("optimize", after_optimization)
    workflow.add_edge("results", END)

    return workflow.compile()


class BayesianOptimizationAgent:
    """
    贝叶斯优化智能体，使用LangGraph中断和人工审批功能。
    """

    def __init__(self, model: Any):
        self.model = model
        self.workflow = create_bayesian_optimization_agent(model)
        self.state = {
            "messages": [],
            "user_instructions": "",
            "input_variables": [],
            "output_variable": "",
            "optimization_goal": "",
            "variable_bounds": {},
            "input_data": None,
            "optimizer_state": None,
            "current_suggestion": None,
            "optimization_results": [],
            "confirmed_settings": False,
            "step": "setup",
            "need_human_approval": False
        }

    def run(self, input_data: Dict[str, Any] = None):
        """运行优化流程"""
        if input_data:
            self.state["input_data"] = input_data

        # 执行工作流
        self.state = self.workflow.invoke(self.state)
        return self.state

    def get_optimization_results(self):
        """获取优化结果"""
        return self.state["optimization_results"]

    def get_best_result(self):
        """获取最佳结果"""
        if not self.state["optimization_results"]:
            return None

        if self.state["optimization_goal"] == "maximize":
            return max(self.state["optimization_results"], key=lambda x: x["result"])
        else:
            return min(self.state["optimization_results"], key=lambda x: x["result"])


def test_bayesian_optimization():
    """测试贝叶斯优化功能"""
    # 创建测试数据
    np.random.seed(42)
    data = np.random.rand(20, 5)
    target = np.sin(data[:, 0]) + np.cos(data[:, 1]) + np.random.normal(0, 0.1, 20)

    df = pd.DataFrame(data, columns=['特征1', '特征2', '特征3', '特征4', '特征5'])
    df['目标值'] = target

    print("生成的测试数据:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")

    # 初始化模型
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=500
    )

    # 创建优化代理
    agent = BayesianOptimizationAgent(llm)

    # 提取数据
    X = df[['特征1', '特征2']].values
    Y = df['目标值'].values

    # 运行优化流程
    print("\n开始贝叶斯优化流程...")
    agent.run({"X": X, "Y": Y})


if __name__ == '__main__':
    # 运行测试
    test_bayesian_optimization()
