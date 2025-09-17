import operator
from typing import Any, Optional, Annotated, Sequence, List, Dict, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from ai_data_science_team.templates import BaseAgent

load_dotenv()

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
        
        # 确保 X 是二维数组 (n_samples, n_features)
        X_array = np.array(self.X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        self.gp.fit(X_array, np.array(self.y))

    def acquisition_function(self, x, xi=0.01):
        """获取函数（预期改进）"""
        if len(self.y) == 0:
            return 0

        # 确保 x 是正确的格式
        x = np.array(x)
        if x.ndim == 0:  # 标量
            x = x.reshape(1, 1)
        elif x.ndim == 1:  # 一维数组
            x = x.reshape(1, -1)
        # 如果已经是二维数组，保持不变
        
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
    max_iterations: int


def validate_bounds(min_val: float, max_val: float) -> bool:
    """验证最小值不大于最大值"""
    return min_val <= max_val


def get_user_confirmation_via_chat(model, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """通过智能对话获取用户确认的输入、输出和目标设置"""
    print("\n=== 智能配置助手 ===")
    print("我将帮助您配置贝叶斯优化的参数。")
    
    # 分析数据结构
    if "columns" in data_info:
        available_columns = data_info["columns"]
        print(f"\n检测到数据包含以下列: {available_columns}")
    else:
        available_columns = [f"特征{i+1}" for i in range(data_info.get("n_features", 2))]
        if "target_col" in data_info:
            available_columns.append(data_info["target_col"])
        print(f"\n数据包含 {data_info.get('n_features', 2)} 个特征")
    
    # 获取输入变量
    print("\n请告诉我哪些是输入变量（用于优化的特征）？")
    while True:
        user_input = input("输入变量（用逗号分隔，例如：特征1,特征2）: ").strip()
        if user_input:
            input_vars = [var.strip() for var in user_input.split(',')]
            if all(var in available_columns for var in input_vars):
                break
            else:
                invalid_vars = [var for var in input_vars if var not in available_columns]
                print(f"错误：以下变量不存在: {invalid_vars}")
                print(f"可用变量: {available_columns}")
        else:
            print("请至少选择一个输入变量")
    
    # 获取输出变量
    print("\n请告诉我哪个是输出变量（优化目标）？")
    while True:
        output_var = input("输出变量: ").strip()
        if output_var in available_columns:
            if output_var not in input_vars:
                break
            else:
                print("输出变量不能与输入变量重复")
        else:
            print(f"错误：变量 '{output_var}' 不存在")
            print(f"可用变量: {available_columns}")
    
    # 获取优化目标
    print("\n您希望最大化还是最小化输出变量？")
    while True:
        goal_input = input("优化目标（最大化/最小化 或 maximize/minimize）: ").strip().lower()
        if goal_input in ["最大化", "maximize", "max"]:
            goal = "maximize"
            break
        elif goal_input in ["最小化", "minimize", "min"]:
            goal = "minimize"
            break
        else:
            print("请输入有效的优化目标：最大化/最小化 或 maximize/minimize")
    
    # 获取变量边界
    variable_bounds = {}
    print("\n现在需要设置每个输入变量的取值范围：")
    for var in input_vars:
        while True:
            try:
                bounds_input = input(f"请输入 '{var}' 的取值范围（格式：最小值,最大值）: ").strip()
                min_val, max_val = map(float, bounds_input.split(','))
                if validate_bounds(min_val, max_val):
                    variable_bounds[var] = (min_val, max_val)
                    break
                else:
                    print("错误：最小值不能大于最大值")
            except ValueError:
                print("错误：请输入有效的数字格式（例如：0,1）")
    
    return {
        "input_variables": input_vars,
        "output_variable": output_var,
        "optimization_goal": goal,
        "variable_bounds": variable_bounds
    }


def make_bayesian_optimization_agent(
    model: Any,
    n_initial_points: int = 5,
    human_in_the_loop: bool = True,
    checkpointer: Checkpointer = None
):
    """
    Creates a bayesian optimization agent that can be run on optimization problems.
    
    Parameters
    ----------
    model : Any
        The language model to use for the agent.
    n_initial_points : int, optional
        Number of initial points for the optimization. Defaults to 5.
    human_in_the_loop : bool, optional
        Whether to enable human-in-the-loop functionality. Defaults to True.
    checkpointer : Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.
        
    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The bayesian optimization agent as a state graph.
    """
    return create_bayesian_optimization_agent(model, n_initial_points, human_in_the_loop, checkpointer)

def create_bayesian_optimization_agent(model: Any, n_initial_points: int = 5, human_in_the_loop: bool = True, checkpointer: Checkpointer = None):
    """创建贝叶斯优化代理"""

    def setup_node(state: AgentState):
        """设置节点 - 通过智能对话获取用户输入、输出和目标"""
        print("\n=== 贝叶斯优化配置 ===")
        
        # 分析输入数据结构
        data_info = {}
        if state.get("input_data") and "X" in state["input_data"]:
            X_data = state["input_data"]["X"]
            if hasattr(X_data, 'shape'):
                data_info["n_features"] = X_data.shape[1] if len(X_data.shape) > 1 else 1
                data_info["n_samples"] = X_data.shape[0]
            
                # 如果有列名信息，使用它们
            if "columns" in state["input_data"]:
                data_info["columns"] = state["input_data"]["columns"]
            else:
                # 生成默认列名
                feature_cols = [f"特征{i+1}" for i in range(data_info.get("n_features", 2))]
                target_col = "目标值"
                data_info["columns"] = feature_cols + [target_col]
                data_info["target_col"] = target_col
        else:
            # 如果没有数据，使用默认设置
            available_columns = ['特征1', '特征2', '特征3', '特征4', '特征5', '目标值']
            data_info = {
                "columns": available_columns,
                "n_features": len(available_columns) - 1
            }
        
        # 通过智能对话获取配置
        config = get_user_confirmation_via_chat(None, data_info)  # 暂时不使用model参数
        
        # 更新状态
        state["input_variables"] = config["input_variables"]
        state["output_variable"] = config["output_variable"]
        state["optimization_goal"] = config["optimization_goal"]
        state["variable_bounds"] = config["variable_bounds"]
        state["step"] = "confirm"
        state["need_human_approval"] = True  # 需要人工确认
        
        print("\n配置完成，等待最终确认...")
        return state

    def approval_node(state: AgentState):
        """确认节点 - 用户最终确认设置"""
        print("\n=== 请确认优化配置 ===")
        print(f"✓ 输入变量（特征）: {', '.join(state['input_variables'])}")
        print(f"✓ 输出变量（目标）: {state['output_variable']}")
        print(f"✓ 优化目标: {'最大化' if state['optimization_goal'] == 'maximize' else '最小化'}")
        print("✓ 变量取值范围:")
        for var, bounds in state['variable_bounds'].items():
            print(f"   {var}: [{bounds[0]}, {bounds[1]}]")
        
        print("\n请确认以上配置是否正确？")
        while True:
            confirm = input("确认开始优化？(是/否 或 y/n): ").strip().lower()
            if confirm in ["是", "y", "yes", "确认"]:
                state["confirmed_settings"] = True
                state["step"] = "optimize"
                state["need_human_approval"] = False
                print("\n✓ 配置已确认，开始贝叶斯优化...")
                break
            elif confirm in ["否", "n", "no", "取消"]:
                print("\n配置已取消。如需重新配置，请重新运行程序。")
                state["step"] = "setup"
                state["need_human_approval"] = True
                break
            else:
                print("请输入有效的选择：是/否 或 y/n")
        
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
            print("将使用默认边界 [0, 1]")
            for var in missing_bounds:
                state["variable_bounds"][var] = (0.0, 1.0)

        # 确保 optimization_results 已初始化
        if "optimization_results" not in state:
            state["optimization_results"] = []

        return state

    def optimization_node(state: AgentState):
        """优化节点 - 执行贝叶斯优化"""
        print(f"\n=== 贝叶斯优化进行中 ===")

        # 验证状态一致性
        state = validate_state(state)

        # 获取或创建优化器
        bounds_list = []
        for var in state["input_variables"]:
            bounds_list.append(state["variable_bounds"][var])

        if state.get("optimizer_state") is None:
            optimizer = BayesianOptimizer(bounds_list)
            # 如果有初始数据，初始化优化器
            if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                X_data = state["input_data"]["X"]
                Y_data = state["input_data"]["Y"]
                columns = state["input_data"].get("columns", [])
                
                if len(X_data) > 0 and len(Y_data) > 0:
                    if hasattr(Y_data, 'shape') and len(Y_data.shape) > 1:
                        Y_data = Y_data.flatten()
                    
                    # 只选择用户指定的输入变量对应的特征
                    if columns and state.get("input_variables"):
                        # 找到用户选择的特征在原始数据中的索引
                        feature_indices = []
                        for var in state["input_variables"]:
                            if var in columns:
                                feature_indices.append(columns.index(var))
                        
                        if feature_indices:
                            # 只提取选中的特征
                            if hasattr(X_data, 'shape') and len(X_data.shape) > 1:
                                X_data = X_data[:, feature_indices]
                            else:
                                # 如果是列表格式，需要重新构建
                                X_data = [[row[i] for i in feature_indices] for row in X_data]
                    
                    # 确保数据类型正确
                    if hasattr(X_data, 'tolist'):
                        X_list = X_data.tolist()
                    else:
                        X_list = X_data
                    if hasattr(Y_data, 'tolist'):
                        Y_list = Y_data.tolist()
                    else:
                        Y_list = Y_data
                    optimizer.initialize(X_list, Y_list)
                    print(f"✓ 使用 {len(X_list)} 个初始数据点初始化优化器")
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
        print("\n🎯 建议的下一组参数:")
        for var, value in formatted_suggestion.items():
            bounds = state["variable_bounds"][var]
            print(f"   {var}: {value:.4f} (范围: [{bounds[0]}, {bounds[1]}])")

        # 保存优化器状态
        state["optimizer_state"] = optimizer.__dict__

        # 请求用户输入实验结果
        print("\n请按照上述参数进行实验，然后输入结果:")
        while True:
            try:
                result_input = input(f"请输入 '{state['output_variable']}' 的值: ").strip()
                result = float(result_input)
                break
            except ValueError:
                print("错误: 请输入有效的数字")

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
                best_idx = optimizer.y.index(best_result)
            else:
                best_result = min(optimizer.y)
                best_idx = optimizer.y.index(best_result)
            
            print(f"\n✓ 结果已记录! 当前最佳结果: {best_result:.4f}")
            print(f"   总评估次数: {len(optimizer.y)}")
            
            # 显示最佳参数
            if len(optimizer.X) > best_idx:
                best_params = optimizer.X[best_idx]
                print(f"   最佳参数: {dict(zip(state['input_variables'], best_params))}")
        else:
            print("\n✓ 结果已记录! 这是第一次评估。")

        # 保存更新后的优化器状态
        state["optimizer_state"] = optimizer.__dict__

        # 询问是否继续优化
        print("\n是否继续优化过程？")
        while True:
            continue_opt = input("继续优化？(是/否 或 y/n): ").strip().lower()
            if continue_opt in ["y", "yes", "是", "继续"]:
                state["step"] = "optimize"
                break
            elif continue_opt in ["n", "no", "否", "停止"]:
                state["step"] = "complete"
                break
            else:
                print("请输入有效的选择：是/否 或 y/n")

        return state

    def results_node(state: AgentState):
        """结果节点 - 显示最终优化结果"""
        print("\n" + "="*50)
        print("🎉 贝叶斯优化完成")
        print("="*50)
        
        results = state["optimization_results"]

        if results:
            print(f"\n✓ 完成了 {len(results)} 次优化迭代")
            print(f"✓ 优化目标: {'最大化' if state['optimization_goal'] == 'maximize' else '最小化'} {state['output_variable']}")

            # 找到最佳结果
            if state["optimization_goal"] == "maximize":
                best_result = max(results, key=lambda x: x["result"])
            else:
                best_result = min(results, key=lambda x: x["result"])

            print(f"\n🏆 最优结果:")
            print(f"   {state['output_variable']}: {best_result['result']:.6f}")
            print(f"\n🎯 最佳参数组合:")
            for param, value in best_result["parameters"].items():
                bounds = state["variable_bounds"][param]
                print(f"   {param}: {value:.6f} (范围: [{bounds[0]}, {bounds[1]}])")

            # 显示优化进程
            if len(results) > 1:
                print(f"\n📈 优化进程:")
                for i, result in enumerate(results, 1):
                    status = "⭐" if result == best_result else "  "
                    print(f"   迭代 {i:2d}: {result['result']:8.4f} {status}")
                
                # 显示改进情况
                first_result = results[0]["result"]
                improvement = best_result["result"] - first_result
                if state["optimization_goal"] == "minimize":
                    improvement = -improvement
                improvement_pct = (improvement / abs(first_result)) * 100 if first_result != 0 else 0
                print(f"\n🚀 改进情况: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        else:
            print("\n⚠️  没有优化结果记录")

        print("\n" + "="*50)
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

    return workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )


class BayesianOptimizationAgent(BaseAgent):
    """
    贝叶斯优化智能体，使用LangGraph中断和人工审批功能。
    继承自BaseAgent，支持invoke_agent和_compiled_graph.get_state方法。
    """

    def __init__(
        self, 
        model: Any,
        n_initial_points: int = 5,
        human_in_the_loop: bool = True,
        checkpointer: Checkpointer = None
    ):
        self._params = {
            "model": model,
            "n_initial_points": n_initial_points,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create the compiled graph for the bayesian optimization agent. 
        Running this method will reset the response to None.
        """
        self.response = None
        return make_bayesian_optimization_agent(**self._params)

    def invoke_agent(
        self, 
        input_data: Dict[str, Any], 
        user_instructions: str = None, 
        max_iterations: int = 10, 
        **kwargs
    ):
        """
        Invokes the bayesian optimization agent. The response is stored in the response attribute.

        Parameters:
        ----------
            input_data (Dict[str, Any]): 
                Dictionary containing 'X' (input features) and 'Y' (target values).
            user_instructions (str): 
                Instructions for the optimization process.
            max_iterations (int): 
                Maximum number of optimization iterations.
            **kwargs
                Additional keyword arguments to pass to invoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "input_data": input_data,
            "max_iterations": max_iterations,
            "optimization_results": [],
        }, **kwargs)
        self.response = response
        return None

    async def ainvoke_agent(
        self, 
        input_data: Dict[str, Any], 
        user_instructions: str = None, 
        max_iterations: int = 10, 
        **kwargs
    ):
        """
        Asynchronously invokes the bayesian optimization agent. 
        The response is stored in the response attribute.

        Parameters:
        ----------
            input_data (Dict[str, Any]): 
                Dictionary containing 'X' (input features) and 'Y' (target values).
            user_instructions (str): 
                Instructions for the optimization process.
            max_iterations (int): 
                Maximum number of optimization iterations.
            **kwargs
                Additional keyword arguments to pass to ainvoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "input_data": input_data,
            "max_iterations": max_iterations,
            "optimization_results": [],
        }, **kwargs)
        self.response = response
        return None

    def get_optimization_results(self):
        """获取优化结果"""
        if self.response:
            return self.response.get("optimization_results", [])
        return []

    def get_best_result(self):
        """获取最佳结果"""
        results = self.get_optimization_results()
        if not results:
            return None

        optimization_goal = self.response.get("optimization_goal", "maximize") if self.response else "maximize"
        
        if optimization_goal == "maximize":
            return max(results, key=lambda x: x["result"])
        else:
            return min(results, key=lambda x: x["result"])

    def get_current_suggestion(self):
        """获取当前建议的参数"""
        if self.response:
            return self.response.get("current_suggestion")
        return None

    def get_optimization_goal(self):
        """获取优化目标"""
        if self.response:
            return self.response.get("optimization_goal")
        return None

    def get_variable_bounds(self):
        """获取变量边界"""
        if self.response:
            return self.response.get("variable_bounds")
        return {}


def test_bayesian_optimization():
    """测试贝叶斯优化功能"""
    print("🗋 生成测试数据...")
    
    # 创建测试数据
    np.random.seed(42)
    data = np.random.rand(20, 5)
    target = np.sin(data[:, 0]) + np.cos(data[:, 1]) + np.random.normal(0, 0.1, 20)

    df = pd.DataFrame(data, columns=['特征1', '特征2', '特征3', '特征4', '特征5'])
    df['目标值'] = target

    print("✓ 测试数据已生成:")
    print(f"   数据形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")
    print(f"   目标值范围: [{target.min():.3f}, {target.max():.3f}]")

    try:
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
        print("\n🚀 开始贝叶斯优化流程...")
        input_data = {
            "X": X, 
            "Y": Y,
            "columns": list(df.columns)
        }
        agent.invoke_agent(input_data, "优化目标值")
        
    except Exception as e:
        print(f"\n⚠️  测试过程中出现错误: {e}")
        print("请检查依赖和配置是否正确")


if __name__ == '__main__':
    # 运行测试
    test_bayesian_optimization()
