import operator
from typing import Any, Optional, Annotated, Sequence, List, Dict, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer

from .constants import AGENT_NAME, AVAILABLE_COLUMNS
from .optimizer import BayesianOptimizer
from .dialog import get_user_confirmation_via_chat
from .utils import serialize_optimizer_state


class AgentState(TypedDict):
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
    step: str
    need_human_approval: bool
    max_iterations: int


def create_bayesian_optimization_agent(
    model: Any,
    n_initial_points: int = 5,
    human_in_the_loop: bool = True,
    checkpointer: Checkpointer = None,
):
    def setup_node(state: AgentState):
        print("\n=== 贝叶斯优化配置 ===")

        # 分析输入数据结构
        data_info: Dict[str, Any] = {}
        if state.get("input_data") and "X" in state["input_data"]:
            X_data = state["input_data"]["X"]
            if hasattr(X_data, "shape"):
                data_info["n_features"] = X_data.shape[1] if len(X_data.shape) > 1 else 1
                data_info["n_samples"] = X_data.shape[0]

            if "columns" in state["input_data"]:
                data_info["columns"] = state["input_data"]["columns"]
            else:
                feature_cols = [f"特征{i+1}" for i in range(data_info.get("n_features", 2))]
                target_col = "目标"
                data_info["columns"] = feature_cols + [target_col]
                data_info["target_col"] = target_col
        else:
            data_info = {"columns": AVAILABLE_COLUMNS, "n_features": len(AVAILABLE_COLUMNS) - 1}

        # 通过智能对话获取配置
        config = get_user_confirmation_via_chat(None, data_info)

        state["input_variables"] = config["input_variables"]
        state["output_variable"] = config["output_variable"]
        state["optimization_goal"] = config["optimization_goal"]
        state["variable_bounds"] = config["variable_bounds"]
        state["step"] = "confirm"
        state["need_human_approval"] = True

        print("\n配置完成，等待最终确认...")
        return state

    def approval_node(state: AgentState):
        print("\n=== 请确认优化配置 ===")
        print(f"输入变量（特征）: {', '.join(state['input_variables'])}")
        print(f"输出变量（目标）: {state['output_variable']}")
        print(f"优化目标: {'最大化' if state['optimization_goal'] == 'maximize' else '最小化'}")
        print("变量取值范围:")
        for var, bounds in state["variable_bounds"].items():
            print(f"   {var}: [{bounds[0]}, {bounds[1]}]")

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
        missing_bounds = [v for v in state["input_variables"] if v not in state["variable_bounds"]]
        if missing_bounds:
            print(f"\n警告: 以下变量缺少边界设置: {missing_bounds}")
            print("将使用默认边界 [0, 1]")
            for v in missing_bounds:
                state["variable_bounds"][v] = (0.0, 1.0)

        if "optimization_results" not in state:
            state["optimization_results"] = []
        return state

    def optimization_node(state: AgentState):
        print(f"\n=== 贝叶斯优化进行中 ===")
        state = validate_state(state)

        bounds_list = [state["variable_bounds"][v] for v in state["input_variables"]]

        if state.get("optimizer_state") is None:
            optimizer = BayesianOptimizer(bounds_list)
            if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                X_data = state["input_data"]["X"]
                Y_data = state["input_data"]["Y"]
                columns = state["input_data"].get("columns", [])
                if len(X_data) > 0 and len(Y_data) > 0:
                    if hasattr(Y_data, "shape") and len(Y_data.shape) > 1:
                        Y_data = Y_data.flatten()

                    if columns and state.get("input_variables"):
                        idxs: List[int] = []
                        for v in state["input_variables"]:
                            if v in columns:
                                idxs.append(columns.index(v))
                        if idxs:
                            if hasattr(X_data, "shape") and len(X_data.shape) > 1:
                                X_data = X_data[:, idxs]
                            else:
                                X_data = [[row[i] for i in idxs] for row in X_data]

                    X_list = X_data.tolist() if hasattr(X_data, "tolist") else X_data
                    Y_list = Y_data.tolist() if hasattr(Y_data, "tolist") else Y_data
                    optimizer.initialize(X_list, Y_list)
                    print(f"✓ 使用 {len(X_list)} 个初始数据点初始化优化器")
        else:
            opt_state = state["optimizer_state"] or {}
            n_ip = int(opt_state.get("n_initial_points", 5))
            optimizer = BayesianOptimizer(bounds_list, n_initial_points=n_ip)
            optimizer.X = opt_state.get("X", [])
            optimizer.y = opt_state.get("y", [])

        suggestion = optimizer.suggest_next_point()
        state["current_suggestion"] = suggestion

        formatted = {v: suggestion[i] for i, v in enumerate(state["input_variables"])}
        print("\n🎯 建议的下一组参数")
        for var, value in formatted.items():
            b = state["variable_bounds"][var]
            print(f"   {var}: {value:.4f} (范围: [{b[0]}, {b[1]}])")

        state["optimizer_state"] = serialize_optimizer_state(optimizer)

        print("\n请按照上述参数进行实验，然后输入结果:")
        while True:
            try:
                result_input = input(f"请输入 '{state['output_variable']}' 的值: ").strip()
                result = float(result_input)
                break
            except ValueError:
                print("错误: 请输入有效的数字")

        optimizer.update(suggestion, result)

        state["optimization_results"].append({"parameters": formatted, "result": result})

        if optimizer.y:
            if state["optimization_goal"] == "maximize":
                best_result = max(optimizer.y)
                best_idx = optimizer.y.index(best_result)
            else:
                best_result = min(optimizer.y)
                best_idx = optimizer.y.index(best_result)

            print(f"\n✓ 结果已记录! 当前最佳结果: {best_result:.4f}")
            print(f"   总评估次数: {len(optimizer.y)}")
            if len(optimizer.X) > best_idx:
                best_params = optimizer.X[best_idx]
                print(f"   最佳参数: {dict(zip(state['input_variables'], best_params))}")
        else:
            print("\n✓ 结果已记录! 这是第一次评估")

        state["optimizer_state"] = serialize_optimizer_state(optimizer)

        print("\n是否继续优化过程？")
        while True:
            cont = input("继续优化？(是/否 或 y/n): ").strip().lower()
            if cont in ["y", "yes", "是", "继续"]:
                state["step"] = "optimize"
                break
            elif cont in ["n", "no", "否", "停止"]:
                state["step"] = "complete"
                break
            else:
                print("请输入有效的选择：是/否 或 y/n")

        return state

    def results_node(state: AgentState):
        print("\n" + "=" * 50)
        print("🎉 贝叶斯优化完成")
        print("=" * 50)

        results = state["optimization_results"]
        if results:
            print(f"\n✓ 完成了 {len(results)} 次优化迭代")
            print(
                f"✓ 优化目标: {'最大化' if state['optimization_goal'] == 'maximize' else '最小化'} {state['output_variable']}"
            )

            if state["optimization_goal"] == "maximize":
                best_result = max(results, key=lambda x: x["result"])
            else:
                best_result = min(results, key=lambda x: x["result"])

            print(f"\n🏆 最优结果:")
            print(f"   {state['output_variable']}: {best_result['result']:.6f}")
            print(f"\n🎯 最佳参数组合:")
            for param, value in best_result["parameters"].items():
                b = state["variable_bounds"][param]
                print(f"   {param}: {value:.6f} (范围: [{b[0]}, {b[1]}])")
        else:
            print("\n⚠️  没有优化结果记录")

        print("\n" + "=" * 50)
        state["step"] = "end"
        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("setup", setup_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("optimize", optimization_node)
    workflow.add_node("results", results_node)

    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "approval")

    def after_approval(state: AgentState):
        return "optimize" if state.get("confirmed_settings", False) else "approval"

    workflow.add_conditional_edges("approval", after_approval)

    def after_optimization(state: AgentState):
        return "results" if state.get("step") == "complete" else "optimize"

    workflow.add_conditional_edges("optimize", after_optimization)
    workflow.add_edge("results", END)

    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)

