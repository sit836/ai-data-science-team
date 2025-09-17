from typing import Any, Dict, List, Optional

from .utils import validate_bounds


def get_user_confirmation_via_chat(model, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """通过智能对话获取用户确认的输入、输出和目标设置。

    说明：此为 v1 版本（一次性问询）。若存在 v2，则建议外层优先调用 v2。
    """
    print("\n=== 智能配置助手 ===")
    print("我将帮助您配置贝叶斯优化的参数。")

    if "columns" in data_info:
        available_columns = data_info["columns"]
        print(f"\n检测到数据包含以下列: {available_columns}")
    else:
        available_columns = [f"特征{i+1}" for i in range(data_info.get("n_features", 2))]
        if "target_col" in data_info:
            available_columns.append(data_info["target_col"])
        print(f"\n数据包含 {data_info.get('n_features', 2)} 个特征")

    # 输入变量 X
    print("\n请告诉我哪些是输入变量（用于优化的特征）。")
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
            print("请至少选择一个输入变量。")

    # 输出变量 Y
    print("\n请告诉我哪个是输出变量（优化目标）？")
    while True:
        output_var = input("输出变量: ").strip()
        if output_var in available_columns:
            if output_var not in input_vars:
                break
            else:
                print("输出变量不能与输入变量重复。")
        else:
            print(f"错误：变量 '{output_var}' 不存在。")
            print(f"可用变量: {available_columns}")

    # 目标 G
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

    # X 的范围
    variable_bounds: Dict[str, tuple] = {}
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
                    print("错误：最小值不能大于最大值。")
            except ValueError:
                print("错误：请输入有效的数字格式（例如：0,1）。")

    return {
        "input_variables": input_vars,
        "output_variable": output_var,
        "optimization_goal": goal,
        "variable_bounds": variable_bounds,
    }


def get_user_confirmation_via_chat_v2(model, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """v2（占位）：暂时委托给 v1；可在此实现流程制造场景的多轮引导。"""
    return get_user_confirmation_via_chat(model, data_info)

