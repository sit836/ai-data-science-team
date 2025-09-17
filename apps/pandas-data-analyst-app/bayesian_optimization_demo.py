import os
import sys

# Add the project root to Python path to ensure we import the local version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ai_data_science_team import BayesianOptimizationAgent

load_dotenv()

MODEL = "deepseek-chat"

# 增大 tokens，减少 JSON 截断概率
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0.0,
    max_tokens=512,
)


def run_auto(data_file: str, user_instructions: str, scenario_name: str):
    """自动模式：直接运行并显示结果。"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    df = pd.read_csv(data_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    agent = BayesianOptimizationAgent(
        model=llm,
        human_in_the_loop=False,
        n_initial_points=3,
    )

    print(f"\nUser instructions: {user_instructions}")

    # Convert DataFrame to the expected format
    input_data = {
        "X": df.drop(columns=[df.columns[-1]]).values,  # All columns except the last one
        "Y": df.iloc[:, -1].values,  # Last column as target
        "columns": list(df.columns)
    }
    agent.invoke_agent(input_data=input_data, user_instructions=user_instructions)

    print("\n--- Optimization Results ---")
    results = agent.get_optimization_results()
    print(results if results else "<no_results>")

    return agent


def run_manual_input(data_file: str, user_instructions: str, scenario_name: str):
    """手动输入模式：获取建议后，手动录入一次实验结果并生成下一建议点。"""
    agent = run_auto(data_file, user_instructions, scenario_name + " (Manual Input)")

    results = agent.get_optimization_results()
    if not results or not isinstance(results, dict):
        print("\n未获取到有效的 optimization_results，无法进入手动输入演示。")
        return

    input_vars = results.get("input_variables", [])
    suggestions = results.get("suggested_points", [])
    if not suggestions:
        print("\n没有生成建议点，无法演示手动录入。")
        return

    print("\n--- Manual Feedback ---")
    print("已有建议点：")
    for s in suggestions:
        sid = s.get("suggestion_id")
        params = s.get("parameters")
        acq = s.get("acquisition_value")
        print(f"  #{sid}: params={params}, acquisition={acq}")

    # 选择一个建议点
    try:
        sel = int(input("选择一个建议点编号 (如 1): ").strip())
    except Exception:
        print("输入无效，跳过手动录入。")
        return

    chosen = next((s for s in suggestions if s.get("suggestion_id") == sel), None)
    if not chosen:
        print("未找到该编号的建议点。")
        return

    # 输入手动测得的目标值
    try:
        measured = float(input("请输入该点测得的目标值 (float): ").strip())
    except Exception:
        print("输入无效，跳过手动录入。")
        return

    params_dict = chosen.get("parameters", {})
    # 将参数按 input_variables 的顺序转换为列表
    new_point = [params_dict.get(v) for v in input_vars]
    ok = agent.update_optimizer_with_feedback(new_point, measured)
    if not ok:
        print("优化器未就绪，无法更新。")
        return

    optimizer = agent.get_bayesian_optimizer()
    next_point = optimizer.suggest_next_point() if optimizer else None
    print(f"已更新优化器；下一建议点：{next_point}")


def run_hitl(data_file: str, user_instructions: str, scenario_name: str):
    """人机协同模式：在 confirm_optimization_problem 处进行人工确认并继续。"""
    print(f"\n{'='*60}")
    print(f"SCENARIO (HITL): {scenario_name}")
    print(f"{'='*60}")

    df = pd.read_csv(data_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    agent = BayesianOptimizationAgent(
        model=llm,
        human_in_the_loop=True,  # 开启人机协同
        n_initial_points=3,
    )

    # 配置线程 ID 以支持中断恢复
    config = {"configurable": {"thread_id": "bo-hitl-demo"}}

    print(f"\nUser instructions: {user_instructions}")

    # Convert DataFrame to the expected format
    input_data = {
        "X": df.drop(columns=[df.columns[-1]]).values,  # All columns except the last one
        "Y": df.iloc[:, -1].values,  # Last column as target
        "columns": list(df.columns)
    }
    
    # 第一次调用：图会在 human_review 节点中断并返回给用户
    try:
        agent.invoke_agent(input_data=input_data, user_instructions=user_instructions, config=config)
        # 若未中断直接完成，也能继续展示结果
    except Exception as e:
        # 某些运行时可能通过异常暴露中断；继续走恢复流程
        print(f"首次调用出现可恢复的中断/异常：{e}")

    # 读取提示并让用户输入 yes 或修改意见；必要时可多轮确认
    print("\n--- Human-in-the-Loop ---")
    print("请根据终端输出的配置提示进行确认或修改（输入 yes 或者你的修改意见）")
    for round_idx in range(3):
        user_text = input("你的输入: ")
        try:
            agent._compiled_graph.invoke(Command(resume=user_text), config=config)
            # 如果是修改意见，自动再发送一次 yes 进行确认，避免用户二次输入
            if user_text.strip().lower() != "yes":
                agent._compiled_graph.invoke(Command(resume="yes"), config=config)
        except Exception as e:
            print(f"恢复执行失败：{e}")
            return

        # 若仍停留在审核环节，继续下一轮；否则尝试输出结果
        results = agent.get_optimization_results()
        if results:
            print("\n--- Optimization Results (HITL) ---")
            print(results)
            break
        else:
            print("\n未获得最终结果，可能仍在审核，请再次输入（通常输入 yes 确认即可）")


def main():
    data_file = "apps/pandas-data-analyst-app/data/bike_sales_data.csv"
    user_instructions = "优化价格和数量参数以最大化总销售额"
    scenario = "Bike Sales Revenue Optimization"

    print("选择模式：\n  1) 自动运行\n  2) 手动输入一次反馈\n  3) 测试 Human-in-the-Loop")
    choice = input("输入 1/2/3 并回车: ").strip()

    if choice == "1":
        run_auto(data_file, user_instructions, scenario)
    elif choice == "2":
        run_manual_input(data_file, user_instructions, scenario)
    elif choice == "3":
        run_hitl(data_file, user_instructions, scenario)
    else:
        print("无效选择，默认执行自动模式。")
        run_auto(data_file, user_instructions, scenario)


if __name__ == "__main__":
    # 允许 Ctrl+C 中断退出
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
