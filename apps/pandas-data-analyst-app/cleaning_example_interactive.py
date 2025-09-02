import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science_team import DataCleaningAgent

load_dotenv()

MODEL = "deepseek-chat"
LOG = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                 temperature=0., max_tokens=100)

# 初始数据
original_df = pd.read_csv("data/bike_sales_data.csv")


def interactive_cleaning_session():
    """交互式数据清洗会话"""
    thread_id = input("请输入会话ID（或按回车使用默认ID）: ") or "interactive_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    # 初始化当前数据为原始数据
    current_data = original_df.copy()

    while True:
        print("\n" + "=" * 60)
        print("数据清洗交互会话")
        print("=" * 60)

        # 显示当前数据信息
        print(f"当前数据形状: {current_data.shape}")
        print("数据列:", list(current_data.columns))
        print("\n前3行数据:")
        print(current_data.head(3))

        # 显示数据基本信息
        print(f"\n数据基本信息:")
        print(f"缺失值数量: {current_data.isnull().sum().sum()}")
        print(f"重复行数量: {current_data.duplicated().sum()}")

        # 获取用户指令
        user_instructions = input("\n请输入清洗指令（输入 'quit' 退出, 'reset' 重置数据）: ")
        if user_instructions.lower() == 'quit':
            break
        elif user_instructions.lower() == 'reset':
            current_data = original_df.copy()
            print("✅ 数据已重置为原始数据")
            continue

        if not user_instructions.strip():
            print("指令不能为空，请重新输入。")
            continue

        try:
            # 创建新的agent实例，使用当前数据
            agent = DataCleaningAgent(
                model=llm,
                log=LOG,
                log_path=LOG_PATH,
                human_in_the_loop=True,
            )

            # 执行清洗
            print(f"\n执行指令: {user_instructions}")
            agent.invoke_agent(
                user_instructions=user_instructions,
                data_raw=current_data,
                config=config
            )

            # 获取状态
            state = agent._compiled_graph.get_state(config=config)

            human_review_required = False
            if state and hasattr(state, 'next') and state.next:
                human_review_required = True
                print("\n🔍 系统需要人类审查:")

                # 尝试获取审查问题
                question = "请确认清洗指令是否正确？"
                if hasattr(state.next[0], 'tasks') and state.next[0].tasks:
                    if state.next[0].tasks[-1].interrupts:
                        question = state.next[0].tasks[-1].interrupts[-1].value
                elif hasattr(state.next[0], 'value'):
                    question = state.next[0].value

                print(f"问题: {question}")

                human_response = input("请输入您的响应（'yes' 继续或提供修改建议）: ")

                # 继续处理
                agent._compiled_graph.invoke(
                    {"resume": human_response},
                    config=config
                )

            # 获取最终结果
            final_state = agent._compiled_graph.get_state(config=config)

            if final_state and hasattr(final_state, 'values'):
                if 'data_cleaned' in final_state.values:
                    cleaned_df = pd.DataFrame(final_state.values['data_cleaned'])

                    print(f"\n✅ 清洗完成！")
                    print(f"处理前形状: {current_data.shape}")
                    print(f"处理后形状: {cleaned_df.shape}")

                    # 更新当前数据为清洗后的数据
                    current_data = cleaned_df

                    view_details = input("\n查看清洗详情？(y/n): ")
                    if view_details.lower() == 'y':
                        print("\n清洗后的前10行数据:")
                        print(current_data.head(10))

                        print("\n数据信息变化:")
                        print(f"缺失值: {original_df.isnull().sum().sum()} → {current_data.isnull().sum().sum()}")
                        print(f"行数: {original_df.shape[0]} → {current_data.shape[0]}")
                        print(f"列数: {original_df.shape[1]} → {current_data.shape[1]}")

                if 'recommended_steps' in final_state.values:
                    view_steps = input("\n查看推荐的清洗步骤？(y/n): ")
                    if view_steps.lower() == 'y':
                        print("\n推荐步骤:")
                        print(final_state.values['recommended_steps'])

                if 'data_cleaner_function' in final_state.values:
                    view_code = input("\n查看生成的代码？(y/n): ")
                    if view_code.lower() == 'y':
                        print("\n生成的清洗函数:")
                        print(final_state.values['data_cleaner_function'])

            # 询问是否继续
            if not human_review_required:
                continue_session = input("\n继续新的清洗指令？(y/n): ")
                if continue_session.lower() != 'y':
                    break

        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

            # 询问是否继续
            continue_session = input("\n是否继续？(y/n): ")
            if continue_session.lower() != 'y':
                break


def multi_step_cleaning_pipeline():
    """多步骤清洗管道示例"""
    config = {"configurable": {"thread_id": "pipeline_1"}}
    current_data = original_df.copy()

    # 定义清洗步骤
    cleaning_steps = [
        {"name": "处理缺失值", "instructions": "删除缺失值超过50%的列，数值列用中位数填充，分类列用众数填充"},
        {"name": "处理异常值", "instructions": "删除数值列中3倍标准差以外的异常值"},
        {"name": "数据类型转换", "instructions": "将日期列转换为datetime类型，确保数值列都是数字类型"},
        {"name": "删除重复行", "instructions": "删除完全重复的行"}
    ]

    print("开始多步骤数据清洗管道...")

    for i, step in enumerate(cleaning_steps, 1):
        print(f"\n步骤 {i}/{len(cleaning_steps)}: {step['name']}")
        print(f"指令: {step['instructions']}")
        print(f"当前数据形状: {current_data.shape}")

        try:
            agent = DataCleaningAgent(
                model=llm,
                log=LOG,
                log_path=LOG_PATH,
                human_in_the_loop=False,  # 管道模式关闭人工审查
            )

            agent.invoke_agent(
                user_instructions=step['instructions'],
                data_raw=current_data,
                config=config
            )

            if agent.response and 'data_cleaned' in agent.response:
                new_data = pd.DataFrame(agent.response['data_cleaned'])
                print(f"✅ {step['name']} 完成")
                print(f"数据形状变化: {current_data.shape} → {new_data.shape}")
                current_data = new_data
            else:
                print(f"⚠️  {step['name']} 未产生变化")

        except Exception as e:
            print(f"❌ {step['name']} 失败: {e}")
            break

    print(f"\n管道完成！最终数据形状: {current_data.shape}")

    # 保存最终结果
    save_option = input("是否保存清洗后的数据？(y/n): ")
    if save_option.lower() == 'y':
        filename = input("输入文件名（默认: cleaned_data.csv）: ") or "cleaned_data.csv"
        current_data.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")


# 运行交互式会话
if __name__ == "__main__":
    print("选择模式:")
    print("1. 交互式清洗会话")
    print("2. 多步骤清洗管道")

    choice = input("请输入选择 (1 或 2): ")

    if choice == "1":
        interactive_cleaning_session()
    elif choice == "2":
        multi_step_cleaning_pipeline()
    else:
        print("无效选择，运行交互式会话")
        interactive_cleaning_session()
