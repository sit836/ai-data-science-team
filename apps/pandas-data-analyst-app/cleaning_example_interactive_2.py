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


def handle_missing_values_interactive(current_data):
    """处理缺失值的交互式流程"""
    print("\n" + "=" * 60)
    print("缺失值处理向导")
    print("=" * 60)

    # 检查缺失值
    missing_counts = current_data.isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if len(missing_columns) == 0:
        print("✅ 数据中没有缺失值！")
        return current_data

    # 打印缺失情况
    print("发现缺失值的列:")
    for col, count in missing_columns.items():
        percentage = (count / len(current_data)) * 100
        print(f"  - {col}: {count} 个缺失值 ({percentage:.2f}%)")

    # 询问缺失原因
    print("\n🔍 请帮助我了解这些缺失值的原因:")
    print("1. 数据收集时遗漏")
    print("2. 数据录入错误")
    print("3. 特定条件下的自然缺失（如未购买产品的客户没有购买记录）")
    print("4. 其他原因")

    missing_reason = input("\n请选择缺失值的主要原因（输入数字）或描述具体情况: ")

    # 协商解决方案
    print("\n💡 基于您的描述，我建议以下处理方案:")

    solutions = []
    for col in missing_columns.index:
        col_type = current_data[col].dtype
        missing_count = missing_counts[col]
        missing_percentage = (missing_count / len(current_data)) * 100

        if missing_percentage > 50:
            solution = f"删除列 '{col}' (缺失值超过50%)"
            solutions.append((col, solution, "delete_column"))
        elif col_type in ['int64', 'float64']:
            solution = f"数值列 '{col}' 使用中位数填充"
            solutions.append((col, solution, "median_impute"))
        else:
            solution = f"分类列 '{col}' 使用众数填充"
            solutions.append((col, solution, "mode_impute"))

    # 显示建议方案
    for i, (col, solution, _) in enumerate(solutions, 1):
        print(f"{i}. {solution}")

    print(f"{len(solutions) + 1}. 自定义处理方案")
    print(f"{len(solutions) + 2}. 跳过缺失值处理")

    # 获取用户选择
    try:
        choice = input("\n请选择处理方案（输入数字）: ")
        choice = int(choice)

        if choice == len(solutions) + 1:
            # 自定义方案
            custom_instruction = input("请输入自定义处理指令: ")
            return apply_custom_missing_value_solution(current_data, custom_instruction)
        elif choice == len(solutions) + 2:
            # 跳过
            print("⏭️ 跳过缺失值处理")
            return current_data
        elif 1 <= choice <= len(solutions):
            # 应用选择的方案
            col, _, action = solutions[choice - 1]
            return apply_missing_value_solution(current_data, col, action)
        else:
            print("❌ 无效选择，使用默认方案")
            return apply_default_missing_value_solution(current_data, missing_columns)

    except ValueError:
        print("❌ 无效输入，使用默认方案")
        return apply_default_missing_value_solution(current_data, missing_columns)


def apply_missing_value_solution(df, column, action):
    """应用特定的缺失值处理方案"""
    if action == "delete_column":
        print(f"🗑️ 删除列: {column}")
        return df.drop(columns=[column])
    elif action == "median_impute":
        median_val = df[column].median()
        print(f"🔢 使用中位数填充 {column}: {median_val}")
        df[column] = df[column].fillna(median_val)
        return df
    elif action == "mode_impute":
        mode_val = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
        print(f"🏷️ 使用众数填充 {column}: {mode_val}")
        df[column] = df[column].fillna(mode_val)
        return df
    else:
        return df


def apply_default_missing_value_solution(df, missing_columns):
    """应用默认的缺失值处理方案"""
    print("🔄 应用默认缺失值处理方案")

    for col in missing_columns.index:
        col_type = df[col].dtype
        missing_count = missing_columns[col]
        missing_percentage = (missing_count / len(df)) * 100

        if missing_percentage > 50:
            print(f"🗑️ 删除列 {col} (缺失值超过50%)")
            df = df.drop(columns=[col])
        elif col_type in ['int64', 'float64']:
            median_val = df[col].median()
            print(f"🔢 使用中位数填充 {col}: {median_val}")
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            print(f"🏷️ 使用众数填充 {col}: {mode_val}")
            df[col] = df[col].fillna(mode_val)

    return df


def apply_custom_missing_value_solution(df, instruction):
    """应用自定义的缺失值处理方案"""
    print(f"🛠️ 应用自定义方案: {instruction}")

    try:
        # 创建agent处理自定义指令
        agent = DataCleaningAgent(
            model=llm,
            log=LOG,
            log_path=LOG_PATH,
            human_in_the_loop=False,
        )

        # 执行清洗
        config = {"configurable": {"thread_id": "missing_values_custom"}}
        agent.invoke_agent(
            user_instructions=f"处理缺失值: {instruction}",
            data_raw=df,
            config=config
        )

        # 获取结果
        if agent.response and 'data_cleaned' in agent.response:
            cleaned_df = pd.DataFrame(agent.response['data_cleaned'])
            print("✅ 自定义缺失值处理完成")
            return cleaned_df
        else:
            print("❌ 自定义处理失败，使用默认方案")
            missing_counts = df.isnull().sum()
            missing_columns = missing_counts[missing_counts > 0]
            return apply_default_missing_value_solution(df, missing_columns)

    except Exception as e:
        print(f"❌ 自定义处理错误: {e}")
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        return apply_default_missing_value_solution(df, missing_columns)


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
        if current_data.isnull().sum().sum() > 0:
            missing_counts = current_data.isnull().sum()
            missing_columns = missing_counts[missing_counts > 0]
            print("缺失值分布:")
            for col, count in missing_columns.items():
                percentage = (count / len(current_data)) * 100
                print(f"  - {col}: {count} 个缺失值 ({percentage:.2f}%)")

        print(f"重复行数量: {current_data.duplicated().sum()}")

        # 提供选项菜单
        print("\n请选择操作:")
        print("1. 处理缺失值")
        print("2. 处理重复值")
        print("3. 处理异常值")
        print("4. 数据类型转换")
        print("5. 自定义清洗指令")
        print("6. 重置数据")
        print("7. 退出")

        choice = input("\n请输入选项数字: ")

        if choice == '7':
            break
        elif choice == '6':
            current_data = original_df.copy()
            print("✅ 数据已重置为原始数据")
            continue
        elif choice == '1':
            # 处理缺失值
            current_data = handle_missing_values_interactive(current_data)
            continue
        elif choice == '2':
            user_instructions = "删除完全重复的行"  # TODO
        elif choice == '3':
            user_instructions = "处理数值列中的异常值（3倍标准差以外）"  # TODO
        elif choice == '4':
            user_instructions = "自动检测并转换数据类型"  # TODO
        elif choice == '5':
            user_instructions = input("请输入自定义清洗指令: ")  # TODO
        else:
            print("❌ 无效选项，请重新选择")
            continue

        if choice in ['2', '3', '4', '5']:
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
