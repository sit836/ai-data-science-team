import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ai_data_science_team import DataCleaningAgent

load_dotenv()


class DataCleaningApp:
    """数据清洗应用程序主类"""

    def __init__(self):
        self.MODEL = "deepseek-chat"
        self.LOG = False
        self.LOG_PATH = os.path.join(os.getcwd(), "logs/")

        self.llm = ChatOpenAI(
            model=self.MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.,
            max_tokens=100
        )
        self.original_df = pd.read_csv("../data/bike_sales_data.csv")
        self.current_data = self.original_df.copy()
        self.missing_values_handled = False
        self.duplicates_handled = False

    class MissingValueHandler:
        """缺失值处理处理器"""

        def __init__(self, app):
            self.app = app

        def handle_interactive(self, current_data):
            """处理缺失值的交互式流程"""
            print("\n" + "=" * 60)
            print("缺失值处理向导")
            print("=" * 60)

            # 检查缺失值
            missing_counts = current_data.isnull().sum()
            missing_columns = missing_counts[missing_counts > 0]

            if len(missing_columns) == 0:
                print("✅ 数据中没有缺失值！")
                self.app.missing_values_handled = True
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

            while True:
                missing_reason = input("\n请选择缺失值的主要原因（输入数字）或描述具体情况: ")
                if missing_reason.strip():  # 确保输入不为空
                    break
                print("❌ 输入不能为空，请重新输入")

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
            while True:
                choice = input("\n请选择处理方案（输入数字）: ")
                try:
                    choice = int(choice)
                    if 1 <= choice <= len(solutions) + 2:
                        break
                    else:
                        print(f"❌ 请输入 1 到 {len(solutions) + 2} 之间的数字")
                except ValueError:
                    print("❌ 请输入有效的数字")

            if choice == len(solutions) + 1:
                # 自定义方案
                while True:
                    custom_instruction = input("请输入自定义处理指令: ")
                    if custom_instruction.strip():
                        break
                    print("❌ 指令不能为空，请重新输入")
                result = self.apply_custom_solution(current_data, custom_instruction)
            elif choice == len(solutions) + 2:
                # 跳过
                print("⏭️ 跳过缺失值处理")
                result = current_data
            else:
                # 应用选择的方案
                col, _, action = solutions[choice - 1]
                result = self.apply_solution(current_data, col, action)

            # 标记缺失值已处理
            self.app.missing_values_handled = True
            return result

        def apply_solution(self, df, column, action):
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

        def apply_default_solution(self, df, missing_columns):
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

        def apply_custom_solution(self, df, instruction):
            """应用自定义的缺失值处理方案"""
            print(f"🛠️ 应用自定义方案: {instruction}")

            try:
                # 创建agent处理自定义指令
                agent = DataCleaningAgent(
                    model=self.app.llm,
                    log=self.app.LOG,
                    log_path=self.app.LOG_PATH,
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
                    return self.apply_default_solution(df, missing_columns)

            except Exception as e:
                print(f"❌ 自定义处理错误: {e}")
                missing_counts = df.isnull().sum()
                missing_columns = missing_counts[missing_counts > 0]
                return self.apply_default_solution(df, missing_columns)

    class DuplicateHandler:
        """重复值处理处理器"""

        def __init__(self, app):
            self.app = app

        def handle_interactive(self, current_data):
            """处理重复值的交互式流程"""
            print("\n" + "=" * 60)
            print("重复值处理向导")
            print("=" * 60)

            # 检查重复行
            duplicate_count = current_data.duplicated().sum()

            if duplicate_count == 0:
                print("✅ 数据中没有重复行！")
                self.app.duplicates_handled = True
                return current_data

            print(f"发现 {duplicate_count} 个重复行")

            # 显示重复行示例
            print("\n重复行示例:")
            duplicates = current_data[current_data.duplicated(keep=False)]
            print(duplicates.head(5))

            # 协商解决方案
            print("\n💡 请选择处理重复值的方式:")
            print("1. 删除所有完全重复的行")
            print("2. 基于关键列删除重复行")
            print("3. 保留第一个出现的重复行，删除后续的")
            print("4. 保留最后一个出现的重复行，删除前面的")
            print("5. 自定义处理方案")

            while True:
                choice = input("\n请选择处理方案（输入数字）: ")
                try:
                    choice = int(choice)
                    if 1 <= choice <= 5:
                        break
                    else:
                        print("❌ 请输入 1 到 5 之间的数字")
                except ValueError:
                    print("❌ 请输入有效的数字")

            if choice == 1:
                result = self.remove_all_duplicates(current_data)
            elif choice == 2:
                while True:
                    key_columns = input("请输入关键列名（用逗号分隔）: ").split(',')
                    key_columns = [col.strip() for col in key_columns if col.strip()]
                    if key_columns:
                        # 验证列名是否存在
                        invalid_cols = [col for col in key_columns if col not in current_data.columns]
                        if not invalid_cols:
                            break
                        print(f"❌ 以下列名不存在: {invalid_cols}")
                    else:
                        print("❌ 请输入至少一个有效的列名")
                result = self.remove_duplicates_by_columns(current_data, key_columns)
            elif choice == 3:
                result = self.remove_duplicates_keep_first(current_data)
            elif choice == 4:
                result = self.remove_duplicates_keep_last(current_data)
            elif choice == 5:
                while True:
                    custom_instruction = input("请输入自定义处理指令: ")
                    if custom_instruction.strip():
                        break
                    print("❌ 指令不能为空，请重新输入")
                result = self.apply_custom_solution(current_data, custom_instruction)

            # 标记重复值已处理
            self.app.duplicates_handled = True
            return result

        def remove_all_duplicates(self, df):
            """删除所有完全重复的行"""
            print("🗑️ 删除所有完全重复的行")
            initial_count = len(df)
            df_cleaned = df.drop_duplicates()
            removed_count = initial_count - len(df_cleaned)
            print(f"删除了 {removed_count} 个重复行")
            return df_cleaned

        def remove_duplicates_by_columns(self, df, columns):
            """基于指定列删除重复行"""
            print(f"🗑️ 基于列 {columns} 删除重复行")
            initial_count = len(df)
            df_cleaned = df.drop_duplicates(subset=columns)
            removed_count = initial_count - len(df_cleaned)
            print(f"删除了 {removed_count} 个重复行")
            return df_cleaned

        def remove_duplicates_keep_first(self, df):
            """保留第一个出现的重复行"""
            print("🗑️ 删除重复行，保留第一个出现的")
            initial_count = len(df)
            df_cleaned = df.drop_duplicates(keep='first')
            removed_count = initial_count - len(df_cleaned)
            print(f"删除了 {removed_count} 个重复行")
            return df_cleaned

        def remove_duplicates_keep_last(self, df):
            """保留最后一个出现的重复行"""
            print("🗑️ 删除重复行，保留最后一个出现的")
            initial_count = len(df)
            df_cleaned = df.drop_duplicates(keep='last')
            removed_count = initial_count - len(df_cleaned)
            print(f"删除了 {removed_count} 个重复行")
            return df_cleaned

        def apply_custom_solution(self, df, instruction):
            """应用自定义的重复值处理方案"""
            print(f"🛠️ 应用自定义方案: {instruction}")

            try:
                agent = DataCleaningAgent(
                    model=self.app.llm,
                    log=self.app.LOG,
                    log_path=self.app.LOG_PATH,
                    human_in_the_loop=False,
                )

                config = {"configurable": {"thread_id": "duplicates_custom"}}
                agent.invoke_agent(
                    user_instructions=f"处理重复值: {instruction}",
                    data_raw=df,
                    config=config
                )

                if agent.response and 'data_cleaned' in agent.response:
                    cleaned_df = pd.DataFrame(agent.response['data_cleaned'])
                    print("✅ 自定义重复值处理完成")
                    return cleaned_df
                else:
                    print("❌ 自定义处理失败，使用默认方案")
                    return self.remove_all_duplicates(df)

            except Exception as e:
                print(f"❌ 自定义处理错误: {e}")
                return self.remove_all_duplicates(df)

    class OutlierHandler:
        """异常值处理处理器"""

        def __init__(self, app):
            self.app = app

        def handle_interactive(self, current_data):
            """处理异常值的交互式流程"""
            # 检查是否已经处理了缺失值和重复行
            if not self.app.missing_values_handled:
                print("❌ 请先处理缺失值！")
                return current_data

            if not self.app.duplicates_handled:
                print("❌ 请先处理重复行！")
                return current_data

            print("\n" + "=" * 60)
            print("异常值处理向导")
            print("=" * 60)

            # 检测数值列中的异常值
            outlier_columns = self.detect_outliers(current_data)

            if len(outlier_columns) == 0:
                print("✅ 数据中没有检测到明显的异常值！")
                return current_data

            # 打印异常值情况
            print("检测到异常值的数值列:")
            for col, info in outlier_columns.items():
                print(f"  - {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
                print(f"    正常值范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

            # 询问异常值原因
            print("\n🔍 请帮助我了解这些异常值的原因:")
            print("1. 数据录入错误")
            print("2. 测量误差")
            print("3. 真实但罕见的极端值")
            print("4. 数据处理错误")
            print("5. 其他原因")

            while True:
                outlier_reason = input("\n请选择异常值的主要原因（输入数字）或描述具体情况: ")
                if outlier_reason.strip():
                    break
                print("❌ 输入不能为空，请重新输入")

            # 协商解决方案
            print("\n💡 基于您的描述，我建议以下处理方案:")

            solutions = []
            for col, info in outlier_columns.items():
                if info['percentage'] < 5:  # 异常值比例小于5%
                    solution = f"保留异常值 '{col}' (比例较小，可能是真实数据)"
                    solutions.append((col, solution, "keep_outliers"))
                elif info['percentage'] > 20:  # 异常值比例大于20%
                    solution = f"转换数据 '{col}' (使用对数变换减少异常值影响)"
                    solutions.append((col, solution, "transform_data"))
                else:
                    solution = f"缩尾处理 '{col}' (将异常值替换为边界值)"
                    solutions.append((col, solution, "winsorize"))

            # 显示建议方案
            for i, (col, solution, _) in enumerate(solutions, 1):
                print(f"{i}. {solution}")

            print(f"{len(solutions) + 1}. 自定义处理方案")
            print(f"{len(solutions) + 2}. 跳过异常值处理")

            # 获取用户选择
            while True:
                choice = input("\n请选择处理方案（输入数字）: ")
                try:
                    choice = int(choice)
                    if 1 <= choice <= len(solutions) + 2:
                        break
                    else:
                        print(f"❌ 请输入 1 到 {len(solutions) + 2} 之间的数字")
                except ValueError:
                    print("❌ 请输入有效的数字")

            if choice == len(solutions) + 1:
                # 自定义方案
                while True:
                    custom_instruction = input("请输入自定义处理指令: ")
                    if custom_instruction.strip():
                        break
                    print("❌ 指令不能为空，请重新输入")
                return self.apply_custom_solution(current_data, custom_instruction)
            elif choice == len(solutions) + 2:
                # 跳过
                print("⏭️ 跳过异常值处理")
                return current_data
            else:
                # 应用选择的方案
                col, _, action = solutions[choice - 1]
                return self.apply_solution(current_data, col, action)

        def apply_solution(self, df, column, action):
            """应用特定的异常值处理方案"""
            if action == "keep_outliers":
                print(f"✅ 保留列 '{column}' 中的异常值")
                return df
            elif action == "transform_data":
                # 对数变换处理异常值
                print(f"📊 对列 '{column}' 进行对数变换")
                if df[column].min() > 0:  # 确保所有值大于0
                    df[column] = np.log1p(df[column])
                else:
                    # 如果有负值，先进行偏移
                    min_val = df[column].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        df[column] = np.log1p(df[column] + offset)
                return df
            elif action == "winsorize":
                # 缩尾处理
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                print(f"🔧 对列 '{column}' 进行缩尾处理，边界值: [{lower_bound:.2f}, {upper_bound:.2f}]")

                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                return df
            else:
                return df

        def apply_default_solution(self, df, outlier_columns):
            """应用默认的异常值处理方案"""
            print("🔄 应用默认异常值处理方案")

            for col, info in outlier_columns.items():
                if info['percentage'] < 5:
                    print(f"✅ 保留列 '{col}' 中的异常值 (比例较小)")
                elif info['percentage'] > 20:
                    print(f"📊 对列 '{col}' 进行对数变换 (异常值比例较高)")
                    if df[col].min() > 0:
                        df[col] = np.log1p(df[col])
                    else:
                        min_val = df[col].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            df[col] = np.log1p(df[col] + offset)
                else:
                    print(f"🔧 对列 '{col}' 进行缩尾处理")
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            return df

        def apply_custom_solution(self, df, instruction):
            """应用自定义的异常值处理方案"""
            print(f"🛠️ 应用自定义方案: {instruction}")

            try:
                # 创建agent处理自定义指令
                agent = DataCleaningAgent(
                    model=self.app.llm,
                    log=self.app.LOG,
                    log_path=self.app.LOG_PATH,
                    human_in_the_loop=False,
                )

                # 执行清洗
                config = {"configurable": {"thread_id": "outliers_custom"}}
                agent.invoke_agent(
                    user_instructions=f"处理异常值: {instruction}",
                    data_raw=df,
                    config=config
                )

                # 获取结果
                if agent.response and 'data_cleaned' in agent.response:
                    cleaned_df = pd.DataFrame(agent.response['data_cleaned'])
                    print("✅ 自定义异常值处理完成")
                    return cleaned_df
                else:
                    print("❌ 自定义处理失败，使用默认方案")
                    return self.apply_default_solution(df, self.detect_outliers(df))

            except Exception as e:
                print(f"❌ 自定义处理错误: {e}")
                return self.apply_default_solution(df, self.detect_outliers(df))

        def detect_outliers(self, df):
            """检测数据中的异常值"""
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            outlier_columns = {}

            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)

                if outlier_count > 0:
                    percentage = (outlier_count / len(df)) * 100
                    outlier_columns[col] = {
                        'count': outlier_count,
                        'percentage': percentage,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }

            return outlier_columns

    def interactive_cleaning_session(self):
        """交互式数据清洗会话"""
        thread_id = input("请输入会话ID（或按回车使用默认ID）: ") or "interactive_session_1"
        config = {"configurable": {"thread_id": thread_id}}

        # 初始化处理器
        missing_handler = self.MissingValueHandler(self)
        duplicate_handler = self.DuplicateHandler(self)
        outlier_handler = self.OutlierHandler(self)

        while True:
            print("\n" + "=" * 60)
            print("数据清洗交互会话")
            print("=" * 60)

            # 显示当前数据信息
            print(f"当前数据形状: {self.current_data.shape}")
            print("数据列:", list(self.current_data.columns))

            # 显示处理状态
            print(f"\n处理状态:")
            print(f"缺失值处理: {'✅' if self.missing_values_handled else '❌'}")
            print(f"重复值处理: {'✅' if self.duplicates_handled else '❌'}")

            print("\n前3行数据:")
            print(self.current_data.head(3))

            # 显示数据基本信息
            print(f"\n数据基本信息:")
            print(f"缺失值数量: {self.current_data.isnull().sum().sum()}")
            if self.current_data.isnull().sum().sum() > 0:
                missing_counts = self.current_data.isnull().sum()
                missing_columns = missing_counts[missing_counts > 0]
                print("缺失值分布:")
                for col, count in missing_columns.items():
                    percentage = (count / len(self.current_data)) * 100
                    print(f"  - {col}: {count} 个缺失值 ({percentage:.2f}%)")

            print(f"重复行数量: {self.current_data.duplicated().sum()}")

            # 提供选项菜单
            print("\n请选择操作:")
            print("1. 处理缺失值")
            print("2. 处理重复值")
            print("3. 处理异常值")
            print("4. 数据类型转换")
            print("5. 自定义清洗指令")
            print("6. 重置数据")
            print("7. 退出")

            while True:
                choice = input("\n请输入选项数字: ")
                if choice in ['1', '2', '3', '4', '5', '6', '7']:
                    break
                print("❌ 无效选项，请输入 1-7 之间的数字")

            if choice == '7':
                break
            elif choice == '6':
                self.current_data = self.original_df.copy()
                self.missing_values_handled = False
                self.duplicates_handled = False
                print("✅ 数据已重置为原始数据")
                continue
            elif choice == '1':
                # 处理缺失值
                self.current_data = missing_handler.handle_interactive(self.current_data)
                continue
            elif choice == '2':
                # 处理重复值
                self.current_data = duplicate_handler.handle_interactive(self.current_data)
                continue
            elif choice == '3':
                # 处理异常值（会检查前置条件）
                self.current_data = outlier_handler.handle_interactive(self.current_data)
                continue
            elif choice in ['4', '5']:
                # 检查前置条件
                if not self.missing_values_handled:
                    print("❌ 请先处理缺失值！")
                    continue
                if not self.duplicates_handled:
                    print("❌ 请先处理重复行！")
                    continue

                if choice == '4':
                    user_instructions = "自动检测并转换数据类型"
                else:
                    while True:
                        user_instructions = input("请输入自定义清洗指令: ")
                        if user_instructions.strip():
                            break
                        print("❌ 指令不能为空，请重新输入")

                try:
                    # 创建新的agent实例，使用当前数据
                    agent = DataCleaningAgent(
                        model=self.llm,
                        log=self.LOG,
                        log_path=self.LOG_PATH,
                        human_in_the_loop=True,
                    )

                    # 执行清洗
                    print(f"\n执行指令: {user_instructions}")
                    agent.invoke_agent(
                        user_instructions=user_instructions,
                        data_raw=self.current_data,
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

                        while True:
                            human_response = input("请输入您的响应（'yes' 继续或提供修改建议）: ")
                            if human_response.strip():
                                break
                            print("❌ 响应不能为空，请重新输入")

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
                            print(f"处理前形状: {self.current_data.shape}")
                            print(f"处理后形状: {cleaned_df.shape}")

                            # 更新当前数据为清洗后的数据
                            self.current_data = cleaned_df

                except Exception as e:
                    print(f"❌ 处理过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()

    def run(self):
        """运行应用程序"""
        print("选择模式:")
        print("1. 交互式清洗会话")
        print("2. 多步骤清洗管道")

        while True:
            choice = input("请输入选择 (1 或 2): ")
            if choice in ["1", "2"]:
                break
            print("❌ 无效选择，请输入 1 或 2")

        if choice == "1":
            self.interactive_cleaning_session()
        elif choice == "2":
            print("❌ 多步骤管道模式暂不可用，请使用交互式会话")
            self.interactive_cleaning_session()


# 运行应用程序
if __name__ == "__main__":
    app = DataCleaningApp()
    app.run()
