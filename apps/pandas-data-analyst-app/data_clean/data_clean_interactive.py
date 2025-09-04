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
        self.processing_steps = []  # 存储需要执行的处理步骤

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
                return current_data, True

            # 打印缺失情况
            print("发现缺失值的列:")
            for col, count in missing_columns.items():
                percentage = (count / len(current_data)) * 100
                print(f"  - {col}: {count} 个缺失值 ({percentage:.2f}%)")

            # 按数据类型分组处理缺失值
            numeric_cols = []
            categorical_cols = []
            high_missing_cols = []

            for col in missing_columns.index:
                col_type = current_data[col].dtype
                missing_percentage = (missing_counts[col] / len(current_data)) * 100

                if missing_percentage > 50:
                    high_missing_cols.append(col)
                elif col_type in ['int64', 'float64']:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

            # 询问缺失原因
            # TODO： 删掉写死的选项
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

            # 协商解决方案 - 按数据类型分组处理
            print("\n💡 基于您的描述，我建议以下处理方案:")

            solutions = []

            # 处理高缺失率列
            for col in high_missing_cols:
                solution = f"删除列 '{col}' (缺失值超过50%)"
                solutions.append((col, solution, "delete_column", "high_missing"))

            # 处理数值列
            if numeric_cols:
                numeric_solution = f"数值列 {numeric_cols} 使用中位数填充"
                solutions.append((numeric_cols, numeric_solution, "median_impute", "numeric"))

            # 处理分类列
            if categorical_cols:
                categorical_solution = f"分类列 {categorical_cols} 使用众数填充"
                solutions.append((categorical_cols, categorical_solution, "mode_impute", "categorical"))

            # 显示建议方案
            for i, (cols, solution, _, _) in enumerate(solutions, 1):
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
                cols, _, action, col_type = solutions[choice - 1]
                result = self.apply_solution(current_data, cols, action, col_type)

            return result, True

        def apply_solution(self, df, columns, action, col_type):
            """应用特定的缺失值处理方案"""
            if isinstance(columns, str):
                columns = [columns]

            if action == "delete_column":
                for col in columns:
                    print(f"🗑️ 删除列: {col}")
                return df.drop(columns=columns)
            elif action == "median_impute":
                for col in columns:
                    median_val = df[col].median()
                    print(f"🔢 使用中位数填充 {col}: {median_val}")
                    df[col] = df[col].fillna(median_val)
                return df
            elif action == "mode_impute":
                for col in columns:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    print(f"🏷️ 使用众数填充 {col}: {mode_val}")
                    df[col] = df[col].fillna(mode_val)
                return df
            else:
                return df

        def apply_default_solution(self, df, missing_columns):
            """应用默认的缺失值处理方案"""
            print("🔄 应用默认缺失值处理方案")

            # 按数据类型分组处理
            numeric_cols = []
            categorical_cols = []
            high_missing_cols = []

            for col in missing_columns.index:
                col_type = df[col].dtype
                missing_percentage = (missing_columns[col] / len(df)) * 100

                if missing_percentage > 50:
                    high_missing_cols.append(col)
                elif col_type in ['int64', 'float64']:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

            # 处理高缺失率列
            if high_missing_cols:
                print(f"🗑️ 删除列 {high_missing_cols} (缺失值超过50%)")
                df = df.drop(columns=high_missing_cols)

            # 处理数值列
            for col in numeric_cols:
                median_val = df[col].median()
                print(f"🔢 使用中位数填充 {col}: {median_val}")
                df[col] = df[col].fillna(median_val)

            # 处理分类列
            for col in categorical_cols:
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
                return current_data, True

            print(f"发现 {duplicate_count} 个重复行")

            # 显示重复行示例
            print("\n重复行示例:")
            duplicates = current_data[current_data.duplicated(keep=False)]
            print(duplicates.head(5))

            # 协商解决方案
            print("\n💡 请选择处理重复值的方式:")
            print("1. 保留第一个出现的重复行，删除后续的")
            print("2. 自定义处理方案")

            while True:
                choice = input("\n请选择处理方案（输入数字）: ")
                try:
                    choice = int(choice)
                    if 1 <= choice <= 2:
                        break
                    else:
                        print("❌ 请输入 1 到 2 之间的数字")
                except ValueError:
                    print("❌ 请输入有效的数字")

            if choice == 1:
                result = self.remove_duplicates_keep_first(current_data)
            elif choice == 2:
                while True:
                    custom_instruction = input("请输入自定义处理指令: ")
                    if custom_instruction.strip():
                        break
                    print("❌ 指令不能为空，请重新输入")
                result = self.apply_custom_solution(current_data, custom_instruction)

            return result, True

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
            print("\n" + "=" * 60)
            print("异常值处理向导")
            print("=" * 60)

            # 检测数值列中的异常值
            outlier_columns = self.detect_outliers(current_data)

            if len(outlier_columns) == 0:
                print("✅ 数据中没有检测到明显的异常值！")
                return current_data, True

            # 打印异常值情况
            print("检测到异常值的数值列:")
            for col, info in outlier_columns.items():
                print(f"  - {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
                print(f"    正常值范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

            # 询问异常值原因
            # print("\n🔍 请帮助我了解这些异常值的原因")
            # print("1. 数据录入错误")
            # print("2. 测量误差")
            # print("3. 真实但罕见的极端值")
            # print("4. 数据处理错误")
            # print("5. 其他原因")

            while True:
                outlier_reason = input("🔍 请帮助我了解这些异常值的原因: ")
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
                return current_data, True
            else:
                # 应用选择的方案
                col, _, action = solutions[choice - 1]
                return self.apply_solution(current_data, col, action), True

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

    def detect_data_issues(self):
        """检测数据中的所有问题"""
        issues = []

        # 检测缺失值
        missing_counts = self.current_data.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        if len(missing_columns) > 0:
            issues.append(('missing', f"发现 {len(missing_columns)} 列有缺失值"))

        # 检测重复值
        duplicate_count = self.current_data.duplicated().sum()
        if duplicate_count > 0:
            issues.append(('duplicate', f"发现 {duplicate_count} 个重复行"))

        # 检测异常值
        outlier_columns = self.detect_outliers(self.current_data)
        if len(outlier_columns) > 0:
            issues.append(('outlier', f"发现 {len(outlier_columns)} 列有异常值"))

        return issues

    def run_data_diagnosis(self):
        """运行数据诊断，检测所有问题"""
        print("\n" + "=" * 60)
        print("数据诊断报告")
        print("=" * 60)

        issues = self.detect_data_issues()

        if not issues:
            print("✅ 数据质量良好，未发现明显问题！")
            return []

        print("发现以下数据问题:")
        for i, (issue_type, description) in enumerate(issues, 1):
            print(f"{i}. {description}")

        return issues

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

            # 运行数据诊断
            issues = self.run_data_diagnosis()

            if not issues:
                # 如果没有发现问题，询问用户下一步操作
                print("\n✅ 数据质量良好，未发现明显问题！")
                print("\n您可以选择:")
                print("1. 执行自定义清洗指令")
                print("2. 重置数据到原始状态")
                print("3. 退出程序")

                while True:
                    choice = input("\n请选择下一步操作（输入数字）: ")
                    if choice == '1':
                        while True:
                            user_instructions = input("请输入自定义清洗指令: ")
                            if user_instructions.strip():
                                break
                            print("❌ 指令不能为空，请重新输入")
                        self.execute_custom_cleaning(user_instructions, config)
                        break
                    elif choice == '2':
                        self.current_data = self.original_df.copy()
                        print("✅ 数据已重置为原始数据")
                        break
                    elif choice == '3':
                        print("👋 感谢使用数据清洗工具，再见！")
                        return
                    else:
                        print("❌ 请输入有效的选项（1-3）")
                continue

            # 引导用户依次处理每个问题
            print(f"\n📋 共发现 {len(issues)} 个数据问题，我将引导您依次处理:")

            for i, (issue_type, description) in enumerate(issues, 1):
                print(f"\n{'=' * 50}")
                print(f"处理问题 {i}/{len(issues)}: {description}")
                print(f"{'=' * 50}")

                if issue_type == 'missing':
                    print("🔍 正在处理缺失值问题...")
                    self.current_data, _ = missing_handler.handle_interactive(self.current_data)
                elif issue_type == 'duplicate':
                    print("🔍 正在处理重复值问题...")
                    self.current_data, _ = duplicate_handler.handle_interactive(self.current_data)
                elif issue_type == 'outlier':
                    print("🔍 正在处理异常值问题...")
                    self.current_data, _ = outlier_handler.handle_interactive(self.current_data)

                # 显示处理后的数据状态
                print(f"\n✅ 当前数据形状: {self.current_data.shape}")

                # 询问是否继续处理下一个问题
                if i < len(issues):
                    while True:
                        continue_choice = input(f"\n是否继续处理下一个问题？(y/n): ").lower()
                        if continue_choice in ['y', 'yes', '是']:
                            break
                        elif continue_choice in ['n', 'no', '否']:
                            print("⏭️ 跳过剩余问题处理")
                            break
                        else:
                            print("❌ 请输入 y/n 或 是/否")

                    if continue_choice in ['n', 'no', '否']:
                        break

            # 所有问题处理完成后，提供额外选项
            print(f"\n{'=' * 60}")
            print("所有检测到的问题已处理完成！")
            print(f"{'=' * 60}")

            while True:
                print("\n请选择下一步操作:")
                print("1. 重新检测数据问题")
                print("2. 执行自定义清洗指令")
                print("3. 保存当前数据")
                print("4. 重置数据到原始状态")
                print("5. 退出程序")

                choice = input("\n请输入选项数字: ")

                if choice == '1':
                    break  # 重新开始循环，再次检测问题
                elif choice == '2':
                    while True:
                        user_instructions = input("请输入自定义清洗指令: ")
                        if user_instructions.strip():
                            break
                        print("❌ 指令不能为空，请重新输入")
                    self.execute_custom_cleaning(user_instructions, config)
                elif choice == '3':
                    output_path = input("请输入保存路径（默认: cleaned_data.csv）: ") or "cleaned_data.csv"
                    self.current_data.to_csv(output_path, index=False)
                    print(f"✅ 数据已保存到 {output_path}")
                elif choice == '4':
                    self.current_data = self.original_df.copy()
                    print("✅ 数据已重置为原始数据")
                    break  # 重新检测问题
                elif choice == '5':
                    print("👋 感谢使用数据清洗工具，再见！")
                    return
                else:
                    print("❌ 请输入有效的选项（1-5）")

    def execute_custom_cleaning(self, user_instructions, config):
        """执行自定义清洗指令"""
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

            if state and hasattr(state, 'next') and state.next:
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
        self.interactive_cleaning_session()


if __name__ == "__main__":
    app = DataCleaningApp()
    app.run()
