import os
from typing import Dict, List, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ai_data_science_team import DataCleaningAgent

load_dotenv()


class ConversationState(BaseModel):
    messages: List[Dict[str, str]] = []
    agreed: bool = False
    data: pd.DataFrame
    processing_type: str
    processing_details: Dict[str, Any]
    current_response: Optional[str] = None
    model_config = {
        "arbitrary_types_allowed": True
    }


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
            temperature=0.7,
            max_tokens=500
        )
        self.original_df = pd.read_csv("../data/bike_sales_data.csv")
        self.current_data = self.original_df.copy()
        self.processing_steps = []

    def run_conversation(self, initial_state: ConversationState) -> ConversationState:
        """运行对话直到用户同意或退出"""
        state = initial_state

        while not state.agreed:
            # 显示AI的响应
            if state.current_response:
                print(f"\n🤖 {state.current_response}")

            # 获取用户输入
            while True:
                user_input = input("\n💬 您有什么疑问吗？如果没有疑问，请输入'继续'开始处理: ")
                if user_input.strip():
                    break
                print("❌ 输入不能为空，请重新输入")

            # 使用语言模型判断用户意图
            intent = self.classify_user_intent(user_input, state.processing_type)

            if intent == "agree":
                state.agreed = True
                break
            elif intent == "disagree":
                print("❌ 您不同意当前方案，将取消处理")
                state.agreed = False
                break

            # 用户有问题，生成回答
            state.messages.append({"role": "user", "content": user_input})

            # 根据处理类型构建系统提示
            if state.processing_type == "missing":
                system_prompt = """你是一个数据清洗专家，正在帮助用户处理缺失值问题。请友好、专业地回答用户的问题。

    处理方案详情：
    - 高缺失率列(>50%): 建议删除
    - 数值列: 建议中位数填充（中位数是将数据排序后位于中间的值，对异常值不敏感）
    - 分类列: 建议众数填充（众数是出现频率最高的值）

    请用中文回答，保持专业但友好的语气。解释清楚为什么选择这种方法，以及它的优缺点。"""
            else:  # outlier
                system_prompt = """你是一个数据清洗专家，正在帮助用户处理异常值问题。请友好、专业地回答用户的问题。

    处理方案详情：
    - 异常值比例<5%: 建议保留（可能是真实的重要数据）
    - 异常值比例>20%: 建议对数变换（压缩极端大值，使分布更正态）
    - 其他情况: 建议缩尾处理（将异常值替换为边界值，保留数据点）

    请用中文回答，保持专业但友好的语气。解释清楚为什么选择这种方法，以及它的优缺点。"""

            # 构建对话上下文
            conversation_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in state.messages[-4:]  # 最近4条消息作为上下文
            ])

            prompt = f"""{system_prompt}

    当前对话上下文：
    {conversation_context}

    请生成友好、专业的回答，帮助用户理解处理方案："""

            try:
                response = self.llm.invoke(prompt)
                ai_response = response.content
            except Exception as e:
                ai_response = "抱歉，我遇到了一些技术问题。中位数填充是将缺失值用该列数据的中位数（排序后的中间值）来填充，这种方法对异常值不敏感，比平均值更稳健。"

            state.messages.append({"role": "assistant", "content": ai_response})
            state.current_response = ai_response

            # 显示回答后继续循环，而不是直接返回
            print(f"\n🤖 {ai_response}")

        return state

    def classify_user_intent(self, user_input: str, processing_type: str) -> str:
        """使用语言模型判断用户意图"""
        intent_prompt = f"""请分析用户的输入意图，判断用户是否同意当前的数据处理方案。

    用户输入: "{user_input}"

    当前处理类型: {processing_type}

    请从以下选项中选择最匹配的意图：
    1. agree - 用户明确表示同意、确认或要求继续处理
    2. disagree - 用户明确表示不同意、拒绝或要求取消处理
    3. question - 用户提出问题或需要更多解释

    请只返回意图关键词（agree/disagree/question），不要返回其他内容。"""

        try:
            response = self.llm.invoke(intent_prompt)
            intent = response.content.strip().lower()

            # 确保返回的是有效的意图
            if intent in ["agree", "disagree", "question"]:
                return intent
            else:
                # 如果模型返回了其他内容，使用备用逻辑
                return self.fallback_intent_detection(user_input)

        except Exception as e:
            print(f"❌ 意图识别失败，使用备用方法: {e}")
            return self.fallback_intent_detection(user_input)

    def fallback_intent_detection(self, user_input: str) -> str:
        """备用意图检测方法（当语言模型失败时使用）"""
        user_input_lower = user_input.lower()

        # 同意意图的关键词
        agree_keywords = ["继续", "开始", "同意", "好的", "没问题", "ok", "yes", "y", "是", "go", "start", "确认",
                          "执行", "处理"]

        # 拒绝意图的关键词
        disagree_keywords = ["不", "不要", "取消", "停止", "退出", "no", "n", "拒绝", "不同意", "算了"]

        # 检查是否包含同意关键词
        if any(keyword in user_input_lower for keyword in agree_keywords):
            return "agree"

        # 检查是否包含拒绝关键词
        if any(keyword in user_input_lower for keyword in disagree_keywords):
            return "disagree"

        # 默认认为是问题
        return "question"

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

            while True:
                missing_reason = input("🔍 请帮助我了解这些缺失值的原因: ")
                if missing_reason.strip():
                    break
                print("❌ 输入不能为空，请重新输入")

            # 构建处理方案
            suggested_solution_parts = []
            if high_missing_cols:
                suggested_solution_parts.append(f"删除列 {high_missing_cols} (缺失值超过50%)")
            if numeric_cols:
                suggested_solution_parts.append(f"数值列 {numeric_cols} 使用中位数填充")
            if categorical_cols:
                suggested_solution_parts.append(f"分类列 {categorical_cols} 使用众数填充")

            suggested_solution = "，".join(suggested_solution_parts) + "。"
            print(f"\n💡 基于您的描述，我建议以下处理方案:\n{suggested_solution}")

            # 初始化对话状态
            initial_state = ConversationState(
                messages=[
                    {"role": "assistant", "content": f"我建议的数据处理方案是：{suggested_solution}"}
                ],
                agreed=False,
                data=current_data,
                processing_type="missing",
                processing_details={
                    "high_missing_cols": high_missing_cols,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                },
                current_response=f"我建议的数据处理方案是：{suggested_solution}\n\n您对这个方案有什么疑问吗？我会详细解释每个处理方法的原理和优势。"
            )

            # 运行对话
            print("\n🔍 您可以询问关于处理方案的任何问题，我会为您详细解释。")
            final_state = self.app.run_conversation(initial_state)

            if final_state.agreed:
                print("\n✅ 开始执行缺失值处理...")
                result = self.apply_suggested_solution(final_state.data,
                                                       high_missing_cols,
                                                       numeric_cols,
                                                       categorical_cols)
                return result, True
            else:
                print("⏭️ 用户取消处理，跳过缺失值处理")
                return current_data, True

        def apply_suggested_solution(self, df, high_missing_cols, numeric_cols, categorical_cols):
            """应用建议的缺失值处理方案"""
            print("🔄 应用建议的缺失值处理方案")

            # 处理高缺失率列
            if high_missing_cols:
                print(f"🗑️ 删除列 {high_missing_cols} (缺失值超过50%)")
                df = df.drop(columns=high_missing_cols)

            # 处理数值列
            for col in numeric_cols:
                if col in df.columns:
                    median_val = df[col].median()
                    print(f"🔢 使用中位数填充 {col}: {median_val}")
                    df[col] = df[col].fillna(median_val)

            # 处理分类列
            for col in categorical_cols:
                if col in df.columns:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    print(f"🏷️ 使用众数填充 {col}: {mode_val}")
                    df[col] = df[col].fillna(mode_val)

            print("✅ 缺失值处理完成！")
            return df

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

            # 构建处理方案
            suggested_solution = f"删除所有重复行，保留第一个出现的重复值。这将删除 {duplicate_count} 个重复行。"
            print(f"\n💡 我建议的处理方案:\n{suggested_solution}")

            # 初始化对话状态
            initial_state = ConversationState(
                messages=[
                    {"role": "assistant", "content": f"我建议的重复值处理方案是：{suggested_solution}"}
                ],
                agreed=False,
                data=current_data,
                processing_type="duplicate",
                processing_details={
                    "duplicate_count": duplicate_count,
                    "suggested_method": "keep_first"
                },
                current_response=f"我建议的重复值处理方案是：{suggested_solution}\n\n您对这个方案有什么疑问吗？我会详细解释处理方法的原理和优势。"
            )

            # 运行对话
            print("\n🔍 您可以询问关于处理方案的任何问题，我会为您详细解释。")
            final_state = self.app.run_conversation(initial_state)

            if final_state.agreed:
                print("\n✅ 开始执行重复值处理...")
                result = self.apply_suggested_solution(final_state.data)
                return result, True
            else:
                print("⏭️ 用户取消处理，跳过重复值处理")
                return current_data, True

        def apply_suggested_solution(self, df):
            """应用建议的重复值处理方案"""
            print("🔄 应用建议的重复值处理方案")

            initial_count = len(df)
            df_cleaned = df.drop_duplicates(keep='first')
            removed_count = initial_count - len(df_cleaned)

            print(f"🗑️ 删除重复行，保留第一个出现的")
            print(f"删除了 {removed_count} 个重复行")
            print("✅ 重复值处理完成！")

            return df_cleaned

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
                print("\n✅ 数据质量良好，未发现明显问题！")
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

                # 显示处理后的数据状态
                print(f"\n✅ 当前数据形状: {self.current_data.shape}")

                # 自动继续处理下一个问题，不再询问
                if i < len(issues):
                    print("⏭️ 自动继续处理下一个问题...")

            # 所有问题处理完成后，提供额外选项
            print(f"\n{'=' * 60}")
            print("所有检测到的问题已处理完成！")
            print(f"{'=' * 60}")

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
