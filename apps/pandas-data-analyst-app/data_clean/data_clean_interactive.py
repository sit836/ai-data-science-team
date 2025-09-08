import os
from typing import Dict, List, Any, Optional
import re
import time

import pandas as pd
import numpy as np
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
    custom_solution_proposed: bool = False
    custom_solution: Optional[str] = None
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
            max_tokens=500,
            timeout=10  # 较短的超时时间
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

            # 使用LLM判断用户意图（带重试和降级处理）
            intent, custom_solution = self.llm_intent_detection_with_fallback(user_input, state.processing_type)

            if intent == "agree":
                state.agreed = True
                break
            elif intent == "disagree":
                print("❌ 您不同意当前方案，将取消处理")
                state.agreed = False
                break
            elif intent == "custom_solution":
                # 用户提出了自定义方案
                state.custom_solution_proposed = True
                state.custom_solution = custom_solution or user_input

                # 使用LLM评估自定义方案的合理性
                is_reasonable, feedback = self.llm_evaluate_solution_with_fallback(
                    state.custom_solution, state.processing_type, state.processing_details
                )

                if is_reasonable:
                    print(f"\n✅ {feedback}")
                    confirmation = input("是否确认执行此方案？(yes/no): ").lower().strip()
                    if confirmation in ['yes', 'y', '是', '确认']:
                        state.agreed = True
                        break
                    else:
                        print("请重新考虑您的方案或提出新的方案")
                        state.current_response = "您的方案看起来合理，但您选择了不执行。请提出新的方案或继续讨论原方案。"
                else:
                    # 即使方案不合理，也询问用户是否坚持
                    print(f"\n⚠️ {feedback}")
                    confirmation = input("您仍然确定要执行此方案吗？(yes/no): ").lower().strip()
                    if confirmation in ['yes', 'y', '是', '确认']:
                        print("✅ 尊重您的选择，将执行您的方案")
                        state.agreed = True
                        break
                    else:
                        print("请修改您的方案或继续讨论原方案")
                        state.current_response = f"关于您的方案: {feedback}\n请考虑修改或继续讨论原方案。"

                continue

            # 用户有问题，使用LLM生成回答
            state.messages.append({"role": "user", "content": user_input})

            ai_response = self.llm_generate_response_with_fallback(
                user_input, state.processing_type, state.messages, state.processing_details
            )

            state.messages.append({"role": "assistant", "content": ai_response})
            state.current_response = ai_response

        return state

    def llm_intent_detection_with_fallback(self, user_input: str, processing_type: str, max_retries: int = 2) -> tuple:
        """使用LLM判断用户意图，带重试和降级处理"""
        for attempt in range(max_retries):
            try:
                intent, custom_solution = self.llm_classify_intent(user_input, processing_type)
                return intent, custom_solution
            except Exception as e:
                print(f"❌ 意图识别尝试 {attempt + 1} 失败: {e}")
                time.sleep(1)  # 短暂等待后重试

        # 所有重试都失败，使用降级逻辑
        print("⚠️ LLM不可用，使用备用意图检测")
        return self.fallback_intent_detection(user_input), user_input

    def llm_classify_intent(self, user_input: str, processing_type: str) -> tuple:
        """使用LLM判断用户意图"""
        intent_prompt = f"""作为数据清洗助手，请分析用户的输入意图。

用户输入: "{user_input}"
当前处理类型: {processing_type}

请判断用户的意图：
1. agree - 用户同意当前方案，要求继续处理
2. disagree - 用户不同意当前方案，要求取消处理  
3. question - 用户提出问题或需要解释
4. custom_solution - 用户提出了自定义的处理方案

如果是自定义方案，请提取方案的核心内容。

请返回格式：意图|方案内容（如适用）

示例：
agree|
question|
custom_solution|使用平均值填充缺失值"""

        try:
            response = self.llm.invoke(intent_prompt)
            result = response.content.strip()

            if "|" in result:
                intent, custom_solution = result.split("|", 1)
                intent = intent.strip().lower()
                custom_solution = custom_solution.strip()

                # 验证意图是否有效
                valid_intents = ["agree", "disagree", "question", "custom_solution"]
                if intent in valid_intents:
                    return intent, custom_solution

            # 如果格式不正确，抛出异常触发重试
            raise ValueError("LLM返回格式不正确")

        except Exception as e:
            print(f"❌ LLM意图识别错误: {e}")
            raise

    def llm_evaluate_solution_with_fallback(self, custom_solution: str, processing_type: str, processing_details: Dict,
                                            max_retries: int = 2) -> tuple:
        """使用LLM评估方案合理性，带重试"""
        for attempt in range(max_retries):
            try:
                return self.llm_evaluate_solution(custom_solution, processing_type, processing_details)
            except Exception as e:
                print(f"❌ 方案评估尝试 {attempt + 1} 失败: {e}")
                time.sleep(1)

        # 评估失败时的降级处理
        print("⚠️ LLM评估不可用，使用默认评估")
        return True, "无法进行评估，将尊重您的选择执行方案"

    def llm_evaluate_solution(self, custom_solution: str, processing_type: str, processing_details: Dict) -> tuple:
        """使用LLM评估自定义方案的合理性"""
        evaluation_prompt = f"""作为数据清洗专家，请评估用户提出的处理方案。

处理类型: {processing_type}
处理详情: {processing_details}
用户方案: "{custom_solution}"

请评估：
1. 方案的技术合理性（是否符合数据清洗最佳实践）
2. 潜在的风险或问题
3. 改进建议（如有）

请用友好、专业的态度回复，返回格式：合理与否(true/false)|评估反馈"""

        try:
            response = self.llm.invoke(evaluation_prompt)
            result = response.content.strip()

            if "|" in result:
                is_reasonable_str, feedback = result.split("|", 1)
                is_reasonable = is_reasonable_str.strip().lower() == "true"
                return is_reasonable, feedback.strip()

            raise ValueError("LLM评估返回格式不正确")

        except Exception as e:
            print(f"❌ LLM方案评估错误: {e}")
            raise

    def llm_generate_response_with_fallback(self, user_input: str, processing_type: str, messages: List[Dict],
                                            processing_details: Dict, max_retries: int = 2) -> str:
        """使用LLM生成回答，带重试"""
        for attempt in range(max_retries):
            try:
                return self.llm_generate_response(user_input, processing_type, messages, processing_details)
            except Exception as e:
                print(f"❌ 回答生成尝试 {attempt + 1} 失败: {e}")
                time.sleep(1)

        # 生成回答失败时的降级处理
        return self.get_fallback_response(processing_type, user_input, processing_details)

    def llm_generate_response(self, user_input: str, processing_type: str, messages: List[Dict],
                              processing_details: Dict) -> str:
        """使用LLM生成专业回答"""
        # 构建对话上下文
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in messages[-3:]  # 最近3条消息
        ])

        system_prompt = self.get_system_prompt(processing_type, processing_details)

        prompt = f"""{system_prompt}

当前对话上下文：
{conversation_context}

用户最新问题: "{user_input}"

请生成友好、专业的回答，帮助用户理解处理方案："""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ LLM回答生成错误: {e}")
            raise

    def get_system_prompt(self, processing_type: str, processing_details: Dict) -> str:
        """获取系统提示"""
        prompts = {
            "missing": f"""你是一个数据清洗专家，正在帮助用户处理缺失值问题。

当前缺失值情况：
{processing_details}

请用中文友好、专业地回答用户的问题，解释清楚处理方法的原理和优势。""",

            "duplicate": f"""你是一个数据清洗专家，正在帮助用户处理重复值问题。

当前重复值情况：
{processing_details}

请用中文友好、专业地回答用户的问题。""",

            "outlier": f"""你是一个数据清洗专家，正在帮助用户处理异常值问题。

当前异常值情况：
{processing_details}

请用中文友好、专业地回答用户的问题。"""
        }
        return prompts.get(processing_type, "你是一个数据清洗专家，请用中文友好、专业地回答用户的问题。")

    def fallback_intent_detection(self, user_input: str) -> str:
        """备用意图检测方法"""
        user_input_lower = user_input.lower().strip()

        # 简单的关键词匹配（仅在LLM不可用时使用）
        agree_keywords = ['继续', '开始', '同意', '好的', '没问题', 'ok', 'yes', 'y', '是']
        disagree_keywords = ['不', '不要', '取消', '停止', '退出', 'no', 'n', '拒绝', '不同意']
        custom_keywords = ['用', '填充', '删除', '保留', '处理', '方案']

        if any(keyword in user_input_lower for keyword in agree_keywords):
            return "agree"
        elif any(keyword in user_input_lower for keyword in disagree_keywords):
            return "disagree"
        elif any(keyword in user_input_lower for keyword in custom_keywords):
            return "custom_solution"
        else:
            return "question"

    def get_fallback_response(self, processing_type: str, user_input: str, processing_details: Dict) -> str:
        """获取备用回答"""
        responses = {
            "missing": f"关于您的输入'{user_input}'，在缺失值处理中，我们通常根据数据类型和缺失比例选择不同的填充方法。",
            "duplicate": f"关于您的输入'{user_input}'，在重复值处理中，我们建议删除完全重复的行。",
            "outlier": f"关于您的输入'{user_input}'，在异常值处理中，需要根据异常值的比例选择不同的处理方法。"
        }
        return responses.get(processing_type, f"关于您的输入'{user_input}'，我会尽力帮助您。")

    # MissingValueHandler 类（使用LLM进行意图判断）
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

            # 构建处理方案
            suggested_solution_parts = []
            if high_missing_cols:
                suggested_solution_parts.append(f"删除列 {high_missing_cols} (缺失值超过50%)")
            if numeric_cols:
                suggested_solution_parts.append(f"数值列 {numeric_cols} 使用中位数填充")
            if categorical_cols:
                suggested_solution_parts.append(f"分类列 {categorical_cols} 使用众数填充")

            suggested_solution = "，".join(suggested_solution_parts) + "。"
            print(f"\n💡 基于分析，我建议以下处理方案:\n{suggested_solution}")

            # 初始化对话状态
            initial_state = ConversationState(
                messages=[],
                agreed=False,
                data=current_data,
                processing_type="missing",
                processing_details={
                    "high_missing_cols": high_missing_cols,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "suggested_solution": suggested_solution
                },
                current_response=f"建议的数据处理方案是：{suggested_solution}\n\n您对这个方案有什么疑问吗？或者您有自定义的处理方案？"
            )

            # 运行对话
            final_state = self.app.run_conversation(initial_state)

            if final_state.agreed:
                if final_state.custom_solution_proposed and final_state.custom_solution:
                    print(f"\n✅ 开始执行您的自定义方案: {final_state.custom_solution}")
                    result = self.apply_custom_solution(final_state.data, final_state.custom_solution,
                                                        final_state.processing_details)
                else:
                    print("\n✅ 开始执行建议的缺失值处理...")
                    result = self.apply_suggested_solution(final_state.data,
                                                           high_missing_cols,
                                                           numeric_cols,
                                                           categorical_cols)
                return result, True
            else:
                print("⏭️ 用户取消处理，跳过缺失值处理")
                return current_data, True

        def apply_custom_solution(self, df, custom_solution, processing_details):
            """应用用户自定义的缺失值处理方案"""
            print(f"🔄 应用自定义方案: {custom_solution}")

            # 使用LLM解析和执行自定义方案
            try:
                execution_prompt = f"""请解析并执行以下缺失值处理方案：

数据集信息:
- 形状: {df.shape}
- 列名: {list(df.columns)}
- 缺失列详情: {processing_details}

自定义方案: "{custom_solution}"

请生成Python代码来执行这个方案，只返回代码部分："""

                response = self.app.llm.invoke(execution_prompt)
                code = response.content.strip()

                # 安全地执行代码
                local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
                exec(code, {}, local_vars)

                result_df = local_vars['df']
                print("✅ 自定义方案执行成功！")
                return result_df

            except Exception as e:
                print(f"❌ 自定义方案执行失败: {e}")
                print("⚠️ 使用建议方案代替")
                return self.apply_suggested_solution(
                    df,
                    processing_details['high_missing_cols'],
                    processing_details['numeric_cols'],
                    processing_details['categorical_cols']
                )

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
            print(duplicates.head(3).to_string())

            # 构建处理方案
            suggested_solution = f"删除所有重复行，保留第一个出现的重复值。这将删除 {duplicate_count} 个重复行。"
            print(f"\n💡 我建议的处理方案:\n{suggested_solution}")

            # 初始化对话状态
            initial_state = ConversationState(
                messages=[],
                agreed=False,
                data=current_data,
                processing_type="duplicate",
                processing_details={
                    "duplicate_count": duplicate_count,
                    "duplicate_sample": duplicates.head(3).to_dict(),
                    "suggested_solution": suggested_solution
                },
                current_response=f"建议的数据处理方案是：{suggested_solution}\n\n您对这个方案有什么疑问吗？或者您有自定义的处理方案？"
            )

            # 运行对话
            final_state = self.run_conversation(initial_state)

            if final_state.agreed:
                if final_state.custom_solution_proposed and final_state.custom_solution:
                    print(f"\n✅ 开始执行您的自定义方案: {final_state.custom_solution}")
                    result = self.apply_custom_solution(final_state.data, final_state.custom_solution,
                                                        final_state.processing_details)
                else:
                    print("\n✅ 开始执行重复值处理...")
                    result = self.apply_suggested_solution(final_state.data)
                return result, True
            else:
                print("⏭️ 用户取消处理，跳过重复值处理")
                return current_data, True

        def run_conversation(self, initial_state: ConversationState) -> ConversationState:
            """运行对话直到用户同意或退出 - 专门处理重复值场景"""
            state = initial_state
            educated_about_duplicates = False  # 标记是否已经教育过用户

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

                # 使用LLM判断用户意图
                intent, custom_solution = self.llm_intent_detection_with_fallback(user_input, state.processing_type)

                if intent == "agree":
                    state.agreed = True
                    break
                elif intent == "disagree":
                    print("❌ 您不同意当前方案，将取消处理")
                    state.agreed = False
                    break
                elif intent == "custom_solution":
                    # 检查是否是保留重复的方案
                    if self.is_keep_duplicates_intent(custom_solution or user_input):
                        if not educated_about_duplicates:
                            # 第一次提出保留重复，教育用户
                            self.educate_user_about_duplicates()
                            educated_about_duplicates = True
                            # 返回初始对话格式
                            state.current_response = f"建议的数据处理方案是：{state.processing_details['suggested_solution']}\n\n您对这个方案有什么疑问吗？或者您有自定义的处理方案？"
                            continue
                        else:
                            # 用户已经受过教育但仍然坚持保留重复
                            print("⚠️ 您坚持保留重复行，但这会导致严重的技术问题。")
                            print("🔧 我们将为您添加一个标识列来标记重复行，而不是完全保留重复数据。")

                            # 创建一个合理的替代方案
                            alternative_solution = "添加重复标识列并保留所有数据"
                            state.custom_solution_proposed = True
                            state.custom_solution = alternative_solution
                            state.agreed = True
                            break

                    # 用户提出了其他自定义方案
                    state.custom_solution_proposed = True
                    state.custom_solution = custom_solution or user_input

                    # 使用LLM评估自定义方案的合理性
                    is_reasonable, feedback = self.llm_evaluate_solution_with_fallback(
                        state.custom_solution, state.processing_type, state.processing_details
                    )

                    if is_reasonable:
                        print(f"\n✅ {feedback}")
                        confirmation = input("是否确认执行此方案？(yes/no): ").lower().strip()
                        if confirmation in ['yes', 'y', '是', '确认']:
                            state.agreed = True
                            break
                        else:
                            print("请重新考虑您的方案或提出新的方案")
                            state.current_response = f"建议的数据处理方案是：{state.processing_details['suggested_solution']}\n\n您对这个方案有什么疑问吗？或者您有自定义的处理方案？"
                    else:
                        print(f"\n⚠️ {feedback}")
                        state.current_response = f"建议的数据处理方案是：{state.processing_details['suggested_solution']}\n\n关于您的方案: {feedback}\n请考虑修改或提出其他处理方案。"

                    continue

                # 用户有问题，使用LLM生成回答
                state.messages.append({"role": "user", "content": user_input})

                ai_response = self.llm_generate_response_with_fallback(
                    user_input, state.processing_type, state.messages, state.processing_details
                )

                state.messages.append({"role": "assistant", "content": ai_response})
                state.current_response = ai_response

            return state

        def is_keep_duplicates_intent(self, user_input: str) -> bool:
            """判断用户是否想要保留重复行"""
            user_input_lower = user_input.lower()
            keep_keywords = [
                '保留所有', '都保留', '两行都要', '不删除', '保留重复', '全部保留', '不要删除',
                'keep all', 'keep both', 'don\'t delete', 'not remove', '保留全部'
            ]
            return any(keyword in user_input_lower for keyword in keep_keywords)

        def educate_user_about_duplicates(self):
            """教育用户关于重复行的问题"""
            education_message = """\n🚨 重要技术说明：为什么必须处理重复行？

    从机器学习和贝叶斯优化的角度，重复行会导致严重的数值稳定性问题：

    1. **贝叶斯优化器内在设计限制**：
       - 重复数据会使协方差矩阵奇异（singular），导致矩阵求逆失败
       - 高斯过程回归中的核函数计算会出现数值不稳定
       - 后验分布计算可能产生错误或无法收敛

    2. **数值稳定性问题**：
       - 重复行会导致特征矩阵的条件数恶化，影响数值精度
       - 梯度计算会出现数值误差累积，影响优化效果
       - 正则化项可能无法有效防止过拟合

    3. **统计偏差和过拟合**：
       - 重复数据扭曲真实的数据分布，导致模型偏见
       - 模型会过度拟合重复的模式，降低泛化能力
       - 评估指标会产生误导性结果，影响模型选择

    基于以上技术原因，完全保留重复行是不可行的。"""

            print(education_message)

        def llm_intent_detection_with_fallback(self, user_input: str, processing_type: str,
                                               max_retries: int = 2) -> tuple:
            """使用LLM判断用户意图，带重试和降级处理"""
            for attempt in range(max_retries):
                try:
                    intent, custom_solution = self.llm_classify_intent(user_input, processing_type)
                    return intent, custom_solution
                except Exception as e:
                    print(f"❌ 意图识别尝试 {attempt + 1} 失败: {e}")
                    time.sleep(1)  # 短暂等待后重试

            # 所有重试都失败，使用降级逻辑
            print("⚠️ LLM不可用，使用备用意图检测")
            return self.fallback_intent_detection(user_input), user_input

        def llm_classify_intent(self, user_input: str, processing_type: str) -> tuple:
            """使用LLM判断用户意图 - 优化版本"""
            intent_prompt = f"""请严格分析用户意图，只能返回以下4种意图之一：

        用户输入: "{user_input}"
        处理场景: 重复值处理

        可选意图：
        1. agree - 用户明确同意当前方案（包含：继续、同意、好的、没问题、行、可以）
        2. disagree - 用户明确拒绝当前方案（包含：不、不要、取消、停止、退出、拒绝）
        3. question - 用户提出问题或需要解释（包含：为什么、怎么、如何、解释、说明、什么）
        4. custom_solution - 用户提出了自定义处理方案（包含：用、改成、建议、自定义、我想、我要）

        判断规则：
        - 如果用户只是简单确认，返回agree
        - 如果用户明确拒绝，返回disagree  
        - 如果用户询问原因或方法，返回question
        - 如果用户提出具体处理方式，返回custom_solution

        返回格式：意图|方案内容（只有custom_solution时才需要方案内容）

        示例：
        用户: "继续" -> agree|
        用户: "不要删除" -> disagree|
        用户: "为什么这样处理" -> question|
        用户: "保留所有重复行" -> custom_solution|保留所有重复行
        用户: "我想只删除完全一样的" -> custom_solution|只删除完全相同的行
        用户: "用众数填充" -> custom_solution|用众数填充"""

            try:
                response = self.app.llm.invoke(intent_prompt)
                result = response.content.strip()
                print(f"DEBUG: LLM返回结果: {result}")  # 调试信息

                if "|" in result:
                    intent, custom_solution = result.split("|", 1)
                    intent = intent.strip().lower()
                    custom_solution = custom_solution.strip()

                    # 验证意图是否有效
                    valid_intents = ["agree", "disagree", "question", "custom_solution"]
                    if intent in valid_intents:
                        # 对于custom_solution，如果没有提取到内容，使用用户输入
                        if intent == "custom_solution" and not custom_solution:
                            custom_solution = user_input
                        return intent, custom_solution
                    else:
                        # 意图不在有效列表中，使用降级检测
                        print(f"⚠️ LLM返回无效意图: {intent}，使用降级检测")
                        return self.fallback_intent_detection(user_input), user_input
                else:
                    # 格式不正确，使用降级检测
                    print("⚠️ LLM返回格式不正确，使用降级检测")
                    return self.fallback_intent_detection(user_input), user_input

            except Exception as e:
                print(f"❌ LLM意图识别错误: {e}")
                # 出错时使用降级检测
                return self.fallback_intent_detection(user_input), user_input

        def fallback_intent_detection(self, user_input: str) -> str:
            """改进的备用意图检测方法"""
            user_input_lower = user_input.lower().strip()

            # 更精确的关键词匹配
            agree_patterns = [
                r'^继续$', r'^开始$', r'^同意$', r'^好的$', r'^没问题$',
                r'^行$', r'^可以$', r'^ok$', r'^yes$', r'^y$', r'^是$',
                r'^确认$', r'^执行$', r'^就这么办$'
            ]

            disagree_patterns = [
                r'^不$', r'^不要$', r'^取消$', r'^停止$', r'^退出$',
                r'^no$', r'^n$', r'^拒绝$', r'^不同意$', r'^放弃$',
                r'^算了$', r'^不用了$'
            ]

            question_patterns = [
                r'为什么', r'怎么', r'如何', r'解释', r'说明', r'什么',
                r'原因', r'方法', r'?\?', r'？', r'请教', r'问一下'
            ]

            custom_patterns = [
                r'用.+', r'改成', r'建议', r'自定义', r'我想', r'我要',
                r'保留', r'删除', r'填充', r'处理', r'方案', r'应该',
                r'不如', r'最好', r'推荐'
            ]

            # 检查匹配模式
            for pattern in agree_patterns:
                if re.search(pattern, user_input_lower):
                    return "agree"

            for pattern in disagree_patterns:
                if re.search(pattern, user_input_lower):
                    return "disagree"

            for pattern in custom_patterns:
                if re.search(pattern, user_input_lower):
                    return "custom_solution"

            for pattern in question_patterns:
                if re.search(pattern, user_input_lower):
                    return "question"

            # 默认认为是问题
            return "question"

        def llm_evaluate_solution_with_fallback(self, custom_solution: str, processing_type: str,
                                                processing_details: Dict,
                                                max_retries: int = 2) -> tuple:
            """使用LLM评估方案合理性，带重试"""
            for attempt in range(max_retries):
                try:
                    return self.llm_evaluate_solution(custom_solution, processing_type, processing_details)
                except Exception as e:
                    print(f"❌ 方案评估尝试 {attempt + 1} 失败: {e}")
                    time.sleep(1)

            # 评估失败时的降级处理
            print("⚠️ LLM评估不可用，使用默认评估")
            return True, "方案评估不可用，将执行您的自定义方案"

        def llm_evaluate_solution(self, custom_solution: str, processing_type: str, processing_details: Dict) -> tuple:
            """使用LLM评估自定义方案的合理性"""
            # 检查是否是保留重复的方案
            if self.is_keep_duplicates_intent(custom_solution):
                return False, "保留所有重复行会导致数值稳定性问题，请选择其他处理方案"

            evaluation_prompt = f"""作为数据清洗专家，请评估用户提出的重复值处理方案。

    当前重复值情况:
    - 重复行数: {processing_details['duplicate_count']}
    - 建议方案: {processing_details['suggested_solution']}

    用户方案: "{custom_solution}"

    请从以下角度评估：
    1. 技术合理性（是否符合数据清洗最佳实践）
    2. 对数据完整性的影响
    3. 对后续机器学习算法的影响
    4. 数值稳定性考虑
    5. 潜在的风险或问题
    6. 改进建议（如有）

    请用友好、专业的态度回复，返回格式：合理与否(true/false)|评估反馈"""

            try:
                response = self.app.llm.invoke(evaluation_prompt)
                result = response.content.strip()

                if "|" in result:
                    is_reasonable_str, feedback = result.split("|", 1)
                    is_reasonable = is_reasonable_str.strip().lower() == "true"
                    return is_reasonable, feedback.strip()

                raise ValueError("LLM评估返回格式不正确")

            except Exception as e:
                print(f"❌ LLM方案评估错误: {e}")
                raise

        def llm_generate_response_with_fallback(self, user_input: str, processing_type: str, messages: List[Dict],
                                                processing_details: Dict, max_retries: int = 2) -> str:
            """使用LLM生成回答，带重试"""
            for attempt in range(max_retries):
                try:
                    return self.llm_generate_response(user_input, processing_type, messages, processing_details)
                except Exception as e:
                    print(f"❌ 回答生成尝试 {attempt + 1} 失败: {e}")
                    time.sleep(1)

            # 生成回答失败时的降级处理
            return self.get_fallback_response(processing_type, user_input, processing_details)

        def llm_generate_response(self, user_input: str, processing_type: str, messages: List[Dict],
                                  processing_details: Dict) -> str:
            """使用LLM生成专业回答"""
            # 构建对话上下文
            conversation_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in messages[-3:]  # 最近3条消息
            ])

            system_prompt = f"""你是一个数据清洗专家，专门处理重复值问题。

    当前重复值情况：
    - 重复行数量: {processing_details['duplicate_count']}
    - 建议方案: {processing_details['suggested_solution']}

    请用中文友好、专业地回答用户关于重复值处理的问题。"""

            prompt = f"""{system_prompt}

    当前对话上下文：
    {conversation_context}

    用户最新问题: "{user_input}"

    请生成友好、专业的回答："""

            try:
                response = self.app.llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"❌ LLM回答生成错误: {e}")
                raise

        def get_fallback_response(self, processing_type: str, user_input: str, processing_details: Dict) -> str:
            """获取备用回答"""
            return f"关于您的输入'{user_input}'，在重复值处理中，我们建议删除完全重复的行以保持数据质量。"

        def apply_custom_solution(self, df, custom_solution, processing_details):
            """应用用户自定义的重复值处理方案"""
            print(f"🔄 应用自定义方案: {custom_solution}")

            # 检查是否是保留重复的方案
            if self.is_keep_duplicates_intent(custom_solution):
                # 为用户创建一个合理的替代方案：添加重复标识列
                print("🔧 为您添加重复标识列并保留所有数据...")
                df_cleaned = df.copy()
                df_cleaned['is_duplicate'] = df_cleaned.duplicated(keep=False)
                print(f"✅ 已添加重复标识列，标记了 {df_cleaned['is_duplicate'].sum()} 个重复行")
                return df_cleaned

            # 使用LLM解析和执行其他自定义方案
            try:
                execution_prompt = f"""请解析并执行以下重复值处理方案：

    数据集信息:
    - 形状: {df.shape}
    - 列名: {list(df.columns)}
    - 重复行数: {processing_details['duplicate_count']}

    自定义方案: "{custom_solution}"

    请生成Python代码来执行这个方案，只返回代码部分："""

                response = self.app.llm.invoke(execution_prompt)
                code = response.content.strip()

                # 安全地执行代码
                local_vars = {'df': df.copy(), 'pd': pd}
                exec(code, {}, local_vars)

                result_df = local_vars['df']
                removed_count = len(df) - len(result_df)
                print(f"✅ 自定义方案执行成功！处理了 {removed_count} 个重复行")
                return result_df

            except Exception as e:
                print(f"❌ 自定义方案执行失败: {e}")
                print("⚠️ 使用建议方案代替")
                return self.apply_suggested_solution(df)

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

    class OutlierHandler:
        """异常值处理处理器"""

        def __init__(self, app):
            self.app = app

        def handle_interactive(self, current_data):
            """处理异常值的交互式流程"""
            print("\n" + "=" * 60)
            print("异常值处理向导")
            print("=" * 60)

            # 检查数值列
            numeric_cols = current_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if not numeric_cols:
                print("✅ 数据中没有数值列，无需处理异常值！")
                return current_data, True

            print(f"发现 {len(numeric_cols)} 个数值列: {numeric_cols}")

            # 显示数值列的基本统计信息
            print("\n数值列统计信息:")
            for col in numeric_cols[:3]:  # 只显示前3列
                col_data = current_data[col].dropna()
                if len(col_data) > 0:
                    print(
                        f"  - {col}: 均值={col_data.mean():.2f}, 标准差={col_data.std():.2f}, 范围=[{col_data.min():.2f}, {col_data.max():.2f}]")

            # 检测异常值
            outlier_info = self.detect_outliers(current_data, numeric_cols)

            # 构建处理方案
            suggested_solution = f"对所有数值列进行异常值检测，根据异常值比例采用不同的处理策略：比例<5%保留，>20%对数变换，其他情况缩尾处理。"
            print(f"\n💡 我建议的处理方案:\n{suggested_solution}")

            # 初始化对话状态
            initial_state = ConversationState(
                messages=[],
                agreed=False,
                data=current_data,
                processing_type="outlier",
                processing_details={
                    "numeric_cols": numeric_cols,
                    "outlier_info": outlier_info,
                    "suggested_solution": suggested_solution
                },
                current_response=f"建议的异常值处理方案是：{suggested_solution}\n\n您对这个方案有什么疑问吗？或者您有自定义的处理方案？"
            )

            # 运行对话
            final_state = self.app.run_conversation(initial_state)

            if final_state.agreed:
                if final_state.custom_solution_proposed and final_state.custom_solution:
                    print(f"\n✅ 开始执行您的自定义方案: {final_state.custom_solution}")
                    result = self.apply_custom_solution(final_state.data, final_state.custom_solution,
                                                        final_state.processing_details)
                else:
                    print("\n✅ 开始执行异常值处理...")
                    result = self.apply_suggested_solution(final_state.data, numeric_cols)
                return result, True
            else:
                print("⏭️ 用户取消处理，跳过异常值处理")
                return current_data, True

        def detect_outliers(self, df, numeric_cols):
            """检测异常值"""
            outlier_info = {}
            for col in numeric_cols:
                if col in df.columns:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        outlier_ratio = outliers / len(df) if len(df) > 0 else 0

                        outlier_info[col] = {
                            'outliers': outliers,
                            'outlier_ratio': outlier_ratio,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
            return outlier_info

        def apply_custom_solution(self, df, custom_solution, processing_details):
            """应用用户自定义的异常值处理方案"""
            print(f"🔄 应用自定义方案: {custom_solution}")

            # 使用LLM解析和执行自定义方案
            try:
                numeric_cols = processing_details.get('numeric_cols', [])
                outlier_info = processing_details.get('outlier_info', {})

                execution_prompt = f"""请解析并执行以下异常值处理方案：

数据集信息:
- 形状: {df.shape}
- 列名: {list(df.columns)}
- 数值列: {numeric_cols}
- 异常值信息: {outlier_info}

自定义方案: "{custom_solution}"

请生成Python代码来执行这个方案，只返回代码部分："""

                response = self.app.llm.invoke(execution_prompt)
                code = response.content.strip()

                # 安全地执行代码
                local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
                exec(code, {}, local_vars)

                result_df = local_vars['df']
                print("✅ 自定义方案执行成功！")
                return result_df

            except Exception as e:
                print(f"❌ 自定义方案执行失败: {e}")
                print("⚠️ 使用建议方案代替")
                return self.apply_suggested_solution(df, processing_details.get('numeric_cols', []))

        def apply_suggested_solution(self, df, numeric_cols):
            """应用建议的异常值处理方案"""
            print("🔄 应用建议的异常值处理方案")

            for col in numeric_cols:
                if col in df.columns:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        outlier_ratio = outliers / len(df) if len(df) > 0 else 0

                        if outlier_ratio < 0.05:
                            print(f"📊 {col}: 异常值比例 {outlier_ratio:.2%} < 5%，保留")
                        elif outlier_ratio > 0.2:
                            print(f"📈 {col}: 异常值比例 {outlier_ratio:.2%} > 20%，进行对数变换")
                            # 避免对非正数取对数
                            if (df[col] > 0).all():
                                df[col] = np.log1p(df[col])
                            else:
                                print(f"  ⚠️ {col} 包含非正值，无法进行对数变换，使用缩尾处理")
                                df[col] = np.clip(df[col], lower_bound, upper_bound)
                        else:
                            print(f"⚖️ {col}: 异常值比例 {outlier_ratio:.2%}，进行缩尾处理")
                            df[col] = np.clip(df[col], lower_bound, upper_bound)

            print("✅ 异常值处理完成！")
            return df

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

        # 检测异常值（简单检测数值列）
        numeric_cols = self.current_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            # 简单检查是否有明显异常值
            has_outliers = False
            for col in numeric_cols:
                col_data = self.current_data[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((self.current_data[col] < lower_bound) | (self.current_data[col] > upper_bound)).sum()
                    if outliers > 0:
                        has_outliers = True
                        break

            if has_outliers:
                issues.append(('outlier', f"发现 {len(numeric_cols)} 个数值列需要异常值检测"))

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
                print("\n✅ 数据质量良好，未发现明显问题！")
                break

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

                # 自动继续处理下一个问题
                if i < len(issues):
                    print("⏭️ 自动继续处理下一个问题...")

            # 所有问题处理完成后，提供额外选项
            print(f"\n{'=' * 60}")
            print("所有检测到的问题已处理完成！")
            print(f"{'=' * 60}")

            # 询问是否继续或退出
            choice = input("\n是否继续检测其他问题？(yes/no): ").lower().strip()
            if choice not in ['yes', 'y', '是']:
                break

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
        """运行应用程序"""
        self.interactive_cleaning_session()


if __name__ == "__main__":
    app = DataCleaningApp()
    app.run()
