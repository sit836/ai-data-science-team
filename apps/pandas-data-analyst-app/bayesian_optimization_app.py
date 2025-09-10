# BUSINESS SCIENCE
# Bayesian Optimization App
# -------------------------

# This app is designed to help you perform Bayesian optimization on your data using natural language requests.

# Imports
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import time
import asyncio

from ai_data_science_team.agents.bayesian_optimization_agent import BayesianOptimizationAgent
import re

# * APP INPUTS ----

TITLE = "贝叶斯优化AI助手"

load_dotenv()

# ---------------------------
# Data Cleaning Classes and Functions
# ---------------------------

class ConversationState:
    """对话状态类"""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.agreed: bool = False
        self.data: Optional[pd.DataFrame] = None
        self.processing_type: str = ""
        self.processing_details: Dict[str, Any] = {}
        self.current_response: Optional[str] = None
        self.custom_solution_proposed: bool = False
        self.custom_solution: Optional[str] = None

class DataCleaningHandler:
    """数据清洗处理器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_data_issues(self, df: pd.DataFrame) -> List[tuple]:
        """检测数据中的所有问题"""
        issues = []
        
        # 检测缺失值
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        if len(missing_columns) > 0:
            issues.append(('missing', f"发现 {len(missing_columns)} 列有缺失值", missing_columns))
        
        # 检测重复值
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(('duplicate', f"发现 {duplicate_count} 个重复行", duplicate_count))
        
        # 检测异常值（简单检测数值列）
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            has_outliers = False
            outlier_cols = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        has_outliers = True
                        outlier_cols.append(col)
            
            if has_outliers:
                issues.append(('outlier', f"发现 {len(outlier_cols)} 个数值列需要异常值检测", outlier_cols))
        
        return issues
    
    def llm_intent_detection_with_fallback(self, user_input: str, processing_type: str, max_retries: int = 2) -> tuple:
        """使用LLM判断用户意图，带重试和降级处理"""
        for attempt in range(max_retries):
            try:
                intent, custom_solution = self.llm_classify_intent(user_input, processing_type)
                return intent, custom_solution
            except Exception as e:
                print(f"❌ 意图识别尝试 {attempt + 1} 失败: {e}")
                time.sleep(1)
        
        # 所有重试都失败，使用降级逻辑
        return self.fallback_intent_detection(user_input), user_input
    
    def llm_classify_intent(self, user_input: str, processing_type: str) -> tuple:
        """使用LLM判断用户意图"""
        intent_prompt = f"""作为数据清洗助手，请分析用户的输入意图。

用户输入: "{user_input}"
当前处理类型: {processing_type}

请判断用户的意图：
1. agree - 用户同意当前方案，要求继续处理（如：继续、开始、同意、好的等）
2. disagree - 用户不同意当前方案，要求取消处理（如：不、不要、取消等）
3. question - 用户提出问题或需要解释（如：为什么、如何、什么等）
4. custom_solution - 用户提出了自定义的处理方案（如：用0补、用平均值填充、删除重复行等）

特别注意：
- 如果用户输入包含具体的处理方法（如"用0补"、"用平均值"、"删除重复"等），应识别为custom_solution
- 如果用户只是简单同意或继续，应识别为agree

如果是自定义方案，请提取方案的核心内容。

请返回格式：意图|方案内容（如适用）

示例：
agree|
question|
custom_solution|用0填充缺失值
custom_solution|删除重复行"""

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
    
    def fallback_intent_detection(self, user_input: str) -> str:
        """备用意图检测方法"""
        user_input_lower = user_input.lower().strip()

        # 简单的关键词匹配（仅在LLM不可用时使用）
        agree_keywords = ['继续', '开始', '同意', '好的', '没问题', 'ok', 'yes', 'y', '是']
        disagree_keywords = ['不', '不要', '取消', '停止', '退出', 'no', 'n', '拒绝', '不同意']
        custom_keywords = ['用', '填充', '删除', '保留', '处理', '方案', '补', '0', '平均值', '中位数', '众数']

        # 优先检查自定义方案关键词
        if any(keyword in user_input_lower for keyword in custom_keywords):
            return "custom_solution"
        elif any(keyword in user_input_lower for keyword in agree_keywords):
            return "agree"
        elif any(keyword in user_input_lower for keyword in disagree_keywords):
            return "disagree"
        else:
            return "question"
    
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
    
    def llm_generate_response_streaming(self, user_input: str, processing_type: str, messages: List[Dict], 
                                       processing_details: Dict, placeholder):
        """流式生成LLM回答"""
        # 构建对话上下文
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in messages[-3:]  # 最近3条消息
        ])
        
        prompt = f"""作为数据清洗助手，请根据对话历史生成友好、专业的回答。

对话历史：
{conversation_context}

当前处理类型: {processing_type}
处理详情: {processing_details}
用户最新输入: "{user_input}"

请生成友好、专业的回答，帮助用户理解处理方案："""

        try:
            # 使用流式输出
            full_response = ""
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(f"**AI:** {full_response}")
                    time.sleep(0.05)  # 控制输出速度
            
            return full_response
        except Exception as e:
            print(f"❌ LLM流式回答生成错误: {e}")
            # 降级到非流式
            return self.llm_generate_response_with_fallback(user_input, processing_type, messages, processing_details)
    
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
    
    def get_fallback_response(self, processing_type: str, user_input: str, processing_details: Dict) -> str:
        """获取备用回答"""
        responses = {
            "missing": f"关于您的输入'{user_input}'，在缺失值处理中，我们通常根据数据类型和缺失比例选择不同的填充方法。",
            "duplicate": f"关于您的输入'{user_input}'，在重复值处理中，我们建议删除完全重复的行。",
            "outlier": f"关于您的输入'{user_input}'，在异常值处理中，需要根据异常值的比例选择不同的处理方法。"
        }
        return responses.get(processing_type, f"关于您的输入'{user_input}'，我会尽力帮助您。")

class MissingValueHandler:
    """缺失值处理处理器"""
    
    def __init__(self, cleaning_handler):
        self.cleaning_handler = cleaning_handler
    
    def handle_interactive(self, current_data: pd.DataFrame) -> tuple:
        """处理缺失值的交互式流程"""
        # 检查缺失值
        missing_counts = current_data.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]

        if len(missing_columns) == 0:
            return current_data, True

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
        
        return {
            'type': 'missing',
            'description': f"发现 {len(missing_columns)} 列有缺失值",
            'suggested_solution': suggested_solution,
            'processing_details': {
                "high_missing_cols": high_missing_cols,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "suggested_solution": suggested_solution
            }
        }
    
    def apply_suggested_solution(self, df: pd.DataFrame, processing_details: Dict) -> pd.DataFrame:
        """应用建议的缺失值处理方案"""
        high_missing_cols = processing_details.get('high_missing_cols', [])
        numeric_cols = processing_details.get('numeric_cols', [])
        categorical_cols = processing_details.get('categorical_cols', [])
        
        # 处理高缺失率列
        if high_missing_cols:
            df = df.drop(columns=high_missing_cols)

        # 处理数值列
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # 处理分类列
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)

        return df
    
    def apply_custom_solution_streaming(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict, placeholder) -> pd.DataFrame:
        """流式执行用户自定义的缺失值处理方案"""
        try:
            # 获取缺失值信息
            missing_info = df.isnull().sum()
            missing_columns = missing_info[missing_info > 0]
            
            show_loading_indicator("🤖 AI正在分析您的需求并生成处理代码...")
            
            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理缺失值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 缺失值统计: {missing_info.to_dict()}
- 有缺失值的列: {list(missing_columns.index) if len(missing_columns) > 0 else '无'}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理所有缺失值
4. 可以使用 pandas 和 numpy 库

示例代码格式：
df = df.fillna(0)  # 用0填充所有缺失值
# 或者针对特定列：
# df['column_name'] = df['column_name'].fillna(0)

请生成代码："""

            # 使用流式输出生成代码
            full_response = ""
            for chunk in self.cleaning_handler.llm.stream(execution_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(f"**AI正在生成代码:**\n```python\n{full_response}\n```")
                    time.sleep(0.05)
            
            code = full_response.strip()
            
            # 清理代码，移除可能的markdown标记和解释文字
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            if not code:
                placeholder.error("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            placeholder.success(f"✅ 代码生成完成！\n\n**生成的代码:**\n```python\n{code}\n```")
            
            # 显示更明显的提示，让用户有时间阅读代码
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #00b894, #00a085);
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">
                📋 请查看上方生成的代码，系统将在3秒后自动执行...
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(3)  # 给用户时间阅读代码
            
            show_loading_indicator("🔧 正在执行代码处理数据...")

            # 调试：显示执行前的数据样本
            print(f"🔧 缺失值处理 - 代码执行前数据样本:")
            print(f"   - 数据形状: {df.shape}")
            print(f"   - 缺失值位置: {df.isnull().sum().sum()}")
            print(f"   - 数据样本:\n{df.head()}")
            print(f"   - 生成的代码: {code}")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 调试：显示执行后的数据样本
            print(f"🔧 缺失值处理 - 代码执行后数据样本:")
            print(f"   - 数据形状: {result_df.shape}")
            print(f"   - 缺失值位置: {result_df.isnull().sum().sum()}")
            print(f"   - 数据样本:\n{result_df.head()}")
            
            # 验证结果
            original_missing = df.isnull().sum().sum()
            remaining_missing = result_df.isnull().sum().sum()
            
            placeholder.success(f"🎉 处理完成！\n\n**处理结果:**\n- 处理前缺失值: {original_missing}\n- 处理后缺失值: {remaining_missing}")
            
            if remaining_missing > 0:
                placeholder.warning(f"⚠️ 注意: 仍有 {remaining_missing} 个缺失值未处理")
            
            return result_df

        except Exception as e:
            placeholder.error(f"❌ 自定义方案执行失败: {e}")
            placeholder.info("🔄 回退到建议方案")
            return self.apply_suggested_solution(df, processing_details)
    
    def apply_custom_solution(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict) -> pd.DataFrame:
        """应用用户自定义的缺失值处理方案"""
        try:
            # 获取缺失值信息
            missing_info = df.isnull().sum()
            missing_columns = missing_info[missing_info > 0]
            
            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理缺失值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 缺失值统计: {missing_info.to_dict()}
- 有缺失值的列: {list(missing_columns.index) if len(missing_columns) > 0 else '无'}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理所有缺失值
4. 可以使用 pandas 和 numpy 库

示例代码格式：
df = df.fillna(0)  # 用0填充所有缺失值
# 或者针对特定列：
# df['column_name'] = df['column_name'].fillna(0)

请生成代码："""

            response = self.cleaning_handler.llm.invoke(execution_prompt)
            code = response.content.strip()
            
            # 清理代码，移除可能的markdown标记和解释文字
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    # 保留代码行，但移除注释
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            # 如果代码为空，使用默认处理
            if not code:
                print("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            print(f"🔧 执行用户自定义方案:")
            print(f"用户要求: {custom_solution}")
            print(f"生成代码: {code}")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 验证结果
            original_missing = df.isnull().sum().sum()
            remaining_missing = result_df.isnull().sum().sum()
            
            print(f"✅ 处理完成:")
            print(f"   - 处理前缺失值: {original_missing}")
            print(f"   - 处理后缺失值: {remaining_missing}")
            
            if remaining_missing > 0:
                print(f"⚠️ 注意: 仍有 {remaining_missing} 个缺失值未处理")
            
            return result_df

        except Exception as e:
            print(f"❌ 自定义方案执行失败: {e}")
            print("🔄 回退到建议方案")
            return self.apply_suggested_solution(df, processing_details)

class DuplicateHandler:
    """重复值处理处理器"""
    
    def __init__(self, cleaning_handler):
        self.cleaning_handler = cleaning_handler
    
    def handle_interactive(self, current_data: pd.DataFrame) -> dict:
        """处理重复值的交互式流程"""
        # 检查重复行
        duplicate_count = current_data.duplicated().sum()

        if duplicate_count == 0:
            return None

        # 构建处理方案
        suggested_solution = f"删除所有重复行，保留第一个出现的重复值。这将删除 {duplicate_count} 个重复行。"
        
        return {
            'type': 'duplicate',
            'description': f"发现 {duplicate_count} 个重复行",
            'suggested_solution': suggested_solution,
            'processing_details': {
                "duplicate_count": duplicate_count,
                "suggested_solution": suggested_solution
            }
        }
    
    def apply_suggested_solution(self, df: pd.DataFrame, processing_details: Dict) -> pd.DataFrame:
        """应用建议的重复值处理方案"""
        return df.drop_duplicates(keep='first')
    
    def apply_custom_solution_streaming(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict, placeholder) -> pd.DataFrame:
        """流式执行用户自定义的重复值处理方案"""
        try:
            duplicate_count = df.duplicated().sum()
            
            show_loading_indicator("🤖 AI正在分析您的需求并生成处理代码...")
            
            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理重复值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 重复行数: {duplicate_count}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理重复值
4. 可以使用 pandas 库

示例代码格式：
df = df.drop_duplicates()  # 删除所有重复行
# 或者保留第一个：
# df = df.drop_duplicates(keep='first')

请生成代码："""

            # 使用流式输出生成代码
            full_response = ""
            for chunk in self.cleaning_handler.llm.stream(execution_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(f"**AI正在生成代码:**\n```python\n{full_response}\n```")
                    time.sleep(0.05)
            
            code = full_response.strip()
            
            # 清理代码
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            if not code:
                placeholder.error("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            placeholder.success(f"✅ 代码生成完成！\n\n**生成的代码:**\n```python\n{code}\n```")
            
            # 显示更明显的提示，让用户有时间阅读代码
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #00b894, #00a085);
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">
                📋 请查看上方生成的代码，系统将在3秒后自动执行...
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(3)  # 给用户时间阅读代码
            
            show_loading_indicator("🔧 正在执行代码处理数据...")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 验证结果
            original_duplicates = duplicate_count
            remaining_duplicates = result_df.duplicated().sum()
            
            placeholder.success(f"🎉 处理完成！\n\n**处理结果:**\n- 处理前重复行: {original_duplicates}\n- 处理后重复行: {remaining_duplicates}")
            
            return result_df

        except Exception as e:
            placeholder.error(f"❌ 自定义方案执行失败: {e}")
            placeholder.info("🔄 回退到建议方案")
            return self.apply_suggested_solution(df, processing_details)
    
    def apply_custom_solution(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict) -> pd.DataFrame:
        """应用用户自定义的重复值处理方案"""
        try:
            duplicate_count = df.duplicated().sum()
            
            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理重复值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 重复行数: {duplicate_count}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理重复值
4. 可以使用 pandas 库

示例代码格式：
df = df.drop_duplicates()  # 删除所有重复行
# 或者保留第一个：
# df = df.drop_duplicates(keep='first')

请生成代码："""

            response = self.cleaning_handler.llm.invoke(execution_prompt)
            code = response.content.strip()
            
            # 清理代码，移除可能的markdown标记和解释文字
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            if not code:
                print("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            print(f"🔧 执行用户自定义方案:")
            print(f"用户要求: {custom_solution}")
            print(f"生成代码: {code}")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 验证结果
            original_duplicates = duplicate_count
            remaining_duplicates = result_df.duplicated().sum()
            
            print(f"✅ 处理完成:")
            print(f"   - 处理前重复行: {original_duplicates}")
            print(f"   - 处理后重复行: {remaining_duplicates}")
            
            return result_df

        except Exception as e:
            print(f"❌ 自定义方案执行失败: {e}")
            return self.apply_suggested_solution(df, processing_details)

class OutlierHandler:
    """异常值处理处理器"""
    
    def __init__(self, cleaning_handler):
        self.cleaning_handler = cleaning_handler
    
    def handle_interactive(self, current_data: pd.DataFrame) -> dict:
        """处理异常值的交互式流程"""
        # 检查数值列
        numeric_cols = current_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if not numeric_cols:
            return None

        # 检测异常值
        outlier_info = self.detect_outliers(current_data, numeric_cols)

        # 构建处理方案
        suggested_solution = f"对所有数值列进行异常值检测，根据异常值比例采用不同的处理策略：比例<5%保留，>20%对数变换，其他情况缩尾处理。"
        
        return {
            'type': 'outlier',
            'description': f"发现 {len(numeric_cols)} 个数值列需要异常值检测",
            'suggested_solution': suggested_solution,
            'processing_details': {
                "numeric_cols": numeric_cols,
                "outlier_info": outlier_info,
                "suggested_solution": suggested_solution
            }
        }
    
    def detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
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
    
    def apply_suggested_solution(self, df: pd.DataFrame, processing_details: Dict) -> pd.DataFrame:
        """应用建议的异常值处理方案"""
        numeric_cols = processing_details.get('numeric_cols', [])
        
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
                        pass  # 保留
                    elif outlier_ratio > 0.2:
                        # 对数变换
                        if (df[col] > 0).all():
                            df[col] = np.log1p(df[col])
                        else:
                            df[col] = np.clip(df[col], lower_bound, upper_bound)
                    else:
                        # 缩尾处理
                        df[col] = np.clip(df[col], lower_bound, upper_bound)

        return df
    
    def apply_custom_solution_streaming(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict, placeholder) -> pd.DataFrame:
        """流式执行用户自定义的异常值处理方案"""
        try:
            numeric_cols = processing_details.get('numeric_cols', [])
            outlier_info = processing_details.get('outlier_info', {})
            
            # 计算异常值统计
            outlier_stats = {}
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
                        outlier_stats[col] = {
                            'outlier_count': outliers,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }

            show_loading_indicator("🤖 AI正在分析您的需求并生成处理代码...")

            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理异常值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 数值列: {numeric_cols}
- 异常值统计: {outlier_stats}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理异常值
4. 可以使用 pandas 和 numpy 库

示例代码格式：
# 删除异常值
df = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
# 或者缩尾处理
df['column'] = np.clip(df['column'], lower_bound, upper_bound)

请生成代码："""

            # 使用流式输出生成代码
            full_response = ""
            for chunk in self.cleaning_handler.llm.stream(execution_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(f"**AI正在生成代码:**\n```python\n{full_response}\n```")
                    time.sleep(0.05)
            
            code = full_response.strip()
            
            # 清理代码
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            if not code:
                placeholder.error("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            placeholder.success(f"✅ 代码生成完成！\n\n**生成的代码:**\n```python\n{code}\n```")
            
            # 显示更明显的提示，让用户有时间阅读代码
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #00b894, #00a085);
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">
                📋 请查看上方生成的代码，系统将在3秒后自动执行...
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(3)  # 给用户时间阅读代码
            
            show_loading_indicator("🔧 正在执行代码处理数据...")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 验证结果
            original_shape = df.shape
            result_shape = result_df.shape
            
            placeholder.success(f"🎉 处理完成！\n\n**处理结果:**\n- 处理前数据形状: {original_shape}\n- 处理后数据形状: {result_shape}")
            
            return result_df

        except Exception as e:
            placeholder.error(f"❌ 自定义方案执行失败: {e}")
            placeholder.info("🔄 回退到建议方案")
            return self.apply_suggested_solution(df, processing_details)
    
    def apply_custom_solution(self, df: pd.DataFrame, custom_solution: str, processing_details: Dict) -> pd.DataFrame:
        """应用用户自定义的异常值处理方案"""
        try:
            numeric_cols = processing_details.get('numeric_cols', [])
            outlier_info = processing_details.get('outlier_info', {})
            
            # 计算异常值统计
            outlier_stats = {}
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
                        outlier_stats[col] = {
                            'outlier_count': outliers,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }

            execution_prompt = f"""作为数据清洗专家，请根据用户要求生成Python代码来处理异常值。

数据集信息:
- 数据形状: {df.shape}
- 列名: {list(df.columns)}
- 数值列: {numeric_cols}
- 异常值统计: {outlier_stats}

用户要求: "{custom_solution}"

请生成Python代码来执行用户的要求。代码要求：
1. 只返回可执行的Python代码，不要包含解释文字
2. 使用变量名 'df' 表示DataFrame
3. 确保代码能正确处理异常值
4. 可以使用 pandas 和 numpy 库

示例代码格式：
# 删除异常值
df = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
# 或者缩尾处理
df['column'] = np.clip(df['column'], lower_bound, upper_bound)

请生成代码："""

            response = self.cleaning_handler.llm.invoke(execution_prompt)
            code = response.content.strip()
            
            # 清理代码，移除可能的markdown标记和解释文字
            lines = code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (line and not line.startswith('#') and not line.startswith('作为') and not line.startswith('请')):
                    if line.startswith('#'):
                        continue
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            
            if not code:
                print("⚠️ LLM未生成有效代码，使用默认处理")
                return self.apply_suggested_solution(df, processing_details)

            print(f"🔧 执行用户自定义方案:")
            print(f"用户要求: {custom_solution}")
            print(f"生成代码: {code}")

            # 安全地执行代码
            local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
            exec(code, {}, local_vars)

            result_df = local_vars['df']
            
            # 验证结果
            original_shape = df.shape
            result_shape = result_df.shape
            
            print(f"✅ 处理完成:")
            print(f"   - 处理前数据形状: {original_shape}")
            print(f"   - 处理后数据形状: {result_shape}")
            
            return result_df

        except Exception as e:
            print(f"❌ 自定义方案执行失败: {e}")
            return self.apply_suggested_solution(df, processing_details)

# ---------------------------
# Helper Functions
# ---------------------------

def show_loading_indicator(message: str, position: str = "below"):
    """显示与主题一致的加载指示器"""
    # 使用与主题一致的蓝色渐变背景
    st.markdown(f"""
    <div style="
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 20px 25px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
        animation: pulse 1.5s ease-in-out infinite;
        border: 2px solid rgba(255,255,255,0.2);
    ">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <div style="
                width: 30px;
                height: 30px;
                border: 4px solid rgba(255,255,255,0.3);
                border-top: 4px solid white;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            "></div>
            <span style="font-size: 20px;">{message}</span>
        </div>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ 
                opacity: 1; 
                transform: scale(1);
            }}
            50% {{ 
                opacity: 0.8; 
                transform: scale(1.02);
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

def get_next_question(step: str, config: dict, available_columns: list) -> str:
    """根据当前步骤生成下一个问题"""
    if step == "ask_input_output":
        return f"我看到您的数据包含以下列: {', '.join(available_columns)}\n\n请告诉我：\n1. 哪些是输入变量（用于优化的特征）？\n2. 哪个是输出变量（优化目标）？\n3. 您希望最大化还是最小化输出变量？\n\n您可以自由描述，我会帮您解析。"
    
    elif step == "confirm_config":
        input_vars = config.get("input_variables", [])
        output_var = config.get("output_variable", "")
        goal = config.get("optimization_goal", "")
        goal_text = "最大化" if goal == "maximize" else "最小化" if goal == "minimize" else "未确定"
        
        return f"让我确认一下您的配置：\n\n📋 **输入变量（特征）**: {', '.join(input_vars)}\n🎯 **输出变量（目标）**: {output_var}\n🎯 **优化目标**: {goal_text}\n\n这个配置正确吗？如果需要修改，请告诉我。"
    
    elif step == "ask_bounds":
        input_vars = config.get("input_variables", [])
        return f"很好！现在需要设置每个输入变量的取值范围。\n\n请为以下变量设置边界：\n{chr(10).join([f'• {var}' for var in input_vars])}\n\n您可以逐个告诉我，或者一次性告诉我所有变量的范围。"
    
    elif step == "confirm_bounds":
        bounds = config.get("variable_bounds", {})
        bounds_text = "\n".join([f"• {var}: {bounds[var][0]:.2f} ~ {bounds[var][1]:.2f}" for var in bounds])
        return f"请确认变量边界设置：\n\n{bounds_text}\n\n这些边界设置正确吗？"
    
    return "请告诉我您想要优化什么？"

# ---------------------------
# Initialize Session State
# ---------------------------

# Initialize all session state variables
if "optimization_step" not in st.session_state:
    st.session_state.optimization_step = "chat_setup"


if "input_variables" not in st.session_state:
    st.session_state.input_variables = []

if "output_variable" not in st.session_state:
    st.session_state.output_variable = None

if "optimization_goal" not in st.session_state:
    st.session_state.optimization_goal = None

if "variable_bounds" not in st.session_state:
    st.session_state.variable_bounds = {}

if "config_confirmed" not in st.session_state:
    st.session_state.config_confirmed = False

if "bounds_confirmed" not in st.session_state:
    st.session_state.bounds_confirmed = False

if "current_suggestion" not in st.session_state:
    st.session_state.current_suggestion = None

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = []

if "bayesian_agent" not in st.session_state:
    st.session_state.bayesian_agent = None

if "optimization_config" not in st.session_state:
    st.session_state.optimization_config = {}

if "current_step" not in st.session_state:
    st.session_state.current_step = "ask_input_output"

if "available_columns" not in st.session_state:
    st.session_state.available_columns = []

if "numeric_columns" not in st.session_state:
    st.session_state.numeric_columns = []

if "data" not in st.session_state:
    st.session_state.data = None

if "optimization_iteration" not in st.session_state:
    st.session_state.optimization_iteration = 0

# 数据清洗相关的session state
if "data_cleaning_step" not in st.session_state:
    st.session_state.data_cleaning_step = "detect"  # detect, process, complete

if "data_issues" not in st.session_state:
    st.session_state.data_issues = []

if "current_issue_index" not in st.session_state:
    st.session_state.current_issue_index = 0

if "data_cleaning_handler" not in st.session_state:
    st.session_state.data_cleaning_handler = None

if "cleaning_conversation_state" not in st.session_state:
    st.session_state.cleaning_conversation_state = None

if "data_cleaned" not in st.session_state:
    st.session_state.data_cleaned = False

if "cleaning_conversation_history" not in st.session_state:
    st.session_state.cleaning_conversation_history = {}

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 简洁白色主题模板
st.markdown("""
<style>
    /* 主应用背景 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 主容器 */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin-top: 1rem;
    }
    
    
    /* 侧边栏 */
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSidebar .stSelectbox > label,
    .stSidebar .stTextInput > label,
    .stSidebar .stTextArea > label {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* 侧边栏文字颜色 */
    .stSidebar .stMarkdown,
    .stSidebar .stText,
    .stSidebar p,
    .stSidebar div {
        color: #2c3e50 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, 
    .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #2c3e50 !important;
        font-weight: 700;
    }
    
    /* 文字颜色 */
    .stMarkdown, .stText, p, div {
        color: #2c3e50 !important;
    }
    
    /* 标题 */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 700;
    }
    
    /* 表单元素 */
    .stSelectbox > label,
    .stSlider > label,
    .stNumberInput > label,
    .stTextInput > label,
    .stTextArea > label,
    .stMultiselect > label {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    
    /* 主要按钮 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #ff6b6b 0%, #ee5a24 100%);
        color: white !important;
    }
    
    /* 文件上传按钮 - 更具体的选择器 */
    .stFileUploader > div > div > button,
    .stFileUploader button,
    .stFileUploader > div > div > button > span,
    .stFileUploader button > span {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader > div > div > button:hover,
    .stFileUploader button:hover,
    .stFileUploader > div > div > button:hover > span,
    .stFileUploader button:hover > span {
        background: linear-gradient(45deg, #5a6fd8 0%, #6a4190 100%) !important;
        color: white !important;
    }
    
    /* 文件上传区域 */
    .stFileUploader > div {
        background: white !important;
        border: 2px dashed #667eea !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #5a6fd8 !important;
        background: #f8f9fa !important;
    }
    
    /* 文件上传按钮文字 - 强制覆盖 */
    .stFileUploader button p,
    .stFileUploader button div,
    .stFileUploader > div > div > button p,
    .stFileUploader > div > div > button div {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* 文件上传按钮所有子元素 */
    .stFileUploader button * {
        color: white !important;
    }
    
    .stFileUploader > div > div > button * {
        color: white !important;
    }
    
    /* 强制覆盖所有可能的文件上传按钮样式 */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploader"] > div > div > button,
    [data-testid="stFileUploader"] > div > div > button * {
        color: white !important;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* 文件上传按钮的文本内容 */
    .stFileUploader button::before,
    .stFileUploader button::after,
    .stFileUploader > div > div > button::before,
    .stFileUploader > div > div > button::after {
        color: white !important;
    }
    
    /* 使用更高优先级的选择器 */
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button * {
        color: white !important;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* 指标卡片 */
    .stMetric {
        background: white;
        border: 2px solid #e8f4fd;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric > div > div {
        color: #2c3e50 !important;
    }
    
    /* 数据框 */
    .stDataFrame {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 输入框 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e8f4fd;
        border-radius: 8px;
        background: white;
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        color: #2c3e50 !important;
    }
    
    /* 选择框 */
    .stSelectbox > div > div,
    .stMultiselect > div > div {
        border: 2px solid #e8f4fd;
        border-radius: 8px;
        background: white;
    }
    
    /* 多选框选项 */
    .stMultiSelect > div > div > div {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        color: #2c3e50;
        font-weight: 500;
    }
    
    .stMultiSelect > div > div > div:hover {
        background: #e9ecef;
        border-color: #667eea;
    }
    
    /* 选择框选项文字 */
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* 多选框标签 */
    .stMultiSelect > div > div > div > span {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* 多选框选中状态 */
    .stMultiSelect > div > div > div[data-baseweb="tag"] {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 6px;
    }
    
    /* 多选框下拉选项 */
    .stMultiSelect > div > div > div[role="option"] {
        color: #2c3e50 !important;
        font-weight: 500;
        padding: 8px 12px;
        background: white !important;
    }
    
    .stMultiSelect > div > div > div[role="option"]:hover {
        background: #e9ecef !important;
        color: #2c3e50 !important;
    }
    
    /* 多选框下拉菜单背景 */
    .stMultiSelect > div > div > div[data-baseweb="popover"] {
        background: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    /* 多选框下拉菜单内容 */
    .stMultiSelect > div > div > div[data-baseweb="popover"] > div {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    /* 多选框输入框 */
    .stMultiSelect > div > div > input {
        color: #2c3e50 !important;
        font-weight: 500;
        background: white !important;
    }
    
    /* 确保所有输入框文字可见 */
    input[type="text"], 
    input[type="number"], 
    textarea {
        color: #2c3e50 !important;
        background: white !important;
    }
    
    /* 占位符文字颜色 */
    input::placeholder,
    textarea::placeholder {
        color: #6c757d !important;
        opacity: 0.8;
    }
    
    /* 选择框下拉选项 */
    .stSelectbox > div > div > div[role="option"] {
        color: #2c3e50 !important;
        font-weight: 500;
        padding: 8px 12px;
        background: white !important;
    }
    
    .stSelectbox > div > div > div[role="option"]:hover {
        background: #e9ecef !important;
        color: #2c3e50 !important;
    }
    
    /* 选择框下拉菜单背景 */
    .stSelectbox > div > div > div[data-baseweb="popover"] {
        background: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    /* 选择框下拉菜单内容 */
    .stSelectbox > div > div > div[data-baseweb="popover"] > div {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    /* 滑块 */
    .stSlider > div > div > div {
        background: #667eea;
    }
    
    /* 通用下拉菜单样式 */
    [data-baseweb="popover"] {
        background: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-baseweb="popover"] > div {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    /* 下拉选项通用样式 */
    [role="option"] {
        color: #2c3e50 !important;
        background: white !important;
        font-weight: 500 !important;
    }
    
    [role="option"]:hover {
        background: #e9ecef !important;
        color: #2c3e50 !important;
    }
    
    /* 确保所有下拉菜单内容可见 */
    .stSelectbox [data-baseweb="popover"],
    .stMultiSelect [data-baseweb="popover"] {
        z-index: 9999 !important;
    }
    
    /* 成功/警告/错误消息 */
    .stSuccess {
        background: linear-gradient(45deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(45deg, #e84393 0%, #d63031 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(45deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* 展开器 */
    .streamlit-expanderHeader {
        background: linear-gradient(45deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border-radius: 8px;
    }
    
    /* 聊天消息 */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* 改善help文字可见性 */
    .stTextInput > div > div > div[data-testid="stMarkdownContainer"] {
        color: #667eea !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-top: 5px !important;
    }
    
    /* 输入框help文字样式 */
    .stTextInput [data-testid="stMarkdownContainer"] p {
        color: #667eea !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        background: rgba(102, 126, 234, 0.1) !important;
        padding: 5px 10px !important;
        border-radius: 5px !important;
        border-left: 3px solid #667eea !important;
    }
    
    /* 隐藏Streamlit默认的按回车提示 */
    .stTextInput [data-testid="stMarkdownContainer"] p {
        display: none !important;
    }
    
    /* 美化help文字样式 */
    .stTextInput [data-testid="stMarkdownContainer"] p:first-child {
        display: block !important;
        color: #667eea !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        border-left: 3px solid #667eea !important;
        margin-top: 8px !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* 为输入框添加优雅的按回车提示 */
    .stTextInput::after {
        content: "💡 输入完成后按回车键提交";
        display: block;
        color: #667eea !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-top: 6px !important;
        text-align: center !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 0 1px 3px rgba(102, 126, 234, 0.1) !important;
    }
</style>

<script>
// 隐藏Streamlit默认的按回车提示
function hideDefaultEnterText() {
    const elements = document.querySelectorAll('*');
    elements.forEach(element => {
        if (element.textContent && 
            (element.textContent.includes('PRESS ENTER TO APPLY') || 
             element.textContent.includes('按回车键应用'))) {
            element.style.display = 'none';
        }
    });
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', hideDefaultEnterText);

// 监听DOM变化，处理动态添加的元素
const observer = new MutationObserver(hideDefaultEnterText);
observer.observe(document.body, {
    childList: true,
    subtree: true
});
</script>
""", unsafe_allow_html=True)
st.title(TITLE)

st.markdown("""
AI助手将通过与您对话来了解您的优化需求，然后使用贝叶斯优化找到最优参数。
""")

with st.expander("优化问题示例", expanded=False):
    st.write(
        """
        ##### 优化示例：
        
        - 寻找函数 f(x1, x2) = sin(x1) * cos(x2) 的最大值
        - 优化机器学习模型的超参数
        - 寻找制造过程的最优参数
        - 优化化学反应条件
        - 寻找系统的最佳配置
        """
    )

# ---------------------------
# Initialize LLM with environment variables
# ---------------------------

# Use environment variables for API configuration
openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Initialize LLM with default model
llm = ChatOpenAI(
    model="deepseek-chat",  # Default model
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] if openai_api_data['base_url'] else None
)

# ---------------------------
# Data Input Section
# ---------------------------

st.markdown("## 数据输入")

# Create two columns for data input options
col1, col2 = st.columns(2)

with col1:
    st.subheader("上传数据")
    uploaded_file = st.file_uploader(
        "选择包含实验数据的CSV文件", 
        type=["csv"],
        help="上传包含输入特征和目标值列的CSV文件"
    )

with col2:
        np.random.seed(42)
        n_points = 20
        
        # 创建2D优化问题：f(x1, x2) = sin(x1) * cos(x2)
        x1 = np.random.uniform(0, 2*np.pi, n_points)
        x2 = np.random.uniform(0, 2*np.pi, n_points)
        y = np.sin(x1) * np.cos(x2) + np.random.normal(0, 0.1, n_points)
        
        # 创建数据框
        df = pd.DataFrame({
            '特征1': x1,
            '特征2': x2,
            '目标值': y
        })
        

# 加载数据
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_data = df
elif hasattr(st.session_state, 'uploaded_data'):
    df = st.session_state.uploaded_data

if df is not None:
    st.subheader("数据预览")
    st.dataframe(df.head(10))
    
    # 显示数据统计
    col1, col2 = st.columns(2)
    with col1:
        st.write("**数据形状:**", df.shape)
    with col2:
        st.write("**数据概览:**")
        st.write(f"• 总行数: {df.shape[0]}")
        st.write(f"• 总列数: {df.shape[1]}")
    
    # 存储可用列名（内部使用，不显示给用户）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.available_columns = list(df.columns)
    st.session_state.numeric_columns = numeric_cols
    
    # 只有在数据清洗未开始且没有进行中的清洗时才更新数据，避免覆盖清洗后的数据
    if (not st.session_state.data_cleaned and 
        st.session_state.data_cleaning_step == "detect" and 
        st.session_state.current_issue_index == 0):
        st.session_state.data = df
        print(f"🔧 数据输入部分更新数据:")
        print(f"   - 更新数据缺失值: {df.isnull().sum().sum()}")
        print(f"   - 数据清洗状态: {st.session_state.data_cleaned}")
        print(f"   - 清洗步骤: {st.session_state.data_cleaning_step}")
    else:
        print(f"🔧 数据输入部分跳过数据更新:")
        print(f"   - 数据清洗状态: {st.session_state.data_cleaned}")
        print(f"   - 清洗步骤: {st.session_state.data_cleaning_step}")
        print(f"   - 当前问题索引: {st.session_state.current_issue_index}")
    
    # 初始化数据清洗处理器
    if st.session_state.data_cleaning_handler is None:
        st.session_state.data_cleaning_handler = DataCleaningHandler(llm)
    
    # ---------------------------
    # Data Cleaning Section
    # ---------------------------
    
    if not st.session_state.data_cleaned:
        st.markdown("## 🧹 数据清洗")
        
        # 使用session_state中的数据，确保使用最新的清洗后数据
        current_df = st.session_state.data
        
        # 检测数据问题
        if st.session_state.data_cleaning_step == "detect":
            st.info("🔍 正在检测数据问题...")
            
            # 显示当前数据状态
            st.write("**当前数据状态:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据形状", f"{current_df.shape[0]} × {current_df.shape[1]}")
            with col2:
                missing_count = current_df.isnull().sum().sum()
                st.metric("缺失值", missing_count)
            with col3:
                duplicate_count = current_df.duplicated().sum()
                st.metric("重复行", duplicate_count)
            
            # 检测问题
            issues = st.session_state.data_cleaning_handler.detect_data_issues(current_df)
            st.session_state.data_issues = issues
            
            if not issues:
                st.success("✅ 数据质量良好，未发现明显问题！")
                st.session_state.data_cleaned = True
                st.session_state.data_cleaning_step = "complete"
            else:
                st.warning(f"⚠️ 发现 {len(issues)} 个数据问题：")
                for i, (issue_type, description, details) in enumerate(issues, 1):
                    st.write(f"{i}. {description}")
                    if issue_type == "missing":
                        st.write(f"   缺失值详情: {details.to_dict()}")
                
                st.session_state.data_cleaning_step = "process"
                st.session_state.current_issue_index = 0
                st.rerun()
        
        # 处理数据问题
        elif st.session_state.data_cleaning_step == "process":
            if st.session_state.current_issue_index < len(st.session_state.data_issues):
                issue_type, description, details = st.session_state.data_issues[st.session_state.current_issue_index]
                
                st.subheader(f"🔧 处理问题 {st.session_state.current_issue_index + 1}/{len(st.session_state.data_issues)}: {description}")
                
                # 根据问题类型选择处理器
                if issue_type == "missing":
                    handler = MissingValueHandler(st.session_state.data_cleaning_handler)
                    issue_info = handler.handle_interactive(current_df)
                elif issue_type == "duplicate":
                    handler = DuplicateHandler(st.session_state.data_cleaning_handler)
                    issue_info = handler.handle_interactive(current_df)
                elif issue_type == "outlier":
                    handler = OutlierHandler(st.session_state.data_cleaning_handler)
                    issue_info = handler.handle_interactive(current_df)
                else:
                    st.error(f"未知问题类型: {issue_type}")
                    st.session_state.current_issue_index += 1
                    st.rerun()
                
                if issue_info is None:
                    st.info("✅ 此问题已自动解决")
                    st.session_state.current_issue_index += 1
                    st.rerun()
                
                # 显示建议方案
                st.write("**建议的处理方案:**")
                st.write(issue_info['suggested_solution'])
                
                # 显示对话历史
                issue_key = f"issue_{st.session_state.current_issue_index}"
                if issue_key in st.session_state.cleaning_conversation_history:
                    st.subheader("💬 对话历史")
                    for msg in st.session_state.cleaning_conversation_history[issue_key]:
                        if msg["role"] == "user":
                            st.write(f"**您:** {msg['content']}")
                        else:
                            st.info(f"**AI:** {msg['content']}")
                
                # 用户交互
                st.subheader("💭 您的选择")
                
                # 直接使用文本输入框，支持回车键提交
                user_input = st.text_input(
                    "您有什么疑问吗？如果没有疑问，请输入'继续'开始处理，或者提出自定义方案：",
                    placeholder="例如：继续、为什么这样处理、用平均值填充等",
                    help="💡 输入完成后按回车键即可提交",
                    key=f"cleaning_input_{st.session_state.current_issue_index}"
                )
                
                # 检查是否有新的输入
                submitted = user_input.strip() != ""
                
                # 处理用户输入 - 检查是否有新的输入
                input_key = f"cleaning_input_{st.session_state.current_issue_index}"
                previous_input_key = f"previous_{input_key}"
                
                # 初始化前一次输入记录
                if previous_input_key not in st.session_state:
                    st.session_state[previous_input_key] = ""
                
                # 检查是否有新的输入（与上次不同且不为空）
                has_new_input = (user_input.strip() != "" and 
                               user_input.strip() != st.session_state[previous_input_key])
                
                if has_new_input:
                    # 初始化对话历史
                    if issue_key not in st.session_state.cleaning_conversation_history:
                        st.session_state.cleaning_conversation_history[issue_key] = []
                    
                    # 添加用户消息到历史
                    st.session_state.cleaning_conversation_history[issue_key].append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # 检查是否在等待确认执行（检查添加用户消息之前的最后一条AI消息）
                    waiting_for_confirmation = False
                    if len(st.session_state.cleaning_conversation_history[issue_key]) >= 2:
                        # 获取倒数第二条消息（添加用户消息之前的最后一条AI消息）
                        last_ai_msg = st.session_state.cleaning_conversation_history[issue_key][-2]
                        if (last_ai_msg["role"] == "assistant" and 
                            ("是否确认执行" in last_ai_msg["content"] or "仍然确定要执行" in last_ai_msg["content"])):
                            waiting_for_confirmation = True
                    
                    if waiting_for_confirmation:
                        # 处理确认执行的情况
                        if any(keyword in user_input.lower() for keyword in ["确认", "执行", "是的", "好的", "同意"]):
                            # 获取自定义方案
                            custom_solution = None
                            for msg in st.session_state.cleaning_conversation_history[issue_key]:
                                if msg["role"] == "user" and any(keyword in msg["content"].lower() for keyword in ["用", "填充", "删除", "保留", "处理", "方案"]):
                                    custom_solution = msg["content"]
                                    break
                            
                            if custom_solution:
                                # 应用自定义方案 - 使用流式输出
                                placeholder = st.empty()
                                show_loading_indicator("🤖 正在执行您的自定义方案...")
                                
                                # 调试信息：执行前数据状态
                                print(f"🔧 自定义方案执行前调试:")
                                print(f"   - 执行前缺失值: {current_df.isnull().sum().sum()}")
                                print(f"   - 执行前数据形状: {current_df.shape}")
                                print(f"   - 自定义方案: {custom_solution}")
                                print(f"   - 问题类型: {issue_type}")
                                
                                if issue_type == "missing":
                                    current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                                elif issue_type == "duplicate":
                                    current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                                elif issue_type == "outlier":
                                    current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                                
                                # 调试信息：执行后数据状态
                                print(f"🔧 自定义方案执行后调试:")
                                print(f"   - 执行后缺失值: {current_df.isnull().sum().sum()}")
                                print(f"   - 执行后数据形状: {current_df.shape}")
                                print(f"   - 数据是否改变: {not current_df.equals(st.session_state.data)}")
                                
                                st.session_state.data = current_df
                                
                                # 调试信息：session状态更新后
                                print(f"🔧 Session状态更新后调试:")
                                print(f"   - Session数据缺失值: {st.session_state.data.isnull().sum().sum()}")
                                print(f"   - Session数据形状: {st.session_state.data.shape}")
                                print(f"   - 数据已被处理: True")
                                print(f"   - 数据内容样本:")
                                print(f"{st.session_state.data.head()}")
                                print(f"   - 数据类型:")
                                print(f"{st.session_state.data.dtypes}")
                                print(f"   - 缺失值位置:")
                                print(f"{st.session_state.data.isnull().sum()}")
                                
                                st.session_state.current_issue_index += 1
                                st.rerun()
                        elif any(keyword in user_input.lower() for keyword in ["取消", "不执行", "不要", "算了"]):
                            st.info("已取消执行自定义方案")
                            # 清除确认状态，继续对话
                            st.rerun()
                        else:
                            # 继续讨论 - 使用流式输出
                            # 在输入框下方显示加载指示器
                            st.markdown("---")  # 添加分隔线
                            show_loading_indicator("🤖 AI正在思考...")
                            placeholder = st.empty()
                            
                            ai_response = st.session_state.data_cleaning_handler.llm_generate_response_streaming(
                                user_input, issue_type, st.session_state.cleaning_conversation_history[issue_key], 
                                issue_info['processing_details'], placeholder
                            )
                            
                            st.session_state.cleaning_conversation_history[issue_key].append({
                                "role": "assistant",
                                "content": ai_response
                            })
                            
                            st.rerun()
                    else:
                        # 使用LLM判断用户意图
                        intent, custom_solution = st.session_state.data_cleaning_handler.llm_intent_detection_with_fallback(
                            user_input, issue_type
                        )
                        
                        if intent == "agree":
                            # 应用建议方案
                            if issue_type == "missing":
                                current_df = handler.apply_suggested_solution(current_df, issue_info['processing_details'])
                            elif issue_type == "duplicate":
                                current_df = handler.apply_suggested_solution(current_df, issue_info['processing_details'])
                            elif issue_type == "outlier":
                                current_df = handler.apply_suggested_solution(current_df, issue_info['processing_details'])
                            
                            st.success("✅ 已应用建议方案")
                            st.session_state.data = current_df
                            
                            # 验证数据更新
                            remaining_missing = st.session_state.data.isnull().sum().sum()
                            st.info(f"📊 数据更新完成，剩余缺失值: {remaining_missing}")
                            
                            # 调试信息：验证数据更新
                            print(f"🔧 建议方案数据更新调试:")
                            print(f"   - 更新后缺失值: {st.session_state.data.isnull().sum().sum()}")
                            print(f"   - 数据形状: {st.session_state.data.shape}")
                            print(f"   - 数据内容样本:")
                            print(f"{st.session_state.data.head()}")
                            print(f"   - 数据类型:")
                            print(f"{st.session_state.data.dtypes}")
                            print(f"   - 缺失值位置:")
                            print(f"{st.session_state.data.isnull().sum()}")
                            
                            st.session_state.current_issue_index += 1
                            st.rerun()
                            
                        elif intent == "custom" or custom_solution:
                            # 直接执行自定义方案，不需要确认
                            placeholder = st.empty()
                            show_loading_indicator("🤖 正在执行您的自定义方案...")
                            
                            # 调试信息：执行前数据状态
                            print(f"🔧 自定义方案执行前调试:")
                            print(f"   - 执行前缺失值: {current_df.isnull().sum().sum()}")
                            print(f"   - 执行前数据形状: {current_df.shape}")
                            print(f"   - 自定义方案: {custom_solution}")
                            print(f"   - 问题类型: {issue_type}")
                            
                            if issue_type == "missing":
                                current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                            elif issue_type == "duplicate":
                                current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                            elif issue_type == "outlier":
                                current_df = handler.apply_custom_solution_streaming(current_df, custom_solution, issue_info['processing_details'], placeholder)
                            
                            # 调试信息：执行后数据状态
                            print(f"🔧 自定义方案执行后调试:")
                            print(f"   - 执行后缺失值: {current_df.isnull().sum().sum()}")
                            print(f"   - 执行后数据形状: {current_df.shape}")
                            print(f"   - 数据是否改变: {not current_df.equals(st.session_state.data)}")
                            
                            st.session_state.data = current_df
                            
                            # 调试信息：session状态更新后
                            print(f"🔧 Session状态更新后调试:")
                            print(f"   - Session数据缺失值: {st.session_state.data.isnull().sum().sum()}")
                            print(f"   - Session数据形状: {st.session_state.data.shape}")
                            print(f"   - 数据已被处理: True")
                            print(f"   - 数据内容样本:")
                            print(f"{st.session_state.data.head()}")
                            print(f"   - 数据类型:")
                            print(f"{st.session_state.data.dtypes}")
                            print(f"   - 缺失值位置:")
                            print(f"{st.session_state.data.isnull().sum()}")
                            
                            st.session_state.current_issue_index += 1
                            st.rerun()
                            
                        elif intent == "disagree":
                            st.info("⏭️ 跳过此问题")
                            st.session_state.current_issue_index += 1
                            st.rerun()
                            
                        elif intent == "custom_solution":
                            # 评估自定义方案
                            is_reasonable, feedback = st.session_state.data_cleaning_handler.llm_evaluate_solution_with_fallback(
                                custom_solution, issue_type, issue_info['processing_details']
                            )
                            
                            if is_reasonable:
                                ai_response = f"✅ {feedback}\n\n您的自定义方案看起来合理。请确认是否执行此方案？"
                            else:
                                ai_response = f"⚠️ {feedback}\n\n您仍然确定要执行此方案吗？"
                            
                            # 添加AI回答到历史
                            st.session_state.cleaning_conversation_history[issue_key].append({
                                "role": "assistant",
                                "content": ai_response
                            })
                            
                            # 重新渲染以显示对话
                            st.rerun()
                            
                        elif intent == "question":
                            # 生成回答 - 使用流式输出
                            # 在输入框下方显示加载指示器
                            st.markdown("---")  # 添加分隔线
                            show_loading_indicator("🤖 AI正在思考...")
                            placeholder = st.empty()
                            
                            ai_response = st.session_state.data_cleaning_handler.llm_generate_response_streaming(
                                user_input, issue_type, st.session_state.cleaning_conversation_history[issue_key], 
                                issue_info['processing_details'], placeholder
                            )
                            
                            # 添加AI回答到历史
                            st.session_state.cleaning_conversation_history[issue_key].append({
                                "role": "assistant",
                                "content": ai_response
                            })
                            
                            # 重新渲染以显示对话
                            st.rerun()
                
                # 更新前一次输入记录
                if has_new_input:
                    st.session_state[previous_input_key] = user_input.strip()
                
                # 处理自定义方案的确认执行
                if issue_key in st.session_state.cleaning_conversation_history:
                    last_msg = st.session_state.cleaning_conversation_history[issue_key][-1]
                    if (last_msg["role"] == "assistant" and 
                        ("是否确认执行" in last_msg["content"] or "仍然确定要执行" in last_msg["content"])):
                        
                        st.subheader("🔧 执行确认")
                        st.info("请在上方输入框中输入您的确认决定：")
                        st.write("**选项：**")
                        st.write("- 输入 '确认' 或 '执行' 来确认执行自定义方案")
                        st.write("- 输入 '取消' 或 '不执行' 来取消执行")
                        st.write("- 输入其他内容来继续讨论")
            
            # 所有问题处理完成
            if st.session_state.current_issue_index >= len(st.session_state.data_issues):
                st.success("🎉 所有数据问题已处理完成！")
                st.session_state.data_cleaned = True
                st.session_state.data_cleaning_step = "complete"
                st.rerun()
    
    # 显示清洗后的数据
    if st.session_state.data_cleaned:
        st.success("✅ 数据清洗完成")
        st.subheader("清洗后数据预览")
        
        # 调试信息：显示当前数据状态
        print(f"🔧 清洗后数据显示调试:")
        print(f"   - 当前数据缺失值: {st.session_state.data.isnull().sum().sum()}")
        print(f"   - 当前数据形状: {st.session_state.data.shape}")
        print(f"   - 数据清洗状态: {st.session_state.data_cleaned}")
        
        # 重新检测数据问题，确保显示的是最新状态
        final_issues = st.session_state.data_cleaning_handler.detect_data_issues(st.session_state.data)
        if final_issues:
            st.warning(f"⚠️ 清洗后仍发现 {len(final_issues)} 个问题，可能需要进一步处理")
            for i, (issue_type, description, details) in enumerate(final_issues, 1):
                st.write(f"{i}. {description}")
        else:
            st.success("🎉 数据质量良好，所有问题已解决！")
        
        # 显示数据统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**清洗后数据形状:**", st.session_state.data.shape)
        with col2:
            missing_count = st.session_state.data.isnull().sum().sum()
            st.write("**剩余缺失值:**", missing_count)
        with col3:
            duplicate_count = st.session_state.data.duplicated().sum()
            st.write("**剩余重复行:**", duplicate_count)
        
        # 显示数据预览
        st.dataframe(st.session_state.data.head(10))
        
        # 控制按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重新清洗数据"):
                st.session_state.data_cleaned = False
                st.session_state.data_cleaning_step = "detect"
                st.session_state.data_issues = []
                st.session_state.current_issue_index = 0
                st.rerun()
        with col2:
            if st.button("📊 查看完整数据"):
                st.dataframe(st.session_state.data, use_container_width=True)
    
# ---------------------------
# Intelligent Chat Interface
# ---------------------------

if st.session_state.optimization_step == "chat_setup" and df is not None and st.session_state.data_cleaned:
    st.markdown("## ⚙️ 优化配置")
    
    # 配置输入变量
    st.subheader("📋 选择输入变量（特征）")
    selected_features = st.multiselect(
        "选择用于优化的输入变量：",
        options=st.session_state.numeric_columns,
        default=st.session_state.input_variables,
        help="选择您想要优化的特征变量"
    )
    
    # 配置输出变量
    st.subheader("🎯 选择输出变量（目标）")
    selected_target = st.selectbox(
        "选择优化目标变量：",
        options=st.session_state.numeric_columns,
        index=st.session_state.numeric_columns.index(st.session_state.output_variable) if st.session_state.output_variable in st.session_state.numeric_columns else 0,
        help="选择您想要优化的目标变量"
    )
    
    # 配置优化目标
    st.subheader("🎯 选择优化目标")
    optimization_goal = st.selectbox(
        "您希望：",
        options=["最大化", "最小化"],
        index=0 if st.session_state.optimization_goal == "maximize" else 1,
        help="选择您希望最大化还是最小化目标变量"
    )
    
    # 显示配置预览
    if selected_features and selected_target:
        st.subheader("📊 配置预览")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("输入变量数量", len(selected_features))
            st.write("**输入变量:**")
            for feature in selected_features:
                st.write(f"• {feature}")
        
        with col2:
            st.metric("输出变量", selected_target)
            st.write("**优化目标:**")
            st.write(f"• {optimization_goal}")
        
        with col3:
            st.metric("数据样本数", len(df))
            st.write("**数据预览:**")
            st.write(f"• 形状: {df.shape}")
    
    # 确认配置按钮
    if st.button("✅ 确认配置", type="primary"):
        if not selected_features:
            st.error("请至少选择一个输入变量")
        elif selected_target in selected_features:
            st.error("输出变量不能与输入变量重复")
        else:
            # 保存配置
            st.session_state.input_variables = selected_features
            st.session_state.output_variable = selected_target
            st.session_state.optimization_goal = "maximize" if optimization_goal == "最大化" else "minimize"
            st.session_state.config_confirmed = True
            
            # 自动设置变量边界
            variable_bounds = {}
            for var in selected_features:
                if var in df.columns:
                    col_data = df[var]
                    variable_bounds[var] = (float(col_data.min()), float(col_data.max()))
                else:
                    variable_bounds[var] = (0.0, 1.0)
            
            st.session_state.variable_bounds = variable_bounds
            st.session_state.bounds_confirmed = True
            
            # 进入优化阶段
            st.session_state.optimization_step = "optimize"
            st.rerun()
    
    
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="bayesian_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("准备帮助您进行贝叶斯优化！您想要优化什么？")

# Session state variables are already initialized above

# ---------------------------
# Chat Interface
# ---------------------------

def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "OPTIMIZATION_RESULT:" in msg.content:
                result_index = int(msg.content.split("OPTIMIZATION_RESULT:")[1])
                result = st.session_state.optimization_results[result_index]
                display_optimization_result(result)
            else:
                st.write(msg.content)

def display_optimization_result(result):
    """以美观的格式显示优化结果"""
    st.markdown("### 🎯 优化结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**最佳参数:**")
        for param, value in result['best_parameters'].items():
            st.write(f"- {param}: {value:.4f}")
    
    with col2:
        st.markdown("**最佳值:**")
        st.metric("最优结果", f"{result['best_value']:.4f}")
    
    # 显示优化历史
    if 'history' in result and len(result['history']) > 1:
        st.markdown("### 📈 优化历史")
        
        # 创建优化进度图表
        history_df = pd.DataFrame(result['history'])

# ---------------------------
# Interactive Optimization Step
# ---------------------------

if st.session_state.optimization_step == "optimize" and st.session_state.config_confirmed and st.session_state.bounds_confirmed:
    st.markdown("## 🔄 交互式优化")
    
    # 初始化代理（如果尚未完成）
    if st.session_state.bayesian_agent is None:
        st.session_state.bayesian_agent = BayesianOptimizationAgent(
            model=llm,
            n_initial_points=5,  # 使用默认值，因为模型会使用全部训练数据
            human_in_the_loop=True
        )
    
    # 显示当前状态
    st.info(f"**当前状态:** 准备生成参数建议")
    
    # 显示建议的参数（如果可用）
    if st.session_state.current_suggestion is not None:
        st.subheader("🎯 建议的参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**参数值:**")
            for i, var in enumerate(st.session_state.input_variables):
                value = st.session_state.current_suggestion[i]
                st.metric(var, f"{value:.4f}")
        
        with col2:
            st.write("**参数边界:**")
            for var in st.session_state.input_variables:
                bounds = st.session_state.variable_bounds.get(var, (0, 1))
                st.write(f"**{var}:** {bounds[0]:.2f} ~ {bounds[1]:.2f}")
        
        # 实验结果输入
        st.subheader("📊 输入实验结果")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            experimental_result = st.number_input(
                "输入您实验的结果值:",
                value=0.0,
                step=0.001,
                format="%.4f",
                key=f"result_input_{st.session_state.optimization_iteration}"
            )
        
        with col2:
            st.write("")  # 间距
            st.write("")  # 间距
            if st.button("✅ 提交结果", type="primary"):
                # 记录结果
                result_entry = {
                    "parameters": {var: st.session_state.current_suggestion[i] for i, var in enumerate(st.session_state.input_variables)},
                    "result": experimental_result,
                    "iteration": 1
                }
                
                st.session_state.optimization_results.append(result_entry)
                st.session_state.optimization_iteration += 1
                
                # 清除当前建议
                st.session_state.current_suggestion = None
                
                # 提交结果后直接完成优化
                st.session_state.optimization_step = "complete"
                
                st.rerun()
    
    else:
        # 生成参数建议
        if st.button("🎲 获取参数建议", type="primary"):
            try:
                # 准备输入数据
                selected_target = st.session_state.output_variable
                selected_features = st.session_state.input_variables
                
                if not selected_target or not selected_features:
                    st.error("请先完成配置对话")
                    st.stop()
                
                X = st.session_state.data[selected_features].dropna().values
                Y = st.session_state.data[selected_target].loc[st.session_state.data[selected_features].dropna().index].values
                
                # 使用贝叶斯优化器生成建议
                if st.session_state.optimization_results:
                    # 使用贝叶斯优化器基于历史数据生成建议
                    bounds_list = []
                    for var in selected_features:
                        bounds_list.append(st.session_state.variable_bounds.get(var, (0, 1)))
                    
                    # 创建临时优化器
                    from ai_data_science_team.agents.bayesian_optimization_agent import BayesianOptimizer
                    temp_optimizer = BayesianOptimizer(bounds_list)
                    
                    # 使用历史结果初始化优化器
                    for result in st.session_state.optimization_results:
                        params = [result["parameters"][var] for var in selected_features]
                        temp_optimizer.update(params, result["result"])
                    
                    # 生成下一个建议点
                    suggestion = temp_optimizer.suggest_next_point()
                    st.session_state.current_suggestion = suggestion
                else:
                    # 第一次迭代 - 使用现有数据点或随机点
                    if len(X) > 0:
                        # 使用现有数据点
                        suggestion = X[0].tolist()  # 使用第一个数据点
                    else:
                        # 随机建议
                        import random
                        suggestion = []
                        for var in selected_features:
                            bounds = st.session_state.variable_bounds.get(var, (0, 1))
                            suggestion.append(random.uniform(bounds[0], bounds[1]))
                    st.session_state.current_suggestion = suggestion
                
                st.rerun()
                
            except Exception as e:
                st.error(f"生成建议时出错: {e}")
    
    # 显示优化历史
    if st.session_state.optimization_results:
        st.subheader("📈 优化历史")
        
        # 创建历史数据框
        history_data = []
        for result in st.session_state.optimization_results:
            row = {"迭代": result["iteration"], "结果": result["result"]}
            for param, value in result["parameters"].items():
                row[param] = value
            history_data.append(row)
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # 显示迄今为止的最佳结果
            if st.session_state.optimization_goal == "maximize":
                best_result = max(st.session_state.optimization_results, key=lambda x: x["result"])
            else:
                best_result = min(st.session_state.optimization_results, key=lambda x: x["result"])
            
            st.success(f"🏆 **迄今为止最佳结果:** {best_result['result']:.4f}")
    
    # 控制按钮
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 重新开始"):
            st.session_state.optimization_step = "chat_setup"
            st.session_state.config_confirmed = False
            st.session_state.bounds_confirmed = False
            st.session_state.chat_messages = []
            st.session_state.current_suggestion = None
            st.session_state.optimization_results = []
            st.rerun()
    
    with col2:
        if st.button("📊 查看结果"):
            st.session_state.optimization_step = "complete"
            st.rerun()
    
    st.stop()

# ---------------------------
# Optimization Complete Step
# ---------------------------

if st.session_state.optimization_step == "complete":
    st.markdown("## 🎉 优化完成！")
    
    if st.session_state.optimization_results:
        # 显示最终结果
        st.subheader("📊 最终结果")
        
        # 找到最佳结果
        if st.session_state.optimization_goal == "maximize":
            best_result = max(st.session_state.optimization_results, key=lambda x: x["result"])
        else:
            best_result = min(st.session_state.optimization_results, key=lambda x: x["result"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("最佳结果", f"{best_result['result']:.4f}")
            st.metric("实验次数", len(st.session_state.optimization_results))
        
        with col2:
            st.write("**最佳参数:**")
            for param, value in best_result["parameters"].items():
                st.write(f"- **{param}:** {value:.4f}")
        
        # 显示优化进度图表
        if len(st.session_state.optimization_results) > 1:
            st.subheader("📈 优化进度")
            

# ---------------------------
# Sidebar Information
# ---------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 数据状态")

if st.session_state.data is not None:
    if st.session_state.data_cleaned:
        st.sidebar.success("✅ 数据清洗完成")
        st.sidebar.metric("数据形状", f"{st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
    else:
        st.sidebar.warning("🧹 数据清洗中...")
        if st.session_state.data_issues:
            st.sidebar.metric("发现问题", len(st.session_state.data_issues))
else:
    st.sidebar.info("📁 等待数据上传")

st.sidebar.markdown("---")
st.sidebar.markdown("### 优化状态")

if st.session_state.optimization_config.get('config_confirmed', False):
    st.sidebar.success("✅ 优化配置已确认")
    
    if 'optimization_state' in st.session_state:
        current_iter = st.session_state.optimization_state['current_iteration']
        max_iter = st.session_state.optimization_state['max_iterations']
        st.sidebar.metric("进度", f"{current_iter}/{max_iter}")
        
        history_count = len(st.session_state.optimization_state['optimization_history'])
        st.sidebar.metric("完成实验", history_count)
else:
    st.sidebar.info("⏳ 等待配置")

st.sidebar.markdown("---")
st.sidebar.markdown("### 使用说明")
st.sidebar.markdown("""
1. 📁 上传包含实验数据的CSV文件
2. 🧹 自动检测并处理数据问题
3. 🎯 按步骤配置优化参数
4. ✅ 确认配置开始优化
5. 🧪 根据建议进行实验
6. 📊 输入实验结果
7. 🔄 重复直到找到最优解
""")
