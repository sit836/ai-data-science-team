# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Interactive Data Cleaning Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any, Optional
import operator
import time
import re

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

import os
import json
import pandas as pd
import numpy as np

from IPython.display import Markdown

from ai_data_science_team.templates import(
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from ai_data_science_team.parsers.parsers import PythonOutputParser
from ai_data_science_team.utils.regex import (
    relocate_imports_inside_function, 
    add_comments_to_top, 
    format_agent_name, 
    format_recommended_steps, 
    get_generic_summary,
)
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.utils.logging import log_ai_function

# Setup
AGENT_NAME = "interactive_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class InteractiveDataCleaningAgent(BaseAgent):
    """
    Creates an interactive data cleaning agent that can process datasets with user confirmation.
    The agent detects data issues, presents solutions to users, and allows for custom modifications.
    
    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data cleaning function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "interactive_data_cleaner.py".
    function_name : str, optional
        Name of the generated data cleaning function. Defaults to "interactive_data_cleaner".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. Defaults to True.
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.
    max_conversation_rounds : int, optional
        Maximum number of conversation rounds. Defaults to 10.
    llm_timeout : int, optional
        LLM timeout in seconds. Defaults to 10.
    fallback_to_auto : bool, optional
        Whether to fallback to automatic cleaning if LLM fails. Defaults to True.
    """
    
    def __init__(
        self, 
        model, 
        n_samples=30, 
        log=False, 
        log_path=None, 
        file_name="interactive_data_cleaner.py", 
        function_name="interactive_data_cleaner",
        overwrite=True, 
        checkpointer: Checkpointer = None,
        max_conversation_rounds=10,
        llm_timeout=10,
        fallback_to_auto=True
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "checkpointer": checkpointer,
            "max_conversation_rounds": max_conversation_rounds,
            "llm_timeout": llm_timeout,
            "fallback_to_auto": fallback_to_auto
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create the compiled graph for the interactive data cleaning agent."""
        self.response = None
        return make_interactive_data_cleaning_agent(**self._params)

    async def ainvoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """Asynchronously invokes the agent."""
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None
    
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """Invokes the agent."""
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

    def get_data_cleaned(self):
        """Retrieves the cleaned data."""
        if self.response:
            return pd.DataFrame(self.response.get("data_cleaned"))
        
    def get_data_raw(self):
        """Retrieves the raw data."""
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self, markdown=False):
        """Retrieves the agent's pipeline function."""
        if self.response:
            if markdown:
                return Markdown(f"```python\n{self.response.get('data_cleaner_function')}\n```")
            else:
                return self.response.get("data_cleaner_function")
            
    def get_recommended_cleaning_steps(self, markdown=False):
        """Retrieves the agent's recommended cleaning steps"""
        if self.response:
            if markdown:
                return Markdown(self.response.get('recommended_steps'))
            else:
                return self.response.get('recommended_steps')

    def get_workflow_summary(self, markdown=False):
        """Retrieves the agent's workflow summary."""
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            else:
                return summary


def make_interactive_data_cleaning_agent(
    model, 
    n_samples=30, 
    log=False, 
    log_path=None, 
    file_name="interactive_data_cleaner.py",
    function_name="interactive_data_cleaner",
    overwrite=True, 
    checkpointer: Checkpointer = None,
    max_conversation_rounds=10,
    llm_timeout=10,
    fallback_to_auto=True
):
    """
    Creates an interactive data cleaning agent that can be run on a dataset.
    The agent detects data issues, presents solutions to users, and allows for custom modifications.
    """
    llm = model
    
    if checkpointer is None:
        print("Interactive data cleaning requires a checkpointer. Setting to MemorySaver().")
        checkpointer = MemorySaver()
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define GraphState for the interactive cleaning agent
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_cleaned: dict
        all_datasets_summary: str
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_file_name: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int
        
        # Interactive cleaning specific state
        detected_issues: list
        current_issue_index: int
        conversation_rounds: int
        max_conversation_rounds: int
        processing_type: str
        processing_details: dict
        agreed: bool
        custom_solution: str
        llm_timeout: int
        fallback_to_auto: bool

    def detect_data_issues(state: GraphState):
        """Detect all data issues that need cleaning."""
        print(format_agent_name(AGENT_NAME))
        print("    * DETECTING DATA ISSUES")

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        issues = []
        
        # Detect missing values
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        if len(missing_columns) > 0:
            issues.append(('missing', f"发现 {len(missing_columns)} 列有缺失值", missing_columns.to_dict()))

        # Detect duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(('duplicate', f"发现 {duplicate_count} 个重复行", {"count": duplicate_count}))

        # Detect outliers (simple detection for numeric columns)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            has_outliers = False
            outlier_info = {}
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
                        outlier_info[col] = {
                            'outliers': outliers,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }

            if has_outliers:
                issues.append(('outlier', f"发现 {len(numeric_cols)} 个数值列需要异常值检测", outlier_info))

        return {
            "detected_issues": issues,
            "current_issue_index": 0,
            "conversation_rounds": 0,
            "max_conversation_rounds": max_conversation_rounds,
            "llm_timeout": llm_timeout,
            "fallback_to_auto": fallback_to_auto
        }

    def interactive_issue_handler(state: GraphState):
        """Handle the current data issue interactively."""
        print("    * INTERACTIVE ISSUE HANDLER")
        
        issues = state.get("detected_issues", [])
        current_index = state.get("current_issue_index", 0)
        
        if current_index >= len(issues):
            # No more issues to handle
            return {
                "agreed": True,
                "processing_type": "completed",
                "processing_details": {"status": "all_issues_handled"}
            }
        
        issue_type, description, details = issues[current_index]
        
        # Generate recommended solution
        if issue_type == 'missing':
            solution = generate_missing_value_solution(details)
        elif issue_type == 'duplicate':
            solution = generate_duplicate_solution(details)
        elif issue_type == 'outlier':
            solution = generate_outlier_solution(details)
        else:
            solution = f"处理 {issue_type} 问题"
        
        return {
            "processing_type": issue_type,
            "processing_details": details,
            "recommended_steps": solution,
            "agreed": False,
            "custom_solution": ""
        }

    def interactive_human_review(state: GraphState) -> Command[Literal["interactive_issue_handler", "create_cleaning_code", "interactive_issue_handler"]]:
        """Interactive human review with LLM-powered intent detection."""
        print("    * INTERACTIVE HUMAN REVIEW")
        
        processing_type = state.get("processing_type")
        processing_details = state.get("processing_details", {})
        recommended_steps = state.get("recommended_steps", "")
        
        # Create the prompt for user interaction
        prompt_text = f"""
🤖 数据清洗建议

处理类型: {processing_type}
问题描述: {recommended_steps}

处理详情: {json.dumps(processing_details, ensure_ascii=False, indent=2)}

请选择您的操作：
1. 输入 'yes' 或 '继续' - 同意当前方案
2. 输入 'no' 或 '取消' - 拒绝当前方案  
3. 输入具体问题 - 我会为您解答
4. 输入自定义方案 - 我会评估并执行您的方案

您的选择: """

        # Use the existing human review function but with custom logic
        user_input = interrupt(value=prompt_text)
        
        # Use LLM to classify user intent
        intent, custom_solution = classify_user_intent_with_fallback(
            user_input, processing_type, state.get("llm_timeout", 10)
        )
        
        if intent == "agree":
            return Command(goto="create_cleaning_code", update={"agreed": True})
        elif intent == "disagree":
            # Skip this issue and move to next
            return Command(goto="interactive_issue_handler", update={
                "current_issue_index": state.get("current_issue_index", 0) + 1,
                "agreed": False
            })
        elif intent == "custom_solution":
            # Evaluate and potentially apply custom solution
            is_reasonable, feedback = evaluate_solution_with_fallback(
                custom_solution, processing_type, processing_details, state.get("llm_timeout", 10)
            )
            
            if is_reasonable:
                return Command(goto="create_cleaning_code", update={
                    "agreed": True,
                    "custom_solution": custom_solution,
                    "recommended_steps": f"自定义方案: {custom_solution}\n评估: {feedback}"
                })
            else:
                # Show feedback and ask again
                new_prompt = f"⚠️ 方案评估: {feedback}\n\n请重新考虑您的方案或选择其他选项: "
                user_input2 = interrupt(value=new_prompt)
                intent2, custom_solution2 = classify_user_intent_with_fallback(
                    user_input2, processing_type, state.get("llm_timeout", 10)
                )
                
                if intent2 == "agree":
                    return Command(goto="create_cleaning_code", update={
                        "agreed": True,
                        "custom_solution": custom_solution,
                        "recommended_steps": f"用户坚持的自定义方案: {custom_solution}"
                    })
                else:
                    return Command(goto="interactive_issue_handler", update={
                        "current_issue_index": state.get("current_issue_index", 0) + 1,
                        "agreed": False
                    })
        else:  # question
            # Generate answer and ask again
            answer = generate_answer_with_fallback(
                user_input, processing_type, processing_details, state.get("llm_timeout", 10)
            )
            new_prompt = f"🤖 {answer}\n\n请重新选择您的操作: "
            user_input2 = interrupt(value=new_prompt)
            intent2, custom_solution2 = classify_user_intent_with_fallback(
                user_input2, processing_type, state.get("llm_timeout", 10)
            )
            
            if intent2 == "agree":
                return Command(goto="create_cleaning_code", update={"agreed": True})
            elif intent2 == "disagree":
                return Command(goto="interactive_issue_handler", update={
                    "current_issue_index": state.get("current_issue_index", 0) + 1,
                    "agreed": False
                })
            else:
                return Command(goto="interactive_issue_handler", update={
                    "current_issue_index": state.get("current_issue_index", 0) + 1,
                    "agreed": False
                })

    def create_cleaning_code(state: GraphState):
        """Create cleaning code based on the agreed solution."""
        print("    * CREATE CLEANING CODE")
        
        processing_type = state.get("processing_type")
        processing_details = state.get("processing_details", {})
        custom_solution = state.get("custom_solution", "")
        recommended_steps = state.get("recommended_steps", "")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        # Generate the cleaning code
        if custom_solution:
            # Use custom solution
            cleaning_code = generate_custom_cleaning_code(
                custom_solution, processing_type, processing_details, df
            )
        else:
            # Use recommended solution
            cleaning_code = generate_recommended_cleaning_code(
                processing_type, processing_details, df
            )
        
        # Add comments and format
        cleaning_code = add_comments_to_top(cleaning_code, agent_name=AGENT_NAME)
        
        # For logging: store the code generated
        file_path, file_name_2 = log_ai_function(
            response=cleaning_code,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )
        
        return {
            "data_cleaner_function": cleaning_code,
            "data_cleaner_function_path": file_path,
            "data_cleaner_file_name": file_name_2,
            "data_cleaner_function_name": function_name,
        }

    def execute_cleaning_code(state: GraphState):
        """Execute the cleaning code."""
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during interactive data cleaning: "
        )

    def fix_cleaning_code(state: GraphState):
        """Fix cleaning code if there are errors."""
        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=llm,
            prompt_template="""
            You are an Interactive Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.
            
            Make sure to only return the function definition for {function_name}().
            
            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
            
            This is the broken code (please fix): 
            {code_snippet}

            Last Known Error:
            {error}
            """,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_cleaner_function_path"),
            function_name=state.get("data_cleaner_function_name"),
        )

    def check_more_issues(state: GraphState):
        """Check if there are more issues to handle."""
        current_index = state.get("current_issue_index", 0)
        issues = state.get("detected_issues", [])
        
        if current_index < len(issues):
            return "interactive_issue_handler"
        else:
            return "report_agent_outputs"

    def report_agent_outputs(state: GraphState):
        """Report the final outputs."""
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_cleaner_function",
                "data_cleaner_function_path",
                "data_cleaner_function_name",
                "data_cleaner_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Interactive Data Cleaning Agent Outputs"
        )

    # Helper functions for solution generation and LLM interaction
    def generate_missing_value_solution(missing_columns):
        """Generate solution for missing values."""
        solution_parts = []
        for col, count in missing_columns.items():
            solution_parts.append(f"列 '{col}' 有 {count} 个缺失值")
        return f"缺失值处理方案: 检测到 {len(missing_columns)} 列有缺失值。建议根据数据类型和缺失比例选择填充方法。"

    def generate_duplicate_solution(duplicate_details):
        """Generate solution for duplicates."""
        count = duplicate_details.get("count", 0)
        return f"重复值处理方案: 发现 {count} 个重复行。建议删除重复行，保留第一个出现的记录。"

    def generate_outlier_solution(outlier_details):
        """Generate solution for outliers."""
        return f"异常值处理方案: 检测到数值列中的异常值。建议根据异常值比例采用不同的处理策略。"

    def classify_user_intent_with_fallback(user_input, processing_type, timeout):
        """Classify user intent with fallback."""
        try:
            intent_prompt = f"""请分析用户意图：

用户输入: "{user_input}"
处理类型: {processing_type}

可选意图：
1. agree - 用户同意当前方案（包含：继续、同意、好的、没问题、行、可以、yes、y）
2. disagree - 用户拒绝当前方案（包含：不、不要、取消、停止、退出、拒绝、no、n）
3. question - 用户提出问题（包含：为什么、怎么、如何、解释、说明、什么）
4. custom_solution - 用户提出自定义方案（包含：用、改成、建议、自定义、我想、我要）

返回格式：意图|方案内容（只有custom_solution时才需要方案内容）"""

            response = llm.invoke(intent_prompt)
            result = response.content.strip()
            
            if "|" in result:
                intent, custom_solution = result.split("|", 1)
                intent = intent.strip().lower()
                custom_solution = custom_solution.strip()
                
                valid_intents = ["agree", "disagree", "question", "custom_solution"]
                if intent in valid_intents:
                    return intent, custom_solution
            
            # Fallback to simple keyword matching
            return fallback_intent_detection(user_input), user_input
            
        except Exception as e:
            print(f"❌ LLM意图识别失败: {e}")
            return fallback_intent_detection(user_input), user_input

    def fallback_intent_detection(user_input):
        """Fallback intent detection using keyword matching."""
        user_input_lower = user_input.lower().strip()
        
        agree_keywords = ['继续', '同意', '好的', '没问题', '行', '可以', 'ok', 'yes', 'y', '是']
        disagree_keywords = ['不', '不要', '取消', '停止', '退出', 'no', 'n', '拒绝', '不同意']
        custom_keywords = ['用', '填充', '删除', '保留', '处理', '方案', '改成', '建议']
        
        if any(keyword in user_input_lower for keyword in agree_keywords):
            return "agree"
        elif any(keyword in user_input_lower for keyword in disagree_keywords):
            return "disagree"
        elif any(keyword in user_input_lower for keyword in custom_keywords):
            return "custom_solution"
        else:
            return "question"

    def evaluate_solution_with_fallback(custom_solution, processing_type, processing_details, timeout):
        """Evaluate custom solution with fallback."""
        try:
            evaluation_prompt = f"""作为数据清洗专家，请评估用户提出的处理方案。

处理类型: {processing_type}
处理详情: {processing_details}
用户方案: "{custom_solution}"

请评估：
1. 方案的技术合理性
2. 潜在的风险或问题
3. 改进建议（如有）

返回格式：合理与否(true/false)|评估反馈"""

            response = llm.invoke(evaluation_prompt)
            result = response.content.strip()
            
            if "|" in result:
                is_reasonable_str, feedback = result.split("|", 1)
                is_reasonable = is_reasonable_str.strip().lower() == "true"
                return is_reasonable, feedback.strip()
            
            return True, "无法评估，将执行您的方案"
            
        except Exception as e:
            print(f"❌ LLM方案评估失败: {e}")
            return True, "评估不可用，将执行您的方案"

    def generate_answer_with_fallback(user_input, processing_type, processing_details, timeout):
        """Generate answer with fallback."""
        try:
            answer_prompt = f"""作为数据清洗专家，请回答用户的问题。

处理类型: {processing_type}
处理详情: {processing_details}
用户问题: "{user_input}"

请用中文友好、专业地回答用户的问题。"""

            response = llm.invoke(answer_prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ LLM回答生成失败: {e}")
            return f"关于您的问题'{user_input}'，我会尽力帮助您。请选择继续、取消或提出自定义方案。"

    def generate_custom_cleaning_code(custom_solution, processing_type, processing_details, df):
        """Generate cleaning code for custom solution."""
        try:
            execution_prompt = f"""请解析并执行以下数据清洗方案：

数据集信息:
- 形状: {df.shape}
- 列名: {list(df.columns)}
- 处理类型: {processing_type}
- 处理详情: {processing_details}

自定义方案: "{custom_solution}"

请生成Python代码来执行这个方案，只返回代码部分："""

            response = llm.invoke(execution_prompt)
            code = response.content.strip()
            
            # Wrap in function
            function_code = f"""def {function_name}(data_raw):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame.from_dict(data_raw)
    
    {code}
    
    return df"""
            
            return function_code
            
        except Exception as e:
            print(f"❌ 自定义方案代码生成失败: {e}")
            return generate_recommended_cleaning_code(processing_type, processing_details, df)

    def generate_recommended_cleaning_code(processing_type, processing_details, df):
        """Generate cleaning code for recommended solution."""
        if processing_type == 'missing':
            return generate_missing_value_cleaning_code(processing_details, df)
        elif processing_type == 'duplicate':
            return generate_duplicate_cleaning_code(processing_details, df)
        elif processing_type == 'outlier':
            return generate_outlier_cleaning_code(processing_details, df)
        else:
            return f"""def {function_name}(data_raw):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame.from_dict(data_raw)
    
    # 基础数据清洗
    # TODO: 添加具体的清洗逻辑
    
    return df"""

    def generate_missing_value_cleaning_code(processing_details, df):
        """Generate code for missing value cleaning."""
        code_parts = ["df = pd.DataFrame.from_dict(data_raw)"]
        
        for col, count in processing_details.items():
            if col in df.columns:
                col_type = df[col].dtype
                if col_type in ['int64', 'float64']:
                    code_parts.append(f"# 数值列 {col} 使用中位数填充")
                    code_parts.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
                else:
                    code_parts.append(f"# 分类列 {col} 使用众数填充")
                    code_parts.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0] if not df['{col}'].mode().empty else 'Unknown')")
        
        code_parts.append("return df")
        
        function_code = f"""def {function_name}(data_raw):
    import pandas as pd
    import numpy as np
    
    {chr(10).join(code_parts)}"""
        
        return function_code

    def generate_duplicate_cleaning_code(processing_details, df):
        """Generate code for duplicate cleaning."""
        function_code = f"""def {function_name}(data_raw):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame.from_dict(data_raw)
    
    # 删除重复行，保留第一个出现的
    initial_count = len(df)
    df = df.drop_duplicates(keep='first')
    removed_count = initial_count - len(df)
    
    print(f"删除了 {{removed_count}} 个重复行")
    
    return df"""
        
        return function_code

    def generate_outlier_cleaning_code(processing_details, df):
        """Generate code for outlier cleaning."""
        function_code = f"""def {function_name}(data_raw):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame.from_dict(data_raw)
    
    # 处理异常值
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 缩尾处理
                df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df"""
        
        return function_code

    # Node functions dictionary
    node_functions = {
        "detect_data_issues": detect_data_issues,
        "interactive_issue_handler": interactive_issue_handler,
        "interactive_human_review": interactive_human_review,
        "create_cleaning_code": create_cleaning_code,
        "execute_cleaning_code": execute_cleaning_code,
        "fix_cleaning_code": fix_cleaning_code,
        "check_more_issues": check_more_issues,
        "report_agent_outputs": report_agent_outputs,
    }

    # Create the workflow using the existing template
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="detect_data_issues",
        create_code_node_name="interactive_issue_handler",
        execute_code_node_name="execute_cleaning_code",
        fix_code_node_name="fix_cleaning_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_cleaner_error",
        human_in_the_loop=True,
        human_review_node_name="interactive_human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=False,
        bypass_explain_code=True,
        agent_name=AGENT_NAME,
    )

    return app
