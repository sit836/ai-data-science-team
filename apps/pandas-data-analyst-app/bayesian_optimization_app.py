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
from openai import OpenAI

from ai_data_science_team.agents.bayesian_optimization_agent import BayesianOptimizationAgent

# * APP INPUTS ----

MODEL_LIST = ["deepseek-v3", "deepseek-chat", "gpt-3.5-turbo", "gpt-4"]
TITLE = "Bayesian Optimization AI Copilot"

load_dotenv()

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="🎯",
    layout="wide"
)
st.title(TITLE)

st.markdown("""
and the AI agent will help you find optimal parameters using Bayesian optimization.
""")

with st.expander("Example Optimization Problems", expanded=False):
    st.write(
        """
        ##### Optimization Examples:
        
        - Find the maximum value of f(x1, x2) = sin(x1) * cos(x2)
        - Optimize hyperparameters for a machine learning model
        - Find optimal process parameters for manufacturing
        - Optimize chemical reaction conditions
        - Find the best configuration for a system
        """
    )

# ---------------------------
# OpenAI API Key Entry and Test
# ---------------------------

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your OpenAI API key is required for the app to function.",
)

openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"],
                    base_url=openai_api_data['base_url'] if openai_api_data['base_url'] else None)

    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()

# * OpenAI Model Selection

model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)

llm = ChatOpenAI(
    model=model_option, 
    api_key=st.session_state["OPENAI_API_KEY"],
    base_url=openai_api_data['base_url'] if openai_api_data['base_url'] else None
)

# ---------------------------
# Data Input Section
# ---------------------------

st.markdown("## Data Input")

# Create two columns for data input options
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with experimental data", 
        type=["csv"],
        help="Upload a CSV file with columns for input features and target values"
    )

with col2:
        np.random.seed(42)
        n_points = 20
        
        # Create a 2D optimization problem: f(x1, x2) = sin(x1) * cos(x2)
        x1 = np.random.uniform(0, 2*np.pi, n_points)
        x2 = np.random.uniform(0, 2*np.pi, n_points)
        y = np.sin(x1) * np.cos(x2) + np.random.normal(0, 0.1, n_points)
        
        # Create DataFrame
        df = pd.DataFrame({
            '特征1': x1,
            '特征2': x2,
            '目标值': y
        })
        

# Load data
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_data = df
elif hasattr(st.session_state, 'uploaded_data'):
    df = st.session_state.uploaded_data

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Show data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Shape:**", df.shape)
        st.write("**All Columns:**", list(df.columns))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write("**Numeric Columns:**", numeric_cols)
    with col2:
        st.write("**Target Statistics:**")
        if '目标值' in df.columns:
            st.write(df['目标值'].describe())
        elif len(numeric_cols) > 0:
            st.write("**Numeric Data Statistics:**")
            st.write(df[numeric_cols].describe())
    
# ---------------------------
# Optimization Configuration
# ---------------------------

st.markdown("## Optimization Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Optimization Settings")
    max_iterations = st.slider("Maximum Iterations", 1, 20, 10)
    n_initial_points = st.slider("Initial Points", 3, 10, 5)
    human_in_the_loop = st.checkbox("Enable Human-in-the-Loop", value=True)
    
    # Prompt optimization goal after variable selection
    optimization_goal = st.selectbox(
        "Optimization Goal",
        ["maximize", "minimize"],
        help="Whether to maximize or minimize the target function"
    )

with col2:
    st.subheader("Variable Bounds")
    if df is not None:
        # Use selected features for bounds
        bounds = {}
        
        if 'selected_feature_cols' in st.session_state and st.session_state.selected_feature_cols:
            for col in st.session_state.selected_feature_cols:
                col_min, col_max = st.slider(
                    f"{col} bounds",
                    min_value=float(df[col].min() - 1),
                    max_value=float(df[col].max() + 1),
                    value=(float(df[col].min()), float(df[col].max())),
                    step=0.1
                )
                bounds[col] = (col_min, col_max)
        else:
            st.warning("请选择用于优化的数值特征列")
        
        st.session_state.variable_bounds = bounds

# ---------------------------
# Optimization Confirmation Step
# ---------------------------

if st.session_state.optimization_step == "setup" and 'selected_feature_cols' in st.session_state and st.session_state.selected_feature_cols:
    st.markdown("## Optimization Setup Confirmation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Current Settings")
        st.write(f"**Input Features:** {st.session_state.selected_feature_cols}")
        st.write(f"**Target Variable:** {st.session_state.get('selected_target_col', 'Not selected')}")
        st.write(f"**Optimization Goal:** {optimization_goal}")
        st.write(f"**Max Iterations:** {max_iterations}")
        st.write(f"**Initial Points:** {n_initial_points}")
        st.write(f"**Human-in-the-Loop:** {'Enabled' if human_in_the_loop else 'Disabled'}")
    
    with col2:
        st.subheader("🎯 Variable Bounds")
        if st.session_state.variable_bounds:
            for var, bounds in st.session_state.variable_bounds.items():
                st.write(f"**{var}:** {bounds[0]:.2f} ~ {bounds[1]:.2f}")
        else:
            st.warning("No bounds set")
    
    # Confirmation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Confirm & Start Optimization", type="primary"):
            st.session_state.optimization_settings_confirmed = True
            st.session_state.optimization_step = "optimize"
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Settings"):
            st.session_state.optimization_step = "setup"
            st.session_state.optimization_settings_confirmed = False
            st.session_state.optimization_iteration = 0
            st.rerun()
    
    with col3:
        if st.button("📊 View Data Summary"):
            st.subheader("Data Summary")
            if df is not None:
                selected_cols = st.session_state.selected_feature_cols + [st.session_state.get('selected_target_col')]
                st.dataframe(df[selected_cols].describe())
    
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="bayesian_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Ready to help you with Bayesian optimization! What would you like to optimize?")

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = []

if "bayesian_agent" not in st.session_state:
    st.session_state.bayesian_agent = None

# Add interactive optimization state
if "optimization_step" not in st.session_state:
    st.session_state.optimization_step = "setup"  # setup, confirm, optimize, complete

if "current_suggestion" not in st.session_state:
    st.session_state.current_suggestion = None

if "optimization_settings_confirmed" not in st.session_state:
    st.session_state.optimization_settings_confirmed = False

if "optimization_iteration" not in st.session_state:
    st.session_state.optimization_iteration = 0

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
    """Display optimization result in a nice format"""
    st.markdown("### 🎯 Optimization Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best Parameters:**")
        for param, value in result['best_parameters'].items():
            st.write(f"- {param}: {value:.4f}")
    
    with col2:
        st.markdown("**Best Value:**")
        st.metric("Optimal Result", f"{result['best_value']:.4f}")
    
    # Show optimization history
    if 'history' in result and len(result['history']) > 1:
        st.markdown("### 📈 Optimization History")
        
        # Create optimization progress chart
        history_df = pd.DataFrame(result['history'])

# ---------------------------
# Interactive Optimization Step
# ---------------------------

if st.session_state.optimization_step == "optimize" and st.session_state.optimization_settings_confirmed:
    st.markdown("## 🔄 Interactive Optimization")
    
    # Initialize agent if not done
    if st.session_state.bayesian_agent is None:
        st.session_state.bayesian_agent = BayesianOptimizationAgent(
            model=llm,
            n_initial_points=n_initial_points,
            human_in_the_loop=human_in_the_loop
        )
    
    # Show current iteration
    st.info(f"**Current Iteration:** {st.session_state.optimization_iteration + 1} / {max_iterations}")
    
    # Show suggested parameters if available
    if st.session_state.current_suggestion is not None:
        st.subheader("🎯 Suggested Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parameter Values:**")
            for i, var in enumerate(st.session_state.selected_feature_cols):
                value = st.session_state.current_suggestion[i]
                st.metric(var, f"{value:.4f}")
        
        with col2:
            st.write("**Parameter Bounds:**")
            for var in st.session_state.selected_feature_cols:
                bounds = st.session_state.variable_bounds.get(var, (0, 1))
                st.write(f"**{var}:** {bounds[0]:.2f} ~ {bounds[1]:.2f}")
        
        # Input for experimental result
        st.subheader("📊 Enter Experimental Result")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            experimental_result = st.number_input(
                "Enter the result value from your experiment:",
                value=0.0,
                step=0.001,
                format="%.4f",
                key=f"result_input_{st.session_state.optimization_iteration}"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("✅ Submit Result", type="primary"):
                # Record the result
                result_entry = {
                    "parameters": {var: st.session_state.current_suggestion[i] for i, var in enumerate(st.session_state.selected_feature_cols)},
                    "result": experimental_result,
                    "iteration": st.session_state.optimization_iteration + 1
                }
                
                st.session_state.optimization_results.append(result_entry)
                st.session_state.optimization_iteration += 1
                
                # Clear current suggestion to get next one
                st.session_state.current_suggestion = None
                
                # Check if we should continue
                if st.session_state.optimization_iteration >= max_iterations:
                    st.session_state.optimization_step = "complete"
                else:
                    st.session_state.optimization_step = "optimize"
                
                st.rerun()
    
    else:
        # Generate next suggestion
        if st.button("🎲 Get Next Parameter Suggestion", type="primary"):
            try:
                # Prepare input data
                selected_target = st.session_state.get('selected_target_col')
                selected_features = st.session_state.get('selected_feature_cols', [])
                
                if not selected_target or not selected_features:
                    st.error("请先选择输入特征列和目标列")
                    st.stop()
                
                X = df[selected_features].dropna().values
                Y = df[selected_target].loc[df[selected_features].dropna().index].values
                
                # Use existing results to initialize optimizer
                if st.session_state.optimization_results:
                    # Use previous results to suggest next point
                    # This is a simplified version - in practice, you'd use the actual Bayesian optimizer
                    import random
                    suggestion = []
                    for var in selected_features:
                        bounds = st.session_state.variable_bounds.get(var, (0, 1))
                        # Simple random suggestion within bounds
                        suggestion.append(random.uniform(bounds[0], bounds[1]))
                    st.session_state.current_suggestion = suggestion
                else:
                    # First iteration - use initial data or random points
                    if len(X) > 0:
                        # Use existing data points
                        suggestion = X[st.session_state.optimization_iteration % len(X)].tolist()
                    else:
                        # Random suggestion
                        suggestion = []
                        for var in selected_features:
                            bounds = st.session_state.variable_bounds.get(var, (0, 1))
                            suggestion.append(random.uniform(bounds[0], bounds[1]))
                    st.session_state.current_suggestion = suggestion
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating suggestion: {e}")
    
    # Show optimization history
    if st.session_state.optimization_results:
        st.subheader("📈 Optimization History")
        
        # Create history DataFrame
        history_data = []
        for result in st.session_state.optimization_results:
            row = {"Iteration": result["iteration"], "Result": result["result"]}
            for param, value in result["parameters"].items():
                row[param] = value
            history_data.append(row)
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Show best result so far
            if optimization_goal == "maximize":
                best_result = max(st.session_state.optimization_results, key=lambda x: x["result"])
            else:
                best_result = min(st.session_state.optimization_results, key=lambda x: x["result"])
            
            st.success(f"🏆 **Best Result So Far:** {best_result['result']:.4f} (Iteration {best_result['iteration']})")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏹️ Stop Optimization"):
            st.session_state.optimization_step = "complete"
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Optimization"):
            st.session_state.optimization_step = "setup"
            st.session_state.optimization_settings_confirmed = False
            st.session_state.optimization_iteration = 0
            st.session_state.current_suggestion = None
            st.session_state.optimization_results = []
            st.rerun()
    
    with col3:
        if st.button("📊 View Results"):
            st.session_state.optimization_step = "complete"
            st.rerun()
    
    st.stop()

# ---------------------------
# Optimization Complete Step
# ---------------------------

if st.session_state.optimization_step == "complete":
    st.markdown("## 🎉 Optimization Complete!")
    
    if st.session_state.optimization_results:
        # Show final results
        st.subheader("📊 Final Results")
        
        # Find best result
        if optimization_goal == "maximize":
            best_result = max(st.session_state.optimization_results, key=lambda x: x["result"])
        else:
            best_result = min(st.session_state.optimization_results, key=lambda x: x["result"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Result", f"{best_result['result']:.4f}")
            st.metric("Total Iterations", len(st.session_state.optimization_results))
        
        with col2:
            st.write("**Best Parameters:**")
            for param, value in best_result["parameters"].items():
                st.write(f"- **{param}:** {value:.4f}")
        
        # Show optimization progress chart
        if len(st.session_state.optimization_results) > 1:
            st.subheader("📈 Optimization Progress")
            

# ---------------------------
# Sidebar Information
# ---------------------------

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
2. 🎯 按步骤配置优化参数
3. ✅ 确认配置开始优化
4. 🧪 根据建议进行实验
5. 📊 输入实验结果
6. 🔄 重复直到找到最优解
""")
