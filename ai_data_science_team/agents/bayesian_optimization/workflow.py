import operator
from typing import Any, Optional, Annotated, Sequence, List, Dict, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer, interrupt

from .constants import AGENT_NAME, AVAILABLE_COLUMNS
from .optimizer import BayesianOptimizer
from .dialog import get_user_confirmation_via_chat, ConsultativeConfigurator
from .utils import serialize_optimizer_state


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instructions: Optional[str]
    input_variables: List[str]
    output_variable: str
    optimization_goal: str
    variable_bounds: Dict[str, tuple]
    input_data: Optional[Dict[str, Any]]
    optimizer_state: Optional[Dict[str, Any]]
    current_suggestion: Optional[List[float]]
    optimization_results: List[Dict[str, Any]]
    confirmed_settings: bool
    step: str
    need_human_approval: bool
    max_iterations: int
    conversation_summary: str
    conversation_notes: List[Dict[str, Any]]
    context_notes: Dict[str, Any]
    info_confidence: Dict[str, float]
    pending_revisions: List[str]
    previous_config: Dict[str, Any]


def create_bayesian_optimization_agent(
    model: Any,
    n_initial_points: int = 5,
    human_in_the_loop: bool = True,
    checkpointer: Checkpointer = None,
):
    def build_summary_from_state(state: AgentState) -> str:
        summary = state.get("conversation_summary")
        if summary:
            return summary
        inputs = ", ".join(state.get("input_variables", [])) or "\u672A\u6307\u5B9A"
        output_var = state.get("output_variable", "\u672A\u6307\u5B9A")
        goal = state.get("optimization_goal", "maximize")
        goal_cn = "\u6700\u5927\u5316" if goal == "maximize" else "\u6700\u5C0F\u5316"
        bounds_lines: List[str] = []
        for var, bounds in (state.get("variable_bounds") or {}).items():
            bounds_lines.append(f"  - {var}: {bounds[0]} ~ {bounds[1]}")
        bounds_text = "\n".join(bounds_lines) if bounds_lines else "  - \u6682\u65E0\u8303\u56F4\u4FE1\u606F"
        context = (state.get("context_notes") or {}).get("overview") or "\uFF08\u672A\u63D0\u4F9B\uFF09"
        return (
            f"\u80CC\u666F\uFF1A{context}\n"
            f"\u8F93\u5165\u53D8\u91CF\uFF1A{inputs}\n"
            f"\u8F93\u51FA\u53D8\u91CF\uFF1A{output_var}\n"
            f"\u4F18\u5316\u65B9\u5411\uFF1A{goal_cn}\n"
            f"\u53D8\u91CF\u8303\u56F4\uFF1A\n{bounds_text}"
        )

    def prompt_alignment_decision(prompt_text: str) -> str:
        current_prompt = prompt_text
        while True:
            if human_in_the_loop:
                raw = interrupt(value=current_prompt)
            else:
                print(current_prompt)
                raw = input("> ").strip()
            text = ConsultativeConfigurator._normalize_response(raw)
            if not text:
                current_prompt = "\u6211\u6682\u65F6\u6CA1\u6536\u5230\u56DE\u590D\uFF0C\u8BF7\u544A\u8BC9\u6211\u662F\u5426\u51C6\u5907\u5F00\u59CB\u6216\u9700\u8981\u8C03\u6574\u3002"
                continue
            if ConsultativeConfigurator._looks_like_question(text):
                current_prompt = (
                    f"\u5173\u4E8E\u60A8\u7684\u7591\u95EE\uFF1A{text}\n"
                    "\u8FD9\u4E2A\u9636\u6BB5\u4E3B\u8981\u786E\u8BA4\u662F\u5426\u8FDB\u5165\u5B9E\u9A8C\uFF0C\u5982\u679C\u9700\u8981\u66F4\u65B0\u914D\u7F6E\uFF0C\u8BF7\u76F4\u63A5\u8BF4\u660E\u54EA\u4E9B\u5185\u5BB9\u9700\u8981\u8C03\u6574\u3002"
                )
                continue
            return text

    def describe_available_data(state: AgentState) -> Dict[str, Any]:
        data_info: Dict[str, Any] = {}
        if state.get("input_data") and "X" in state["input_data"]:
            X_data = state["input_data"]["X"]
            if hasattr(X_data, "shape"):
                data_info["n_features"] = X_data.shape[1] if len(X_data.shape) > 1 else 1
                data_info["n_samples"] = X_data.shape[0]
            if "columns" in state["input_data"]:
                data_info["columns"] = state["input_data"]["columns"]
            else:
                feature_cols = [f"\u7279\u5F81{i+1}" for i in range(data_info.get("n_features", 2))]
                target_col = "\u76EE\u6807"
                data_info["columns"] = feature_cols + [target_col]
                data_info["target_col"] = target_col
        else:
            data_info = {"columns": AVAILABLE_COLUMNS, "n_features": len(AVAILABLE_COLUMNS) - 1}
        return data_info

    def setup_node(state: AgentState):
        print("\n=== \u8D1D\u53F6\u65AF\u4F18\u5316\u914D\u7F6E\u9636\u6BB5 ===")
        data_info = describe_available_data(state)
        previous_config = state.get("previous_config")
        revisions = state.get("pending_revisions") or None
        config = get_user_confirmation_via_chat(
            model=model,
            data_info=data_info,
            previous_config=previous_config,
            human_in_the_loop=human_in_the_loop,
            fields_to_update=revisions,
        )
        state["input_variables"] = config["input_variables"]
        state["output_variable"] = config["output_variable"]
        state["optimization_goal"] = config["optimization_goal"]
        state["variable_bounds"] = config["variable_bounds"]
        state["conversation_summary"] = config.get("conversation_summary", "")
        state["conversation_notes"] = config.get("conversation_notes", [])
        state["context_notes"] = config.get("context_notes", {})
        state["info_confidence"] = config.get("confidence", {})
        state["confirmed_settings"] = config.get("confirmed", False)
        state["pending_revisions"] = []
        state["previous_config"] = {
            "input_variables": state["input_variables"],
            "output_variable": state["output_variable"],
            "optimization_goal": state["optimization_goal"],
            "variable_bounds": state["variable_bounds"],
            "confidence": state["info_confidence"],
            "context_notes": state["context_notes"],
        }
        state["step"] = "alignment"
        state["need_human_approval"] = human_in_the_loop
        print("\n\u76EE\u6807\u548C\u53D8\u91CF\u5DF2\u7ECF\u6574\u7406\u597D\uFF0C\u7EE7\u7EED\u4E4B\u524D\u6211\u4EEC\u518D\u540C\u6B65\u4E00\u6B21\u671F\u671B\u3002")
        return state

    def approval_node(state: AgentState):
        summary_text = build_summary_from_state(state)
        prompt_text = (
            "\n=== \u914D\u7F6E\u56DE\u987E\u4E0E\u540C\u6B65 ===\n"
            f"{summary_text}\n\n"
            "\u6211\u4EEC\u662F\u5426\u53EF\u4EE5\u5F00\u59CB\u65B0\u4E00\u8F6E\u5B9E\u9A8C\uFF1F\u5982\u679C\u60F3\u8C03\u6574\uFF0C\u8BF7\u76F4\u63A5\u544A\u8BC9\u6211\u9700\u8981\u4FEE\u6539\u7684\u5185\u5BB9\u6216\u65B0\u7684\u7591\u95EE\u3002"
        )
        while True:
            response = prompt_alignment_decision(prompt_text)
            if ConsultativeConfigurator._is_positive(response):
                state["confirmed_settings"] = True
                state["need_human_approval"] = False
                state["pending_revisions"] = []
                state["step"] = "optimize"
                print("\n\u660E\u767D\u4E86\uff0c\u6211\u4F1A\u6839\u636E\u8BE5\u914D\u7F6E\u5F00\u59CB\u4F18\u5316\u8FC7\u7A0B\u3002")
                return state
            fields = ConsultativeConfigurator._interpret_revision_request(response)
            if not fields and not ConsultativeConfigurator._is_negative(response):
                prompt_text = (
                    "\u4E3A\u4E86\u5E2E\u60A8\u66F4\u65B0\u914D\u7F6E\uFF0C\u8BF7\u6307\u660E\u9700\u8981\u8FDB\u4E00\u6B65\u67E5\u8BE2\u7684\u9879\uFF0C\u6216\u544A\u8BC9\u6211\u5DF2\u7ECF\u53EF\u4EE5\u5F00\u59CB\u3002"
                )
                continue
            if not fields:
                fields = {"context", "inputs", "output", "goal", "bounds"}
            if "inputs" in fields:
                fields.add("bounds")
            state["pending_revisions"] = sorted(fields)
            state["confirmed_settings"] = False
            state["need_human_approval"] = human_in_the_loop
            state["step"] = "setup"
            print("\n\u597D\u7684\uff0c\u6211\u4F1A\u5E26\u4F60\u56DE\u5230\u9700\u6C42\u63D0\u70BC\u9636\u6BB5\u91CD\u65B0\u8C03\u6574\u3002")
            return state

    def validate_state(state: AgentState):
        missing_bounds = [v for v in state.get("input_variables", []) if v not in state.get("variable_bounds", {})]
        if missing_bounds:
            print(f"\n\u8B66\u544A: \u4EE5\u4E0B\u53D8\u91CF\u6682\u65F6\u7F3A\u5C11\u8303\u56F4\uFF1A{missing_bounds}")
            print("\u5C06\u4F7F\u7528\u9ED8\u8BA4\u8303\u56F4 [0, 1]")
            for v in missing_bounds:
                state.setdefault("variable_bounds", {})[v] = (0.0, 1.0)
        state.setdefault("optimization_results", [])
        return state

    def prompt_numeric(question: str) -> float:
        current_prompt = question
        while True:
            if human_in_the_loop:
                raw = interrupt(value=current_prompt)
            else:
                print(current_prompt)
                raw = input("> ")
            text = ConsultativeConfigurator._normalize_response(raw)
            if not text:
                current_prompt = "\u8BF7\u8F93\u5165\u6570\u503C\u7ED3\u679C\uFF08\u4F8B\u5982 1.23\uFF09"
                continue
            if ConsultativeConfigurator._looks_like_question(text):
                current_prompt = (
                    f"\u5173\u4E8E\u7ED3\u679C\u586B\u62A5\u7684\u7591\u95EE\uFF1A{text}\n"
                    "\u60F3\u83B7\u5F97\u6B64\u6B65\u6570\u503C\uFF0C\u4EE5\u4FBF\u6211\u66F4\u65B0\u6A21\u578B\u3002\u8BF7\u63D0\u4F9B\u5B9E\u9A8C\u7684\u6570\u503C\u3002"
                )
                continue
            try:
                return float(text)
            except ValueError:
                current_prompt = "\u65E0\u6CD5\u89E3\u6790\u8F93\u5165\uFF0C\u8BF7\u4F7F\u7528\u6570\u503C\u683C\u5F0F\uFF08\u4F8B\u5982 1.23\uFF09\u3002"

    def prompt_yes_no(question: str) -> bool:
        current_prompt = question
        while True:
            response = prompt_alignment_decision(current_prompt)
            if ConsultativeConfigurator._is_positive(response):
                return True
            if ConsultativeConfigurator._is_negative(response):
                return False
            interpreted = ConsultativeConfigurator._interpret_revision_request(response)
            if interpreted:
                return False
            current_prompt = "\u6211\u9700\u8981\u77E5\u9053\u662F\u5426\u7EE7\u7EED\u5B8C\u6210\u4E0B\u4E00\u6B21\u8C03\u7528\u3002\u53EF\u4EE5\u56DE\u590D\u7EE7\u7EED\u6216\u6682\u505C\u3002"

    def optimization_node(state: AgentState):
        print("\n=== \u8D1D\u53F6\u65AF\u4F18\u5316\u6D41\u7A0B ===")
        state = validate_state(state)
        input_vars = state.get("input_variables", [])
        bounds_list = [state["variable_bounds"][v] for v in input_vars]
        if state.get("optimizer_state") is None:
            optimizer = BayesianOptimizer(bounds_list)
            if state.get("input_data") and "X" in state["input_data"] and "Y" in state["input_data"]:
                X_data = state["input_data"]["X"]
                Y_data = state["input_data"]["Y"]
                if hasattr(X_data, "tolist"):
                    X_list = X_data.tolist()
                else:
                    X_list = X_data
                if hasattr(Y_data, "tolist"):
                    Y_list = Y_data.tolist()
                else:
                    Y_list = Y_data
                if X_list and Y_list:
                    optimizer.initialize(X_list, Y_list)
                    print(f"\n\u57FA\u4E8E\u5386\u53F2\u6570\u636E\u521D\u59CB\u5316 {len(Y_list)} \u4E2A\u6D4B\u70B9\u3002")
        else:
            opt_state = state["optimizer_state"] or {}
            optimizer = BayesianOptimizer(bounds_list, n_initial_points=int(opt_state.get("n_initial_points", n_initial_points)))
            optimizer.X = opt_state.get("X", [])
            optimizer.y = opt_state.get("y", [])
        suggestion = optimizer.suggest_next_point()
        state["current_suggestion"] = suggestion
        formatted = {v: suggestion[i] for i, v in enumerate(input_vars)}
        print("\n\u6211\u7684\u63A8\u8350\u5C0F\u6B65\uFF1A")
        for var, value in formatted.items():
            b = state["variable_bounds"][var]
            print(f"  - {var}: {value:.4f} (\u8303\u56F4 {b[0]} ~ {b[1]})")
        result = prompt_numeric(f"\n\u8BF7\u544A\u8BC9\u6211\u8F93\u51FA\u53D8\u91CF '{state['output_variable']}' \u7684\u5B9E\u9A8C\u6570\u503C\uFF1A")
        optimizer.update(suggestion, result)
        state.setdefault("optimization_results", []).append({"parameters": formatted, "result": result})
        if optimizer.y:
            if state["optimization_goal"] == "maximize":
                best_result = max(optimizer.y)
                best_idx = optimizer.y.index(best_result)
            else:
                best_result = min(optimizer.y)
                best_idx = optimizer.y.index(best_result)
            best_params = optimizer.X[best_idx] if len(optimizer.X) > best_idx else suggestion
            best_map = {v: best_params[i] for i, v in enumerate(input_vars)}
            print(f"\n\u5DF2\u66F4\u65B0\u7EDF\u8BA1\uFF0C\u5F53\u524D\u6700\u4F73\u7ED3\u679C {best_result:.4f}\uFF0C\u76F8\u5BF9\u53C2\u6570 {best_map}.")
        else:
            print("\n\u5DF2\u8BB0\u5F55\u7B2C\u4E00\u4E2A\u5B9E\u9A8C\u7ED3\u679C\u3002")
        state["optimizer_state"] = serialize_optimizer_state(optimizer)
        continue_prompt = (
            "\n\u662F\u5426\u7EE7\u7EED\u4E0B\u4E00\u8F6E\u8C03\u6574\uFF1F\u53EF\u56DE\u590D\u7EE7\u7EED\u6216\u6682\u505C\uFF0C\u4E5F\u53EF\u4EE5\u63D0\u51FA\u95EE\u9898\u3002"
        )
        if prompt_yes_no(continue_prompt):
            state["step"] = "optimize"
        else:
            state["step"] = "complete"
        return state

    def results_node(state: AgentState):
        print("\n" + "=" * 40)
        print("\u5DF2\u5B8C\u6210\u8D1D\u53F6\u65AF\u4F18\u5316\u8FDB\u7A0B")
        print("=" * 40)
        results = state.get("optimization_results", [])
        if results:
            print(f"\n\u5171\u62A5\u544A {len(results)} \u6B21\u5B9E\u9A8C\u3002")
            goal = state.get("optimization_goal", "maximize")
            output_var = state.get("output_variable", "\u76EE\u6807")
            if goal == "maximize":
                best_entry = max(results, key=lambda x: x["result"])
            else:
                best_entry = min(results, key=lambda x: x["result"])
            print(f"\n\u6700\u4F73 {output_var}: {best_entry['result']:.6f}")
            print("\u5BF9\u5E94\u53C2\u6570\uFF1A")
            for param, value in best_entry["parameters"].items():
                bounds = state.get("variable_bounds", {}).get(param)
                if bounds:
                    print(f"  - {param}: {value:.6f} (\u8303\u56F4 {bounds[0]} ~ {bounds[1]})")
                else:
                    print(f"  - {param}: {value:.6f}")
        else:
            print("\n\u6682\u65F6\u6CA1\u6709\u6536\u96C6\u5230\u4F18\u5316\u7ED3\u679C\u3002")
        print("\n" + "=" * 40)
        state["step"] = "end"
        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("setup", setup_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("optimize", optimization_node)
    workflow.add_node("results", results_node)

    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "approval")

    def after_approval(state: AgentState):
        if state.get("step") == "setup":
            return "setup"
        return "optimize" if state.get("confirmed_settings") else "approval"

    workflow.add_conditional_edges("approval", after_approval)

    def after_optimization(state: AgentState):
        return "results" if state.get("step") == "complete" else "optimize"

    workflow.add_conditional_edges("optimize", after_optimization)
    workflow.add_edge("results", END)

    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
