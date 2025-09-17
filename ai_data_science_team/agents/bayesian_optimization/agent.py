from typing import Any, Dict

from langgraph.types import Checkpointer, Command
from ai_data_science_team.templates import BaseAgent
from .workflow import create_bayesian_optimization_agent


def make_bayesian_optimization_agent(
    model: Any,
    n_initial_points: int = 5,
    human_in_the_loop: bool = True,
    checkpointer: Checkpointer = None,
):
    # 默认提供内存检查点，确保可使用 get_state()
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return create_bayesian_optimization_agent(
        model, n_initial_points, human_in_the_loop, checkpointer
    )


class BayesianOptimizationAgent(BaseAgent):
    """
    贝叶斯优化智能体
    """

    def __init__(
        self,
        model: Any,
        n_initial_points: int = 5,
        human_in_the_loop: bool = True,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "n_initial_points": n_initial_points,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_bayesian_optimization_agent(**self._params)

    def _handle_interrupts(
        self,
        result,
        config=None,
        **kwargs,
    ):
        if not isinstance(result, dict):
            return result
        if not self._params.get("human_in_the_loop", True):
            return result
        while result.get("__interrupt__"):
            interrupts = result.get("__interrupt__") or []
            prompt_text = None
            for interrupt_event in reversed(interrupts):
                prompt_text = getattr(interrupt_event, "value", None)
                if prompt_text:
                    break
            if prompt_text is None:
                prompt_text = "请输入后续指示："
            print("\n" + str(prompt_text))
            try:
                user_reply = input("> ").strip()
            except KeyboardInterrupt:
                print("\n用户中断了对话流程。")
                break
            while user_reply == "":
                user_reply = input("> ").strip()
            result = self._compiled_graph.invoke(Command(resume=user_reply), config=config, **kwargs)
            self.response = result
            if not isinstance(result, dict):
                break
        return result
    def invoke_agent(
        self,
        input_data: Dict[str, Any],
        user_instructions: str = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        initial_state = {
            "user_instructions": user_instructions,
            "input_data": input_data,
            "max_iterations": max_iterations,
            "optimization_results": [],
        }
        result = self._compiled_graph.invoke(initial_state, config=config, **kwargs)
        result = self._handle_interrupts(result, config=config, **kwargs)
        self.response = result
        return None

    async def ainvoke_agent(
        self,
        input_data: Dict[str, Any],
        user_instructions: str = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        self.response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "input_data": input_data,
                "max_iterations": max_iterations,
                "optimization_results": [],
            },
            config=config,
            **kwargs,
        )
        return None

    def get_optimization_results(self):
        if self.response:
            return self.response.get("optimization_results", [])
        return []

    def get_best_result(self):
        results = self.get_optimization_results()
        if not results:
            return None

        optimization_goal = (
            self.response.get("optimization_goal", "maximize") if self.response else "maximize"
        )
        if optimization_goal == "maximize":
            return max(results, key=lambda x: x["result"])
        else:
            return min(results, key=lambda x: x["result"])

    def get_current_suggestion(self):
        if self.response:
            return self.response.get("current_suggestion")
        return None

    def get_optimization_goal(self):
        if self.response:
            return self.response.get("optimization_goal")
        return None

    def get_variable_bounds(self):
        if self.response:
            return self.response.get("variable_bounds")
        return {}

