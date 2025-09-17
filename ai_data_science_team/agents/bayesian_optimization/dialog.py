import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from langgraph.types import Command, interrupt

from .utils import validate_bounds


_QUESTION_TOKENS = [
    "?",
    "\uFF1F",
    "\u5417",
    "\u4E48",
    "\u4F55",
    "\u5982\u4F55",
    "\u600E\u4E48",
    "\u4E3A\u4F55",
    "\u4E3A\u5565",
    "\u54EA",
    "\u591A\u5C11",
]

_POSITIVE_CUES = [
    "\u662F",
    "\u597D",
    "\u786E\u8BA4",
    "\u53EF\u4EE5",
    "\u6CA1\u95EE\u9898",
    "\u884C",
    "ok",
    "okay",
    "\u5F00\u59CB",
    "\u51C6\u5907",
]

_NEGATIVE_CUES = [
    "\u4E0D",
    "\u5426",
    "\u518D",
    "\u5148\u7B49",
    "\u7A0D\u7B49",
    "\u7B49\u7B49",
    "\u6539",
    "\u8C03",
    "\u91CD",
    "\u6362",
]


@dataclass
class ConversationTurn:
    stage: str
    prompt: str
    user_input: str
    notes: Dict[str, Any] = field(default_factory=dict)


class ConsultativeConfigurator:
    def __init__(
        self,
        model: Any,
        data_info: Dict[str, Any],
        human_in_the_loop: bool = True,
        previous_config: Optional[Dict[str, Any]] = None,
        fields_to_update: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.data_info = data_info
        self.human_in_the_loop = human_in_the_loop
        columns = data_info.get("columns", []) or []
        self.available_columns = [str(col) for col in columns]
        self._available_column_map = {col.casefold(): col for col in self.available_columns}
        self._keep_existing_keywords = [
            "保留",
            "沿用",
            "保持",
            "不变",
            "一样",
            "维持",
            "沿袭",
            "和之前一样",
            "上次",
            "原来的",
            "same",
            "不用改",
        ]
        self.history: List[ConversationTurn] = []
        self.context_notes: Dict[str, Any] = {}
        self.current_config: Dict[str, Any] = {
            "input_variables": [],
            "output_variable": "",
            "optimization_goal": "maximize",
            "variable_bounds": {},
        }
        self.confidence: Dict[str, float] = {}
        self.confirmed: bool = False
        self._intro_done: bool = False
        self.fields_to_update: Set[str] = set(fields_to_update or [])

        if previous_config:
            self.current_config.update({
                "input_variables": previous_config.get("input_variables", []),
                "output_variable": previous_config.get("output_variable", ""),
                "optimization_goal": previous_config.get("optimization_goal", "maximize"),
                "variable_bounds": previous_config.get("variable_bounds", {}),
            })
            self.confidence.update(previous_config.get("confidence", {}))
            self.context_notes.update(previous_config.get("context_notes", {}))

        if not self.fields_to_update:
            self.fields_to_update = {"context", "inputs", "output", "goal", "bounds"}

    def run(self) -> Dict[str, Any]:
        self._intro_if_needed()
        while not self.confirmed:
            self._collect_information()
            self.confirmed = self._confirm_understanding()
        summary = self._build_summary()
        return {
            **self.current_config,
            "context_notes": self.context_notes,
            "confidence": self.confidence,
            "conversation_notes": [turn.__dict__ for turn in self.history],
            "conversation_summary": summary,
            "confirmed": self.confirmed,
        }

    def _collect_information(self) -> None:
        if "context" in self.fields_to_update:
            self._gather_context()
        if "inputs" in self.fields_to_update:
            self._gather_input_variables()
        if "output" in self.fields_to_update:
            self._gather_output_variable()
        if "goal" in self.fields_to_update:
            self._gather_optimization_goal()
        if "bounds" in self.fields_to_update:
            self._gather_bounds()
        self.fields_to_update.clear()

    def _intro_if_needed(self) -> None:
        if self._intro_done:
            return
        intro_prompt = (
            "\u60A8\u597D\uff0c\u6211\u662F\u8D1D\u53F6\u65AF\u4F18\u5316\u54A8\u8BE2\u987E\u95EE\u3002\n"
            "\u6211\u4F1A\u548C\u60A8\u4E00\u8D77\u68B3\u7406\u76EE\u6807\u3001\u6570\u636E\u548C\u7EA6\u675F\uFF0C\u7136\u540E\u518D\u542F\u52A8\u4F18\u5316\u6D41\u7A0B\u3002\n"
            "\u5982\u679C\u5728\u8FC7\u7A0B\u4E2D\u5BF9\u6211\u7684\u95EE\u9898\u6709\u7591\u95EE\uFF0C\u53EF\u4EE5\u968F\u65F6\u5148\u95EE\u6211\u3002\n\n"
            "\u4E3A\u4E86\u5F00\u59CB\uFF0C\u8BF7\u5148\u7B80\u8981\u4ECB\u7ECD\u5F53\u524D\u60F3\u8981\u4F18\u5316\u7684\u76EE\u6807\u3001\u5DF2\u6709\u7684\u63A2\u7D22\u4EE5\u53CA\u9700\u8981\u6CE8\u610F\u7684\u7EA6\u675F\u3002"
        )
        response = self._prompt_user(intro_prompt, stage="intro", allow_empty=True)
        cleaned = response.strip() if response else ""
        if cleaned:
            self.history.append(ConversationTurn("intro", intro_prompt, response))
            self.context_notes["initial_comment"] = cleaned
            if not self.context_notes.get("overview"):
                self.context_notes["overview"] = cleaned
                self.confidence["context"] = 0.8 if len(cleaned) < 15 else 0.9
        elif response is not None:
            self.history.append(ConversationTurn("intro", intro_prompt, response, {"status": "empty"}))
        self._intro_done = True

    def _gather_context(self) -> None:
        existing = self.context_notes.get("overview")
        if existing:
            summary = existing.strip()
            prompt = (
                f"我目前记录的背景是：{summary or '（尚未记录）'}。\n"
                "是否还有其他关键约束或补充信息？如果没有，可以回复‘没有’或直接按回车。"
            )
            response = self._prompt_user(prompt, stage="context", allow_empty=True)
            normalized = response.strip() if response else ""
            has_addition = bool(normalized) and not any(
                phrase in normalized for phrase in ["没有", "無", "无", "没了", "none", "nothing"]
            )
            if has_addition:
                updated = f"{summary}\n{normalized}" if summary else normalized
                self.context_notes["overview"] = updated
                self.confidence["context"] = 0.9
                self.history.append(
                    ConversationTurn("context_followup", prompt, response, {"updated": True})
                )
            else:
                self.history.append(
                    ConversationTurn("context_followup", prompt, response, {"updated": False})
                )
            return
        prompt = (
            "为了确保我理解业务背景，能和我分享一下当前的目标、已有探索以及关键约束吗？\n"
            "可以自由描述，我会帮您整理要点。"
        )
        response = self._prompt_user(prompt, stage="context")
        normalized = response.strip()
        self.context_notes["overview"] = normalized
        self.confidence["context"] = 0.8 if len(normalized) < 15 else 0.9
        self.history.append(ConversationTurn("context", prompt, response))

    def _gather_input_variables(self) -> None:
        existing = self.current_config.get("input_variables") or []
        available_display = ", ".join(self.available_columns) if self.available_columns else "（未提供列信息）"
        base_prompt = (
            "我们来确定需要优化的输入变量。\n"
            f"可选字段：{available_display}。\n"
            "可以直接告诉我要关注哪些变量，如果想保留上一次的选择，也请说明。"
        )
        if existing:
            formatted_existing = ", ".join(existing)
            base_prompt += f"\n（上次记录：{formatted_existing}）"
        while True:
            response = self._prompt_user(base_prompt, stage="inputs")
            normalized = response.strip() if response else ""
            normalized_cf = normalized.casefold() if normalized else ""
            keep_existing = bool(existing) and normalized_cf and any(
                keyword.casefold() in normalized_cf for keyword in self._keep_existing_keywords
            )
            raw_tokens = self._parse_variables(response)
            if keep_existing:
                resolved = list(existing)
                unmatched = []
                source = "keep_existing"
            else:
                resolved, unmatched = self._resolve_input_tokens(raw_tokens)
                source = "user_input"
            if resolved:
                self.current_config["input_variables"] = resolved
                self.confidence["input_variables"] = 0.9 if not unmatched else 0.85
                notes = {"parsed": resolved, "source": source}
                if raw_tokens:
                    notes["parsed_tokens"] = raw_tokens
                if unmatched:
                    notes["unmatched"] = unmatched
                self.history.append(ConversationTurn("inputs", base_prompt, response, notes))
                break
            goal_hint = self._parse_goal(response)
            retry_notes = {"status": "retry"}
            if raw_tokens:
                retry_notes["parsed_tokens"] = raw_tokens
            if unmatched:
                retry_notes["unmatched"] = unmatched
            if goal_hint:
                retry_notes["hint"] = "goal_detected"
            self.history.append(ConversationTurn("inputs_retry", base_prompt, response, retry_notes))
            if goal_hint:
                base_prompt = (
                    "听起来您在描述优化方向，我们稍后会确认是最大化还是最小化。\n"
                    f"现在请告诉我需要调节的输入变量名称，例如：{available_display}。"
                )
            elif unmatched and self.available_columns:
                base_prompt = (
                    f"我没有在已知字段中找到这些名称：{', '.join(unmatched)}。\n"
                    f"可选字段：{available_display}。请再试一次，或说明这些字段是否有别名。"
                )
            else:
                base_prompt = (
                    "我暂时没能提取出具体的输入变量。\n"
                    "请使用字段名（可以用逗号或空格分隔）再描述一次。"
                )

    def _resolve_input_tokens(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        if not tokens:
            return [], []
        if not self._available_column_map:
            resolved: List[str] = []
            for token in tokens:
                if token not in resolved:
                    resolved.append(token)
            return resolved, []
        resolved: List[str] = []
        unmatched: List[str] = []
        for token in tokens:
            key = token.casefold()
            column = self._available_column_map.get(key)
            if column:
                if column not in resolved:
                    resolved.append(column)
            else:
                unmatched.append(token)
        return resolved, unmatched

    def _gather_output_variable(self) -> None:
        existing = self.current_config.get("output_variable")
        base_prompt = (
            "\u63A5\u4E0B\u6765\u786E\u5B9A\u4F18\u5316\u7684\u8861\u91CF\u6307\u6807\uFF08\u8F93\u51FA\u53D8\u91CF\uFF09\u3002\n"
            "\u8BF7\u544A\u8BC9\u6211\u54EA\u4E2A\u5B57\u6BB5\u53EF\u4EE5\u4EE3\u8868\u5B9E\u9A8C\u7684\u6210\u6548\u6216\u635F\u5931\u3002"
        )
        if existing:
            base_prompt += f"\n\uFF08\u4E0A\u6B21\u8BB0\u5F55\uFF1A{existing}\uFF09"
        while True:
            response = self._prompt_user(base_prompt, stage="output")
            output_var = self._parse_single_variable(response)
            if output_var:
                if output_var in self.current_config.get("input_variables", []):
                    base_prompt = (
                        "\u8FD9\u4E2A\u5B57\u6BB5\u76EE\u524D\u5728\u8F93\u5165\u53D8\u91CF\u91CC\uFF0C\u8F93\u51FA\u53D8\u91CF\u9700\u8981\u4E0E\u8F93\u5165\u53D8\u91CF\u4E0D\u540C\u3002\n"
                        "\u8BF7\u91CD\u65B0\u6307\u5B9A\u8F93\u51FA\u53D8\u91CF\u3002"
                    )
                    continue
                self.current_config["output_variable"] = output_var
                self.confidence["output_variable"] = 0.85
                self.history.append(ConversationTurn("output", base_prompt, response, {"parsed": output_var}))
                break
            base_prompt = (
                "\u6211\u6682\u65F6\u6CA1\u6709\u8BC6\u522B\u51FA\u5177\u4F53\u7684\u8F93\u51FA\u53D8\u91CF\u3002\n"
                "\u8BF7\u63CF\u8FF0\u6216\u76F4\u63A5\u8F93\u5165\u6307\u6807\u540D\u79F0\u3002"
            )

    def _gather_optimization_goal(self) -> None:
        existing = self.current_config.get("optimization_goal")
        base_prompt = (
            "\u4F18\u5316\u65B9\u5411\u662F\u6700\u5927\u5316\u8FD8\u662F\u6700\u5C0F\u5316\uFF1F\n"
            "\u4E5F\u53EF\u4EE5\u7528\u5177\u4F53\u63CF\u8FF0\uFF0C\u4F8B\u5982\u201C\u5E0C\u671B\u6307\u6807\u8D8A\u5927\u8D8A\u597D\u201D\u6216\u201C\u60F3\u964D\u4F4E\u6210\u672C\u201D\u3002"
        )
        if existing:
            base_prompt += f"\n\uFF08\u4E0A\u6B21\u8BB0\u5F55\uFF1A{existing}\uFF09"
        while True:
            response = self._prompt_user(base_prompt, stage="goal")
            goal = self._parse_goal(response)
            if goal:
                self.current_config["optimization_goal"] = goal
                self.confidence["optimization_goal"] = 0.9
                self.history.append(ConversationTurn("goal", base_prompt, response, {"parsed": goal}))
                break
            base_prompt = (
                "\u6211\u7406\u89E3\u60A8\u53EF\u80FD\u5728\u63CF\u8FF0\uFF0C\u4F46\u8FD8\u6CA1\u80FD\u5224\u65AD\u662F\u6700\u5927\u5316\u8FD8\u662F\u6700\u5C0F\u5316\u3002\n"
                "\u53EF\u4EE5\u5C1D\u8BD5\u76F4\u63A5\u8BF4\u201C\u6700\u5927\u5316\u201D\u3001\u201C\u6700\u5C0F\u5316\u201D\uFF0C\u6216\u8005\u63CF\u8FF0\u65B9\u5411\u3002"
            )

    def _gather_bounds(self) -> None:
        input_vars = self.current_config.get("input_variables", [])
        bounds: Dict[str, Tuple[float, float]] = {}
        for var in input_vars:
            existing = self.current_config.get("variable_bounds", {}).get(var)
            base_prompt = (
                f"\u8BF7\u544A\u8BC9\u6211\u53D8\u91CF\u201C{var}\u201D\u7684\u53EF\u7528\u8303\u56F4\uFF0C\u4F8B\u5982 0 \u5230 1\u3002\n"
                "\u53EF\u4EE5\u5199\u6210\u201C0, 1\u201D\u6216\u63CF\u8FF0\u5F62\u5F0F\u3002"
            )
            if existing:
                base_prompt += f"\n\uFF08\u4E0A\u6B21\u8BB0\u5F55\uFF1A{existing[0]} \u5230 {existing[1]}\uFF09"
            while True:
                response = self._prompt_user(base_prompt, stage=f"bounds::{var}")
                parsed = self._parse_bounds(response)
                if parsed and validate_bounds(*parsed):
                    bounds[var] = parsed
                    self.history.append(
                        ConversationTurn("bounds", base_prompt, response, {"variable": var, "parsed": parsed})
                    )
                    break
                base_prompt = (
                    "\u6211\u6682\u65F6\u6CA1\u80FD\u8BC6\u522B\u6709\u6548\u7684\u8303\u56F4\u3002\n"
                    "\u8BF7\u786E\u4FDD\u7ED9\u51FA\u4E24\u4E2A\u6570\u5B57\uFF08\u6700\u5C0F\u503C\u3001\u6700\u5927\u503C\uFF09\uFF0C\u6216\u8005\u8BF4\u660E\u662F\u5426\u6709\u56FA\u5B9A\u533A\u95F4\u3002"
                )
        self.current_config["variable_bounds"] = bounds
        self.confidence["variable_bounds"] = 0.8

    def _confirm_understanding(self) -> bool:
        summary = self._build_summary()
        prompt = (
            f"\u6211\u6574\u7406\u7684\u8981\u70B9\u5982\u4E0B\uFF1A\n{summary}\n"
            "\u8FD9\u4E9B\u662F\u5426\u51C6\u786E\uFF1F\u5982\u679C\u6709\u4EFB\u4F55\u504F\u5DEE\uFF0C\u8BF7\u544A\u8BC9\u6211\u5177\u4F53\u8981\u8C03\u6574\u7684\u90E8\u5206\u3002"
        )
        while True:
            response = self._prompt_user(prompt, stage="confirmation")
            if self._is_positive(response):
                self.history.append(ConversationTurn("confirmation", prompt, response, {"status": "confirmed"}))
                return True
            if self._is_negative(response):
                fields = self._interpret_revision_request(response)
                if not fields:
                    fields = {"context", "inputs", "output", "goal", "bounds"}
                if "inputs" in fields:
                    fields.add("bounds")
                self.fields_to_update = fields
                self.history.append(
                    ConversationTurn(
                        "confirmation",
                        prompt,
                        response,
                        {"status": "revise", "fields": list(fields)},
                    )
                )
                return False
            inferred_fields = self._interpret_revision_request(response)
            if inferred_fields:
                if "inputs" in inferred_fields:
                    inferred_fields.add("bounds")
                self.fields_to_update = inferred_fields
                self.history.append(
                    ConversationTurn(
                        "confirmation",
                        prompt,
                        response,
                        {"status": "revise", "fields": list(inferred_fields)},
                    )
                )
                return False
            prompt = (
                "\u611F\u8C22\u53CD\u9988\u3002\u4E3A\u4E86\u6539\u8FDB\u8BBE\u7F6E\uFF0C\u9700\u8981\u77E5\u9053\u8981\u8C03\u6574\u54EA\u4E00\u90E8\u5206\uFF1A\u8F93\u5165\u53D8\u91CF\u3001\u8F93\u51FA\u53D8\u91CF\u3001\u4F18\u5316\u76EE\u6807\u6216\u8303\u56F4\uFF1F"
            )

    def _build_summary(self) -> str:
        inputs = ", ".join(self.current_config.get("input_variables", [])) or "\u672A\u6307\u5B9A"
        output_var = self.current_config.get("output_variable", "\u672A\u6307\u5B9A")
        goal = self.current_config.get("optimization_goal", "maximize")
        goal_cn = "\u6700\u5927\u5316" if goal == "maximize" else "\u6700\u5C0F\u5316"
        bounds_lines = []
        for var, bounds in self.current_config.get("variable_bounds", {}).items():
            bounds_lines.append(f"  - {var}: {bounds[0]} ~ {bounds[1]}")
        bounds_text = "\n".join(bounds_lines) if bounds_lines else "  - \u6682\u65E0\u8303\u56F4\u4FE1\u606F"
        context = self.context_notes.get("overview") or "\uFF08\u672A\u63D0\u4F9B\uFF09"
        return (
            f"\u80CC\u666F\uFF1A{context}\n"
            f"\u8F93\u5165\u53D8\u91CF\uFF1A{inputs}\n"
            f"\u8F93\u51FA\u53D8\u91CF\uFF1A{output_var}\n"
            f"\u4F18\u5316\u65B9\u5411\uFF1A{goal_cn}\n"
            f"\u53D8\u91CF\u8303\u56F4\uFF1A\n{bounds_text}"
        )

    def _prompt_user(self, prompt: str, stage: str, allow_empty: bool = False) -> str:
        current_prompt = prompt
        while True:
            raw = self._get_user_input(current_prompt)
            text = self._normalize_response(raw)
            if not text:
                if allow_empty:
                    return ""
                current_prompt = "\u62B1\u6B49\uFF0C\u6211\u6CA1\u6709\u6536\u5230\u5185\u5BB9\u3002\u53EF\u4EE5\u518D\u8BF4\u4E00\u6B21\u5417\uFF1F"
                continue
            if self._looks_like_question(text):
                current_prompt = self._compose_question_reply(stage, text)
                continue
            return text

    def _get_user_input(self, prompt: str) -> Any:
        if self.human_in_the_loop:
            return interrupt(value=prompt)
        print(prompt)
        return input("> ").strip()

    @staticmethod
    def _normalize_response(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, Command):
            resume_val = getattr(raw, "resume", None)
            if resume_val is not None:
                return str(resume_val).strip()
            update_payload = getattr(raw, "update", None) or {}
            if isinstance(update_payload, dict):
                for key in ("text", "message", "response"):
                    if update_payload.get(key):
                        return str(update_payload[key]).strip()
            goto_target = getattr(raw, "goto", None)
            if goto_target is not None:
                return str(goto_target).strip()
            return ""
        return str(raw).strip()

    def _compose_question_reply(self, stage: str, question: str) -> str:
        stage_explanations = {
            "intro": "\u6211\u4F1A\u5148\u4E86\u89E3\u80CC\u666F\uFF0C\u7136\u540E\u518D\u9010\u6B65\u786E\u8BA4\u53D8\u91CF\u548C\u76EE\u6807\u3002",
            "context": "\u4E86\u89E3\u80CC\u666F\u53EF\u4EE5\u5E2E\u6211\u5224\u65AD\u54EA\u4E9B\u7EA6\u675F\u91CD\u8981\uFF0C\u54EA\u4E9B\u4FE1\u606F\u8FD8\u7F3A\u5931\u3002",
            "inputs": "\u786E\u5B9A\u8F93\u5165\u53D8\u91CF\u662F\u4E3A\u4E86\u660E\u786E\u4F18\u5316\u7A7A\u95F4\uFF0C\u4E4B\u540E\u6211\u4F1A\u534F\u52A9\u8BBE\u5B9A\u8303\u56F4\u3002",
            "output": "\u8F93\u51FA\u53D8\u91CF\u662F\u8861\u91CF\u5B9E\u9A8C\u6210\u6548\u7684\u6307\u6807\uFF0C\u7528\u6765\u5224\u65AD\u597D\u574F\u3002",
            "goal": "\u4F18\u5316\u65B9\u5411\u544A\u8BC9\u6211\u9700\u8981\u63D0\u5347\u8FD8\u662F\u964D\u4F4E\u6307\u6807\u3002",
            "confirmation": "\u5728\u786E\u8BA4\u9636\u6BB5\uFF0C\u6211\u4EEC\u4E00\u8D77\u68C0\u67E5\u524D\u9762\u6574\u7406\u7684\u5185\u5BB9\u662F\u5426\u51C6\u786E\u3002",
        }
        base = stage_explanations.get(stage.split("::")[0], "\u6211\u4F1A\u5C3D\u91CF\u56DE\u7B54\u60A8\u7684\u7591\u95EE\u3002")
        return (
            f"\u5173\u4E8E\u60A8\u7684\u95EE\u9898\uFF1A{question}\n"
            f"{base}\n"
            "\u5982\u679C\u8FD8\u6709\u7591\u95EE\uFF0C\u53EF\u4EE5\u518D\u95EE\uFF1B\u5426\u5219\u8BF7\u7EE7\u7EED\u544A\u8BC9\u6211\u76F8\u5173\u4FE1\u606F\u3002"
        )

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in _QUESTION_TOKENS)

    @staticmethod
    def _parse_variables(text: str) -> List[str]:
        normalized = (
            text.replace("\u548C", " ")
            .replace("\u3001", " ")
            .replace(",", " ")
            .replace("\uFF0C", " ")
        )
        candidates = [item.strip() for item in normalized.split() if item.strip()]
        unique: List[str] = []
        for item in candidates:
            if item not in unique:
                unique.append(item)
        return unique

    @staticmethod
    def _parse_single_variable(text: str) -> str:
        tokens = ConsultativeConfigurator._parse_variables(text)
        return tokens[0] if tokens else ""

    @staticmethod
    def _parse_goal(text: str) -> Optional[str]:
        normalized = text.lower()
        if any(word in normalized for word in ["\u6700\u5927", "max", "increase", "\u63D0\u9AD8", "\u63D0\u5347", "\u589E\u52A0"]):
            return "maximize"
        if any(word in normalized for word in ["\u6700\u5C0F", "min", "decrease", "\u964D\u4F4E", "\u51CF\u5C11"]):
            return "minimize"
        return None

    @staticmethod
    def _parse_bounds(text: str) -> Optional[Tuple[float, float]]:
        cleaned = text.replace("，", ",")
        for token in ("～", "〜", "~", "到"):
            cleaned = cleaned.replace(token, " ")
        matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if len(matches) < 2:
            return None
        lower, upper = float(matches[0]), float(matches[1])
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper
    @staticmethod
    def _is_positive(text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in _POSITIVE_CUES)

    @staticmethod
    def _is_negative(text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in _NEGATIVE_CUES)

    @staticmethod
    def _interpret_revision_request(text: str) -> Set[str]:
        lowered = text.lower()
        fields: Set[str] = set()
        if "\u8F93\u5165" in text or "\u7279\u5F81" in text or ("\u53D8\u91CF" in text and "\u8F93\u5165" in text):
            fields.add("inputs")
        if "\u8F93\u51FA" in text or "\u76EE\u6807\u5217" in text or ("\u6307\u6807" in text and "\u8F93\u51FA" in text):
            fields.add("output")
        if "\u4F18\u5316" in text or "\u65B9\u5411" in text or "\u6700\u5927" in text or "\u6700\u5C0F" in text:
            fields.add("goal")
        if "\u8303\u56F4" in text or "\u533A\u95F4" in text or "\u754C\u9650" in text:
            fields.add("bounds")
        if "\u80CC\u666F" in text or "\u573A\u666F" in text or "\u9650\u5236" in text:
            fields.add("context")
        if "\u5168\u90E8" in text or "\u91CD\u65B0" in text or "\u91CD\u6765" in text or "\u90FD" in text:
            fields.update({"context", "inputs", "output", "goal", "bounds"})
        return fields


def get_user_confirmation_via_chat(
    model: Any,
    data_info: Dict[str, Any],
    previous_config: Optional[Dict[str, Any]] = None,
    human_in_the_loop: bool = True,
    fields_to_update: Optional[List[str]] = None,
) -> Dict[str, Any]:
    configurator = ConsultativeConfigurator(
        model=model,
        data_info=data_info,
        human_in_the_loop=human_in_the_loop,
        previous_config=previous_config,
        fields_to_update=fields_to_update,
    )
    return configurator.run()


def get_user_confirmation_via_chat_v2(
    model: Any,
    data_info: Dict[str, Any],
    previous_config: Optional[Dict[str, Any]] = None,
    human_in_the_loop: bool = True,
    fields_to_update: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return get_user_confirmation_via_chat(
        model=model,
        data_info=data_info,
        previous_config=previous_config,
        human_in_the_loop=human_in_the_loop,
        fields_to_update=fields_to_update,
    )


