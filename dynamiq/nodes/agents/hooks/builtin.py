"""Built-in hooks, registered under short names for YAML and UI use.

transform         JSONPath rewrite of tool input (merged into the model's arguments) and of tool output.
tool_policy       deny a tool, or allow it only for matching users.
call_limit        at most N calls per run.
regex             block or mask regex matches, with presets.
pii               detect, mask and restore PII with numbered placeholders.
prompt_injection  prompt-injection check of the user input and of tool results through a detector connection.
"""

import importlib
import json
import re
from collections.abc import Iterator
from typing import Any, Literal

import regex
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from dynamiq.connections import HuggingFace, Lakera, Replicate
from dynamiq.nodes.agents.hooks.core import (
    ALLOW,
    Block,
    BlockAs,
    Decision,
    Hook,
    HookContext,
    HookPoint,
    Modify,
    ToolCall,
    ToolResult,
    register_hook,
    sanitize_tool_name,
)
from dynamiq.nodes.detectors import LlamaGuardDetector, PromptInjectionDetector
from dynamiq.nodes.node import InputTransformer, Node, OutputTransformer
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.jsonpath import _is_rooted, is_jsonpath
from dynamiq.utils.jsonpath import mapper as jsonpath_mapper
from dynamiq.utils.logger import logger

REGEX_TIMEOUT_SECONDS = 1.0

RunPoint = Literal["input", "tool_result", "output"]
_RUN_POINTS = {"input": HookPoint.ON_INPUT, "tool_result": HookPoint.AFTER_TOOL, "output": HookPoint.ON_OUTPUT}


def iter_strings(value: Any) -> Iterator[str]:
    """Every string leaf of a payload: dicts, lists and tuples are walked, and so is a ``ToolResult``."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, ToolResult):
        yield from iter_strings([value.content, value.output, value.error])
    elif isinstance(value, dict):
        yield from iter_strings(list(value.values()))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


def map_strings(value: Any, fn) -> Any:
    """Copy of a payload with ``fn`` applied to every string leaf."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, ToolResult):
        return ToolResult(map_strings(value.content, fn), map_strings(value.output, fn), map_strings(value.error, fn))
    if isinstance(value, dict):
        return {key: map_strings(item, fn) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_strings(item, fn) for item in value)
    return value


def compile_regex(pattern: str, label: str = "regex") -> "regex.Pattern":
    try:
        return regex.compile(pattern)
    except regex.error as e:
        raise ValueError(f"invalid {label} {pattern!r}: {e}") from e


@register_hook("transform")
class TransformHook(Hook):
    """JSONPath rewrite of tool input and output, the declarative way to select fields.

    ``input_transformer.selector`` is merged into the model's arguments, like the agent's per-tool
    ``input_transformer``: listed keys are set and every other argument is kept. A path that matches nothing is
    skipped with a warning, never written as ``None``. A selector value that is not a JSONPath is a constant.

    ``output_transformer`` is applied to ``{"content": <result>}``; a JSON text result is parsed first so that paths
    can select from it.
    """

    input_transformer: InputTransformer = Field(default_factory=InputTransformer)
    output_transformer: OutputTransformer = Field(default_factory=OutputTransformer)

    @model_validator(mode="after")
    def _validate(self):
        if self.input_transformer.path:
            raise ValueError("transform.input_transformer.path is not supported; use `selector`")
        for label, transformer in (("input", self.input_transformer), ("output", self.output_transformer)):
            if transformer.path and not is_jsonpath(transformer.path):
                raise ValueError(f"transform.{label}_transformer.path {transformer.path!r} is not a valid JSONPath")
            for key, expression in (transformer.selector or {}).items():
                if _is_rooted(expression) and not is_jsonpath(expression):
                    raise ValueError(
                        f"transform.{label}_transformer.selector[{key!r}] {expression!r} is not a valid JSONPath"
                    )
        if not (self.input_transformer.selector or self.output_transformer.path or self.output_transformer.selector):
            raise ValueError("transform needs an input_transformer.selector or an output_transformer")
        return self

    def points(self) -> set[HookPoint]:
        points = set()
        if self.input_transformer.selector:
            points.add(HookPoint.BEFORE_TOOL)
        if self.output_transformer.path or self.output_transformer.selector:
            points.add(HookPoint.AFTER_TOOL)
        return points

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        if not isinstance(call.input, dict):
            return ALLOW
        merged = dict(call.input)
        for key, expression in self.input_transformer.selector.items():
            if not _is_rooted(expression):
                merged[key] = expression
            elif (value := jsonpath_mapper(call.input, {key: expression}).get(key)) is not None:
                merged[key] = value
            else:
                logger.warning(f"transform hook for '{call.name}': {expression!r} matched nothing; {key!r} skipped")
        return Modify(merged) if merged != call.input else ALLOW

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        if result.error is not None:
            return ALLOW
        data, parsed = result.content, False
        if isinstance(data, str):
            try:
                loaded = json.loads(data)
            except ValueError:
                loaded = None
            if isinstance(loaded, (dict, list)):
                data, parsed = loaded, True
        transformed = Node.transform({"content": data}, self.output_transformer)
        content = transformed["content"] if isinstance(transformed, dict) and "content" in transformed else transformed
        if content is None:
            logger.warning(f"transform hook for '{call.name}': output_transformer matched nothing; content is None")
        elif parsed and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return Modify(ToolResult(content=content, output=result.output))


class GuardHook(Hook):
    """A hook that can refuse. ``on_violation`` is ``observation`` (the model is told; at tool points only, elsewhere
    it ends the run with the message), ``answer`` or ``fail``."""

    message: str = "Blocked by policy."
    on_violation: BlockAs = BlockAs.OBSERVATION

    def violation(self) -> Block:
        return Block(self.message, self.on_violation)


@register_hook("tool_policy")
class ToolPolicyHook(GuardHook):
    """Deny a tool, or allow it only when the run context matches.

    Attributes:
        deny: Always block the matching tools.
        allow_if: ``{context path: expected value or list of values}``; every entry must match. Paths: ``user_id``,
            ``session_id``, ``metadata.<key>``.

    Human approval (``approval: true``) is not implemented yet.
    """

    message: str = "This tool is not available."
    deny: bool = False
    allow_if: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _approval_not_supported(cls, data):
        if isinstance(data, dict) and "approval" in data:
            raise ValueError("tool_policy `approval` is not supported yet; use `deny` or `allow_if`")
        return data

    @model_validator(mode="after")
    def _needs_a_rule(self):
        if not self.deny and not self.allow_if:
            raise ValueError("tool_policy needs `deny: true` or `allow_if`; otherwise it would allow everything")
        return self

    def points(self) -> set[HookPoint]:
        return {HookPoint.BEFORE_TOOL}

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        if self.deny:
            return self.violation()
        for path, expected in (self.allow_if or {}).items():
            if ctx.lookup(path) not in (expected if isinstance(expected, list) else [expected]):
                return self.violation()
        return ALLOW


@register_hook("call_limit")
class CallLimitHook(GuardHook):
    """At most ``max_per_run`` calls of the matching tools per run, sub-agent calls included.

    The counter lives in the run context, so it is exact with parallel tool calls and separate for every Map item.
    A blocked call does not count.
    """

    message: str = "Call limit reached for this tool in this run. Do not call it again."
    max_per_run: int = Field(ge=1)

    def points(self) -> set[HookPoint]:
        return {HookPoint.BEFORE_TOOL}

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        with ctx.lock:
            state = ctx.hook_state
            if state.get("calls", 0) >= self.max_per_run:
                return self.violation()
            state["calls"] = state.get("calls", 0) + 1
        return ALLOW


class TextGuardHook(GuardHook):
    """Shared by the hooks that scan text: the user input, tool results (failures included) and the final answer.

    Subclasses define ``_violates`` (``action: block``) and ``_mask`` (``action: mask``).

    Attributes:
        on: ``input`` (the user's message, once per run), ``tool_result`` (each tool result), ``output`` (the answer).
    """

    on: list[RunPoint] = Field(default_factory=lambda: ["input", "tool_result", "output"], min_length=1)
    action: Literal["block", "mask"] = "block"

    def points(self) -> set[HookPoint]:
        return {_RUN_POINTS[name] for name in self.on}

    def _violates(self, text: str) -> bool:
        raise NotImplementedError

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        raise NotImplementedError

    def _screen(self, ctx: HookContext, value: Any, origin: str) -> Decision:
        if self.action == "block":
            return self.violation() if any(self._violates(text) for text in iter_strings(value)) else ALLOW
        masked = map_strings(value, lambda text: self._mask(ctx, text, origin))
        return Modify(masked) if masked != value else ALLOW

    def on_input(self, ctx: HookContext, text: str) -> Decision:
        return self._screen(ctx, text, "input")

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        return self._screen(ctx, result, "tool_result")

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        return self._screen(ctx, answer, "output")


REGEX_PRESETS: dict[str, list[str]] = {
    "injection_basic": [
        r"(?i)\bignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)\b",
        r"(?i)\bdisregard\s+(all\s+)?(the\s+)?(system|previous|prior)\s+(prompt|instructions)\b",
        r"(?i)\b(reveal|print|show)\s+(me\s+)?(your\s+)?(system\s+prompt|hidden\s+instructions)\b",
        r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s+mode\b",
    ],
    "secrets": [
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bsk-[A-Za-z0-9_\-]{20,}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*\S{8,}",
    ],
}


@register_hook("regex")
class RegexHook(TextGuardHook):
    """Block or mask text that matches a pattern.

    Matching runs on the ``regex`` engine with a timeout (``REGEX_TIMEOUT_SECONDS`` per scan): a catastrophic pattern
    raises instead of stalling the process, and the hook then fails per ``on_error``.

    Attributes:
        patterns: Regexes. ``presets``: named sets (``injection_basic``, ``secrets``). One of the two is required.
        replacement: What ``action: mask`` puts in place of a match.
    """

    message: str = "The request was blocked by a content policy."
    patterns: list[str] = Field(default_factory=list)
    presets: list[Literal["injection_basic", "secrets"]] = Field(default_factory=list)
    replacement: str = "[REDACTED]"

    _compiled: list["regex.Pattern"] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _compile(self):
        patterns = [*self.patterns, *(pattern for name in self.presets for pattern in REGEX_PRESETS[name])]
        if not patterns:
            raise ValueError("regex needs at least one pattern or preset; otherwise it would match nothing")
        self._compiled = [compile_regex(pattern) for pattern in patterns]
        return self

    def _violates(self, text: str) -> bool:
        return any(pattern.search(text, timeout=REGEX_TIMEOUT_SECONDS) for pattern in self._compiled)

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        for pattern in self._compiled:
            text = pattern.sub(self.replacement, text, timeout=REGEX_TIMEOUT_SECONDS)
        return text


PIIEntity = Literal["email", "phone", "credit_card", "ssn", "ip_address"]

# Most specific first, so that a card number is not also taken for a phone number.
_PII_PATTERNS: dict[str, str] = {
    "credit_card": r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])",
    "ssn": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}",
    "ip_address": r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
    "phone": r"(?<![\w.])\+?\d[\d\s().\-]{7,}\d(?![\w])",
}
_PLACEHOLDER = regex.compile(r"<[A-Z_]+_\d+>")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        number = int(char) * (2 if position % 2 else 1)
        total += number - 9 if number > 9 else number
    return total % 10 == 0


def _is_valid_match(entity: str | None, text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if entity == "credit_card":
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if entity == "phone":
        return 9 <= len(digits) <= 15 and not re.match(r"\d{4}-\d{2}-\d{2}", text.strip())
    return True


@register_hook("pii")
class PIIHook(TextGuardHook):
    """Detect PII and mask it with numbered placeholders (``<EMAIL_1>``) so that the agent keeps working.

    Masking happens at the source: the user's message is masked once and each tool result is masked once, so the
    conversation history, memory, checkpoints and the summarizer only ever hold placeholders and nothing is
    re-scanned on later LLM calls. One mapping per run lives in the run context and is checkpointed with the agent.

    Attributes:
        entities: What to detect. ``custom_patterns``: ``{NAME: regex}`` for more entity types.
        restore_in_tools: Tools that receive the real values: placeholders in their arguments are replaced right
            before the call. Other tools get placeholders. ``*`` means all tools.
        restore_in_output: Put the real values back in the final answer, only those the user supplied (never values
            that came from a tool result).

    Limits: the mapping lives in the run, so a later run that reloads history from memory sees placeholders without
    the mapping, and numbering restarts. Tools that call LLMs inside, and their traces, see what the tool received.
    Detection is regex-based: names and addresses are not detected.
    """

    message: str = "The request contains personal data and was blocked."
    action: Literal["block", "mask"] = "mask"
    entities: list[PIIEntity] = Field(default_factory=lambda: list(_PII_PATTERNS))
    custom_patterns: dict[str, str] = Field(default_factory=dict)
    restore_in_tools: list[str] = Field(default_factory=list)
    restore_in_output: bool = False

    _detectors: list[tuple[str, str | None, "regex.Pattern"]] = PrivateAttr(default_factory=list)

    @field_validator("custom_patterns")
    @classmethod
    def _check_custom_names(cls, value):
        for name in value:
            if not re.fullmatch(r"[A-Z_]+", name):
                raise ValueError(f"custom_patterns name {name!r} must be UPPER_SNAKE_CASE letters")
        return value

    @model_validator(mode="after")
    def _compile(self):
        if not self.entities and not self.custom_patterns:
            raise ValueError("pii needs at least one entity or custom pattern")
        built_in = [(name.upper(), name, _PII_PATTERNS[name]) for name in _PII_PATTERNS if name in self.entities]
        custom = [(name, None, pattern) for name, pattern in self.custom_patterns.items()]
        self._detectors = [
            (label, entity, compile_regex(pattern, f"pattern for {label}"))
            for label, entity, pattern in built_in + custom
        ]
        return self

    def points(self) -> set[HookPoint]:
        points = super().points()
        if self.restore_in_tools:
            points.add(HookPoint.BEFORE_TOOL)
        if self.restore_in_output:
            points.add(HookPoint.ON_OUTPUT)
        return points

    def _placeholder(self, ctx: HookContext, label: str, value: str, origin: str) -> str:
        with ctx.lock:
            store = ctx.hook_state
            placeholders = store.setdefault("value_to_placeholder", {})
            key = f"{label}\x00{value}"
            if (placeholder := placeholders.get(key)) is None:
                counters = store.setdefault("counters", {})
                counters[label] = counters.get(label, 0) + 1
                placeholder = placeholders[key] = f"<{label}_{counters[label]}>"
                store.setdefault("placeholder_to_value", {})[placeholder] = value
            origins = store.setdefault("origin", {})
            if origin == "input" or placeholder not in origins:
                origins[placeholder] = origin
        return placeholder

    def _violates(self, text: str) -> bool:
        return any(
            _is_valid_match(entity, match.group(0))
            for _, entity, pattern in self._detectors
            for match in pattern.finditer(text, timeout=REGEX_TIMEOUT_SECONDS)
        )

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        for label, entity, pattern in self._detectors:

            def replace(match, label=label, entity=entity):
                found = match.group(0)
                return self._placeholder(ctx, label, found, origin) if _is_valid_match(entity, found) else found

            text = pattern.sub(replace, text, timeout=REGEX_TIMEOUT_SECONDS)
        return text

    def _restore(self, ctx: HookContext, text: str, only_origin: str | None = None) -> str:
        with ctx.lock:
            store = ctx.hook_state
            values, origins = dict(store.get("placeholder_to_value", {})), dict(store.get("origin", {}))

        def put_back(match):
            placeholder = match.group(0)
            if placeholder not in values or (only_origin and origins.get(placeholder) != only_origin):
                return placeholder
            return values[placeholder]

        return _PLACEHOLDER.sub(put_back, text)

    def _restores_for(self, tool_name: str) -> bool:
        return "*" in self.restore_in_tools or sanitize_tool_name(tool_name) in {
            sanitize_tool_name(name) for name in self.restore_in_tools
        }

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        if not self._restores_for(call.name):
            return ALLOW
        restored = map_strings(call.input, lambda text: self._restore(ctx, text))
        return Modify(restored) if restored != call.input else ALLOW

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        value = answer
        if "output" in self.on:
            decision = self._screen(ctx, answer, "output")
            if isinstance(decision, Block):
                return decision
            if isinstance(decision, Modify):
                value = decision.value
        if self.restore_in_output:
            value = map_strings(value, lambda text: self._restore(ctx, text, only_origin="input"))
        return Modify(value) if value != answer else ALLOW


class DetectorConfig(BaseModel):
    """The detector node to use and the workflow connection it talks through (no API key in the hook)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: Literal["prompt_injection", "llama_guard"] = "prompt_injection"
    connection: HuggingFace | Lakera | Replicate
    model: str | None = None
    timeout: float | None = None

    @field_validator("connection", mode="before")
    @classmethod
    def _build_connection(cls, value):
        """A dumped connection (a dict with its dotted ``type``) is rebuilt as that connection class."""
        if not (isinstance(value, dict) and value.get("type")):
            return value
        module_name, _, class_name = str(value["type"]).rpartition(".")
        try:
            connection_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"connection type {value['type']!r} could not be imported: {e}") from e
        return connection_class(**{key: item for key, item in value.items() if key != "type"})

    @model_validator(mode="after")
    def _connection_matches_kind(self):
        if (self.kind == "llama_guard") != isinstance(self.connection, Replicate):
            raise ValueError(
                "detector kind `llama_guard` needs a Replicate connection, "
                "`prompt_injection` a HuggingFace or Lakera one"
            )
        return self


@register_hook("prompt_injection")
class PromptInjectionHook(GuardHook):
    """Check the user's input and tool results (failed calls included) with a hosted detector.

    ``detector.connection`` references a connection of the workflow (``connection: lakera-conn``). A detector that
    errors fails per ``on_error`` (closed by default). Each message and each tool result is checked once.
    """

    message: str = "Potential prompt injection detected; the content was withheld."
    detector: DetectorConfig
    on: list[Literal["input", "tool_result"]] = Field(default_factory=lambda: ["input", "tool_result"], min_length=1)

    _node: LlamaGuardDetector | PromptInjectionDetector | None = PrivateAttr(default=None)

    def points(self) -> set[HookPoint]:
        return {_RUN_POINTS[name] for name in self.on}

    def _detector_node(self) -> LlamaGuardDetector | PromptInjectionDetector:
        if self._node is None:
            node_class = LlamaGuardDetector if self.detector.kind == "llama_guard" else PromptInjectionDetector
            options = self.detector.model_dump(include={"model", "timeout"}, exclude_none=True)
            self._node = node_class(connection=self.detector.connection, **options)
        return self._node

    def detect(self, ctx: HookContext, text: str) -> bool:
        """Whether the detector flags ``text``. Override to plug another detector."""
        config = ctx.config or RunnableConfig(callbacks=[])
        result = self._detector_node().run_sync(input_data={"message": text}, config=config)
        if result.status != RunnableStatus.SUCCESS:
            raise RuntimeError(f"detector failed: {result.error.message if result.error else result.status}")
        output = result.output or {}
        return bool(output["prompt_detected"]) if "prompt_detected" in output else output.get("is_safe") is False

    def on_input(self, ctx: HookContext, text: str) -> Decision:
        return self.violation() if text and self.detect(ctx, text) else ALLOW

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        return self.violation() if any(text and self.detect(ctx, text) for text in iter_strings(result)) else ALLOW
