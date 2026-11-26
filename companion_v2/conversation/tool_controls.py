"""Role-shared deny-list controls for every tool surface in a reply turn.

The configuration stores canonical action names where possible.  Host tools
are matched by their native function name as well, so the same control can
hide Companion tools, Relay tools, AstrBot extensions and MCP tools.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import re

from companion_v2.configuration.models import ConfigurationError
from companion_v2.execution.models import TaskError, digest


FIELD = "mcp_disabled_tools"
DEFAULT = "[]"
MAX_ITEMS = 256
MAX_NAME_CHARS = 192


def native_action_name(action: str) -> str:
    """Return the AstrBot native function name without importing the host."""
    name = "companion_v2_" + re.sub(r"[^a-zA-Z0-9_]", "_", action)
    return name if len(name) <= 64 else name[:47] + "_" + digest(action)[:16]


def disabled_tool_set(raw, *, strict=True):
    """Parse and canonicalise a JSON/list deny-list.

    ``strict`` is used by configuration validation. Runtime callers may pass
    ``strict=False`` for a damaged older snapshot; in that case only valid
    entries are retained so one malformed item cannot prevent startup.
    """
    if raw is None or raw == "":
        return frozenset()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return frozenset()
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError) as error:
            if not strict:
                return frozenset()
            raise ConfigurationError(
                "reply_configuration_invalid",
                "disabled tools must be a JSON array",
            ) from error
        raw = parsed
    if isinstance(raw, (tuple, list, set, frozenset)):
        values = list(raw)
    elif not strict:
        return frozenset()
    else:
        raise ConfigurationError(
            "reply_configuration_invalid",
            "disabled tools must be a JSON array",
        )
    if len(values) > MAX_ITEMS:
        if not strict:
            values = values[:MAX_ITEMS]
        else:
            raise ConfigurationError(
                "reply_configuration_invalid",
                f"at most {MAX_ITEMS} tools may be disabled",
            )
    result = []
    for value in values:
        if not isinstance(value, str):
            if not strict:
                continue
            raise ConfigurationError(
                "reply_configuration_invalid",
                "every disabled tool must be a string",
            )
        name = value.strip()
        if not name:
            continue
        if (len(name) > MAX_NAME_CHARS
                or any(ord(char) < 32 or ord(char) == 127 for char in name)):
            if not strict:
                continue
            raise ConfigurationError(
                "reply_configuration_invalid",
                "disabled tool names must be short printable strings",
            )
        if name not in result:
            result.append(name)
    return frozenset(result)


def is_tool_disabled(disabled, *, action=None, native_name=None):
    blocked = disabled if isinstance(disabled, (set, frozenset, tuple, list)) else ()
    if action and action in blocked:
        return True
    if native_name and native_name in blocked:
        return True
    if action and native_action_name(action) in blocked:
        return True
    return False


def require_tool_enabled(disabled, *, action=None, native_name=None):
    if is_tool_disabled(disabled, action=action, native_name=native_name):
        raise TaskError(
            "tool_disabled",
            "this tool is disabled by the role tool settings",
        )
    return True


def filter_specs(tools: Iterable, disabled, *, action_getter):
    return tuple(
        tool for tool in tools
        if not is_tool_disabled(
            disabled,
            action=action_getter(tool),
        )
    )


def filter_host_tools(tools: Iterable, disabled):
    return [
        tool for tool in tools
        if not is_tool_disabled(
            disabled,
            native_name=getattr(tool, "name", ""),
        )
    ]
