"""Role tool controls cover Companion actions and host/MCP native names."""

import unittest

from companion_v2.conversation.tool_controls import (
    disabled_tool_set,
    filter_host_tools,
    filter_specs,
    is_tool_disabled,
    native_action_name,
    require_tool_enabled,
)
from companion_v2.execution.models import TaskError
from companion_v2.model_gateway.models import ToolSpec


class ToolControlTests(unittest.TestCase):
    def test_json_list_and_native_alias_both_disable_companion_tools(self):
        disabled = disabled_tool_set(
            '["memory.global.query","companion_v2_relay_send"]'
        )

        self.assertTrue(is_tool_disabled(
            disabled, action="memory.global.query"
        ))
        self.assertTrue(is_tool_disabled(
            disabled, action="relay_send"
        ))
        self.assertFalse(is_tool_disabled(
            disabled, action="group.memory.query"
        ))
        self.assertEqual(
            native_action_name("memory.global.query"),
            "companion_v2_memory_global_query",
        )

    def test_filters_remove_companion_and_native_mcp_tools(self):
        specs = (
            ToolSpec("memory.global.query", "memory", "{}"),
            ToolSpec("group.interest.manage", "interest", "{}"),
        )
        disabled = disabled_tool_set(
            '["companion_v2_memory_global_query","mcp_docs_search"]'
        )
        host = (
            type("HostTool", (), {"name": "mcp_docs_search"})(),
            type("HostTool", (), {"name": "web_search"})(),
        )

        self.assertEqual(
            [item.action for item in filter_specs(
                specs, disabled, action_getter=lambda item: item.action
            )],
            ["group.interest.manage"],
        )
        self.assertEqual(
            [item.name for item in filter_host_tools(host, disabled)],
            ["web_search"],
        )

    def test_execution_guard_reports_the_stable_task_error(self):
        disabled = disabled_tool_set('["memory.global.query"]')

        with self.assertRaises(TaskError) as blocked:
            require_tool_enabled(disabled, action="memory.global.query")
        self.assertEqual(blocked.exception.code, "tool_disabled")

    def test_damaged_runtime_snapshot_fails_closed_only_for_bad_entries(self):
        disabled = disabled_tool_set(
            '["web_search", 7, "", "mcp_docs_search"]', strict=False
        )

        self.assertEqual(disabled, frozenset(("web_search", "mcp_docs_search")))
        self.assertEqual(
            disabled_tool_set("{broken", strict=False),
            frozenset(),
        )


if __name__ == "__main__":
    unittest.main()
