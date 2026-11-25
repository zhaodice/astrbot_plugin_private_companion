"""Autonomous group interests are role-owned, revisioned, and soft-retracted."""

from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
from types import SimpleNamespace as NS
import unittest

from companion_v2.conversation.group_interests import (
    GroupAutonomousInterestTools,
    GroupAutonomousInterests,
)
from companion_v2.execution.models import TaskError
from companion_v2.foundation.refs import ScopeKind, ScopeRef


class _Runtime:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE runtime_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def read_one(self, sql, values=()):
        return self.connection.execute(sql, values).fetchone()

    @contextmanager
    def transaction(self):
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise


class _Messages:
    def __init__(self):
        self.runtime = _Runtime()
        self.tasks = NS(now=lambda: datetime(
            2026, 11, 25, 6, tzinfo=timezone.utc
        ))
        self.group_context = NS(authorize=lambda *_args: None)


def _context(group_id="group-a"):
    scope = ScopeRef(
        "installation",
        "qq",
        ScopeKind.GROUP,
        group_id,
        "bot",
        "bot-account",
        "conversation-" + group_id,
        "session-" + group_id,
    )
    return NS(
        scope=scope,
        key="turn-" + group_id,
        role_binding=NS(role=NS(subject_id="role")),
    )


class GroupAutonomousInterestTests(unittest.TestCase):
    def setUp(self):
        self.messages = _Messages()
        self.interests = GroupAutonomousInterests(self.messages)
        self.context = _context()

    def test_role_can_create_update_match_and_soft_retract_an_interest(self):
        created = self.interests.execute(self.context, {
            "operation": "create",
            "topic": "夜空与流星",
            "keywords": ["流星", "星图"],
            "weight": 1.2,
            "basis": {
                "context": "群友连续讨论了观星",
                "persona": "角色本身喜欢夜空",
                "memory": "此前也见过相关话题",
                "willingness": "今晚有精力参与",
            },
        })["payload"]

        self.assertEqual(created["active_count"], 1)
        item = created["items"][0]
        self.assertEqual((item["state"], item["revision"]), ("active", 1))
        matched = self.interests.match(
            self.context, "周末有人去看流星吗？"
        )
        self.assertEqual(matched["basis"], "role_owned_interest")
        self.assertEqual(matched["items"][0]["interest_id"], item["interest_id"])

        updated = self.interests.execute(self.context, {
            "operation": "update",
            "interest_id": item["interest_id"],
            "expected_revision": 1,
            "topic": "观星",
            "keywords": ["观星", "望远镜"],
            "basis": {"memory": "兴趣已从流星扩展到观星"},
        })["payload"]
        revised = updated["items"][0]
        self.assertEqual((revised["state"], revised["revision"]), ("active", 2))
        self.assertIsNone(self.interests.match(self.context, "有人在看流星吗？"))
        self.assertIsNotNone(self.interests.match(self.context, "今晚去观星？"))

        with self.assertRaises(TaskError) as stale:
            self.interests.execute(self.context, {
                "operation": "update",
                "interest_id": item["interest_id"],
                "expected_revision": 1,
                "keywords": ["旧关键词"],
            })
        self.assertEqual(stale.exception.code, "group_interest_revision")

        deactivated = self.interests.execute(self.context, {
            "operation": "deactivate",
            "interest_id": item["interest_id"],
            "expected_revision": 2,
            "reason": "角色现在不想继续关注这个话题",
        })["payload"]
        self.assertEqual(deactivated["active_count"], 0)
        self.assertEqual(deactivated["items"][0]["state"], "inactive")
        self.assertEqual(deactivated["items"][0]["revision"], 3)
        self.assertIsNone(self.interests.match(self.context, "今晚去观星？"))

    def test_interest_owner_is_the_role_not_one_group(self):
        self.interests.execute(self.context, {
            "operation": "create",
            "topic": "桌游",
            "keywords": ["桌游"],
        })

        other_group = _context("group-b")
        snapshot = self.interests.snapshot(other_group)

        self.assertEqual(snapshot["owner"], "role")
        self.assertEqual(snapshot["active_count"], 1)
        self.assertEqual(snapshot["items"][0]["topic"], "桌游")

    def test_disabled_switch_removes_tool_and_blocks_execution(self):
        tools = GroupAutonomousInterestTools(self.interests)
        disabled = {"group_autonomous_interests_enabled": False}

        self.assertEqual(tools.catalog_for(self.context, disabled), ())
        with self.assertRaises(TaskError) as blocked:
            import asyncio
            asyncio.run(tools.execute(self.context, {
                "operation": "create",
                "topic": "天文",
                "keywords": ["天文"],
            }, disabled))
        self.assertEqual(blocked.exception.code, "group_interest_disabled")


if __name__ == "__main__":
    unittest.main()
