"""Role-owned global memory must cross windows without crossing private audiences."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import uuid4
import unittest

from companion_v2.conversation.recalls import MessageRecall
from companion_v2.execution.models import TaskContext, digest, encode, stamp
from companion_v2.foundation.refs import (
    AudienceRef,
    ScopeKind,
    ScopeRef,
    SourceRef,
    SubjectKind,
    SubjectRef,
    VersionRef,
)
from companion_v2.identity_policy.models import (
    AuthenticatedHostPrincipal,
    ResolvedIdentity,
    RoleBinding,
)
from companion_v2.memory.role_global import (
    RoleGlobalMemory,
    RoleGlobalMemoryTools,
    _project,
    role_global_owner,
)


NOW = datetime(2026, 11, 25, 4, 0, tzinfo=timezone.utc)


def make_context(*, kind=ScopeKind.PRIVATE, native_id="friend-a",
                 actor_id="stranger-a", role_id="bot-role",
                 installation="install", bot="bot", account="account"):
    scope = ScopeRef(
        installation, "qq", kind, native_id, bot, account,
        "conversation-" + native_id, "session-" + native_id)
    principal = AuthenticatedHostPrincipal(
        installation, "host", "qq", actor_id, bot, account)
    subject = SubjectRef(installation, SubjectKind.HUMAN, "subject-" + actor_id)
    actor = ResolvedIdentity(principal, subject, "binding-" + actor_id, 1, 1)
    role = SubjectRef(installation, SubjectKind.BOT, role_id)
    return TaskContext(
        actor, RoleBinding(scope, role, 1),
        AudienceRef(scope, (subject,)))


class _Runtime:
    def __init__(self, root):
        self.data_root = root
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE runtime_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute(
            "CREATE TABLE inbox (consumer TEXT NOT NULL, event_id TEXT NOT NULL, "
            "PRIMARY KEY(consumer, event_id))")

    def read_one(self, sql, values=()):
        return self.connection.execute(sql, values).fetchone()

    def read_all(self, sql, values=()):
        return tuple(self.connection.execute(sql, values).fetchall())


class _Messages:
    def __init__(self, root, *, enabled=True):
        self.runtime = _Runtime(root)
        self.tasks = NS(now=lambda: NOW)
        self.tools = NS(semantic=None)
        self.reply_settings = NS(snapshot=lambda *_args, **_kwargs: NS(values={
            "conversation.presentation.role_global_memory_enabled": enabled,
        }))


class _CandidateMemory(RoleGlobalMemory):
    def __init__(self, messages, candidates):
        super().__init__(messages)
        self.candidates = candidates

    def _candidates(self, context, *, since, until, scope_filter):
        if scope_filter == "group":
            return [candidate for candidate in self.candidates
                    if candidate["row"]["scope_kind"] == "group"]
        if scope_filter == "private":
            return [candidate for candidate in self.candidates
                    if candidate["row"]["scope_kind"] == "private"]
        if scope_filter == "current":
            return [candidate for candidate in self.candidates
                    if (candidate["row"]["original"].scope.kind
                        is context.scope.kind
                        and candidate["row"]["original"].scope.native_scope_id
                        == context.scope.native_scope_id
                        and (context.scope.kind is not ScopeKind.PRIVATE
                             or candidate["row"]["original"].actor.subject
                             == context.actor.subject))]
        return list(self.candidates)

    def _validate(self, row, context):
        return row["candidate"]


def candidate(context, text, *, kind="input", observed_at=NOW, event_id="event"):
    reference = SourceRef(
        "source-" + event_id, "conversation.native_input", context.scope,
        VersionRef("source-" + event_id, 1), observed_at)
    return {
        "row": {"scope_kind": context.scope.kind.value, "original": context},
        "candidate": {
            "id": "observation-" + event_id,
            "original": context,
            "reference": reference,
            "kind": kind,
            "event_id": event_id,
            "effective_at": stamp(observed_at),
            "speaker_id": (
                context.actor.principal.native_account_id
                if kind == "input" else context.role_binding.role.subject_id
            ),
            "payload": text,
        },
    }


class RoleGlobalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".tmp-role-global-" + uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_owner_is_shared_by_strangers_private_and_every_group(self):
        first = make_context(native_id="stranger-1", actor_id="stranger-1")
        second = make_context(native_id="stranger-2", actor_id="stranger-2")
        group = make_context(
            kind=ScopeKind.GROUP, native_id="group-a", actor_id="member-a")
        self.assertEqual(role_global_owner(first), role_global_owner(second))
        self.assertEqual(role_global_owner(first), role_global_owner(group))

        other_role = make_context(role_id="other-role")
        other_bot = make_context(bot="other-bot", account="other-account")
        self.assertNotEqual(role_global_owner(first), role_global_owner(other_role))
        self.assertNotEqual(role_global_owner(first), role_global_owner(other_bot))

    def test_stranger_private_exchange_is_remembered_but_not_disclosed_elsewhere(self):
        current = make_context(native_id="my-private", actor_id="me")
        stranger = make_context(native_id="stranger-private", actor_id="stranger")
        messages = _Messages(self.root)
        memory = _CandidateMemory(messages, [
            candidate(stranger, "我对海边的旧灯塔特别感兴趣"),
        ])

        result = asyncio.run(memory.query(
            current, query="今天有什么好玩的事情", limit=8))

        self.assertEqual(result["state"], "ok")
        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["scope"]["kind"], "private")
        self.assertEqual(item["projection"]["visibility"], "other_private_withheld")
        self.assertEqual(item["text"], "")
        self.assertNotIn("旧灯塔", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["coverage"]["withheld_private"], 1)

    def test_same_private_window_and_group_public_text_are_returned(self):
        current = make_context(native_id="friend-a", actor_id="friend-a")
        same = candidate(current, "我上周去了海边", event_id="same-private")
        group_context = make_context(
            kind=ScopeKind.GROUP, native_id="group-a", actor_id="member-a")
        group = candidate(group_context, "今晚一起看流星雨", event_id="group-public")
        memory = _CandidateMemory(_Messages(self.root), [same, group])

        result = asyncio.run(memory.query(
            current, query="海边 流星雨", scope_filter="any", limit=8))

        by_visibility = {
            item["projection"]["visibility"]: item for item in result["items"]}
        self.assertEqual(by_visibility["current_private"]["text"], "我上周去了海边")
        self.assertEqual(by_visibility["group_public"]["text"], "今晚一起看流星雨")
        self.assertTrue(all(item["evidence"]["source_id"] for item in result["items"]))

    def test_current_private_text_survives_reset_and_role_revision(self):
        original = make_context(native_id="friend-a", actor_id="friend-a")
        scope = replace(
            original.scope,
            conversation_id="after-reset-conversation",
            session_id="after-reset-session",
        )
        current = TaskContext(
            original.actor,
            RoleBinding(scope, original.role_binding.role, 2),
            AudienceRef(scope, (original.actor.subject,)),
        )
        memory = _CandidateMemory(
            _Messages(self.root),
            [candidate(original, "旧灯塔旁边有一家深夜书店", event_id="before-reset")],
        )

        result = asyncio.run(memory.query(
            current, query="深夜书店", scope_filter="current", limit=4))

        self.assertEqual(result["state"], "ok")
        self.assertEqual(result["items"][0]["projection"]["visibility"], "current_private")
        self.assertEqual(result["items"][0]["text"], "旧灯塔旁边有一家深夜书店")

    def test_role_own_private_words_are_recalled(self):
        current = make_context(native_id="my-private", actor_id="me")
        other = make_context(native_id="other-private", actor_id="other")
        own = candidate(other, "我答应过下次认真听他讲完", kind="assistant",
                        event_id="own-reply")
        memory = _CandidateMemory(_Messages(self.root), [own])

        result = asyncio.run(memory.query(current, query="答应 听讲", limit=4))

        item = result["items"][0]
        self.assertEqual(item["projection"]["visibility"], "role_own_private")
        self.assertIn("认真听", item["text"])

    def test_generic_query_falls_back_to_recent_messages(self):
        current = make_context(native_id="my-private", actor_id="me")
        older = candidate(current, "一条不太相关的老消息",
                          observed_at=NOW - timedelta(hours=4), event_id="older")
        recent = candidate(current, "刚刚遇到一只会开门的猫",
                           observed_at=NOW, event_id="recent")
        memory = _CandidateMemory(_Messages(self.root), [older, recent])

        result = asyncio.run(memory.query(
            current, query="今天有什么好玩的事情", limit=1))

        self.assertEqual(result["items"][0]["observation_id"], "observation-recent")

    def test_embedding_failure_keeps_lexical_recall(self):
        current = make_context(native_id="my-private", actor_id="me")
        item = candidate(current, "周末计划去看一场电影", event_id="movie")
        messages = _Messages(self.root)

        class BrokenSemantic:
            store = NS(settings=lambda: {"embedding_provider": "embed"})

            class Gateway:
                async def embed(self, *args, **kwargs):
                    raise RuntimeError("embedding offline")

            gateway = Gateway()

        messages.tools = NS(semantic=BrokenSemantic())
        memory = _CandidateMemory(messages, [item])

        result = asyncio.run(memory.query(current, query="电影", limit=4))

        self.assertEqual(result["mode"], "lexical")
        self.assertEqual(result["items"][0]["text"], "周末计划去看一场电影")

    def test_disabled_role_memory_has_no_catalog_and_returns_disabled(self):
        current = make_context()
        messages = _Messages(self.root, enabled=False)
        memory = _CandidateMemory(messages, [candidate(current, "秘密")])
        result = asyncio.run(memory.query(current, query="秘密"))
        self.assertEqual(result["state"], "disabled")
        self.assertEqual(result["items"], [])


class RoleGlobalMemoryToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".tmp-role-global-tool-" + uuid4().hex)
        self.root.mkdir()
        self.current = make_context()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_disabled_memory_exposes_no_tool(self):
        memory = _CandidateMemory(
            _Messages(self.root, enabled=False),
            [candidate(self.current, "hello")],
        )

        self.assertEqual(RoleGlobalMemoryTools(memory).catalog_for(self.current), ())

    def test_enabled_memory_exposes_one_read_only_tool(self):
        memory = _CandidateMemory(
            _Messages(self.root),
            [candidate(self.current, "hello")],
        )
        tools = RoleGlobalMemoryTools(memory)

        self.assertEqual(
            [tool.action for tool in tools.catalog_for(self.current)],
            ["memory.global.query"],
        )
        self.assertEqual(tools.actions, frozenset(("memory.global.query",)))


class RoleGlobalSourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".tmp-role-global-source-" + uuid4().hex)
        self.root.mkdir()
        self.messages = _Messages(self.root)
        self.memory = RoleGlobalMemory(self.messages)
        self.context = make_context(native_id="stranger", actor_id="stranger")
        self.payload = "陌生人第一次来聊天"
        self.reference = SourceRef(
            "source-1", "conversation.native_input", self.context.scope,
            VersionRef("source-1", 1), NOW)
        self.row = {
            "observation_id": "obs-1",
            "context_key": self.context.key,
            "event_id": "event-1",
            "kind": "input",
            "delivery_state": "input_observed",
            "context_json": encode(self.context),
            "source_json": encode(self.reference),
            "content_hash": digest(self.payload),
            "observed_at": stamp(NOW),
            "effective_at": stamp(NOW),
            "scope_kind": "private",
            "payload_json": json.dumps(self.payload, ensure_ascii=False),
            "source_state": "active",
            "source_expires_at": None,
            "source_hash": digest(self.payload),
            "source_id": "source-1",
            "lineage_id": "source-1",
            "source_revision": 1,
            "issued_at": stamp(NOW),
            "owner": "conversation.native_input",
            "installation_id": self.context.scope.installation_id,
            "platform": self.context.scope.platform,
            "native_scope_id": self.context.scope.native_scope_id,
            "bot_id": self.context.scope.bot_id,
            "account_id": self.context.scope.account_id,
            "conversation_id": self.context.scope.conversation_id,
            "session_id": self.context.scope.session_id,
        }

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_source_fields_are_validated_without_cross_user_owner_check(self):
        result = self.memory._validate(self.row, make_context(
            native_id="another", actor_id="another"))
        self.assertIsNotNone(result)
        self.assertEqual(result["payload"], self.payload)

    def test_revoked_changed_withdrawn_or_wrong_role_sources_are_invisible(self):
        checks = []
        for changes in (
            {"source_state": "revoked"},
            {"content_hash": digest("changed")},
            {"source_hash": digest("changed")},
            {"bot_id": "other-bot"},
        ):
            row = {**self.row, **changes}
            checks.append(self.memory._validate(row, self.context))
        self.assertTrue(all(item is None for item in checks))

        marker = MessageRecall.from_scope(
            self.context.scope, "event-1",
            self.context.actor.principal.native_account_id)
        with patch.object(self.messages.runtime, "read_one") as read:
            def recalled(sql, values=()):
                if "FROM inbox" in sql:
                    return {"found": 1}
                return None
            read.side_effect = recalled
            self.assertIsNone(self.memory._validate(self.row, self.context))


if __name__ == "__main__":
    unittest.main()
