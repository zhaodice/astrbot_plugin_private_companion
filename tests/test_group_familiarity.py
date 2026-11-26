"""Focused contracts for Laya's opt-in group familiarity candidates."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from astrbot_host.core_reply_decision import (
    _group_memory_summary,
    _group_participation_summary,
)
from companion_v2.conversation.group_admission import GroupReplyAdmission
from companion_v2.conversation.group_familiarity import GroupFamiliarity
from companion_v2.conversation.reply_settings import GROUP, GROUP_WAKEUP_DEFAULTS, group_wakeup_settings
from companion_v2.decision.prompt import build_state, decision_questions, render_model_state
from companion_v2.foundation.refs import ScopeKind, ScopeRef


class _Runtime:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE runtime_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

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
    def __init__(self, settings, *, laya_enabled=True, action_enabled=True, now=None):
        self.runtime = _Runtime()
        self.values = {GROUP + "." + key: value for key, value in settings.items()}
        self.laya = {"enabled": laya_enabled, "decision_fields": {"action": action_enabled}}
        moment = now or datetime(2026, 11, 20, 5, tzinfo=timezone.utc)
        self.tasks = NS(now=lambda: moment)
        self.reply_settings = NS(snapshot=lambda *_args, **_kwargs: NS(values=self.values))
        self.laya_decision_resolver = lambda _context: NS(store=NS(raw=lambda: dict(self.laya)))


def _context(account_id="100"):
    scope = ScopeRef("installation", "qq", ScopeKind.GROUP, "group", "bot", "bot-account",
                     "conversation", "session")
    return NS(
        scope=scope,
        key="turn-" + account_id,
        actor=NS(principal=NS(native_account_id=account_id)),
        role_binding=NS(role=NS(subject_id="role"), revision=1),
    )


def _values(**changes):
    return {**GROUP_WAKEUP_DEFAULTS, **changes}


class GroupFamiliarityTests(unittest.TestCase):
    def setUp(self):
        self.settings = _values(
            group_wakeup_direct_words="[]",
            group_wakeup_context_words="[]",
            group_wakeup_interest_keywords="[]",
            group_familiarity_min_messages=2,
            group_familiarity_min_active_days=2,
            group_familiarity_group_min_messages=20,
            group_familiarity_group_min_members=3,
            group_familiarity_probability=1.0,
            group_familiarity_cooldown_seconds=180,
        )
        self.messages = _Messages(self.settings)
        self.familiarity = GroupFamiliarity(self.messages)
        self.context = _context()
        self.start = datetime(2026, 11, 20, 5, tzinfo=timezone.utc)

    def observe(self, sender, text, offset, *, event_id=None):
        return self.familiarity.observe(
            self.context,
            text=text,
            speaker_id=sender,
            received_at=self.start + timedelta(seconds=offset),
            event_id=event_id or f"event-{sender}-{offset}",
            wakeup=group_wakeup_settings(self.settings),
        )

    def seed_member(self, sender, name, *, messages=2, days=2):
        for day in range(days):
            for index in range(messages):
                offset = day * 86400 + index * 10
                self.observe(sender, "普通聊天", offset)
        self.familiarity.record_names(_context(sender), nickname=name, at=self.start)

    def test_laya_and_action_switch_gate_collection_and_candidates(self):
        for enabled, action in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, action=action):
                messages = _Messages(self.settings, laya_enabled=enabled, action_enabled=action)
                familiarity = GroupFamiliarity(messages)
                self.assertFalse(familiarity.observe(
                    self.context, text="熟悉的人说话", speaker_id="200",
                    received_at=self.start, event_id=f"gated-{enabled}-{action}",
                    wakeup=group_wakeup_settings(self.settings)))
                result = familiarity.candidate(
                    self.context, text="熟悉的人说话", at=self.start,
                    settings=self.settings, event_id=f"gated-{enabled}-{action}")
                self.assertFalse(result["enabled"])
                self.assertEqual(result["candidate_reason"], "")

    def test_activity_ledger_never_persists_message_bodies(self):
        body = "message-body-must-not-be-persisted"
        self.assertTrue(self.observe("200", body, 0, event_id="privacy-event"))
        rows = self.messages.runtime.connection.execute(
            "SELECT value FROM runtime_metadata").fetchall()
        self.assertTrue(rows)
        self.assertNotIn(body, "\n".join(row["value"] for row in rows))

    def test_active_speaker_becomes_a_familiar_contact_candidate(self):
        self.seed_member("200", "小星")
        result = self.familiarity.candidate(
            _context("200"), text="今天终于有空来看看", at=self.start + timedelta(days=2),
            settings=self.settings, event_id="speaker-current")
        self.assertEqual(result["candidate_reason"], "familiar_contact")
        self.assertTrue(result["current_speaker"]["familiar"])
        self.assertTrue(result["sampled"])
        self.assertFalse(result["current_speaker"]["thresholds"]["messages"] > 2)

    def test_unique_mention_of_an_observed_contact_is_a_candidate(self):
        self.seed_member("200", "小星")
        context = _context("201")
        result = self.familiarity.candidate(
            context, text="刚才小星提到周末聚会", at=self.start + timedelta(days=2),
            settings=self.settings, event_id="mention-current")
        self.assertEqual(result["candidate_reason"], "familiar_contact")
        self.assertEqual([item["display_name"] for item in result["mentioned_contacts"]], ["小星"])
        self.assertFalse(result["current_speaker"]["familiar"])

    def test_mention_keeps_the_exact_alias_and_configured_primary_is_familiar(self):
        self.seed_member("200", "阿青")
        self.familiarity.record_names(
            _context("200"), nickname="阿青", card="青柠同学", at=self.start)
        mention = self.familiarity.inspect(
            _context("201"), text="青柠同学刚才说了什么？",
            at=self.start + timedelta(days=2), settings=self.settings)
        self.assertEqual(mention["potential_reason"], "familiar_contact")
        self.assertEqual(mention["mentioned_contacts"][0]["matched_name"], "青柠同学")

        settings = _values(
            **{**self.settings, "group_wakeup_primary_member_ids": json.dumps(["300"], ensure_ascii=False)}
        )
        messages = _Messages(settings)
        familiarity = GroupFamiliarity(messages)
        familiarity.record_names(_context("300"), nickname="小南", at=self.start)
        speaker = familiarity.inspect(
            _context("300"), text="晚上好", at=self.start + timedelta(days=2), settings=settings)
        self.assertEqual(speaker["potential_reason"], "familiar_contact")
        self.assertTrue(speaker["current_speaker"]["configured_primary"])
        self.assertTrue(speaker["current_speaker"]["familiar"])

        referred = familiarity.inspect(
            _context("301"), text="小南今天来吗？",
            at=self.start + timedelta(days=2), settings=settings)
        self.assertEqual(referred["potential_reason"], "familiar_contact")
        self.assertTrue(referred["mentioned_contacts"][0]["configured_primary"])

    def test_same_name_or_longer_alias_ambiguity_does_not_identify_contact(self):
        self.familiarity.record_names(self.context, nickname="小明", at=self.start)
        context = _context("201")
        self.familiarity.record_names(context, nickname="小明", at=self.start)
        result = self.familiarity.candidate(
            context, text="小明今天没来", at=self.start + timedelta(days=2),
            settings=self.settings, event_id="same-name")
        self.assertEqual(result["candidate_reason"], "")
        self.assertEqual(result["mentioned_contacts"], [])

        other = _context("202")
        self.familiarity.record_names(other, nickname="小明同学", at=self.start)
        result = self.familiarity.candidate(
            _context("203"), text="小明今天没来", at=self.start + timedelta(days=2),
            settings=self.settings, event_id="longer-alias")
        self.assertEqual(result["candidate_reason"], "")
        self.assertEqual(result["mentioned_contacts"], [])

    def test_familiar_group_can_admit_an_ordinary_member_message(self):
        settings = _values(
            group_wakeup_direct_words="[]",
            group_wakeup_context_words="[]",
            group_wakeup_interest_keywords="[]",
            group_familiarity_min_messages=20,
            group_familiarity_min_active_days=2,
            group_familiarity_group_min_messages=4,
            group_familiarity_group_min_members=2,
            group_familiarity_probability=1.0,
        )
        messages = _Messages(settings)
        familiarity = GroupFamiliarity(messages)
        for index, sender in enumerate(("200", "201")):
            for offset in (index, index + 86400):
                familiarity.observe(
                    self.context, text="群友闲聊", speaker_id=sender,
                    received_at=self.start + timedelta(seconds=offset),
                    event_id=f"group-{sender}-{offset}",
                    wakeup=group_wakeup_settings(settings))
        result = familiarity.candidate(
            _context("202"), text="今天群里挺热闹", at=self.start + timedelta(days=2),
            settings=settings, event_id="familiar-group")
        self.assertEqual(result["candidate_reason"], "familiar_group")
        self.assertTrue(result["group"]["familiar"])
        self.assertFalse(result["current_speaker"]["familiar"])

    def test_probability_zero_and_cooldown_suppress_candidates(self):
        zero_values = dict(self.settings)
        zero_values["group_familiarity_probability"] = 0.0
        zero = _values(**zero_values)
        messages = _Messages(zero)
        familiarity = GroupFamiliarity(messages)
        for offset in (0, 86400):
            familiarity.observe(
                self.context, text="普通聊天", speaker_id="200",
                received_at=self.start + timedelta(seconds=offset),
                event_id=f"zero-{offset}", wakeup=group_wakeup_settings(zero))
        result = familiarity.candidate(
            self.context, text="继续聊", at=self.start + timedelta(days=2),
            settings=zero, event_id="zero-candidate")
        self.assertEqual(result["candidate_reason"], "")
        self.assertFalse(result["sampled"])

        self.seed_member("200", "小星")
        first = self.familiarity.candidate(
            _context("200"), text="我先说一句", at=self.start + timedelta(days=2),
            settings=self.settings, event_id="cooldown-first")
        second = self.familiarity.candidate(
            _context("200"), text="再补一句", at=self.start + timedelta(days=2, seconds=1),
            settings=self.settings, event_id="cooldown-second")
        self.assertEqual(first["candidate_reason"], "familiar_contact")
        self.assertEqual(second["candidate_reason"], "")
        self.assertGreater(second["cooldown"]["remaining_seconds"], 0)


class FamiliarityPromptTests(unittest.TestCase):
    def test_familiar_candidate_keeps_silence_and_natural_interjection(self):
        questions = decision_questions({
            "scope_kind": "group",
            "candidate_reason": "familiar_contact",
            "group_participation_policy": {
                "question_enabled": False,
                "interjection_enabled": True,
            },
            "group_familiarity": {"candidate_reason": "familiar_contact"},
        }, {})
        action = questions["action"]
        self.assertIn("不是必答", action["instructions"])
        self.assertIn("角色人设", action["instructions"])
        self.assertIn("记忆线索", action["instructions"])
        self.assertIn("当前意愿", action["instructions"])
        self.assertIn("沉默", action["criteria"]["silent"])
        self.assertIn("插话", action["criteria"]["interject"])

    def test_rendered_state_says_familiarity_is_not_an_invitation(self):
        facts = {
            "scope_kind": "group",
            "message_kind": "text",
            "user_text": "今天人真多",
            "candidate_reason": "familiar_group",
            "group_participation_policy": {"question_enabled": False, "interjection_enabled": True},
            "group_familiarity": {
                "candidate_reason": "familiar_group",
                "current_speaker": {"display_name": "小星", "messages": 8, "active_days": 3,
                                    "familiar": True, "thresholds": {"messages": 6, "active_days": 2}},
                "mentioned_contacts": [],
                "group": {"messages": 32, "active_days": 4, "member_count": 5,
                          "familiar": True, "thresholds": {"messages": 20, "members": 3}},
                "sampled": True,
            },
            "role_profile_summary": {
                "personality": "慢热但会主动关心熟人",
                "social_distance": "熟人之间说话自然",
            },
        }
        model = json.loads(render_model_state(build_state(facts)))
        rendered = model["facts"]
        self.assertIn("熟人/熟群候选", rendered["address_context"])
        self.assertIn("不是必答", rendered["address_context"])
        self.assertEqual(rendered["group_familiarity"]["basis"], "observed_group_activity_proxy")
        self.assertFalse(rendered["group_familiarity"]["candidate_is_invitation"])
        self.assertEqual(rendered["role_profile_summary"]["personality"], "慢热但会主动关心熟人")


    def test_group_memory_and_willingness_are_bounded_participation_references(self):
        facts = {
            "scope_kind": "group",
            "message_kind": "text",
            "user_text": "刚才那个安排呢？",
            "candidate_reason": "familiar_contact",
            "group_participation_policy": {"question_enabled": False, "interjection_enabled": True},
            "group_familiarity": {"candidate_reason": "familiar_contact"},
            "group_memory_summary": {
                "items": [{"speaker_id": "200", "said_at": "2026-11-20T05:00:00+00:00",
                           "statement": "周末想去散步，但还没定时间。",
                           "speaks_for_current_actor": False}],
            },
            "group_participation_summary": {
                "observation": {"mood": {"top": "轻松", "score": 12, "tension": 0}},
                "self_initiated_speech": {
                    "allowed": True, "remaining": 2, "max_daily": 3,
                    "elapsed_min": 20, "min_interval_min": 10,
                },
            },
        }
        model = json.loads(render_model_state(build_state(facts)))["facts"]
        self.assertEqual(model["group_memory_summary"]["basis"], "same_group_public_memory")
        self.assertEqual(model["group_memory_summary"]["items"][0]["speaker"], "其他群成员")
        self.assertIn("散步", model["group_memory_summary"]["items"][0]["statement"])
        reference = model["group_participation_reference"]
        self.assertTrue(reference["self_initiated_speech"]["allowed"])
        self.assertEqual(reference["self_initiated_speech"]["remaining"], 2)
        self.assertEqual(reference["group_atmosphere"]["mood"]["top"], "轻松")
        self.assertIn("不是必须接话", reference["meaning"])


class FamiliarityAdmissionTests(unittest.TestCase):
    def test_familiar_mention_wins_before_a_broader_behavior_candidate(self):
        settings = _values(
            group_wakeup_direct_words="[]",
            group_wakeup_context_words="[]",
            group_wakeup_interest_keywords="[]",
            group_familiarity_min_messages=2,
            group_familiarity_min_active_days=2,
            group_familiarity_probability=1.0,
        )
        messages = _Messages(settings)
        familiarity = GroupFamiliarity(messages)
        start = datetime(2026, 11, 20, 5, tzinfo=timezone.utc)
        for day in range(2):
            for index in range(2):
                familiarity.observe(
                    _context("200"), text="普通聊天",
                    speaker_id="200", event_id=f"mention-{day}-{index}",
                    received_at=start + timedelta(days=day, seconds=index),
                    wakeup=group_wakeup_settings(settings))
        familiarity.record_names(_context("200"), nickname="小星", at=start)
        messages.group_context = NS(authorize=lambda *_args: None)
        admission = GroupReplyAdmission(messages)
        admission._behavior_candidate = lambda *_args: True
        reason = admission.candidate(
            _context("201"), text="小星刚才说周末聚会", instruction="",
            at=start + timedelta(days=2), settings=settings,
            participation={"enable_group_repeat_follow": False},
            mentions_others=True, event_id="familiar-mention-priority",
            continuation={"limit_reached": False, "candidate_reason": ""})
        self.assertEqual(reason, "familiar_contact")


class _MemoryOwner:
    def __init__(self):
        self.calls = 0

    def capture(self, context, key):
        self.calls += 1
        return {"context_key": context.key, "batch_id": key}

    def contributions(self, snapshot, context, key):
        return (
            NS(payload=json.dumps({
                "speaker_id": "200",
                "said_at": "2026-11-20T05:00:00+00:00",
                "statement": "We should decide the weekend plan later.",
            })),
        )


class CoreDecisionGroupFactsTests(unittest.TestCase):
    def test_group_memory_summary_reads_only_the_same_group_owner(self):
        owner = _MemoryOwner()
        application = NS(
            messages=NS(group_memory=NS(context=owner), runtime=NS()))
        at = datetime(2026, 11, 20, 6, tzinfo=timezone.utc)

        group = _group_memory_summary(application, _context(), "event-1", at)
        self.assertEqual(group["basis"], "same_group_public_memory")
        self.assertEqual(group["visibility"], "group")
        self.assertEqual(group["items"][0]["speaker_id"], "200")
        self.assertFalse(group["items"][0]["speaks_for_current_actor"])
        self.assertEqual(owner.calls, 1)

        private_scope = NS(
            **{**vars(_context()), "scope": ScopeRef(
                "installation", "qq", ScopeKind.PRIVATE, "private", "bot",
                "bot-account", "conversation", "session")})
        self.assertIsNone(_group_memory_summary(
            application, private_scope, "event-2", at))
        self.assertEqual(owner.calls, 1)

    def test_group_participation_summary_exposes_willingness_not_a_command(self):
        participation = object()
        application = NS(
            tools=NS(group_participation=participation),
            reply_settings=NS(snapshot=lambda *_args: NS(values={
                "conversation.presentation.group_wakeup_enabled": True,
            })),
        )
        compact = {
            "observation": {"mood": {"top": "light", "score": 12}},
            "interjection": {"allowed": True, "remaining": 2, "max_daily": 3},
        }
        at = datetime(2026, 11, 20, 6, tzinfo=timezone.utc)
        with patch(
                "companion_v2.conversation.native_participation.native_participation_turn",
                return_value={"compact_facts": compact}):
            result = _group_participation_summary(
                application, _context(), NS(), {}, at)
        self.assertEqual(result["observation"]["mood"]["top"], "light")
        self.assertEqual(result["self_initiated_speech"]["remaining"], 2)


if __name__ == "__main__":
    unittest.main()
