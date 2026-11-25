"""Role-owned autonomous interests used only to widen group candidates.

Interests belong to the bot role, not to the group currently speaking.  They
are learned from the role's persona, its observations and memories, and remain
separate from administrator-provided wake-up keywords.  An active match only
invites Laya to consider participating; it never forces a reply.
"""

from __future__ import annotations

import json
import re

from companion_v2.execution.models import TaskError, digest, encode, stamp
from companion_v2.foundation.refs import ScopeKind
from companion_v2.model_gateway.models import ToolSpec


ACTION = "group.interest.manage"
VERSION = "conversation.group_autonomous_interests.v1"
PREFIX = VERSION + ":"
MAX_ACTIVE = 24
MAX_ITEMS = 64
MAX_KEYWORDS = 12
MAX_TOPIC_CHARS = 120
MAX_KEYWORD_CHARS = 64
MAX_BASIS_CHARS = 600
MAX_REASON_CHARS = 300
BASIS_FIELDS = frozenset(("context", "persona", "memory", "willingness"))


def _owner(context):
    scope = context.scope
    return digest([
        "role-autonomous-interests.v1",
        scope.installation_id,
        scope.platform,
        scope.bot_id,
        scope.account_id,
        context.role_binding.role.subject_id,
    ])


def _key(context):
    return PREFIX + _owner(context)


def _empty():
    return {"version": VERSION, "revision": 0, "items": []}


def _valid(document):
    return (
        type(document) is dict
        and document.get("version") == VERSION
        and type(document.get("revision")) is int
        and document["revision"] >= 0
        and type(document.get("items")) is list
    )


def _clean(value, name, maximum, *, required=False):
    if type(value) is not str:
        if not required and value is None:
            return ""
        raise TaskError("group_interest_invalid", f"{name} must be text")
    value = " ".join(value.split())
    if (required and not value) or len(value) > maximum:
        raise TaskError(
            "group_interest_invalid",
            f"{name} must be between 1 and {maximum} characters",
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise TaskError("group_interest_invalid", f"{name} contains control characters")
    return value


def _keywords(values):
    if type(values) is not list or not 1 <= len(values) <= MAX_KEYWORDS:
        raise TaskError(
            "group_interest_invalid",
            f"keywords must contain between 1 and {MAX_KEYWORDS} items",
        )
    result = []
    seen = set()
    for value in values:
        keyword = _clean(value, "keyword", MAX_KEYWORD_CHARS, required=True)
        folded = keyword.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        result.append(keyword)
    if not result:
        raise TaskError("group_interest_invalid", "at least one keyword is required")
    return result


def _basis(value):
    if value is None:
        return {}
    if type(value) is not dict or set(value) - BASIS_FIELDS:
        raise TaskError(
            "group_interest_invalid",
            "basis may contain only context, persona, memory and willingness",
        )
    result = {}
    for name in BASIS_FIELDS:
        if name not in value:
            continue
        text = _clean(value[name], f"basis.{name}", MAX_BASIS_CHARS)
        if text:
            result[name] = text
    return result


def _matches(text, keyword):
    folded = keyword.casefold()
    if not folded:
        return False
    if folded.isascii() and any(char.isalnum() for char in folded):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(folded) + r"(?![A-Za-z0-9])"
        return re.search(pattern, text.casefold()) is not None
    return folded in text.casefold()


class GroupAutonomousInterests:
    def __init__(self, messages):
        self.messages = messages
        self.runtime = messages.runtime
        self.tasks = messages.tasks

    def enabled(self, context, settings=None):
        if context.scope.kind is not ScopeKind.GROUP:
            return False
        if isinstance(settings, dict):
            return settings.get("group_autonomous_interests_enabled", True) is True
        reply_settings = getattr(self.messages, "reply_settings", None)
        if reply_settings is None:
            return True
        try:
            snapshot = reply_settings.snapshot(
                context.actor, context, self.tasks.now())
            return snapshot.values.get(
                "conversation.presentation.group_autonomous_interests_enabled",
                True,
            ) is True
        except Exception:
            return True

    def _read(self, context):
        row = self.runtime.read_one(
            "SELECT value FROM runtime_metadata WHERE key = ?", (_key(context),))
        if row is None:
            return _empty()
        try:
            document = json.loads(row["value"])
        except (TypeError, ValueError):
            return _empty()
        return document if _valid(document) else _empty()

    def _snapshot(self, document):
        items = []
        for item in document["items"]:
            if type(item) is not dict:
                continue
            items.append({
                "interest_id": item.get("interest_id"),
                "revision": item.get("revision"),
                "state": item.get("state"),
                "topic": item.get("topic"),
                "keywords": list(item.get("keywords") or ()),
                "weight": item.get("weight"),
                "basis": dict(item.get("basis") or {}),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "deactivated_at": item.get("deactivated_at"),
                "deactivation_reason": item.get("deactivation_reason"),
            })
        return {
            "version": VERSION,
            "owner": "role",
            "revision": document["revision"],
            "active_count": sum(item["state"] == "active" for item in items),
            "items": items[:MAX_ITEMS],
        }

    def snapshot(self, context):
        self.messages.group_context.authorize(context, self.tasks.now())
        return self._snapshot(self._read(context))

    def active_items(self, context, settings=None):
        if not self.enabled(context, settings):
            return ()
        document = self._read(context)
        return tuple(
            item for item in document["items"]
            if type(item) is dict and item.get("state") == "active"
        )

    def prompt_items(self, context, settings=None, *, limit=8):
        return tuple({
            "interest_id": item.get("interest_id"),
            "revision": item.get("revision"),
            "topic": item.get("topic"),
            "keywords": list(item.get("keywords") or ()),
        } for item in self.active_items(context, settings)[:limit])

    def match(self, context, text, *, settings=None):
        if type(text) is not str or not text.strip():
            return None
        matched = []
        for item in self.active_items(context, settings):
            keywords = item.get("keywords")
            if type(keywords) is not list:
                continue
            hits = [keyword for keyword in keywords
                    if type(keyword) is str and _matches(text, keyword)]
            if not hits:
                continue
            matched.append({
                "interest_id": item.get("interest_id"),
                "revision": item.get("revision"),
                "topic": item.get("topic"),
                "matched_keywords": hits,
                "weight": item.get("weight", 1.0),
            })
        if not matched:
            return None
        return {
            "basis": "role_owned_interest",
            "items": matched,
            "weight": max(float(item["weight"]) for item in matched),
        }

    def execute(self, context, arguments):
        if context.scope.kind is not ScopeKind.GROUP:
            raise TaskError("group_interest_scope", "autonomous interests are managed in an authenticated group")
        self.messages.group_context.authorize(context, self.tasks.now())
        operation = arguments.get("operation") if isinstance(arguments, dict) else None
        if operation not in ("create", "update", "deactivate"):
            raise TaskError("group_interest_invalid", "operation must be create, update or deactivate")
        allowed = {
            "operation", "interest_id", "expected_revision", "topic",
            "keywords", "weight", "basis", "reason",
        }
        if set(arguments) - allowed:
            raise TaskError("group_interest_invalid", "unexpected interest fields")
        now = stamp(self.tasks.now())
        key = _key(context)
        with self.runtime.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = ?", (key,)
            ).fetchone()
            try:
                document = json.loads(row["value"]) if row is not None else _empty()
            except (TypeError, ValueError):
                document = _empty()
            if not _valid(document):
                document = _empty()

            if operation == "create":
                if set(arguments) - {"operation", "topic", "keywords", "weight", "basis"}:
                    raise TaskError("group_interest_invalid", "create accepts topic, keywords, weight and basis")
                active = [item for item in document["items"]
                          if item.get("state") == "active"]
                if len(active) >= MAX_ACTIVE:
                    raise TaskError("group_interest_limit", "the role has too many active interests")
                if len(document["items"]) >= MAX_ITEMS:
                    raise TaskError("group_interest_limit", "the role interest history is full")
                topic = _clean(
                    arguments.get("topic"), "topic", MAX_TOPIC_CHARS, required=True)
                keywords = _keywords(arguments.get("keywords"))
                if any(str(item.get("topic") or "").casefold() == topic.casefold()
                       for item in active):
                    raise TaskError("group_interest_exists", "an active interest already has this topic")
                weight = arguments.get("weight", 1.0)
                if type(weight) not in (int, float) or isinstance(weight, bool) or not 0.25 <= weight <= 2.0:
                    raise TaskError("group_interest_invalid", "weight must be between 0.25 and 2")
                item = {
                    "interest_id": "interest-" + digest([
                        _owner(context), document["revision"] + 1, topic, now,
                    ])[:24],
                    "revision": 1,
                    "state": "active",
                    "topic": topic,
                    "keywords": keywords,
                    "weight": float(weight),
                    "basis": _basis(arguments.get("basis")),
                    "created_at": now,
                    "updated_at": now,
                    "deactivated_at": None,
                    "deactivation_reason": None,
                }
                document["items"].append(item)
            else:
                interest_id = arguments.get("interest_id")
                expected = arguments.get("expected_revision")
                if type(interest_id) is not str or type(expected) is not int or expected < 1:
                    raise TaskError(
                        "group_interest_invalid",
                        "interest_id and expected_revision are required",
                    )
                item = next(
                    (entry for entry in document["items"]
                     if entry.get("interest_id") == interest_id),
                    None,
                )
                if item is None:
                    raise TaskError("group_interest_missing", "the interest no longer exists")
                if item.get("revision") != expected:
                    raise TaskError("group_interest_revision", "the interest changed; refresh before editing")
                if item.get("state") != "active":
                    raise TaskError("group_interest_inactive", "the interest is already inactive")

                if operation == "deactivate":
                    if set(arguments) - {
                            "operation", "interest_id", "expected_revision", "reason"}:
                        raise TaskError(
                            "group_interest_invalid",
                            "deactivate accepts interest_id, expected_revision and reason",
                        )
                    reason = _clean(
                        arguments.get("reason"), "reason", MAX_REASON_CHARS)
                    item.update({
                        "revision": expected + 1,
                        "state": "inactive",
                        "updated_at": now,
                        "deactivated_at": now,
                        "deactivation_reason": reason or None,
                    })
                else:
                    if set(arguments) - {
                            "operation", "interest_id", "expected_revision",
                            "topic", "keywords", "weight", "basis"}:
                        raise TaskError(
                            "group_interest_invalid",
                            "update accepts topic, keywords, weight and basis",
                        )
                    if "topic" in arguments:
                        topic = _clean(
                            arguments["topic"], "topic", MAX_TOPIC_CHARS, required=True)
                        if any(
                            entry is not item
                            and entry.get("state") == "active"
                            and str(entry.get("topic") or "").casefold() == topic.casefold()
                            for entry in document["items"]
                        ):
                            raise TaskError("group_interest_exists", "an active interest already has this topic")
                        item["topic"] = topic
                    if "keywords" in arguments:
                        item["keywords"] = _keywords(arguments["keywords"])
                    if "weight" in arguments:
                        weight = arguments["weight"]
                        if (type(weight) not in (int, float) or isinstance(weight, bool)
                                or not 0.25 <= weight <= 2.0):
                            raise TaskError("group_interest_invalid", "weight must be between 0.25 and 2")
                        item["weight"] = float(weight)
                    if "basis" in arguments:
                        item["basis"] = _basis(arguments["basis"])
                    item["revision"] = expected + 1
                    item["updated_at"] = now

            document["revision"] += 1
            connection.execute(
                """INSERT INTO runtime_metadata(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, json.dumps(document, ensure_ascii=False, sort_keys=True)),
            )
        return {
            "state": "succeeded",
            "effect": "committed",
            "payload": {
                **self._snapshot(document),
                "operation": operation,
                "meaning": (
                    "Active interests widen the role's candidate opportunities. "
                    "Deactivation is a reversible soft withdrawal."
                ),
            },
        }


class GroupAutonomousInterestTools:
    actions = frozenset((ACTION,))
    instruction = (
        "【自主兴趣】你可以根据当前群聊所见所闻、原人格、群公共记忆和角色全局记忆，"
        "判断某个话题是否真正持续吸引自己。形成稳定兴趣时调用 group.interest.manage："
        "create 建立话题与少量关键词，update 修订，deactivate 软撤回；"
        "软撤回保留历史但不再参与候选。不要把某位成员一时的喜好直接当成自己的爱好，"
        "也不要仅凭单个表情或一句玩笑频繁新建。兴趣命中只表示可以申请参与，"
        "仍须结合当前语境和意愿决定是否接话。"
    )
    catalog = (
        ToolSpec(
            ACTION,
            "管理 Bot 角色自己的群聊自主兴趣。create/update/deactivate 都是角色主动决定；"
            "兴趣只会提高候选机会，不会强制回复。",
            encode({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "update", "deactivate"],
                    },
                    "interest_id": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "topic": {"type": "string", "maxLength": MAX_TOPIC_CHARS},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": MAX_KEYWORD_CHARS},
                        "minItems": 1,
                        "maxItems": MAX_KEYWORDS,
                        "uniqueItems": True,
                    },
                    "weight": {
                        "type": "number",
                        "minimum": 0.25,
                        "maximum": 2.0,
                    },
                    "basis": {
                        "type": "object",
                        "properties": {
                            "context": {"type": "string", "maxLength": MAX_BASIS_CHARS},
                            "persona": {"type": "string", "maxLength": MAX_BASIS_CHARS},
                            "memory": {"type": "string", "maxLength": MAX_BASIS_CHARS},
                            "willingness": {"type": "string", "maxLength": MAX_BASIS_CHARS},
                        },
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
                },
                "required": ["operation"],
                "additionalProperties": False,
            }),
        ),
    )

    def __init__(self, interests):
        self.interests = interests

    def catalog_for(self, context, settings=None):
        if not self.interests.enabled(context, settings):
            return ()
        self.interests.messages.group_context.authorize(
            context, self.interests.tasks.now())
        return self.catalog

    def execute(self, context, arguments, settings=None):
        if not self.interests.enabled(context, settings):
            raise TaskError(
                "group_interest_disabled",
                "autonomous interests are disabled for this group turn",
            )
        return self.interests.execute(context, arguments)
