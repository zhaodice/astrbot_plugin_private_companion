"""The bot role's own cross-window recall of observed conversations.

This owner belongs to the role, never to the current participant.  A stranger's
first private message is therefore something the role can remember later.  Raw
text is still projected for the current audience: public group messages may be
recalled elsewhere, while another person's private wording is withheld.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json

from companion_v2.conversation.recalls import CONSUMER as RECALL_CONSUMER
from companion_v2.conversation.recalls import MessageRecall
from companion_v2.conversation.reset import source_current
from companion_v2.execution.models import TaskContext, TaskError, digest, encode, stamp
from companion_v2.foundation.refs import ScopeKind, SourceRef, normalize_utc
from companion_v2.memory.codec import source as decode_source
from companion_v2.memory.models import integer, text
from companion_v2.memory.search import SearchIndex, terms
from companion_v2.model_gateway.models import ToolSpec
from companion_v2.relationships.provenance import same_private_route
from companion_v2.storage.codec import decode_timestamp, scope_from_row
from companion_v2.storage.native_dedup import resolve_context_json


ACTION = "memory.global.query"
SCOPE_FILTERS = frozenset(("any", "group", "private", "current"))
MAX_CANDIDATES = 256
MAX_RESULTS = 20
MAX_TEXT = 4000
GENERIC_QUERY_WORDS = (
    "今天", "最近", "刚刚", "有什么", "什么", "好玩", "有趣", "事情",
    "新鲜事", "趣事", "发生了", "聊了", "说说", "其他群", "别的群",
    "群里", "群聊", "大家", "见过", "记得", "吗", "呢", "的",
)
ENABLED_SETTING = "role_global_memory_enabled"
ENABLED_SETTING_ID = "conversation.presentation." + ENABLED_SETTING


def _setting_enabled(settings, name=ENABLED_SETTING):
    """Read a frozen value while accepting both bare and fully qualified keys."""

    if not isinstance(settings, Mapping):
        return None
    for key in (name, "conversation.presentation." + name):
        if key in settings:
            return settings[key] is True
    return None


def role_global_owner(context):
    """Stable role owner that deliberately excludes actor, window and session."""

    scope = context.scope
    return digest([
        "role-global-memory.v1",
        scope.installation_id,
        scope.platform,
        scope.bot_id,
        scope.account_id,
        context.role_binding.role.subject_id,
    ])


def _parse_time(value, name):
    if value is None:
        return None
    if type(value) is not str or not 1 <= len(value) <= 64:
        raise ValueError(f"{name} must be an ISO-8601 time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        # Chat callers naturally omit an offset.  Interpret those values in the
        # host's configured local time rather than the machine's ambient zone.
        from zoneinfo import ZoneInfo
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return normalize_utc(parsed, field=name)


def _same_role(context, original):
    scope, other = context.scope, original.scope
    return (
        scope.installation_id == other.installation_id
        and scope.platform == other.platform
        and scope.bot_id == other.bot_id
        and scope.account_id == other.account_id
        and context.role_binding.role == original.role_binding.role
        and other.kind in (ScopeKind.PRIVATE, ScopeKind.GROUP)
    )


def _same_private_participant(original, current):
    """Keep one private participant's wording across resets and role revisions."""

    return (
        same_private_route(original.scope, current.scope)
        and original.actor.subject == current.actor.subject
    )


def _project(original, kind, payload, current):
    """Return text only when this audience may hear that exact source."""

    if original.scope.kind is ScopeKind.GROUP:
        return {
            "visibility": "group_public",
            "text": payload,
            "reason": "This was spoken in a group audience and may be recalled across windows.",
        }
    if _same_private_participant(original, current):
        return {
            "visibility": "current_private",
            "text": payload,
            "reason": "This private exchange belongs to the current participant.",
        }
    if kind == "assistant":
        return {
            "visibility": "role_own_private",
            "text": payload,
            "reason": "These are the role's own words, remembered without exposing another user's input.",
        }
    return {
        "visibility": "other_private_withheld",
        "text": "",
        "reason": "The role remembers this private exchange, but its original wording is withheld from this audience.",
    }


def _is_generic_query(query):
    remaining = query
    for phrase in GENERIC_QUERY_WORDS:
        remaining = remaining.replace(phrase, "")
    return not any(char.isalnum() for char in remaining)


class RoleGlobalMemory:
    """Authoritative, role-scoped native conversation recall."""

    def __init__(self, messages, *, index=None):
        self.messages = messages
        self.runtime = messages.runtime
        self.tasks = messages.tasks
        self.reply_settings = getattr(messages, "reply_settings", None)
        self.index = index or SearchIndex(self.runtime.data_root)

    def _enabled(self, context, settings=None):
        frozen = _setting_enabled(settings)
        if frozen is not None:
            return frozen
        if self.reply_settings is None:
            return True
        try:
            snapshot = self.reply_settings.snapshot(
                context.actor, context, self.tasks.now())
            values = getattr(snapshot, "values", {})
        except Exception:
            return True
        return values.get(ENABLED_SETTING_ID, True) is True

    def enabled(self, context, settings=None):
        return self._enabled(context, settings)

    def _candidates(self, context, *, since, until, scope_filter):
        scope = context.scope
        clauses = [
            "n.kind IN ('input', 'assistant')",
            "n.delivery_state IN ('input_observed', 'send_returned')",
            "json_extract(COALESCE(c.context_json, n.context_json), '$.role_binding.scope.installation_id') = ?",
            "json_extract(COALESCE(c.context_json, n.context_json), '$.role_binding.scope.platform') = ?",
            "json_extract(COALESCE(c.context_json, n.context_json), '$.role_binding.scope.bot_id') = ?",
            "json_extract(COALESCE(c.context_json, n.context_json), '$.role_binding.scope.account_id') = ?",
            "json_extract(COALESCE(c.context_json, n.context_json), '$.role_binding.role.subject_id') = ?",
        ]
        values = [scope.installation_id, scope.platform, scope.bot_id,
                  scope.account_id, context.role_binding.role.subject_id]
        if scope_filter == "group":
            clauses.append("n.scope_kind = 'group'")
        elif scope_filter == "private":
            clauses.append("n.scope_kind = 'private'")
        elif scope_filter == "current":
            # Reset rotates conversation/session ids while preserving the
            # authenticated native route. Recall "current" by that stable
            # route instead of returning only the exact context-key window.
            clauses.extend((
                "n.scope_kind = ?",
                "json_extract(COALESCE(c.context_json, n.context_json), "
                "'$.role_binding.scope.native_scope_id') = ?",
            ))
            values.extend((scope.kind.value, scope.native_scope_id))
        if since is not None:
            clauses.append("n.effective_at >= ?")
            values.append(stamp(since))
        if until is not None:
            clauses.append("n.effective_at < ?")
            values.append(stamp(until))
        values.append(MAX_CANDIDATES)
        return self.runtime.read_all("""
            SELECT n.observation_id, n.context_key, n.event_id, n.kind,
                   n.delivery_state, n.context_json, n.source_json,
                   n.content_hash, n.observed_at, n.effective_at,
                   n.scope_kind, s.payload_json, s.state AS source_state,
                   s.expires_at AS source_expires_at, s.content_hash AS source_hash,
                   s.source_id, s.lineage_id, s.source_revision, s.issued_at,
                   s.owner, s.installation_id, s.platform, s.native_scope_id,
                   s.bot_id, s.account_id, s.conversation_id, s.session_id
            FROM conversation_native_observations n
            LEFT JOIN conversation_contexts c ON c.context_hash = n.context_hash
            JOIN sources s
              ON s.source_id = json_extract(n.source_json, '$.source_id')
             AND s.lineage_id = json_extract(n.source_json, '$.version.lineage_id')
             AND s.source_revision = json_extract(n.source_json, '$.version.revision')
            WHERE """ + " AND ".join(clauses) + """
            ORDER BY n.effective_at DESC, n.observation_id DESC LIMIT ?
        """, tuple(values))

    def _validate(self, row, context):
        try:
            original = TaskContext.restore(
                json.loads(resolve_context_json(self.runtime, row["context_json"])))
            reference = decode_source(json.loads(row["source_json"]))
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(reference, SourceRef) or not _same_role(context, original):
            return None
        stored = SourceRef(
            row["source_id"], row["owner"], scope_from_row(row),
            reference.version, decode_timestamp(row["issued_at"]),
            decode_timestamp(row["source_expires_at"])
            if row["source_expires_at"] else None,
        )
        if stored != reference or reference.scope != original.scope:
            return None
        if row["source_state"] != "active" or row["source_hash"] != row["content_hash"]:
            return None
        if row["scope_kind"] != original.scope.kind.value:
            return None
        now = self.tasks.now()
        if reference.is_expired_at(now) or now < reference.issued_at:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return None
        if type(payload) is not str or digest(payload) != row["content_hash"]:
            return None
        try:
            if not source_current(self.runtime, reference):
                return None
        except Exception:
            return None
        if row["kind"] == "input":
            marker = MessageRecall.from_scope(
                original.scope, row["event_id"],
                original.actor.principal.native_account_id)
            if any(self.runtime.read_one(
                    "SELECT 1 FROM inbox WHERE consumer = ? AND event_id = ?",
                    (RECALL_CONSUMER, key)) is not None
                   for key in (marker.key, marker.legacy_key)):
                return None
        return {
            "id": row["observation_id"],
            "original": original,
            "reference": reference,
            "kind": row["kind"],
            "event_id": row["event_id"],
            "effective_at": row["effective_at"],
            "speaker_id": (
                original.actor.principal.native_account_id
                if row["kind"] == "input"
                else original.role_binding.role.subject_id
            ),
            "payload": payload,
        }

    def _documents(self, candidates):
        documents = []
        for candidate in candidates:
            kind = "人类发言" if candidate["kind"] == "input" else "机器人发言"
            scope = candidate["original"].scope.kind.value
            documents.append({
                "id": candidate["id"],
                "revision": candidate["reference"].version.revision,
                "scope": scope,
                "speaker_id": candidate["speaker_id"],
                "effective_at": candidate["effective_at"],
                "text": f"{kind} {scope} {candidate['payload']}"[:MAX_TEXT],
            })
        return documents

    async def _vector_recall(self, owner, query, documents):
        semantic = getattr(getattr(self.messages, "tools", None), "semantic", None)
        if semantic is None:
            return None, None
        try:
            provider = semantic.store.settings()["embedding_provider"]
            corpus = [document["text"] for document in documents[:48]]
            binding, vectors = await semantic.gateway.embed(owner, corpus, provider)
            for document, vector in zip(documents, vectors):
                self.index.put_vector(
                    owner, document["id"], model=str(binding),
                    values=tuple(vector), expected_hash=digest(document))
            _, query_vector = await semantic.gateway.embed(owner, [query], provider)
            if not query_vector or len(query_vector[0]) != len(vectors[0]):
                return None, None
            return tuple(query_vector[0]), str(binding)
        except Exception:
            return None, None

    async def query(self, context, *, query, since=None, until=None,
                    limit=12, scope_filter="any", settings=None):
        scope = context.scope
        text(query, "query", 2000)
        integer(limit, "limit", 1, MAX_RESULTS)
        if scope_filter not in SCOPE_FILTERS:
            raise ValueError("scope_filter must be any, group, private or current")
        if not self._enabled(context, settings):
            return {"state": "disabled", "owner": "role", "items": [],
                    "reason": "role_global_memory_disabled"}
        start, end = _parse_time(since, "since"), _parse_time(until, "until")
        if start is not None and end is not None and start >= end:
            raise ValueError("since must precede until")
        rows = self._candidates(
            context, since=start, until=end, scope_filter=scope_filter)
        candidates = [
            candidate for row in rows
            if (candidate := self._validate(row, context)) is not None
        ]
        if scope_filter == "current" and scope.kind is ScopeKind.PRIVATE:
            candidates = [
                candidate for candidate in candidates
                if _same_private_participant(candidate["original"], context)
            ]
        if not candidates:
            return {
                "state": "empty",
                "owner": "role",
                "mode": "lexical",
                "items": [],
                "coverage": {"authoritative_candidates": len(rows), "valid": 0},
            }

        owner = role_global_owner(context)
        documents = self._documents(candidates)
        self.index.upsert_documents(owner, documents, relationship_owner=owner)
        allowed = {document["id"] for document in documents}
        vector, model = await self._vector_recall(owner, query, documents)
        if _is_generic_query(query):
            ranked = [
                candidate["id"] for candidate in sorted(
                    candidates,
                    key=lambda item: (item["effective_at"], item["id"]),
                    reverse=True,
                )
            ]
            mode, coverage = "recent", {
                "embedded": 0,
                "eligible": len(allowed),
                "model": model,
            }
        else:
            ranked, mode, coverage = self.index.rank(
                owner, query, allowed=allowed, vector=vector, model=model)

        # A conversational query such as "今天有什么好玩的" should still see
        # recent events when lexical terms do not overlap exactly.
        if len(ranked) < min(limit, 4):
            ranked.extend(candidate["id"] for candidate in candidates
                          if candidate["id"] not in ranked)
        by_id = {candidate["id"]: candidate for candidate in candidates}
        selected = []
        for identifier in ranked[:limit]:
            candidate = by_id[identifier]
            projection = _project(
                candidate["original"], candidate["kind"],
                candidate["payload"], context)
            selected.append({
                "observation_id": identifier,
                "kind": candidate["kind"],
                "said_at": candidate["effective_at"],
                "speaker_id": candidate["speaker_id"],
                "scope": {
                    "kind": candidate["original"].scope.kind.value,
                    "native_scope_id": candidate["original"].scope.native_scope_id,
                    "session_id": candidate["original"].scope.session_id,
                },
                "projection": projection,
                "text": projection["text"][:MAX_TEXT],
                "evidence": {
                    "source_id": candidate["reference"].source_id,
                    "owner": candidate["reference"].owner,
                    "lineage_id": candidate["reference"].version.lineage_id,
                    "revision": candidate["reference"].version.revision,
                    "issued_at": stamp(candidate["reference"].issued_at),
                },
            })
        return {
            "state": "ok" if selected else "empty",
            "owner": "role",
            "meaning": (
                "These are the role's own observations across private chats and groups. "
                "Withheld private wording is still remembered by the role but must not be disclosed."
            ),
            "mode": mode,
            "items": selected,
            "coverage": {
                **(coverage or {}),
                "authoritative_candidates": len(rows),
                "valid": len(candidates),
                "withheld_private": sum(
                    item["projection"]["visibility"] == "other_private_withheld"
                    for item in selected),
            },
        }

    def validate_result(self, context, payload, settings=None):
        """Recheck a persisted tool result against current sources and audience."""

        if type(payload) is not dict or payload.get("owner") != "role":
            raise TaskError("role_memory_result_invalid",
                            "the persisted global-memory result is malformed")
        if payload.get("state") not in ("ok", "empty", "disabled"):
            raise TaskError("role_memory_result_invalid",
                            "the persisted global-memory state is invalid")
        if not self._enabled(context, settings):
            raise TaskError("role_memory_disabled",
                            "role global memory is disabled")
        items = payload.get("items", [])
        if type(items) is not list or len(items) > MAX_RESULTS:
            raise TaskError("role_memory_result_invalid",
                            "the persisted global-memory result is malformed")
        for item in items:
            if type(item) is not dict:
                raise TaskError("role_memory_result_invalid",
                                "the persisted global-memory item is malformed")
            row = self.runtime.read_one(
                "SELECT * FROM conversation_native_observations "
                "WHERE observation_id = ?", (item.get("observation_id"),))
            if row is None:
                raise TaskError("role_memory_source_changed",
                                "an observed conversation is no longer available")
            reference_doc = json.loads(row["source_json"])
            source_row = self.runtime.read_one(
                """SELECT s.* FROM sources s
                   WHERE s.source_id = ? AND s.lineage_id = ? AND s.source_revision = ?""",
                (reference_doc["source_id"],
                 reference_doc["version"]["lineage_id"],
                 reference_doc["version"]["revision"]),
            )
            if source_row is None:
                raise TaskError("role_memory_source_changed",
                                "the original observation source is unavailable")
            merged = {
                **dict(row),
                **dict(source_row),
                "source_state": source_row["state"],
                "source_expires_at": source_row["expires_at"],
                "source_hash": source_row["content_hash"],
            }
            candidate = self._validate(merged, context)
            if candidate is None:
                raise TaskError("role_memory_source_changed",
                                "an observed conversation changed or was withdrawn")
            projection = _project(
                candidate["original"], candidate["kind"],
                candidate["payload"], context)
            if item.get("projection") != projection:
                raise TaskError("role_memory_audience_changed",
                                "the global-memory result no longer fits this audience")
        return payload


class RoleGlobalMemoryTools:
    actions = frozenset((ACTION,))
    instruction = (
        "【角色全局记忆】需要回忆其他群、私聊或陌生人以前说过什么时，调用 "
        "memory.global.query。它按当前 Bot 角色检索，不要求对方是当前会话用户。"
        "group_public 内容可以跨群引用；other_private_withheld 表示角色确实记得"
        "这段私聊，但不得向当前受众复述原文。不要说“我没记忆”。"
    )
    catalog = (
        ToolSpec(
            ACTION,
            "查询本 Bot 角色跨私聊、跨群、跨用户的全局所见所闻。"
            "适合回答“今天有什么好玩的事”“别的群聊了什么”。"
            "返回逐条来源与受众投影；other_private_withheld 的原文不得外泄。",
            encode({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
                    "scope_filter": {
                        "type": "string",
                        "enum": ["any", "group", "private", "current"],
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            }),
        ),
    )

    def __init__(self, memory):
        self.memory = memory

    def catalog_for(self, context, settings=None):
        return self.catalog if self.memory.enabled(context, settings) else ()

    async def execute(self, context, arguments, settings=None):
        if not self.memory.enabled(context, settings):
            raise TaskError("role_memory_disabled",
                            "role global memory is disabled")
        if type(arguments) is not dict or set(arguments) - {
                "query", "since", "until", "limit", "scope_filter"}:
            raise TaskError("invalid_tool_arguments",
                            "global memory arguments do not match the registered tool")
        if type(arguments.get("query")) is not str:
            raise TaskError("invalid_tool_arguments",
                            "global memory query is required")
        payload = await self.memory.query(context, settings=settings, **arguments)
        return {"state": "succeeded", "effect": "none", "payload": payload}
