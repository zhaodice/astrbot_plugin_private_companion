"""Opt-in group familiarity evidence for Laya's pre-reply decision.

The ledger stores only bounded counters and names captured by the host.  It
never stores message bodies.  Familiarity here means observed activity in this
role's group, not a claim about a user's real-world relationship.

Only messages that Laya cannot otherwise see need this path.  Familiarity is a
candidate source: Laya still decides whether to reply, wait, interject, or stay
silent using the current conversation, role profile, same-group memory
references, mood, and willingness.  A member explicitly listed in the existing
primary-member setting is also treated as familiar; this is a configuration
fact rather than an inference from the chat.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import re

from companion_v2.execution.models import digest, stamp
from companion_v2.foundation.refs import ScopeKind
from companion_v2.relationships.group_network import owner_key as group_owner_key
from .reply_settings import group_wakeup_settings


VERSION = "conversation.group_familiarity.v1"
PREFIX = VERSION + ":"
MAX_MEMBERS = 512
MAX_EVENTS = 512
MAX_NAMES = 8
MAX_TEXT_CHARS = 12000


def _clean_name(value):
    return " ".join(str(value or "").split())[:64]


def _number(value):
    return int(value) if type(value) is int and value >= 0 else 0


def _empty():
    return {"version": VERSION, "group": {"messages": 0, "active_days": 0,
            "last_active_day": None, "last_seen": None, "members": {}},
            "event_keys": [], "last_candidate": None}


def _valid(value):
    return (type(value) is dict and value.get("version") == VERSION
            and type(value.get("group")) is dict
            and type(value["group"].get("members")) is dict
            and type(value.get("event_keys")) is list)


def _day(at):
    return at.date().isoformat()


def _event_key(context, event_id, speaker_id, at, text):
    stable = event_id or digest([context.scope.key, speaker_id, stamp(at), str(text or "")[:500]])
    return context.scope.key + ":" + str(stable)


def _member(document, account_id, at):
    members = document["group"]["members"]
    item = members.get(account_id)
    if not type(item) is dict:
        item = {"messages": 0, "active_days": 0, "last_active_day": None,
                "last_seen": None, "names": []}
        members[account_id] = item
    return item


def _alias(item, value, kind, at):
    value = _clean_name(value)
    if not value:
        return False
    names = [entry for entry in item.get("names", ())
             if type(entry) is dict and entry.get("value") != value]
    names.insert(0, {"value": value, "kind": kind, "last_seen": stamp(at)})
    changed = names[:MAX_NAMES] != item.get("names")
    item["names"] = names[:MAX_NAMES]
    return changed


def _bounded_members(document):
    members = document["group"]["members"]
    if len(members) <= MAX_MEMBERS:
        return
    ranked = sorted(members.items(),
                    key=lambda pair: str(pair[1].get("last_seen") or ""),
                    reverse=True)
    document["group"]["members"] = dict(ranked[:MAX_MEMBERS])


def _labels(item):
    result = []
    for entry in item.get("names", ()):
        if type(entry) is dict:
            value = _clean_name(entry.get("value"))
            if value:
                result.append(value)
    return tuple(dict.fromkeys(result))


def _ascii_label(label):
    return bool(label) and label.isascii() and any(char.isalnum() for char in label)


def _contains_label(text, label):
    if not _ascii_label(label):
        return label.casefold() in text.casefold()
    pattern = r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _mentions(text, document):
    """Return unique familiar-looking members named in plain text.

    Equal labels owned by multiple accounts are excluded. A short label that is
    only part of another account's longer label is also excluded; "小明" cannot
    silently identify someone when the group also has a distinct "小明同学".
    """

    owners, labels_by_account = {}, {}
    for account_id, item in document["group"]["members"].items():
        for label in _labels(item):
            folded = label.casefold()
            owners.setdefault(folded, set()).add(account_id)
            labels_by_account.setdefault(account_id, {})[folded] = label
    matched = {}
    for account_id, labels in labels_by_account.items():
        for folded, label in labels.items():
            if len(folded) < 2 or len(owners.get(folded, ())) != 1:
                continue
            longer_elsewhere = any(
                folded in other and other != folded and account != account_id
                for other, accounts in owners.items() for account in accounts)
            if longer_elsewhere or not _contains_label(text, folded):
                continue
            item = document["group"]["members"][account_id]
            matched[account_id] = {"account_id": account_id,
                                   "display_name": _labels(item)[0],
                                   "matched_name": label}
    return tuple(matched.values())


class GroupFamiliarity:
    def __init__(self, messages):
        self.messages = messages
        self.runtime = messages.runtime

    @staticmethod
    def key(context):
        return PREFIX + group_owner_key(context)

    def _load(self, context):
        row = self.runtime.read_one("SELECT value FROM runtime_metadata WHERE key = ?", (self.key(context),))
        if row is None:
            return _empty()
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError):
            return _empty()
        return value if _valid(value) else _empty()

    def _read(self, context):
        return self._load(context)

    def laya_enabled(self, context):
        """Whether the global Laya switch and its reply-action question are on."""

        if context.scope.kind is not ScopeKind.GROUP:
            return False
        resolver = getattr(self.messages, "laya_decision_resolver", None)
        if not callable(resolver):
            return False
        try:
            service = resolver(context)
            values = service.store.raw() if service is not None else None
        except Exception:
            return False
        fields = values.get("decision_fields") if type(values) is dict else None
        return (type(values) is dict and values.get("enabled") is True
                and type(fields) is dict and fields.get("action") is True)

    def _settings(self, context):
        reply_settings = getattr(self.messages, "reply_settings", None)
        if reply_settings is None:
            return None
        try:
            snapshot = reply_settings.snapshot(
                context.actor, context, self.messages.tasks.now())
            values = {key.removeprefix("conversation.presentation."): value
                      for key, value in snapshot.values.items()
                      if key.startswith("conversation.presentation.")}
            return group_wakeup_settings(values)
        except Exception:
            return None

    def observe(self, context, *, text, speaker_id, received_at, event_id=None,
                wakeup=None):
        """Add one activity sample when Laya and the familiarity feature are on."""

        if (context.scope.kind is not ScopeKind.GROUP
                or type(speaker_id) is not str or not speaker_id
                or not isinstance(received_at, datetime)
                or received_at.tzinfo is None):
            return False
        if not self.laya_enabled(context):
            return False
        settings = wakeup or self._settings(context)
        if settings is None or settings.get("group_familiarity_enabled") is not True:
            return False
        from .reset import boundary
        reset = boundary(self.runtime, context.scope)
        if reset is not None and stamp(received_at) <= reset["reset_at"]:
            return False
        event_key = _event_key(context, event_id, speaker_id, received_at, text)
        key = self.key(context)
        with self.runtime.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = ?", (key,)).fetchone()
            try:
                document = json.loads(row["value"]) if row is not None else _empty()
            except (TypeError, ValueError):
                document = _empty()
            if not _valid(document):
                document = _empty()
            if event_key in document["event_keys"]:
                return False
            group = document["group"]
            day = _day(received_at)
            group["messages"] = _number(group.get("messages")) + 1
            group["last_seen"] = stamp(received_at)
            if group.get("last_active_day") != day:
                group["active_days"] = _number(group.get("active_days")) + 1
                group["last_active_day"] = day
            item = _member(document, speaker_id, received_at)
            item["messages"] = _number(item.get("messages")) + 1
            item["last_seen"] = stamp(received_at)
            if item.get("last_active_day") != day:
                item["active_days"] = _number(item.get("active_days")) + 1
                item["last_active_day"] = day
            document["event_keys"] = [*document["event_keys"], event_key][-MAX_EVENTS:]
            _bounded_members(document)
            connection.execute(
                """INSERT INTO runtime_metadata(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, json.dumps(document, ensure_ascii=False, sort_keys=True)))
        return True

    def record_names(self, context, *, nickname=None, card=None, at=None):
        """Attach host-captured display names to the bounded familiarity ledger."""

        if context.scope.kind is not ScopeKind.GROUP:
            return False
        if not self.laya_enabled(context):
            return False
        settings = self._settings(context)
        if settings is None or settings.get("group_familiarity_enabled") is not True:
            return False
        at = at or self.messages.tasks.now()
        account_id = str(context.actor.principal.native_account_id)
        key = self.key(context)
        with self.runtime.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = ?", (key,)).fetchone()
            try:
                document = json.loads(row["value"]) if row is not None else _empty()
            except (TypeError, ValueError):
                document = _empty()
            if not _valid(document):
                document = _empty()
            item = _member(document, account_id, at)
            changed = False
            if nickname:
                changed = _alias(item, nickname, "platform_nickname", at) or changed
            if card:
                changed = _alias(item, card, "group_card", at) or changed
            if not changed:
                return False
            connection.execute(
                """INSERT INTO runtime_metadata(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, json.dumps(document, ensure_ascii=False, sort_keys=True)))
        return True

    def _evidence(self, context, *, text, at):
        document = self._read(context)
        current = str(context.actor.principal.native_account_id)
        speaker = document["group"]["members"].get(current) or {}
        members = document["group"]["members"]
        mentioned = []
        for item in _mentions(text[:MAX_TEXT_CHARS], document):
            stats = members.get(item["account_id"]) or {}
            mentioned.append({**item,
                              "messages": _number(stats.get("messages")),
                              "active_days": _number(stats.get("active_days"))})
        return {
            "version": VERSION,
            "basis": "observed_group_activity_proxy",
            "current_speaker": {
                "account_id": current,
                "display_name": (_labels(speaker) or [""])[0],
                "messages": _number(speaker.get("messages")),
                "active_days": _number(speaker.get("active_days")),
                "last_seen": speaker.get("last_seen"),
            },
            "mentioned_contacts": mentioned,
            "group": {
                "messages": _number(document["group"].get("messages")),
                "active_days": _number(document["group"].get("active_days")),
                "member_count": len(members),
                "last_seen": document["group"].get("last_seen"),
            },
            "evaluated_at": stamp(at),
        }

    def inspect(self, context, *, text, at, settings):
        """Evaluate bounded familiarity evidence without consuming a candidate."""

        wakeup = group_wakeup_settings(settings)
        if wakeup.get("group_familiarity_enabled") is not True or not self.laya_enabled(context):
            return {"candidate_reason": "", "enabled": False}
        evidence = self._evidence(context, text=text, at=at)
        minimum_messages = wakeup["group_familiarity_min_messages"]
        minimum_days = wakeup["group_familiarity_min_active_days"]
        primary_members = set(wakeup["group_wakeup_primary_member_ids"])
        speaker = evidence["current_speaker"]
        speaker_primary = speaker["account_id"] in primary_members
        speaker_familiar = (speaker_primary or (
            speaker["messages"] >= minimum_messages
            and speaker["active_days"] >= minimum_days))
        mentioned_familiar = {
            item["account_id"] for item in evidence["mentioned_contacts"]
            if item["account_id"] in primary_members or (
                item["messages"] >= minimum_messages
                and item["active_days"] >= minimum_days)
        }
        group = evidence["group"]
        group_familiar = (group["messages"] >= wakeup["group_familiarity_group_min_messages"]
                          and group["member_count"] >= wakeup["group_familiarity_group_min_members"])
        potential_reason = ""
        if (wakeup["group_familiarity_speaker_enabled"] and speaker_familiar
                or wakeup["group_familiarity_mention_enabled"] and mentioned_familiar):
            potential_reason = "familiar_contact"
        elif wakeup["group_familiarity_group_enabled"] and group_familiar:
            potential_reason = "familiar_group"
        evidence.update({
            "enabled": True,
            "candidate_reason": "",
            "potential_reason": potential_reason,
            "sampled": False,
            "current_speaker": {**speaker, "familiar": bool(speaker_familiar),
                "configured_primary": bool(speaker_primary),
                "thresholds": {"messages": minimum_messages, "active_days": minimum_days}},
            "mentioned_contacts": [{**item, "familiar": item["account_id"] in mentioned_familiar,
                "configured_primary": item["account_id"] in primary_members}
                for item in evidence["mentioned_contacts"]],
            "group": {**group, "familiar": bool(group_familiar),
                "thresholds": {
                    "messages": wakeup["group_familiarity_group_min_messages"],
                    "members": wakeup["group_familiarity_group_min_members"],
                }},
            "probability": wakeup["group_familiarity_probability"],
            "cooldown_seconds": wakeup["group_familiarity_cooldown_seconds"],
        })
        return evidence

    def candidate(self, context, *, text, at, settings, event_id=None, inspection=None):
        """Return familiarity evidence and an optional candidate reason.

        Sampling and cooldown are applied only after a configured familiarity
        threshold passes. The persisted claim is idempotent for one event.
        """

        evidence = inspection if isinstance(inspection, dict) else self.inspect(
            context, text=text, at=at, settings=settings)
        if evidence.get("enabled") is not True:
            return {"candidate_reason": "", "enabled": False}
        reason = str(evidence.get("potential_reason") or "")
        if not reason:
            evidence["sampled"] = False
            return evidence
        wakeup = group_wakeup_settings(settings)
        speaker = evidence["current_speaker"]
        event_key = _event_key(context, event_id, speaker["account_id"], at, text)
        probability = wakeup["group_familiarity_probability"]
        if not self._sample(context, reason, probability, event_id, at, text):
            evidence.update({"candidate_reason": "", "sampled": False})
            return evidence
        evidence["sampled"] = True
        key = self.key(context)
        with self.runtime.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = ?", (key,)).fetchone()
            try:
                document = json.loads(row["value"]) if row is not None else _empty()
            except (TypeError, ValueError):
                document = _empty()
            if not _valid(document):
                document = _empty()
            last = document.get("last_candidate")
            from .reset import boundary
            reset = boundary(self.runtime, context.scope)
            if (type(last) is dict and reset is not None
                    and str(last.get("at") or "") <= reset["reset_at"]):
                last = None
            if type(last) is dict and last.get("event_key") == event_key:
                evidence["candidate_reason"] = reason
                evidence["cooldown"] = {"remaining_seconds": 0, "reused_event": True}
                return evidence
            previous = datetime.fromisoformat(str(last.get("at"))) if (
                type(last) is dict and last.get("at")) else None
            if previous is not None and previous.tzinfo is None:
                previous = previous.replace(tzinfo=at.tzinfo)
            remaining = (max(0, math.ceil(wakeup["group_familiarity_cooldown_seconds"]
                                           - (at - previous).total_seconds()))
                         if previous is not None else 0)
            if remaining > 0:
                evidence.update({"candidate_reason": "", "sampled": False,
                                 "cooldown": {"remaining_seconds": remaining,
                                              "reused_event": False}})
                return evidence
            document["last_candidate"] = {"event_key": event_key, "at": stamp(at),
                                          "reason": reason}
            connection.execute(
                """INSERT INTO runtime_metadata(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, json.dumps(document, ensure_ascii=False, sort_keys=True)))
        evidence["candidate_reason"] = reason
        evidence["cooldown"] = {"remaining_seconds": wakeup["group_familiarity_cooldown_seconds"],
                                "reused_event": False}
        return evidence

    @staticmethod
    def _sample(context, reason, probability, event_id, at, text):
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        key = [context.key, reason, event_id or [stamp(at), text]]
        return int(digest(key)[:16], 16) / (1 << 64) < probability
