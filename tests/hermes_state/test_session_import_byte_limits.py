"""Per-store import byte allowances retain the native safety boundary."""

from __future__ import annotations

import json

import pytest

from hermes_state import SessionDB

MIB = 1024 * 1024


def test_only_explicit_store_accepts_a_session_above_the_default(tmp_path):
    payload = {
        "id": "large-original",
        "system_prompt": "Historical instructions stay historical.",
        "messages": [{"role": "user", "content": "x" * (5 * MIB)}],
    }
    default = SessionDB(db_path=tmp_path / "default.db")
    try:
        rejected = default.import_sessions([payload])
        assert rejected["ok"] is False
        assert rejected["imported"] == 0
        assert "import size limit" in rejected["errors"][0]["error"]
        assert default.get_session(payload["id"]) is None
    finally:
        default.close()

    configured = SessionDB(
        db_path=tmp_path / "configured.db", import_max_session_bytes=25 * MIB,
    )
    try:
        restored = configured.import_sessions([payload])
        assert restored["ok"] is True
        assert restored["imported_ids"] == [payload["id"]]
        exported = configured.export_session(payload["id"], include_inactive=True)
        assert exported["id"] == payload["id"]
        assert exported["system_prompt"] == payload["system_prompt"]
        assert exported["messages"][0]["content"] == payload["messages"][0]["content"]
    finally:
        configured.close()

    ordinary = SessionDB(db_path=tmp_path / "ordinary.db")
    try:
        assert ordinary.import_sessions([payload])["ok"] is False
        assert ordinary.get_session(payload["id"]) is None
    finally:
        ordinary.close()


@pytest.mark.parametrize("invalid", [0, -1, True, False, "26214400", 1.5, 25 * MIB + 1])
def test_allowance_is_positive_integer_bounded_by_transaction_ceiling(tmp_path, invalid):
    path = tmp_path / "invalid.db"
    with pytest.raises(ValueError, match="import_max_session_bytes"):
        store = SessionDB(db_path=path, import_max_session_bytes=invalid)
        store.close()
    assert not path.exists(), "invalid policy must fail before opening a database"


def test_configured_limit_counts_utf8_bytes_at_the_exact_boundary(tmp_path):
    payload = {"id": "unicode", "messages": [{"role": "user", "content": "雪" * 100}]}
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for limit, accepted in ((size, True), (size - 1, False)):
        store = SessionDB(db_path=tmp_path / f"boundary-{limit}.db", import_max_session_bytes=limit)
        try:
            result = store.import_sessions([payload])
            assert result["ok"] is accepted
            assert (store.get_session(payload["id"]) is not None) is accepted
        finally:
            store.close()


def test_configured_store_still_rejects_above_its_session_allowance(tmp_path):
    store = SessionDB(db_path=tmp_path / "oversize.db", import_max_session_bytes=25 * MIB)
    try:
        result = store.import_sessions([
            {"id": "too-large", "messages": [{"role": "user", "content": "x" * (25 * MIB)}]},
        ])
        assert result["ok"] is False
        assert "session exceeds the import size limit" in result["errors"][0]["error"]
        assert store.get_session("too-large") is None
    finally:
        store.close()


def test_larger_session_allowance_does_not_widen_the_transaction(tmp_path):
    store = SessionDB(db_path=tmp_path / "batch.db", import_max_session_bytes=25 * MIB)
    try:
        payload = [
            {"id": sid, "messages": [{"role": "user", "content": "x" * (13 * MIB)}]}
            for sid in ("first", "second")
        ]
        result = store.import_sessions(payload)
        assert result["ok"] is False
        assert result["imported"] == 0
        assert "import exceeds the total size limit" in result["errors"][0]["error"]
        assert all(store.get_session(item["id"]) is None for item in payload)
    finally:
        store.close()


def test_large_native_backup_retains_all_stored_generations(tmp_path):
    source = SessionDB(db_path=tmp_path / "source.db")
    target = SessionDB(db_path=tmp_path / "target.db", import_max_session_bytes=25 * MIB)
    sid = "original-history"
    try:
        source.create_session(sid, source="slack", system_prompt="Original historical prompt.")
        archived = source.append_message(sid, "user", "An archived generation", platform_message_id="old-turn")
        compacted = source.append_message(sid, "assistant", "A compacted generation")
        source.append_message(sid, "user", "x" * (5 * MIB), message_uid="original-user")
        source.append_message(sid, "assistant", "Using a tool", tool_calls=[{
            "id": "original-call", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"historical-path"}'},
        }], reasoning="Original reasoning", message_uid="original-assistant")
        source.append_message(sid, "tool", "Original tool result", tool_name="read_file",
                              tool_call_id="original-call", message_uid="original-tool")
        # Fixture-only generation states model rows retained after rewind and compaction.
        source._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (archived,))
        source._conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (compacted,))
        source._conn.execute(
            "UPDATE sessions SET archived = 1, auto_archived = 1, pinned = 1, hidden = 1 WHERE id = ?", (sid,),
        )
        source._conn.commit()
        payload = json.loads(json.dumps(source.export_session(sid, include_inactive=True)))
        assert len(payload["messages"]) == 5

        result = target.import_sessions([payload])
        assert result["ok"] is True
        assert result["imported_ids"] == [sid]
        restored = target.export_session(sid, include_inactive=True)
        assert restored["system_prompt"] == payload["system_prompt"]
        for flag in ("archived", "auto_archived", "pinned", "hidden"):
            assert restored[flag] == payload[flag]
        # SQLite-local primary keys may differ. Native batch insertion also fills an absent
        # tool-result link from the original assistant call's durable UID, not a new identity.
        tool_uids = payload["messages"][3]["tool_call_uids"]
        if isinstance(tool_uids, str):
            tool_uids = json.loads(tool_uids)
        assert set(tool_uids) == {"original-call"}
        def without_row_ids(rows):
            return [{k: v for k, v in row.items() if k != "id"} for row in rows]
        for index, (original, copy) in enumerate(zip(payload["messages"], restored["messages"], strict=True)):
            for key, value in original.items():
                if key != "id":
                    expected = tool_uids["original-call"] if (index, key) == (4, "tool_call_uid") else value
                    assert copy[key] == expected, (index, key)
        repeat = target.import_sessions([payload])
        assert repeat["imported"] == 0
        assert repeat["skipped_ids"] == [sid]
        assert without_row_ids(target.get_messages(sid, include_inactive=True)) == without_row_ids(restored["messages"])
    finally:
        source.close()
        target.close()
