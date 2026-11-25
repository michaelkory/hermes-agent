"""Native history import preserves metadata without reviving runtime ownership."""

from __future__ import annotations

import hashlib
import json

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def stores(tmp_path):
    source = SessionDB(db_path=tmp_path / "source.db")
    target = SessionDB(db_path=tmp_path / "target.db")
    yield source, target
    source.close()
    target.close()


@pytest.mark.parametrize("pin", [
    # Ordered full declarations, noncanonical whitespace, escaping and numeric spelling.
    '{ "version" : "synthetic-code", "tools" : [\n'
    ' {"type":"function","function":{"name":"zeta","description":"雪\\u2603",'
    '"parameters":{"type":"object","properties":{"n":{"default":1e+00}}}}},\n'
    ' {"type":"function","function":{"name":"alpha","parameters":{"type":"object"}}} ] }\n',
    '[ "zeta", "alpha", "\\u2603" ]\n',
    "",
    None,
], ids=["full-definition", "legacy-names", "empty", "null"])
def test_export_import_preserves_exact_tool_pin(stores, pin):
    source, target = stores
    source.create_session("history", source="cli")
    # Historical fixtures can carry spellings the current JSON-producing setter cannot emit.
    source._conn.execute("UPDATE sessions SET tool_names = ? WHERE id = ?", (
        source._store_system_prompt(source._conn, pin), "history",
    ))
    source._conn.commit()
    payload = json.loads(json.dumps(source.export_session("history", include_inactive=True)))
    assert payload["tool_names"] == pin

    result = target.import_sessions([payload])
    assert result["ok"] is True and result["imported"] == 1
    assert target.export_session("history", include_inactive=True)["tool_names"] == pin
    stored = target._conn.execute("SELECT tool_names FROM sessions WHERE id = 'history'").fetchone()[0]
    if pin is None:
        assert stored is None
    else:
        assert stored == hashlib.sha256(pin.encode("utf-8")).hexdigest()
        assert target._conn.execute("SELECT prompt FROM system_prompts WHERE hash = ?", (stored,)).fetchone()[0] == pin


@pytest.mark.parametrize("title", ["Historical title", None])
@pytest.mark.parametrize("title_source,created_source", [
    ("llm", "slack"), ("derived", "tui"), ("user", "cli"),
    (None, None), ("", ""), ("future-author", "future-surface"),
])
def test_export_import_preserves_nullable_provenance(stores, title, title_source, created_source):
    source, target = stores
    source.create_session("provenance", source="unknown")
    source._conn.execute(
        "UPDATE sessions SET title = ?, title_source = ?, created_source = ? WHERE id = ?",
        (title, title_source, created_source, "provenance"),
    )
    source._conn.commit()
    payload = json.loads(json.dumps(source.export_session("provenance", include_inactive=True)))
    assert target.import_sessions([payload])["imported"] == 1
    restored = target.export_session("provenance", include_inactive=True)
    assert {key: restored[key] for key in ("title", "title_source", "created_source", "source")} == {
        "title": title, "title_source": title_source, "created_source": created_source, "source": "unknown",
    }


def test_old_exports_do_not_infer_missing_metadata(stores):
    _, target = stores
    assert target.import_sessions([{"id": "legacy", "source": "cli", "title": "Old title"}])["imported"] == 1
    restored = target.export_session("legacy")
    assert all(restored[field] is None for field in ("tool_names", "title_source", "created_source"))
