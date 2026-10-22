from tools import web_tools
from tools.web_result_cache import search_memo


def test_provider_specific_cache_identity(monkeypatch):
    class Provider:
        name = "camofox-service"
        backend = "auto"
        calls = []

        @property
        def search_cache_key(self):
            return self.name + ":" + self.backend

        def search(self, query, limit):
            self.calls.append(self.backend)
            return {"success": True, "data": {"web": [{"title": self.backend}]}}

    p = Provider()
    search_memo.clear()
    monkeypatch.setattr("tools.web_result_cache.cache_enabled", lambda: True)
    assert web_tools._memoized_search(p, "same", 1)["data"]["web"][0]["title"] == "auto"
    p.backend = "brave"
    assert (
        web_tools._memoized_search(p, "same", 1)["data"]["web"][0]["title"] == "brave"
    )
    web_tools._memoized_search(p, "same", 1)
    assert p.calls == ["auto", "brave"]
    search_memo.clear()
