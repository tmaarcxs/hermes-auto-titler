"""Tests for plugin unload thread cleanup and retry backoff on unavailable auxiliary client."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from hermes_auto_titler import register
from hermes_auto_titler.titler import AutoTitler, _retry_class


def test_retry_class_marks_auxiliary_client_unavailable_blocked():
    assert _retry_class("AuxiliaryClientUnavailable: No LLM provider configured") == "blocked"
    assert _retry_class("UnscopedSecretError: denied") == "blocked"
    assert _retry_class("RuntimeError: transient timeout") == "temporary"


def test_register_registers_on_unload_and_stops_retry_thread(monkeypatch):
    unload_callbacks = []

    class DummyCtx:
        def register_hook(self, *args, **kwargs):
            pass

        def register_command(self, *args, **kwargs):
            pass

        def on_unload(self, cb):
            unload_callbacks.append(cb)

        def get_config(self, key, default=None):
            return default

    monkeypatch.setattr("hermes_auto_titler.migrate_legacy_state", lambda: None)
    monkeypatch.setattr(AutoTitler, "restore_state", lambda self: None)
    monkeypatch.setattr(AutoTitler, "_requeue_untitled_sessions", lambda self: None)

    before = {t.ident for t in threading.enumerate() if t.name == "autotitler-retry"}
    register(DummyCtx())
    assert len(unload_callbacks) == 1

    # Give the daemon thread a moment to enter wait()
    time.sleep(0.05)
    during = [t for t in threading.enumerate() if t.name == "autotitler-retry" and t.ident not in before]
    assert len(during) == 1

    unload_callbacks[0]()
    after = [t for t in threading.enumerate() if t.name == "autotitler-retry" and t.ident not in before]
    assert after == []
