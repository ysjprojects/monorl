"""Provider wiring: registry lookup, mode dispatch, BackendUnavailable without sglang."""

from __future__ import annotations

import httpx
import pytest
from fake_sglang import make_config

from monorl.adapters import PROVIDERS
from monorl.adapters.sglang import PATCHES, is_available, make_engine, register
from monorl.adapters.sglang.client import SGLangServerClient
from monorl.adapters.sglang.engine import SGLangEngine
from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.core.registry import Registry, registry
from monorl.patching import get_patchsets


def test_registry_resolves_sglang_via_lazy_provider():
    assert PROVIDERS["sglang"] == "monorl.adapters.sglang"
    assert registry.get("rollout_engine", "sglang") is make_engine


def test_register_on_fresh_registry():
    reg = Registry()
    register(reg)
    assert reg.get("rollout_engine", "sglang") is make_engine


def test_patchset_registered_at_import():
    assert any(ps is PATCHES for ps in get_patchsets("sglang"))


def test_instantiation_without_sglang_raises_actionable_error(no_sglang, sglang_config):
    assert not is_available()
    factory = registry.get("rollout_engine", "sglang")
    with pytest.raises(BackendUnavailable) as info:
        factory(sglang_config)
    message = str(info.value)
    assert "monorl[sglang]" in message and "rollout.extra.mode='server'" in message and "backend='hf'" in message


def test_factory_dispatches_modes(fake_sglang, run_config_dict):
    assert is_available()
    colocated = make_engine(make_config(run_config_dict), replica=2)
    assert isinstance(colocated, SGLangEngine) and colocated.replica == 2
    colocated.shutdown()
    server = make_engine(
        make_config(run_config_dict, extra={"mode": "server", "url": "http://s"}),
        replica=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    assert isinstance(server, SGLangServerClient) and server.replica == 1
    server.shutdown()
    with pytest.raises(ConfigError, match="rollout.extra.mode"):
        make_engine(make_config(run_config_dict, extra={"mode": "remote"}))


def test_server_mode_does_not_need_sglang(no_sglang, run_config_dict):
    client = make_engine(
        make_config(run_config_dict, extra={"mode": "server", "url": "http://s"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    assert isinstance(client, SGLangServerClient)
    client.shutdown()
