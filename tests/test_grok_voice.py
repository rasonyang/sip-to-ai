"""Unit tests for GrokVoiceClient and Grok config wiring."""

import os
from unittest.mock import patch

import pytest


class TestGrokConfig:
    """Tests for Grok-related fields in AIConfig."""

    def test_grok_vendor_is_accepted(self) -> None:
        """AI_VENDOR=grok should be accepted (not coerced to mock)."""
        with patch.dict(os.environ, {"AI_VENDOR": "grok", "XAI_API_KEY": "x"}, clear=False):
            # Re-import to pick up the new env
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.vendor == "grok"

    def test_grok_defaults(self) -> None:
        """Grok config defaults match the spec."""
        with patch.dict(os.environ, {"AI_VENDOR": "grok", "XAI_API_KEY": "x"}, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.grok_model == "grok-voice-think-fast-1.0"
            assert cfg_module.config.ai.grok_voice == "eve"
            assert cfg_module.config.ai.grok_ws_endpoint == "wss://api.x.ai/v1/realtime"

    def test_grok_env_overrides(self) -> None:
        """GROK_MODEL / GROK_VOICE / GROK_WS_ENDPOINT env vars override defaults."""
        env = {
            "AI_VENDOR": "grok",
            "XAI_API_KEY": "x",
            "GROK_MODEL": "grok-voice-fast-1.0",
            "GROK_VOICE": "rex",
            "GROK_WS_ENDPOINT": "wss://example.invalid/realtime",
        }
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.grok_model == "grok-voice-fast-1.0"
            assert cfg_module.config.ai.grok_voice == "rex"
            assert cfg_module.config.ai.grok_ws_endpoint == "wss://example.invalid/realtime"


class TestGrokConstructor:
    """Tests for GrokVoiceClient.__init__."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor raises ValueError if no api_key arg and XAI_API_KEY not set."""
        from app.ai.grok_voice import GrokVoiceClient

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Grok API key"):
            GrokVoiceClient(api_key=None)

    def test_constructor_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor reads XAI_API_KEY from env when api_key arg is None."""
        from app.ai.grok_voice import GrokVoiceClient

        monkeypatch.setenv("XAI_API_KEY", "env-key")
        client = GrokVoiceClient(api_key=None)
        assert client._api_key == "env-key"

    def test_constructor_defaults(self) -> None:
        """Default model, voice, endpoint match spec."""
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        assert client._model == "grok-voice-think-fast-1.0"
        assert client._voice == "eve"
        assert client._ws_url == "wss://api.x.ai/v1/realtime"
        assert client._sample_rate == 8000
        assert client._frame_ms == 20
