"""Unit tests for GrokVoiceClient and Grok config wiring."""

import asyncio
import base64
import json
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


class _FakeWebSocket:
    """Minimal stand-in for websockets.WebSocketClientProtocol used in tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    async def recv(self) -> str:  # pragma: no cover - overridden per test as needed
        await asyncio.Event().wait()
        return ""


class TestGrokUplink:
    """Tests for send_pcm16_8k."""

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_sends_input_audio_buffer_append(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]
        client._connected = True

        # 320 bytes of PCM16 silence = 160 mu-law bytes
        await client.send_pcm16_8k(b"\x00" * 320)

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "input_audio_buffer.append"
        decoded = base64.b64decode(msg["audio"])
        assert len(decoded) == 160  # mu-law @ 8kHz, 20ms = 160 bytes

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_validates_frame_size(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._ws = _FakeWebSocket()  # type: ignore[assignment]
        client._connected = True

        with pytest.raises(ValueError, match="320"):
            await client.send_pcm16_8k(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_raises_when_not_connected(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        with pytest.raises(ConnectionError):
            await client.send_pcm16_8k(b"\x00" * 320)
