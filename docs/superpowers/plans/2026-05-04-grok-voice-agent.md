# Grok Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add xAI Grok Voice as a fourth realtime AI vendor in SIP-to-AI, enabling SIP calls to be answered by Grok's voice models alongside the existing OpenAI, Deepgram, and Gemini integrations.

**Architecture:** New standalone `app/ai/grok_voice.py` (`GrokVoiceClient`) that implements the existing `AiDuplexClient` Protocol, patterned line-for-line on `app/ai/openai_realtime.py`. Wire it into `app/config.py` and `app/main.py`'s vendor factory. The Grok Voice realtime API mirrors OpenAI Realtime almost 1:1 (same event names, same audio buffer model, native G.711 μ-law @ 8kHz), so no changes are required to `app/bridge/`, `app/sip_async/`, or `app/ai/duplex_base.py`. No resampling: SIP G.711 μ-law @ 8kHz ↔ PCM16 (within `AudioAdapter`) ↔ PCM16 → μ-law @ 8kHz (within `GrokVoiceClient`) ↔ Grok WebSocket.

**Tech Stack:** Python 3.12, asyncio, `websockets` 15+, `structlog`, `numpy`, `pytest` + `pytest-asyncio`, `uv`. No new dependencies introduced.

**Spec:** `docs/superpowers/specs/2026-05-04-grok-voice-agent-design.md`

**Conventions used in this plan:**
- All commands assume CWD = repo root (`/Users/rason/workspaces/sip-to-ai` or wherever the worktree was made).
- All `pytest` invocations use `uv run pytest`.
- All commit messages follow the existing repo style: lowercase, sentence-case, no scope prefix (e.g., `add grok voice client`, not `feat(ai): ...`).
- The existing `OpenAIRealtimeClient` is the structural template throughout. Where a step says "mirror OpenAI", it means same method signatures, same handshake sequencing, same logging cadence, same buffer sizes — only the URL, header form, model/voice fields, and `audio_format` schema differ.

---

## File Structure

**New files (2):**

| Path | Responsibility | LOC est. |
|---|---|---|
| `app/ai/grok_voice.py` | `GrokVoiceClient(AiDuplexBase)` — WebSocket client for `wss://api.x.ai/v1/realtime`, G.711 μ-law @ 8kHz both directions, full `AiDuplexClient` Protocol | ~420 |
| `tests/test_grok_voice.py` | Unit tests for `GrokVoiceClient`: connect handshake, session config, audio uplink/downlink, frame validation, error mapping, disconnect, greeting, missing API key | ~350 |

**Modified files (4):**

| Path | Change | Why |
|---|---|---|
| `app/config.py` | Extend `AIConfig.vendor` Literal to include `"grok"`; add `grok_api_key`, `grok_model`, `grok_voice`, `grok_ws_endpoint` fields; add env-var loading; update validation list | Surface Grok as a selectable vendor |
| `app/main.py` | Add `elif vendor == "grok":` branch to `create_ai_client()` | Vendor factory wiring |
| `.env.example` | Add `XAI_API_KEY`, `GROK_MODEL`, `GROK_VOICE`, `GROK_WS_ENDPOINT` entries; update `AI_VENDOR` comment | Operator documentation |
| `README.md` | Add "Grok Voice Agent Setup" section | Operator documentation |

**Untouched (do not modify):**
- `app/bridge/`, `app/sip_async/`, `app/utils/` — vendor-agnostic.
- `app/ai/duplex_base.py` — Protocol already matches.
- `app/ai/openai_realtime.py`, `deepgram_agent.py`, `gemini_live.py` — additive change only.

---

## Pre-flight

### Task 0: Verify baseline

**Files:** none (read-only)

- [ ] **Step 0.1: Confirm clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean` (or only the spec/plan docs untracked).

- [ ] **Step 0.2: Confirm test suite is green before any changes**

Run: `uv run pytest -q`
Expected: All existing tests pass. Note the count for later comparison.

- [ ] **Step 0.3: Confirm Python and uv versions**

Run: `uv run python --version && uv --version`
Expected: Python 3.12+ and any uv version.

- [ ] **Step 0.4: Read the spec one more time**

Open `docs/superpowers/specs/2026-05-04-grok-voice-agent-design.md`. The architecture section is authoritative; every choice in this plan derives from it.

---

## Phase 1: Configuration plumbing

### Task 1: Extend `AIConfig` with Grok fields (TDD)

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_grok_voice.py` (new file — first test goes here for config wiring; further tests added in later tasks)

- [ ] **Step 1.1: Create `tests/test_grok_voice.py` with one failing test for the config**

Create file `tests/test_grok_voice.py` with the following content:

```python
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
```

- [ ] **Step 1.2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_grok_voice.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'grok_model'` (or vendor coerced to `"mock"` because `"grok"` not in allowed list).

- [ ] **Step 1.3: Edit `app/config.py` — extend the vendor Literal and add Grok fields**

In `app/config.py`, change `AIConfig.vendor` from:

```python
    vendor: Literal["openai", "deepgram", "gemini"] = "openai"
```

to:

```python
    vendor: Literal["openai", "deepgram", "gemini", "grok"] = "openai"
```

Then, immediately after the Gemini block (after the line `gemini_voice: str = "Puck"  # Voice: Puck, Charon, Kore, Fenrir, Aoede`), insert:

```python

    # xAI Grok Voice Configuration
    grok_api_key: str = ""
    grok_ws_endpoint: str = "wss://api.x.ai/v1/realtime"
    grok_model: str = "grok-voice-think-fast-1.0"
    grok_voice: str = "eve"  # Built-in voices: eve (default), ara, leo, rex, sal
```

- [ ] **Step 1.4: Update `Config.__init__` in `app/config.py`**

Find the line:

```python
        if ai_vendor not in ["mock", "openai", "deepgram", "gemini"]:
```

Change to:

```python
        if ai_vendor not in ["mock", "openai", "deepgram", "gemini", "grok"]:
```

Find the `self.ai = AIConfig(...)` block. Immediately after the line `gemini_voice=os.getenv("GEMINI_VOICE", "Puck"),` (and inside the `AIConfig(...)` call), add:

```python
            grok_api_key=os.getenv("XAI_API_KEY", ""),
            grok_ws_endpoint=os.getenv("GROK_WS_ENDPOINT", "wss://api.x.ai/v1/realtime"),
            grok_model=os.getenv("GROK_MODEL", "grok-voice-think-fast-1.0"),
            grok_voice=os.getenv("GROK_VOICE", "eve"),
```

- [ ] **Step 1.5: Run the config tests — they should now pass**

Run: `uv run pytest tests/test_grok_voice.py -v`
Expected: All 3 tests in `TestGrokConfig` PASS.

- [ ] **Step 1.6: Run the full test suite — nothing else should regress**

Run: `uv run pytest -q`
Expected: All previously-green tests still pass; new 3 tests pass.

- [ ] **Step 1.7: Commit**

```bash
git add app/config.py tests/test_grok_voice.py
git commit -m "add grok vendor config fields"
```

---

## Phase 2: `GrokVoiceClient` — TDD, one capability at a time

The client is built incrementally. Each task adds one capability with its own test, and we keep the file compilable and importable at every step. The structural template is `app/ai/openai_realtime.py` — keep it open in another window.

### Task 2: Skeleton — constructor + module imports

**Files:**
- Create: `app/ai/grok_voice.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 2.1: Write failing test for constructor + missing-key error**

Append to `tests/test_grok_voice.py`:

```python


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
```

- [ ] **Step 2.2: Run — expect import failure**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokConstructor -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ai.grok_voice'`.

- [ ] **Step 2.3: Create `app/ai/grok_voice.py` skeleton**

Create file `app/ai/grok_voice.py` with the following content:

```python
"""xAI Grok Voice realtime API adapter.

Patterned on OpenAIRealtimeClient. The Grok Voice realtime protocol mirrors
OpenAI Realtime almost 1:1 (event names, handshake, audio buffer model), and
natively supports G.711 mu-law @ 8kHz so no resampling is needed.

Audio Flow:
- Input:  PCM16 @ 8kHz -> G.711 mu-law @ 8kHz -> Grok
- Output: Grok -> G.711 mu-law @ 8kHz -> PCM16 @ 8kHz

Endpoint: wss://api.x.ai/v1/realtime?model=<model>
Auth:     Authorization: Bearer <XAI_API_KEY>
"""

import asyncio
import base64
import json
import os
import time
from typing import AsyncIterator, Dict, Optional

import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType
from app.utils.codec import Codec


class GrokVoiceClient(AiDuplexBase):
    """xAI Grok Voice realtime API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-voice-think-fast-1.0",
        voice: str = "eve",
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None,
        ws_endpoint: str = "wss://api.x.ai/v1/realtime",
    ) -> None:
        """Initialize Grok Voice client.

        Args:
            api_key: xAI API key (falls back to XAI_API_KEY env var).
            model: Grok voice model (e.g. grok-voice-think-fast-1.0).
            voice: Built-in voice (eve, ara, leo, rex, sal) or custom 8-char id.
            instructions: System prompt for the agent.
            greeting: Optional greeting played when the call connects.
            ws_endpoint: WebSocket endpoint (overridable for testing).
        """
        self._sip_sample_rate = 8000
        self._grok_sample_rate = 8000
        self._frame_ms = 20
        self._sample_rate = self._sip_sample_rate

        super().__init__(self._sample_rate, self._frame_ms)

        self._api_key = api_key or os.getenv("XAI_API_KEY")
        if not self._api_key:
            raise ValueError("Grok API key not provided")

        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._greeting = greeting
        self._ws: Optional[WebSocketClientProtocol] = None
        self._ws_url = ws_endpoint

        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        self._connected = False
        self._stop_event = asyncio.Event()
        self._session_created_event = asyncio.Event()
        self._session_updated_event = asyncio.Event()
        self._message_handler_task: Optional[asyncio.Task[None]] = None

        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        self._logger = structlog.get_logger(__name__)
```

- [ ] **Step 2.4: Run constructor tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokConstructor -v`
Expected: All 3 PASS.

- [ ] **Step 2.5: Run mypy on the new file**

Run: `uv run mypy app/ai/grok_voice.py`
Expected: No errors. (If `Codec` import is unused, leave it — Tasks 3 and 4 use it.)

- [ ] **Step 2.6: Commit**

```bash
git add app/ai/grok_voice.py tests/test_grok_voice.py
git commit -m "add grok voice client skeleton"
```

---

### Task 3: `send_pcm16_8k` — uplink encoding

**Files:**
- Modify: `app/ai/grok_voice.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 3.1: Write failing tests for uplink**

Append to `tests/test_grok_voice.py`:

```python


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
```

Add `import asyncio`, `import base64`, `import json` to the top of `tests/test_grok_voice.py` if not already present.

- [ ] **Step 3.2: Run — expect AttributeError (no send_pcm16_8k yet)**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokUplink -v`
Expected: FAIL — `AttributeError: 'GrokVoiceClient' object has no attribute 'send_pcm16_8k'`.

- [ ] **Step 3.3: Implement `send_pcm16_8k` in `app/ai/grok_voice.py`**

Append the following method to the `GrokVoiceClient` class:

```python
    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to Grok.

        Converts PCM16 -> G.711 mu-law @ 8kHz before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes).
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

        g711_ulaw = Codec.pcm16_to_ulaw(frame_20ms)

        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(g711_ulaw).decode("utf-8"),
        }
        await self._ws.send(json.dumps(message))

        self._audio_frames_sent += 1
        if self._audio_frames_sent % 50 == 0:
            self._logger.info("📤 Sent audio frames to Grok", frames=self._audio_frames_sent)
```

- [ ] **Step 3.4: Run uplink tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokUplink -v`
Expected: All 3 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add app/ai/grok_voice.py tests/test_grok_voice.py
git commit -m "add grok uplink encoding"
```

---

### Task 4: `_process_message` — downlink decoding + event mapping

**Files:**
- Modify: `app/ai/grok_voice.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 4.1: Write failing tests for downlink + events**

Append to `tests/test_grok_voice.py`:

```python


class TestGrokMessageProcessing:
    """Tests for _process_message → event/audio queue dispatch."""

    @pytest.mark.asyncio
    async def test_session_created_sets_event_and_emits_connected(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "session.created", "session": {"id": "s1"}})

        assert client._session_created_event.is_set()
        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.CONNECTED

    @pytest.mark.asyncio
    async def test_session_updated_sets_event_and_emits(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "session.updated", "session": {}})

        assert client._session_updated_event.is_set()
        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.SESSION_UPDATED

    @pytest.mark.asyncio
    async def test_audio_delta_decoded_to_pcm16(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        # 160 bytes mu-law silence (0x7F is ~zero in mu-law)
        ulaw = bytes([0x7F] * 160)
        await client._process_message({
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(ulaw).decode("utf-8"),
        })

        chunk = client._audio_queue.get_nowait()
        assert len(chunk) == 320  # PCM16 = 2x mu-law

    @pytest.mark.asyncio
    async def test_speech_started_emits_partial_event(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "input_audio_buffer.speech_started"})

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.TRANSCRIPT_PARTIAL
        assert evt.data == {"event": "speech_started"}

    @pytest.mark.asyncio
    async def test_transcription_completed_emits_final(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello world",
        })

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.TRANSCRIPT_FINAL
        assert evt.data == {"text": "hello world"}

    @pytest.mark.asyncio
    async def test_error_event_emitted(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({
            "type": "error",
            "error": {"message": "bad thing"},
        })

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.ERROR
        assert evt.error == "bad thing"

    @pytest.mark.asyncio
    async def test_unknown_event_does_not_raise(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        # Should be silently logged, no exception, no queue entries
        await client._process_message({"type": "some.unknown.event"})

        assert client._event_queue.empty()
        assert client._audio_queue.empty()
```

- [ ] **Step 4.2: Run — expect AttributeError**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokMessageProcessing -v`
Expected: FAIL — `AttributeError: 'GrokVoiceClient' object has no attribute '_process_message'`.

- [ ] **Step 4.3: Implement `_process_message`**

Append to the `GrokVoiceClient` class in `app/ai/grok_voice.py`:

```python
    async def _process_message(self, data: Dict) -> None:
        """Dispatch a single Grok server event."""
        msg_type = data.get("type")
        self._logger.debug("Received Grok event", msg_type=msg_type)

        if msg_type == "session.created":
            self._session_created_event.set()
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.CONNECTED,
                    data=data.get("session"),
                    timestamp=time.time(),
                )
            )

        elif msg_type == "session.updated":
            self._session_updated_event.set()
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.SESSION_UPDATED,
                    data=data.get("session", {}),
                    timestamp=time.time(),
                )
            )

        elif msg_type == "input_audio_buffer.speech_started":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_started"},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "input_audio_buffer.speech_stopped":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_stopped"},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = data.get("transcript")
            self._logger.info("✅ User transcript", text=transcript)
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_FINAL,
                    data={"text": transcript},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "response.output_audio.delta":
            audio_b64 = data.get("delta")
            if audio_b64:
                ulaw = base64.b64decode(audio_b64)
                pcm16 = Codec.ulaw_to_pcm16(ulaw)
                await self._audio_queue.put(pcm16)
                self._audio_chunks_received += 1
                if self._audio_chunks_received % 10 == 0:
                    self._logger.info(
                        "📢 Received audio chunks",
                        chunks=self._audio_chunks_received,
                        ulaw_bytes=len(ulaw),
                        pcm16_bytes=len(pcm16),
                    )

        elif msg_type in ("response.output_audio_transcript.delta", "response.output_audio_transcript.done"):
            self._logger.info("🤖 AI transcript", **{k: data.get(k) for k in ("delta", "transcript") if k in data})

        elif msg_type == "response.done":
            self._logger.debug("Grok response.done")

        elif msg_type == "error":
            err = data.get("error", {})
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.ERROR,
                    error=err.get("message"),
                    timestamp=time.time(),
                )
            )
            self._logger.error("Grok error", error=err)

        else:
            self._logger.debug("Unhandled Grok event", msg_type=msg_type, data=data)
```

- [ ] **Step 4.4: Run downlink tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokMessageProcessing -v`
Expected: All 7 PASS.

- [ ] **Step 4.5: Run full test file — sanity**

Run: `uv run pytest tests/test_grok_voice.py -v`
Expected: All previous + new tests pass. Existing OpenAI/Deepgram/etc. tests unaffected.

- [ ] **Step 4.6: Commit**

```bash
git add app/ai/grok_voice.py tests/test_grok_voice.py
git commit -m "add grok message processing and downlink decoding"
```

---

### Task 5: `_configure_session` and `_send_greeting`

**Files:**
- Modify: `app/ai/grok_voice.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 5.1: Write failing tests for session config + greeting**

Append to `tests/test_grok_voice.py`:

```python


class TestGrokSessionConfig:
    """Tests for _configure_session and _send_greeting."""

    @pytest.mark.asyncio
    async def test_configure_session_payload(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(
            api_key="k",
            model="grok-voice-think-fast-1.0",
            voice="eve",
            instructions="be terse",
        )
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._configure_session()

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "session.update"
        sess = msg["session"]
        assert sess["model"] == "grok-voice-think-fast-1.0"
        assert sess["voice"] == "eve"
        assert sess["system_prompt"] == "be terse"
        assert sess["audio_format"]["input"]["type"] == "mulaw"
        assert sess["audio_format"]["input"]["sample_rate"] == 8000
        assert sess["audio_format"]["output"]["type"] == "mulaw"
        assert sess["audio_format"]["output"]["sample_rate"] == 8000
        assert sess["turn_detection"]["type"] == "server_vad"

    @pytest.mark.asyncio
    async def test_send_greeting_sends_response_create(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k", greeting="hi there")
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._send_greeting()

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "response.create"
        assert msg["response"]["instructions"] == "hi there"
        assert msg["response"]["metadata"]["response_purpose"] == "greeting"

    @pytest.mark.asyncio
    async def test_send_greeting_noop_when_unset(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k", greeting=None)
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._send_greeting()

        assert ws.sent == []
```

- [ ] **Step 5.2: Run — expect failures**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokSessionConfig -v`
Expected: FAIL — methods don't exist.

- [ ] **Step 5.3: Implement `_configure_session` and `_send_greeting`**

Append to the `GrokVoiceClient` class in `app/ai/grok_voice.py`:

```python
    async def _configure_session(self) -> None:
        """Send the initial session.update payload.

        Schema note: the audio_format field shape used here ({"input": {"type": "mulaw",
        "sample_rate": 8000}, "output": {...}}) follows the doc summary. If the live
        API rejects this shape, the design intent (mu-law @ 8kHz both directions,
        server VAD) is what matters — adjust the key path to match the API error.
        """
        if not self._ws:
            return

        config = {
            "type": "session.update",
            "session": {
                "model": self._model,
                "voice": self._voice,
                "system_prompt": self._instructions,
                "audio_format": {
                    "input": {"type": "mulaw", "sample_rate": 8000},
                    "output": {"type": "mulaw", "sample_rate": 8000},
                },
                "turn_detection": {"type": "server_vad"},
            },
        }
        self._logger.info(
            "Configuring Grok session",
            model=self._model,
            voice=self._voice,
            has_greeting=self._greeting is not None,
            instructions_length=len(self._instructions),
        )
        await self._ws.send(json.dumps(config))

    async def _send_greeting(self) -> None:
        """Send greeting response.create after session.updated."""
        if not self._ws or not self._greeting:
            return

        message = {
            "type": "response.create",
            "response": {
                "instructions": self._greeting,
                "conversation": "none",
                "output_modalities": ["audio"],
                "metadata": {"response_purpose": "greeting"},
            },
        }
        await self._ws.send(json.dumps(message))
        self._logger.info("Greeting request sent", greeting_preview=self._greeting[:50])
```

- [ ] **Step 5.4: Run tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokSessionConfig -v`
Expected: All 3 PASS.

- [ ] **Step 5.5: Commit**

```bash
git add app/ai/grok_voice.py tests/test_grok_voice.py
git commit -m "add grok session config and greeting"
```

---

### Task 6: `connect`, `close`, `_message_handler`, `events`, `receive_chunks`, `update_session`, `ping`, `reconnect`

This task brings the client up to the full `AiDuplexClient` Protocol surface. Each method is small and largely identical to OpenAI's; we group them in one task because they're tightly coupled (lifecycle).

**Files:**
- Modify: `app/ai/grok_voice.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 6.1: Write failing tests for the full lifecycle**

Append to `tests/test_grok_voice.py`:

```python


class _RecvControlledFakeWS(_FakeWebSocket):
    """Fake WS where recv() yields scripted messages, then blocks."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        for m in messages:
            self._queue.put_nowait(m)
        self._closed_event = asyncio.Event()

    async def recv(self) -> str:
        if not self._queue.empty():
            return await self._queue.get()
        # After scripted messages exhausted, block until close
        await self._closed_event.wait()
        raise websockets.exceptions.ConnectionClosed(None, None)  # type: ignore[arg-type]

    async def close(self) -> None:
        await super().close()
        self._closed_event.set()


class TestGrokLifecycle:
    """Tests for connect/close/message-handler integration."""

    @pytest.mark.asyncio
    async def test_connect_completes_after_session_created_and_updated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ai import grok_voice
        from app.ai.grok_voice import GrokVoiceClient

        scripted = [
            json.dumps({"type": "session.created", "session": {"id": "s"}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ]
        fake_ws = _RecvControlledFakeWS(scripted)

        async def fake_connect(*args: object, **kwargs: object) -> _RecvControlledFakeWS:
            return fake_ws

        monkeypatch.setattr(grok_voice.websockets, "connect", fake_connect)

        client = GrokVoiceClient(api_key="k")
        await client.connect()

        assert client._connected is True
        assert client._session_created_event.is_set()
        assert client._session_updated_event.is_set()
        # session.update was sent during connect
        types = [json.loads(m)["type"] for m in fake_ws.sent]
        assert "session.update" in types

        await client.close()

    @pytest.mark.asyncio
    async def test_connect_sends_greeting_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ai import grok_voice
        from app.ai.grok_voice import GrokVoiceClient

        scripted = [
            json.dumps({"type": "session.created", "session": {}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ]
        fake_ws = _RecvControlledFakeWS(scripted)

        async def fake_connect(*args: object, **kwargs: object) -> _RecvControlledFakeWS:
            return fake_ws

        monkeypatch.setattr(grok_voice.websockets, "connect", fake_connect)

        client = GrokVoiceClient(api_key="k", greeting="welcome")
        await client.connect()

        types = [json.loads(m)["type"] for m in fake_ws.sent]
        assert "response.create" in types
        await client.close()

    @pytest.mark.asyncio
    async def test_close_unblocks_receive_chunks(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True

        # close() should put a sentinel and flip _connected
        await client.close()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_receive_chunks_yields_then_stops(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True
        await client._audio_queue.put(b"\x00" * 320)

        gen = client.receive_chunks()
        first = await gen.__anext__()
        assert first == b"\x00" * 320

        # Simulate close: flip flag and push sentinel
        client._connected = False
        await client._audio_queue.put(b"")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    @pytest.mark.asyncio
    async def test_events_yields_queued_event(self) -> None:
        from app.ai.duplex_base import AiEvent, AiEventType
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True
        await client._event_queue.put(AiEvent(type=AiEventType.CONNECTED))

        gen = client.events()
        evt = await gen.__anext__()
        assert evt.type == AiEventType.CONNECTED
```

- [ ] **Step 6.2: Run — expect failures**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokLifecycle -v`
Expected: FAIL — `connect`, `close`, `receive_chunks`, `events` not implemented.

- [ ] **Step 6.3: Implement the lifecycle methods**

Append to the `GrokVoiceClient` class in `app/ai/grok_voice.py`:

```python
    async def connect(self) -> None:
        """Connect to Grok Voice WebSocket and complete the handshake."""
        if self._connected:
            return

        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}

            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    f"{self._ws_url}?model={self._model}",
                    additional_headers=headers,
                    open_timeout=10.0,
                )

            self._connected = True
            self._stop_event.clear()
            self._session_created_event.clear()
            self._session_updated_event.clear()

            self._message_handler_task = asyncio.create_task(
                self._message_handler(),
                name="grok-message-handler",
            )

            self._logger.info("Waiting for session.created from Grok...")
            async with asyncio.timeout(5.0):
                await self._session_created_event.wait()

            self._logger.info("Configuring Grok session...")
            await self._configure_session()

            self._logger.info("Waiting for session.updated from Grok...")
            async with asyncio.timeout(5.0):
                await self._session_updated_event.wait()

            if self._greeting:
                await self._send_greeting()

            self._logger.info("Grok Voice connected", model=self._model, voice=self._voice)

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def close(self) -> None:
        """Close the Grok WebSocket and cancel background tasks."""
        if not self._connected:
            return

        self._connected = False
        self._stop_event.set()
        try:
            self._audio_queue.put_nowait(b"")
        except asyncio.QueueFull:
            pass

        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                self._logger.debug("Message handler cancelled")

        if self._ws:
            await self._ws.close()

        self._logger.info("Grok Voice disconnected")

    async def _message_handler(self) -> None:
        """Read frames from the WebSocket and dispatch via _process_message."""
        if not self._ws:
            return

        while not self._stop_event.is_set():
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                await self._process_message(data)
            except websockets.exceptions.ConnectionClosed:
                self._logger.warning("Grok WebSocket closed")
                self._connected = False
                self._stop_event.set()
                try:
                    self._event_queue.put_nowait(
                        AiEvent(type=AiEventType.DISCONNECTED, timestamp=time.time())
                    )
                except asyncio.QueueFull:
                    pass
                try:
                    self._audio_queue.put_nowait(b"")
                except asyncio.QueueFull:
                    pass
                break
            except Exception as e:
                self._logger.error("Grok message handler error", error=str(e))

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM16 @ 8kHz audio chunks decoded from Grok."""
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
                if not self._connected and chunk == b"":
                    break
                yield chunk
            except Exception as e:
                self._logger.error("Grok audio stream error", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Yield AI events (CONNECTED, ERROR, etc.)."""
        while self._connected:
            try:
                event = await self._event_queue.get()
                yield event
            except Exception as e:
                self._logger.error("Grok event stream error", error=str(e))
                break

    async def update_session(self, config: Dict) -> None:
        """Send a session.update with the given config payload."""
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")
        message = {"type": "session.update", "session": config}
        await self._ws.send(json.dumps(message))
        self._logger.info("Grok session updated")

    async def ping(self) -> bool:
        """Health check via WebSocket ping."""
        if not self._connected or not self._ws:
            return False
        try:
            pong_waiter = await self._ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=5.0)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def reconnect(self) -> None:
        """Close and reconnect."""
        await self.close()
        await asyncio.sleep(1.0)
        await self.connect()
```

- [ ] **Step 6.4: Run lifecycle tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokLifecycle -v`
Expected: All 5 PASS.

- [ ] **Step 6.5: Run mypy strict on the full file**

Run: `uv run mypy app/ai/grok_voice.py`
Expected: No errors.

- [ ] **Step 6.6: Run full file tests + ensure no regressions**

Run: `uv run pytest tests/test_grok_voice.py -v && uv run pytest -q`
Expected: All `test_grok_voice.py` tests pass; all pre-existing tests still pass.

- [ ] **Step 6.7: Commit**

```bash
git add app/ai/grok_voice.py tests/test_grok_voice.py
git commit -m "add grok client lifecycle (connect/close/receive/events)"
```

---

## Phase 3: Wire Grok into the vendor factory

### Task 7: Add `grok` branch to `create_ai_client()`

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_grok_voice.py`

- [ ] **Step 7.1: Write failing test for the factory branch**

Append to `tests/test_grok_voice.py`:

```python


class TestGrokFactory:
    """Tests for create_ai_client() vendor=grok branch."""

    def test_create_ai_client_returns_grok_client(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # Build minimal agent_prompt.yaml
        prompt = tmp_path / "agent_prompt.yaml"
        prompt.write_text("instructions: be helpful\ngreeting: hi\n")

        env = {
            "AI_VENDOR": "grok",
            "XAI_API_KEY": "k",
            "AGENT_PROMPT_FILE": str(prompt),
        }
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            # Re-import main so it picks up the reloaded config singleton
            from app import main as main_module
            reload(main_module)

            client = main_module.create_ai_client()

        from app.ai.grok_voice import GrokVoiceClient
        assert isinstance(client, GrokVoiceClient)
        assert client._model == "grok-voice-think-fast-1.0"
        assert client._voice == "eve"
        assert client._greeting == "hi"

    def test_create_ai_client_grok_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = {"AI_VENDOR": "grok"}
        # Ensure XAI_API_KEY absent
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            from app import main as main_module
            reload(main_module)

            with pytest.raises(ValueError, match="Grok API key"):
                main_module.create_ai_client()
```

- [ ] **Step 7.2: Run — expect failure**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokFactory -v`
Expected: FAIL — `ValueError: Unsupported AI vendor: grok` (raised by the existing else-branch in `create_ai_client`).

- [ ] **Step 7.3: Add the `grok` branch to `create_ai_client` in `app/main.py`**

In `app/main.py`, find the line beginning with `elif vendor == "gemini":` and locate the end of that block (just before the final `else: raise ValueError(f"Unsupported AI vendor: {vendor}")`).

Insert this new branch immediately before the final `else:`:

```python
    elif vendor == "grok":
        if not config.ai.grok_api_key:
            raise ValueError("Grok API key not configured")

        instructions, greeting = _load_agent_config(logger)

        logger.info(
            "Using Grok Voice client",
            model=config.ai.grok_model,
            voice=config.ai.grok_voice,
            has_greeting=greeting is not None,
            instructions_length=len(instructions),
            greeting_preview=greeting[:50] if greeting else None,
        )

        from app.ai.grok_voice import GrokVoiceClient
        return GrokVoiceClient(
            api_key=config.ai.grok_api_key,
            model=config.ai.grok_model,
            voice=config.ai.grok_voice,
            instructions=instructions,
            greeting=greeting,
            ws_endpoint=config.ai.grok_ws_endpoint,
        )

```

(The local `from app.ai.grok_voice import GrokVoiceClient` mirrors a per-branch import, but a top-of-file import is also acceptable. If the existing imports at the top of `main.py` group all AI clients together, prefer adding the import there for consistency. To match style: add `from app.ai.grok_voice import GrokVoiceClient` near the other `from app.ai.*` imports at the top of `main.py` and remove the local import inside the branch.)

- [ ] **Step 7.4: Run factory tests — should pass**

Run: `uv run pytest tests/test_grok_voice.py::TestGrokFactory -v`
Expected: Both tests PASS.

- [ ] **Step 7.5: Run the full suite**

Run: `uv run pytest -q`
Expected: All previously-green tests still green; new tests all green.

- [ ] **Step 7.6: Commit**

```bash
git add app/main.py tests/test_grok_voice.py
git commit -m "wire grok vendor into ai client factory"
```

---

## Phase 4: Documentation

### Task 8: Update `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 8.1: Edit `.env.example`**

Find the line:

```
# Options: openai, deepgram, gemini
```

Change to:

```
# Options: openai, deepgram, gemini, grok
```

After the Gemini block (after the line `GEMINI_VOICE=Puck  # Voice: Puck, Charon, Kore, Fenrir, Aoede`), add a blank line, then:

```
# xAI Grok Voice Configuration (when AI_VENDOR=grok)
# Get API key from: https://console.x.ai/
XAI_API_KEY=your_xai_api_key_here
GROK_MODEL=grok-voice-think-fast-1.0  # Options: grok-voice-fast-1.0, grok-voice-think-fast-1.0
GROK_VOICE=eve  # Built-in voices: eve (default), ara, leo, rex, sal
GROK_WS_ENDPOINT=wss://api.x.ai/v1/realtime
```

- [ ] **Step 8.2: Verify the file parses as a dotenv (no syntax errors)**

Run: `uv run python -c "from dotenv import dotenv_values; v = dotenv_values('.env.example'); assert 'XAI_API_KEY' in v; assert v['GROK_MODEL'] == 'grok-voice-think-fast-1.0'; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 8.3: Commit**

```bash
git add .env.example
git commit -m "document grok vendor in env example"
```

---

### Task 9: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 9.1: Update the vendor list near the top of the README**

In `README.md`, find the bullet list:

```
- ✅ **OpenAI Realtime API** (gpt-realtime GA)
- ✅ **Deepgram Voice Agent**
- ✅ **Gemini Live** (Gemini 2.5 Flash)
```

Add one line at the end:

```
- ✅ **xAI Grok Voice** (grok-voice-think-fast-1.0)
```

In the next paragraph that begins `Simple passthrough bridge:`, update it to mention Grok where it lists native G.711 vendors. Find:

```
Simple passthrough bridge: **SIP (G.711 μ-law @ 8kHz)** ↔ **AI voice models**. OpenAI and Deepgram support native G.711, Gemini requires PCM16 resampling (8kHz ↔ 16kHz/24kHz).
```

Change to:

```
Simple passthrough bridge: **SIP (G.711 μ-law @ 8kHz)** ↔ **AI voice models**. OpenAI, Deepgram, and Grok support native G.711, Gemini requires PCM16 resampling (8kHz ↔ 16kHz/24kHz).
```

- [ ] **Step 9.2: Add a new "Grok Voice Setup" section after the Gemini Live Setup section**

Find the closing of the "Gemini Live Setup" section (the line `**Note:** Gemini Live uses PCM16 audio (16kHz input, 24kHz output), so the bridge performs resampling from/to 8kHz SIP audio. This adds minimal latency (<5ms).`).

After that line, insert a blank line, then:

```
## Grok Voice Setup

Set `AI_VENDOR=grok` in `.env`:

```bash
AI_VENDOR=grok
XAI_API_KEY=your-key-here
AGENT_PROMPT_FILE=agent_prompt.yaml
GROK_MODEL=grok-voice-think-fast-1.0
GROK_VOICE=eve
```

Available built-in voices: `eve` (default), `ara`, `leo`, `rex`, `sal`.

Available models:
- `grok-voice-think-fast-1.0` (recommended — best UX with reasoning)
- `grok-voice-fast-1.0` (faster, cheaper)

Get your API key from [xAI Console](https://console.x.ai/).

**Note:** Grok Voice supports native G.711 μ-law @ 8kHz — same as OpenAI and Deepgram, so no resampling overhead. The realtime protocol mirrors OpenAI's, including server-side VAD and barge-in.
```

(Note: in the actual README edit, the inline triple-backticks above need to be the literal Markdown — not nested. If your editor escapes them, just paste the raw section.)

- [ ] **Step 9.3: Verify Markdown renders sanely**

Run: `uv run python -c "import pathlib; t = pathlib.Path('README.md').read_text(); assert 'Grok Voice Setup' in t; assert 'XAI_API_KEY' in t; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 9.4: Commit**

```bash
git add README.md
git commit -m "document grok voice setup in readme"
```

---

## Phase 5: Final verification

### Task 10: Full-suite sanity, type check, lint

**Files:** none (read-only verification)

- [ ] **Step 10.1: Run the full test suite**

Run: `uv run pytest -q`
Expected: All tests pass. Test count = (baseline from Step 0.2) + ~25 new tests.

- [ ] **Step 10.2: Run mypy strict on the new module**

Run: `uv run mypy app/ai/grok_voice.py`
Expected: No errors.

- [ ] **Step 10.3: Run mypy on touched modules**

Run: `uv run mypy app/config.py app/main.py app/ai/grok_voice.py`
Expected: No errors. (If pre-existing modules have unrelated errors — `OpenAIRealtimeClient`, etc. — they are out of scope; do not "fix" them in this plan.)

- [ ] **Step 10.4: Run ruff lint on changed files**

Run: `uv run ruff check app/ai/grok_voice.py app/config.py app/main.py tests/test_grok_voice.py`
Expected: `All checks passed!`

- [ ] **Step 10.5: Run ruff format check**

Run: `uv run ruff format --check app/ai/grok_voice.py tests/test_grok_voice.py`
Expected: `0 files would be reformatted` (or run `uv run ruff format` to auto-fix and commit a formatting follow-up).

- [ ] **Step 10.6: Confirm git history is clean and tells a story**

Run: `git log --oneline -15`
Expected: A series of commits matching the task structure (config, skeleton, uplink, message processing, session config, lifecycle, factory, env example, readme).

- [ ] **Step 10.7: Confirm `.env.example` actually parses end-to-end**

Run: `uv run python -c "from app.config import Config; import os; os.environ['AI_VENDOR']='grok'; os.environ['XAI_API_KEY']='x'; c = Config.load(); print(c.ai.vendor, c.ai.grok_model, c.ai.grok_voice)"`
Expected: prints `grok grok-voice-think-fast-1.0 eve`.

---

### Task 11: Manual integration smoke test (optional, requires real API key)

**Files:** none — operator-only step

- [ ] **Step 11.1: Set up `.env` with real key**

Copy `.env.example` to `.env` if not present. Set:
```
AI_VENDOR=grok
XAI_API_KEY=<real key>
AGENT_PROMPT_FILE=agent_prompt.yaml
SIP_DOMAIN=<your bind ip>
SIP_PORT=6060
```

- [ ] **Step 11.2: Start the server**

Run: `uv run python -m app.main`
Expected: log line `SIP server ready - waiting for INVITE requests`, vendor=`grok`.

- [ ] **Step 11.3: Place a test SIP call**

Use a softphone (e.g. Linphone, Zoiper) to dial `sip:<SIP_DOMAIN>:6060`.
Expected:
- Greeting from `agent_prompt.yaml` is heard within ~2s of pickup.
- Two-way conversation works.
- Speaking over the agent triggers barge-in (Grok stops within ~300ms).
- Hanging up cleanly closes the WebSocket (check logs for "Grok Voice disconnected").

- [ ] **Step 11.4: Schema verification**

If the connect log line `Configuring Grok session` is followed shortly by an `error` event from Grok complaining about the `audio_format` shape, this confirms the spec's flagged uncertainty about the exact field nesting. In that case:
- Read the error message — it will name the offending field.
- Adjust `_configure_session` in `app/ai/grok_voice.py` to match the API's expected shape (e.g., move to OpenAI-style `audio.input.format.type: "audio/pcmu"`).
- Update the corresponding test in `TestGrokSessionConfig::test_configure_session_payload`.
- Re-run `uv run pytest tests/test_grok_voice.py::TestGrokSessionConfig -v`.
- Commit: `git commit -am "fix grok session audio_format shape per live api"`.

If the connection succeeds without an `audio_format` error, the spec's assumption was correct and no further action is needed.

---

## Self-Review Notes

I checked this plan against `docs/superpowers/specs/2026-05-04-grok-voice-agent-design.md`:

- **Spec coverage:**
  - File layout (new `grok_voice.py`, modified `config.py`/`main.py`/`README.md`/`.env.example`) → Tasks 1, 2-6, 7, 8, 9.
  - Data flow (PCM16 → mu-law → WS, no resampling) → Task 3 (uplink) + Task 4 (downlink).
  - Protocol mapping table → Task 4 covers all listed events; the "log only" events (`response.output_audio_transcript.delta/done`, `response.done`) are explicitly handled.
  - Session config payload → Task 5.
  - Greeting → Task 5 (`_send_greeting`) and Task 6 (sent on connect after `session.updated`).
  - Configuration env vars + `AIConfig` additions + vendor literal expansion → Task 1.
  - `create_ai_client()` branch → Task 7.
  - Error handling table → covered: missing key (Task 2 + Task 7), connect timeout (Task 6 `connect`), `session.created/updated` 5s timeouts (Task 6), WS closed mid-call (Task 6 `_message_handler`), `error` event (Task 4), audio queue full (Task 3 — uses `asyncio.Queue` with `maxsize=100`, `QueueFull` propagates from `put_nowait` paths in `_message_handler`).
  - Logging cadence (50/10) → Task 3 and Task 4.
  - Unit tests list (9 items in spec) → all 9 covered: connect lifecycle (6.1), session config payload (5.1), audio uplink (3.1), audio downlink (4.1), frame size validation (3.1), error event mapping (4.1), disconnect handling (6.1 — covered by `_RecvControlledFakeWS.close` triggering `ConnectionClosed`), greeting (5.1, 6.1), missing API key (2.1).
  - Manual smoke test → Task 11.
  - Schema verification step from spec ("verified against live API on first connect") → Task 11.4.

- **Placeholder scan:** No "TBD"/"TODO"/"implement later". Every step has either exact code or an exact command. The schema-verification step in 11.4 is conditional but concrete — it gives the operator the exact remediation path if the audio_format shape is wrong.

- **Type consistency:** Method names match between tasks: `send_pcm16_8k`, `_process_message`, `_configure_session`, `_send_greeting`, `connect`, `close`, `_message_handler`, `receive_chunks`, `events`, `update_session`, `ping`, `reconnect`. Test class names are unique. `_FakeWebSocket` is defined once (Task 3) and extended once (`_RecvControlledFakeWS` in Task 6).

- **Known soft spots flagged:**
  1. The `audio_format` nesting in Task 5 is the spec's known unknown. If the live API rejects it, Task 11.4 documents the fix.
  2. The factory test in Task 7 reloads modules, which is brittle but matches the only sensible way to test config-singleton behavior. If this proves flaky, it can be refactored to inject `Config` rather than rely on the module-level singleton — but that's a separate refactor and is intentionally out of scope.
  3. The `_RecvControlledFakeWS` test fake intentionally raises `websockets.exceptions.ConnectionClosed(None, None)` with `None` args; if the installed websockets version requires a `Close` frame object, the test will need a stub. This is small enough to fix inline if it surfaces.

No gaps found. No fixes needed.

---

## Post-Execution Corrections (2026-05-04 smoke test)

Tasks 1–10 landed as planned. Task 11 (manual smoke test against the live xAI API) revealed that several details derived from my pre-implementation doc summary diverged from the actual `https://docs.x.ai/voice-realtime.ws.json` schema. Three follow-up commits on the feature branch corrected the live-API contract before merge to `main`. The plan steps above reflect what was originally specified — code on `main` is the authoritative current state.

| Aspect | Plan said | Live API actually requires | Fix commit |
|---|---|---|---|
| Connection-ready event | `session.created` | `conversation.created` (we now tolerate either) | `b577549` |
| `response.create` payload | `metadata: {response_purpose}` | adds required `metadata.client_event_id` (uuid4) | `92e487d` |
| Session audio config path | `session.audio_format.{input,output}.{type, sample_rate}` with type `mulaw` | OpenAI-style `session.audio.{input,output}.format.{type, rate}` with type `audio/pcmu` | `721488b` |
| System prompt field | `session.system_prompt` | `session.instructions` (without this fix the agent_prompt.yaml was silently ignored) | `721488b` |
| Default output rate | implicit 8000 | server default is **24000** — explicit `rate: 8000` is required | `721488b` |
| `ping` server keepalive | not in plan | added explicit no-op handler | `b577549` |

If you are reading this plan to follow it task-by-task in a fresh implementation, apply the corrections above directly when writing `_configure_session` and `_send_greeting` (Task 5) and `_process_message` (Task 4) — don't reproduce the original wrong values just to "fix them later".

---

## Done.
