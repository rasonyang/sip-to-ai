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
