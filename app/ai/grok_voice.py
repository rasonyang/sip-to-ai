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
