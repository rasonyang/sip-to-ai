# Grok Voice Agent — Design Spec

**Date:** 2026-05-04
**Status:** Approved (pending user spec review)
**Scope:** Add xAI Grok Voice as a fourth realtime AI vendor in SIP-to-AI.

## Goal

Add a new AI vendor `grok` to the SIP-to-AI bridge so incoming SIP calls can be served by xAI's Grok Voice realtime API, alongside the existing `openai`, `deepgram`, and `gemini` vendors.

## Background

xAI's Grok Voice realtime API (`wss://api.x.ai/v1/realtime`) exposes a duplex audio WebSocket whose protocol mirrors OpenAI Realtime almost 1:1: the same event names (`session.update`, `session.created`, `input_audio_buffer.append`, `input_audio_buffer.speech_started`, `response.output_audio.delta`, `response.done`, `error`), the same handshake (`session.created` → `session.update` → `session.updated`), and native G.711 μ-law @ 8kHz support — meaning no resampling between SIP and the AI service.

The existing project already has a clean per-vendor pattern: each AI is a single file under `app/ai/` implementing `AiDuplexClient`, registered through a factory in `app/main.py` and configured via env vars in `app/config.py`. Grok fits this pattern directly.

## Non-Goals

- Function calling / tool use — out of scope for v1.
- Custom voices (8-char alphanumeric IDs) — built-in voices only for v1.
- Ephemeral client-secret auth (`POST /v1/realtime/client_secrets`) — server-side bridge uses direct Bearer auth.
- Manual VAD mode — server VAD only.
- Refactoring `OpenAIRealtimeClient` to share a base class with Grok — duplicate-and-adapt to keep blast radius minimal.
- New tests for OpenAI / Deepgram / Gemini vendors — their code paths are untouched.
- CI integration test against the live xAI service — manual smoke test only, matching project posture.

## Architecture

### File Layout

**New file:**
- `app/ai/grok_voice.py` — `GrokVoiceClient(AiDuplexBase)`. WebSocket client patterned line-for-line on `app/ai/openai_realtime.py`. ~400 LOC. WebSocket URL `wss://api.x.ai/v1/realtime?model=<model>`. G.711 μ-law @ 8kHz both directions. Three async tasks (uplink send via `send_pcm16_8k`, message handler, downlink chunk drain via `receive_chunks`). Same handshake: `connect → wait session.created → send session.update → wait session.updated → optional greeting`.

**Modified files:**
- `app/config.py` — extend `AIConfig.vendor` Literal to include `"grok"`; add fields `grok_api_key`, `grok_model`, `grok_voice`, `grok_ws_endpoint`. Update the allowed-vendor list in `Config.__init__`. Read env vars `XAI_API_KEY`, `GROK_MODEL`, `GROK_VOICE`, `GROK_WS_ENDPOINT`.
- `app/main.py` — add `elif vendor == "grok":` branch to `create_ai_client()`, mirroring the OpenAI branch (validate key, load agent prompt, instantiate `GrokVoiceClient`).
- `README.md` — add "Grok Voice Agent Setup" section parallel to OpenAI / Deepgram / Gemini sections.
- `.env.example` — add `XAI_API_KEY`, `GROK_MODEL`, `GROK_VOICE` entries.

**Untouched:**
- `app/bridge/` — `CallSession` and `AudioAdapter` are vendor-agnostic.
- `app/sip_async/` — pure SIP/RTP layer.
- `app/ai/duplex_base.py` — `AiDuplexClient` Protocol already matches Grok's needs.
- `app/ai/openai_realtime.py`, `deepgram_agent.py`, `gemini_live.py` — no changes.

### Data Flow

The bridge pipeline is unchanged. Grok plugs in at the WebSocket boundary, identically to OpenAI:

```
RTP G.711 μ-law (160B / 20ms)
  ↔ RTPAudioBridge (G.711 ↔ PCM16)
  ↔ AudioAdapter (PCM16 320B frames, asyncio.Queue)
  ↔ CallSession.uplink_task / receive_task
  ↔ GrokVoiceClient.send_pcm16_8k() / receive_chunks()
  ↔ Codec.pcm16_to_ulaw() / ulaw_to_pcm16()  [internal to GrokVoiceClient]
  ↔ WebSocket wss://api.x.ai/v1/realtime
```

No resampling — same as OpenAI/Deepgram. PCM16 320B/20ms ↔ G.711 μ-law 160B/20ms within `GrokVoiceClient`, base64-encoded over JSON on the wire.

### Protocol Mapping

| Grok event (server → client) | Action in `_process_message` | Emitted `AiEventType` |
|---|---|---|
| `session.created` | set `_session_created_event` | `CONNECTED` |
| `session.updated` | set `_session_updated_event` | `SESSION_UPDATED` |
| `input_audio_buffer.speech_started` | barge-in signal | `TRANSCRIPT_PARTIAL` (data: `speech_started`) |
| `input_audio_buffer.speech_stopped` | — | `TRANSCRIPT_PARTIAL` (data: `speech_stopped`) |
| `conversation.item.input_audio_transcription.completed` | user transcript | `TRANSCRIPT_FINAL` |
| `response.output_audio.delta` | base64-decode → ulaw → PCM16 → enqueue | (no event; pushed to `_audio_queue`) |
| `response.output_audio_transcript.delta` | log AI's spoken text | (debug log only) |
| `response.output_audio_transcript.done` | log finalized AI transcript | (debug log only) |
| `response.done` | turn complete | (debug log only) |
| `error` | error payload | `ERROR` |
| anything else | log unhandled | — |

| Client → server | When sent |
|---|---|
| `session.update` | After `session.created` arrives, during `connect()` |
| `input_audio_buffer.append` | Per uplink frame, base64-encoded 160B μ-law |
| `response.create` (greeting) | After `session.updated`, only if `greeting` is configured |

### Session Configuration

Sent on connect:

```json
{
  "type": "session.update",
  "session": {
    "model": "grok-voice-think-fast-1.0",
    "voice": "eve",
    "system_prompt": "<from agent_prompt.yaml>",
    "audio_format": {
      "input":  {"type": "mulaw", "sample_rate": 8000},
      "output": {"type": "mulaw", "sample_rate": 8000}
    },
    "turn_detection": {"type": "server_vad"}
  }
}
```

**Schema verification note:** The exact `audio_format` field shape will be verified against the live xAI API during implementation. The doc summary listed `mulaw`/`alaw` and sample rates including 8000, but the precise nesting (e.g., `input.format.type` vs `audio_format.input.type` vs OpenAI-style `audio.input.format.type: "audio/pcmu"`) needs confirmation. The design intent — "G.711 μ-law @ 8kHz both directions, server-side VAD on" — holds regardless of the exact key path.

**Greeting:** identical to OpenAI's pattern — after `session.updated`, send:
```json
{
  "type": "response.create",
  "response": {
    "instructions": "<greeting text>",
    "conversation": "none",
    "output_modalities": ["audio"],
    "metadata": {"response_purpose": "greeting"}
  }
}
```

### Configuration

**Environment variables:**

```bash
AI_VENDOR=grok
XAI_API_KEY=xai-...                          # required
GROK_MODEL=grok-voice-think-fast-1.0         # default
GROK_VOICE=eve                               # default; alternatives: ara, leo, rex, sal
GROK_WS_ENDPOINT=wss://api.x.ai/v1/realtime  # default; overridable
AGENT_PROMPT_FILE=agent_prompt.yaml          # shared with other vendors
```

**`AIConfig` additions:**

```python
# xAI Grok Voice Configuration
grok_api_key: str = ""
grok_ws_endpoint: str = "wss://api.x.ai/v1/realtime"
grok_model: str = "grok-voice-think-fast-1.0"
grok_voice: str = "eve"
```

**Vendor literal:** `Literal["openai", "deepgram", "gemini", "grok"]`. The `ai_vendor not in [...]` validation list in `Config.__init__` adds `"grok"`.

**`create_ai_client()` branch** in `app/main.py`:
- Validate `XAI_API_KEY` is set; raise `ValueError("Grok API key not configured")` if missing — matches OpenAI's behavior.
- Load `instructions` and `greeting` via the existing vendor-agnostic `_load_agent_config()`.
- Instantiate `GrokVoiceClient(api_key=..., model=..., voice=..., instructions=..., greeting=..., ws_endpoint=...)`.

### Error Handling

Reuses the patterns from `openai_realtime.py`:

| Failure | Behavior |
|---|---|
| Missing `XAI_API_KEY` at startup | `ValueError`; server fails fast — no calls accepted |
| WebSocket connect timeout (10s) | `ConnectionError` from `connect()`; `CallSession` propagates; call ends |
| Auth rejection (401/403) | Error logged; call ends |
| `session.created` not received within 5s | Timeout → `ConnectionError` |
| `session.updated` not received within 5s | Timeout → `ConnectionError` |
| WebSocket closed mid-call | `_message_handler` emits `DISCONNECTED`; `CallSession` health task may trigger reconnect via existing `reconnect()` (close + sleep 1s + connect) |
| `error` event from Grok | Emit `AiEventType.ERROR` with message; logged but non-fatal — let server decide if WS survives |
| Audio queue full (back-pressure) | `asyncio.QueueFull` → drop frame, log, identical to existing clients |

**Logging:** structlog with same call-id binding pattern. Frame counters every 50 uplink / 10 downlink, matching OpenAI cadence.

**No new dependencies.** `websockets`, `structlog`, `numpy` already in `pyproject.toml`.

## Testing

### Unit Tests

`tests/test_grok_voice.py` — flat `tests/` layout matches the existing `test_codec.py`, `test_ring_buffer.py`, `test_bridge_end2end.py` pattern. No existing per-vendor tests under `tests/`, so this is the first.

- **Connection lifecycle:** mock `websockets.connect` to deliver `session.created` then `session.updated`; assert `connect()` completes within timeout, both `_session_*_event` flags set, `session.update` payload was sent.
- **Session config payload:** assert outbound `session.update` JSON contains correct `model`, `voice`, `system_prompt`, `audio_format`, `turn_detection`.
- **Audio uplink:** feed 320B PCM16 frame to `send_pcm16_8k()`; assert outbound `input_audio_buffer.append` contains base64-encoded 160B μ-law payload.
- **Audio downlink:** push fake `response.output_audio.delta` (base64 μ-law) into mock socket; assert `receive_chunks()` yields decoded 320B PCM16.
- **Frame size validation:** wrong-size input raises `ValueError`.
- **Error event mapping:** `error` message → `AiEventType.ERROR` on `events()`.
- **Disconnect handling:** closing socket → `DISCONNECTED` emitted, `_audio_queue` unblocks.
- **Greeting:** when `greeting` is set, after `session.updated` a `response.create` with greeting payload is sent.
- **Missing API key:** constructor with no key and no env raises `ValueError`.

### Integration

Manual smoke test, documented in README:
1. Set `AI_VENDOR=grok`, `XAI_API_KEY=...`, real `agent_prompt.yaml`.
2. Run `uv run python -m app.main`.
3. Dial in via SIP softphone.
4. Verify: greeting plays, two-way conversation works, barge-in interrupts.

No CI integration against the live xAI service — matches the project's current posture for OpenAI / Deepgram / Gemini.

## Out of Scope (deferred for future work)

- Function/tool calling — Grok supports it but no current bridge use case.
- Custom voices.
- Ephemeral `client_secrets` auth.
- Manual VAD mode.
- Per-call dynamic model/voice selection (e.g., picked from SIP `From` header).

## Open Questions

None at design time. The one item flagged for verification during implementation is the exact `audio_format` field nesting in `session.update` — confirmed against live API on first connect, with `mulaw` @ 8kHz as the target shape regardless.
