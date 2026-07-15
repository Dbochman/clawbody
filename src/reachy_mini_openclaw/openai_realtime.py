"""OpenAI Realtime speech transport for OpenClaw-owned Reachy turns.

Realtime provides VAD and transcription; the Speech endpoint renders OpenClaw's
exact text. OpenClaw owns memory, personality, reasoning, tools, and every reply.
"""

import asyncio
import base64
import logging
import random
from typing import Any, Final, Literal

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray
from openai import AsyncOpenAI
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.prompts import get_session_voice
from reachy_mini_openclaw.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000
TTS_CHUNK_BYTES: Final[int] = 4800


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Speech transport for turns owned entirely by the OpenClaw agent.

    Realtime provides VAD, transcription, and voice rendering only. Every completed
    transcript is sent to OpenClaw, which owns memory, reasoning, tools, robot
    actions, and the exact response text that Reachy vocalizes.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        openclaw_bridge: Any | None = None,
        gradio_mode: bool = False,
    ):
        """Initialize the handler.

        Args:
            deps: Tool dependencies for robot control
            openclaw_bridge: Bridge to OpenClaw gateway
            gradio_mode: Whether running with Gradio UI
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )

        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.gradio_mode = gradio_mode

        # OpenAI connection
        self.client: AsyncOpenAI | None = None
        self.connection: Any = None

        # Output queue
        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        # State tracking
        self.last_activity_time = 0.0
        self.start_time = 0.0
        self._speaking = False  # True when robot is speaking
        self._speaking_until = 0.0
        self._audio_playback_until = 0.0
        self._last_speech_stopped_at = 0.0
        self._playback_marker_turn_started: float | None = None

        # Lifecycle flags
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        self._turn_lock = asyncio.Lock()
        self._speech_lock = asyncio.Lock()
        self._turn_tasks: set[asyncio.Task] = set()

    def copy(self) -> "OpenAIRealtimeHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return OpenAIRealtimeHandler(self.deps, self.openclaw_bridge, self.gradio_mode)

    async def start_up(self) -> None:
        """Start the handler and connect to OpenAI.

        Runs an infinite reconnection loop so the robot stays alive
        even if the WebSocket drops (network blip, idle timeout, etc.).
        """
        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.error("OPENAI_API_KEY not configured")
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time

        attempt = 0
        max_backoff = 30  # Cap backoff at 30 seconds

        while not self._shutdown_requested:
            attempt += 1
            try:
                await self._run_session()
                # Session ended cleanly (shouldn't normally happen)
                if self._shutdown_requested:
                    return
                # Reset attempt counter on a clean exit
                attempt = 0
            except ConnectionClosedError as e:
                logger.warning("WebSocket closed unexpectedly (attempt %d): %s", attempt, e)
            except Exception as e:
                logger.error("Session error (attempt %d): %s", attempt, e)
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

            if self._shutdown_requested:
                return

            # Exponential backoff with jitter, capped at max_backoff
            delay = min(max_backoff, (2 ** min(attempt - 1, 5))) + random.uniform(0, 1)
            logger.info("Reconnecting in %.1f seconds...", delay)
            await asyncio.sleep(delay)

    async def _run_session(self) -> None:
        """Run a single OpenAI Realtime session."""
        model = config.OPENAI_MODEL
        logger.info("Connecting to OpenAI Realtime API with model: %s", model)

        async with self.client.realtime.connect(model=model) as conn:
            # Realtime is deliberately not the conversational agent. It detects
            # turns and transcribes speech, but never creates an automatic reply.
            await conn.session.update(
                session={
                    "type": "realtime",
                    "model": model,
                    "output_modalities": ["audio"],
                    "instructions": ("Transcribe input speech only. Never answer users or create assistant content."),
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                            "transcription": {
                                "model": config.OPENAI_TRANSCRIPTION_MODEL,
                                "language": config.OPENAI_TRANSCRIPTION_LANGUAGE,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": config.OPENAI_VAD_SILENCE_MS,
                                "create_response": False,
                                "interrupt_response": False,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                            "voice": get_session_voice(),
                        },
                    },
                    "tools": [],
                    "tool_choice": "none",
                },
            )
            logger.info(
                "OpenAI Realtime configured as speech transport "
                "(automatic responses disabled, 0 tools, transcription=%s, "
                "language=%s, silence=%dms)",
                config.OPENAI_TRANSCRIPTION_MODEL,
                config.OPENAI_TRANSCRIPTION_LANGUAGE,
                config.OPENAI_VAD_SILENCE_MS,
            )

            self.connection = conn
            self._connected_event.set()

            # Process events
            async for event in conn:
                await self._handle_event(event)

    async def _handle_event(self, event: Any) -> None:
        """Handle an event from the OpenAI Realtime API."""
        event_type = event.type

        # The wireless speaker is close to the microphones. Ignore any delayed
        # VAD events caused by Reachy's own playback so TTS cannot open a turn.
        if (
            event_type
            in {
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "conversation.item.input_audio_transcription.completed",
            }
            and self._is_playing_speech()
        ):
            logger.debug("Ignoring input event during Reachy speech: %s", event_type)
            return

        # Speech detection
        if event_type == "input_audio_buffer.speech_started":
            # User started speaking - stop any current output
            self._speaking = False
            self.deps.movement_manager.set_processing(False)
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            self._set_speech_tracking(True)
            logger.info("User started speaking")

        if event_type == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
            self._set_speech_tracking(False)
            self._last_speech_stopped_at = asyncio.get_event_loop().time()
            logger.info("User stopped speaking")

        # OpenClaw owns the turn as soon as transcription completes.
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                now = asyncio.get_event_loop().time()
                if self._last_speech_stopped_at:
                    logger.info(
                        "Latency transcription-ready: %.0fms",
                        (now - self._last_speech_stopped_at) * 1000,
                    )
                logger.info("User: %s", transcript)
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                task = asyncio.create_task(
                    self._run_openclaw_turn(transcript.strip()),
                    name="openclaw-voice-turn",
                )
                self._turn_tasks.add(task)
                task.add_done_callback(self._turn_tasks.discard)

            # The transcript now lives in OpenClaw's session. Remove the audio
            # item from Realtime so its unused conversation cannot grow forever.
            item_id = getattr(event, "item_id", None)
            if item_id and self.connection:
                try:
                    await self.connection.conversation.item.delete(item_id=item_id)
                except Exception as exc:
                    logger.debug("Could not delete transcribed Realtime item: %s", exc)

        # Errors
        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)

    def _set_speech_tracking(self, listening: bool) -> None:
        """Give tracking the head only while the user is actively speaking."""
        if not self.deps.daemon_face_tracking:
            return
        try:
            self.deps.robot.start_head_tracking(weight=0.85 if listening else 0.25)
            logger.info(
                "Speech-gated face tracking %s",
                "active" if listening else "released",
            )
        except Exception as exc:
            logger.warning("Could not update speech-gated face tracking: %s", exc)

    def _is_playing_speech(self) -> bool:
        return asyncio.get_event_loop().time() < self._speaking_until

    def suppress_input_for(self, duration_seconds: float) -> None:
        """Prevent robot audio playback from being transcribed as user speech."""
        now = asyncio.get_event_loop().time()
        self._speaking = True
        self._audio_playback_until = max(now, self._audio_playback_until) + duration_seconds
        self._speaking_until = self._audio_playback_until + 0.75

    def note_audio_playback_started(self) -> None:
        """Record when the first streamed PCM chunk reaches Reachy's speaker."""
        if self._playback_marker_turn_started is None:
            return
        logger.info(
            "Latency first-speaker-push: %.0fms",
            (asyncio.get_event_loop().time() - self._playback_marker_turn_started) * 1000,
        )
        self._playback_marker_turn_started = None

    async def _run_openclaw_turn(self, transcript: str) -> None:
        """Buffer one OpenClaw-owned turn, then stream one continuous speech clip."""
        async with self._turn_lock:
            loop = asyncio.get_event_loop()
            turn_started = loop.time()
            self.deps.movement_manager.set_processing(True)

            if self.openclaw_bridge is None:
                await self.speak_text("I can't reach OpenClaw right now.")
                return
            if not self.openclaw_bridge.is_connected:
                connected = await self.openclaw_bridge.connect()
                if not connected:
                    await self.speak_text("I can't reach OpenClaw right now.")
                    return

            logger.info("Routing voice turn directly to OpenClaw (full-response buffer)")
            reply_parts: list[str] = []
            first_text_seen = False
            try:
                async for delta in self.openclaw_bridge.stream_chat(
                    transcript,
                    system_context=(
                        "This is a direct voice turn from the authenticated, physically "
                        "secured Reachy Mini session. You are the sole conversational "
                        "agent: use your memory, personality, tools, and the reachy-control "
                        "skill directly. Perform requested robot actions before replying "
                        "and never claim an action succeeded unless its command confirmed "
                        "success. Your complete final response is spoken verbatim through "
                        "Reachy, so make it concise and natural for voice. Do not call "
                        "reachyctl speak during this session; the bridge vocalizes your "
                        "response automatically."
                    ),
                ):
                    if not first_text_seen:
                        first_text_seen = True
                        logger.info(
                            "Latency OpenClaw first text: %.0fms",
                            (loop.time() - turn_started) * 1000,
                        )
                    reply_parts.append(delta)
            except Exception as exc:
                logger.warning("OpenClaw turn failed: %s", exc)
                reply = "I'm having trouble reaching my OpenClaw brain right now."
            else:
                reply = "".join(reply_parts).strip()
                if not reply:
                    reply = "I didn't get a response from OpenClaw."

            logger.info(
                "Latency OpenClaw final text: %.0fms",
                (loop.time() - turn_started) * 1000,
            )
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": reply}))
            result = await self.speak_text(reply, turn_started=turn_started)
            if result.get("error"):
                logger.warning("Could not render buffered OpenClaw speech: %s", result["error"])

    async def speak_text(
        self,
        text: str,
        *,
        turn_started: float | None = None,
    ) -> dict[str, Any]:
        """Render OpenClaw-provided text through the active Realtime voice."""
        text = text.strip()
        if not text:
            return {"error": "Speech text is empty"}
        if len(text) > 4000:
            return {"error": "Speech text exceeds 4000 characters"}

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=15)
        except TimeoutError:
            return {"error": "OpenAI Realtime speech transport is unavailable"}
        if self.connection is None:
            return {"error": "OpenAI Realtime speech transport is unavailable"}

        async with self._speech_lock:
            try:
                self._speaking = True
                request_started = asyncio.get_event_loop().time()
                first_chunk = True
                total_bytes = 0
                trailing_byte = b""
                async with self.client.audio.speech.with_streaming_response.create(
                    model=config.OPENAI_TTS_MODEL,
                    voice=config.OPENAI_TTS_VOICE,
                    input=text,
                    response_format="pcm",
                ) as response:
                    async for raw_chunk in response.iter_bytes(chunk_size=TTS_CHUNK_BYTES):
                        chunk = trailing_byte + raw_chunk
                        trailing_byte = b""
                        if len(chunk) % 2:
                            trailing_byte = chunk[-1:]
                            chunk = chunk[:-1]
                        if not chunk:
                            continue

                        if first_chunk:
                            first_chunk = False
                            if turn_started is not None:
                                logger.info(
                                    "Latency first TTS byte: %.0fms",
                                    (asyncio.get_event_loop().time() - turn_started) * 1000,
                                )
                                if self._playback_marker_turn_started is None:
                                    self._playback_marker_turn_started = turn_started
                            logger.info(
                                "TTS time to first byte: %.0fms",
                                (asyncio.get_event_loop().time() - request_started) * 1000,
                            )
                            # Keep the thinking pose until playable audio exists.
                            self.deps.movement_manager.set_processing(False)

                        total_bytes += len(chunk)
                        self.suppress_input_for(len(chunk) / (2 * OPENAI_SAMPLE_RATE))
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed(base64.b64encode(chunk).decode("ascii"))
                        audio_data = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
                        await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_data))

                if first_chunk:
                    raise RuntimeError("OpenAI returned no speech audio")
                if trailing_byte:
                    logger.debug("Discarding incomplete PCM sample from TTS stream")

                duration_seconds = total_bytes / (2 * OPENAI_SAMPLE_RATE)
                self.last_activity_time = asyncio.get_event_loop().time()
                logger.info(
                    "Streamed complete OpenClaw speech in %.0fms: %s",
                    (asyncio.get_event_loop().time() - request_started) * 1000,
                    text[:100] if len(text) > 100 else text,
                )
                return {
                    "status": "success",
                    "speech": "queued",
                    "characters": len(text),
                    "backend": f"openai-{config.OPENAI_TTS_MODEL}",
                    "duration_seconds": round(duration_seconds, 2),
                }
            except Exception as exc:
                self.deps.movement_manager.set_processing(False)
                logger.error("Speech rendering failed: %s", exc)
                return {"error": str(exc)}
            finally:
                if not self._is_playing_speech():
                    self._speaking = False

    async def receive(self, frame: tuple[int, NDArray]) -> None:
        """Receive audio from the robot microphone."""
        if not self.connection:
            return
        if self._is_playing_speech():
            return
        if self._speaking:
            self._speaking = False
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

        input_sr, audio = frame

        # Handle stereo
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        audio = audio.flatten()

        # Convert to float for resampling
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample to OpenAI sample rate
        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio) * OPENAI_SAMPLE_RATE / input_sr)
            audio = resample(audio, num_samples).astype(np.float32)

        # Convert to int16 for OpenAI
        audio_int16 = (audio * 32767).astype(np.int16)

        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output (audio or transcript)."""
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("Connection close: %s", e)
            self.connection = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
