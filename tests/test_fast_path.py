import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from reachy_mini_openclaw.control_server import ClawBodyControlServer
from reachy_mini_openclaw.openclaw_bridge import (
    OpenClawBridge,
    OpenClawContinuityError,
)
from reachy_mini_openclaw.openai_realtime import (
    OpenAIRealtimeHandler,
    build_direct_voice_instructions,
    build_turn_detection,
)


class GatewayEventBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_completion_events_are_replayed_after_run_registration(self) -> None:
        bridge = OpenClawBridge()
        bridge._connected = True

        async def send_request(*_args, **_kwargs):
            await bridge._dispatch({
                "type": "event",
                "event": "agent",
                "payload": {
                    "runId": "run-fast",
                    "stream": "assistant",
                    "data": {"text": "REACHY_DELEGATION_OK"},
                },
            })
            await bridge._dispatch({
                "type": "event",
                "event": "chat",
                "payload": {
                    "runId": "run-fast",
                    "state": "delta",
                    "message": {
                        "content": [{"type": "text", "text": "REACHY_DELEGATION_OK"}]
                    },
                },
            })
            await bridge._dispatch({
                "type": "event",
                "event": "agent",
                "payload": {
                    "runId": "run-fast",
                    "stream": "lifecycle",
                    "data": {"phase": "finishing"},
                },
            })
            return {"ok": True, "payload": {"runId": "run-fast"}}

        bridge._send_request = AsyncMock(side_effect=send_request)

        response = await bridge.chat("Reply with the marker")

        self.assertEqual(response.error, None)
        self.assertEqual(response.content, "REACHY_DELEGATION_OK")
        self.assertNotIn("run-fast", bridge._early_run_events)
        self.assertNotIn("run-fast", bridge._run_events)


class RealtimeToolSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_does_not_block_realtime_event_handling(self) -> None:
        handler = object.__new__(OpenAIRealtimeHandler)
        handler._active_tool_calls = 0
        handler._turn_tasks = set()
        handler._response_active = True
        handler.deps = SimpleNamespace(
            movement_manager=SimpleNamespace(set_processing=Mock())
        )
        release_tool = asyncio.Event()

        async def handle_tool(_event) -> None:
            await release_tool.wait()

        handler._handle_tool_call = handle_tool
        event = SimpleNamespace(
            type="response.function_call_arguments.done",
            name="ask_openclaw",
        )

        await handler._handle_event(event)

        self.assertEqual(handler._active_tool_calls, 1)
        self.assertEqual(len(handler._turn_tasks), 1)
        self.assertFalse(next(iter(handler._turn_tasks)).done())

        release_tool.set()
        await asyncio.gather(*handler._turn_tasks)
        await asyncio.sleep(0)
        self.assertEqual(handler._active_tool_calls, 0)

    async def test_delegation_timeout_returns_a_tool_error(self) -> None:
        handler = object.__new__(OpenAIRealtimeHandler)
        handler.openclaw_bridge = SimpleNamespace(
            is_connected=True,
            chat=AsyncMock(side_effect=TimeoutError),
        )

        result = await handler._handle_openclaw_query(
            '{"query":"Use an OpenClaw skill"}'
        )

        self.assertIn("took too long", result["error"])


class RealtimeBargeInTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_flushes_playback_and_truncates_assistant_audio(self) -> None:
        handler = object.__new__(OpenAIRealtimeHandler)
        now = asyncio.get_running_loop().time()
        clear_player = Mock()
        truncate = AsyncMock()
        handler.output_queue = asyncio.Queue()
        await handler.output_queue.put((24000, object()))
        handler._response_audio = bytearray(b"buffered")
        handler._response_audio_started = True
        handler._speaking = True
        handler._speaking_until = now + 3
        handler._audio_playback_until = now + 3
        handler._playback_started_at = now - 1
        handler._current_audio_duration = 2.0
        handler._current_audio_item_id = "item-1"
        handler._current_audio_content_index = 0
        handler._turn_tasks = set()
        handler.connection = SimpleNamespace(
            conversation=SimpleNamespace(
                item=SimpleNamespace(truncate=truncate)
            )
        )
        handler.deps = SimpleNamespace(
            robot=SimpleNamespace(
                media=SimpleNamespace(
                    audio=SimpleNamespace(clear_player=clear_player)
                )
            ),
            head_wobbler=None,
        )

        await handler._interrupt_playback()
        await asyncio.gather(*handler._turn_tasks)

        self.assertTrue(handler.output_queue.empty())
        self.assertEqual(handler._response_audio, bytearray())
        clear_player.assert_called_once_with()
        truncate.assert_awaited_once()
        call = truncate.await_args.kwargs
        self.assertEqual(call["item_id"], "item-1")
        self.assertEqual(call["content_index"], 0)
        self.assertGreaterEqual(call["audio_end_ms"], 900)
        self.assertLessEqual(call["audio_end_ms"], 1100)


class RealtimeVadTests(unittest.TestCase):
    def test_direct_voice_enables_server_side_response_interruption(self) -> None:
        settings = build_turn_detection(create_response=True, barge_in=True)

        self.assertTrue(settings["create_response"])
        self.assertTrue(settings["interrupt_response"])

    def test_transcription_fallback_remains_non_interrupting(self) -> None:
        settings = build_turn_detection(create_response=False, barge_in=True)

        self.assertFalse(settings["create_response"])
        self.assertFalse(settings["interrupt_response"])


class ContinuityRpcTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_rpc_does_not_run_chat(self) -> None:
        bridge = OpenClawBridge()
        bridge._send_request = AsyncMock(
            return_value={
                "ok": True,
                "payload": {
                    "revision": "abc",
                    "identity": "My name is Claude.",
                    "soul": "Be useful.",
                    "user": "Dylan is the owner.",
                    "capsule": "No context.",
                },
            }
        )

        payload = await bridge.get_reachy_continuity_context()

        self.assertEqual(payload["revision"], "abc")
        self.assertEqual(payload["identity"], "My name is Claude.")
        bridge._send_request.assert_awaited_once_with(
            "reachy.continuity.context", {}, timeout=5
        )

    async def test_context_rpc_fails_closed_on_invalid_payload(self) -> None:
        bridge = OpenClawBridge()
        bridge._send_request = AsyncMock(return_value={"ok": True, "payload": {}})

        with self.assertRaises(OpenClawContinuityError):
            await bridge.get_reachy_continuity_context()

    async def test_summary_append_uses_plugin_rpc(self) -> None:
        bridge = OpenClawBridge()
        bridge._send_request = AsyncMock(
            return_value={"ok": True, "payload": {"status": "success"}}
        )

        await bridge.append_reachy_continuity_summary("Discussed dinner.", "turn-1")

        bridge._send_request.assert_awaited_once_with(
            "reachy.continuity.append",
            {"summary": "Discussed dinner.", "turnId": "turn-1"},
            timeout=5,
        )


class ControlLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_proactive_speech_holds_exclusive_control_lease(self) -> None:
        events: list[str] = []
        server = ClawBodyControlServer(SimpleNamespace())

        async def acquire(command: str) -> None:
            events.append(f"acquire:{command}")

        def release() -> None:
            events.append("release")

        async def speak(text: str) -> dict:
            events.append(f"speak:{text}")
            return {"status": "success"}

        server.set_control_callbacks(acquire, release, lambda: "direct_voice")
        server.set_speak_callback(speak)

        result = await server._execute(
            {"command": "speak", "arguments": {"text": "Hello"}}
        )

        self.assertEqual(result, {"status": "success"})
        self.assertEqual(events, ["acquire:speak", "speak:Hello", "release"])


class DirectVoicePromptTests(unittest.TestCase):
    def test_prompt_makes_the_complete_identity_stack_authoritative(self) -> None:
        prompt = build_direct_voice_instructions(
            "My name is Claude Bochman.",
            "Be warm, capable, and opinionated.",
            "Dylan and Julia share two homes.",
            "Recently discussed dinner.",
        )

        self.assertLess(prompt.index("# Voice identity contract"), prompt.index("<identity-md>"))
        self.assertIn("separate generic voice", prompt)
        self.assertIn("My name is Claude Bochman.", prompt)
        self.assertIn("Be warm, capable, and opinionated.", prompt)
        self.assertIn("Dylan and Julia share two homes.", prompt)
        self.assertIn("Recently discussed dinner.", prompt)


if __name__ == "__main__":
    unittest.main()
