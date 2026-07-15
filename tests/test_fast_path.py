import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from reachy_mini_openclaw.control_server import ClawBodyControlServer
from reachy_mini_openclaw.openclaw_bridge import (
    OpenClawBridge,
    OpenClawContinuityError,
)
from reachy_mini_openclaw.openai_realtime import build_direct_voice_instructions


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
