import asyncio

import pytest

from reachy_mini_openclaw import config as config_module
from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge


@pytest.mark.asyncio
async def test_chat_applies_low_latency_turn_options() -> None:
    bridge = OpenClawBridge(gateway_url="ws://example.invalid")
    bridge._connected = True
    bridge._ws = object()

    async def send_request(method: str, params: dict, timeout: float | None = None) -> dict:
        assert method == "chat.send"
        assert params["thinking"] == "minimal"
        assert params["fastMode"] is True
        return {"ok": True, "payload": {}}

    bridge._send_request = send_request
    response = await bridge.chat("hello")

    assert response.error == "No runId in response"


@pytest.mark.asyncio
async def test_stream_chat_converts_cumulative_text_to_deltas() -> None:
    bridge = OpenClawBridge(gateway_url="ws://example.invalid")
    bridge._connected = True
    bridge._ws = object()

    async def send_request(method: str, params: dict, timeout: float | None = None) -> dict:
        assert method == "chat.send"
        assert params["thinking"] == "minimal"
        assert params["fastMode"] is True
        return {"ok": True, "payload": {"runId": "run-1"}}

    bridge._send_request = send_request
    stream = bridge.stream_chat("hello")

    first_delta = asyncio.create_task(anext(stream))
    while "run-1" not in bridge._run_events:
        await asyncio.sleep(0)
    event_queue = bridge._run_events["run-1"]
    await event_queue.put(
        {
            "event": "agent",
            "payload": {
                "runId": "run-1",
                "stream": "assistant",
                "data": {"text": "First sentence."},
            },
        }
    )
    assert await first_delta == "First sentence."

    second_delta = asyncio.create_task(anext(stream))
    await event_queue.put(
        {
            "event": "agent",
            "payload": {
                "runId": "run-1",
                "stream": "assistant",
                "data": {"text": "First sentence. Second sentence!"},
            },
        }
    )
    assert await second_delta == " Second sentence!"

    finished = asyncio.create_task(anext(stream))
    await event_queue.put(
        {
            "event": "agent",
            "payload": {
                "runId": "run-1",
                "stream": "lifecycle",
                "data": {"phase": "end"},
            },
        }
    )
    with pytest.raises(StopAsyncIteration):
        await finished


@pytest.mark.asyncio
async def test_stream_chat_settles_after_text_without_completion_event(monkeypatch) -> None:
    monkeypatch.setattr(config_module.config, "OPENCLAW_STREAM_SETTLE_MS", 10)
    bridge = OpenClawBridge(gateway_url="ws://example.invalid")
    bridge._connected = True
    bridge._ws = object()

    async def send_request(method: str, params: dict, timeout: float | None = None) -> dict:
        return {"ok": True, "payload": {"runId": "run-without-completion"}}

    bridge._send_request = send_request
    stream = bridge.stream_chat("hello")

    first_delta = asyncio.create_task(anext(stream))
    while "run-without-completion" not in bridge._run_events:
        await asyncio.sleep(0)
    await bridge._run_events["run-without-completion"].put(
        {
            "event": "agent",
            "payload": {
                "runId": "run-without-completion",
                "stream": "assistant",
                "data": {"text": "Complete response."},
            },
        }
    )
    assert await first_delta == "Complete response."

    with pytest.raises(StopAsyncIteration):
        await anext(stream)
