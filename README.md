---
title: ClawBody
emoji: 🦞
colorFrom: red
colorTo: purple
sdk: static
pinned: false
short_description: OpenClaw AI with robot body and face tracking
tags:
 - reachy_mini
 - reachy_mini_python_app
 - openclaw
 - clawson
 - embodied-ai
 - ai-assistant
 - voice-assistant
 - robotics
 - openai-realtime
 - conversational-ai
 - physical-ai
 - robot-body
 - speech-to-speech
 - multimodal
 - vision
 - expressive-robot
 - simulation
 - mujoco
 - face-tracking
 - face-detection
 - eye-contact
 - human-robot-interaction
---

# 🦞🤖 ClawBody

**Give your OpenClaw AI agent a physical robot body!**

ClawBody combines OpenClaw's skills and personality with Reachy Mini's expressive robot body. Ordinary conversation runs directly through OpenAI Realtime using a live SOUL/capsule snapshot; requests needing skills or external actions delegate to OpenClaw on demand.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 👁️ NEW: Face Tracking & Eye Contact

**The robot looks at you when you speak!**

ClawBody now includes real-time face tracking that makes conversations feel natural and engaging:

- **Automatic Face Detection**: Uses MediaPipe or YOLO to detect faces at 25Hz
- **Smooth Head Tracking**: Robot smoothly follows your face as you move
- **Natural Eye Contact**: Maintains engagement during conversation
- **Graceful Fallback**: Smoothly returns to neutral position when you leave

```bash
# Face tracking is enabled by default
clawbody

# Choose your tracker (MediaPipe is lighter, YOLO is more accurate)
clawbody --head-tracker mediapipe
clawbody --head-tracker yolo

# Disable if needed
clawbody --no-face-tracking
```

---

## 🎮 No Robot? No Problem!

**You don't need a physical Reachy Mini robot to use ClawBody!**

ClawBody works with the [Reachy Mini Simulator](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started), a MuJoCo-based physics simulation that runs on your computer. Watch Clawson move and express emotions on screen while you talk to your OpenClaw agent.

```bash
# Install simulator support
pip install "reachy-mini[mujoco]"

# Start the simulator (opens a 3D window)
reachy-mini-daemon --sim

# In another terminal, run ClawBody
clawbody --gradio
```

> 🍎 **Mac Users**: Use `mjpython -m reachy_mini.daemon.app.main --sim` instead.

---

## ✨ Features

- **👁️ Face Tracking**: Robot tracks your face and maintains eye contact during conversation
- **🎤 Real-time Voice Conversation**: OpenAI Realtime API for sub-second response latency
- **🧠 OpenClaw Skills on Demand**: Fast local voice turns delegate to the full OpenClaw agent whenever a request needs tools, memory, or external actions
- **🔗 Private Continuity**: SOUL and compact expiring capsule summaries keep Reachy and OpenClaw aligned without copying raw room transcripts
- **🎛️ Exclusive Control Lease**: OpenClaw can proactively see, speak, move, emote, or dance through Reachy without competing with the direct voice session
- **👀 Vision**: See through the robot's camera and describe the environment
- **💃 Expressive Movements**: Natural head movements, emotions, dances, and audio-driven wobble
- **🦞 Clawson Embodied**: Your friendly space lobster AI assistant, now with a body!
- **🖥️ Simulator Support**: Works with or without physical hardware

## 🏗️ Architecture

```
Voice → Reachy mic → persistent OpenAI Realtime session → Reachy speaker
                         │                         │
                         │ local robot tools       └─ look / emotion / dance
                         │
                         ├─ SOUL + capsule RPC (no agent/model turn)
                         │
                         └─ ask_openclaw, only when needed
                                      │
                                      ▼
                           OpenClaw skills and memory

OpenClaw proactive action → exclusive control lease → camera / motion / speech
```

## 📋 Prerequisites

### Option A: With Physical Robot
- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot (Wireless or Lite)

### Option B: With Simulator (No Hardware Required!)
- Any computer with Python 3.11+
- Install: `pip install "reachy-mini[mujoco]"`
- [Simulation Setup Guide](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)

### Software (Both Options)
- Python 3.11+
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini) installed
- [OpenClaw](https://github.com/openclaw/openclaw) gateway running
- OpenAI API key with Realtime API access

## 🚀 Installation

### Quick Start with Simulator

```bash
# Clone ClawBody
git clone https://github.com/tomrikert/clawbody
cd clawbody

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install ClawBody + simulator support + face tracking
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# Or for more accurate face tracking (requires more resources)
# pip install -e ".[yolo_vision]"

# Configure (see Configuration section)
cp .env.example .env
# Edit .env with your keys

# Terminal 1: Start the simulator
reachy-mini-daemon --sim

# Terminal 2: Run ClawBody
clawbody --gradio
```

### On a Physical Reachy Mini Robot

```bash
# SSH into the robot
ssh pollen@reachy-mini.local

# Clone the repository
git clone https://github.com/tomrikert/clawbody
cd clawbody

# Install in the apps virtual environment
/venvs/apps_venv/bin/pip install -e .
```

## ⚙️ Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:

```bash
# Required
OPENAI_API_KEY=sk-...your-key...

# OpenClaw Gateway (capsule/control RPC and delegated skill requests)
OPENCLAW_GATEWAY_URL=http://localhost:18789  # or your host IP
OPENCLAW_TOKEN=your-gateway-token
OPENCLAW_AGENT_ID=main

# Native direct voice fast path (set openclaw for compatibility fallback)
REACHY_VOICE_MODE=direct
OPENAI_MODEL=gpt-realtime-2.1-mini
OPENAI_VOICE=cedar
OPENAI_AUDIO_JITTER_MS=220

# Optional local wake-word gate (see the device install note below)
REACHY_WAKE_WORD_ENABLED=true
REACHY_WAKE_WORD_THRESHOLD=0.5
REACHY_WAKE_WORD_INITIAL_TIMEOUT_SECONDS=10
REACHY_WAKE_WORD_FOLLOWUP_TIMEOUT_SECONDS=20

# Optional - Face tracking (enabled by default)
ENABLE_FACE_TRACKING=true
HEAD_TRACKER_TYPE=mediapipe  # or "yolo" for more accuracy
```

### Local “Hey Claude” wake word

When `REACHY_WAKE_WORD_ENABLED=true`, ClawBody runs the bundled “Hey Claude”
openWakeWord model on Reachy. While sleeping, microphone frames stay on the
robot and are not forwarded to OpenAI. Detection opens a ten-second window in
which to begin speaking; once speech begins, the utterance can continue for any
length supported by the normal Realtime turn detector. After a reply, Reachy
remains engaged for 20 seconds so follow-ups and barge-in do not require another
wake phrase.

The face tracker and listening pose activate immediately when the wake phrase is
detected. `reachyctl status` reports `wake_word_state` as `sleeping`, `waiting`,
or `engaged`. Raise `REACHY_WAKE_WORD_THRESHOLD` to reduce false activations, or
lower it if intended wakes are missed.

Install the optional runtime with `pip install -e ".[wake_word]"` on platforms
where openWakeWord's full dependencies are available. Reachy Mini's Python 3.12
ARM environment uses only ONNX, while openWakeWord declares an unused
`tflite-runtime` dependency for all Linux systems that has no matching wheel.
Install the same pinned package there without that unused dependency:

```bash
/venvs/apps_venv/bin/pip install --no-deps openwakeword==0.6.0
/venvs/apps_venv/bin/pip install -e .
```

## 🎮 Usage

### With Simulator

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Run ClawBody with web UI (recommended for simulator)
clawbody --gradio
```

The simulator opens a 3D window where you can watch the robot move. The Gradio web UI at http://localhost:7860 lets you interact via your browser's microphone.

### With Physical Robot

```bash
# Basic usage
clawbody

# With debug logging
clawbody --debug

# With specific robot
clawbody --robot-name my-reachy
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug logging |
| `--gradio` | Launch web UI instead of console mode |
| `--robot-name NAME` | Specify robot name for connection |
| `--gateway-url URL` | OpenClaw gateway URL |
| `--no-camera` | Disable camera functionality |
| `--no-openclaw` | Disable OpenClaw integration |
| `--head-tracker TYPE` | Face tracker: `mediapipe` (lighter) or `yolo` (more accurate) |
| `--no-face-tracking` | Disable face tracking |

## 🛠️ Robot Capabilities

ClawBody gives Clawson these physical abilities:

| Capability | Description |
|------------|-------------|
| **Face Tracking** | Automatically tracks and looks at people during conversation |
| **Look** | Move head to look in directions (left, right, up, down) |
| **See** | Capture images through the robot's camera |
| **Dance** | Perform expressive dance animations |
| **Emotions** | Express emotions through movement (happy, curious, thinking, etc.) |
| **Speak** | Voice output through the robot's speaker |
| **Listen** | Hear through the robot's microphone |
| **Mute** | Mute its microphone on request; restore it through Reachy's UI or an authenticated remote control command |

## 🖥️ Simulator Features

When running with the simulator:

- **3D Visualization**: Watch Clawson's movements in real-time
- **Scene Options**: Use `--scene minimal` to add objects (apple, duck, croissant)
- **Full SDK Compatibility**: The simulator behaves exactly like a real robot
- **Dashboard Access**: Visit http://localhost:8000 to see the robot dashboard

```bash
# Start simulator with objects on a table
reachy-mini-daemon --sim --scene minimal
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ClawBody builds on:

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini robot, SDK, and simulator
- [OpenClaw](https://github.com/openclaw/openclaw) - AI assistant framework (Clawson!)
- [OpenAI](https://openai.com/) - Realtime API for voice I/O
- [MuJoCo](https://mujoco.org/) - Physics simulation engine
- [pollen-robotics/reachy_mini_conversation_app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app) - Movement and audio systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- **This project**: [GitHub Issues](https://github.com/tomrikert/clawbody/issues)
- **OpenClaw Skills**: Submit ClawBody as a skill to [ClawHub](https://docs.openclaw.ai/tools/clawhub)
- **Reachy Mini Apps**: Submit to [Pollen Robotics](https://github.com/pollen-robotics)
