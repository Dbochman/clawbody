"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-realtime-2.1-mini"))
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))
    OPENAI_TTS_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "tts-1"))
    OPENAI_TTS_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "onyx"))
    OPENAI_TRANSCRIPTION_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    )
    OPENAI_TRANSCRIPTION_LANGUAGE: str = field(
        default_factory=lambda: os.getenv("OPENAI_TRANSCRIPTION_LANGUAGE", "en")
    )
    OPENAI_VAD_SILENCE_MS: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_VAD_SILENCE_MS", "400"))
    )
    OPENAI_AUDIO_JITTER_MS: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_AUDIO_JITTER_MS", "220"))
    )
    REACHY_VOICE_MODE: str = field(
        default_factory=lambda: os.getenv("REACHY_VOICE_MODE", "direct").lower()
    )
    CONTINUITY_REFRESH_SECONDS: float = field(
        default_factory=lambda: float(os.getenv("CONTINUITY_REFRESH_SECONDS", "2"))
    )
    CONTINUITY_SUMMARY_MODEL: str = field(
        default_factory=lambda: os.getenv("CONTINUITY_SUMMARY_MODEL", "gpt-5.4-mini")
    )
    
    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789"))
    OPENCLAW_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("OPENCLAW_TOKEN"))
    OPENCLAW_AGENT_ID: str = field(default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    OPENCLAW_SESSION_KEY: str = field(default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))
    OPENCLAW_THINKING_LEVEL: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_THINKING_LEVEL", "minimal")
    )
    OPENCLAW_FAST_MODE: bool = field(
        default_factory=lambda: os.getenv("OPENCLAW_FAST_MODE", "true").lower() == "true"
    )
    OPENCLAW_STREAM_SETTLE_MS: int = field(
        default_factory=lambda: int(os.getenv("OPENCLAW_STREAM_SETTLE_MS", "350"))
    )
    OPENCLAW_DELEGATION_TIMEOUT_SECONDS: float = field(
        default_factory=lambda: float(os.getenv("OPENCLAW_DELEGATION_TIMEOUT_SECONDS", "75"))
    )
    
    # Robot Configuration
    ROBOT_NAME: Optional[str] = field(default_factory=lambda: os.getenv("ROBOT_NAME"))
    
    # Feature Flags
    ENABLE_OPENCLAW_TOOLS: bool = field(default_factory=lambda: os.getenv("ENABLE_OPENCLAW_TOOLS", "true").lower() == "true")
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true")
    
    # Face Tracking Configuration
    # Options: "daemon" (Reachy Wireless 1.9+), "yolo", or "mediapipe"
    HEAD_TRACKER_TYPE: Optional[str] = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "yolo"))
    
    # Local Vision Processing
    ENABLE_LOCAL_VISION: bool = field(default_factory=lambda: os.getenv("ENABLE_LOCAL_VISION", "false").lower() == "true")
    LOCAL_VISION_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"))
    VISION_DEVICE: str = field(default_factory=lambda: os.getenv("VISION_DEVICE", "auto"))  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    
    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        if self.REACHY_VOICE_MODE not in {"direct", "openclaw"}:
            errors.append("REACHY_VOICE_MODE must be 'direct' or 'openclaw'")
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: Optional[str]) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_local_vision_enabled(enabled: bool) -> None:
    """Enable or disable local vision processing at runtime."""
    global config
    config.ENABLE_LOCAL_VISION = enabled
