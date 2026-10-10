"""Persisted user settings (JSON in %APPDATA%\\StreamVoiceFocus\\settings.json).

Devices are stored by `DeviceInfo.ident` (host API + full endpoint name). PortAudio does not
expose Windows endpoint GUIDs, and device *indices* change whenever devices are added or
removed, so names are the most stable identifier available. If a saved device is missing at
startup the UI asks for a replacement instead of silently picking another endpoint.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field


def settings_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "StreamVoiceFocus")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "settings.json")


@dataclass
class Settings:
    input_ident: str = ""
    output_ident: str = ""
    model_key: str = "dpdfnet2_48k"
    reduction: float = 1.0
    output_gain: float = 1.0
    show_all_hostapis: bool = False
    prefill_ms: float = 20.0
    window_geometry: str = ""

    @classmethod
    def load(cls) -> "Settings":
        p = settings_path()
        s = cls()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except FileNotFoundError:
            pass
        except Exception:
            pass  # corrupt file: fall back to defaults
        return s

    def save(self) -> None:
        p = settings_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, p)
