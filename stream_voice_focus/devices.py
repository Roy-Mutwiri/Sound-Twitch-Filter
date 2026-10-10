"""Audio endpoint enumeration, VB-Cable detection and feedback-loop checks.

Terminology (VB-Cable): "CABLE Input" is the *playback* endpoint applications play into;
"CABLE Output" is the *recording* endpoint that returns the same audio. This app captures
from CABLE Output and must never play into CABLE Input (that would feed its own output back).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import sounddevice as sd

WASAPI = "Windows WASAPI"

VIRTUAL_PLAYBACK_HINTS = ("CABLE Input", "CABLE In", "Voicemeeter Input", "Voicemeeter AUX Input",
                          "Voicemeeter VAIO3 Input", "Voicemeeter In ", "VB-Audio Point")
VIRTUAL_CAPTURE_HINTS = ("CABLE Output", "Voicemeeter Out", "VB-Audio Point")
MIC_HINTS = ("Microphone", "Mic ", "Headset", "Hands-Free", "Line In", "Stereo Mix", "WebCam")


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def ident(self) -> str:
        """Stable-ish identifier for settings: host API + full endpoint name (PortAudio does
        not expose Windows endpoint GUIDs; WASAPI names are the full endpoint names)."""
        return f"{self.hostapi}|{self.name}"

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


def _hostapi_name(idx: int) -> str:
    return sd.query_hostapis(idx)["name"]


def list_devices(hostapi: Optional[str] = WASAPI, refresh: bool = True) -> List[DeviceInfo]:
    if refresh:
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
    out = []
    for i, d in enumerate(sd.query_devices()):
        ha = _hostapi_name(d["hostapi"])
        if hostapi and ha != hostapi:
            continue
        out.append(DeviceInfo(i, d["name"], ha, d["max_input_channels"], d["max_output_channels"], d["default_samplerate"]))
    return out


def input_devices(hostapi: Optional[str] = WASAPI, refresh: bool = True) -> List[DeviceInfo]:
    return [d for d in list_devices(hostapi, refresh) if d.is_input]


def output_devices(hostapi: Optional[str] = WASAPI, refresh: bool = True) -> List[DeviceInfo]:
    return [d for d in list_devices(hostapi, refresh) if d.is_output]


def find_by_ident(ident: str, devices: List[DeviceInfo]) -> Optional[DeviceInfo]:
    for d in devices:
        if d.ident == ident:
            return d
    return None


def is_vb_cable_capture(name: str) -> bool:
    return name.startswith("CABLE Output")


def is_vb_cable_playback(name: str) -> bool:
    return name.startswith("CABLE Input") or name.startswith("CABLE In ")


def is_virtual_playback(name: str) -> bool:
    return any(h in name for h in VIRTUAL_PLAYBACK_HINTS)


def is_virtual_capture(name: str) -> bool:
    return any(h in name for h in VIRTUAL_CAPTURE_HINTS)


def looks_like_microphone(name: str) -> bool:
    return any(h in name for h in MIC_HINTS)


def find_vb_cable(devices: Optional[List[DeviceInfo]] = None) -> Tuple[Optional[DeviceInfo], Optional[DeviceInfo]]:
    """Returns (capture endpoint 'CABLE Output', playback endpoint 'CABLE Input') or Nones."""
    devices = devices if devices is not None else list_devices()
    cap = next((d for d in devices if d.is_input and is_vb_cable_capture(d.name)), None)
    play = next((d for d in devices if d.is_output and is_vb_cable_playback(d.name)), None)
    return cap, play


def vb_cable_missing_message() -> str:
    return ("VB-Audio Virtual Cable was not found. Install it from the official source "
            "https://vb-audio.com/Cable/ (run the installer as administrator, then reboot). "
            "This app does not install drivers itself.")


def default_output_device(devices: List[DeviceInfo]) -> Optional[DeviceInfo]:
    """Windows default playback endpoint, unless it is a virtual cable (unsafe)."""
    try:
        idx = sd.default.device[1]
        name = sd.query_devices(idx)["name"] if idx is not None and idx >= 0 else None
    except Exception:
        name = None
    cands = [d for d in devices if d.is_output and not is_virtual_playback(d.name)]
    if name:
        for d in cands:
            if d.name == name:
                return d
    # fall back to something that looks like headphones/speakers
    for hint in ("Headphones", "Headset", "Speakers"):
        for d in cands:
            if hint in d.name:
                return d
    return cands[0] if cands else None


def default_input_device(devices: List[DeviceInfo]) -> Optional[DeviceInfo]:
    cap, _ = find_vb_cable(devices)
    return cap  # never a microphone by default


@dataclass
class RoutingIssue:
    level: str   # "error" | "warning"
    message: str


def check_routing(inp: Optional[DeviceInfo], out: Optional[DeviceInfo]) -> List[RoutingIssue]:
    issues: List[RoutingIssue] = []
    if inp is None or out is None:
        issues.append(RoutingIssue("error", "Select both an input and an output device."))
        return issues
    if is_vb_cable_capture(inp.name) and is_vb_cable_playback(out.name):
        issues.append(RoutingIssue("error",
            "Feedback loop: input is CABLE Output and output is CABLE Input (the same virtual cable). "
            "Processed audio would be captured again and fed back endlessly. Choose your headphones as output."))
    if inp.name.split(" (")[0] == out.name.split(" (")[0] and inp.index != out.index:
        issues.append(RoutingIssue("warning", "Input and output appear to be the same device."))
    if is_virtual_playback(out.name) and not is_vb_cable_playback(out.name):
        issues.append(RoutingIssue("warning",
            f"Output '{out.name}' is a virtual device. Make sure nothing routes it back into '{inp.name}', "
            "or you will get a feedback loop. Normally the output should be your physical headphones."))
    if is_vb_cable_playback(out.name):
        issues.append(RoutingIssue("error",
            f"Output '{out.name}' is the VB-Cable playback endpoint the browser plays into. Never output there."))
    if looks_like_microphone(inp.name) and not is_virtual_capture(inp.name):
        issues.append(RoutingIssue("warning",
            f"Input '{inp.name}' looks like a microphone. The intended input is 'CABLE Output' "
            "(the browser's audio via VB-Cable); a microphone would process room sound instead."))
    if not is_vb_cable_capture(inp.name) and not is_virtual_capture(inp.name):
        issues.append(RoutingIssue("warning",
            "Input is not a virtual cable endpoint. If this is a loopback of your speakers you will "
            "hear the original and the processed audio at the same time."))
    return issues


def wasapi_settings(hostapi: str):
    """Let the Windows audio engine convert sample rate / format so streams can always open at 48 kHz."""
    if hostapi == WASAPI:
        return sd.WasapiSettings(auto_convert=True)
    return None
