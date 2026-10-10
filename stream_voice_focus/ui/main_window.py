"""PySide6 desktop UI for Stream Voice Focus."""
from __future__ import annotations

import math
import os
import sys
import traceback
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QObject, QUrl
from PySide6.QtGui import QAction, QIcon, QDesktopServices, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton, QSlider,
                               QVBoxLayout, QWidget, QSizePolicy, QFileDialog, QFrame, QDialog, QTextBrowser,
                               QDialogButtonBox)

from .. import devices as dv
from ..engine import AudioEngine, EngineConfig, EngineStats
from ..models.registry import MODEL_INFOS, load_model, model_info
from ..settings import Settings

NO_SELECTION = "— select a device —"
VB_CABLE_URL = "https://vb-audio.com/Cable/"
ACCENT = "#9146ff"

STYLE = """
QMainWindow, QDialog { background: #14121b; }
QWidget { color: #ececf1; font-size: 13px; }
QGroupBox { background: #1e1b29; border: 1px solid #2d2940; border-radius: 10px; margin-top: 14px; padding: 12px 10px 8px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #b9b4d0; font-weight: 600; }
QComboBox, QLineEdit { background: #262236; border: 1px solid #3a3550; border-radius: 6px; padding: 5px 8px; min-height: 22px; }
QComboBox:disabled { color: #8a86a0; }
QComboBox QAbstractItemView { background: #262236; selection-background-color: #9146ff; border: 1px solid #3a3550; }
QPushButton { background: #2b2740; border: 1px solid #3f3a58; border-radius: 7px; padding: 6px 14px; }
QPushButton:hover { background: #353052; }
QPushButton:disabled { color: #7d7892; }
QPushButton#start { background: #9146ff; border: none; color: white; font-weight: 700; font-size: 15px; border-radius: 9px; }
QPushButton#start:hover { background: #a060ff; }
QPushButton#start[running="true"] { background: #d9534f; }
QPushButton#start[running="true"]:hover { background: #e36460; }
QPushButton#link { background: transparent; border: 1px solid #9146ff; color: #c9b3ff; }
QSlider::groove:horizontal { height: 6px; background: #2b2740; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #9146ff; border-radius: 3px; }
QSlider::handle:horizontal { background: #ffffff; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
QProgressBar { background: #262236; border: 1px solid #3a3550; border-radius: 5px; text-align: center; color: #ececf1; }
QProgressBar::chunk { border-radius: 4px; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; background: #262236; border: 1px solid #5a5478; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #9146ff; border: 1px solid #c9b3ff; }
QRadioButton::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #ffffff, stop:0.45 #ffffff, stop:0.55 #9146ff, stop:1 #9146ff); }
QLabel#title { font-size: 20px; font-weight: 700; color: white; }
QLabel#subtitle { color: #b9b4d0; }
QLabel#pill { border-radius: 11px; padding: 3px 12px; font-weight: 600; background: #2b2740; color: #c9c4dc; }
QLabel#pill[state="running"] { background: #1f6b3a; color: #c8ffd8; }
QLabel#pill[state="error"] { background: #7a2b2b; color: #ffd0d0; }
QLabel#pill[state="loading"] { background: #5a4a1f; color: #ffe9b0; }
QLabel#hint { color: #8f8aa6; }
QLabel#statval { color: #ffffff; }
QStatusBar { background: #1a1725; color: #b9b4d0; }
QMenuBar { background: #14121b; } QMenuBar::item:selected { background: #2b2740; }
QMenu { background: #262236; border: 1px solid #3a3550; } QMenu::item:selected { background: #9146ff; }
QToolTip { background: #262236; color: #ececf1; border: 1px solid #3a3550; }
"""


def asset_path(name: str) -> str:
    if getattr(sys, "frozen", False):
        base = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "assets")
    else:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
    return os.path.join(base, name)


class ModelLoader(QThread):
    done = Signal(object, str)   # model or None, error text

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def run(self):
        try:
            m = load_model(self.key)
            self.done.emit(m, "")
        except Exception as e:  # noqa
            self.done.emit(None, f"{e}\n\n{traceback.format_exc()}")


class EngineBridge(QObject):
    event = Signal(str, str)


def dbfs(x: float) -> float:
    return 20 * math.log10(max(x, 1e-6))


class LevelMeter(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setRange(-60, 0)
        self.setValue(-60)
        self.setTextVisible(True)
        self.setFormat("%v dBFS")
        self.setFixedHeight(20)
        self._color = ""

    def set_level(self, lin: float):
        v = int(max(-60, min(0, dbfs(lin))))
        self.setValue(v)
        color = "#d9534f" if v > -1 else ("#f0ad4e" if v > -6 else "#5cb85c")
        if color != self._color:
            self._color = color
            self.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")


SETUP_HTML = """
<h2 style="color:#c9b3ff">Setup in 4 steps</h2>
<ol>
<li><b>Install VB-Cable</b> (free): <a href="https://vb-audio.com/Cable/">vb-audio.com/Cable</a> → run the installer
<i>as administrator</i> → reboot. This app only detects it; it never installs drivers.</li>
<li><b>Route the browser to the cable.</b> Windows Settings → System → Sound → <b>Volume mixer</b>
(Windows 10: "App volume and device preferences"). Find your browser and set its <b>Output</b> to
<b>CABLE Input (VB-Audio Virtual Cable)</b>. Reload the Twitch tab. Every tab of that browser is now captured;
keep other things in a different browser or profile.</li>
<li><b>In this app:</b> Input = <b>CABLE Output</b> (auto-selected), Output = your <b>headphones</b>
(use the "Headphones (…)" entry, not "Headset (…)" which is the low-quality hands-free profile). Press <b>Start</b>.</li>
<li><b>Check:</b> the Input meter moves when the stream plays. Use <b>Listen to: Original / Filtered</b> to compare;
both are time-aligned so switching does not jump.</li>
</ol>
<p><b>Hearing original + filtered at once?</b> The browser is still also playing to the headphones; redo step 2.</p>
<p><b>Audio behind the video?</b> That is the added latency (see the latency rows). Choose DeepFilterNet3 (30 ms model delay)
and wired headphones for the smallest delay; Bluetooth adds 100–250 ms by itself.</p>
<p><b>Never</b> select CABLE Input as this app's output: that would feed the processed audio back into the capture (loop).
The app refuses that combination.</p>
"""


class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup guide")
        self.resize(640, 520)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(SETUP_HTML)
        tb.setStyleSheet("QTextBrowser { background: #1e1b29; border: 1px solid #2d2940; border-radius: 8px; padding: 8px; }")
        lay.addWidget(tb)
        hb = QHBoxLayout()
        b1 = QPushButton("Open Windows sound settings")
        b1.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("ms-settings:apps-volume")))
        b2 = QPushButton("Get VB-Cable")
        b2.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(VB_CABLE_URL)))
        hb.addWidget(b1)
        hb.addWidget(b2)
        hb.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        hb.addWidget(bb)
        lay.addLayout(hb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stream Voice Focus")
        ico = asset_path("icon.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QIcon(ico))
        self.settings = Settings.load()
        self.engine = AudioEngine(on_event=self._engine_event_threadsafe)
        self.bridge = EngineBridge()
        self.bridge.event.connect(self._on_engine_event)
        self.model = None
        self.model_key_loaded = ""
        self.loader: Optional[ModelLoader] = None
        self.devices: List[dv.DeviceInfo] = []
        self._build_ui()
        self.refresh_devices(initial=True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        if self.settings.window_geometry:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromHex(self.settings.window_geometry.encode()))
            except Exception:
                pass

    # ------------------------------------------------------------------ UI
    def _set_pill(self, text: str, state: str):
        self.pill.setText(text)
        self.pill.setProperty("state", state)
        self.pill.style().unpolish(self.pill)
        self.pill.style().polish(self.pill)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 10, 14, 10)

        # Header
        hdr = QHBoxLayout()
        logo = QLabel()
        png = asset_path("icon.png")
        if os.path.exists(png):
            logo.setPixmap(QPixmap(png).scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        hdr.addWidget(logo)
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        t = QLabel("Stream Voice Focus")
        t.setObjectName("title")
        st = QLabel("Keeps the streamer's voice, reduces music and game sounds — in real time, on your PC.")
        st.setObjectName("subtitle")
        tcol.addWidget(t)
        tcol.addWidget(st)
        hdr.addLayout(tcol, 1)
        self.pill = QLabel("Idle")
        self.pill.setObjectName("pill")
        hdr.addWidget(self.pill)
        self.btn_guide = QPushButton("Setup guide")
        self.btn_guide.setObjectName("link")
        self.btn_guide.clicked.connect(lambda: SetupDialog(self).exec())
        hdr.addWidget(self.btn_guide)
        lay.addLayout(hdr)

        # Devices
        g = QGroupBox("1 · Audio routing     browser → CABLE Input   |   app ← CABLE Output   |   app → headphones")
        f = QGridLayout(g)
        f.setVerticalSpacing(8)
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.in_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.out_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_refresh = QPushButton("Refresh devices")
        self.btn_refresh.clicked.connect(lambda: self.refresh_devices())
        self.btn_sound = QPushButton("Open Windows sound settings")
        self.btn_sound.setToolTip("Volume mixer: set your browser's output to 'CABLE Input'")
        self.btn_sound.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("ms-settings:apps-volume")))
        self.chk_all = QCheckBox("Show all host APIs")
        self.chk_all.setChecked(self.settings.show_all_hostapis)
        self.chk_all.toggled.connect(lambda _: self.refresh_devices())
        self.lbl_cable = QLabel("")
        self.lbl_cable.setObjectName("hint")
        self.lbl_cable.setWordWrap(True)
        self.lbl_cable.setOpenExternalLinks(True)
        f.addWidget(QLabel("Input (capture):"), 0, 0)
        f.addWidget(self.in_combo, 0, 1)
        f.addWidget(QLabel("Output (playback):"), 1, 0)
        f.addWidget(self.out_combo, 1, 1)
        hb = QHBoxLayout()
        hb.addWidget(self.btn_refresh)
        hb.addWidget(self.btn_sound)
        hb.addWidget(self.chk_all)
        hb.addStretch(1)
        f.addLayout(hb, 2, 0, 1, 2)
        f.addWidget(self.lbl_cable, 3, 0, 1, 2)
        lay.addWidget(g)

        # Model
        g = QGroupBox("2 · Model")
        f = QGridLayout(g)
        f.setVerticalSpacing(8)
        self.model_combo = QComboBox()
        for mi in MODEL_INFOS:
            self.model_combo.addItem(mi.display_name, mi.key)
        idx = self.model_combo.findData(self.settings.model_key)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.lbl_model = QLabel("")
        self.lbl_model.setObjectName("hint")
        self.lbl_model.setWordWrap(True)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("start")
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setMinimumWidth(170)
        self.btn_start.clicked.connect(self.toggle_start)
        f.addWidget(QLabel("Model:"), 0, 0)
        f.addWidget(self.model_combo, 0, 1)
        f.addWidget(self.btn_start, 0, 2, 2, 1)
        f.addWidget(self.lbl_model, 1, 0, 1, 2)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)
        self._model_changed()

        # Controls
        g = QGroupBox("3 · Controls")
        f = QGridLayout(g)
        f.setVerticalSpacing(8)
        self.sl_red = QSlider(Qt.Horizontal)
        self.sl_red.setRange(0, 100)
        self.sl_red.setValue(int(round(self.settings.reduction * 100)))
        self.sl_red.valueChanged.connect(self._reduction_changed)
        self.lbl_red = QLabel("")
        self.lbl_red.setMinimumWidth(230)
        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(0, 150)
        self.sl_vol.setValue(int(round(self.settings.output_gain * 100)))
        self.sl_vol.valueChanged.connect(self._gain_changed)
        self.lbl_vol = QLabel("")
        self.rb_filtered = QRadioButton("Filtered")
        self.rb_original = QRadioButton("Original (latency-matched)")
        self.rb_filtered.setChecked(True)
        self.rb_filtered.toggled.connect(self._bypass_changed)
        f.addWidget(QLabel("Background reduction:"), 0, 0)
        f.addWidget(self.sl_red, 0, 1)
        f.addWidget(self.lbl_red, 0, 2)
        f.addWidget(QLabel("Output volume:"), 1, 0)
        f.addWidget(self.sl_vol, 1, 1)
        f.addWidget(self.lbl_vol, 1, 2)
        f.addWidget(QLabel("Listen to:"), 2, 0)
        hb = QHBoxLayout()
        hb.addWidget(self.rb_filtered)
        hb.addWidget(self.rb_original)
        hb.addStretch(1)
        f.addLayout(hb, 2, 1, 1, 2)
        note = QLabel("Reduction blends the model output with the original delayed by exactly the model's "
                      "algorithmic delay (sample-aligned, no echo). It is a blend, not a model strength parameter.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        f.addWidget(note, 3, 0, 1, 3)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)
        self._reduction_changed(self.sl_red.value())
        self._gain_changed(self.sl_vol.value())

        # Meters
        g = QGroupBox("Levels")
        f = QGridLayout(g)
        self.m_in = LevelMeter()
        self.m_out = LevelMeter()
        f.addWidget(QLabel("Input:"), 0, 0)
        f.addWidget(self.m_in, 0, 1)
        f.addWidget(QLabel("Output:"), 1, 0)
        f.addWidget(self.m_out, 1, 1)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)

        # Stats
        g = QGroupBox("Performance and latency")
        f = QGridLayout(g)
        f.setVerticalSpacing(6)
        self.stat_labels = {}
        rows = [
            ("infer", "Inference time (avg / p95 / max per 10 ms frame)"),
            ("cpu", "Worker load (inference time / frame time)"),
            ("buf", "Buffers: input backlog / output queue / underruns / overflows"),
            ("drops", "Recovery: backlog drops / output drops / drift slips (samples)"),
            ("algo", "Model algorithmic delay"),
            ("devlat", "Device latency reported by PortAudio (capture / playback)"),
            ("est", "Estimated total added latency (sum of the above, conservative)"),
            ("meas", "Measured in-app latency (device ADC→DAC timestamps)"),
            ("frames", "Frames processed / uptime"),
        ]
        for i, (k, title) in enumerate(rows):
            t = QLabel(title + ":")
            t.setObjectName("hint")
            t.setMinimumHeight(20)
            f.addWidget(t, i, 0)
            lab = QLabel("–")
            lab.setObjectName("statval")
            lab.setMinimumHeight(20)
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.stat_labels[k] = lab
            f.addWidget(lab, i, 1)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)
        lay.addStretch(1)

        self.status = self.statusBar()
        self.status.showMessage("Idle. Route your browser to 'CABLE Input' (Setup guide), choose your headphones, press Start.")

        menu = self.menuBar().addMenu("Tools")
        act = QAction("Process a WAV file (test mode)…", self)
        act.triggered.connect(self._file_mode)
        menu.addAction(act)
        act2 = QAction("Setup guide", self)
        act2.triggered.connect(lambda: SetupDialog(self).exec())
        menu.addAction(act2)
        act3 = QAction("Get VB-Cable (official site)", self)
        act3.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(VB_CABLE_URL)))
        menu.addAction(act3)

    # ------------------------------------------------------------------ devices
    def refresh_devices(self, initial: bool = False):
        hostapi = None if self.chk_all.isChecked() else dv.WASAPI
        try:
            self.devices = dv.list_devices(hostapi, refresh=not self.engine.running)
        except Exception as e:
            QMessageBox.critical(self, "Device error", str(e))
            return
        ins = [d for d in self.devices if d.is_input]
        outs = [d for d in self.devices if d.is_output]
        cur_in = self.in_combo.currentData() or self.settings.input_ident
        cur_out = self.out_combo.currentData() or self.settings.output_ident

        def fill(combo, items, want, fallback):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(NO_SELECTION, "")
            for d in items:
                combo.addItem(f"{d.name}   [{d.hostapi}]", d.ident)
            found = combo.findData(want) if want else -1
            missing = bool(want) and found < 0
            if found < 0 and fallback is not None and not want:
                found = combo.findData(fallback.ident)
            combo.setCurrentIndex(found if found >= 0 else 0)
            combo.blockSignals(False)
            return missing

        miss_in = fill(self.in_combo, ins, cur_in, dv.default_input_device(self.devices))
        miss_out = fill(self.out_combo, outs, cur_out, dv.default_output_device(self.devices))
        cap, play = dv.find_vb_cable(self.devices)
        if cap and play:
            self.lbl_cable.setText(f"✔ VB-Cable found. Capture: '{cap.name}'. Route only the browser to '{play.name}' "
                                   "(Open Windows sound settings → Volume mixer → browser → Output).")
        else:
            self.lbl_cable.setText("<b style='color:#ffb3b3'>VB-Cable not found.</b> " + dv.vb_cable_missing_message().replace(
                VB_CABLE_URL, f"<a style='color:#c9b3ff' href='{VB_CABLE_URL}'>{VB_CABLE_URL}</a>"))
            if initial:
                box = QMessageBox(self)
                box.setWindowTitle("VB-Cable not installed")
                box.setIcon(QMessageBox.Information)
                box.setText("VB-Audio Virtual Cable was not found on this PC.\n\nIt is the free virtual audio cable that "
                            "carries the browser's sound into this app. Install it from the official site (run the installer "
                            "as administrator, then reboot). This app never installs drivers itself.")
                b = box.addButton("Open vb-audio.com/Cable", QMessageBox.AcceptRole)
                box.addButton("Later", QMessageBox.RejectRole)
                box.exec()
                if box.clickedButton() is b:
                    QDesktopServices.openUrl(QUrl(VB_CABLE_URL))
        if initial and (miss_in or miss_out):
            what = []
            if miss_in:
                what.append(f"input '{cur_in.split('|')[-1]}'")
            if miss_out:
                what.append(f"output '{cur_out.split('|')[-1]}'")
            QMessageBox.warning(self, "Saved device missing",
                                "The saved " + " and ".join(what) + " is not available. Please choose a replacement. "
                                "Nothing was selected automatically to avoid routing audio to the wrong endpoint.")

    def _selected(self, combo) -> Optional[dv.DeviceInfo]:
        ident = combo.currentData()
        return dv.find_by_ident(ident, self.devices) if ident else None

    # ------------------------------------------------------------------ controls
    def _model_changed(self):
        key = self.model_combo.currentData()
        mi = model_info(key)
        self.lbl_model.setText(f"{mi.description}  Algorithmic delay: {1000 * mi.algorithmic_delay / mi.sample_rate:.0f} ms. "
                               f"License: {mi.license}")
        if self.engine.running:
            self.status.showMessage("Model change takes effect after Stop / Start.")

    def _reduction_changed(self, v: int):
        r = v / 100.0
        if r >= 0.999:
            txt = "100 % (full model output)"
        elif r <= 0.001:
            txt = "0 % (original)"
        else:
            txt = f"{v} %  (background limited to about -{-20 * math.log10(1 - r):.0f} dB)"
        self.lbl_red.setText(txt)
        self.engine.set_reduction(r)

    def _gain_changed(self, v: int):
        self.lbl_vol.setText(f"{v} %  ({20 * math.log10(max(v, 1) / 100):+.1f} dB)")
        self.engine.set_gain(v / 100.0)

    def _bypass_changed(self, _):
        self.engine.set_bypass(self.rb_original.isChecked())

    # ------------------------------------------------------------------ start/stop
    def _set_running_style(self, running: bool):
        self.btn_start.setProperty("running", "true" if running else "false")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)

    def toggle_start(self):
        if self.engine.running or self.loader is not None:
            self.stop()
        else:
            self.start()

    def start(self):
        inp, out = self._selected(self.in_combo), self._selected(self.out_combo)
        issues = dv.check_routing(inp, out)
        errs = [i.message for i in issues if i.level == "error"]
        warns = [i.message for i in issues if i.level == "warning"]
        if errs:
            QMessageBox.critical(self, "Cannot start", "\n\n".join(errs))
            return
        if warns:
            r = QMessageBox.question(self, "Routing warning", "\n\n".join(warns) + "\n\nStart anyway?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        key = self.model_combo.currentData()
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Loading model…")
        self._set_pill("Loading model", "loading")
        self.status.showMessage(f"Loading model {key} (first time downloads the weights)…")
        if self.model is not None and self.model_key_loaded == key:
            self._start_engine(self.model)
            return
        self.loader = ModelLoader(key)
        self.loader.done.connect(self._model_loaded)
        self.loader.start()

    def _model_loaded(self, model, err):
        self.loader = None
        if model is None:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("▶  Start")
            self._set_pill("Error", "error")
            self.status.showMessage("Model failed to load.")
            QMessageBox.critical(self, "Model error",
                                 "The model could not be loaded. The first start needs an internet connection to "
                                 "download the model weights (about 30 MB); afterwards it works offline.\n\n" + err)
            return
        if self.model is not None and self.model is not model:
            try:
                self.model.close()
            except Exception:
                pass
        self.model = model
        self.model_key_loaded = model.info.key
        self._start_engine(model)

    def _start_engine(self, model):
        inp, out = self._selected(self.in_combo), self._selected(self.out_combo)
        cfg = EngineConfig(input_device=inp, output_device=out, reduction=self.sl_red.value() / 100.0,
                           output_gain=self.sl_vol.value() / 100.0, bypass=self.rb_original.isChecked(),
                           prefill_ms=self.settings.prefill_ms)
        try:
            self.engine.start(cfg, model)
        except Exception as e:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("▶  Start")
            self._set_pill("Error", "error")
            self.status.showMessage("Start failed.")
            QMessageBox.critical(self, "Cannot start audio",
                                 f"{e}\n\nTips: make sure the device is not in exclusive use by another app, "
                                 "try 'Refresh devices', or choose another endpoint.")
            return
        self.btn_start.setEnabled(True)
        self.btn_start.setText("■  Stop")
        self._set_running_style(True)
        self._set_pill("Running", "running")
        self.in_combo.setEnabled(False)
        self.out_combo.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.status.showMessage(f"Running: {inp.name} → {model.info.display_name} → {out.name}")
        self._save_settings()

    def stop(self):
        self.engine.stop()
        self.btn_start.setText("▶  Start")
        self.btn_start.setEnabled(True)
        self._set_running_style(False)
        self._set_pill("Idle", "idle")
        self.in_combo.setEnabled(True)
        self.out_combo.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.status.showMessage("Stopped.")

    # ------------------------------------------------------------------ events / stats
    def _engine_event_threadsafe(self, kind: str, msg: str):
        self.bridge.event.emit(kind, msg)

    def _on_engine_event(self, kind: str, msg: str):
        if kind == "error":
            self.stop()
            self._set_pill("Error", "error")
            self.status.showMessage("Error: " + msg)
            QMessageBox.critical(self, "Audio engine stopped", msg + "\n\nPress 'Refresh devices', check the selection and start again.")

    def _tick(self):
        s: EngineStats = self.engine.stats()
        self.m_in.set_level(s.in_peak)
        self.m_out.set_level(s.out_peak)
        L = self.stat_labels
        if not s.running:
            if self.engine.last_error == "" and s.frames_processed == 0:
                for k in L:
                    L[k].setText("–")
            return
        L["infer"].setText(f"{s.infer_ms_avg:.2f} / {s.infer_ms_p95:.2f} / {s.infer_ms_max:.2f} ms")
        L["cpu"].setText(f"{100 * s.cpu_load:.0f} %   (above ~80 % expect drops; choose a lighter model)")
        L["buf"].setText(f"{s.backlog_ms:.1f} ms / {s.out_fill_ms:.1f} ms / {s.underruns} / {s.in_overflows}")
        L["drops"].setText(f"{s.backlog_drops} / {s.out_drops} / {s.slips:+d}")
        L["algo"].setText(f"{s.algo_delay_ms:.0f} ms (fixed by the model)")
        L["devlat"].setText(f"{s.input_latency_ms:.0f} ms / {s.output_latency_ms:.0f} ms")
        L["est"].setText(f"≈ {s.est_total_ms:.0f} ms  = capture {s.input_latency_ms:.0f} + block wait {s.backlog_ms:.0f} + "
                         f"inference {s.infer_ms_avg:.0f} + model {s.algo_delay_ms:.0f} + queue {s.out_fill_ms:.0f} + playback {s.output_latency_ms:.0f}")
        if s.measured_latency_ms is not None:
            L["meas"].setText(f"≈ {s.measured_latency_ms:.0f} ms   ({s.measured_latency_note}; not an acoustic measurement, "
                              "Bluetooth/driver output buffering may add more)")
        else:
            L["meas"].setText(f"unavailable ({s.measured_latency_note})")
        L["frames"].setText(f"{s.frames_processed} / {s.uptime_s:.0f} s")
        if s.in_peak < 0.001 and s.uptime_s > 5:
            self.status.showMessage("Running, but no input signal: is the browser routed to 'CABLE Input' and playing?")
        elif s.backlog_drops and s.cpu_load > 0.8:
            self.status.showMessage("Inference is too slow for real time: audio is being dropped. Choose a lighter model.")

    # ------------------------------------------------------------------ misc
    def _file_mode(self):
        src, _ = QFileDialog.getOpenFileName(self, "Input WAV", "", "Audio (*.wav *.flac *.ogg)")
        if not src:
            return
        dst, _ = QFileDialog.getSaveFileName(self, "Output WAV", src.rsplit(".", 1)[0] + "_filtered.wav", "WAV (*.wav)")
        if not dst:
            return
        from ..file_mode import main as fm
        try:
            fm([src, dst, "--model", self.model_combo.currentData(), "--reduction", str(self.sl_red.value() / 100.0), "--keep-rate"])
            QMessageBox.information(self, "Done", f"Wrote {dst}")
        except Exception as e:
            QMessageBox.critical(self, "File mode failed", str(e))

    def _save_settings(self):
        st = self.settings
        st.input_ident = self.in_combo.currentData() or ""
        st.output_ident = self.out_combo.currentData() or ""
        st.model_key = self.model_combo.currentData()
        st.reduction = self.sl_red.value() / 100.0
        st.output_gain = self.sl_vol.value() / 100.0
        st.show_all_hostapis = self.chk_all.isChecked()
        try:
            st.window_geometry = bytes(self.saveGeometry().toHex()).decode()
        except Exception:
            pass
        try:
            st.save()
        except Exception:
            pass

    def closeEvent(self, ev):
        self.engine.stop()
        self._save_settings()
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                pass
        super().closeEvent(ev)


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    ico = asset_path("icon.ico")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Stream Voice Focus")
    apply_theme(app)
    w = MainWindow()
    w.resize(900, 860)
    w.show()
    sys.exit(app.exec())
