"""Entry point: `python -m stream_voice_focus` starts the desktop UI.

Options:
  --screenshot PATH   render the window, save a PNG after ~2 s and exit (used for automated checks)
  --file IN OUT ...   file-processing test mode (same as `python -m stream_voice_focus.file_mode`)
"""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--file":
        from .file_mode import main as fm
        return fm(argv[1:])
    def take(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    shot = take("--screenshot")
    auto_out = take("--autostart-out")      # test automation: output device name substring
    auto_vol = float(take("--volume", "100"))
    after = float(take("--after", "2"))
    try:  # own taskbar identity so the window icon (not python's) shows in the taskbar
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RoyMutwiri.StreamVoiceFocus")
    except Exception:
        pass
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from .ui.main_window import MainWindow, apply_theme
    app = QApplication([sys.argv[0]] + argv)
    app.setApplicationName("Stream Voice Focus")
    apply_theme(app)
    w = MainWindow()
    w.resize(900, 860)
    w.show()
    if auto_out:
        for i in range(w.out_combo.count()):
            if auto_out.lower() in w.out_combo.itemText(i).lower():
                w.out_combo.setCurrentIndex(i)
                break
        w.sl_vol.setValue(int(auto_vol))
        QTimer.singleShot(300, w.start)
    if shot:
        def grab():
            w.grab().save(shot)
            if w.engine.running:
                w.stop()
            app.quit()
        QTimer.singleShot(int(after * 1000), grab)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
