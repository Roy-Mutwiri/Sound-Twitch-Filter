"""PyInstaller entry point. The exe is windowed (no console), so any start-up exception is written
to %LOCALAPPDATA%\StreamVoiceFocus\error.log and shown in a message box."""
import multiprocessing
import os
import sys
import traceback


def _log_path():
    d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "StreamVoiceFocus")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "error.log")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        from stream_voice_focus.__main__ import main
        code = main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
        try:
            if "--screenshot" in sys.argv:
                raise RuntimeError("automated run: no dialog")
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, tb[-1500:], "Stream Voice Focus failed to start", 0x10)
        except Exception:
            pass
        code = 1
    sys.exit(code)
