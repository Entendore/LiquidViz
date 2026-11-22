# recorder.py - Asynchronous video recording via ffmpeg
#
# IMPROVEMENTS:
#   - Pre-checks ffmpeg availability
#   - Returns meaningful error messages
#   - Handles BrokenPipeError and subprocess crashes gracefully
#   - Configurable via config.py constants
#   - Tracks recording stats (frames, duration)

import subprocess
import threading
import queue
import shutil
import time
from config import EXPORT_CRF, EXPORT_PRESET, FPS


class VideoRecorder:
    def __init__(self):
        self.process = None
        self.thread = None
        self.queue = None
        self.active = False
        self._frame_count = 0
        self._start_time = 0.0

    @staticmethod
    def is_ffmpeg_available():
        """Check if ffmpeg is installed and accessible."""
        return shutil.which("ffmpeg") is not None

    def start(self, filename, fps=None, sim_size=None, export_size=None):
        """Start recording. Returns (success, error_message)."""
        self.stop()

        if not self.is_ffmpeg_available():
            return False, "ffmpeg not found. Install it or add to PATH."

        if fps is None:
            fps = FPS
        if sim_size is None or export_size is None:
            return False, "sim_size and export_size must be specified."

        input_w, input_h = sim_size
        target_w, target_h = export_size

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{input_w}x{input_h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", EXPORT_PRESET,
            "-crf", str(EXPORT_CRF),
            "-pix_fmt", "yuv420p",
            filename,
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,   # IMPROVEMENT: capture stderr for diagnostics
                stdout=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False, "ffmpeg binary not found."
        except OSError as e:
            return False, f"Failed to launch ffmpeg: {e}"

        self.queue = queue.Queue(maxsize=300)
        self.active = True
        self._frame_count = 0
        self._start_time = time.perf_counter()
        self._error_msg = None

        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        return True, ""

    def _writer_loop(self):
        """Background thread that feeds frames to ffmpeg stdin."""
        while self.active:
            try:
                frame = self.queue.get(timeout=0.5)
                if frame is None:
                    break
                try:
                    self.process.stdin.write(frame.tobytes())
                    self._frame_count += 1
                except BrokenPipeError:
                    self._error_msg = "ffmpeg process crashed (broken pipe)."
                    break
                except OSError as e:
                    self._error_msg = f"Write error to ffmpeg: {e}"
                    break
            except queue.Empty:
                continue

        # Flush and close
        if self.process:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            self.process.wait()
            if self.process.returncode != 0:
                stderr = self.process.stderr.read().decode(errors="replace")
                # Grab just the last line for a concise message
                last_line = stderr.strip().split("\n")[-1] if stderr.strip() else "unknown error"
                self._error_msg = f"ffmpeg exited with code {self.process.returncode}: {last_line}"

    def add_frame(self, frame):
        if not self.active:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame rather than block the render loop

    def stop(self):
        """Stop recording and wait for writer thread to finish."""
        if not self.active:
            return
        self.active = False
        if self.queue:
            self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=5.0)
        self.process = None
        self.thread = None

    @property
    def stats(self):
        """Return (frame_count, duration_seconds) of the last recording."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        return self._frame_count, elapsed

    @property
    def last_error(self):
        return getattr(self, "_error_msg", None)