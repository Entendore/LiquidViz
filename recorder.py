import subprocess
import threading
import queue
import numpy as np

class VideoRecorder:
    def __init__(self):
        self.process = None
        self.thread = None
        self.queue = None
        self.active = False

    def start(self, filename, fps, sim_size, export_size):
        self.stop()
        input_w, input_h = sim_size
        target_w, target_h = export_size
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{input_w}x{input_h}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-',
            '-vf', f'scale={target_w}:{target_h}:flags=lanczos',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', filename
        ]
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return False

        self.queue = queue.Queue(maxsize=300)
        self.active = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        return True

    def _writer_loop(self):
        while self.active:
            try:
                frame = self.queue.get(timeout=0.5)
                if frame is None: break
                try:
                    self.process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break
            except queue.Empty: continue
        if self.process:
            self.process.stdin.close()
            self.process.wait()

    def add_frame(self, frame):
        if not self.active: return
        try: self.queue.put_nowait(frame)
        except queue.Full: pass

    def stop(self):
        if not self.active: return
        self.active = False
        if self.queue: self.queue.put(None)
        if self.thread: self.thread.join(timeout=2.0)
        self.process = None; self.thread = None