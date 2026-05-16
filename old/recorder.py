import subprocess
import threading
import queue
import numpy as np
from config import FPS, SIM_W, SIM_H

class Recorder:
    def __init__(self):
        self.process = None
        self.thread = None
        self.queue = None
        self.active = False

    def start(self, filename, resolution):
        self.stop()
        
        target_w, target_h = resolution
        input_w, input_h = SIM_W, SIM_H
        
        vf_filter = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )

        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{input_w}x{input_h}',
            '-pix_fmt', 'rgb24',
            '-r', str(FPS),
            '-i', '-',
            '-vf', vf_filter,
            '-c:v', 'h264_nvenc',      
            '-preset', 'p4',           
            '-rc', 'vbr',              
            '-cq', '21',               
            '-pix_fmt', 'yuv420p',
            filename
        ]

        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            # Fallback to CPU
            cmd[10] = 'libx264'
            cmd[11] = '-preset'
            cmd[12] = 'fast'
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        self.queue = queue.Queue(maxsize=300)
        self.active = True
        self.thread = threading.Thread(target=self._writer_loop)
        self.thread.start()
        return True

    def _writer_loop(self):
        while self.active:
            try:
                frame = self.queue.get(timeout=0.1)
                if frame is None: break
                try:
                    self.process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    self.active = False
                    break
            except queue.Empty:
                continue
        
        if self.process:
            self.process.stdin.close()
            self.process.wait()

    def add_frame(self, frame):
        if not self.active: return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        if not self.active: return
        self.active = False
        if self.queue: self.queue.put(None)
        if self.thread: self.thread.join(timeout=2.0)
        self.process = None
        self.thread = None