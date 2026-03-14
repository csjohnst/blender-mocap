# blender_mocap/capture_server/audio.py
"""Audio capture via sounddevice for synchronized recording."""
import os
import threading
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioRecorder:
    """Records audio from an input device to a WAV file."""

    SAMPLE_RATE = 44100
    CHANNELS = 1
    DTYPE = "int16"

    def __init__(self, device_index: int | None = None):
        self._device_index = device_index
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self, output_path: str) -> None:
        """Begin recording audio to the specified WAV file path."""
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        self._output_path = output_path
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            device=self._device_index,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def stop(self) -> str | None:
        """Stop recording and write WAV file. Returns the file path."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return None
            audio_data = np.concatenate(self._frames, axis=0)
            self._frames = []

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with wave.open(self._output_path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        return self._output_path

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return list of available audio input devices."""
        if sd is None:
            return []
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                inputs.append({"index": i, "name": dev["name"]})
        return inputs

    def _write_test_wav(self, path: str, duration_sec: float = 0.1) -> None:
        """Write a silent WAV file for testing."""
        num_frames = int(self.SAMPLE_RATE * duration_sec)
        silence = np.zeros(num_frames, dtype=np.int16)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(silence.tobytes())
