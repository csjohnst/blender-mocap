# blender_mocap/ipc_client.py
"""Unix socket client for the Blender addon. Reads pose data, sends commands."""
import json
import socket
import select


class IPCClient:
    """Connects to the capture server's Unix socket."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._buffer = ""

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)
        self._sock.setblocking(False)

    def read_message(self, timeout: float = 5.0) -> dict | None:
        """Read a single JSON message. Blocks up to timeout."""
        if not self._sock:
            return None
        deadline = timeout
        while deadline > 0:
            # Check buffer first
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    return json.loads(line)
            ready, _, _ = select.select([self._sock], [], [], min(deadline, 0.5))
            if ready:
                try:
                    data = self._sock.recv(8192)
                except OSError:
                    return None  # Server disconnected or crashed
                if not data:
                    return None  # Server disconnected
                self._buffer += data.decode("utf-8")
            deadline -= 0.5
        return None

    def drain_latest_pose(self) -> tuple[dict | None, list[dict]]:
        """Read all available messages. Returns (latest_pose, other_messages).

        other_messages includes heartbeats, status, errors — needed for liveness tracking.
        """
        if not self._sock:
            return None, []
        # Read all available data
        while True:
            ready, _, _ = select.select([self._sock], [], [], 0)
            if not ready:
                break
            try:
                data = self._sock.recv(65536)
            except OSError:
                break
            if not data:
                break
            self._buffer += data.decode("utf-8")

        # Parse all messages, keep latest pose, collect others
        latest_pose = None
        other_messages = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                msg = json.loads(line)
                if msg.get("type") == "pose":
                    latest_pose = msg
                else:
                    other_messages.append(msg)

        return latest_pose, other_messages

    def send_command(self, action: str) -> None:
        if self._sock:
            msg = json.dumps({"type": "command", "action": action}) + "\n"
            self._sock.sendall(msg.encode("utf-8"))

    def is_connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
