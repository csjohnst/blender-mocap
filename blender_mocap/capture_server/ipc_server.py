# blender_mocap/capture_server/ipc_server.py
"""Unix socket server for the capture process. Sends pose data, receives commands."""
import json
import os
import socket
import select
import threading
import queue


PROTOCOL_VERSION = 1


class IPCServer:
    """Newline-delimited JSON server over a Unix socket."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._command_queue: queue.Queue = queue.Queue()
        self._recv_buffer = ""

    def start(self) -> None:
        # Remove stale socket
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(1)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)
        while self._running:
            try:
                client, _ = self._server_sock.accept()
                self._client_sock = client
                self._client_sock.setblocking(False)
                # Send handshake
                self._raw_send({"type": "hello", "protocol_version": PROTOCOL_VERSION})
                self._read_commands_loop()
            except socket.timeout:
                continue
            except OSError:
                break

    def _read_commands_loop(self) -> None:
        while self._running and self._client_sock:
            try:
                ready, _, _ = select.select([self._client_sock], [], [], 0.5)
                if ready:
                    data = self._client_sock.recv(4096)
                    if not data:
                        break  # Client disconnected
                    self._recv_buffer += data.decode("utf-8")
                    while "\n" in self._recv_buffer:
                        line, self._recv_buffer = self._recv_buffer.split("\n", 1)
                        if line.strip():
                            msg = json.loads(line)
                            self._command_queue.put(msg)
            except (OSError, ConnectionError):
                break

    def _raw_send(self, msg: dict) -> None:
        if self._client_sock:
            data = json.dumps(msg) + "\n"
            self._client_sock.sendall(data.encode("utf-8"))

    def send(self, msg: dict) -> None:
        self._raw_send(msg)

    def send_heartbeat(self) -> None:
        self.send({"type": "heartbeat"})

    def read_command(self, timeout: float = 0.0) -> dict | None:
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_client(self) -> bool:
        return self._client_sock is not None

    def stop(self) -> None:
        self._running = False
        if self._client_sock:
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        if self._accept_thread:
            self._accept_thread.join(timeout=3.0)
