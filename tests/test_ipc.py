# tests/test_ipc.py
import json
import os
import tempfile
import threading
import time
import pytest
from blender_mocap.capture_server.ipc_server import IPCServer
from blender_mocap.ipc_client import IPCClient


@pytest.fixture
def socket_path():
    path = os.path.join(tempfile.mkdtemp(), "test.sock")
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestIPCProtocol:
    def test_handshake(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            hello = client.read_message()
            assert hello["type"] == "hello"
            assert hello["protocol_version"] == 1
            client.close()
        finally:
            server.stop()

    def test_server_sends_pose(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            pose = {"type": "pose", "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}], "timestamp": 1.0}
            server.send(pose)
            msg = client.read_message()
            assert msg["type"] == "pose"
            assert msg["landmarks"][0]["x"] == 0.5
            client.close()
        finally:
            server.stop()

    def test_client_sends_command(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            client.send_command("start_preview")
            msg = server.read_command(timeout=2.0)
            assert msg["type"] == "command"
            assert msg["action"] == "start_preview"
            client.close()
        finally:
            server.stop()

    def test_backpressure_keeps_latest(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            # Send multiple poses rapidly
            for i in range(10):
                server.send({"type": "pose", "landmarks": [], "timestamp": float(i)})
            time.sleep(0.1)  # Let them buffer

            # Drain should return only the latest pose, plus other messages
            latest, others = client.drain_latest_pose()
            assert latest is not None
            assert latest["timestamp"] == 9.0
            client.close()
        finally:
            server.stop()

    def test_stale_socket_cleanup(self, socket_path):
        # Create a stale socket file
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)
        with open(socket_path, "w") as f:
            f.write("stale")
        # Server should remove it and bind successfully
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            hello = client.read_message()
            assert hello["type"] == "hello"
            client.close()
        finally:
            server.stop()

    def test_heartbeat(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            server.send_heartbeat()
            msg = client.read_message()
            assert msg["type"] == "heartbeat"
            client.close()
        finally:
            server.stop()
