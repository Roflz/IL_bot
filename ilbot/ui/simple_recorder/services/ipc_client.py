# ilbot/ui/simple_recorder/services/ipc_client.py
import json
import socket
import time

class RuneLiteIPC:
    """
    Pure move from main_window; behavior unchanged.
    """
    def __init__(self, host="127.0.0.1", port=17000, pre_action_ms=250, timeout_s=2.0):
        self.host = host
        self.port = port
        self.pre_action_ms = pre_action_ms
        self.timeout_s = timeout_s

    def _send(self, obj: dict) -> dict:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
                s.sendall(data)
                s.shutdown(socket.SHUT_WR)
                resp = s.makefile("r", encoding="utf-8").readline().strip()
                return json.loads(resp) if resp else {"ok": False, "err": "empty-response"}
        except socket.timeout:
            return {"ok": False, "err": "timeout"}
        except ConnectionRefusedError:
            return {"ok": False, "err": "connection-refused"}
        except Exception as e:
            return {"ok": False, "err": f"{type(e).__name__}: {e}"}

    def ping(self) -> bool:
        try:
            r = self._send({"cmd": "ping"})
            return bool(r.get("ok"))
        except Exception:
            return False

    def focus(self):
        try:
            self._send({"cmd": "focus"})
        except Exception:
            pass

    def click_canvas(self, x: int, y: int, button: int = 1, pre_ms: int | None = None):
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            time.sleep(delay / 1000.0)
        return self._send({"cmd": "click", "x": int(x), "y": int(y), "button": int(button)})

    def key(self, k: str, pre_ms: int | None = None):
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            time.sleep(delay / 1000.0)
        return self._send({"cmd": "key", "k": str(k)})

    def project_world_tile(self, world_x: int, world_y: int, plane: int = 0) -> dict:
        """
        Asks the IPC plugin to project a world tile into canvas coordinates.
        Returns dict like: {"ok":true, "x":123, "y":456, "onScreen":true}
        If the IPC doesn't implement 'project', you'll get {"ok":false,...}.
        """
        try:
            return self._send({"cmd": "project", "worldX": int(world_x), "worldY": int(world_y), "plane": int(plane)})
        except Exception as e:
            return {"ok": False, "err": f"{type(e).__name__}: {e}"}