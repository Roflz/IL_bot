import json, socket, time
from typing import Optional, Tuple, List, Dict, Any
from heapq import heappush, heappop

from ilbot.ui.simple_recorder.helpers.context import get_payload


def ipc_port_from_payload(payload: dict, default: int = 17000) -> int:
    return payload["__ipc"].port

# in _ipc_send(...) inside action_plans.py
def ipc_send(msg: dict, payload: dict | None = None, timeout: float = 0.35) -> Optional[dict]:
    if payload is None:
        payload = get_payload()
    host = "127.0.0.1"
    port = ipc_port_from_payload(payload)
    t0 = time.time()
    try:
        line = json.dumps(msg, separators=(",", ":"))
        # print(f"[IPC->] port={port} {line}")
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((line + "\n").encode("utf-8"))
            data = b""
            while True:
                ch = s.recv(1)
                if not ch or ch == b"\n":
                    break
                data += ch
        resp = json.loads(data.decode("utf-8")) if data else None
        dt = int((time.time() - t0)*1000)
        # print(f"[<-IPC] {resp} ({dt} ms)")
        return resp
    except Exception as e:
        dt = int((time.time() - t0)*1000)
        print(f"[IPC ERR] {type(e).__name__}: {e} ({dt} ms)")
        return None

def ipc_project_many(payload, wps):
    """
    Project world tiles to canvas AND preserve 'door' metadata.
    Returns:
      out: [{onscreen, canvas, world:{x,y,p}, door?}, ...]
      dbg: raw resp from IPC
    """
    tiles = [{"x": int(w["x"]), "y": int(w["y"])} for w in wps]
    resp = ipc_send({"cmd": "tilexy_many", "tiles": tiles})
    results = resp.get("results", []) or []

    out = []
    for i, (w, r) in enumerate(zip(wps, results)):
        row = {
            "onscreen": (r or {}).get("onscreen"),
            "canvas":  (r or {}).get("canvas"),
            "world":   {"x": int(w["x"]), "y": int(w["y"]), "p": int(w.get("p", 0))},
        }
        # ✨ keep door info from projection if present, else from the waypoint itself
        door = None
        if isinstance(r, dict) and isinstance(r.get("door"), dict):
            door = r["door"]
        elif isinstance(w, dict) and isinstance(w.get("door"), dict):
            door = w["door"]
        if door:
            row["door"] = door

        out.append(row)

    return out, resp

def ipc_mask(payload: dict, radius: int = 15) -> dict | None:
    ipc = payload.get("__ipc")
    if not ipc:
        return None
    try:
        m = ipc._send({"cmd": "mask", "radius": int(radius)})
        return m if isinstance(m, dict) and m.get("ok") else None
    except Exception:
        return None

def astar_on_rows(rows: list[str], start_rc: tuple[int,int], goal_rc: tuple[int,int]) -> list[tuple[int,int]]:
    # rows[0] is northmost, columns left->right; r,c are indices into rows
    R, C = len(rows), len(rows[0]) if rows else 0
    def walkable(r,c): return 0 <= r < R and 0 <= c < C and rows[r][c] == '.'
    sr, sc = start_rc; gr, gc = goal_rc
    if not (walkable(sr,sc) and 0 <= gr < R and 0 <= gc < C): return []
    # If goal blocked, search for nearest walkable in 3×3 then 5×5; else clamp later
    if not walkable(gr,gc):
        found = None
        for rad in (1,2,3):
            for rr in range(gr-rad, gr+rad+1):
                for cc in range(gc-rad, gc+rad+1):
                    if walkable(rr,cc): found = (rr,cc); break
                if found: break
            if found: break
        if found: gr,gc = found

    openh = []; heappush(openh, (0, (sr,sc)))
    came = { (sr,sc): None }
    gscore = { (sr,sc): 0 }
    def h(r,c): return abs(r-gr)+abs(c-gc)
    while openh:
        _, (r,c) = heappop(openh)
        if (r,c) == (gr,gc): break
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if not walkable(nr,nc): continue
            ng = gscore[(r,c)] + 1
            if ng < gscore.get((nr,nc), 1e9):
                gscore[(nr,nc)] = ng
                came[(nr,nc)] = (r,c)
                heappush(openh, (ng + h(nr,nc), (nr,nc)))
    if (gr,gc) not in came: return []
    # Reconstruct
    path = []
    cur = (gr,gc)
    while cur is not None:
        path.append(cur)
        cur = came.get(cur)
    path.reverse()
    return path

def rows_to_world(mask: dict, rc_path: list[tuple[int,int]]) -> list[tuple[int,int]]:
    """Map mask row/col indices back to world x,y."""
    if not rc_path: return []
    radius = int(mask["radius"])
    origin = mask["origin"]; wx0, wy0 = int(origin["x"]), int(origin["y"])
    # rows indexing: r=0 is wy0+radius (north), c=0 is wx0-radius (west)
    out = []
    for r,c in rc_path:
        wx = (wx0 - radius) + c
        wy = (wy0 + radius) - r
        out.append((wx, wy))
    return out

def ipc_path(payload, rect=None, goal=None, max_wps=None):
    req = {"cmd": "path"}
    if rect:
        minX, maxX, minY, maxY = rect
        req.update({"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY})
    if goal:
        gx, gy = goal
        req.update({"goalX": gx, "goalY": gy})
    if max_wps is not None:
        req["maxWps"] = int(max_wps)

    resp = ipc_send(req)
    # ← keep waypoints exactly as provided, including "door"
    wps = resp.get("waypoints", []) or []
    return wps, resp
