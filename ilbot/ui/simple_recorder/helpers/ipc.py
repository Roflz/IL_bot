import json, socket, time
from typing import Optional, Tuple, List, Dict, Any
from heapq import heappush, heappop

def ipc_port_from_payload(payload: dict, default: int = 17000) -> int:
    return payload["__ipc"].port

# in _ipc_send(...) inside action_plans.py
def ipc_send(payload: dict, msg: dict, timeout: float = 0.35) -> Optional[dict]:
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

def ipc_project_many(payload: dict, tiles_w: List[Dict[str, int]]) -> Tuple[List[Dict[str, Any]], dict]:
    """
    Batch project world tiles to canvas using tilexy_many.
    Returns (list_of_results, debug). Each result ≈
      {"world":{"x":..,"y":..,"p":..}, "projection":{...}, "canvas":{"x":..,"y":..}} when onscreen.
    """
    if not tiles_w:
        return [], {"warn": "no tiles to project"}

    req = {"cmd": "tilexy_many", "tiles": [{"x": int(t["x"]), "y": int(t["y"])} for t in tiles_w]}
    resp = ipc_send(payload, req)
    dbg = {"ipc_req": req, "ipc_resp": resp}

    out: List[Dict[str, Any]] = []
    results = (resp or {}).get("results") or []
    for t, r in zip(tiles_w, results):
        item = {"world": {"x": t["x"], "y": t["y"], "p": t.get("p", 0)}, "projection": r}
        if r and r.get("ok") and r.get("onscreen") and isinstance(r.get("canvas"), dict):
            item["canvas"] = {"x": int(r["canvas"]["x"]), "y": int(r["canvas"]["y"])}
        out.append(item)
    return out, dbg

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

def ipc_path(payload: dict, *, rect: Tuple[int,int,int,int] | None = None,
              goal: Tuple[int,int] | None = None, max_wps: int = 20) -> Tuple[List[Dict[str,int]], dict]:
    if rect is None and goal is None:
        return [], {"err":"no rect/goal"}
    req = {"cmd": "path", "maxWps": int(max_wps)}
    if rect: req.update({"minX": rect[0], "maxX": rect[1], "minY": rect[2], "maxY": rect[3]})
    if goal: req.update({"goalX": int(goal[0]), "goalY": int(goal[1])})
    resp = ipc_send(payload, req); dbg = {"ipc_req": req, "ipc_resp": resp}
    if not resp or not resp.get("ok"): return [], dbg
    out = []
    for w in resp.get("waypoints") or []:
        try: out.append({"x": int(w["x"]), "y": int(w["y"]), "p": int(w.get("p",0))})
        except Exception: pass
    return out, dbg