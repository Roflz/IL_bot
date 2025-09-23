from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

from ..helpers.context import get_payload
from ..helpers.ipc import ipc_send

def _all_npcs(payload: Optional[dict] = None) -> List[dict]:
    """Combine visible NPC lists; filter out null/empty names."""
    if payload is None:
        payload = get_payload() or {}
    npcs = (payload.get("closestNPCs") or []) + (payload.get("npcs") or [])
    out: List[dict] = []
    for n in npcs:
        nm = (n.get("name") or "").strip()
        if not nm or nm.lower() == "null":
            continue
        out.append(n)
    return out

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def npc_exists(name: str, payload: Optional[dict] = None) -> bool:
    return closest_npc_by_name(name, payload) is not None

def closest_npc_by_name(name: str, payload: Optional[dict] = None) -> Optional[dict]:
    """Find closest NPC whose name contains `name` (case-insensitive)."""
    want = _norm(name)
    if not want:
        return None
    
    # First try to find in payload
    best = None
    best_d = None
    for n in _all_npcs(payload):
        nm = _norm(n.get("name") or "")
        if want in nm:
            d = n.get("distance")
            try:
                d = int(d)
            except Exception:
                d = 1_000_000
            if best is None or d < best_d:
                best, best_d = n, d
    
    # If found in payload, return it
    if best is not None:
        return best
    
    # Fallback: Use IPC to search for NPCs in the area
    return _ipc_search_npc(name, payload)

def npc_action_index(npc: dict, action: str) -> Optional[int]:
    """Return 0-based index of action in npc['actions'] (case-insensitive), or None if absent."""
    try:
        acts = [a.lower() for a in (npc.get("actions") or []) if a]
        a = action.strip().lower()
        return acts.index(a) if a in acts else None
    except Exception:
        return None

def npc_anchor_point(npc: dict) -> Optional[Tuple[int, int]]:
    """
    Prefer rect center from clickbox; else fall back to canvasX/Y.
    (No imports here; action layer will compute rect center if provided.)
    """
    cb = npc.get("clickbox")
    if isinstance(cb, dict) and all(k in cb for k in ("x", "y", "width", "height")):
        # just indicate we have a rect via a sentinel; the action will compute center
        return None  # center computed in actions when rect available
    x = npc.get("canvasX")
    y = npc.get("canvasY")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (int(x), int(y))
    return None

def _ipc_search_npc(name: str, payload: Optional[dict] = None) -> Optional[dict]:
    """
    Use IPC to search for NPCs in the current area as a fallback.
    This searches for NPCs using the new 'npcs' IPC command.
    """
    if payload is None:
        payload = get_payload() or {}
    
    try:
        # Get player position
        player = payload.get("player", {})
        if not player:
            return None
            
        px = player.get("x", 0)
        py = player.get("y", 0)
        plane = player.get("p", 0)
        
        # Use IPC to get all NPCs
        resp = ipc_send({
            "cmd": "npcs",
            "name": name  # Filter by name on the server side
        }, payload)
        
        if not resp or not resp.get("ok"):
            return None
            
        # Look for NPCs in the response
        npcs = resp.get("npcs", [])
        want = _norm(name)
        
        best_npc = None
        best_distance = float('inf')
        
        for npc in npcs:
            npc_name = _norm(npc.get("name", ""))
            if want in npc_name:
                # Calculate distance
                npc_world = npc.get("world", {})
                npc_x = npc_world.get("x", 0)
                npc_y = npc_world.get("y", 0)
                distance = abs(npc_x - px) + abs(npc_y - py)
                
                if distance < best_distance:
                    best_distance = distance
                    best_npc = {
                        "name": npc.get("name"),
                        "x": npc_x,
                        "y": npc_y,
                        "p": plane,
                        "distance": distance,
                        "worldX": npc_x,
                        "worldY": npc_y,
                        "canvasX": npc.get("canvas", {}).get("x"),
                        "canvasY": npc.get("canvas", {}).get("y"),
                        "clickbox": npc.get("bounds"),
                        "actions": npc.get("actions", [])
                    }
        
        return best_npc
        
    except Exception:
        # If IPC search fails, return None
        return None
