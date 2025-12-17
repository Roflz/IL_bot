from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

from ..helpers.runtime_utils import ipc

def _all_npcs() -> List[dict]:
    """Get all visible NPCs; filter out null/empty names."""
    # Use the proper IPC method to get all NPCs in the area
    npcs = ipc.get_closest_npcs()
    
    out: List[dict] = []
    for n in npcs:
        nm = (n.get("name") or "").strip()
        if not nm or nm.lower() == "null":
            continue
        out.append(n)
    return out

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def closest_npc_by_name(name: str) -> Optional[dict]:
    """Find closest NPC whose name contains `name` (case-insensitive)."""
    want = _norm(name)
    if not want:
        return None
    
    # First try to find in payload
    best = None
    best_d = None
    for n in _all_npcs():
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
    return _ipc_search_npc(name)

def npc_action_index(npc: dict, action: str) -> Optional[int]:
    """Return 0-based index of action in npc['actions'] (case-insensitive), or None if absent."""
    try:
        acts = [a.lower() for a in (npc.get("actions") or []) if a]
        a = action.strip().lower()
        return acts.index(a) if a in acts else None
    except Exception:
        return None

def _ipc_search_npc(name: str) -> Optional[dict]:
    """
    Use IPC to search for NPCs in the current area as a fallback.
    This searches for NPCs using the new IPC commands.
    """
    try:
        # Use the find_npc method which returns the closest match
        result = ipc.find_npc(name)
        if result and result.get("ok") and result.get("found"):
            npc = result.get("npc")
            if npc:
                # Convert to the expected format
                world = npc.get("world", {})
                return {
                    "name": npc.get("name"),
                    "x": world.get("x", 0),
                    "y": world.get("y", 0),
                    "p": world.get("p", 0),
                    "distance": npc.get("distance", 0),
                    "worldX": world.get("x", 0),
                    "worldY": world.get("y", 0),
                    "canvasX": npc.get("canvas", {}).get("x"),
                    "canvasY": npc.get("canvas", {}).get("y"),
                    "clickbox": npc.get("bounds"),
                    "actions": npc.get("actions", [])
                }
        return None
        
    except Exception:
        # If IPC search fails, return None
        return None

def get_all_npcs() -> List[dict]:
    """Get all NPCs in the current area."""
    return ipc.get_closest_npcs()

def get_npcs_by_name(name: str) -> List[dict]:
    """Get all NPCs matching the given name."""
    return ipc.get_npcs_by_name(name)

def get_npcs_in_radius(radius: int = 26) -> List[dict]:
    """Get all NPCs within the specified radius."""
    return ipc.get_npcs_in_radius(radius)

def get_npcs_in_combat() -> List[dict]:
    """Get all NPCs currently in combat."""
    return ipc.get_npcs_in_combat()

def get_npcs_by_action(action: str) -> List[dict]:
    """Get all NPCs that have the specified action available."""
    return ipc.get_npcs_by_action(action)

def find_npc_by_id(npc_id: int) -> Optional[dict]:
    """Find an NPC by its ID."""
    npcs = get_all_npcs()
    for npc in npcs:
        if npc.get("id") == npc_id:
            return npc
    return None

def get_closest_npc_by_action(action: str) -> Optional[dict]:
    """Get the closest NPC that has the specified action available."""
    npcs = get_npcs_by_action(action)
    if npcs:
        # NPCs are already sorted by distance from the IPC call
        return npcs[0]
    return None

def is_npc_in_combat(npc_name: str) -> bool:
    """Check if a specific NPC is in combat."""
    npcs = get_npcs_by_name(npc_name)
    for npc in npcs:
        if npc.get("inCombat", False):
            return True
    return False

def get_npc_actions(npc_name: str) -> List[str]:
    """Get all available actions for a specific NPC."""
    npc = closest_npc_by_name(npc_name)
    if npc:
        return npc.get("actions", [])
    return []
