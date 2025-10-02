from typing import Optional, Union

from ilbot.ui.simple_recorder.actions import inventory, tab
from ilbot.ui.simple_recorder.helpers.context import get_payload
from ilbot.ui.simple_recorder.helpers.vars import get_var
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send
from ilbot.ui.simple_recorder.constants import PLAYER_ANIMATIONS


def get_player_plane(payload: dict | None = None, default=None):
    """
    Return the player's plane (0/1/2/3) from a payload.
    Works if payload is the full game payload or the `player` sub-dict.
    """
    if payload is None:
        payload = get_payload()
    if not isinstance(payload, dict):
        return default
    # If it's the full payload, expect a "player" key; otherwise treat it as the player dict.
    player = payload.get("player", payload)
    plane = player.get("plane")
    return int(plane) if isinstance(plane, (int, float)) else default


def get_x(payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's current X coordinate using direct IPC calls.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Player X coordinate (int) if found
        - None if failed to get data or player not found
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("worldX")


def get_y(payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's current Y coordinate using direct IPC calls.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Player Y coordinate (int) if found
        - None if failed to get data or player not found
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("worldY")

def in_cutscene(payload: dict | None = None, timeout: float = 0.35) -> bool:
    """
    Check if the player is in a cutscene.
    Returns True if varbit 542 == 1, False otherwise.
    """
    val = get_var(542, payload=payload, timeout=timeout)
    return val == 1


def get_player_animation(payload: Optional[dict] = None) -> Union[int, str, None]:
    """
    Get the player's current animation ID using direct IPC calls.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Animation ID (int) if not found in constants
        - Animation name (str) if found in PLAYER_ANIMATIONS mapping
        - None if failed to get animation or player not found
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    animation_id = player_data.get("animation")
    if animation_id is None:
        return None
    
    # Return the mapped name if it exists, otherwise return the raw ID
    return PLAYER_ANIMATIONS.get(animation_id, animation_id)


def get_skill_level(skill_name: str, payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's current level for a specific skill using direct IPC calls.
    
    Args:
        skill_name: Name of the skill (e.g., "fishing", "cooking", "attack")
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Skill level (int) if found
        - None if skill not found or failed to get data
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    skills = player_data.get("skills", {})
    skill_data = skills.get(skill_name.lower())
    if not skill_data:
        return None
    
    return skill_data.get("level")


def get_skill_boosted_level(skill_name: str, payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's boosted level for a specific skill using direct IPC calls.
    
    Args:
        skill_name: Name of the skill (e.g., "fishing", "cooking", "attack")
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Boosted skill level (int) if found
        - None if skill not found or failed to get data
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    skills = player_data.get("skills", {})
    skill_data = skills.get(skill_name.lower())
    if not skill_data:
        return None
    
    return skill_data.get("boostedLevel")


def get_skill_xp(skill_name: str, payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's experience points for a specific skill using direct IPC calls.
    
    Args:
        skill_name: Name of the skill (e.g., "fishing", "cooking", "attack")
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Skill XP (int) if found
        - None if skill not found or failed to get data
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    skills = player_data.get("skills", {})
    skill_data = skills.get(skill_name.lower())
    if not skill_data:
        return None
    
    return skill_data.get("xp")


def get_all_skills(payload: Optional[dict] = None) -> Optional[dict]:
    """
    Get all player skills data using direct IPC calls.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - Dictionary of all skills with their data if successful
        - None if failed to get data
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("skills")


def is_in_combat(payload: Optional[dict] = None) -> bool:
    """
    Check if the player is currently in combat.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        True if player is in combat (interacting with something), False otherwise
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return False
    
    player_data = resp.get("player")
    if not player_data:
        return False
    
    return player_data.get("isInteracting", False)


def get_combat_target(payload: Optional[dict] = None) -> Optional[dict]:
    """
    Get information about what the player is currently fighting/interacting with.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        Dictionary with target info if in combat, None otherwise
        {
            "name": "Goblin",
            "id": 1234,
            "type": "NPC",  # "NPC", "Player", or "Unknown"
            "worldX": 3000,
            "worldY": 3000,
            "worldP": 0,
            "healthRatio": 10,
            "healthScale": 10,
            "combatLevel": 2
        }
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("interactingTarget")


def get_combat_level(payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the player's combat level.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        Player's combat level if successful, None otherwise
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("combatLevel")


def get_tile_objects(payload: Optional[dict] = None) -> list[dict]:
    """
    Get all objects on the player's current tile using direct IPC calls.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - List of objects on the player's tile, each with type, id, name, actions
        - Empty list if no objects or failed to get data
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_player"}, payload)
    if not resp or not resp.get("ok"):
        return []
    
    player_data = resp.get("player")
    if not player_data:
        return []
    
    return player_data.get("tileObjects", [])


def has_fire_on_tile(payload: Optional[dict] = None) -> bool:
    """
    Check if there's already a fire on the player's current tile.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        - True if there's a fire on the tile, False otherwise
    """
    tile_objects = get_tile_objects(payload)
    
    for obj in tile_objects:
        obj_name = (obj.get("name") or "").lower()
        if "fire" in obj_name:
            return True
    
    return False


def make_fire(payload: Optional[dict] = None, ui=None) -> bool:
    """
    Make a fire using logs and tinderbox, but only if there isn't already a fire on the tile.
    
    Args:
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance
        
    Returns:
        - True if fire was made or already exists, False if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        from ..helpers.context import get_ui
        ui = get_ui()
    
    # Check if there's already a fire on the tile
    if has_fire_on_tile(payload):
        return True  # Fire already exists, no need to make another
    
    # Check if player has required items
    if not inventory.has_items(["logs", "tinderbox"]):
        return False
    
    # Check if player is already firemaking
    if get_player_animation(payload) == "FIREMAKING":
        return True  # Already firemaking
    
    # Get current firemaking XP to detect when fire is made
    current_xp = get_skill_xp("firemaking", payload)
    if current_xp is None:
        return False
    
    # Ensure inventory tab is open
    tab.ensure_tab_open("INVENTORY")
    
    # Use logs on tinderbox
    result = inventory.use_item_on_item("logs", "tinderbox")
    if not result:
        return False
    
    # Wait for firemaking animation to start
    from .timing import wait_until
    if not wait_until(lambda: get_player_animation() == "FIREMAKING"):
        return False
    
    # Wait for XP to change (indicating fire was made)
    if not wait_until(lambda: get_skill_xp("firemaking") != current_xp):
        return False
    
    return True

