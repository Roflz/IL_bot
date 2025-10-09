from typing import Optional, Union

from ilbot.ui.simple_recorder.actions import inventory, tab
from ..helpers.runtime_utils import ipc
from ilbot.ui.simple_recorder.helpers.vars import get_var
from ilbot.ui.simple_recorder.constants import PLAYER_ANIMATIONS


def get_player_plane(default=None):
    """
    Return the player's plane (0/1/2/3) from IPC.
    """
    player_data = ipc.get_player()
    player = player_data.get("player", {})
    plane = player.get("plane")
    return int(plane) if isinstance(plane, (int, float)) else default


def get_x() -> Optional[int]:
    """
    Get the player's current X coordinate using direct IPC calls.
    
    Returns:
        - Player X coordinate (int) if found
        - None if failed to get data or player not found
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("worldX")


def get_y() -> Optional[int]:
    """
    Get the player's current Y coordinate using direct IPC calls.
    
    Returns:
        - Player Y coordinate (int) if found
        - None if failed to get data or player not found
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("worldY")

def get_player_position():
    position  = (get_x(), get_y())
    return position

def in_cutscene(timeout: float = 0.35) -> bool:
    """
    Check if the player is in a cutscene.
    Returns True if varbit 542 == 1, False otherwise.
    """
    val = get_var(542, timeout=timeout)
    return val == 1


def get_player_animation() -> Union[int, str, None]:
    """
    Get the player's current animation ID using direct IPC calls.
    
    Returns:
        - Animation ID (int) if not found in constants
        - Animation name (str) if found in PLAYER_ANIMATIONS mapping
        - None if failed to get animation or player not found
    """
    resp = ipc.get_player()
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


def get_skill_level(skill_name: str) -> Optional[int]:
    """
    Get the player's current level for a specific skill using direct IPC calls.
    
    Args:
        skill_name: Name of the skill (e.g., "fishing", "cooking", "attack")
        
    Returns:
        - Skill level (int) if found
        - None if skill not found or failed to get data
    """
    resp = ipc.get_player()
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

def get_skill_xp(skill_name: str) -> Optional[int]:
    """
    Get the player's experience points for a specific skill using direct IPC calls.
    
    Args:
        skill_name: Name of the skill (e.g., "fishing", "cooking", "attack")
        
    Returns:
        - Skill XP (int) if found
        - None if skill not found or failed to get data
    """
    resp = ipc.get_player()
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


def get_all_skills() -> Optional[dict]:
    """
    Get all player skills data using direct IPC calls.
    
    Returns:
        - Dictionary of all skills with their data if successful
        - None if failed to get data
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("skills")


def is_in_combat() -> bool:
    """
    Check if the player is currently in combat.
    
    Returns:
        True if player is in combat (interacting with something), False otherwise
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return False
    
    player_data = resp.get("player")
    if not player_data:
        return False
    
    return player_data.get("isInteracting", False)


def get_combat_target() -> Optional[dict]:
    """
    Get information about what the player is currently fighting/interacting with.
    
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
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("interactingTarget")


def get_combat_level() -> Optional[int]:
    """
    Get the player's combat level.
    
    Returns:
        Player's combat level if successful, None otherwise
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    return player_data.get("combatLevel")


def get_health() -> Optional[int]:
    """
    Get the player's current hitpoints.
    
    Returns:
        Current hitpoints (int) if successful, None otherwise
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return None
    
    player_data = resp.get("player")
    if not player_data:
        return None
    
    # Try healthRatio first (current hitpoints)
    health_ratio = player_data.get("healthRatio")
    if health_ratio is not None and health_ratio > 0:
        return health_ratio
    
    # Fallback to hitpoints skill level
    skills = player_data.get("skills", {})
    hitpoints_data = skills.get("hitpoints")
    if hitpoints_data:
        return hitpoints_data.get("boostedLevel")  # Current hitpoints (boosted level)
    
    return None


def get_tile_objects() -> list[dict]:
    """
    Get all objects on the player's current tile using direct IPC calls.
    
    Returns:
        - List of objects on the player's tile, each with type, id, name, actions
        - Empty list if no objects or failed to get data
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return []
    
    player_data = resp.get("player")
    if not player_data:
        return []
    
    return player_data.get("tileObjects", [])


def has_fire_on_tile() -> bool:
    """
    Check if there's already a fire on the player's current tile.
    
    Returns:
        - True if there's a fire on the tile, False otherwise
    """
    tile_objects = get_tile_objects()
    
    for obj in tile_objects:
        obj_name = (obj.get("name") or "").lower()
        if "fire" in obj_name:
            return True
    
    return False


def make_fire() -> bool:
    """
    Make a fire using logs and tinderbox, but only if there isn't already a fire on the tile.

    Returns:
        - True if fire was made or already exists, False if failed
    """
    # Check if there's already a fire on the tile
    if has_fire_on_tile():
        return True  # Fire already exists, no need to make another
    
    # Check if player has required items
    if not inventory.has_items(["logs", "tinderbox"]):
        return False
    
    # Check if player is already firemaking
    if get_player_animation() == "FIREMAKING":
        return True  # Already firemaking
    
    # Get current firemaking XP to detect when fire is made
    current_xp = get_skill_xp("firemaking")
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

