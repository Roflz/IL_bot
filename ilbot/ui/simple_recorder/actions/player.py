import time
from typing import Optional, Union

from ilbot.ui.simple_recorder.actions import inventory, tab, widgets
from .timing import wait_until
from ..helpers.runtime_utils import ipc
from ilbot.ui.simple_recorder.helpers.vars import get_var
from ilbot.ui.simple_recorder.constants import PLAYER_ANIMATIONS
from ..helpers.utils import press_spacebar


def get_player_plane(default=None):
    """
    Return the player's plane (0/1/2/3) from IPC.
    """
    player_data = ipc.get_player()
    player = player_data.get("player", {})
    plane = player.get("plane")
    return int(plane) if isinstance(plane, (int, float)) else default


def logged_in() -> bool:
    """
    Check if the player is logged into the game.
    
    Returns:
        - True if logged in (game state is LOGGED_IN)
        - False if at login screen or other state
    """
    try:
        resp = ipc.get_game_state()
        if not resp or not resp.get("ok"):
            return False
        if widgets.widget_exists(24772680):
            return False
        
        state = resp.get("state", "").upper()
        return state == "LOGGED_IN"
    except Exception:
        return False

def check_total_level(required_level: int) -> bool:
    """
    Check if the player's total level meets the required level.
    
    Args:
        required_level: Minimum total level required
    
    Returns:
        True if total level >= required_level, False otherwise
    """
    try:
        player_data = ipc.get_player()
        if not player_data.get("ok"):
            return False
        
        player_info = player_data.get("player", {})
        skills = player_info.get("skills", {})
        
        total_level = 0
        for skill_name, skill_data in skills.items():
            if isinstance(skill_data, dict) and "level" in skill_data:
                level = int(skill_data.get("level", 1))
                total_level += level
        
        return total_level >= required_level
        
    except Exception as e:
        logging.error(f"[check_total_level] actions/ge.py: {e}")
        return False

def login() -> bool:
    """
    Attempt to log into the game by clicking the "Play Now" button and waiting for the WelcomeScreen.PLAY widget.
    
    Args:
        username: RuneScape username (not used in this simple approach)
        password: RuneScape password (not used in this simple approach)
        
    Returns:
        - True if login was successful
        - False if login failed or already logged in
    """
    # Check if already logged in
    if logged_in():
        print("[LOGIN] Already logged in")
        return True
    
    # Check if we're at the login screen
    resp = ipc.get_game_state()
    if not resp or not resp.get("ok"):
        print("[LOGIN] Could not get game state")
        return False

    state = resp.get("state", "").upper()
    if state == "LOGIN_SCREEN":

        print(f"[LOGIN] At login screen, clicking 'Play Now' button")

        import random

        # Define the rectangle for the "Play Now" button
        min_x, min_y = 860, 265
        max_x, max_y = 1020, 305

        # Try clicking in the rectangle up to 3 times
        max_attempts = 3
        for attempt in range(max_attempts):
            # Generate random coordinates within the rectangle
            click_x = random.randint(min_x, max_x)
            click_y = random.randint(min_y, max_y)

            print(f"[LOGIN] Attempt {attempt + 1}: Clicking at ({click_x}, {click_y})")
            ipc.click(click_x, click_y)
            time.sleep(1)

            print("[LOGIN] Clicked 'Play Now' button, waiting for WelcomeScreen.PLAY widget...")

            # Wait for the widget to appear (with 3 second timeout)
            if wait_until(lambda: widgets.widget_exists(24772680), max_wait_ms=3000):
                print("[LOGIN] WelcomeScreen.PLAY widget appeared!")
                return True
            else:
                print(f"[LOGIN] WelcomeScreen.PLAY widget did not appear within 3 seconds (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    print("[LOGIN] Trying again...")
                    continue
                else:
                    print("[LOGIN] All attempts failed")
                    return False
    else:
        if widgets.widget_exists(24772680):
            print("[LOGIN] WelcomeScreen.PLAY widget appeared, clicking it...")
            widgets.click_widget(24772680)  # WelcomeScreen.PLAY widget ID
            return True
        return False


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


def get_skills() -> Optional[int]:
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

    return player_data.get("skills", {})

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
    if not inventory.has_items({"logs": 1, "tinderbox": 1}):
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


def get_run_energy() -> Optional[int]:
    """
    Get the player's current run energy percentage.
    
    Returns:
        - Run energy percentage (0-100) if successful
        - None if failed to get data
    """
    resp = ipc.get_player()
    if not resp or not resp.get("ok"):
        return 10000
    
    player_data = resp.get("player")
    if not player_data:
        return 10000
    
    return player_data.get("runEnergy")


def is_run_on() -> bool:
    """
    Check if the player's run mode is currently enabled.

    Returns:
        - True if run mode is on
        - False if run mode is off or failed to get data
    """
    try:
        run = widgets.get_widget_info(10485793)
        if not run or not run.get('data'):
            return False

        run_id = run['data'].get('spriteId')
        if run_id is None:
            return False

        if run_id == 1069:
            return False
        if run_id == 1070:
            return True
        return True
    except (KeyError, TypeError, AttributeError):
        return True

def toggle_run() -> bool:
    """
    Toggle the player's run mode on/off by clicking the run icon.
    
    Returns:
        - True if run was successfully toggled
        - False if failed to toggle
    """
    try:
        # Click the run icon (widget ID 10485782)
        widgets.click_widget(10485793)
        time.sleep(0.1)  # Small delay to ensure click registers
        return True
    except Exception as e:
        print(f"[RUN] Error toggling run: {e}")
        return False


def ensure_run_on() -> bool:
    """
    Ensure run mode is on if we have enough energy (20+).
    
    Returns:
        - True if run is on or was successfully turned on
        - False if not enough energy or failed to toggle
    """
    # Check if already running
    if is_run_on():
        return True
    
    # Check run energy
    energy = get_run_energy()
    if energy is None:
        print("[RUN] Could not get run energy")
        return False
    
    if energy < 20:
        print(f"[RUN] Not enough energy to run ({energy}%)")
        return False
    
    # Toggle run on
    print(f"[RUN] Turning run on (energy: {energy}%)")
    return toggle_run()


def ensure_run_off() -> bool:
    """
    Ensure run mode is off.
    
    Returns:
        - True if run is off or was successfully turned off
        - False if failed to toggle
    """
    # Check if already not running
    if not is_run_on():
        return True
    
    # Toggle run off
    print("[RUN] Turning run off")
    return toggle_run()


def get_world() -> Optional[int]:
    """
    Get the current world number using IPC.
    
    Returns:
        - World number (int) if successful
        - None if failed to get data
    """
    resp = ipc.get_world()
    if not resp or not resp.get("ok"):
        return None
    
    return resp.get("world")


def hop_world(world_id: int) -> bool:
    """
    Hop to a specific world using IPC.
    
    Args:
        world_id: Target world number to hop to
        
    Returns:
        - True if hop was successful
        - False if hop failed
    """
    resp1 = ipc.open_world_hopper()
    if not resp1 or not resp1.get("ok"):
        return False
    # time.sleep(0.5)
    resp2 = ipc.hop_world(world_id)
    if not resp2 or not resp2.get("ok"):
        return False
    if not (wait_until(lambda: widgets.widget_exists(12648448), max_wait_ms=3000)):
        return False
    current_world = get_world()
    press_spacebar()
    if not (wait_until(lambda: not logged_in())):
        return False
    if not (wait_until(lambda: logged_in())):
        return False

    return True

