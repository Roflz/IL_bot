import re, time, csv, random
from pathlib import Path
# from ilbot.ui.simple_recorder.helpers.runtime_utils import dispatch

_STEP_HITS: dict[str, int] = {}
_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')

def clean_rs(s: str | None) -> str:
    if not s:
        return ""
    return _RS_TAG_RE.sub('', s)

def norm_name(s: str | None) -> str:
    return clean_rs(s or "").strip().lower()

def now_ms() -> int:
    return int(time.time() * 1000)

def closest_object_by_names(names: list[str]) -> dict | None:
    from .runtime_utils import ipc
    objects_data = ipc.get_closest_objects() or {}
    wanted = [n.lower() for n in names]

    # Fallback to generic nearby objects
    for obj in (objects_data.get("objects") or []):
        nm = norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj

    return None

def get_world_from_csv(username: str) -> int | None:
    """
    Get the world number for a specific character from the character_stats.csv file.
    
    Args:
        username: Character username to look up
        
    Returns:
        - World number (int) if character found
        - None if character not found or error occurred
    """
    try:
        csv_file = Path("D:/repos/bot_runelite_IL/ilbot/ui/simple_recorder/character_data/character_stats.csv")
        
        if not csv_file.exists():
            return None
            
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('username') == username:
                    world_str = row.get('world_number')
                    if world_str:
                        return int(world_str)
                    return None
                    
        return None
        
    except Exception as e:
        print(f"[get_world_from_csv] Error reading CSV: {e}")
        return None


# ============================================================================
# NUMBER GENERATION METHODS
# ============================================================================

def random_number(min_val: float, max_val: float, output_type: str = "int") -> float | int:
    """
    Generate a random number between min and max values.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        output_type: "int" for integer, "float" for float
        
    Returns:
        Random number in specified range
    """
    result = random.uniform(min_val, max_val)
    return int(result) if output_type == "int" else result


def normal_number(min_val: float, max_val: float, center_bias: float = 0.5, output_type: str = "int") -> float | int:
    """
    Generate a number using normal distribution with center bias.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        center_bias: How much to bias toward center (0.0 = uniform, 1.0 = very centered)
        output_type: "int" for integer, "float" for float
        
    Returns:
        Random number using normal distribution
    """
    center = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / (4 + center_bias * 4)
    
    result = random.normalvariate(center, std_dev)
    result = max(min_val, min(max_val, result))
    
    return int(result) if output_type == "int" else result


def exponential_number(min_val: float, max_val: float, lambda_param: float = 1.0, output_type: str = "int") -> float | int:
    """
    Generate a number using exponential distribution.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        lambda_param: Rate parameter (higher = more likely to be shorter)
        output_type: "int" for integer, "float" for float
        
    Returns:
        Random number using exponential distribution
    """
    exp_value = random.expovariate(lambda_param)
    result = min_val + (exp_value / (1 + exp_value)) * (max_val - min_val)
    result = max(min_val, min(max_val, result))
    
    return int(result) if output_type == "int" else result


def beta_number(min_val: float, max_val: float, alpha: float = 2.0, beta: float = 2.0, output_type: str = "int") -> float | int:
    """
    Generate a number using beta distribution.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        alpha: Shape parameter (higher = more weight toward max)
        beta: Shape parameter (higher = more weight toward min)
        output_type: "int" for integer, "float" for float
        
    Returns:
        Random number using beta distribution
    """
    beta_value = random.betavariate(alpha, beta)
    result = min_val + beta_value * (max_val - min_val)
    
    return int(result) if output_type == "int" else result


def triangular_number(min_val: float, max_val: float, mode_bias: float = 0.5, output_type: str = "int") -> float | int:
    """
    Generate a number using triangular distribution.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        mode_bias: Where the peak is (0.0 = min, 0.5 = center, 1.0 = max)
        output_type: "int" for integer, "float" for float
        
    Returns:
        Random number using triangular distribution
    """
    mode = min_val + mode_bias * (max_val - min_val)
    result = random.triangular(min_val, max_val, mode)
    
    return int(result) if output_type == "int" else result


# ============================================================================
# TIMING METHODS
# ============================================================================

def sleep_random(min_seconds: float, max_seconds: float) -> None:
    """
    Sleep for a random amount of time between min and max seconds.
    
    Args:
        min_seconds: Minimum sleep time in seconds
        max_seconds: Maximum sleep time in seconds
    """
    sleep_time = random_number(min_seconds, max_seconds, "float")
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)


def sleep_normal(min_seconds: float, max_seconds: float, center_bias: float = 0.5) -> None:
    """
    Sleep using normal distribution with center bias.
    Higher center_bias (0-1) makes it more likely to sleep closer to the center.
    
    Args:
        min_seconds: Minimum sleep time in seconds
        max_seconds: Maximum sleep time in seconds
        center_bias: How much to bias toward center (0.0 = uniform, 1.0 = very centered)
    """
    sleep_time = normal_number(min_seconds, max_seconds, center_bias, "float")
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)


def sleep_exponential(min_seconds: float, max_seconds: float, lambda_param: float = 1.0) -> None:
    """
    Sleep using exponential distribution (good for human-like timing).
    
    Args:
        min_seconds: Minimum sleep time in seconds
        max_seconds: Maximum sleep time in seconds
        lambda_param: Rate parameter (higher = more likely to be shorter)
    """
    sleep_time = exponential_number(min_seconds, max_seconds, lambda_param, "float")
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)


def sleep_beta(min_seconds: float, max_seconds: float, alpha: float = 2.0, beta: float = 2.0) -> None:
    """
    Sleep using beta distribution (good for bounded distributions with custom shapes).
    
    Args:
        min_seconds: Minimum sleep time in seconds
        max_seconds: Maximum sleep time in seconds
        alpha: Shape parameter (higher = more weight toward max)
        beta: Shape parameter (higher = more weight toward min)
    """
    sleep_time = beta_number(min_seconds, max_seconds, alpha, beta, "float")
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)


# ============================================================================
# RECTANGLE CLICKING METHODS
# ============================================================================

def rect_center_xy(rect: tuple) -> tuple[int, int]:
    """
    Get center coordinates of a rectangle.
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        
    Returns:
        (center_x, center_y) tuple
    """
    min_x, max_x, min_y, max_y = rect
    return ((min_x + max_x) // 2, (min_y + max_y) // 2)


def rect_random_xy(rect: tuple) -> tuple[int, int]:
    """
    Get random coordinates within a rectangle.
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        
    Returns:
        (random_x, random_y) tuple
    """
    min_x, max_x, min_y, max_y = rect
    x = random_number(min_x, max_x, "int")
    y = random_number(min_y, max_y, "int")
    return (x, y)


def rect_normal_xy(rect: tuple, center_bias: float = 0.5) -> tuple[int, int]:
    """
    Get coordinates using normal distribution with center bias.
    Higher center_bias makes clicks more likely to be near center.
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        center_bias: How much to bias toward center (0.0 = uniform, 1.0 = very centered)
        
    Returns:
        (x, y) tuple
    """
    min_x, max_x, min_y, max_y = rect
    x = normal_number(min_x, max_x, center_bias, "int")
    y = normal_number(min_y, max_y, center_bias, "int")
    return (x, y)


def rect_beta_xy(rect: tuple, alpha: float = 2.0, beta: float = 2.0) -> tuple[int, int]:
    """
    Get coordinates using beta distribution (higher probability near center when alpha=beta).
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        alpha: Shape parameter (higher = more weight toward max)
        beta: Shape parameter (higher = more weight toward min)
        
    Returns:
        (x, y) tuple
    """
    min_x, max_x, min_y, max_y = rect
    x = beta_number(min_x, max_x, alpha, beta, "int")
    y = beta_number(min_y, max_y, alpha, beta, "int")
    return (x, y)


def rect_triangular_xy(rect: tuple, mode_bias: float = 0.5) -> tuple[int, int]:
    """
    Get coordinates using triangular distribution.
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        mode_bias: Where the peak is (0.0 = min, 0.5 = center, 1.0 = max)
        
    Returns:
        (x, y) tuple
    """
    min_x, max_x, min_y, max_y = rect
    x = triangular_number(min_x, max_x, mode_bias, "int")
    y = triangular_number(min_y, max_y, mode_bias, "int")
    return (x, y)


# ============================================================================
# DESTINATION TILE SELECTION METHODS
# ============================================================================

def get_random_walkable_tile(destination_key: str, custom_dest_rect: tuple = None) -> tuple[int, int] | None:
    """
    Get a random walkable tile within the destination rectangle.
    
    Args:
        destination_key: Navigation key for destination
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
        
    Returns:
        (x, y) tuple of random walkable tile, or None if none found
    """
    try:
        from collision_cache.pathfinder import load_collision_data, get_walkable_tiles
        from ..helpers.navigation import get_nav_rect
        
        # Get destination coordinates
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            dest_rect = custom_dest_rect
        else:
            dest_rect = get_nav_rect(destination_key)
            if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
                return None
        
        min_x, max_x, min_y, max_y = dest_rect
        
        # Load collision data for the destination area
        dest_center_x = (min_x + max_x) // 2
        dest_center_y = (min_y + max_y) // 2
        destination = (dest_center_x, dest_center_y)
        
        collision_data = load_collision_data(destination, destination)
        if not collision_data:
            return None
        
        # Get walkable tiles
        walkable_tiles, _, _, _ = get_walkable_tiles(collision_data)
        
        # Find walkable tiles within the destination rectangle
        walkable_in_dest = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in walkable_tiles:
                    walkable_in_dest.append((x, y))
        
        if not walkable_in_dest:
            return None
        
        # Return random walkable tile
        return random.choice(walkable_in_dest)
        
    except Exception as e:
        print(f"[get_random_walkable_tile] Error: {e}")
        return None


def get_center_walkable_tile(destination_key: str, custom_dest_rect: tuple = None) -> tuple[int, int] | None:
    """
    Get the center walkable tile within the destination rectangle.
    
    Args:
        destination_key: Navigation key for destination
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
        
    Returns:
        (x, y) tuple of center walkable tile, or None if none found
    """
    try:
        from collision_cache.pathfinder import load_collision_data, get_walkable_tiles
        from ..helpers.navigation import get_nav_rect
        
        # Get destination coordinates
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            dest_rect = custom_dest_rect
        else:
            dest_rect = get_nav_rect(destination_key)
            if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
                return None
        
        min_x, max_x, min_y, max_y = dest_rect
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Load collision data for the destination area
        destination = (center_x, center_y)
        collision_data = load_collision_data(destination, destination)
        if not collision_data:
            return None
        
        # Get walkable tiles
        walkable_tiles, _, _, _ = get_walkable_tiles(collision_data)
        
        # Find walkable tiles within the destination rectangle
        walkable_in_dest = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in walkable_tiles:
                    walkable_in_dest.append((x, y))
        
        if not walkable_in_dest:
            return None
        
        # Find closest walkable tile to center
        min_dist = float('inf')
        closest_tile = None
        
        for tile in walkable_in_dest:
            dist = ((tile[0] - center_x)**2 + (tile[1] - center_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_tile = tile
        
        return closest_tile
        
    except Exception as e:
        print(f"[get_center_walkable_tile] Error: {e}")
        return None


def get_center_weighted_walkable_tile(destination_key: str, custom_dest_rect: tuple = None, center_bias: float = 2.0) -> tuple[int, int] | None:
    """
    Get a walkable tile using center-weighted selection (higher probability near center).
    
    Args:
        destination_key: Navigation key for destination
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
        center_bias: How strong the center bias is (1.0 = mild, 3.0 = very strong)
        
    Returns:
        (x, y) tuple of walkable tile, or None if none found
    """
    try:
        from collision_cache.pathfinder import load_collision_data, get_walkable_tiles
        from ..helpers.navigation import get_nav_rect
        
        # Get destination coordinates
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            dest_rect = custom_dest_rect
        else:
            dest_rect = get_nav_rect(destination_key)
            if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
                return None
        
        min_x, max_x, min_y, max_y = dest_rect
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Load collision data for the destination area
        destination = (int(center_x), int(center_y))
        collision_data = load_collision_data(destination, destination)
        if not collision_data:
            return None
        
        # Get walkable tiles
        walkable_tiles, _, _, _ = get_walkable_tiles(collision_data)
        
        # Find walkable tiles within the destination rectangle
        walkable_in_dest = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in walkable_tiles:
                    walkable_in_dest.append((x, y))
        
        if not walkable_in_dest:
            return None
        
        # Use weighted selection to select tile
        # Calculate weights based on distance from center
        weights = []
        for tile in walkable_in_dest:
            dist = ((tile[0] - center_x)**2 + (tile[1] - center_y)**2)**0.5
            # Higher weight for tiles closer to center
            weight = 1.0 / (1.0 + dist**center_bias)
            weights.append(weight)
        
        # Select tile based on weights
        return random.choices(walkable_in_dest, weights=weights, k=1)[0]
        
    except Exception as e:
        print(f"[get_center_weighted_walkable_tile] Error: {e}")
        return None