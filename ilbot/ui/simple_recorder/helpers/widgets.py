from ilbot.ui.simple_recorder.helpers.context import get_payload
from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy


def craft_widget_rect(payload: dict, key: str) -> dict | None:
    w = (payload.get("crafting_widgets", {}) or {}).get(key)
    return unwrap_rect((w or {}).get("bounds") if isinstance(w, dict) else None)

def bank_widget_rect(payload: dict, key: str) -> dict | None:
    """Return screen-rect for a bank widget exported under data.bank_widgets[key]."""
    w = ((payload.get("bank_widgets") or {}).get(key) or {})
    b = (w.get("bounds") if isinstance(w, dict) else None)
    if isinstance(b, dict) and all(k in b for k in ("x","y","width","height")):
        return b
    return None

def rect_center_from_widget(w: dict | None) -> tuple[int | None, int | None]:
    rect = unwrap_rect((w or {}).get("bounds"))
    return rect_center_xy(rect)

def get_widget_text(widget_id: int, payload: dict = None) -> str | None:
    """Get text content from a widget by its ID."""
    if payload is None:
        payload = get_payload() or {}
    
    # Try to find the widget in various possible locations in the payload
    # This is a generic implementation that looks for widgets with the given ID
    
    # Check if there's a widgets section
    widgets = payload.get("widgets", {})
    if isinstance(widgets, dict):
        widget = widgets.get(str(widget_id))
        if isinstance(widget, dict):
            return widget.get("text")
    
    # Check if there's a tutorial section with widgets
    tutorial = payload.get("tutorial", {})
    if isinstance(tutorial, dict):
        # Look for widgets in tutorial data
        for key, value in tutorial.items():
            if isinstance(value, dict) and value.get("id") == widget_id:
                return value.get("text")
    
    # Check other possible widget locations
    for section_name in ["ui_widgets", "game_widgets", "interface_widgets"]:
        section = payload.get(section_name, {})
        if isinstance(section, dict):
            widget = section.get(str(widget_id))
            if isinstance(widget, dict):
                return widget.get("text")
    
    return None

def get_tutorial_set_name(payload: dict = None) -> dict | None:
    """Get the tutorial SET_NAME widget."""
    if payload is None:
        payload = get_payload() or {}
    
    tutorial = payload.get("tutorial", {})
    if isinstance(tutorial, dict):
        return tutorial.get("setName")
    
    return None

def get_tutorial_lookup_name(payload: dict = None) -> dict | None:
    """Get the tutorial LOOK_UP_NAME widget."""
    if payload is None:
        payload = get_payload() or {}
    
    tutorial = payload.get("tutorial", {})
    if isinstance(tutorial, dict):
        return tutorial.get("lookupName")
    
    return None

def get_character_design_widget(widget_id: int, payload: dict = None) -> dict | None:
    """Get a character design widget by its ID."""
    if payload is None:
        payload = get_payload() or {}
    
    # Check if there's a character_design section
    character_design = payload.get("character_design", {})
    if isinstance(character_design, dict):
        widget = character_design.get(str(widget_id))
        if isinstance(widget, dict):
            return widget
    
    # Check if there's a player_design section
    player_design = payload.get("player_design", {})
    if isinstance(player_design, dict):
        widget = player_design.get(str(widget_id))
        if isinstance(widget, dict):
            return widget
    
    # Check if there's a design section
    design = payload.get("design", {})
    if isinstance(design, dict):
        widget = design.get(str(widget_id))
        if isinstance(widget, dict):
            return widget
    
    # Check other possible widget locations
    for section_name in ["widgets", "ui_widgets", "game_widgets", "interface_widgets"]:
        section = payload.get(section_name, {})
        if isinstance(section, dict):
            widget = section.get(str(widget_id))
            if isinstance(widget, dict):
                return widget
    
    return None

def get_character_design_main(payload: dict = None) -> dict | None:
    """Get the main character design widget (PlayerDesign.MAIN)."""
    return get_character_design_widget(44498948, payload)

def get_character_design_widgets(payload: dict = None) -> dict:
    """Get all character design widgets from the payload."""
    if payload is None:
        payload = get_payload() or {}
    
    widgets = {}
    
    # Check various possible sections for character design widgets
    for section_name in ["character_design", "player_design", "design", "widgets", "ui_widgets", "game_widgets", "interface_widgets"]:
        section = payload.get(section_name, {})
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, dict) and value.get("id"):
                    widget_id = value.get("id")
                    # Check if this is a character design related widget
                    if (isinstance(widget_id, int) and 
                        (widget_id >= 44498948 or  # PlayerDesign.MAIN and below
                         "design" in key.lower() or 
                         "character" in key.lower() or
                         "player" in key.lower())):
                        widgets[str(widget_id)] = value
    
    return widgets

# Character Design Widget IDs (PlayerDesign enum values)
PLAYER_DESIGN_WIDGETS = {
    # Body parts - LEFT/RIGHT buttons
    "HEAD_LEFT": 679.15,
    "HEAD_RIGHT": 679.16,
    "JAW_LEFT": 679.19,
    "JAW_RIGHT": 679.20,
    "TORSO_LEFT": 679.23,
    "TORSO_RIGHT": 679.24,
    "ARMS_LEFT": 679.27,
    "ARMS_RIGHT": 679.28,
    "HANDS_LEFT": 679.31,
    "HANDS_RIGHT": 679.32,
    "LEGS_LEFT": 679.35,
    "LEGS_RIGHT": 679.36,
    "FEET_LEFT": 679.39,
    "FEET_RIGHT": 679.40,
    
    # Colors - LEFT/RIGHT buttons
    "HAIR_LEFT": 679.46,
    "HAIR_RIGHT": 679.47,
    "TORSO_COL_LEFT": 679.50,
    "TORSO_COL_RIGHT": 679.51,
    "LEGS_COL_LEFT": 679.54,
    "LEGS_COL_RIGHT": 679.55,
    "FEET_COL_LEFT": 679.58,
    "FEET_COL_RIGHT": 679.59,
    "SKIN_LEFT": 679.62,
    "SKIN_RIGHT": 679.63,
}

def get_character_design_widget_realtime(widget_name: str) -> dict | None:
    """Get character design widget via real-time IPC lookup by name."""
    if widget_name not in PLAYER_DESIGN_WIDGETS:
        print(f"[ERROR] Unknown character design widget: {widget_name}")
        return None
    
    widget_id = PLAYER_DESIGN_WIDGETS[widget_name]
    
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        # Create IPC client
        ipc = RuneLiteIPC()
        
        # Send widget data request
        response = ipc._send({
            "cmd": "get_widget",
            "widget_id": int(widget_id)
        })
        
        if response.get("ok"):
            return response.get("widget")
        else:
            print(f"[ERROR] IPC widget lookup failed for {widget_name}: {response.get('err', 'unknown error')}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to get widget {widget_name}: {e}")
        return None

def get_character_design_button_realtime(part: str, direction: str) -> dict | None:
    """Get a character design button (LEFT or RIGHT) for a specific body part/color."""
    if direction not in ["LEFT", "RIGHT"]:
        print(f"[ERROR] Invalid direction: {direction}. Must be LEFT or RIGHT")
        return None
    
    widget_name = f"{part}_{direction}"
    return get_character_design_widget_realtime(widget_name)

def get_button_name_by_id(widget_id: int) -> str:
    """Map widget ID to descriptive button name."""
    # Map widget IDs to button names based on the widget inspector
    id_to_name = {
        # Body parts - LEFT/RIGHT buttons
        44498960: "HEAD_LEFT",    # S 679.15 PlayerDesign.HEAD_LEFT
        44498961: "HEAD_RIGHT",   # S 679.16 PlayerDesign.HEAD_RIGHT
        44498964: "JAW_LEFT",     # S 679.19 PlayerDesign.JAW_LEFT
        44498965: "JAW_RIGHT",    # S 679.20 PlayerDesign.JAW_RIGHT
        44498968: "TORSO_LEFT",   # S 679.23 PlayerDesign.TORSO_LEFT
        44498969: "TORSO_RIGHT",  # S 679.24 PlayerDesign.TORSO_RIGHT
        44498972: "ARMS_LEFT",    # S 679.27 PlayerDesign.ARMS_LEFT
        44498973: "ARMS_RIGHT",   # S 679.28 PlayerDesign.ARMS_RIGHT
        44498976: "HANDS_LEFT",   # S 679.31 PlayerDesign.HANDS_LEFT
        44498977: "HANDS_RIGHT",  # S 679.32 PlayerDesign.HANDS_RIGHT
        44498980: "LEGS_LEFT",    # S 679.35 PlayerDesign.LEGS_LEFT
        44498981: "LEGS_RIGHT",   # S 679.36 PlayerDesign.LEGS_RIGHT
        44498984: "FEET_LEFT",    # S 679.39 PlayerDesign.FEET_LEFT
        44498985: "FEET_RIGHT",   # S 679.40 PlayerDesign.FEET_RIGHT
        
        # Colors - LEFT/RIGHT buttons
        44498991: "HAIR_LEFT",    # S 679.46 PlayerDesign.HAIR_LEFT
        44498992: "HAIR_RIGHT",   # S 679.47 PlayerDesign.HAIR_RIGHT
        44498995: "TORSO_COL_LEFT",   # S 679.50 PlayerDesign.TORSO_COL_LEFT
        44498996: "TORSO_COL_RIGHT",  # S 679.51 PlayerDesign.TORSO_COL_RIGHT
        44498999: "LEGS_COL_LEFT",    # S 679.54 PlayerDesign.LEGS_COL_LEFT
        44499000: "LEGS_COL_RIGHT",   # S 679.55 PlayerDesign.LEGS_COL_RIGHT
        44499003: "FEET_COL_LEFT",    # S 679.58 PlayerDesign.FEET_COL_LEFT
        44499004: "FEET_COL_RIGHT",   # S 679.59 PlayerDesign.FEET_COL_RIGHT
        44499007: "SKIN_LEFT",    # S 679.62 PlayerDesign.SKIN_LEFT
        44499008: "SKIN_RIGHT",   # S 679.63 PlayerDesign.SKIN_RIGHT
    }
    
    return id_to_name.get(widget_id, "")

def get_all_character_design_buttons() -> dict:
    """Get all character design LEFT/RIGHT buttons via real-time IPC calls."""
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        # Create IPC client
        ipc = RuneLiteIPC()
        
        print(f"[DEBUG] Getting character design widgets from parent 44498948...")
        
        # Get all child widgets of the main PlayerDesign widget
        response = ipc._send({
            "cmd": "get_widget_children",
            "widget_id": 44498948  # PlayerDesign.MAIN
        })
        
        print(f"[DEBUG] IPC response: {response}")
        
        if not response.get("ok"):
            print(f"[ERROR] Failed to get character design widgets: {response.get('err', 'unknown error')}")
            return {}
        
        children = response.get("children", [])
        print(f"[DEBUG] Found {len(children)} total child widgets")
        
        buttons = {}
        clickable_widgets = []
        
        # Filter for widgets with listeners (clickable widgets)
        for child in children:
            has_listener = child.get("hasListener", False)
            widget_id = child.get("id")
            
            if has_listener:
                clickable_widgets.append(child)
                print(f"[DEBUG] Found widget with listener: ID={widget_id}, hasListener={has_listener}")
                
                if widget_id:
                    # Map widget ID to descriptive name
                    button_name = get_button_name_by_id(widget_id)
                    if button_name:
                        buttons[button_name] = child
                        print(f"[DEBUG] Mapped widget {widget_id} to {button_name}")
                    else:
                        # Fallback: use widget ID as key if no name mapping found
                        buttons[f"WIDGET_{widget_id}"] = child
                        print(f"[DEBUG] No name mapping found for widget ID {widget_id}, using fallback key")
        
        print(f"[DEBUG] Found {len(clickable_widgets)} widgets with listeners, mapped {len(buttons)} to button names")
        
        # If we found clickable widgets but no mapped names, return them with fallback keys
        if clickable_widgets and not buttons:
            print(f"[DEBUG] No mapped names found, returning all clickable widgets with fallback keys")
            for i, widget in enumerate(clickable_widgets):
                buttons[f"CLICKABLE_{i}"] = widget
        
        return buttons
        
    except Exception as e:
        print(f"[ERROR] Failed to get character design buttons: {e}")
        import traceback
        traceback.print_exc()
        return {}

def widget_exists(widget_id: int, payload: dict = None) -> bool:
    """Check if a widget exists AND is visible via real-time IPC lookup by its ID."""
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        # Create IPC client
        ipc = RuneLiteIPC()
        
        # Send widget existence check request
        response = ipc._send({
            "cmd": "widget_exists",
            "widget_id": int(widget_id)
        })
        
        if response.get("ok"):
            # Return true only if widget exists AND is visible
            return response.get("exists", False) and response.get("visible", False)
        else:
            print(f"[ERROR] IPC widget existence check failed: {response.get('err', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to check widget {widget_id} existence: {e}")
        return False

def character_design_widget_exists(widget_name: str, payload: dict = None) -> bool:
    """Check if a character design widget exists by name via IPC lookup."""
    if widget_name not in PLAYER_DESIGN_WIDGETS:
        return False
    
    widget_id = PLAYER_DESIGN_WIDGETS[widget_name]
    return widget_exists(widget_id, payload)

def get_widget_info(widget_id: int, payload: dict = None) -> dict | None:
    """Get detailed information about a widget via IPC lookup if it exists."""
    if not widget_exists(widget_id, payload):
        return None
    
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        # Create IPC client
        ipc = RuneLiteIPC()
        
        # Send widget info request
        response = ipc._send({
            "cmd": "get_widget_info",
            "widget_id": int(widget_id)
        })
        
        if response.get("ok"):
            widget_data = response.get("widget")
            if widget_data:
                return {
                    "id": widget_id,
                    "section": response.get("section", "unknown"),
                    "key": str(widget_id),
                    "data": widget_data
                }
            return None
        else:
            print(f"[ERROR] IPC widget info lookup failed: {response.get('err', 'unknown error')}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to get widget {widget_id} info: {e}")
        return None