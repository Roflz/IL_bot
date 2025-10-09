from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy
from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc


def craft_widget_rect(key: str) -> dict | None:
    from .runtime_utils import ipc
    crafting_widgets_data = ipc.get_crafting_widgets()
    w = crafting_widgets_data.get(key)
    return unwrap_rect((w or {}).get("bounds") if isinstance(w, dict) else None)

def bank_widget_rect(key: str) -> dict | None:
    """Return screen-rect for a bank widget exported under data.bank_widgets[key]."""
    from .runtime_utils import ipc
    bank_widgets_data = ipc.get_bank_widgets()
    w = bank_widgets_data.get(key) or {}
    b = (w.get("bounds") if isinstance(w, dict) else None)
    if isinstance(b, dict) and all(k in b for k in ("x","y","width","height")):
        return b
    return None

def rect_center_from_widget(w: dict | None) -> tuple[int | None, int | None]:
    rect = unwrap_rect((w or {}).get("bounds"))
    return rect_center_xy(rect)

def get_widget_text(widget_id: int) -> str | None:
    """Get text content from a widget by its ID."""
    from .runtime_utils import ipc
    widget_data = ipc.send({"cmd": "get_widget_info", "widget_id": widget_id}) or {}
    
    # Get widget text from the widget data
    if widget_data and widget_data.get("ok"):
        data = widget_data.get("data", {})
        return data.get("text")
    
    return None

def get_tutorial_set_name() -> dict | None:
    """Get the tutorial SET_NAME widget."""
    from .runtime_utils import ipc
    tutorial_data = ipc.send({"cmd": "get_tutorial"}) or {}
    
    if tutorial_data and tutorial_data.get("ok"):
        return tutorial_data.get("setName")
    
    return None

def get_tutorial_lookup_name() -> dict | None:
    """Get the tutorial LOOK_UP_NAME widget."""
    from .runtime_utils import ipc
    tutorial_data = ipc.send({"cmd": "get_tutorial"}) or {}
    
    if tutorial_data and tutorial_data.get("ok"):
        return tutorial_data.get("lookupName")
    
    return None

def get_character_design_widget(widget_id: int) -> dict | None:
    """Get a character design widget by its ID."""
    from .runtime_utils import ipc
    widget_data = ipc.send({"cmd": "get_widget_info", "widget_id": widget_id}) or {}
    
    # Get widget data from IPC response
    if widget_data and widget_data.get("ok"):
        return widget_data.get("data")
    
    return None

def get_character_design_main() -> dict | None:
    """Get the main character design widget (PlayerDesign.MAIN)."""
    return get_character_design_widget(44498948)

def get_character_design_widgets() -> dict:
    """Get all character design widgets from the payload."""
    from .runtime_utils import ipc
    character_design_data = ipc.send({"cmd": "get_character_design"}) or {}
    
    widgets = {}
    
    if character_design_data and character_design_data.get("ok"):
        return character_design_data.get("widgets", {})
    
    return {}

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

def widget_exists(widget_id: int) -> bool:
    """Check if a widget exists AND is visible via real-time IPC lookup by its ID."""
    try:
        from .runtime_utils import ipc
        
        # Check widget existence using dedicated IPC method
        response = ipc.widget_exists(int(widget_id))
        
        if response and response.get("ok"):
            # Return true only if widget exists AND is visible
            return response.get("exists", False) and response.get("visible", False)
        else:
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to check widget {widget_id} existence: {e}")
        return False

def character_design_widget_exists(widget_name: str) -> bool:
    """Check if a character design widget exists by name via IPC lookup."""
    if widget_name not in PLAYER_DESIGN_WIDGETS:
        return False
    
    widget_id = PLAYER_DESIGN_WIDGETS[widget_name]
    return widget_exists(widget_id)

def get_widget_info(widget_id: int) -> dict | None:
    """Get detailed information about a widget via IPC lookup if it exists."""
    if not widget_exists(widget_id):
        return None
    
    try:
        # Send widget info request
        response = ipc.get_widget_info(widget_id)
        
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


def click_listener_on(widget_id: int) -> bool:
    """
    Check if a widget has an active click listener (OnOpListener).
    
    Args:
        widget_id: The widget ID to check
        
    Returns:
        True if the widget has an active click listener, False otherwise
    """
    try:
        # Get widget info
        response = ipc.get_widget_info(int(widget_id))
        
        if response and response.get("ok"):
            widget_data = response.get("widget")
            if widget_data:
                on_op_listener = widget_data.get("onOpListener")
                # Return True if onOpListener is not None and not empty
                return on_op_listener is None
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to check click listener for widget {widget_id}: {e}")
        return False