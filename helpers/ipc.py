import json, socket, time
from typing import Optional

# Global variable to store the last interaction
_last_interaction = None


class IPCClient:
    """
    Consolidated IPC client that handles all communication with RuneLite.
    This replaces the scattered ipc_send functions and RuneLiteIPC class.
    """
    
    def __init__(self, host="127.0.0.1", port=17000, pre_action_ms=0, timeout_s=2.0):
        self.host = host
        self.port = port
        self.pre_action_ms = pre_action_ms
        self.timeout_s = timeout_s

    def _send(self, msg: dict, timeout: float = None) -> Optional[dict]:
        """Internal method to send IPC messages."""
        timeout = timeout or self.timeout_s
        t0 = time.time()
        try:
            line = json.dumps(msg, separators=(",", ":"))
            with socket.create_connection((self.host, self.port), timeout=timeout) as s:
                s.settimeout(timeout)
                s.sendall((line + "\n").encode("utf-8"))
                data = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in chunk:
                        data = data.split(b"\n", 1)[0]
                        break
                resp = json.loads(data.decode("utf-8")) if data else None
                dt = int((time.time() - t0)*1000)
                return resp
        except Exception as e:
            dt = int((time.time() - t0)*1000)
            print(f"[IPC ERR] {type(e).__name__}: {e} ({dt} ms)")
            return None

    # ===== BASIC IPC METHODS =====
    
    def ping(self) -> bool:
        """Test if RuneLite is responding."""
        try:
            r = self._send({"cmd": "ping"})
            return bool(r.get("ok"))
        except Exception:
            return False

    def focus(self):
        """Focus the RuneLite window."""
        try:
            self._send({"cmd": "focus"})
        except Exception:
            pass

    # ===== CLICKING METHODS =====
    
    def click(self, x: int, y: int, button: int = 1, hover_only: bool = False, pre_ms: int = None):
        """Click at canvas coordinates."""
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            from .utils import sleep_exponential
            sleep_exponential(delay / 1000.0 * 0.8, delay / 1000.0 * 1.2, 1.0)
        
        msg = {"cmd": "click", "x": int(x), "y": int(y), "button": int(button)}
        if hover_only:
            msg["hover_only"] = True
        
        result = self._send(msg)
        
        # Capture the last interaction after clicking
        if not hover_only and button == 1:
            self._capture_last_interaction()
        
        return result

    def click_canvas(self, x: int, y: int, button: int = 1, pre_ms: int = None):
        """Alias for click method."""
        return self.click(x, y, button, pre_ms=pre_ms)

    # ===== KEYBOARD METHODS =====
    
    def key(self, k: str, pre_ms: int = None):
        """Send a key press."""
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            from .utils import sleep_exponential
            sleep_exponential(delay / 1000.0 * 0.8, delay / 1000.0 * 1.2, 1.0)
        return self._send({"cmd": "key", "k": str(k)})

    def type(self, text: str, enter: bool = True, per_char_ms: int = 30):
        """Type text with optional enter."""
        payload = {
            "cmd": "type",
            "text": str(text),
            "enter": bool(enter),
            "perCharMs": int(per_char_ms),
        }
        return self._send(payload)

    def key_hold(self, key: str, ms: int = 180):
        """Hold a key for specified milliseconds."""
        return self._send({"cmd": "keyHold", "key": str(key), "ms": int(ms)})
    
    def key_press(self, key: str):
        """Press a key down."""
        return self._send({"cmd": "keyPress", "key": str(key)})
    
    def key_release(self, key: str):
        """Release a key."""
        return self._send({"cmd": "keyRelease", "key": str(key)})

    def scroll(self, amount: int) -> dict:
        """Mouse wheel scroll (positive = zoom in, negative = zoom out)."""
        return self._send({"cmd": "scroll", "amount": int(amount)})

    # ===== GAME STATE METHODS =====
    
    def get_player(self) -> dict:
        """Get player information."""
        return self._send({"cmd": "get_player"})
    
    def get_game_state(self) -> dict:
        """Get the current game state (LOGIN_SCREEN, LOGGED_IN, etc.)."""
        return self._send({"cmd": "get_game_state"}) or {}

    def get_inventory(self) -> dict:
        """Get inventory information."""
        return self._send({"cmd": "get_inventory"}) or {}

    def get_equipment(self) -> dict:
        """Get equipment information."""
        return self._send({"cmd": "get_equipment"}) or {}

    def get_bank(self) -> dict:
        """Get bank information."""
        return self._send({"cmd": "get_bank"}) or {}

    def get_bank_inventory(self) -> dict:
        """Get bank inventory information."""
        return self._send({"cmd": "get_bank_inventory"}) or {}

    def get_npcs(self, name: str = None) -> dict:
        """Get NPCs, optionally filtered by name."""
        msg = {"cmd": "npcs"}
        if name:
            msg["name"] = name
        return self._send(msg) or {}

    def get_closest_npcs(self) -> list:
        """Get closest NPCs (returns list of NPCs sorted by distance)."""
        response = self._send({"cmd": "npcs"})
        if response and response.get("ok"):
            return response.get("npcs", [])
        return []

    def find_npc(self, name: str) -> dict:
        """Find a specific NPC by name (returns closest match)."""
        response = self._send({"cmd": "npcs", "name": name})
        if response and response.get("ok"):
            npcs = response.get("npcs", [])
            if npcs:
                return {"ok": True, "found": True, "npc": npcs[0]}
            else:
                return {"ok": True, "found": False, "npc": None}
        return {"ok": False, "found": False, "npc": None}

    def get_npcs_by_name(self, name: str) -> list:
        """Get all NPCs matching the given name."""
        response = self._send({"cmd": "npcs", "name": name})
        if response and response.get("ok"):
            return response.get("npcs", [])
        return []

    def get_npcs_in_radius(self, radius: int = 26) -> list:
        """Get all NPCs within the specified radius."""
        response = self._send({"cmd": "npcs"})
        if response and response.get("ok"):
            npcs = response.get("npcs", [])
            # Filter by radius (the Java command already filters by radius 26 by default)
            return [npc for npc in npcs if npc.get("distance", 999) <= radius]
        return []

    def get_npcs_in_combat(self) -> list:
        """Get all NPCs currently in combat."""
        response = self._send({"cmd": "npcs"})
        if response and response.get("ok"):
            npcs = response.get("npcs", [])
            return [npc for npc in npcs if npc.get("inCombat", False)]
        return []

    def get_npcs_by_action(self, action: str) -> list:
        """Get all NPCs that have the specified action available."""
        response = self._send({"cmd": "npcs"})
        if response and response.get("ok"):
            npcs = response.get("npcs", [])
            action_lower = action.lower()
            return [npc for npc in npcs if action_lower in [a.lower() for a in (npc.get("actions") or [])]]
        return []

    def get_chat_widgets(self) -> dict:
        """Get chat widget information (ChatLeft, ChatMenu, ChatRight)."""
        return self._send({"cmd": "get_chat_widgets"}) or {}

    def get_chat(self) -> dict:
        """Get chat widget information (alias for get_chat_widgets)."""
        return self.get_chat_widgets()

    def click_continue_button(self, chat_type: str = "left") -> bool:
        """
        Click the continue button for chat dialogue.
        
        Args:
            chat_type: "left" for NPC dialogue continue, "right" for player dialogue continue
        
        Returns:
            True if click was successful, False otherwise
        """
        chat_widgets = self.get_chat_widgets()
        if not chat_widgets.get("ok"):
            return False
        
        # Determine which continue button to click
        if chat_type.lower() == "left":
            continue_info = chat_widgets.get("chatLeft", {}).get("continue", {})
        elif chat_type.lower() == "right":
            continue_info = chat_widgets.get("chatRight", {}).get("continue", {})
        else:
            return False
        
        # Check if continue button exists and is clickable
        if not continue_info.get("exists") or not continue_info.get("hasListener"):
            return False
        
        # Get center coordinates for clicking
        center = continue_info.get("center")
        if not center:
            return False
        
        # Click the continue button
        click_result = self._send({
            "cmd": "click",
            "x": center.get("x"),
            "y": center.get("y"),
            "button": 1
        })
        
        return click_result and click_result.get("ok", False)

    def get_objects(self, name: str = None, types: list = None, radius: int = None) -> list:
        """Get objects, optionally filtered by name, types, and radius."""
        msg = {"cmd": "objects"}
        if name:
            msg["name"] = name
        if types:
            msg["types"] = types
        if radius:
            msg["radius"] = radius
        return self._send(msg) or []

    def get_closest_objects(self) -> list:
        """Get closest objects."""
        return self._send({"cmd": "objects"}) or []

    def find_object(self, name: str, types: list = None, exact_match: bool = False) -> dict:
        """Find a specific object by name."""
        msg = {"cmd": "find_object", "name": name, "exactMatch": exact_match}
        if types:
            msg["types"] = types
        return self._send(msg) or {}

    def find_object_by_path(self, name: str, types: list = None) -> dict:
        """Find the closest object by path distance (waypoint count)."""
        msg = {"cmd": "find_object_by_path", "name": name}
        if types:
            msg["types"] = types
        return self._send(msg, timeout=20000) or {}

    def get_object_at_tile(self, x: int, y: int, plane: int = None, name: str = None, types: list = None) -> dict:
        """Get object(s) at a specific tile coordinate."""
        msg = {"cmd": "get_object_at_tile", "x": int(x), "y": int(y)}
        if plane is not None:
            msg["plane"] = int(plane)
        if name:
            msg["name"] = name
        if types:
            msg["types"] = types
        return self._send(msg) or {}

    def find_object_in_area(self, name: str, min_x: int, max_x: int, min_y: int, max_y: int, types: list = None) -> dict:
        """Find the closest object within a specific area."""
        msg = {
            "cmd": "find_object_in_area", 
            "name": name,
            "minX": int(min_x),
            "maxX": int(max_x), 
            "minY": int(min_y),
            "maxY": int(max_y)
        }
        if types:
            msg["types"] = types
        return self._send(msg) or {}

    def get_ground_items(self, name: str = None, radius: int = None) -> dict:
        """Get ground items."""
        req = {"cmd": "ground_items"}
        if name:
            req["name"] = name
        if radius:
            req["radius"] = radius
        return self._send(req) or {}

    def get_camera(self) -> dict:
        """Get camera information."""
        return self._send({"cmd": "get_camera"}) or {}

    def get_tab(self) -> dict:
        """Get current tab information."""
        return self._send({"cmd": "tab"}) or {}

    def get_ge(self) -> dict:
        """Get Grand Exchange information using comprehensive widget data."""
        return self.get_ge_widgets()

    def get_ge_widgets(self) -> dict:
        """Get all GE widgets with comprehensive data."""
        return self._send({"cmd": "get_ge_widgets"}) or {}


    def get_inventory_widgets(self) -> dict:
        """Get inventory widget information."""
        return self._send({"cmd": "get_inventory_widgets"}) or {}

    def get_crafting_widgets(self) -> dict:
        """Get crafting widget information."""
        return self._send({"cmd": "get_crafting_widgets"}) or {}

    def get_widget_info(self, widget_id: int) -> dict:
        """Get widget information by ID."""
        return self._send({"cmd": "get_widget_info", "widget_id": widget_id}) or {}

    def get_widget_children(self, widget_id: int) -> list:
        """Get widget children by parent ID."""
        return self._send({"cmd": "get_widget_children", "widget_id": widget_id}) or []

    def get_equipment_inventory(self) -> dict:
        """Get equipment inventory information."""
        return self._send({"cmd": "get_equipment_inventory"}) or {}

    def get_tutorial(self) -> dict:
        """Get tutorial information."""
        return self._send({"cmd": "get_tutorial"}) or {}

    def get_character_design(self) -> dict:
        """Get character design information."""
        return self._send({"cmd": "get_character_design"}) or {}

    def get_quests(self) -> dict:
        """Get quest information."""
        return self._send({"cmd": "get_quests"}) or {}

    def get_spellbook(self) -> dict:
        """Get spellbook information."""
        return self._send({"cmd": "get_spellbook"}) or {}

    def get_menu(self) -> dict:
        """Get menu information."""
        return self._send({"cmd": "menu"}) or {}

    def get_varp(self, varp_id: int) -> dict:
        """Get VarPlayer value by ID."""
        return self._send({"cmd": "get_varp", "id": varp_id}) or {}

    def get_var(self, var_id: int, timeout: float = None) -> dict:
        """Get variable value by ID."""
        return self._send({"cmd": "get-var", "id": int(var_id)}, timeout=timeout) or {}

    def get_mask(self) -> dict:
        """Get mask information."""
        return self._send({"cmd": "get_mask"}) or {}

    def where(self) -> dict:
        """Get window dimensions and position."""
        return self._send({"cmd": "where"}) or {}

    def widget_exists(self, widget_id: int) -> dict:
        """Check if a widget exists and is visible."""
        return self._send({"cmd": "widget_exists", "widget_id": int(widget_id)}) or {}

    def get_bank_items(self) -> dict:
        """Get all bank item slots and their data."""
        return self._send({"cmd": "get_bank_items"}) or {}

    def get_bank_tabs(self) -> dict:
        """Get bank organization tabs."""
        return self._send({"cmd": "get_bank_tabs"}) or {}

    def get_bank_quantity_buttons(self) -> dict:
        """Get bank withdraw quantity buttons (1, 5, 10, X, All)."""
        return self._send({"cmd": "get_bank_quantity_buttons"}) or {}

    def get_bank_deposit_buttons(self) -> dict:
        """Get bank deposit buttons (inventory, equipment)."""
        return self._send({"cmd": "get_bank_deposit_buttons"}) or {}

    def get_bank_note_toggle(self) -> dict:
        """Get bank note/item toggle buttons."""
        return self._send({"cmd": "get_bank_note_toggle"}) or {}

    def get_bank_search(self) -> dict:
        """Get bank search interface widgets."""
        return self._send({"cmd": "get_bank_search"}) or {}

    def get_bank_xvalue(self) -> dict:
        """Get current bank withdraw mode and X value using RuneLite Varbits."""
        return self._send({"cmd": "bank-xvalue"}) or {}

    # ===== GRAND EXCHANGE METHODS =====
    
    def get_ge_offers(self) -> dict:
        """Get all GE offer slots (465.2[0] through 465.2[11])."""
        return self._send({"cmd": "get_ge_offers"}) or {}

    def get_ge_setup(self) -> dict:
        """Get GE setup widgets (465.26[0] through 465.26[58])."""
        return self._send({"cmd": "get_ge_setup"}) or {}

    def get_ge_confirm(self) -> dict:
        """Get GE confirm widgets (465.30[0] through 465.30[8])."""
        return self._send({"cmd": "get_ge_confirm"}) or {}

    def get_ge_buttons(self) -> dict:
        """Get main GE buttons (BACK, INDEX, COLLECTALL)."""
        return self._send({"cmd": "get_ge_buttons"}) or {}

    # ===== PATHFINDING METHODS =====
    
    def get_path(self, rect=None, goal=None, max_wps=15):
        """Get pathfinding waypoints without visualization."""
        req = {"cmd": "path", "visualize": False}  # Disable visualization
        if rect:
            minX, maxX, minY, maxY = rect
            req.update({"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY})
        if goal:
            gx, gy = goal
            req.update({"goalX": gx, "goalY": gy})
        if max_wps is not None:
            req["maxWps"] = int(max_wps)

        resp = self._send(req)
        wps = resp.get("waypoints", []) or [] if resp else []
        return wps, resp

    def path(self, rect=None, goal=None, visualize=True, max_wps=None):
        """Get pathfinding waypoints with optional visualization."""
        req = {"cmd": "path", "visualize": visualize}
        if rect:
            minX, maxX, minY, maxY = rect
            req.update({"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY})
        if goal:
            gx, gy = goal
            req.update({"goalX": gx, "goalY": gy})

        resp = self._send(req)
        wps = resp.get("waypoints", []) or [] if resp else []
        
        # Filter waypoints to only include those within 14x14 box around player
        if wps:
            player_info = self.get_player()
            if player_info and player_info.get("ok"):
                player_x = player_info['player'].get("worldX")
                player_y = player_info['player'].get("worldY")
                
                if isinstance(player_x, int) and isinstance(player_y, int):
                    # Calculate 29x29 box bounds (14 tiles in each direction from player)
                    min_x = player_x - 14
                    max_x = player_x + 14
                    min_y = player_y - 14
                    max_y = player_y + 14
                    
                    # Filter waypoints to only include those within the box
                    filtered_wps = []
                    for wp in wps:
                        wp_x = wp.get("x")
                        wp_y = wp.get("y")
                        if (isinstance(wp_x, int) and isinstance(wp_y, int) and
                            min_x <= wp_x <= max_x and min_y <= wp_y <= max_y):
                            filtered_wps.append(wp)
                        else:
                            # Stop at first waypoint outside the box
                            break
                    
                    wps = filtered_wps
        
        return wps, resp

    def project_world_tile(self, world_x: int, world_y: int) -> dict:
        """Project world tile to canvas coordinates."""
        try:
            return self._send({"cmd": "tilexy", "x": int(world_x), "y": int(world_y)}) or {}
        except Exception as e:
            return {"ok": False, "err": f"{type(e).__name__}: {e}"}

    def project_many(self, wps):
        """"
        Project world tiles to canvas AND preserve 'door' metadata.
        Returns:
          out: [{onscreen, canvas, world:{x,y,p}, door?}, ...]
          dbg: raw resp from IPC
        """
        tiles = [{"x": int(w["x"]), "y": int(w["y"])} for w in wps]
        resp = self._send({"cmd": "tilexy_many", "tiles": tiles})
        results = resp.get("results", []) or [] if resp else []

        out = []
        for i, (w, r) in enumerate(zip(wps, results)):
            row = {
                "onscreen": (r or {}).get("onscreen"),
                "canvas":  (r or {}).get("canvas"),
                "world":   {"x": int(w["x"]), "y": int(w["y"]), "p": int(w.get("p", 0))},
            }
                                # Keep door info from projection if present, else from the waypoint itself
            door = None
            if isinstance(r, dict) and isinstance(r.get("door"), dict):
                door = r["door"]
            elif isinstance(w, dict) and isinstance(w.get("door"), dict):
                door = w["door"]
            if door:
                row["door"] = door

            out.append(row)

        return out, resp

    # ===== DOOR TRAVERSAL =====

    def check_door_traversal(self, door_id: int) -> dict:
        """Check if a door can be traversed."""
        return self._send({"cmd": "check_door_traversal", "door_id": door_id}) or {}
    
    def get_ge_prices(self) -> dict:
        """Get current GE prices from RuneLite GE plugin."""
        return self._send({"cmd": "get_ge_prices"}) or {}
    
    def get_last_interaction(self) -> dict:
        """Get the last interaction data from StateExporter2Plugin."""
        return self._send({"cmd": "get_last_interaction"}) or {}
    
    def get_players(self) -> dict:
        """Get information about all players around the local player."""
        return self._send({"cmd": "get_players"}) or {}
    
    def get_world(self) -> dict:
        """Get the current world number."""
        return self._send({"cmd": "get_world"}) or {}
    
    def get_worlds(self) -> dict:
        """
        Get the available world list (expects a list under key 'worlds').

        Expected world objects typically include:
          - id (int)
          - members (bool)
        """
        return self._send({"cmd": "get_worlds"}) or {}

    def hop_world(self, world_id: int) -> dict:
        """Hop to a specific world."""
        return self._send({"cmd": "hop_world", "world_id": world_id}) or {}
    
    def open_world_hopper(self) -> dict:
        """Open the world hopper interface."""
        return self._send({"cmd": "openWorldHopper"}) or {}
    
    def _capture_last_interaction(self):
        """Capture the last interaction data and store it globally."""
        global _last_interaction
        try:
            # Small delay to ensure the interaction is captured
            from .utils import sleep_exponential
            time.sleep(0.1)
            
            # Get the last interaction data via IPC
            result = self._send({"cmd": "get_last_interaction"})
            if result and result.get("ok"):
                _last_interaction = result.get("interaction")
        except Exception as e:
            print(f"[IPC] get_last_interaction error: {e}")


def get_last_interaction():
    """Get the last captured interaction data."""
    global _last_interaction
    return _last_interaction