import logging
from .runtime_utils import ipc, dispatch
from .utils import norm_name, clean_rs
from .widgets import unwrap_rect, widget_exists
from ..actions import widgets


def ge_open() -> bool:
  from .runtime_utils import ipc
  # Check if GE main widget (465.1) exists and is visible
  ge_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
  return bool(ge_data.get("ok", False) and ge_data.get("children"))


def ge_widgets() -> dict:
  from .runtime_utils import ipc
  # Get all GE widgets using the comprehensive command
  ge_data = ipc.get_ge_widgets() or {}
  if ge_data.get("ok"):
    return {str(widget.get("id", "")): widget for widget in ge_data.get("widgets", [])}
  return {}


def ge_widget(key: str) -> dict | None:
  return ge_widgets().get(key)


def ge_offer_text_contains_anywhere(substr: str) -> bool:
  """Scan the 30474266 tree for a substring in any 'text'/'textStripped'."""
  W = ge_widgets()
  sub = norm_name(substr)
  for k, v in W.items():
    if not (isinstance(k, str) and (k == "30474266" or k.startswith("30474266:"))):
      continue
    t = norm_name((v or {}).get("text") or (v or {}).get("textStripped"))
    if t and sub in t:
      return True
  return False

def ge_offer_open() -> bool:
    """
    Offer panel is OPEN iff GE is open AND we can find setup-related widgets.
    """
    if not ge_open():
        return False

    # Look for setup widgets using widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return False
    
    # If we have children in the setup widget, offer panel is open
    return bool(setup_data.get("children"))

def ge_selected_item_is(name: str) -> bool:
  from .runtime_utils import ipc
  # Look for the selected item text in the setup widget children
  setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
  if not setup_data.get("ok"):
    return False
  
  children = setup_data.get("children", [])
  for child in children:
    text = (child.get("text") or "").strip()
    if text and norm_name(text) == norm_name(name):
      return True
  return False


def ge_first_buy_slot_btn() -> dict | None:
  from .runtime_utils import ipc
  # Look for offer slot widgets using widget children
  offers_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
  if not offers_data.get("ok"):
    return None
  
  children = offers_data.get("children", [])
  for child in children:
    widget_id = child.get("id", 0)
    # Check for offer slot widgets (30474242-30474253 range for 465.2[0-11])
    if 30474242 <= widget_id <= 30474253:
      return child
  return None

def ge_offer_item_label() -> str | None:
  from .runtime_utils import ipc
  # Look for item name in setup widget children
  setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
  if not setup_data.get("ok"):
    return None
  
  children = setup_data.get("children", [])
  for child in children:
    text = (child.get("text") or "").strip()
    if text and len(text) > 0:
      return norm_name(text)
  return None

def widget_by_id_text(wid: int, txt: str | None) -> dict | None:
    from .runtime_utils import ipc
    # Get widget by ID using widget children from the appropriate parent
    # For GE widgets, we need to check the main GE container
    ge_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not ge_data.get("ok"):
        return None
    
    children = ge_data.get("children", [])
    for child in children:
        if child.get("id") == wid:
            if txt is None:
                return child
            text = (child.get("text") or "").strip()
            if norm_name(text) == norm_name(txt):
                return child
    return None


def widget_by_id_text_contains(wid: int, substr: str) -> dict | None:
    from .runtime_utils import ipc
    sub = norm_name(substr)

    data = widgets.get_widget_info(wid)
    if not data:
        return None

    data = widgets.get_widget_info(wid).get("data")
    text = (data.get("text") or "").strip()
    if sub in norm_name(text) and data.get("bounds"):
        return data

    data = ipc.get_widget_children(wid)
    if not data.get("ok"):
        return None

    children = data.get("children", [])
    for child in children:
        if child.get("id") == wid:
            text = (child.get("text") or "").strip()
            if sub in norm_name(text) and child.get("bounds"):
                return child
    return None

def widget_by_id_sprite(parent_wid: int, sprite_id: int) -> dict | None:
  # Sprite-based widget finding not supported with current targeted commands
  return None

def ge_buy_minus_widget() -> dict | None:
    return widget_by_id_text(30474266, "-5%")


def ge_buy_confirm_widget() -> dict | None:
  return widget_by_id_text_contains(30474266, "confirm")

def ge_price_widget() -> dict | None:
    """Find the widget whose text looks like '51,300 coins (...)'."""
    from .runtime_utils import ipc
    # Look for price widget in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").lower()
        if "coins" in text and child.get("bounds"):
            return child
    return None

def ge_price_value() -> int | None:
    """Return the integer price before the word 'coins', e.g. 51300 from '51,300 coins (...)'."""
    w = ge_price_widget()
    if not w:
        return None
    txt = (w.get("text") or "").split(" coins", 1)[0].strip()
    # strip commas and tags if any slipped in
    txt = txt.replace(",", "")
    try:
        return int(txt)
    except Exception:
        return None

def ge_inv_item_by_name(name: str) -> dict | None:
    from .runtime_utils import ipc
    # Look for inventory items in GE inventory widget children
    # GE inventory is typically in widget 465.15 (30474255)
    inv_data = ipc.get_widget_children(30474255)  # 465.15 GeOffers.INVENTORY
    if not inv_data.get("ok"):
        return None
    
    children = inv_data.get("children", [])
    for child in children:
        # Check both name and text fields for item name
        child_name = child.get("name", "")
        child_text = child.get("text", "")
        nm = norm_name(child_name or child_text)
        if nm == norm_name(name):
            return child
    return None

def find_ge_plus5_bounds():
    """
    Locate the '+5%' price adjust button on the GE Buy offer.
    Returns {x, y, width, height} or None.
    """
    from .runtime_utils import ipc
    # Look for +5% button in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").strip().replace(" ", "")
        if text.lower() == "+5%":
            bounds = child.get("bounds", {})
            if bounds and int(bounds.get("width", 0)) > 0 and int(bounds.get("height", 0)) > 0:
                return bounds
    
    return None

def ge_qty_button() -> dict | None:
    from .runtime_utils import ipc
    # Look for quantity button in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").strip()
        if "..." in text:
            return child
    return None

def ge_qty_value_widget() -> dict | None:
    from .runtime_utils import ipc
    # Look for quantity value widget in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    for child in children:
        text = (clean_rs(child.get("text")) or "")
        text = text.replace(',', '')
        id =  (child.get("id") or "")
        if text.isdigit() and id == 30474266:
            return child
    return None

def ge_qty_matches(want_qty: int) -> bool:
    w = ge_qty_value_widget() or {}
    t = int(w.get("text").replace(',', ''))
    return want_qty == t  # GE usually shows exact number; `in` is robust to formatting

def chatbox_qty_prompt_visible() -> bool:
    """
    We'll treat the quantity prompt as visible if BOTH 10616874 and 10616875 exist.
    Your exporter will publish them under payload['chatbox']['widgets'] (see exporter patch).
    """
    from .runtime_utils import ipc
    chatbox_data = ipc.get_chat() or {}
    cbw = chatbox_data.get("widgets") or {}
    return ("10616874" in cbw) and ("10616875" in cbw)


def chat_qty_prompt_active() -> bool:
    """
    Check if the GE quantity prompt is active by looking for a child widget
    with the text "How many do you wish to buy?".
    """
    from .runtime_utils import ipc

    ge_chatbox_data = ipc.get_widget_children(10616866).get('children') or []

    # Search through the children list for the quantity prompt text
    for child in ge_chatbox_data:
        if isinstance(child, dict):
            text = child.get('text', '')
            if text and ('How many do you wish to buy?' in text or 'Enter amount' in text or 'Set a price for each item' in text):
                return True

    return False

def ge_inv_slot_bounds(item_name: str | int) -> dict | None:
    """
    Returns the bounds dict for a GE-inventory item chosen by name.
    - Prefer exact match on nameStripped (e.g., 'Coins', 'Sapphire ring').
    - Falls back to exact match on raw 'name' and then substring contains.
    - If item_name is int, preserves legacy behavior: uses items[index] when valid.
    """
    from .runtime_utils import ipc
    # Look for inventory items in GE inventory widget children
    inv_data = ipc.get_widget_children(30474255)  # 465.15 GeOffers.INVENTORY
    if not inv_data.get("ok"):
        return None
    
    children = inv_data.get("children", [])

    # Legacy: allow slot index
    if isinstance(item_name, int):
        try:
            child = children[item_name] if item_name < len(children) else None
            if child:
                rect = unwrap_rect(child.get("bounds"))
                return rect if rect and rect.get("x", -1) >= 0 and rect.get("y", -1) >= 0 else None
        except Exception:
            return None

    needle = norm_name(str(item_name))

    def _nm(child):
        # prefer name field; fall back to text
        return norm_name(child.get("name") or child.get("text"))

    target = None

    # 1) exact match on name/text
    for child in children:
        if _nm(child) == needle:
            target = child
            break

    # 2) contains match if exact not found
    if not target and needle:
        for child in children:
            nm = _nm(child)
            if needle in nm:
                target = child
                break

    rect = unwrap_rect((target or {}).get("bounds"))
    if not rect:
        return None
    # ignore invisible placeholders (-1, -1)
    if int(rect.get("x", -1)) < 0 or int(rect.get("y", -1)) < 0:
        return None
    return rect

def price(name: str) -> int:
    # This function would need price data from a different source
    # For now, return 0 as we don't have price data in widget children
    return 0

def nearest_clerk() -> dict | None:
    from .runtime_utils import ipc
    player_data = ipc.get_player() or {}
    me = player_data.get("player") or {}
    mx, my, mp = int(me.get("worldX") or 0), int(me.get("worldY") or 0), int(me.get("plane") or 0)

    closest_npcs = ipc.get_npcs() or []
    all_npcs = ipc.get_npcs() or []
    
    best, best_d2 = None, 1e18
    for npc in closest_npcs + all_npcs:
        nm = (npc.get("name") or "").lower()
        nid = int(npc.get("id") or -1)
        if "grand exchange clerk" not in nm and not (2148 <= nid <= 2151):
            continue
        if int(npc.get("plane") or 0) != mp:
            continue
        nx, ny = int(npc.get("worldX") or 0), int(npc.get("worldY") or 0)
        dx, dy = nx - mx, ny - my
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = npc, d2
    return best


def selected_item_matches(name: str) -> bool:
    W = ge_widgets()
    w = W.get("30474266:27") or {}
    t = norm_name(w.get("text") or w.get("textStripped"))
    return bool(t) and (t == norm_name(name))


# ===== NEW TARGETED GE COMMAND HELPERS =====

def get_ge_offers() -> list[dict]:
    """Get all GE offer slots using widget children."""
    from .runtime_utils import ipc
    # Get offer slots from the main GE container
    offers_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not offers_data.get("ok"):
        return []
    
    children = offers_data.get("children", [])
    # Filter for offer slot widgets (30474242-30474253 range)
    offer_slots = []
    for child in children:
        widget_id = child.get("id", 0)
        if 30474242 <= widget_id <= 30474253:
            offer_slots.append(child)
    return offer_slots

def get_ge_history() -> list[dict]:
    """Get GE history slots using widget children."""
    from .runtime_utils import ipc
    # Get history slots from the main GE container
    history_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not history_data.get("ok"):
        return []
    
    children = history_data.get("children", [])
    # Filter for history slot widgets (30474254-30474265 range)
    history_slots = []
    for child in children:
        widget_id = child.get("id", 0)
        if 30474254 <= widget_id <= 30474265:
            history_slots.append(child)
    return history_slots

def get_ge_setup() -> list[dict]:
    """Get GE setup widgets using widget children."""
    from .runtime_utils import ipc
    # Get setup widgets from the setup container
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return []
    return setup_data.get("children", [])

def get_ge_confirm() -> list[dict]:
    """Get GE confirm widgets using widget children."""
    from .runtime_utils import ipc
    # Get confirm widgets from the confirm container
    confirm_data = ipc.get_widget_children(30474270)  # 465.30 GeOffers.CONFIRM
    if not confirm_data.get("ok"):
        return []
    return confirm_data.get("children", [])

def get_ge_buttons() -> list[dict]:
    """Get main GE buttons using widget children."""
    from .runtime_utils import ipc
    # Get buttons from the main GE container
    buttons_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not buttons_data.get("ok"):
        return []
    
    children = buttons_data.get("children", [])
    # Filter for button widgets (back, index, collect_all, etc.)
    buttons = []
    for child in children:
        widget_id = child.get("id", 0)
        # Check for button widgets (30474246-30474248 range for main buttons)
        if 30474246 <= widget_id <= 30474248:
            buttons.append(child)
    return buttons

def find_ge_button_by_name(button_name: str) -> dict | None:
    """Find a GE button by name (back, index, collect_all, index_0)."""
    buttons = get_ge_buttons()
    for button in buttons:
        if button.get("name") == button_name:
            return button
    return None

def find_ge_setup_widget_by_text(text: str) -> dict | None:
    """Find a GE setup widget by text content."""
    setup_widgets = get_ge_setup()
    for widget in setup_widgets:
        widget_text = (widget.get("text") or "").strip()
        if text.lower() in widget_text.lower():
            return widget
    return None

def find_ge_confirm_widget_by_text(text: str) -> dict | None:
    """Find a GE confirm widget by text content."""
    confirm_widgets = get_ge_confirm()
    for widget in confirm_widgets:
        widget_text = (widget.get("text") or "").strip()
        if text.lower() in widget_text.lower():
            return widget
    return None


# ===== GE OFFER SCREEN METHODS =====

def is_offer_screen_open() -> bool:
    """Check if the GE offer setup screen is open."""
    return widget_exists(30474266)


def open_history() -> dict | None:
    """Open the GE history screen."""
    try:
        # Get the history button widget using get_widget_children
        main_data = ipc.get_widget_children(30474242)  # Main container
        if not main_data.get("ok"):
            logging.error("[open_history] helpers/ge.py: Failed to get main container widget data")
            return None
        
        children = main_data.get("children", [])
        history_button = None
        
        # Find the history button by looking for the History button (index 13 from widget analysis)
        for i, child in enumerate(children):
            if i == 13:  # History button is at index 13
                history_button = child
                break
        
        if not history_button:
            logging.error("[open_history] helpers/ge.py: History button not found")
            return None
        
        bounds = history_button.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[open_history] helpers/ge.py: History button has invalid bounds")
            return None
        
        step = {
            "action": "ge-open-history",
            "click": {"type": "rect-center"},
            "target": {"domain": "ge-widget", "name": "History", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[open_history] helpers/ge.py: {e}")
        return None


def click_item_search() -> dict | None:
    """Click the item search icon to open item selection."""
    try:
        # Get the item search icon widget using get_widget_children
        setup_data = ipc.get_widget_children(30474266)  # Setup container
        if not setup_data.get("ok"):
            logging.error("[click_item_search] helpers/ge.py: Failed to get setup container widget data")
            return None
        
        children = setup_data.get("children", [])
        search_icon = None
        
        # Find the item search icon by looking for the search icon (index 62 from widget analysis)
        for i, child in enumerate(children):
            if i == 62:  # Item search icon is at index 62
                search_icon = child
                break
        
        if not search_icon:
            logging.error("[click_item_search] helpers/ge.py: Item search icon not found")
            return None
        
        bounds = search_icon.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[click_item_search] helpers/ge.py: Item search icon has invalid bounds")
            return None
        
        step = {
            "action": "ge-click-item-search",
            "click": {"type": "rect-center"},
            "target": {"domain": "ge-widget", "name": "Item Search", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[click_item_search] helpers/ge.py: {e}")
        return None


def set_quantity(amount: int) -> dict | None:
    """Set the quantity using the appropriate button."""
    try:
        if amount == 1:
            return _click_quantity_button("QUANTITY_PLUS_1")
        elif amount == 10:
            return _click_quantity_button("QUANTITY_PLUS_10")
        elif amount == 100:
            return _click_quantity_button("QUANTITY_PLUS_100")
        elif amount == 1000:
            return _click_quantity_button("QUANTITY_PLUS_1K")
        else:
            # For custom amounts, use the custom button
            return _click_quantity_button("QUANTITY_CUSTOM")
    except Exception as e:
        logging.error(f"[set_quantity] helpers/ge.py: {e}")
        return None


def _click_quantity_button(button_name: str) -> dict | None:
    """Click a quantity button."""
    try:
        # Get the setup container widgets using get_widget_children
        setup_data = ipc.get_widget_children(30474266)  # Setup container
        if not setup_data.get("ok"):
            logging.error(f"[_click_quantity_button] helpers/ge.py: Failed to get setup container widget data")
            return None
        
        children = setup_data.get("children", [])
        target_button = None
        
        # Map button names to their indices from widget analysis
        button_indices = {
            "QUANTITY_PLUS_1": 41,
            "QUANTITY_PLUS_10": 42,
            "QUANTITY_PLUS_100": 43,
            "QUANTITY_PLUS_1K": 44,
            "QUANTITY_CUSTOM": 45,
            "QUANTITY_LEFT_ARROW": 39,
            "QUANTITY_RIGHT_ARROW": 40,
        }
        
        button_index = button_indices.get(button_name)
        if button_index is None:
            logging.error(f"[_click_quantity_button] helpers/ge.py: Unknown button name '{button_name}'")
            return None
        
        # Find the button by index
        for i, child in enumerate(children):
            if i == button_index:
                target_button = child
                break
        
        if not target_button:
            logging.error(f"[_click_quantity_button] helpers/ge.py: {button_name} button not found at index {button_index}")
            return None
        
        bounds = target_button.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error(f"[_click_quantity_button] helpers/ge.py: {button_name} button has invalid bounds")
            return None
        
        step = {
            "action": f"ge-click-{button_name.lower()}",
            "click": {"type": "rect-center"},
            "target": {"domain": "ge-widget", "name": button_name.replace("_", " ").title(), "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[_click_quantity_button] helpers/ge.py: {e}")
        return None


def adjust_quantity(direction: str) -> dict | None:
    """Adjust quantity using arrow buttons."""
    try:
        if direction.lower() == "up" or direction.lower() == "increase":
            return _click_quantity_button("QUANTITY_RIGHT_ARROW")
        elif direction.lower() == "down" or direction.lower() == "decrease":
            return _click_quantity_button("QUANTITY_LEFT_ARROW")
        else:
            logging.error(f"[adjust_quantity] helpers/ge.py: Invalid direction '{direction}', use 'up'/'down' or 'increase'/'decrease'")
            return None
    except Exception as e:
        logging.error(f"[adjust_quantity] helpers/ge.py: {e}")
        return None


def set_price_percentage(percentage: int) -> dict | None:
    """Set price using percentage buttons."""
    try:
        if percentage == -5:
            return _click_price_button("PRICE_MINUS_5_PERCENT")
        elif percentage == 5:
            return _click_price_button("PRICE_PLUS_5_PERCENT")
        elif percentage < 0:
            return _click_price_button("PRICE_MINUS_X_PERCENT")
        elif percentage > 0:
            return _click_price_button("PRICE_PLUS_X_PERCENT")
        else:
            logging.error(f"[set_price_percentage] helpers/ge.py: Invalid percentage '{percentage}'")
            return None
    except Exception as e:
        logging.error(f"[set_price_percentage] helpers/ge.py: {e}")
        return None


def _click_price_button(button_name: str) -> dict | None:
    """Click a price button."""
    try:
        # Get the setup container widgets using get_widget_children
        setup_data = ipc.get_widget_children(30474266)  # Setup container
        if not setup_data.get("ok"):
            logging.error(f"[_click_price_button] helpers/ge.py: Failed to get setup container widget data")
            return None
        
        children = setup_data.get("children", [])
        target_button = None
        
        # Map button names to their indices from widget analysis
        button_indices = {
            "PRICE_MINUS_5_PERCENT": 48,
            "PRICE_PLUS_5_PERCENT": 51,
            "PRICE_MINUS_X_PERCENT": 52,
            "PRICE_PLUS_X_PERCENT": 53,
            "PRICE_LEFT_ARROW": 46,
            "PRICE_RIGHT_ARROW": 47,
            "MARKET_PRICE_BUTTON": 49,
        }
        
        button_index = button_indices.get(button_name)
        if button_index is None:
            logging.error(f"[_click_price_button] helpers/ge.py: Unknown button name '{button_name}'")
            return None
        
        # Find the button by index
        for i, child in enumerate(children):
            if i == button_index:
                target_button = child
                break
        
        if not target_button:
            logging.error(f"[_click_price_button] helpers/ge.py: {button_name} button not found at index {button_index}")
            return None
        
        bounds = target_button.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error(f"[_click_price_button] helpers/ge.py: {button_name} button has invalid bounds")
            return None
        
        step = {
            "action": f"ge-click-{button_name.lower()}",
            "click": {"type": "rect-center"},
            "target": {"domain": "ge-widget", "name": button_name.replace("_", " ").title(), "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[_click_price_button] helpers/ge.py: {e}")
        return None


def adjust_price(direction: str) -> dict | None:
    """Adjust price using arrow buttons."""
    try:
        if direction.lower() == "up" or direction.lower() == "increase":
            return _click_price_button("PRICE_RIGHT_ARROW")
        elif direction.lower() == "down" or direction.lower() == "decrease":
            return _click_price_button("PRICE_LEFT_ARROW")
        else:
            logging.error(f"[adjust_price] helpers/ge.py: Invalid direction '{direction}', use 'up'/'down' or 'increase'/'decrease'")
            return None
    except Exception as e:
        logging.error(f"[adjust_price] helpers/ge.py: {e}")
        return None


def set_market_price() -> dict | None:
    """Set price to current market price."""
    try:
        return _click_price_button("MARKET_PRICE_BUTTON")
    except Exception as e:
        logging.error(f"[set_market_price] helpers/ge.py: {e}")
        return None


def confirm_offer() -> dict | None:
    """Confirm and place the offer."""
    try:
        # Get the confirm button widget using get_widget_children
        main_data = ipc.get_widget_children(30474242)  # Main container
        if not main_data.get("ok"):
            logging.error("[confirm_offer] helpers/ge.py: Failed to get main container widget data")
            return None
        
        children = main_data.get("children", [])
        confirm_button = None
        
        # Find the confirm button by looking for the confirm button (index 24 from widget analysis)
        for i, child in enumerate(children):
            if i == 24:  # Confirm button is at index 24
                confirm_button = child
                break
        
        if not confirm_button:
            logging.error("[confirm_offer] helpers/ge.py: Confirm button not found")
            return None
        
        bounds = confirm_button.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[confirm_offer] helpers/ge.py: Confirm button has invalid bounds")
            return None
        
        step = {
            "action": "ge-confirm-offer",
            "click": {"type": "rect-center"},
            "target": {"domain": "ge-widget", "name": "Confirm Offer", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[confirm_offer] helpers/ge.py: {e}")
        return None


def get_offer_screen_data() -> dict | None:
    """Get all offer screen widget data for analysis."""
    try:
        # Get the main container and all its children
        response = ipc.get_widget_children(30474242)  # Main container
        if not response or not response.get("ok"):
            logging.error("[get_offer_screen_data] helpers/ge.py: Failed to get offer screen widget data")
            return None
        
        return response.get("children", [])
    except Exception as e:
        logging.error(f"[get_offer_screen_data] helpers/ge.py: {e}")
        return None


def is_offer_ready() -> bool:
    """Check if the offer is ready to be confirmed (has item, quantity, and price)."""
    try:
        # This would need to be implemented based on the actual widget states
        # For now, just check if the screen is open
        return is_offer_screen_open()
    except Exception as e:
        logging.error(f"[is_offer_ready] helpers/ge.py: {e}")
        return False


def get_current_quantity() -> str | None:
    """Get the current quantity value from the offer screen."""
    try:
        # This would need to be implemented by reading the quantity text field
        # For now, return None as we'd need to identify the specific text widget
        return None
    except Exception as e:
        logging.error(f"[get_current_quantity] helpers/ge.py: {e}")
        return None


def get_current_price() -> str | None:
    """Get the current price value from the offer screen."""
    try:
        # This would need to be implemented by reading the price text field
        # For now, return None as we'd need to identify the specific text widget
        return None
    except Exception as e:
        logging.error(f"[get_current_price] helpers/ge.py: {e}")
        return None