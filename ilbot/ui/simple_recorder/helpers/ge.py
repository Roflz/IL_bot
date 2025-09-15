from .context import get_payload
from .utils import norm_name
from .widgets import unwrap_rect


def ge_open(payload: dict) -> bool:
  return bool(((payload.get("grand_exchange") or {}).get("open")))


def ge_widgets(payload: dict) -> dict:
  return (payload.get("grand_exchange") or {}).get("widgets") or {}


def ge_widget(payload: dict, key: str) -> dict | None:
  return (ge_widgets(payload) or {}).get(key)


def ge_offer_text_contains_anywhere(payload: dict, substr: str) -> bool:
  """Scan the 30474266 tree for a substring in any 'text'/'textStripped'."""
  W = ge_widgets(payload) or {}
  sub = norm_name(substr)
  for k, v in W.items():
    if not (isinstance(k, str) and (k == "30474266" or k.startswith("30474266:"))):
      continue
    t = norm_name((v or {}).get("text") or (v or {}).get("textStripped"))
    if t and sub in t:
      return True
  return False

def ge_offer_open(payload: dict) -> bool:
    """
    Offer panel is OPEN iff GE is open AND widget id 30474246 is NOT present.
    """
    if not ge_open(payload):
      return False

    W = ge_widgets(payload)
    if not isinstance(W, dict):
      return True  # no widgets exported → treat as offer open

    # If any key is "30474246" or a child like "30474246:*", chooser is visible → offer NOT open
    for k in W.keys():
      if k == "30474246" or k.startswith("30474246:"):
        return False

    return True


def ge_selected_item_is(payload: dict, name: str) -> bool:
  W = ge_widgets(payload) or {}
  w = W.get("30474266:27") or {}
  t = norm_name(w.get("text") or w.get("textStripped"))
  return bool(t) and t == norm_name(name)


def ge_first_buy_slot_btn(payload: dict) -> dict | None:
  # Step 1: "the 3rd widget" under any of 30474247..30474254
  W = ge_widgets(payload)
  for pid in (30474247, 30474248, 30474249, 30474250, 30474251, 30474252, 30474253, 30474254):
    w = W.get(f"{pid}:3")
    if w and w.get("bounds"):
      return w
  return None


def ge_offer_shell(payload: dict) -> dict | None:
  # Offer shell container: 30474266 root exists -> offer UI present
  return (ge_widgets(payload) or {}).get("30474266")


def ge_offer_item_label(payload: dict) -> str | None:
  # Your spec: selected item name sits in 30474266:27
  w = (ge_widgets(payload) or {}).get("30474266:27")
  t = (w or {}).get("text") or ""
  return norm_name(t) if t else None

def widget_by_id_text(wid: int, txt: str | None, payload: dict | None = None) -> dict | None:
    if payload is None:
        payload = get_payload()
    W = ge_widgets(payload)
    if not isinstance(W, dict):
      return None

    # No text constraint: return the root widget by id (if present)
    if txt is None:
      return W.get(str(wid))

    needle = norm_name(txt)

    # 1) Prefer children "wid:index" with exact text match
    for k, v in W.items():
      if not k.startswith(f"{wid}:"):
        continue
      vt = norm_name((v or {}).get("text"))
      if vt == needle:
        return v

    # 2) If root widget itself has matching text, allow that too
    root = W.get(str(wid))
    if root:
      rt = norm_name((root or {}).get("text"))
      if rt == needle:
        return root

    # No match on id+text
    return None

def widget_by_id_text_contains(payload: dict, wid: int, substr: str) -> dict | None:
  W = ge_widgets(payload)
  sub = norm_name(substr)
  for k, v in W.items():
    if not k.startswith(f"{wid}:"): continue
    vt = norm_name((v or {}).get("text"))
    if sub in vt and v.get("bounds"):
      return v
  return None


def widget_by_id_sprite(payload: dict, parent_wid: int, sprite_id: int) -> dict | None:
  W = ge_widgets(payload)
  for k, v in W.items():
    if not k.startswith(f"{parent_wid}:"): continue
    if int(v.get("spriteId") or -1) == int(sprite_id) and v.get("bounds"):
      return v
  return None

def ge_buy_minus_widget(payload: dict) -> dict | None:
    return widget_by_id_text(payload, 30474266, "-5%")


def ge_buy_confirm_widget(payload: dict) -> dict | None:
  return widget_by_id_text_contains(payload, 30474266, "confirm")

def ge_price_widget(payload: dict) -> dict | None:
    """Find the 30474266 child whose text looks like '51,300 coins (...)'."""
    W = (payload.get("grand_exchange") or {}).get("widgets") or {}
    best = None
    for k, v in W.items():
      if not k.startswith("30474266:"):
        continue
      t = (v or {}).get("text") or ""
      tl = t.lower()
      if "coins" in tl and v.get("bounds"):
        best = v
        # prefer longer text that includes the '(...-% )' tail
        # but we can just take the first match; break for determinism
        break
    return best

def ge_price_value(payload: dict) -> int | None:
    """Return the integer price before the word 'coins', e.g. 51300 from '51,300 coins (...)'."""
    w = ge_price_widget(payload)
    if not w:
        return None
    txt = (w.get("text") or "").split(" coins", 1)[0].strip()
    # strip commas and tags if any slipped in
    txt = txt.replace(",", "")
    try:
        return int(txt)
    except Exception:
        return None

def ge_inv_item_by_name(payload: dict, name: str) -> dict | None:
    inv = (payload.get("ge_inventory") or {})
    for it in (inv.get("items") or []):
        nm = norm_name(it.get("nameStripped") or it.get("name"))
        if nm == norm_name(name):
            return it
    return None

def find_ge_plus5_bounds(payload: dict):
    """
    Locate the '+5%' price adjust button on the GE Buy offer.
    Returns {x, y, width, height} or None.

    Strategy:
      1) Scan widget tree for a node whose text/textStripped equals '+5%' (case-insensitive, strip spaces).
      2) Prefer nodes under the GE group (30474266:*).
    """
    widgets = (payload or {}).get("widgets") or {}

    def iter_nodes(node):
        if not isinstance(node, dict):
            return
        yield node
        for child in (node.get("children") or []):
            yield from iter_nodes(child)

    def is_plus5(node):
        t = (node.get("textStripped") or node.get("text") or "").strip().replace(" ", "")
        return t.lower() == "+5%"

    # Pass 1: Prefer under GE root group (30474266:*)
    for k, root in widgets.items():
        if isinstance(k, str) and k.startswith("30474266:"):
            for n in iter_nodes(root):
                if is_plus5(n):
                    b = n.get("bounds") or {}
                    if b and int(b.get("width", 0)) > 0 and int(b.get("height", 0)) > 0:
                        return b

    # Pass 2: Search entire tree
    for root in widgets.values():
        for n in iter_nodes(root):
            if is_plus5(n):
                b = n.get("bounds") or {}
                if b and int(b.get("width", 0)) > 0 and int(b.get("height", 0)) > 0:
                    return b

    return None

def ge_qty_button(payload: dict) -> dict | None:
    # The literal "..." button: parent 30474266, child index 51
    return (ge_widgets(payload) or {}).get("30474266:51")

def ge_qty_value_widget(payload: dict) -> dict | None:
    # Where GE echoes the quantity after you confirm the number (child index 34)
    return (ge_widgets(payload) or {}).get("30474266:34")

def ge_qty_matches(payload: dict, want_qty: int) -> bool:
    w = ge_qty_value_widget(payload) or {}
    t = (w.get("text") or w.get("textStripped") or "")
    return str(want_qty) in t  # GE usually shows exact number; `in` is robust to formatting

def chatbox_qty_prompt_visible(payload: dict) -> bool:
    """
    We’ll treat the quantity prompt as visible if BOTH 10616874 and 10616875 exist.
    Your exporter will publish them under payload['chatbox']['widgets'] (see exporter patch).
    """
    cbw = (payload.get("chatbox") or {}).get("widgets") or {}
    return ("10616874" in cbw) and ("10616875" in cbw)

def chat_qty_prompt_active(payload: dict) -> bool:
    # prefer your exporter’s flag if present
    cb = payload.get("chatbox") or {}
    if isinstance(cb.get("promptActive"), bool):
        return cb["promptActive"]
    # fallback: require BOTH widgets and positive bounds
    W = ((ge_widgets(payload) or {}))  # not used here; keeping shape similar
    chat = (payload.get("chatbox") or {}).get("widgets") or {}
    a = chat.get("10616874") or {}
    b = chat.get("10616875") or {}
    def _ok(w):
        b = (w.get("bounds") or {})
        return bool(b.get("width")) and bool(b.get("height"))
    return _ok(a) and _ok(b)

def ge_inv_slot_bounds(payload: dict, item_name: str | int) -> dict | None:
    """
    Returns the bounds dict for a GE-inventory item chosen by name.
    - Prefer exact match on nameStripped (e.g., 'Coins', 'Sapphire ring').
    - Falls back to exact match on raw 'name' and then substring contains.
    - If item_name is int, preserves legacy behavior: uses items[index] when valid.
    """
    inv = (payload.get("ge_inventory") or {})
    items = inv.get("items") or []

    # Legacy: allow slot index
    if isinstance(item_name, int):
        try:
            rect = unwrap_rect((items[item_name] or {}).get("bounds"))
            return rect if rect and rect.get("x", -1) >= 0 and rect.get("y", -1) >= 0 else None
        except Exception:
            return None

    needle = norm_name(str(item_name))

    def _nm(it):
        # prefer stripped name; fall back to raw
        return norm_name(it.get("nameStripped") or it.get("name"))

    target = None

    # 1) exact match on stripped/raw
    for it in items:
        if _nm(it) == needle:
            target = it
            break

    # 2) contains match if exact not found
    if not target and needle:
        for it in items:
            nm = _nm(it)
            if needle in nm:
                target = it
                break

    rect = unwrap_rect((target or {}).get("bounds"))
    if not rect:
        return None
    # ignore invisible placeholders (-1, -1)
    if int(rect.get("x", -1)) < 0 or int(rect.get("y", -1)) < 0:
        return None
    return rect

def price(payload: dict, name: str) -> int:
    p = (payload.get("ge_prices") or {}).get(name, 0)
    try: return int(p)
    except Exception: return 0

def nearest_clerk(payload: dict) -> dict | None:
    me = payload.get("player") or {}
    mx, my, mp = int(me.get("worldX") or 0), int(me.get("worldY") or 0), int(me.get("plane") or 0)

    best, best_d2 = None, 1e18
    for npc in (payload.get("closestNPCs") or []) + (payload.get("npcs") or []):
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
    w = ge_widgets.get("30474266:27") or {}
    t = norm_name(w.get("text") or w.get("textStripped"))
    return bool(t) and (t == norm_name(name))