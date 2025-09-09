# action_plans.py
import re
from typing import Dict, Callable
from .nav_simple import next_tile_toward
from .constants import GE_MIN_X, GE_MAX_X, GE_MIN_Y, GE_MAX_Y

# --- tiny in-process memory to advance steps even when the plugin doesn't see IPC clicks ---
import time

from .constants import (
    GE_MIN_X, GE_MAX_X, GE_MIN_Y, GE_MAX_Y,
    EDGE_BANK_MIN_X, EDGE_BANK_MAX_X, EDGE_BANK_MIN_Y, EDGE_BANK_MAX_Y,
)


_STEP_HITS: dict[str, int] = {}

# --- GE helpers ---
def _ge_open(payload: dict) -> bool:
    return bool(((payload.get("grand_exchange") or {}).get("open")))

def _ge_widgets(payload: dict) -> dict:
    return (payload.get("grand_exchange") or {}).get("widgets") or {}

def _ge_offer_open(payload: dict) -> bool:
    """
    GE offer panel is considered OPEN iff the GE is open AND we do NOT see the
    placeholder text "Choose an item..." anywhere under grand_exchange.widgets.
    """
    if not _ge_open(payload):
        return False

    W = _ge_widgets(payload)
    if not isinstance(W, dict):
        # GE open but no widgets exported — follow your rule: absence of the text => offer open
        return True

    spriteId = 1108     # sprite id signifying main ge screen
    for v in W.values():
        t = v.get("spriteId")
        if t == spriteId:
            return False  # chooser visible => offer NOT open

    return True  # chooser absent => offer OPEN

def _ge_selected_item_is(payload: dict, name: str) -> bool:
    W = _ge_widgets(payload) or {}
    w = W.get("30474266:27") or {}
    t = _norm_name(w.get("text") or w.get("textStripped"))
    return bool(t) and t == _norm_name(name)


def _ge_first_buy_slot_btn(payload: dict) -> dict | None:
    # Step 1: "the 3rd widget" under any of 30474247..30474254
    W = _ge_widgets(payload)
    for pid in (30474247, 30474248, 30474249, 30474250, 30474251, 30474252, 30474253, 30474254):
        w = W.get(f"{pid}:3")
        if w and w.get("bounds"):
            return w
    return None

def _ge_offer_shell(payload: dict) -> dict | None:
    # Offer shell container: 30474266 root exists -> offer UI present
    return (_ge_widgets(payload) or {}).get("30474266")

def _ge_offer_item_label(payload: dict) -> str | None:
    # Your spec: selected item name sits in 30474266:27
    w = (_ge_widgets(payload) or {}).get("30474266:27")
    t = (w or {}).get("text") or ""
    return _norm_name(t) if t else None

def _ge_buy_minus_widget(payload: dict) -> dict | None:
    return _widget_by_id_text(payload, 30474266, "-5%")

def _ge_buy_confirm_widget(payload: dict) -> dict | None:
    return _widget_by_id_text_contains(payload, 30474266, "confirm")

def _widget_by_id_text(payload: dict, wid: int, txt: str | None) -> dict | None:
    W = _ge_widgets(payload)
    if not isinstance(W, dict):
        return None

    # No text constraint: return the root widget by id (if present)
    if txt is None:
        return W.get(str(wid))

    needle = _norm_name(txt)

    # 1) Prefer children "wid:index" with exact text match
    for k, v in W.items():
        if not k.startswith(f"{wid}:"):
            continue
        vt = _norm_name((v or {}).get("text"))
        if vt == needle:
            return v

    # 2) If root widget itself has matching text, allow that too
    root = W.get(str(wid))
    if root:
        rt = _norm_name((root or {}).get("text"))
        if rt == needle:
            return root

    # No match on id+text
    return None

def _find_ge_plus5_bounds(payload: dict):
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


# choose ring from GE-inventory by name (Sapphire → Emerald)
def _ge_inv_item_by_name(payload: dict, name: str) -> dict | None:
    inv = (payload.get("ge_inventory") or {})
    for it in (inv.get("items") or []):
        nm = _norm_name(it.get("nameStripped") or it.get("name"))
        if nm == _norm_name(name):
            return it
    return None

# --- GE price helpers ---
def _ge_price_widget(payload: dict) -> dict | None:
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

def _ge_price_value(payload: dict) -> int | None:
    """Return the integer price before the word 'coins', e.g. 51300 from '51,300 coins (...)'."""
    w = _ge_price_widget(payload)
    if not w:
        return None
    txt = (w.get("text") or "").split(" coins", 1)[0].strip()
    # strip commas and tags if any slipped in
    txt = txt.replace(",", "")
    try:
        return int(txt)
    except Exception:
        return None


def _widget_by_id_text_contains(payload: dict, wid: int, substr: str) -> dict | None:
    W = _ge_widgets(payload)
    sub = _norm_name(substr)
    for k, v in W.items():
        if not k.startswith(f"{wid}:"): continue
        vt = _norm_name((v or {}).get("text"))
        if sub in vt and v.get("bounds"):
            return v
    return None

def _widget_by_id_sprite(payload: dict, parent_wid: int, sprite_id: int) -> dict | None:
    W = _ge_widgets(payload)
    for k, v in W.items():
        if not k.startswith(f"{parent_wid}:"): continue
        if int(v.get("spriteId") or -1) == int(sprite_id) and v.get("bounds"):
            return v
    return None

def _rect_center_from_widget(w: dict | None) -> tuple[int | None, int | None]:
    rect = _unwrap_rect((w or {}).get("bounds"))
    return _rect_center_xy(rect)

def _inventory_ring_slots(payload: dict) -> list[dict]:
    out = []
    for s in _inv_slots(payload):
        nm = _norm_name(s.get("itemName"))
        if "ring" in nm and "mould" not in nm and int(s.get("quantity") or 0) > 0:
            out.append(s)
    return out

def _inv_slot_bounds(payload: dict, slot_id: int) -> dict | None:
    iw = (payload.get("inventory_widgets") or {}).get(str(slot_id)) or {}
    return _unwrap_rect(iw.get("bounds") if isinstance(iw, dict) else None)

def _ge_inv_slot_bounds(payload: dict, item_name: str | int) -> dict | None:
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
            rect = _unwrap_rect((items[item_name] or {}).get("bounds"))
            return rect if rect and rect.get("x", -1) >= 0 and rect.get("y", -1) >= 0 else None
        except Exception:
            return None

    needle = _norm_name(str(item_name))

    def _nm(it):
        # prefer stripped name; fall back to raw
        return _norm_name(it.get("nameStripped") or it.get("name"))

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

    rect = _unwrap_rect((target or {}).get("bounds"))
    if not rect:
        return None
    # ignore invisible placeholders (-1, -1)
    if int(rect.get("x", -1)) < 0 or int(rect.get("y", -1)) < 0:
        return None
    return rect


def _coins(payload: dict) -> int:
    return _inv_count(payload, "Coins")

def _price(payload: dict, name: str) -> int:
    p = (payload.get("ge_prices") or {}).get(name, 0)
    try: return int(p)
    except Exception: return 0

def _type_string_steps(s: str) -> list[dict]:
    steps = []
    for ch in (s or ""):
        key = ch.lower()
        if key == " ": key = "space"
        steps.append({"action": "click", "click": {"type": "key", "key": key}, "description": f"type '{ch}'"})
    return steps

def _now_ms() -> int:
    return int(time.time() * 1000)

def mark_step_done(step_id: str):
    """Record that we just executed this step (used to advance UI flows)."""
    if step_id:
        _STEP_HITS[step_id] = _now_ms()

def step_recent(step_id: str, max_ms: int = 1800) -> bool:
    """True if step_id was executed in the last max_ms milliseconds."""
    t = _STEP_HITS.get(step_id)
    return isinstance(t, int) and (_now_ms() - t) <= max_ms


def _bank_widget_rect(payload: dict, key: str) -> dict | None:
    """Return screen-rect for a bank widget exported under data.bank_widgets[key]."""
    w = ((payload.get("bank_widgets") or {}).get(key) or {})
    b = (w.get("bounds") if isinstance(w, dict) else None)
    if isinstance(b, dict) and all(k in b for k in ("x","y","width","height")):
        return b
    return None

def _bank_slots_matching(payload: dict, names: list[str]) -> list[dict]:
    """Return bank slots whose itemName matches (case-insensitive) any of names."""
    want = { (n or "").strip().lower() for n in names if n }
    out = []
    for s in (payload.get("bank", {}).get("slots") or []):
        nm = (s.get("itemName") or "").strip().lower()
        qty = int(s.get("quantity") or 0)
        if nm in want and qty > 0:
            out.append(s)
    return out



# ---------------- shared helpers (minimal copies to stay decoupled) ----------------
_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')
def _clean_rs(s: str | None) -> str:
    if not s:
        return ""
    return _RS_TAG_RE.sub('', s)

def _norm_name(s: str | None) -> str:
    return _clean_rs(s or "").strip().lower()

def _inv_slots(payload: dict) -> list[dict]:
    return (payload.get("inventory", {}) or {}).get("slots", []) or []

def _bank_slots(payload: dict) -> list[dict]:
    return (payload.get("bank", {}) or {}).get("slots", []) or []

def _inv_has(payload: dict, name: str) -> bool:
    n = _norm_name(name)
    return any(_norm_name(s.get("itemName")) == n for s in _inv_slots(payload))

def _inv_count(payload: dict, name: str) -> int:
    n = _norm_name(name)
    return sum(int(s.get("quantity") or 0) for s in _inv_slots(payload)
               if _norm_name(s.get("itemName")) == n)

def _first_inv_slot(payload: dict, name: str) -> dict | None:
    n = _norm_name(name)
    for s in _inv_slots(payload):
        if _norm_name(s.get("itemName")) == n:
            return s
    return None

def _first_bank_slot(payload: dict, name: str) -> dict | None:
    n = _norm_name(name)
    best = None
    for s in _bank_slots(payload):
        if _norm_name(s.get("itemName")) == n:
            if best is None:
                best = s
            else:
                q1 = int(s.get("quantity") or 0)
                q2 = int(best.get("quantity") or 0)
                if q1 > q2 or (q1 == q2 and int(s.get("slotId") or 9_999) < int(best.get("slotId") or 9_999)):
                    best = s
    return best

def _unwrap_rect(maybe_rect_dict: dict | None) -> dict | None:
    if not isinstance(maybe_rect_dict, dict):
        return None
    if {"x","y","width","height"} <= set(maybe_rect_dict.keys()):
        return maybe_rect_dict
    inner = maybe_rect_dict.get("bounds")
    if isinstance(inner, dict) and {"x","y","width","height"} <= set(inner.keys()):
        return inner
    return None

def _rect_center_xy(rect: dict | None) -> tuple[int | None, int | None]:
    if not rect:
        return None, None
    try:
        return int(rect["x"] + rect["width"]/2), int(rect["y"] + rect["height"]/2)
    except Exception:
        return None, None

def _closest_object_by_names(payload: dict, names: list[str]) -> dict | None:
    wanted = [n.lower() for n in names]

    # Prefer the GE-specific list your exporter writes
    for obj in (payload.get("ge_booths") or []):
        nm = _norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj

    # Fallback to generic nearby objects
    for obj in (payload.get("closestGameObjects") or []):
        nm = _norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj

    return None



def _craft_widget_rect(payload: dict, key: str) -> dict | None:
    w = (payload.get("crafting_widgets", {}) or {}).get(key)
    return _unwrap_rect((w or {}).get("bounds") if isinstance(w, dict) else None)

# Known crafting animations
_CRAFT_ANIMS = {899}
def _is_crafting_anim(anim_id: int) -> bool:
    return anim_id in _CRAFT_ANIMS

# ------------- Plan interface -------------
class Plan:
    """
    Each plan exposes:
      - id: str
      - label: str
      - compute_phase(payload: dict, craft_recent: bool) -> str
      - build_action_plan(payload: dict, phase: str) -> dict
    """
    id: str
    label: str
    def compute_phase(self, payload: dict, craft_recent: bool) -> str: ...
    def build_action_plan(self, payload: dict, phase: str) -> dict: ...

# ------------- Sapphire Rings (your current behavior) -------------
class SapphireRingsPlan(Plan):
    id = "SAPPHIRE_RINGS"
    label = "Sapphire Rings"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        bank_open   = bool((payload.get("bank") or {}).get("bankOpen", False))
        craft_open  = bool(payload.get("craftingInterfaceOpen", False))
        has_mould   = _inv_has(payload, "Ring mould")
        has_gold    = _inv_count(payload, "Gold bar") > 0
        has_sapph   = _inv_count(payload, "Sapphire") > 0
        out_of_mats = (not has_gold) or (not has_sapph) or (not has_mould)

        if bank_open:
            return "Banking"
        if (craft_open or craft_recent):
            return "Crafting" if not out_of_mats else "Moving to bank"
        if out_of_mats:
            return "Moving to bank"
        return "Moving to furnace"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}

        if phase == "Moving to bank":
            obj = _closest_object_by_names(payload, ["bank booth", "banker"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action": "click",
                    "description": "Click nearest bank booth",
                    "click": ({"type": "rect-center"} if rect else
                              {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}),
                    "target": {
                        "domain": "object", "name": obj.get("name"), "id": obj.get("id"),
                        "clickbox": rect,
                        "canvas": {"x": obj.get("canvasX"), "y": obj.get("canvasY")}
                    },
                    "preconditions": ["bankOpen == false"],
                    "postconditions": ["bankOpen == true"],
                    "confidence": 0.92 if rect else 0.6
                }
                plan["steps"].append(step)
            return plan

        if phase == "Banking":
            TARGET_SAPP = 13
            TARGET_GOLD = 13
            inv_sapp  = _inv_count(payload, "Sapphire")
            inv_gold  = _inv_count(payload, "Gold bar")
            has_mould = _inv_has(payload, "Ring mould")
            inv_ring  = _first_inv_slot(payload, "Sapphire ring")

            if inv_ring:
                rect = _unwrap_rect(inv_ring.get("bounds"))
                plan["steps"].append({
                    "action": "deposit-inventory-item",
                    "description": "Deposit Sapphire ring from inventory",
                    "click": {"type": "rect-center"} if rect else {"type":"none"},
                    "target": {"domain":"inventory", "name":"Sapphire ring",
                               "slotId": inv_ring.get("slotId"), "bounds": rect},
                    "preconditions": ["bankOpen == true", "inventory contains 'Sapphire ring'"],
                    "postconditions": ["inventory does not contain 'Sapphire ring'"],
                    "confidence": 0.9 if rect else 0.4,
                })
                return plan

            if inv_sapp < TARGET_SAPP:
                bank_sapp = _first_bank_slot(payload, "Sapphire")
                if bank_sapp:
                    rect = _unwrap_rect(bank_sapp.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": f"Withdraw Sapphires (need {TARGET_SAPP - inv_sapp} more)",
                        "click": {"type": "rect-center"} if rect else {"type":"none"},
                        "target": {"domain":"bank","name":"Sapphire","slotId":bank_sapp.get("slotId"),"bounds":rect},
                        "preconditions": ["bankOpen == true", f"inventory count('Sapphire') < {TARGET_SAPP}"],
                        "postconditions": [f"inventory count('Sapphire') >= {TARGET_SAPP}"],
                        "confidence": 0.9 if rect else 0.4,
                    })
                    return plan
                else:
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": "Could not find Sapphires in bank",
                        "click": {"type":"none"},
                        "target": {"domain":"bank","name":"Sapphire"},
                        "preconditions": ["bankOpen == true"],
                        "postconditions": [],
                        "confidence": 0.0
                    })
                    return plan

            if inv_gold < TARGET_GOLD:
                bank_gold = _first_bank_slot(payload, "Gold bar")
                if bank_gold:
                    rect = _unwrap_rect(bank_gold.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": f"Withdraw Gold bars (need {TARGET_GOLD - inv_gold} more)",
                        "click": {"type":"rect-center"} if rect else {"type":"none"},
                        "target": {"domain":"bank","name":"Gold bar","slotId":bank_gold.get("slotId"),"bounds":rect},
                        "preconditions": ["bankOpen == true", f"inventory count('Gold bar') < {TARGET_GOLD}"],
                        "postconditions": [f"inventory count('Gold bar') >= {TARGET_GOLD}"],
                        "confidence": 0.9 if rect else 0.4,
                    })
                    return plan
                else:
                    plan["steps"].append({
                        "action":"withdraw-item",
                        "description":"Could not find Gold bars in bank",
                        "click":{"type":"none"},
                        "target":{"domain":"bank","name":"Gold bar"},
                        "preconditions":["bankOpen == true"],
                        "postconditions":[],
                        "confidence":0.0
                    })
                    return plan

            if not has_mould:
                bank_mould = _first_bank_slot(payload, "Ring mould")
                if bank_mould:
                    rect = _unwrap_rect(bank_mould.get("bounds"))
                    plan["steps"].append({
                        "action":"withdraw-item",
                        "description":"Withdraw Ring mould",
                        "click":{"type":"rect-center"} if rect else {"type":"none"},
                        "target":{"domain":"bank","name":"Ring mould","slotId":bank_mould.get("slotId"),"bounds":rect},
                        "preconditions":["bankOpen == true","inventory does not contain 'Ring mould'"],
                        "postconditions":["inventory contains 'Ring mould'"],
                        "confidence":0.9 if rect else 0.4,
                    })
                    return plan

            plan["steps"].append({
                "action":"close-bank",
                "description":"Close bank with ESC",
                "click":{"type":"key","key":"ESC"},
                "target":{"domain":"widget","name":"bank_close"},
                "preconditions":[
                    "bankOpen == true",
                    "inventory contains 'Ring mould'",
                    f"inventory count('Sapphire') >= {TARGET_SAPP}",
                    f"inventory count('Gold bar') >= {TARGET_GOLD}",
                    "inventory does not contain 'Sapphire ring'",
                ],
                "postconditions":["bankOpen == false"],
                "confidence":0.95
            })
            return plan

        if phase == "Moving to furnace":
            obj = _closest_object_by_names(payload, ["furnace"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action":"click-furnace",
                    "description":"Click nearest furnace",
                    "click": ({"type":"rect-center"} if rect else
                              {"type":"point","x":int(obj.get("canvasX") or 0),"y":int(obj.get("canvasY") or 0)}),
                    "target":{
                        "domain":"object","name":obj.get("name"),"id":obj.get("id"),
                        "clickbox":rect,"canvas":{"x":obj.get("canvasX"),"y":obj.get("canvasY")}
                    },
                    "preconditions":[
                        "bankOpen == false",
                        "inventory contains 'Ring mould'",
                        "inventory count('Sapphire') > 0",
                        "inventory count('Gold bar') > 0"
                    ],
                    "postconditions":["craftingInterfaceOpen == true"],
                    "confidence":0.92 if rect else 0.6
                }
                plan["steps"].append(step)
            return plan

        if phase == "Crafting":
            make_rect = _craft_widget_rect(payload, "make_sapphire_rings")
            plan["steps"].append({
                "action":"click-make-widget",
                "description":"Click the 'Make sapphire rings' button",
                "click":{"type":"rect-center"} if make_rect else {"type":"none"},
                "target":{"domain":"widget","name":"make_sapphire_rings","bounds":make_rect},
                "preconditions":[
                    "craftingInterfaceOpen == true",
                    "inventory count('Sapphire') > 0",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions":[
                    "player.animation == 899 OR crafting in progress"
                ],
                "confidence":0.95 if make_rect else 0.4
            })
            plan["steps"].append({
                "action":"wait-crafting-complete",
                "description":"Wait until sapphires and gold bars are consumed",
                "click":{"type":"none"},
                "target":{"domain":"none","name":"crafting_wait"},
                "preconditions":[
                    "inventory count('Sapphire') > 0",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions":[
                    "inventory count('Sapphire') == 0 OR inventory count('Gold bar') == 0"
                ],
                "confidence":1.0
            })
            return plan

        plan["steps"].append({"action":"idle","description":"No actionable step for this phase",
                              "click":{"type":"none"},"target":{"domain":"none","name":"n/a"},
                              "preconditions":[],"postconditions":[],"confidence":0.0})
        return plan

# ------------- Gold Rings -------------
class GoldRingsPlan(Plan):
    id = "GOLD_RINGS"
    label = "Gold Rings"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        bank_open   = bool((payload.get("bank") or {}).get("bankOpen", False))
        craft_open  = bool(payload.get("craftingInterfaceOpen", False))
        has_mould   = _inv_has(payload, "Ring mould")
        has_gold    = _inv_count(payload, "Gold bar") > 0
        out_of_mats = (not has_gold) or (not has_mould)

        if bank_open:
            return "Banking"
        if (craft_open or craft_recent):
            return "Crafting" if not out_of_mats else "Moving to bank"
        if out_of_mats:
            return "Moving to bank"
        return "Moving to furnace"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}

        if phase == "Moving to bank":
            obj = _closest_object_by_names(payload, ["bank booth", "banker"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action": "click",
                    "description": "Click nearest bank booth",
                    "click": ({"type":"rect-center"} if rect else
                              {"type":"point","x":int(obj.get("canvasX") or 0),"y":int(obj.get("canvasY") or 0)}),
                    "target": {
                        "domain":"object","name":obj.get("name"),"id":obj.get("id"),
                        "clickbox":rect,"canvas":{"x":obj.get("canvasX"),"y":obj.get("canvasY")}
                    },
                    "preconditions": ["bankOpen == false"],
                    "postconditions": ["bankOpen == true"],
                    "confidence": 0.92 if rect else 0.6
                }
                plan["steps"].append(step)
            return plan

        if phase == "Banking":
            TARGET_GOLD = 27  # full inv minus mould
            inv_gold  = _inv_count(payload, "Gold bar")
            has_mould = _inv_has(payload, "Ring mould")
            inv_ring  = _first_inv_slot(payload, "Gold ring")

            if inv_ring:
                rect = _unwrap_rect(inv_ring.get("bounds"))
                plan["steps"].append({
                    "action":"deposit-inventory-item",
                    "description":"Deposit Gold ring from inventory",
                    "click":{"type":"rect-center"} if rect else {"type":"none"},
                    "target":{"domain":"inventory","name":"Gold ring",
                              "slotId": inv_ring.get("slotId"), "bounds": rect},
                    "preconditions":["bankOpen == true", "inventory contains 'Gold ring'"],
                    "postconditions":["inventory does not contain 'Gold ring'"],
                    "confidence":0.9 if rect else 0.4,
                })
                return plan

            if inv_gold < TARGET_GOLD:
                bank_gold = _first_bank_slot(payload, "Gold bar")
                if bank_gold:
                    rect = _unwrap_rect(bank_gold.get("bounds"))
                    plan["steps"].append({
                        "action":"withdraw-item",
                        "description":f"Withdraw Gold bars (need {TARGET_GOLD - inv_gold} more)",
                        "click":{"type":"rect-center"} if rect else {"type":"none"},
                        "target":{"domain":"bank","name":"Gold bar","slotId":bank_gold.get("slotId"),"bounds":rect},
                        "preconditions":["bankOpen == true", f"inventory count('Gold bar') < {TARGET_GOLD}"],
                        "postconditions":[f"inventory count('Gold bar') >= {TARGET_GOLD}"],
                        "confidence":0.9 if rect else 0.4,
                    })
                    return plan
                else:
                    plan["steps"].append({
                        "action":"withdraw-item",
                        "description":"Could not find Gold bars in bank",
                        "click":{"type":"none"},
                        "target":{"domain":"bank","name":"Gold bar"},
                        "preconditions":["bankOpen == true"],
                        "postconditions":[],
                        "confidence":0.0
                    })
                    return plan

            if not has_mould:
                bank_mould = _first_bank_slot(payload, "Ring mould")
                if bank_mould:
                    rect = _unwrap_rect(bank_mould.get("bounds"))
                    plan["steps"].append({
                        "action":"withdraw-item",
                        "description":"Withdraw Ring mould",
                        "click":{"type":"rect-center"} if rect else {"type":"none"},
                        "target":{"domain":"bank","name":"Ring mould","slotId":bank_mould.get("slotId"),"bounds":rect},
                        "preconditions":["bankOpen == true", "inventory does not contain 'Ring mould'"],
                        "postconditions":["inventory contains 'Ring mould'"],
                        "confidence":0.9 if rect else 0.4,
                    })
                    return plan

            plan["steps"].append({
                "action":"close-bank",
                "description":"Close bank with ESC",
                "click":{"type":"key","key":"ESC"},
                "target":{"domain":"widget","name":"bank_close"},
                "preconditions":[
                    "bankOpen == true",
                    "inventory contains 'Ring mould'",
                    f"inventory count('Gold bar') >= {TARGET_GOLD}",
                    "inventory does not contain 'Gold ring'",
                ],
                "postconditions":["bankOpen == false"],
                "confidence":0.95
            })
            return plan

        if phase == "Moving to furnace":
            obj = _closest_object_by_names(payload, ["furnace"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action":"click-furnace",
                    "description":"Click nearest furnace",
                    "click": ({"type":"rect-center"} if rect else
                              {"type":"point","x":int(obj.get("canvasX") or 0),"y":int(obj.get("canvasY") or 0)}),
                    "target":{
                        "domain":"object","name":obj.get("name"),"id":obj.get("id"),
                        "clickbox":rect,"canvas":{"x":obj.get("canvasX"),"y":obj.get("canvasY")}
                    },
                    "preconditions":[
                        "bankOpen == false",
                        "inventory contains 'Ring mould'",
                        "inventory count('Gold bar') > 0"
                    ],
                    "postconditions":["craftingInterfaceOpen == true"],
                    "confidence":0.92 if rect else 0.6
                }
                plan["steps"].append(step)
            return plan

        if phase == "Crafting":
            # your plugin should export the proper widget bounds under this key
            make_rect = _craft_widget_rect(payload, "make_gold_rings")
            plan["steps"].append({
                "action":"click-make-widget",
                "description":"Click the 'Make gold rings' button",
                "click":{"type":"rect-center"} if make_rect else {"type":"none"},
                "target":{"domain":"widget","name":"make_gold_rings","bounds":make_rect},
                "preconditions":[
                    "craftingInterfaceOpen == true",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions":[
                    "player.animation == 899 OR crafting in progress"
                ],
                "confidence":0.95 if make_rect else 0.4
            })
            plan["steps"].append({
                "action":"wait-crafting-complete",
                "description":"Wait until gold bars are consumed",
                "click":{"type":"none"},
                "target":{"domain":"none","name":"crafting_wait"},
                "preconditions":[ "inventory count('Gold bar') > 0" ],
                "postconditions":[ "inventory count('Gold bar') == 0" ],
                "confidence":1.0
            })
            return plan

        plan["steps"].append({"action":"idle","description":"No actionable step for this phase",
                              "click":{"type":"none"},"target":{"domain":"none","name":"n/a"},
                              "preconditions":[],"postconditions":[],"confidence":0.0})
        return plan

class EmeraldRingsPlan(Plan):
    id = "EMERALD_RINGS"
    label = "Emerald Rings"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        bank_open   = bool((payload.get("bank") or {}).get("bankOpen", False))
        craft_open  = bool(payload.get("craftingInterfaceOpen", False))
        has_mould   = _inv_has(payload, "Ring mould")
        has_gold    = _inv_count(payload, "Gold bar") > 0
        has_emerald = _inv_count(payload, "Emerald") > 0
        out_of_mats = (not has_gold) or (not has_emerald) or (not has_mould)

        if bank_open:
            return "Banking"
        if (craft_open or craft_recent):
            return "Crafting" if not out_of_mats else "Moving to bank"
        if out_of_mats:
            return "Moving to bank"
        return "Moving to furnace"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}

        if phase == "Moving to bank":
            # ---- compute coordinates (prefer clickbox center; else world-tile; else canvas) ----
            obj = _closest_object_by_names(payload, ["grand exchange booth"])
            if not obj:
                return {"phase": "No target", "steps": []}

            cb = (obj.get("clickbox") or {})
            has_rect = all(k in cb for k in ("x", "y", "width", "height")) and cb["width"] and cb["height"]
            click = None
            click_kind = None

            if has_rect:
                x = int(cb["x"] + cb["width"] // 2)
                y = int(cb["y"] + cb["height"] // 2)
                click = {"type": "point", "x": x, "y": y}
                click_kind = "rect-center(point)"
            elif obj.get("worldX") is not None and obj.get("worldY") is not None:
                click = {
                    "type": "world-tile",
                    "worldX": int(obj.get("worldX") or 0),
                    "worldY": int(obj.get("worldY") or 0),
                    "plane": int(obj.get("plane") or 0),
                }
                click_kind = "world-tile"
            elif obj.get("canvasX") is not None and obj.get("canvasY") is not None:
                click = {"type": "point", "x": int(obj.get("canvasX")), "y": int(obj.get("canvasY"))}
                click_kind = "canvas-point"

            if not click:
                return {"phase": "No target coords", "steps": []}

            # Build a super explicit debug bundle we can print in the executor
            debug_payload = {
                "chosen_obj": {
                    "id": obj.get("id"),
                    "name": obj.get("name"),
                    "worldX": obj.get("worldX"),
                    "worldY": obj.get("worldY"),
                    "plane": obj.get("plane"),
                    "canvasX": obj.get("canvasX"),
                    "canvasY": obj.get("canvasY"),
                    "has_clickbox": bool(has_rect),
                    "source": "ge_booths" if (obj in (payload.get("ge_booths") or [])) else "closestGameObjects",
                },
                "computed_click": click,
                "click_kind": click_kind,
            }

            step = {
                "id": "ge-bank-open",
                "action": "click",
                "description": "Click Grand Exchange bank booth",
                "click": click,
                "target": {
                    "domain": "object",
                    "name": obj.get("name"),
                    "id": obj.get("id"),
                },
                "preconditions": ["bankOpen == false"],
                "postconditions": ["bankOpen == true"],
                "confidence": 0.92 if has_rect else 0.6,
                "summary": f"GE booth: id={obj.get('id')} {click_kind} -> {click}",
                "debug": debug_payload,  # <— add this
            }
            return {"phase": phase, "steps": [step]}

        if phase == "Banking":
            TARGET_EME  = 13
            TARGET_GOLD = 13
            inv_emerald = _inv_count(payload, "Emerald")
            inv_gold    = _inv_count(payload, "Gold bar")
            has_mould   = _inv_has(payload, "Ring mould")
            inv_ring    = _first_inv_slot(payload, "Emerald ring")

            # Deposit outputs first
            if inv_ring:
                rect = _unwrap_rect(inv_ring.get("bounds"))
                plan["steps"].append({
                    "action": "deposit-inventory-item",
                    "description": "Deposit Emerald ring from inventory",
                    "click": {"type": "rect-center"} if rect else {"type": "none"},
                    "target": {"domain": "inventory", "name": "Emerald ring",
                               "slotId": inv_ring.get("slotId"), "bounds": rect},
                    "preconditions": ["bankOpen == true", "inventory contains 'Emerald ring'"],
                    "postconditions": ["inventory does not contain 'Emerald ring'"],
                    "confidence": 0.9 if rect else 0.4,
                })
                return plan

            # Top off Emeralds
            if inv_emerald < TARGET_EME:
                bank_emerald = _first_bank_slot(payload, "Emerald")
                if bank_emerald:
                    rect = _unwrap_rect(bank_emerald.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": f"Withdraw Emeralds (need {TARGET_EME - inv_emerald} more)",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "bank", "name": "Emerald",
                                   "slotId": bank_emerald.get("slotId"), "bounds": rect},
                        "preconditions": ["bankOpen == true", f"inventory count('Emerald') < {TARGET_EME}"],
                        "postconditions": [f"inventory count('Emerald') >= {TARGET_EME}"],
                        "confidence": 0.9 if rect else 0.4,
                    })
                    return plan
                else:
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": "Could not find Emeralds in bank",
                        "click": {"type": "none"},
                        "target": {"domain": "bank", "name": "Emerald"},
                        "preconditions": ["bankOpen == true"],
                        "postconditions": [],
                        "confidence": 0.0
                    })
                    return plan

            # Top off Gold bars
            if inv_gold < TARGET_GOLD:
                bank_gold = _first_bank_slot(payload, "Gold bar")
                if bank_gold:
                    rect = _unwrap_rect(bank_gold.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": f"Withdraw Gold bars (need {TARGET_GOLD - inv_gold} more)",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "bank", "name": "Gold bar",
                                   "slotId": bank_gold.get("slotId"), "bounds": rect},
                        "preconditions": ["bankOpen == true", f"inventory count('Gold bar') < {TARGET_GOLD}"],
                        "postconditions": [f"inventory count('Gold bar') >= {TARGET_GOLD}"],
                        "confidence": 0.9 if rect else 0.4,
                    })
                    return plan
                else:
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": "Could not find Gold bars in bank",
                        "click": {"type": "none"},
                        "target": {"domain": "bank", "name": "Gold bar"},
                        "preconditions": ["bankOpen == true"],
                        "postconditions": [],
                        "confidence": 0.0
                    })
                    return plan

            # Ensure Ring mould
            if not has_mould:
                bank_mould = _first_bank_slot(payload, "Ring mould")
                if bank_mould:
                    rect = _unwrap_rect(bank_mould.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": "Withdraw Ring mould",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "bank", "name": "Ring mould",
                                   "slotId": bank_mould.get("slotId"), "bounds": rect},
                        "preconditions": ["bankOpen == true", "inventory does not contain 'Ring mould'"],
                        "postconditions": ["inventory contains 'Ring mould'"],
                        "confidence": 0.9 if rect else 0.4,
                    })
                    return plan

            # Close bank when ready
            plan["steps"].append({
                "action": "close-bank",
                "description": "Close bank with ESC",
                "click": {"type": "key", "key": "ESC"},
                "target": {"domain": "widget", "name": "bank_close"},
                "preconditions": [
                    "bankOpen == true",
                    "inventory contains 'Ring mould'",
                    f"inventory count('Emerald') >= {TARGET_EME}",
                    f"inventory count('Gold bar') >= {TARGET_GOLD}",
                    "inventory does not contain 'Emerald ring'",
                ],
                "postconditions": ["bankOpen == false"],
                "confidence": 0.95
            })
            return plan

        if phase == "Moving to furnace":
            obj = _closest_object_by_names(payload, ["furnace"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action": "click-furnace",
                    "description": "Click nearest furnace",
                    "click": ({"type": "rect-center"} if rect else
                              {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}),
                    "target": {
                        "domain": "object", "name": obj.get("name"), "id": obj.get("id"),
                        "clickbox": rect, "canvas": {"x": obj.get("canvasX"), "y": obj.get("canvasY")}
                    },
                    "preconditions": [
                        "bankOpen == false",
                        "inventory contains 'Ring mould'",
                        "inventory count('Emerald') > 0",
                        "inventory count('Gold bar') > 0"
                    ],
                    "postconditions": ["craftingInterfaceOpen == true"],
                    "confidence": 0.92 if rect else 0.6
                }
                plan["steps"].append(step)
            return plan

        if phase == "Crafting":
            make_rect = _craft_widget_rect(payload, "make_emerald_rings")
            plan["steps"].append({
                "action": "click-make-widget",
                "description": "Click the 'Make emerald rings' button",
                "click": {"type": "rect-center"} if make_rect else {"type": "none"},
                "target": {"domain": "widget", "name": "make_emerald_rings", "bounds": make_rect},
                "preconditions": [
                    "craftingInterfaceOpen == true",
                    "inventory count('Emerald') > 0",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions": [
                    "player.animation == 899 OR crafting in progress"
                ],
                "confidence": 0.95 if make_rect else 0.4
            })
            plan["steps"].append({
                "action": "wait-crafting-complete",
                "description": "Wait until emeralds and gold bars are consumed",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "crafting_wait"},
                "preconditions": [
                    "inventory count('Emerald') > 0",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions": [
                    "inventory count('Emerald') == 0 OR inventory count('Gold bar') == 0"
                ],
                "confidence": 1.0
            })
            return plan

        plan["steps"].append({
            "action": "idle",
            "description": "No actionable step for this phase",
            "click": {"type": "none"},
            "target": {"domain": "none", "name": "n/a"},
            "preconditions": [],
            "postconditions": [],
            "confidence": 0.0
        })
        return plan

class GoToGEPlan(Plan):
    """
    Chooses a ground tile from tiles_15x15 that best advances the player toward the
    Grand Exchange world coords, and clicks it using the tile's canvas coordinates.
    """
    id = "GO_TO_GE"
    label = "Go to GE"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        me = (payload.get("player") or {})
        tiles = payload.get("tiles_15x15") or []
        if not me or not tiles:
            return "No GE/Player/Tiles"

        wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))

        # Check if inside GE region
        if GE_MIN_X <= wx <= GE_MAX_X and GE_MIN_Y <= wy <= GE_MAX_Y:
            return "Arrived at GE"

        return "Moving to GE"

    # inside GoToGEPlan
    def _ge_center(self) -> tuple[int, int]:
        # Center of the GE region from constants.py
        gx = (GE_MIN_X + GE_MAX_X) // 2
        gy = (GE_MIN_Y + GE_MAX_Y) // 2
        return gx, gy

    def _line_to(self, x0: int, y0: int, x1: int, y1: int, max_steps: int = 14):
        """
        Integer grid line from (x0,y0) to (x1,y1), up to max_steps ahead (not including start).
        Classic Bresenham; returns a list of (x,y) points along the path.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1 if x0 > x1 else 0
        sy = 1 if y0 < y1 else -1 if y0 > y1 else 0
        x, y = x0, y0

        if dx >= dy:
            err = dx // 2
            for _ in range(max_steps):
                if x == x1 and y == y1:
                    break
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                points.append((x, y))
        else:
            err = dy // 2
            for _ in range(max_steps):
                if x == x1 and y == y1:
                    break
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                points.append((x, y))
        return points

    def _pick_tile_toward_ge(self, payload: dict) -> dict | None:
        me = (payload.get("player") or {})
        tiles = payload.get("tiles_15x15") or []

        try:
            wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
        except Exception:
            return None

        # target = GE center (from constants)
        gx, gy = self._ge_center()

        # Build a fast lookup for available tiles by world coord
        # Keep only tiles that have canvas coords (clickable now)
        by_world = {}
        for t in tiles:
            try:
                tx, ty = int(t.get("worldX")), int(t.get("worldY"))
                cx, cy = t.get("canvasX"), t.get("canvasY")
            except Exception:
                continue
            if isinstance(cx, int) and isinstance(cy, int):
                by_world[(tx, ty)] = t

        # Walk the straight line from player to GE center (up to 14 steps ahead),
        # pick the FARTHEST point along that line that actually exists in tiles_15x15.
        path = self._line_to(wx, wy, gx, gy, max_steps=14)
        pick = None
        for pt in reversed(path):  # farthest first
            if pt in by_world:
                pick = by_world[pt]
                break

        return pick

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}

        if phase == "Arrived at GE":
            plan["steps"].append({
                "action": "idle",
                "description": "Player is inside GE region, no further movement",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "n/a"},
                "preconditions": [],
                "postconditions": [],
                "confidence": 1.0
            })
            return plan

        if phase != "Moving to GE":
            plan["steps"].append({
                "action": "idle",
                "description": "No actionable step",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "n/a"},
                "preconditions": [],
                "postconditions": [],
                "confidence": 0.0
            })
            return plan

        pick = self._pick_tile_toward_ge(payload)
        if pick:
            # player + GE center (for debug visibility)
            me = (payload.get("player") or {})
            wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
            gx, gy = self._ge_center()

            cx, cy = int(pick["canvasX"]), int(pick["canvasY"])
            tx, ty = int(pick.get("worldX", 0)), int(pick.get("worldY", 0))
            plane = int(pick.get("plane", 0))

            plan["steps"].append({
                "action": "click-ground",
                # keep description short; the GE center is surfaced in the target name for the UI
                "description": f"toward GE center {gx},{gy} from {wx},{wy}",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {
                    "domain": "ground",
                    # ↓↓↓ This shows up in the Next Action row: "<action> → <name> @ (x,y)"
                    "name": f"Tile→GE_CENTER({gx},{gy})",
                    "world": {"x": tx, "y": ty, "plane": plane},
                    "canvas": {"x": cx, "y": cy},
                    # optional extra debug payload if you inspect the step JSON
                    "debug": {
                        "player": {"x": wx, "y": wy},
                        "ge_center": {"x": gx, "y": gy},
                        "goal_vec": {"dx": gx - wx, "dy": gy - wy},
                        "chosen_tile_world": {"x": tx, "y": ty, "plane": plane},
                        "chosen_tile_canvas": {"x": cx, "y": cy},
                    }
                },
                "preconditions": [],
                "postconditions": [],
                "confidence": 0.85
            })
            return plan

class OpenGEBankPlan(Plan):
    id = "OPEN_GE_BANK"
    label = "GE: Open Bank"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        bank_open = bool((payload.get("bank") or {}).get("bankOpen", False))
        if bank_open:
            return "Bank already open"
        # Only target the exact GE bank booth name the user wants
        obj = _closest_object_by_names(payload, ["grand exchange booth"])
        return "Click GE bank" if obj else "No target"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}
        if phase != "Click GE bank":
            return plan

        # --- 1) Find a GE booth (for a stable canvas anchor) ---
        booth = _closest_object_by_names(payload, ["grand exchange booth"])
        if not booth:
            return plan

        # Pull booth canvas coords (exported by your plugin under ge_booths/closestGameObjects)
        canvasX = booth.get("canvasX")
        canvasY = booth.get("canvasY")

        # If missing in the chosen object, try enrich from ge_booths by id+world match
        if (not isinstance(canvasX, (int, float)) or canvasX < 0 or
                not isinstance(canvasY, (int, float)) or canvasY < 0):
            for ge in (payload.get("ge_booths") or []):
                try:
                    same = (int(ge.get("id") or -1) == int(booth.get("id") or -2) and
                            int(ge.get("worldX") or -1) == int(booth.get("worldX") or -2) and
                            int(ge.get("worldY") or -1) == int(booth.get("worldY") or -2))
                except Exception:
                    same = False
                if same:
                    canvasX = ge.get("canvasX", canvasX)
                    canvasY = ge.get("canvasY", canvasY)
                    break

        # Bail if we still don’t have a usable canvas anchor
        if not isinstance(canvasX, (int, float)) or not isinstance(canvasY, (int, float)):
            return plan

        # --- 2) Find the nearest Banker (world coords only are fine) ---
        def _nearest_banker(payload: dict, booth_obj: dict) -> dict | None:
            bx, by, bp = int(booth_obj.get("worldX") or 0), int(booth_obj.get("worldY") or 0), int(
                booth_obj.get("plane") or 0)
            best, best_d2 = None, 1e18
            for npc in (payload.get("closestNPCs") or []):
                nm = (npc.get("name") or "").lower()
                if "banker" not in nm:
                    continue
                nx, ny, np = int(npc.get("worldX") or 0), int(npc.get("worldY") or 0), int(npc.get("plane") or 0)
                if np != bp:
                    continue
                dx, dy = (nx - bx), (ny - by)
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best, best_d2 = npc, d2
            return best

        banker = _nearest_banker(payload, booth)
        if not banker:
            # No banker seen? Keep your old behavior (click the booth anchor with a small lift)
            click = {"type": "point", "x": int(canvasX), "y": int(canvasY) - 16}
            step = {
                "id": "ge-bank-open",
                "action": "click",
                "description": "Click GE booth (fallback: no banker found)",
                "click": click,
                "target": {
                    "domain": "object",
                    "name": booth.get("name"),
                    "id": booth.get("id"),
                    "worldX": booth.get("worldX"),
                    "worldY": booth.get("worldY"),
                    "plane": booth.get("plane", 0),
                    "canvasX": canvasX, "canvasY": canvasY,
                },
                "preconditions": ["bankOpen == false"],
                "postconditions": ["bankOpen == true"],
                "confidence": 0.85,
                "debug": {
                    "click_kind": "canvas-point (no banker)",
                    "chosen_booth": booth,
                    "computed_click": click,
                },
            }
            plan["steps"].append(step)
            return plan

        # --- 3) Compute a simple nudge toward the banker (pixel space) ---
        # Convert booth->banker world delta into a small canvas offset.
        # Tunables (pixels per world-tile influence). Start modest; adjust if needed.
        PX_PER_TILE = 40  # coarse mapping tile→pixels at your typical zoom
        SCALE = 0.8  # use 60% of the vector length to avoid overshoot
        LIFT_Y = -6  # a small upward bias helps catch the banker sprite

        dx_tiles = int(banker.get("worldX") or 0) - int(booth.get("worldX") or 0)
        dy_tiles = int(banker.get("worldY") or 0) - int(booth.get("worldY") or 0)

        # Map world delta to a rough screen delta; note: y sign may need flipping depending on your camera
        # Empirically at GE with a default camera, positive worldY tends to go "up" in canvas (smaller Y).
        # If it goes the other way for you, change dy_px to (-dy_tiles * PX_PER_TILE).
        dx_px = int(dx_tiles * PX_PER_TILE * SCALE)
        dy_px = int(-dy_tiles * PX_PER_TILE * SCALE) + int(LIFT_Y)

        click_x = int(canvasX) + dx_px
        click_y = int(canvasY) + dy_px

        click = {
            "type": "point",
            "x": click_x,
            "y": click_y,
        }

        step = {
            "id": "ge-bank-open",
            "action": "click",
            "description": "Click Banker (via booth anchor nudged toward banker)",
            "click": click,
            "target": {
                "domain": "npc",
                "name": banker.get("name"),
                "id": banker.get("id"),
                "worldX": banker.get("worldX"),
                "worldY": banker.get("worldY"),
                "plane": banker.get("plane", 0),
                # keep the booth anchor for debugging
                "anchor": {
                    "name": booth.get("name"),
                    "worldX": booth.get("worldX"),
                    "worldY": booth.get("worldY"),
                    "plane": booth.get("plane", 0),
                    "canvasX": canvasX, "canvasY": canvasY,
                },
            },
            "preconditions": ["bankOpen == false"],
            "postconditions": ["bankOpen == true"],
            "confidence": 0.93,
            "summary": f"Banker via booth anchor → click=({click_x},{click_y})",
            "debug": {
                "click_kind": "canvas-point toward banker",
                "booth": {
                    "id": booth.get("id"),
                    "worldX": booth.get("worldX"),
                    "worldY": booth.get("worldY"),
                    "canvasX": canvasX, "canvasY": canvasY,
                },
                "banker": {
                    "id": banker.get("id"),
                    "name": banker.get("name"),
                    "worldX": banker.get("worldX"),
                    "worldY": banker.get("worldY"),
                    "plane": banker.get("plane"),
                },
                "vec_world": {"dx_tiles": dx_tiles, "dy_tiles": dy_tiles},
                "vec_canvas": {"dx_px": dx_px, "dy_px": dy_px},
                "computed_click": click,
            },
        }

        plan["steps"].append(step)
        return plan

class GEWithdrawRingsAsNotesPlan(Plan):
    id = "GE_WITHDRAW_NOTED_RINGS"

    def _bank_openish(self, payload: dict) -> bool:
        b = payload.get("bank") or {}
        if b.get("bankOpen"):
            return True
        bw = payload.get("bank_widgets") or {}
        if bw.get("withdraw_note_toggle") or bw.get("withdraw_quantity_all"):
            return True
        return False

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        return "Withdraw rings as notes" if self._bank_openish(payload) else "Idle"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}
        if phase != "Withdraw rings as notes":
            return plan

        bw = payload.get("bank_widgets") or {}
        note_rect = ((bw.get("withdraw_note_toggle") or {}).get("bounds") or None)
        all_rect  = ((bw.get("withdraw_quantity_all") or {}).get("bounds") or None)

        def _rect_ok(r):
            return isinstance(r, dict) and all(k in r for k in ("x","y","width","height"))

        # 1) Click NOTE once (skip if we just did it recently)
        if _rect_ok(note_rect) and not step_recent("bank-note-toggle", 6000):
            plan["steps"].append({
                "id": "bank-note-toggle",
                "action": "click",
                "description": "Enable Withdraw as Note",
                "target": {"name": "Withdraw as Note", "bounds": note_rect},
                "click": {"type": "rect-center"},
                "preconditions": ["bankOpen == true"],
                "postconditions": [],
            })
            return plan

        # 2) Click ALL once (skip if we just did it recently)
        if _rect_ok(all_rect) and not step_recent("bank-qty-all", 6000):
            plan["steps"].append({
                "id": "bank-qty-all",
                "action": "click",
                "description": "Set Withdraw Quantity to All",
                "target": {"name": "Quantity All", "bounds": all_rect},
                "click": {"type": "rect-center"},
                "preconditions": ["bankOpen == true"],
                "postconditions": [],
            })
            return plan

        # 3) Withdraw rings (one stack per tick)
        bank_slots = (payload.get("bank") or {}).get("slots") or []
        def _slot_bounds(slot: dict):
            wrap = slot.get("bounds") or {}
            return wrap.get("bounds")

        ring_names = {"sapphire ring", "emerald ring", "sapphire rings", "emerald rings"}
        for s in bank_slots:
            nm = (s.get("itemName") or "").strip().lower()
            qty = int(s.get("quantity") or 0)
            if nm in ring_names and qty > 0:
                r = _slot_bounds(s)
                if _rect_ok(r):
                    plan["steps"].append({
                        "id": f"bank-withdraw-{s.get('slotId')}",
                        "action": "click",
                        "description": f"Withdraw all: {s.get('itemName')}",
                        "target": {"name": s.get("itemName"), "bounds": r},
                        "click": {"type": "rect-center"},
                        "preconditions": ["bankOpen == true"],
                        "postconditions": [],
                    })
                    return plan

        # 4) Withdraw coins (one stack per tick)
        for s in bank_slots:
            nm = (s.get("itemName") or "").strip().lower()
            qty = int(s.get("quantity") or 0)
            if nm == "coins" and qty > 0:
                r = _slot_bounds(s)
                if _rect_ok(r):
                    plan["steps"].append({
                        "id": f"bank-withdraw-coins-{s.get('slotId')}",
                        "action": "click",
                        "description": "Withdraw all: Coins",
                        "target": {"name": "Coins", "bounds": r},
                        "click": {"type": "rect-center"},
                        "preconditions": ["bankOpen == true"],
                        "postconditions": [],
                    })
                    return plan

        # 5) Close bank
        plan["steps"].append({
            "id": "bank-close-esc",
            "action": "click",
            "description": "Close Bank (Esc)",
            "click": {"type": "key", "key": "esc"},
            "preconditions": ["bankOpen == true"],
            "postconditions": ["bankOpen == false"],
        })
        return plan

class OpenGEExchangePlan(Plan):
    id = "OPEN_GE_EXCHANGE"
    label = "GE: Open via Clerk"

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        ge_open = bool((payload.get("grand_exchange") or {}).get("open", False))
        if ge_open:
            return "GE already open"

        clerk = self._nearest_clerk(payload)
        return "Click GE clerk" if clerk else "No target"

    def _nearest_clerk(self, payload: dict) -> dict | None:
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

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}
        if phase != "Click GE clerk":
            return plan

        clerk = self._nearest_clerk(payload)
        if not clerk:
            return plan

        canvasX = clerk.get("canvasX")
        canvasY = clerk.get("canvasY")
        if not isinstance(canvasX, (int, float)) or not isinstance(canvasY, (int, float)):
            return plan

        click = {"type": "point", "x": int(canvasX), "y": int(canvasY) - 8}

        plan["steps"].append({
            "id": "ge-exchange-open",
            "action": "click",
            "description": "Click Grand Exchange Clerk",
            "click": click,
            "target": {
                "domain": "npc",
                "name": clerk.get("name"),
                "id": clerk.get("id"),
                "worldX": clerk.get("worldX"),
                "worldY": clerk.get("worldY"),
                "plane": clerk.get("plane", 0),
                "canvasX": canvasX,
                "canvasY": canvasY,
            },
            "preconditions": ["geOpen == false"],
            "postconditions": ["geOpen == true"],
            "confidence": 0.95,
        })
        return plan

class GETradePlan(Plan):
    id = "GE_SELL_BUY"
    label = "GE: Sell Rings & Buy Mats"

    BUY_PAIR = ("Sapphire", "Gold bar")  # flip to ("Emerald", "Gold bar") if you want

    def __init__(self):
        self.state = {
            "phase": "ENSURE_GE_OPEN",
            "sell": {
                "queue": [],  # will be seeded once from GE inventory (in-order)
                "queue_seeded": False,  # ← one-time seeding flag
                "pending_pick": None,
                "active": None,
                "clicks": 0,
                "last_price": None,
            },
            "buy": {
                "items": ["Sapphire", "Gold bar"],
                "idx": 0,
                "step": "OPEN_SLOT",
            },
        }

    def _seed_sell_queue_once(self, payload: dict) -> None:
        s = self.state["sell"]
        if s.get("queue_seeded"):
            return
        inv = (payload.get("ge_inventory") or {})
        items = inv.get("items") or []

        q, seen = [], set()
        for it in items:
            nm = _norm_name(it.get("nameStripped") or it.get("name"))
            if nm not in ("sapphire ring", "emerald ring"):
                continue
            b = _unwrap_rect(it.get("bounds"))
            if not b or int(b.get("x", -1)) < 0 or int(b.get("y", -1)) < 0:
                continue
            if int(it.get("itemQuantity") or 0) <= 0:
                continue

            nice = "Sapphire ring" if nm == "sapphire ring" else "Emerald ring"
            if nice not in seen:
                q.append(nice)
                seen.add(nice)

        s["queue"] = q
        s["queue_seeded"] = True

    # ----------------------------- PHASE ENGINE -----------------------------
    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        st = self.state
        sell = st["sell"]
        buy = st.setdefault("buy", {"want": [], "idx": 0, "typed": False, "clicks": 0, "last_price": None})

        # --- hard facts each tick ---
        ge_open = _ge_open(payload)
        offer_open = _ge_offer_open(payload)  # chooser absent -> True
        price_now = _ge_price_value(payload)  # int or None
        can_confirm = bool(_widget_by_id_text_contains(payload, 30474266, "confirm"))
        has_collect = bool(_widget_by_id_text_contains(payload, 30474246, "collect"))
        ge_widgets = (_ge_widgets(payload) or {})
        print(st)

        # helper: check for any open-slot child "pid:3"
        def _open_slot_child_exists() -> bool:
            for pid in (30474247, 30474248, 30474249, 30474250, 30474251, 30474252, 30474253, 30474254):
                child = ge_widgets.get(f"{pid}:3") or {}
                b = _unwrap_rect(child.get("bounds"))
                if b and int(b.get("width", 0)) > 0 and int(b.get("height", 0)) > 0:
                    return True
            return False

        # helper: verify selected item text in 30474266:27
        def _selected_item_matches(name: str) -> bool:
            w = ge_widgets.get("30474266:27") or {}
            t = _norm_name(w.get("text") or w.get("textStripped"))
            return bool(t) and (t == _norm_name(name))

        # --- 1) GE closed -> ensure open (do NOT touch persistent queues) ---
        if not ge_open:
            st["phase"] = "ENSURE_GE_OPEN"
            sell.update({"pending_pick": None, "active": None, "clicks": 0, "last_price": None})
            # keep buy.want intact if you want to resume mid-buy; otherwise clear transient
            buy.update({"typed": False, "clicks": 0, "last_price": None})
            return st["phase"]

        # --- 2) Seed SELL queue exactly once from current GE inventory ---
        if not sell.get("queue_seeded"):
            q = []
            if _ge_inv_item_by_name(payload, "Sapphire ring"): q.append("Sapphire ring")
            if _ge_inv_item_by_name(payload, "Emerald ring"):  q.append("Emerald ring")
            sell["queue"] = q
            sell["queue_seeded"] = True

        # --- 3) SELL: if an offer finished (panel closed) after 3 cuts -> pop head & reset ---
        if (not offer_open) and sell.get("active") and sell.get("clicks", 0) >= 3:
            if sell.get("queue") and sell["queue"][0] == sell["active"]:
                sell["queue"].pop(0)
            sell.update({"pending_pick": None, "active": None, "clicks": 0, "last_price": None})
            # If Collect is visible, go collect first
            if has_collect:
                st["phase"] = "SELL_COLLECT"
                return st["phase"]

        # --- 4) SELL_COLLECT gate (stay until it disappears, then continue) ---
        if st["phase"] == "SELL_COLLECT":
            if not has_collect:
                st["phase"] = "SELL_PICK" if sell["queue"] else "BUY_OPEN_SLOT"
            return st["phase"]

        # ===== SELL path while there is anything left to sell =====
        if sell.get("queue"):
            head = sell["queue"][0]

            # need to open an offer first
            if not offer_open:
                if sell.get("pending_pick") != head:
                    sell["pending_pick"] = head
                st["phase"] = "SELL_PICK"
                return st["phase"]

            # offer is open: set active (once) and track price drops
            if sell.get("active") is None:
                sell["active"] = sell.get("pending_pick") or head

            if price_now is not None and sell.get("last_price") is None:
                sell["last_price"] = price_now
            if (price_now is not None) and (sell.get("last_price") is not None) and (price_now < sell["last_price"]):
                sell["clicks"] += 1
                sell["last_price"] = price_now

            if sell.get("clicks", 0) < 3:
                st["phase"] = "SELL_MINUS"
                return st["phase"]

            st["phase"] = "SELL_CONFIRM" if can_confirm else "SELL_WAIT_CLOSED"
            return st["phase"]

        # ===== BUY path (no items left to sell) =====

        # 5) One-time BUY want-list seed (when we first switch from sell→buy)
        if not buy.get("want"):
            coins = _coins(payload)
            itemA, itemB = self.BUY_PAIR
            pA = max(1, _price(payload, itemA) or 1)
            pB = max(1, _price(payload, itemB) or 1)
            budget = max(1, int(coins * 0.80))
            qtyA = max(1, (budget // 2) // pA)
            qtyB = max(1, (budget // 2) // pB)
            buy["want"] = [{"name": itemA, "qty": qtyA}, {"name": itemB, "qty": qtyB}]
            buy["idx"] = 0
            buy.update({"typed": False, "clicks": 0, "last_price": None})

        # If all buys done → collect if available, else close GE
        if buy["idx"] >= len(buy["want"]):
            if has_collect:
                st["phase"] = "BUY_COLLECT"
            else:
                st["phase"] = "BUY_CLOSE_GE"
            return st["phase"]

        cur = buy["want"][buy["idx"]]

        # 6) BUY_OPEN_SLOT: wait for any open slot child pid:3 we can click
        if st["phase"] not in (
        "BUY_WAIT_OFFER", "BUY_TYPE_NAME", "BUY_VERIFY_ITEM", "BUY_PLUS", "BUY_CONFIRM", "BUY_WAIT_CLOSED",
        "BUY_COLLECT", "BUY_CLOSE_GE", "BUY_OPEN_SLOT"):
            st["phase"] = "BUY_OPEN_SLOT"
            return st["phase"]

        if st["phase"] == "BUY_OPEN_SLOT":
            if offer_open and ge_widgets.get("30474266"):
                buy.update({"typed": False, "clicks": 0, "last_price": None})
                st["phase"] = "BUY_TYPE_NAME"
            return st["phase"]

        # 8) BUY_TYPE_NAME → after builder types+enter we’ll transition on next tick
        if st["phase"] == "BUY_TYPE_NAME":
            buy = st["buy"]
            items = buy.get("want") or buy.get("items") or []
            idx = int(buy.get("idx") or 0)
            if 0 <= idx < len(items):
                cur = items[idx]
                item_name = (cur.get("name") if isinstance(cur, dict) else str(cur)).strip()
                if _ge_selected_item_is(payload, item_name):
                    st["phase"] = "BUY_PLUS"   # concrete UI proof met → advance
            # else stay in BUY_TYPE_NAME
            return st["phase"]

        # 9) BUY_VERIFY_ITEM: wait until 30474266:27 shows our item name
        if st["phase"] == "BUY_VERIFY_ITEM":
            if _selected_item_matches(cur["name"]):
                st["phase"] = "BUY_PLUS"
            return st["phase"]

        # 10) BUY_PLUS: press +5% until 3 confirmed price raises
        if st["phase"] == "BUY_PLUS":
            # initialize last_price when we first see it
            if (price_now is not None) and (buy.get("last_price") is None):
                buy["last_price"] = price_now

            # count only confirmed increases (i.e., UI price changed upward)
            if (price_now is not None) and (buy.get("last_price") is not None) and (price_now > buy["last_price"]):
                buy["clicks"] = int(buy.get("clicks") or 0) + 1
                buy["last_price"] = price_now

            # after 3 bumps, move to confirm if available; otherwise stay here
            if int(buy.get("clicks") or 0) >= 3:
                st["phase"] = "BUY_CONFIRM" if can_confirm else "BUY_PLUS"
            return st["phase"]

        # 11) BUY_CONFIRM: wait for builder to click Confirm; then we wait for close
        if st["phase"] == "BUY_CONFIRM":
            if not offer_open:
                st["phase"] = "BUY_WAIT_CLOSED"
            return st["phase"]

        # 12) BUY_WAIT_CLOSED: when panel closes, advance to next item or collecting
        if st["phase"] == "BUY_WAIT_CLOSED":
            if not offer_open:
                buy["idx"] += 1
                buy.update({"typed": False, "clicks": 0, "last_price": None})
                if buy["idx"] >= len(buy["want"]):
                    st["phase"] = "BUY_COLLECT" if has_collect else "BUY_CLOSE_GE"
                else:
                    st["phase"] = "BUY_OPEN_SLOT"
            return st["phase"]

        # 13) BUY_COLLECT: stay here until Collect disappears, then close GE
        if st["phase"] == "BUY_COLLECT":
            if not has_collect:
                st["phase"] = "BUY_CLOSE_GE"
            return st["phase"]

        # 14) BUY_CLOSE_GE: builder will press ESC; when GE is closed we’ll drop back to ENSURE_GE_OPEN
        if st["phase"] == "BUY_CLOSE_GE":
            if not ge_open:
                st["phase"] = "ENSURE_GE_OPEN"
            return st["phase"]

        # fallback (should not happen)
        return st["phase"]

    # ----------------------------- ACTION BUILDER ---------------------------
    # IMPORTANT: this NEVER mutates self.state["phase"].
    def build_action_plan(self, payload: dict, phase: str) -> dict:
        st = self.state
        sell = st["sell"]
        plan = {"phase": phase, "steps": []}

        # ENSURE_GE_OPEN → actually click the clerk
        if phase == "ENSURE_GE_OPEN":
            if not _ge_open(payload):
                me = payload.get("player") or {}
                def _nearest_clerk():
                    best, best_d2 = None, 1e18
                    for npc in (payload.get("closestNPCs") or []) + (payload.get("npcs") or []):
                        nm = (npc.get("name") or "").lower()
                        nid = int(npc.get("id") or -1)
                        if "grand exchange clerk" not in nm and not (2148 <= nid <= 2151):
                            continue
                        if int(npc.get("plane") or 0) != int(me.get("plane") or 0):
                            continue
                        dx = int(npc.get("worldX") or 0) - int(me.get("worldX") or 0)
                        dy = int(npc.get("worldY") or 0) - int(me.get("worldY") or 0)
                        d2 = dx*dx + dy*dy
                        if d2 < best_d2:
                            best, best_d2 = npc, d2
                    return best
                clerk = _nearest_clerk()
                if clerk and isinstance(clerk.get("canvasX"), (int, float)) and isinstance(clerk.get("canvasY"), (int, float)):
                    plan["steps"].append({
                        "id": "ge-exchange-open",
                        "action": "click",
                        "description": "Click Grand Exchange Clerk",
                        "click": {"type": "point", "x": int(clerk["canvasX"]), "y": int(clerk["canvasY"]) - 8},
                        "target": {"domain": "npc", "name": clerk.get("name"), "id": clerk.get("id")},
                        "preconditions": ["geOpen == false"],
                        "postconditions": ["geOpen == true"],
                        "confidence": 0.95,
                    })
            return plan

        # --------------------- SELL_PICK --------------------------
        if st["phase"] == "SELL_PICK":
            nm = st["sell"]["pending_pick"]
            if nm:
                rect = _ge_inv_slot_bounds(payload, nm)
                if rect:
                    plan["steps"].append({
                        "id": "ge-offer-open",
                        "action": "click",
                        "description": f"Offer {nm}",
                        "target": {"name": "GE inv item", "bounds": rect},
                        "click": {"type": "rect-center"},
                        "preconditions": ["geOpen == true"],
                        "postconditions": [],
                        "confidence": 0.95,
                    })
            return plan

        if phase == "SELL_MINUS":
            minus = _widget_by_id_text(payload, 30474266, "-5%")
            if minus:
                cx, cy = _rect_center_from_widget(minus)
                plan["steps"].append({
                    "id": "ge-minus5",
                    "action": "click",
                    "description": f"GE price -5% ({sell['clicks'] + 1}/3)",
                    "target": {"name": "-5%", "bounds": minus.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": ["geOpen == true"],
                    "postconditions": [],  # compute_phase confirms the drop via price text
                    "confidence": 0.95,
                })
            return plan

        # --------------------- SELL_CONFIRM -----------------------
        if st["phase"] == "SELL_CONFIRM":
            confirm = _widget_by_id_text_contains(payload, 30474266, "confirm")
            if confirm:
                cx, cy = _rect_center_from_widget(confirm)
                plan["steps"].append({
                    "id": "ge-confirm-offer",
                    "action": "click",
                    "description": f"Confirm offer for {st['sell']['active']}",
                    "target": {"name": "Confirm", "bounds": confirm.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": ["geOpen == true"],
                    "postconditions": [],
                    "confidence": 0.95,
                })
            return plan

        # --------------------- SELL_WAIT_CLOSED -------------------
        if st["phase"] == "SELL_WAIT_CLOSED":
            # chooser present again → panel closed
            if not _ge_offer_open(payload):
                # remove the item we just sold from the queue (only here)
                try:
                    if st["sell"]["active"] in st["sell"]["queue"]:
                        st["sell"]["queue"].remove(st["sell"]["active"])
                except Exception:
                    pass
                st["sell"].update({
                    "active": None,
                    "pending_pick": None,
                    "clicks": 0,
                    "last_price": None
                })
            return plan

        # --------------------- SELL_COLLECT -----------------------
        if st["phase"] == "SELL_COLLECT":
            collect = _widget_by_id_text_contains(payload, 30474246, "collect")
            if collect:
                cx, cy = _rect_center_from_widget(collect)
                plan["steps"].append({
                    "id": "ge-collect",
                    "action": "click",
                    "description": "Collect proceeds",
                    "target": {"name": "Collect", "bounds": collect.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": ["geOpen == true"],
                    "postconditions": [],
                    # next tick: compute_phase will see Collect gone, then move to BUY_INIT (or SELL_PICK if queue not empty)
                    "confidence": 0.95,
                })
            return plan

        if phase == "BUY_OPEN_SLOT":
            btn = _ge_first_buy_slot_btn(payload)
            if btn:
                x, y = _rect_center_from_widget(btn)
                plan["steps"].append({
                    "id": "ge-buy-open-slot",
                    "action": "click",
                    "description": "Open buy slot",
                    "target": {"name": "GE slot button", "bounds": btn.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                    "preconditions": ["geOpen == true"],
                    "postconditions": []
                })
            return plan

        if phase == "BUY_TYPE_NAME":
            buy = st["buy"]
            items = buy.get("want") or buy.get("items") or []
            idx = int(buy.get("idx") or 0)
            if not isinstance(items, list) or not (0 <= idx < len(items)):
                return plan

            cur = items[idx]
            item_name = (cur.get("name") if isinstance(cur, dict) else str(cur)).strip()
            if not item_name:
                return plan

            # three-step sequencing: TYPE -> WAIT -> ENTER
            per_ms = 50  # tune typing speed as you like
            window_ms = 60000  # treat steps as "done" for up to a minute for this item

            type_id = f"ge-type-name-{idx}-{item_name}"
            wait_id = f"ge-type-wait-{idx}"
            enter_id = f"ge-type-enter-{idx}"

            # 1) If we haven't typed the name yet, do that first (and don't press Enter yet)
            if not step_recent(type_id, window_ms):
                plan["steps"].append({
                    "id": type_id,
                    "action": "click",
                    "click": {
                        "type": "type",
                        "text": item_name,
                        "enter": False,  # <-- important: no Enter here
                        "per_char_ms": per_ms,
                        "focus": True
                    },
                    "description": f"type '{item_name}'"
                })
                return plan

            # 2) After typing, wait a short moment so the UI settles
            if not step_recent(wait_id, window_ms):
                plan["steps"].append({
                    "id": wait_id,
                    "action": "click",
                    "click": {"type": "wait", "ms": max(300, per_ms * len(item_name) + 50)},
                    "description": "Pause before pressing Enter"
                })
                return plan

            # 3) Finally, press Enter once
            if not step_recent(enter_id, window_ms):
                plan["steps"].append({
                    "id": enter_id,
                    "action": "click",
                    "click": {"type": "key", "key": "enter"},
                    "description": "Confirm search"
                })
                return plan

            # Nothing to do; compute_phase will advance once it sees the selected item
            return plan

        if phase == "BUY_VERIFY_ITEM":
            return plan  # pure wait; compute_phase checks 30474266:27

        if phase == "BUY_PLUS":
            plus = _widget_by_id_text(payload, 30474266, "+5%")
            if plus:
                cx, cy = _rect_center_from_widget(plus)
                plan["steps"].append({
                    "id": "ge-plus5",
                    "action": "click",
                    "description": f"GE price +5% ({int(st['buy'].get('clicks') or 0) + 1}/3)",
                    "target": {"name": "+5%", "bounds": plus.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": ["geOpen == true"],
                    "postconditions": [],  # compute_phase confirms the raise via price text
                    "confidence": 0.95,
                })
            return plan

        if phase == "BUY_CONFIRM":
            conf = _ge_buy_confirm_widget(payload)
            if conf:
                x, y = _rect_center_from_widget(conf)
                plan["steps"].append({
                    "id": "ge-buy-confirm", "action": "click",
                    "description": "Confirm buy",
                    "target": {"name": "Confirm", "bounds": conf.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                })
            return plan

        if phase == "BUY_WAIT_CLOSED":
            return plan  # pure wait

        if phase == "BUY_COLLECT":
            collect = _widget_by_id_text_contains(payload, 30474246, "collect")
            if collect:
                x, y = _rect_center_from_widget(collect)
                plan["steps"].append({
                    "id": "ge-collect", "action": "click",
                    "description": "Collect proceeds",
                    "target": {"name": "Collect", "bounds": collect.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                })
            return plan

        if phase == "BUY_CLOSE_GE":
            # Press Esc once; compute_phase will see ge closed on next tick and naturally settle in ENSURE_GE_OPEN
            plan["steps"].append({"id": "ge-close", "action": "click", "description": "Close GE",
                                  "click": {"type": "key", "key": "escape"}})
            return plan

        # fallback
        return plan

    # ----- local (class) helper, not a global duplicate -----
    def _nearest_clerk(self, payload: dict) -> dict | None:
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

    def _ge_slot_child3(self, payload: dict) -> dict | None:
        """Return the first slot's child index 3 (30474247..54:3) that has bounds."""
        W = (_ge_widgets(payload) or {})
        for pid in ("30474247", "30474248", "30474249", "30474250", "30474251", "30474252", "30474253", "30474254"):
            w = W.get(f"{pid}:3")
            if w and (w.get("bounds") or {}).get("width"):
                return w
        return None

class GoToEdgevilleBankPlan(Plan):
    """
    Chooses a ground tile from tiles_15x15 that best advances the player toward the
    Edgeville bank region, and clicks it using the tile's canvas coordinates.
    """
    id = "GO_TO_EDGE_BANK"
    label = "Go to Edgeville Bank"

    # -------------------- PHASE LOGIC --------------------
    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        me = (payload.get("player") or {})
        tiles = payload.get("tiles_15x15") or []
        if not me or not tiles:
            return "No Player/Tiles"

        try:
            wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
        except Exception:
            return "No Player/Tiles"

        # Inside Edgeville bank region?
        if EDGE_BANK_MIN_X <= wx <= EDGE_BANK_MAX_X and EDGE_BANK_MIN_Y <= wy <= EDGE_BANK_MAX_Y:
            return "Arrived at Bank"

        return "Moving to Bank"

    # -------------------- HELPERS --------------------
    def _edge_bank_center(self) -> tuple[int, int]:
        # Center of the Edgeville bank region from constants.py
        bx = (EDGE_BANK_MIN_X + EDGE_BANK_MAX_X) // 2
        by = (EDGE_BANK_MIN_Y + EDGE_BANK_MAX_Y) // 2
        return bx, by

    def _line_to(self, x0: int, y0: int, x1: int, y1: int, max_steps: int = 14):
        """
        Integer grid line from (x0,y0) to (x1,y1), up to max_steps ahead (not including start).
        Classic Bresenham; returns a list of (x,y) points along the path.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1 if x0 > x1 else 0
        sy = 1 if y0 < y1 else -1 if y0 > y1 else 0
        x, y = x0, y0

        if dx >= dy:
            err = dx // 2
            for _ in range(max_steps):
                if x == x1 and y == y1:
                    break
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                points.append((x, y))
        else:
            err = dy // 2
            for _ in range(max_steps):
                if x == x1 and y == y1:
                    break
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                points.append((x, y))
        return points

    def _pick_tile_toward_edge(self, payload: dict) -> dict | None:
        me = (payload.get("player") or {})
        tiles = payload.get("tiles_15x15") or []

        try:
            wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
        except Exception:
            return None

        # target = Edgeville bank center (from constants)
        bx, by = self._edge_bank_center()

        # Build a lookup of clickable tiles (must have canvas coords)
        by_world = {}
        for t in tiles:
            try:
                tx, ty = int(t.get("worldX")), int(t.get("worldY"))
                cx, cy = t.get("canvasX"), t.get("canvasY")
            except Exception:
                continue
            if isinstance(cx, int) and isinstance(cy, int):
                by_world[(tx, ty)] = t

        # Walk straight line toward bank center, pick the farthest visible tile
        path = self._line_to(wx, wy, bx, by, max_steps=14)
        pick = None
        for pt in reversed(path):  # farthest first
            if pt in by_world:
                pick = by_world[pt]
                break

        return pick

    # -------------------- ACTION BUILDER --------------------
    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}

        if phase == "Arrived at Bank":
            plan["steps"].append({
                "action": "idle",
                "description": "Player is inside Edgeville bank region",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "n/a"},
                "preconditions": [],
                "postconditions": [],
                "confidence": 1.0
            })
            return plan

        if phase != "Moving to Bank":
            plan["steps"].append({
                "action": "idle",
                "description": "No actionable step",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "n/a"},
                "preconditions": [],
                "postconditions": [],
                "confidence": 0.0
            })
            return plan

        pick = self._pick_tile_toward_edge(payload)
        if pick:
            # Debug context
            me = (payload.get("player") or {})
            wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
            bx, by = self._edge_bank_center()

            cx, cy = int(pick["canvasX"]), int(pick["canvasY"])
            tx, ty = int(pick.get("worldX", 0)), int(pick.get("worldY", 0))
            plane = int(pick.get("plane", 0))

            plan["steps"].append({
                "action": "click-ground",
                "description": f"toward Edge bank {bx},{by} from {wx},{wy}",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {
                    "domain": "ground",
                    "name": f"Tile→EDGE_BANK({bx},{by})",
                    "world": {"x": tx, "y": ty, "plane": plane},
                    "canvas": {"x": cx, "y": cy},
                    "debug": {
                        "player": {"x": wx, "y": wy},
                        "edge_bank_center": {"x": bx, "y": by},
                        "goal_vec": {"dx": bx - wx, "dy": by - wy},
                        "chosen_tile_world": {"x": tx, "y": ty, "plane": plane},
                        "chosen_tile_canvas": {"x": cx, "y": cy},
                    }
                },
                "preconditions": [],
                "postconditions": [],
                "confidence": 0.85
            })
            return plan

        # No reachable tile on this tick
        plan["steps"].append({
            "action": "idle",
            "description": "No clickable tile toward Edgeville bank on this tick",
            "click": {"type": "none"},
            "target": {"domain": "none", "name": "n/a"},
            "preconditions": [],
            "postconditions": [],
            "confidence": 0.0
        })
        return plan



# ------------- Registry & accessors -------------
PLAN_REGISTRY: Dict[str, Plan] = {
    SapphireRingsPlan.id: SapphireRingsPlan(),
    GoldRingsPlan.id:     GoldRingsPlan(),
    EmeraldRingsPlan.id: EmeraldRingsPlan(),
    GoToGEPlan.id:        GoToGEPlan(),
    OpenGEBankPlan.id:    OpenGEBankPlan(),
    GEWithdrawRingsAsNotesPlan.id: GEWithdrawRingsAsNotesPlan(),
    OpenGEExchangePlan.id: OpenGEExchangePlan(),
    GETradePlan.id: GETradePlan(),
    GoToEdgevilleBankPlan.id: GoToEdgevilleBankPlan(),
}

def get_plan(plan_id: str) -> Plan:
    return PLAN_REGISTRY.get(plan_id, PLAN_REGISTRY["SAPPHIRE_RINGS"])
