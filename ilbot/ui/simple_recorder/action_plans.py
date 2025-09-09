# action_plans.py
import re
from typing import Dict, Callable
from .nav_simple import next_tile_toward
from .constants import GE_MIN_X, GE_MAX_X, GE_MIN_Y, GE_MAX_Y


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
                    "action": "click-bank",
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
                    "action": "click-bank",
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
            obj = _closest_object_by_names(payload, ["bank booth", "banker"])
            if obj:
                rect = _unwrap_rect(obj.get("clickbox"))
                step = {
                    "action": "click-bank",
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


# ------------- Registry & accessors -------------
PLAN_REGISTRY: Dict[str, Plan] = {
    SapphireRingsPlan.id: SapphireRingsPlan(),
    GoldRingsPlan.id:     GoldRingsPlan(),
    EmeraldRingsPlan.id: EmeraldRingsPlan(),
    GoToGEPlan.id:        GoToGEPlan(),
}

def get_plan(plan_id: str) -> Plan:
    return PLAN_REGISTRY.get(plan_id, PLAN_REGISTRY["SAPPHIRE_RINGS"])
