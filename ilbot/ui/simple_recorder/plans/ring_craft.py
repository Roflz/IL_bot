from .base import Plan
from ilbot.ui.simple_recorder.actions import *
from ..helpers.bank import first_bank_slot
from ..helpers.inventory import inv_count, inv_has, first_inv_slot
from ..helpers.rects import unwrap_rect
from ..helpers.utils import closest_object_by_names
from ..helpers.widgets import craft_widget_rect


class RingCraftPlan(Plan):
    """
    Crafts best-allowed ring based on crafting level + materials in inventory/bank.
    Priority (strict): Emerald (>=27) → Sapphire (>=20) → Gold (<20 only).
    Banking rules:
      - Keep inventory as: [1x Ring mould] + [N Gold bars] + [N Gems (Sapp+Emer total)] and nothing else.
      - Prefer N = 13 when possible; otherwise N = min(13, available pairs).
      - If inventory deviates (extra items, unequal gold vs gems), Deposit-All then restock.
      - Ensure Withdraw-X is selected before withdrawing.
      - Finish by crafting ALL remaining materials, then Deposit-All and close bank.
    """
    id = "RING_CRAFT"
    label = "Craft Rings"

    DEFAULT_REQS = {"Gold ring": 5, "Sapphire ring": 20, "Emerald ring": 27}

    def __init__(self):
        # Start in a phase your build_action_plan understands.
        self.state = {
            "phase": "Moving to bank",
            "done": False,
        }

    # ------ Level & choice helpers ------
    def _level(self, payload: dict) -> int:
        # From your plugin addition: payload["skills"]["craftingLevel"]
        return int((payload.get("skills") or {}).get("craftingLevel") or 1)

    def _reqs(self, payload: dict) -> dict:
        r = dict(self.DEFAULT_REQS)
        r.update(payload.get("craftingLevelReqs") or {})
        return r

    def _can_make(self, payload: dict, ring_name: str) -> bool:
        lvl = self._level(payload)
        reqs = self._reqs(payload)
        need = int(reqs.get(ring_name, 999))
        if ring_name == "Gold ring":
            # hard gate: only allow Gold ring when level < 20 (rule #6)
            return lvl < 20 and lvl >= need
        return lvl >= need

    def _choose_ring(self, payload: dict) -> str | None:
        """Choose which ring to craft next given level and mats in *inventory* (not bank)."""
        inv_gold = inv_count(payload, "Gold bar")
        inv_sapp = inv_count(payload, "Sapphire")
        inv_emer = inv_count(payload, "Emerald")
        has_mould = inv_has(payload, "Ring mould")

        # prioritize Emerald → Sapphire; Gold is only for lvl < 20
        if has_mould and inv_gold > 0:
            if inv_emer > 0 and self._can_make(payload, "Emerald ring"):
                return "Emerald ring"
            if inv_sapp > 0 and self._can_make(payload, "Sapphire ring"):
                return "Sapphire ring"
            if self._can_make(payload, "Gold ring"):
                return "Gold ring"
        return None

    # ------ Bank & inventory shaping helpers ------
    def _bank_qty_x_selected(self, payload: dict) -> bool:
        bw = payload.get("bank_widgets") or {}
        layer = (bw.get("withdraw_quantity_layer") or {})
        for opt in (layer.get("options") or []):
            t = (opt.get("text") or "").strip().lower()
            if t == "x":
                return bool(opt.get("selected"))
        return False

    def _deposit_all_button_bounds(self, payload: dict) -> dict | None:
        bw = payload.get("bank_widgets") or {}
        dep = bw.get("deposit_inventory") or {}
        b = dep.get("bounds") or None
        if isinstance(b, dict) and int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0:
            return b
        return None

    def _bank_available_counts(self, payload: dict) -> tuple[int, int, int]:
        """Return (goldBars, sapphires, emeralds) available in bank (by quantity)."""
        gold = sapp = emer = 0
        for s in (payload.get("bank") or {}).get("slots") or []:
            nm = (s.get("itemName") or "")
            q = int(s.get("quantity") or 0)
            if nm == "Gold bar":
                gold += q
            elif nm == "Sapphire":
                sapp += q
            elif nm == "Emerald":
                emer += q
        return gold, sapp, emer

    def _inventory_gem_total(self, payload: dict) -> int:
        return inv_count(payload, "Sapphire") + inv_count(payload, "Emerald")

    def _inventory_is_clean_balanced(self, payload: dict, allow_unequal: bool = False) -> bool:
        """
        True when inventory matches banking shape:
          - Exactly 1 Ring mould
          - Only Gold bars + Sapphires + Emeralds besides the mould
          - If allow_unequal=False: counts of gold == total gems and > 0
            If allow_unequal=True: we only require mould present and no foreign items
          - SPECIAL: an entirely empty inventory is considered "clean" (so we don't spam Deposit-All)
        """
        inv_meta = (payload.get("inventory") or {})
        if int(inv_meta.get("totalItems") or 0) == 0:
            return True  # <- empty inventory = clean; don't click Deposit-All

        # counts
        mould = inv_count(payload, "Ring mould")
        gold = inv_count(payload, "Gold bar")
        sapp = inv_count(payload, "Sapphire")
        emer = inv_count(payload, "Emerald")
        gems = sapp + emer

        # ensure no foreign items
        allowed_names = {"Ring mould", "Gold bar", "Sapphire", "Emerald"}
        for slot in inv_meta.get("slots") or []:
            nm = slot.get("itemName")
            qty = int(slot.get("quantity") or 0)
            if qty > 0 and (nm not in allowed_names):
                return False

        if mould != 1:
            return False
        if allow_unequal:
            return True
        return (gold > 0) and (gold == gems)

    def _target_withdraw_amount(self, payload: dict) -> int:
        """
        We aim for 13 pairs; otherwise min(13, bank_gold, bank_gems_total).
        """
        bank_gold, bank_sapp, bank_emer = self._bank_available_counts(payload)
        bank_gems = bank_sapp + bank_emer
        if bank_gold <= 0 or bank_gems <= 0:
            return 0
        return max(1, min(13, bank_gold, bank_gems))

    # ------ Phase engine ------
    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        st = self.state
        prev = st.get("phase") or "Moving to bank"

        inv = (payload.get("inventory") or {}).get("slots") or []
        bank_open = bool((payload.get("bank") or {}).get("bankOpen", False))

        def _qty(inv_like, name):
            name = (name or "").lower()
            for s in inv_like:
                if (s.get("itemName") or "").lower() == name:
                    return int(s.get("quantity") or 0)
            return 0

        have_gold = _qty(inv, "Gold bar")
        have_sapph = _qty(inv, "Sapphire")
        have_emerl = _qty(inv, "Emerald")
        have_mould = any((s.get("itemName") or "").lower() == "ring mould" for s in inv)

        # Heuristic: if any make_* widget has bounds, we’re on the craft screen
        def _craft_ui_open() -> bool:
            keys = ("make_gold_rings", "make_sapphire_rings", "make_emerald_rings")
            return any(craft_widget_rect(payload, k) for k in keys)

        # Can we (still) craft from current inventory?
        can_craft_now = have_mould and have_gold > 0 and (have_sapph > 0 or have_emerl > 0)

        # 2) If bank is open, shape inventory / withdraw mats
        if bank_open:
            st["phase"] = "Banking"
            return st["phase"]

        # --- STICKY CRAFTING ---
        # If we were already crafting and still have mats, keep waiting in Crafting even if the UI closed.
        if prev == "Crafting" and can_craft_now:
            st["phase"] = "Crafting"
            return st["phase"]

        # 1) If we can craft now but aren’t crafting yet: Craft if UI open; otherwise go open it.
        if can_craft_now:
            st["phase"] = "Crafting" if _craft_ui_open() else "Moving to furnace"
            return st["phase"]


        # 3) If we’ve just crafted and have rings/leftovers, go bank
        if craft_recent or any(_qty(inv, nm) > 0 for nm in ("Gold ring", "Sapphire ring", "Emerald ring")):
            st["phase"] = "Moving to bank"
            return st["phase"]

        # 4) Default: go check bank; don’t assume emptiness with bank closed
        st["phase"] = "Moving to bank"
        return st["phase"]

    # ------ Actions ------
    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}
        ring = self._choose_ring(payload) or ("Gold ring" if self._level(payload) < 20 else "Sapphire ring")

        # --- Moving to bank ---
        if phase == "Moving to bank":
            obj = closest_object_by_names(payload, ["bank booth", "banker"])
            if obj:
                rect = unwrap_rect(obj.get("clickbox"))
                plan["steps"].append({
                    "action": "click",
                    "description": "Open nearest bank",
                    "click": ({"type": "rect-center"} if rect else
                              {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}),
                    "target": {"domain": "object", "name": obj.get("name"), "id": obj.get("id"), "clickbox": rect},
                    "preconditions": ["bankOpen == false"], "postconditions": ["bankOpen == true"],
                    "confidence": 0.9 if rect else 0.6
                })
            return plan

        # --- Banking ---
        if phase == "Banking":
            # A) If we have any finished rings, deposit those FIRST (preferred path after crafting)
            for nm in ("Gold ring", "Sapphire ring", "Emerald ring"):
                inv_slot = first_inv_slot(payload, nm)
                if inv_slot:
                    rect = unwrap_rect(inv_slot.get("bounds"))
                    plan["steps"].append({
                        "action": "deposit-inventory-item",
                        "description": f"Deposit {nm}",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "inventory", "name": nm, "slotId": inv_slot.get("slotId"), "bounds": rect},
                        "preconditions": ["bankOpen == true"], "postconditions": [], "confidence": 0.9 if rect else 0.4
                    })
                    return plan

            # B) STRICT (progress-aware): Inventory must match the expected banking state or we Deposit-All.
            inv = (payload.get("inventory") or {})
            inv_has_any = int(inv.get("totalItems") or 0) > 0

            if inv_has_any:
                # Expected counts and allowed gem type (must be craftable AND available in bank)
                N = self._target_withdraw_amount(payload)

                prefer_emer = self._can_make(payload, "Emerald ring") and (
                            first_bank_slot(payload, "Emerald") is not None)
                prefer_sapp = self._can_make(payload, "Sapphire ring") and (
                            first_bank_slot(payload, "Sapphire") is not None)

                allowed_gem = "Emerald" if prefer_emer else ("Sapphire" if prefer_sapp else None)

                # Actual counts
                mould = inv_count(payload, "Ring mould")
                gold = inv_count(payload, "Gold bar")
                sapp = inv_count(payload, "Sapphire")
                emer = inv_count(payload, "Emerald")
                gems = sapp + emer

                strict_ok = True

                # 1) No foreign items
                allowed_names = {"ring mould", "gold bar", "sapphire", "emerald"}
                for s in (inv.get("slots") or []):
                    q = int(s.get("quantity") or 0)
                    if q <= 0:
                        continue
                    nm = (s.get("itemName") or "").strip().lower()
                    if nm not in allowed_names:
                        strict_ok = False
                        break

                # 2) Mould: ≤ 1
                if strict_ok and mould > 1:
                    strict_ok = False

                # 3) Gem type gating
                if strict_ok:
                    if allowed_gem is None:
                        # Can't craft any gem → there should be no gold/gems present
                        if gold > 0 or gems > 0:
                            strict_ok = False
                    else:
                        if allowed_gem == "Emerald" and sapp > 0:
                            strict_ok = False
                        if allowed_gem == "Sapphire" and emer > 0:
                            strict_ok = False

                # 4) Pairing & progression-aware bounds:
                #    Accept these transient states:
                #      - (0,0) nothing yet
                #      - (gold>0<=N, gems==0) after withdrawing gold, before gems
                #      - (gold==N, 0<gems<=N) withdrawing gems up to N
                #      - (gold==gems<=N) fully paired
                if strict_ok:
                    acceptable = False
                    if gold == 0 and gems == 0:
                        acceptable = True
                    elif gold > 0 and gems == 0 and gold <= N:
                        acceptable = True
                    elif gold == N and 0 < gems <= N:
                        acceptable = True
                    elif gold == gems and gold <= N:
                        acceptable = True
                    if not acceptable:
                        strict_ok = False

                if not strict_ok:
                    dep_bounds = self._deposit_all_button_bounds(payload)
                    if dep_bounds:
                        plan["steps"].append({
                            "action": "click",
                            "description": "Deposit-all (inventory not in expected banking state)",
                            "click": {"type": "rect-center"},
                            "target": {"domain": "widget", "name": "deposit_inventory", "bounds": dep_bounds},
                            "preconditions": ["bankOpen == true"], "postconditions": []
                        })
                        return plan
            # (If inventory is empty or exactly matches expectations, continue.)

            # C) Ensure Withdraw-X is selected
            bw = (payload.get("bank_widgets") or {})
            qx = (bw.get("withdraw_quantity_X") or {})  # {"bounds": {...}, "selected": bool}
            b = qx.get("bounds") or {}

            if not bool(qx.get("selected")) and int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0:
                plan["steps"].append({
                    "id": "bank-qty-x",
                    "action": "click",
                    "description": "Set bank withdraw quantity to X",
                    "target": {"domain": "widget", "name": "bank_qty_X", "bounds": b},
                    "click": {"type": "rect-center"},
                    "preconditions": ["bankOpen == true"],
                    "postconditions": []
                })
                return plan

            # D) Ensure we have exactly 1 ring mould
            if inv_count(payload, "Ring mould") < 1:
                bslot = first_bank_slot(payload, "Ring mould")
                if bslot:
                    rect = unwrap_rect(bslot.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": "Withdraw Ring mould",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "bank", "name": "Ring mould", "slotId": bslot.get("slotId"),
                                   "bounds": rect},
                        "preconditions": ["bankOpen == true"], "postconditions": []
                    })
                    return plan

            # E) Compute desired batch size N
            N = self._target_withdraw_amount(payload)
            if N == 0:
                plan["steps"].append({
                    "action": "click",
                    "description": "Close bank (no pairs available)",
                    "click": {"type": "key", "key": "ESC"},
                    "target": {"domain": "widget", "name": "bank_close"},
                    "preconditions": ["bankOpen == true"], "postconditions": ["bankOpen == false"], "confidence": 0.95
                })
                return plan

            # F) Withdraw materials to reach exactly: 1 mould + N gold + N gems (emeralds preferred if craftable)
            gold = inv_count(payload, "Gold bar")
            sapp = inv_count(payload, "Sapphire")
            emer = inv_count(payload, "Emerald")
            gems = sapp + emer

            if gold < N:
                bslot = first_bank_slot(payload, "Gold bar")
                if bslot:
                    rect = unwrap_rect(bslot.get("bounds"))
                    plan["steps"].append({
                        "action": "withdraw-item",
                        "description": f"Withdraw Gold bar (need {N - gold})",
                        "click": {"type": "rect-center"} if rect else {"type": "none"},
                        "target": {"domain": "bank", "name": "Gold bar", "slotId": bslot.get("slotId"), "bounds": rect},
                        "preconditions": ["bankOpen == true"], "postconditions": []
                    })
                    return plan

            need_gems = max(0, N - gems)
            if need_gems > 0:
                if self._can_make(payload, "Emerald ring"):
                    bslot = first_bank_slot(payload, "Emerald")
                    if bslot:
                        rect = unwrap_rect(bslot.get("bounds"))
                        plan["steps"].append({
                            "action": "withdraw-item",
                            "description": f"Withdraw Emerald (need {need_gems})",
                            "click": {"type": "rect-center"} if rect else {"type": "none"},
                            "target": {"domain": "bank", "name": "Emerald", "slotId": bslot.get("slotId"),
                                       "bounds": rect},
                            "preconditions": ["bankOpen == true"], "postconditions": []
                        })
                        return plan
                if self._can_make(payload, "Sapphire ring"):
                    bslot = first_bank_slot(payload, "Sapphire")
                    if bslot:
                        rect = unwrap_rect(bslot.get("bounds"))
                        plan["steps"].append({
                            "action": "withdraw-item",
                            "description": f"Withdraw Sapphire (need {need_gems})",
                            "click": {"type": "rect-center"} if rect else {"type": "none"},
                            "target": {"domain": "bank", "name": "Sapphire", "slotId": bslot.get("slotId"),
                                       "bounds": rect},
                            "preconditions": ["bankOpen == true"], "postconditions": []
                        })
                        return plan

            # G) Final guard: if something weird shows up, allow a re-clean (rare)
            if inv_has_any and self._inv_has_foreign_items(payload):
                dep_bounds = self._deposit_all_button_bounds(payload)
                if dep_bounds:
                    plan["steps"].append({
                        "action": "click",
                        "description": "Re-clean inventory (Deposit-all) due to mismatch",
                        "click": {"type": "rect-center"},
                        "target": {"domain": "widget", "name": "deposit_inventory", "bounds": dep_bounds},
                        "preconditions": ["bankOpen == true"], "postconditions": []
                    })
                    return plan

            # H) Ready: close bank
            plan["steps"].append({
                "action": "click",
                "description": "Close bank (ready to craft)",
                "click": {"type": "key", "key": "ESC"},
                "target": {"domain": "widget", "name": "bank_close"},
                "preconditions": ["bankOpen == true"], "postconditions": ["bankOpen == false"], "confidence": 0.95
            })
            return plan

        # --- Moving to furnace ---
        if phase == "Moving to furnace":
            obj = closest_object_by_names(payload, ["furnace"])
            if obj:
                rect = unwrap_rect(obj.get("clickbox"))
                plan["steps"].append({
                    "action": "click-furnace",
                    "description": "Click nearest furnace",
                    "click": ({"type": "rect-center"} if rect else
                              {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}),
                    "target": {"domain": "object", "name": obj.get("name"), "id": obj.get("id"), "clickbox": rect},
                    "preconditions": ["bankOpen == false"], "postconditions": ["craftingInterfaceOpen == true"],
                    "confidence": 0.9 if rect else 0.6
                })
            return plan

        # --- Crafting (respect strict ring priority and level gates; rule #6) ---
        if phase == "Crafting":
            # pick best allowed *given current inventory*
            ring = self._choose_ring(payload) or ("Gold ring" if self._level(payload) < 20 else "Sapphire ring")
            # if level >= 20, never choose Gold
            if self._level(payload) >= 20 and ring == "Gold ring":
                ring = "Sapphire ring" if inv_count(payload, "Sapphire") > 0 else "Emerald ring"

            key = "make_gold_rings" if ring == "Gold ring" else (
                  "make_sapphire_rings" if ring == "Sapphire ring" else "make_emerald_rings")
            make_rect = craft_widget_rect(payload, key)
            plan["steps"].append({
                "action": "click-make-widget",
                "description": f"Make {ring}",
                "click": {"type": "rect-center"} if make_rect else {"type": "none"},
                "target": {"domain": "widget", "name": key, "bounds": make_rect},
                "preconditions": ["craftingInterfaceOpen == true"], "postconditions": [], "confidence": 0.95 if make_rect else 0.4
            })
            plan["steps"].append({
                "action": "wait-crafting-complete",
                "description": "Wait until bars/gems consumed",
                "click": {"type": "none"},
                "target": {"domain": "none", "name": "crafting_wait"},
                "preconditions": [], "postconditions": [], "confidence": 1.0
            })
            return plan

        # --- Finalize Banking (rule #5) ---
        if phase == "Finalize Banking":
            # If bank is closed, open it
            if not bool((payload.get("bank") or {}).get("bankOpen", False)):
                obj = closest_object_by_names(payload, ["bank booth", "banker", "grand exchange booth"])
                if obj:
                    rect = unwrap_rect(obj.get("clickbox"))
                    plan["steps"].append({
                        "action": "click",
                        "description": "Open bank to finalize",
                        "click": ({"type": "rect-center"} if rect else
                                  {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}),
                        "target": {"domain": "object", "name": obj.get("name"), "id": obj.get("id"), "clickbox": rect},
                        "preconditions": ["bankOpen == false"], "postconditions": ["bankOpen == true"],
                        "confidence": 0.9 if rect else 0.6
                    })
                    return plan

            # Deposit-all anything left (rings, leftovers, etc.)
            dep_bounds = self._deposit_all_button_bounds(payload)
            if dep_bounds:
                plan["steps"].append({
                    "action": "click",
                    "description": "Deposit-all before ending",
                    "click": {"type": "rect-center"},
                    "target": {"domain": "widget", "name": "deposit_inventory", "bounds": dep_bounds},
                    "preconditions": ["bankOpen == true"], "postconditions": []
                })
                return plan

            # Close bank to finish
            plan["steps"].append({
                "action": "click",
                "description": "Close bank (done)",
                "click": {"type": "key", "key": "ESC"},
                "target": {"domain": "widget", "name": "bank_close"},
                "preconditions": ["bankOpen == true"], "postconditions": ["bankOpen == false"], "confidence": 0.95
            })
            return plan

        # Fallback
        plan["steps"].append({"action": "idle", "description": "No actionable step",
                              "click": {"type": "none"},
                              "target": {"domain": "none", "name": "n/a"},
                              "preconditions": [], "postconditions": [], "confidence": 0.0})
        return plan

    def _inv_has_any_rings(self, payload: dict) -> bool:
        return any(inv_count(payload, nm) > 0 for nm in ("Gold ring", "Sapphire ring", "Emerald ring"))

    def _inv_has_foreign_items(self, payload: dict) -> bool:
        allowed = {"Ring mould", "Gold bar", "Sapphire", "Emerald", "Gold ring", "Sapphire ring", "Emerald ring"}
        for s in (payload.get("inventory") or {}).get("slots") or []:
            nm = s.get("itemName") or ""
            qty = int(s.get("quantity") or 0)
            if qty > 0 and nm not in allowed:
                return True
        return False
