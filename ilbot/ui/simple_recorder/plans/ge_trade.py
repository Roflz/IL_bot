from .base import Plan
from ..helpers.bank import nearest_banker, bank_note_selected, bank_qty_all_selected, deposit_all_button_bounds
from ..helpers.ge import ge_open, ge_offer_open, ge_price_value, widget_by_id_text_contains, ge_widgets, \
    ge_inv_item_by_name, price, ge_selected_item_is, chat_qty_prompt_active, ge_qty_matches, ge_inv_slot_bounds, \
    widget_by_id_text, ge_first_buy_slot_btn, ge_qty_button, ge_buy_confirm_widget, nearest_clerk
from ..helpers.inventory import coins, inv_has_any
from ..helpers.rects import unwrap_rect
from ..helpers.utils import norm_name, closest_object_by_names, step_recent
from ..helpers.widgets import rect_center_from_widget


class GETradePlan(Plan):
    id = "GE_SELL_BUY"
    label = "GE: Sell Rings & Buy Mats"

    BUY_PAIR = ("Sapphire", "Gold bar")  # flip to ("Emerald", "Gold bar") if you want

    def __init__(self):
        self.state = {
            "phase": "ENSURE_GE_OPEN",
            "done": False,  # <- added: handy flag for callers/loop
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
                # added defensive defaults used in compute_phase:
                "want": [],
                "typed": False,
                "clicks": 0,
                "last_price": None,
            },
            "prep": {"mode": "OPEN_BANK"},  # 🚑 start prelude immediately
            # used for balancing purchases; compute_phase reads this:
            "bank_counts": {"gold": 0, "sapphire": 0, "emerald": 0},
        }

    # ----------------------------- PHASE ENGINE -----------------------------
    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        st = self.state

        # --- if we've finished, stay finished ---
        if st.get("phase") == "DONE":
            st["done"] = True
            return "DONE"

        # ensure subdicts exist (defensive; keeps your original structure)
        sell = st.setdefault("sell", {"queue": [], "queue_seeded": False, "pending_pick": None,
                                      "active": None, "clicks": 0, "last_price": None})
        buy = st.setdefault("buy", {"items": ["Sapphire", "Gold bar"], "idx": 0, "step": "OPEN_SLOT",
                                    "want": [], "typed": False, "clicks": 0, "last_price": None})
        prep = st.setdefault("prep", {"mode": "OPEN_BANK"})
        st.setdefault("bank_counts", {"gold": 0, "sapphire": 0, "emerald": 0})

        # --- hard facts each tick ---
        _ge_open = ge_open(payload)
        offer_open = ge_offer_open(payload)  # chooser absent -> True
        price_now = ge_price_value(payload)  # int or None
        bank_open = bool((payload.get("bank") or {}).get("bankOpen", False))
        can_confirm = bool(widget_by_id_text_contains(payload, 30474266, "confirm"))
        has_collect = bool(widget_by_id_text_contains(payload, 30474246, "collect"))
        _ge_widgets = (ge_widgets(payload) or {})

        # ---------- small helpers ----------
        def _rings_exist_in_bank() -> bool:
            for s_ in (payload.get("bank") or {}).get("slots") or []:
                nm = (s_.get("itemName") or "").lower()
                if nm in ("sapphire ring", "emerald ring", "sapphire rings", "emerald rings"):
                    if int(s_.get("quantity") or 0) > 0:
                        return True
            return False

        def _coins_exist_in_bank() -> bool:
            for s_ in (payload.get("bank") or {}).get("slots") or []:
                if (s_.get("itemName") or "").lower() == "coins" and int(s_.get("quantity") or 0) > 0:
                    return True
            return False

        def _selected_item_matches(name: str) -> bool:
            w = _ge_widgets.get("30474266:27") or {}
            t = norm_name(w.get("text") or w.get("textStripped"))
            return bool(t) and (t == norm_name(name))

        # --- PRELUDE: small, linear, one-click phases ---
        if prep.get("mode") != "DONE":
            bank_open = bool((payload.get("bank") or {}).get("bankOpen", False))

            if prep.get("mode") is None:
                prep["mode"] = "OPEN_BANK"

            if prep["mode"] == "OPEN_BANK":
                if bank_open:
                    prep["mode"] = "DEPOSIT_ALL" if inv_has_any(payload) else "SELECT_NOTE"
                st["phase"] = "PREP_OPEN_BANK"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "DEPOSIT_ALL":
                if not bank_open:
                    st["phase"] = "PREP_OPEN_BANK"
                    st["done"] = False
                    return st["phase"]
                if not inv_has_any(payload):
                    prep["mode"] = "SELECT_NOTE"
                st["phase"] = "PREP_DEPOSIT_ALL"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "SELECT_NOTE":
                if not bank_open:
                    st["phase"] = "PREP_OPEN_BANK"
                    st["done"] = False
                    return st["phase"]
                if bank_note_selected(payload):
                    prep["mode"] = "SELECT_ALL"
                st["phase"] = "PREP_SELECT_NOTE"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "SELECT_ALL":
                if not bank_open:
                    st["phase"] = "PREP_OPEN_BANK"
                    st["done"] = False
                    return st["phase"]
                if bank_qty_all_selected(payload):
                    prep["mode"] = "PUMP_RINGS"
                st["phase"] = "PREP_SELECT_ALL"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "PUMP_RINGS":
                if not bank_open:
                    st["phase"] = "PREP_OPEN_BANK"
                    st["done"] = False
                    return st["phase"]
                if not _rings_exist_in_bank():
                    prep["mode"] = "PUMP_COINS"
                st["phase"] = "PREP_PUMP_RINGS"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "PUMP_COINS":
                if not bank_open:
                    st["phase"] = "PREP_OPEN_BANK"
                    st["done"] = False
                    return st["phase"]
                if not _coins_exist_in_bank():
                    prep["mode"] = "CLOSE_BANK"
                st["phase"] = "PREP_PUMP_COINS"
                st["done"] = False
                return st["phase"]

            if prep["mode"] == "CLOSE_BANK":
                if not bank_open:
                    prep["mode"] = "DONE"
                st["phase"] = "PREP_CLOSE_BANK"
                st["done"] = False
                return st["phase"]

        # --- 1) GE closed handling:
        if not _ge_open:
            if st.get("phase") == "BUY_CLOSE_GE":
                st.update({"phase": "DONE", "done": True})
                return "DONE"
            st["phase"] = "ENSURE_GE_OPEN"
            sell.update({"pending_pick": None, "active": None, "clicks": 0, "last_price": None})
            buy.update({"typed": False, "clicks": 0, "last_price": None})
            st["done"] = False
            return st["phase"]

        # --- 2) Seed SELL queue exactly once from current GE inventory ---
        if not sell.get("queue_seeded"):
            q = []
            if ge_inv_item_by_name(payload, "Sapphire ring"): q.append("Sapphire ring")
            if ge_inv_item_by_name(payload, "Emerald ring"):  q.append("Emerald ring")
            sell["queue"] = q
            sell["queue_seeded"] = True

        # --- 3) SELL finished → pop & reset ---
        if (not offer_open) and sell.get("active") and sell.get("clicks", 0) >= 3:
            if sell.get("queue") and sell["queue"][0] == sell["active"]:
                sell["queue"].pop(0)
            sell.update({"pending_pick": None, "active": None, "clicks": 0, "last_price": None})
            if has_collect:
                st["phase"] = "SELL_COLLECT"
                st["done"] = False
                return st["phase"]

        # --- 4) SELL_COLLECT gate ---
        if st["phase"] == "SELL_COLLECT":
            if not has_collect:
                st["phase"] = "SELL_PICK" if sell["queue"] else "BUY_OPEN_SLOT"
            st["done"] = False
            return st["phase"]

        # ===== SELL path =====
        if sell.get("queue"):
            head = sell["queue"][0]

            if not offer_open:
                if sell.get("pending_pick") != head:
                    sell["pending_pick"] = head
                st["phase"] = "SELL_PICK"
                st["done"] = False
                return st["phase"]

            if sell.get("active") is None:
                sell["active"] = sell.get("pending_pick") or head

            if price_now is not None and sell.get("last_price") is None:
                sell["last_price"] = price_now
            if (price_now is not None) and (sell.get("last_price") is not None) and (price_now < sell["last_price"]):
                sell["clicks"] += 1
                sell["last_price"] = price_now

            if sell.get("clicks", 0) < 3:
                st["phase"] = "SELL_MINUS"
                st["done"] = False
                return st["phase"]

            st["phase"] = "SELL_CONFIRM" if can_confirm else "SELL_WAIT_CLOSED"
            st["done"] = False
            return st["phase"]

        # ===== BUY path =====
        if not buy.get("want"):
            _coins = max(0, int(coins(payload)))
            budget = max(0, int(_coins * 0.80))

            pG = max(1, price(payload, "Gold bar") or 1)
            pS = max(1, price(payload, "Sapphire") or 1)
            pE = max(1, price(payload, "Emerald") or 1)

            bc = st.get("bank_counts") or {}
            nG = int(bc.get("gold", 0))
            nS = int(bc.get("sapphire", 0))
            nE = int(bc.get("emerald", 0))
            gems_now = nS + nE
            gold_now = nG

            want_gold = want_sap = want_em = 0
            diff = gems_now - gold_now
            spend = 0

            if diff > 0:
                can_buy = min(diff, budget // pG)
                want_gold += can_buy
                spend += can_buy * pG
            elif diff < 0:
                need_gems = -diff
                s_need = (need_gems + 1) // 2
                e_need = need_gems - s_need
                s_buy = min(s_need, (budget // pS) if pS else 0)
                spend_s = s_buy * pS
                rem = budget - spend_s
                e_buy = min(e_need, (rem // pE) if pE else 0)
                spend_e = e_buy * pE
                want_sap += s_buy
                want_em += e_buy
                spend += spend_s + spend_e

            rem = max(0, budget - spend)
            if rem >= min(pG + pS, pG + pE):
                cost_pair_s = pG + pS
                cost_pair_e = pG + pE
                pair_order = [("Sapphire", cost_pair_s), ("Emerald", cost_pair_e)]
                pair_order.sort(key=lambda t: t[1], reverse=True)
                for gem_name, cpair in pair_order:
                    if rem < cpair:
                        continue
                    k = rem // cpair
                    if k <= 0:
                        continue
                    want_gold += k
                    if gem_name == "Sapphire":
                        want_sap += k
                    else:
                        want_em += k
                    rem -= k * cpair

            want_list = []
            if want_gold > 0: want_list.append({"name": "Gold bar", "qty": int(want_gold)})
            if want_sap > 0: want_list.append({"name": "Sapphire", "qty": int(want_sap)})
            if want_em > 0: want_list.append({"name": "Emerald", "qty": int(want_em)})

            if not want_list:
                itemA, itemB = self.BUY_PAIR
                pA = max(1, price(payload, itemA) or 1)
                pB = max(1, price(payload, itemB) or 1)
                qtyA = max(1, (budget // 2) // pA) if budget > 0 else 1
                qtyB = max(1, (budget // 2) // pB) if budget > 0 else 1
                want_list = [{"name": itemA, "qty": qtyA}, {"name": itemB, "qty": qtyB}]

            buy["want"] = want_list
            buy["idx"] = 0
            buy.update({"typed": False, "clicks": 0, "last_price": None})

        if buy["idx"] >= len(buy["want"]):
            st["phase"] = "BUY_COLLECT" if has_collect else "BUY_CLOSE_GE"
            st["done"] = False
            return st["phase"]

        cur = buy["want"][buy["idx"]]

        if st["phase"] not in (
                "BUY_WAIT_OFFER", "BUY_TYPE_NAME", "BUY_VERIFY_ITEM",
                "BUY_OPEN_QTY_BTN", "BUY_TYPE_QTY",
                "BUY_PLUS", "BUY_CONFIRM", "BUY_WAIT_CLOSED", "BUY_COLLECT", "BUY_CLOSE_GE", "BUY_OPEN_SLOT"):
            st["phase"] = "BUY_OPEN_SLOT"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_OPEN_SLOT":
            if offer_open and _ge_widgets.get("30474266"):
                buy.update({"typed": False, "clicks": 0, "last_price": None})
                st["phase"] = "BUY_TYPE_NAME"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_TYPE_NAME":
            items = buy.get("want") or buy.get("items") or []
            idx = int(buy.get("idx") or 0)
            if 0 <= idx < len(items):
                cur = items[idx]
                item_name = (cur.get("name") if isinstance(cur, dict) else str(cur)).strip()
                if ge_selected_item_is(payload, item_name):
                    st["phase"] = "BUY_OPEN_QTY_BTN"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_VERIFY_ITEM":
            if _selected_item_matches(cur["name"]):
                st["phase"] = "BUY_OPEN_QTY_BTN"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_OPEN_QTY_BTN":
            if chat_qty_prompt_active(payload):
                st["phase"] = "BUY_TYPE_QTY"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_TYPE_QTY":
            q = int(cur.get("qty") or 1)
            if ge_qty_matches(payload, q):
                st["phase"] = "BUY_PLUS"
            else:
                st["phase"] = "BUY_TYPE_QTY"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_PLUS":
            if (price_now is not None) and (buy.get("last_price") is None):
                buy["last_price"] = price_now
            if (price_now is not None) and (buy.get("last_price") is not None) and (price_now > buy["last_price"]):
                buy["clicks"] = int(buy.get("clicks") or 0) + 1
                buy["last_price"] = price_now
            if int(buy.get("clicks") or 0) >= 3:
                st["phase"] = "BUY_CONFIRM" if can_confirm else "BUY_PLUS"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_CONFIRM":
            if not offer_open:
                st["phase"] = "BUY_WAIT_CLOSED"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_WAIT_CLOSED":
            if not offer_open:
                buy["idx"] += 1
                buy.update({"typed": False, "clicks": 0, "last_price": None})
                if buy["idx"] >= len(buy["want"]):
                    st["phase"] = "BUY_COLLECT" if has_collect else "BUY_CLOSE_GE"
                else:
                    st["phase"] = "BUY_OPEN_SLOT"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_COLLECT":
            if not has_collect:
                st["phase"] = "BUY_CLOSE_GE"
            st["done"] = False
            return st["phase"]

        if st["phase"] == "BUY_CLOSE_GE":
            if not _ge_open:
                st.update({"phase": "DONE", "done": True})
                return "DONE"
            st["done"] = False
            return st["phase"]

        # fallback
        st["done"] = False
        return st["phase"]

    # ----------------------------- ACTION BUILDER ---------------------------
    # IMPORTANT: this NEVER mutates self.state["phase"].
    # ----------------------------- ACTION BUILDER ---------------------------
    # IMPORTANT: this NEVER mutates self.state["phase"] and only produces steps.
    def build_action_plan(self, payload: dict, phase: str) -> dict:
        st = self.state
        sell = st["sell"]
        plan = {"phase": phase, "steps": []}

        if phase == "DONE":
            return {"phase": phase, "steps": []}

        if phase == "PREP_OPEN_BANK":
            # Close GE first if open
            if ge_open(payload):
                plan["steps"].append({
                    "id": "prelude-close-ge",
                    "action": "click",
                    "description": "Close GE (Esc) before opening bank",
                    "click": {"type": "key", "key": "esc"},
                    "preconditions": [], "postconditions": []
                })
                return plan

            if not (payload.get("bank") or {}).get("bankOpen", False):
                # 1) Banker
                banker = nearest_banker(payload)
                if banker and isinstance(banker.get("canvasX"), (int, float)) and isinstance(banker.get("canvasY"),
                                                                                             (int, float)):
                    plan["steps"].append({
                        "id": "ge-bank-open-npc",
                        "action": "click",
                        "description": "Open bank (click Banker)",
                        "click": {"type": "point", "x": int(banker["canvasX"]), "y": int(banker["canvasY"]) - 8},
                        "target": {"domain": "npc", "name": banker.get("name"), "id": banker.get("id")},
                        "preconditions": ["bankOpen == false"], "postconditions": ["bankOpen == true"],
                        "confidence": 0.95
                    })
                    return plan

                # 2) Fallback: booth/banker object (old path)
                booth = closest_object_by_names(payload, ["bank booth", "grand exchange booth", "banker"])
                if booth and isinstance(booth.get("canvasX"), (int, float)) and isinstance(booth.get("canvasY"),
                                                                                           (int, float)):
                    plan["steps"].append({
                        "id": "ge-bank-open",
                        "action": "click",
                        "description": "Open bank (object fallback)",
                        "click": {"type": "point", "x": int(booth["canvasX"]), "y": int(booth["canvasY"]) - 16},
                        "target": {"domain": "object", "name": booth.get("name"), "id": booth.get("id")},
                        "preconditions": ["bankOpen == false"], "postconditions": ["bankOpen == true"],
                        "confidence": 0.9
                    })
            return plan

        if phase == "PREP_DEPOSIT_ALL":
            dep_bounds = deposit_all_button_bounds(payload)
            if dep_bounds:
                plan["steps"].append({
                    "id": "prelude-deposit-all",
                    "action": "click",
                    "description": "Deposit inventory to start clean",
                    "click": {"type": "rect-center"},
                    "target": {"domain": "widget", "name": "deposit_inventory", "bounds": dep_bounds},
                    "preconditions": ["bankOpen == true"], "postconditions": []
                })
            return plan

        if phase == "PREP_SELECT_NOTE":
            bw = payload.get("bank_widgets") or {}
            node = (bw.get("withdraw_note_toggle") or {})
            b = node.get("bounds") or {}
            if int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0 and not bank_note_selected(payload):
                plan["steps"].append({
                    "id": "bank-note-toggle",
                    "action": "click",
                    "description": "Enable Withdraw as Note",
                    "target": {"name": "Withdraw as Note", "bounds": b},
                    "click": {"type": "rect-center"},
                    "preconditions": ["bankOpen == true"], "postconditions": []
                })
            return plan

        # PREP_SELECT_ALL — click the "All" quantity button if it's not selected.
        if phase == "PREP_SELECT_ALL":
            bw = (payload.get("bank_widgets") or {})
            qall = (bw.get("withdraw_quantity_all") or {})  # {"bounds": {...}, "selected": bool}
            b = qall.get("bounds") or {}
            if not bool(qall.get("selected")) and int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0:
                plan["steps"].append({
                    "id": "bank-qty-all",
                    "action": "click",
                    "description": "Set Withdraw Quantity to All",
                    "target": {"name": "Quantity All", "bounds": b},
                    "click": {"type": "rect-center"},
                    "preconditions": ["bankOpen == true"],
                    "postconditions": []
                })
            return plan

        if phase == "PREP_PUMP_RINGS":
            bank_slots = (payload.get("bank") or {}).get("slots") or []
            want = {"sapphire ring", "emerald ring", "sapphire rings", "emerald rings"}
            for s in bank_slots:
                nm = (s.get("itemName") or "").lower()
                if nm in want and int(s.get("quantity") or 0) > 0:
                    r = ((s.get("bounds") or {}) or {}).get("bounds")
                    if r and r.get("width"):
                        plan["steps"].append({
                            "id": f"bank-withdraw-{s.get('slotId')}",
                            "action": "click", "description": f"Withdraw all: {s.get('itemName')}",
                            "target": {"name": s.get("itemName"), "bounds": r},
                            "click": {"type": "rect-center"}, "preconditions": ["bankOpen == true"],
                            "postconditions": []
                        })
                        return plan
            return plan

        if phase == "PREP_PUMP_COINS":
            bank_slots = (payload.get("bank") or {}).get("slots") or []
            for s in bank_slots:
                if (s.get("itemName") or "").lower() == "coins" and int(s.get("quantity") or 0) > 0:
                    r = ((s.get("bounds") or {}) or {}).get("bounds")
                    if r and r.get("width"):
                        plan["steps"].append({
                            "id": f"bank-withdraw-coins-{s.get('slotId')}",
                            "action": "click", "description": "Withdraw all: Coins",
                            "target": {"name": "Coins", "bounds": r},
                            "click": {"type": "rect-center"}, "preconditions": ["bankOpen == true"],
                            "postconditions": []
                        })
                        return plan
            return plan

        if phase == "PREP_CLOSE_BANK":
            plan["steps"].append({
                "id": "bank-close-esc", "action": "click",
                "description": "Close Bank (Esc)",
                "click": {"type": "key", "key": "esc"},
                "preconditions": ["bankOpen == true"], "postconditions": ["bankOpen == false"]
            })
            return plan

        # === GE open/entry ===
        if phase == "ENSURE_GE_OPEN":
            if not ge_open(payload):
                # 1) Grand Exchange Clerk (best)
                clerk = nearest_clerk(payload)
                if clerk and isinstance(clerk.get("canvasX"), (int, float)) and isinstance(clerk.get("canvasY"),
                                                                                           (int, float)):
                    plan["steps"].append({
                        "id": "ge-exchange-open",
                        "action": "click",
                        "description": "Open GE (Grand Exchange Clerk)",
                        "click": {"type": "point", "x": int(clerk["canvasX"]), "y": int(clerk["canvasY"]) - 8},
                        "target": {"domain": "npc", "name": clerk.get("name"), "id": clerk.get("id")},
                        "preconditions": [], "postconditions": [],
                        "confidence": 0.95
                    })
                    return plan

                # 2) Fallback: booth object
                booth = closest_object_by_names(payload, ["grand exchange booth"])
                if booth and isinstance(booth.get("canvasX"), (int, float)) and isinstance(booth.get("canvasY"),
                                                                                           (int, float)):
                    plan["steps"].append({
                        "id": "ge-exchange-open-booth",
                        "action": "click",
                        "description": "Open GE (booth fallback)",
                        "click": {"type": "point", "x": int(booth["canvasX"]), "y": int(booth["canvasY"]) - 12},
                        "target": {"domain": "object", "name": booth.get("name"), "id": booth.get("id")},
                        "preconditions": [], "postconditions": [],
                        "confidence": 0.9
                    })
                    return plan
            return plan

        # --------------------- SELL path actions --------------------------
        if phase == "SELL_PICK":
            nm = st["sell"]["pending_pick"]
            if nm:
                rect = ge_inv_slot_bounds(payload, nm)
                if rect:
                    plan["steps"].append({
                        "id": "ge-offer-open",
                        "action": "click",
                        "description": f"Offer {nm}",
                        "target": {"name": "GE inv item", "bounds": rect},
                        "click": {"type": "rect-center"},
                        "preconditions": [], "postconditions": [],
                        "confidence": 0.95,
                    })
            return plan

        if phase == "SELL_MINUS":
            minus = widget_by_id_text(payload, 30474266, "-5%")
            if minus:
                cx, cy = rect_center_from_widget(minus)
                plan["steps"].append({
                    "id": "ge-minus5",
                    "action": "click",
                    "description": f"GE price -5% ({sell['clicks'] + 1}/3)",
                    "target": {"name": "-5%", "bounds": minus.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": [], "postconditions": [],
                    "confidence": 0.95,
                })
            return plan

        if phase == "SELL_CONFIRM":
            confirm = widget_by_id_text_contains(payload, 30474266, "confirm")
            if confirm:
                cx, cy = rect_center_from_widget(confirm)
                plan["steps"].append({
                    "id": "ge-confirm-offer",
                    "action": "click",
                    "description": f"Confirm offer for {st['sell']['active']}",
                    "target": {"name": "Confirm", "bounds": confirm.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": [], "postconditions": [],
                    "confidence": 0.95,
                })
            return plan

        if phase == "SELL_WAIT_CLOSED":
            return plan  # pure wait (compute_phase handles state)

        if phase == "SELL_COLLECT":
            collect = widget_by_id_text_contains(payload, 30474246, "collect")
            if collect:
                cx, cy = rect_center_from_widget(collect)
                plan["steps"].append({
                    "id": "ge-collect",
                    "action": "click",
                    "description": "Collect proceeds",
                    "target": {"name": "Collect", "bounds": collect.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": [], "postconditions": [],
                    "confidence": 0.95,
                })
            return plan

        # --------------------- BUY path actions --------------------------
        if phase == "BUY_OPEN_SLOT":
            btn = ge_first_buy_slot_btn(payload)
            if btn:
                x, y = rect_center_from_widget(btn)
                plan["steps"].append({
                    "id": "ge-buy-open-slot",
                    "action": "click",
                    "description": "Open buy slot",
                    "target": {"name": "GE slot button", "bounds": btn.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                    "preconditions": [], "postconditions": []
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
            per_ms = 50
            window_ms = 60000
            type_id = f"ge-type-name-{idx}-{item_name}"
            wait_id = f"ge-type-wait-{idx}"
            enter_id = f"ge-type-enter-{idx}"

            if not step_recent(type_id, window_ms):
                plan["steps"].append({
                    "id": type_id,
                    "action": "click",
                    "click": {
                        "type": "type", "text": item_name,
                        "enter": False, "per_char_ms": per_ms, "focus": True
                    },
                    "description": f"type '{item_name}'"
                })
                return plan

            if not step_recent(wait_id, window_ms):
                plan["steps"].append({
                    "id": wait_id,
                    "action": "click",
                    "click": {"type": "wait", "ms": max(300, per_ms * len(item_name) + 50)},
                    "description": "Pause before pressing Enter"
                })
                return plan

            if not step_recent(enter_id, window_ms):
                plan["steps"].append({
                    "id": enter_id,
                    "action": "click",
                    "click": {"type": "key", "key": "enter"},
                    "description": "Confirm search"
                })
                return plan

            return plan

        if phase == "BUY_VERIFY_ITEM":
            return plan  # pure wait

        if phase == "BUY_OPEN_QTY_BTN":
            btn = ge_qty_button(payload)  # must resolve to 30474266:51 ("…")
            if btn:
                x, y = rect_center_from_widget(btn)
                plan["steps"].append({
                    "id": "ge-open-qty",
                    "action": "click",
                    "description": "Open quantity dialog (…) ",
                    "target": {"name": "Qty …", "bounds": btn.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                    "preconditions": [], "postconditions": []
                })
            return plan

        if phase == "BUY_TYPE_QTY":
            if chat_qty_prompt_active(payload):
                q = int((st["buy"]["want"][st["buy"]["idx"]] or {}).get("qty") or 1)
                type_id = f"ge-type-qty-{st['buy']['idx']}-{q}"
                if not step_recent(type_id, 60000):
                    plan["steps"].append({
                        "id": type_id,
                        "action": "click",
                        "description": f"Type buy quantity {q} and Enter",
                        "click": {"type": "type", "text": str(q), "enter": True, "per_char_ms": 40, "focus": True}
                    })
            return plan

        if phase == "BUY_PLUS":
            plus = widget_by_id_text(payload, 30474266, "+5%")
            if plus:
                cx, cy = rect_center_from_widget(plus)
                plan["steps"].append({
                    "id": "ge-plus5",
                    "action": "click",
                    "description": f"GE price +5% ({int(st['buy'].get('clicks') or 0) + 1}/3)",
                    "target": {"name": "+5%", "bounds": plus.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                    "preconditions": [], "postconditions": [],
                    "confidence": 0.95,
                })
            return plan

        if phase == "BUY_CONFIRM":
            conf = ge_buy_confirm_widget(payload)
            if conf:
                x, y = rect_center_from_widget(conf)
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
            collect = widget_by_id_text_contains(payload, 30474246, "collect")
            if collect:
                x, y = rect_center_from_widget(collect)
                plan["steps"].append({
                    "id": "ge-collect", "action": "click",
                    "description": "Collect proceeds",
                    "target": {"name": "Collect", "bounds": collect.get("bounds")},
                    "click": {"type": "point", "x": x, "y": y},
                })
            return plan

        if phase == "BUY_CLOSE_GE":
            plan["steps"].append({
                "id": "ge-close", "action": "click", "description": "Close GE",
                "click": {"type": "key", "key": "escape"}
            })
            return plan

        # fallback
        return plan
