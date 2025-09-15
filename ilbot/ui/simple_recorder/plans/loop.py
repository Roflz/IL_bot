from .base import Plan
from ilbot.ui.simple_recorder.actions import *

class LoopPlan(Plan):
    """
    Craft rings -> go to GE -> buy/sell -> go to Edgeville bank -> repeat.
    Delegates to existing plans; this class is just a top-level FSM coordinator.
    """
    id = "CRAFT_GE_BANK_LOOP"
    label = "Loop: Craft ⇄ GE ⇄ Bank"

    def __init__(self):
        # persistent minimal state across ticks
        self.phase = "TO_EDGE_BANK"  # starting leg; change if you want to start elsewhere
        self._sub_cache = {
            "go":  GoToRectPlan(),   # reuse implementation
            "craft": RingCraftPlan(),
            "ge": GETradePlan()
        }

    def _edge_rect(self):
        # rect = (minX, maxX, minY, maxY)
        from .constants import EDGE_BANK_MIN_X, EDGE_BANK_MAX_X, EDGE_BANK_MIN_Y, EDGE_BANK_MAX_Y
        return (EDGE_BANK_MIN_X, EDGE_BANK_MAX_X, EDGE_BANK_MIN_Y, EDGE_BANK_MAX_Y)

    def _ge_rect(self):
        from .constants import GE_MIN_X, GE_MAX_X, GE_MIN_Y, GE_MAX_Y
        return (GE_MIN_X, GE_MAX_X, GE_MIN_Y, GE_MAX_Y)

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        # We keep our own phase and only advance when a delegated plan reports "done/arrived"
        me = (payload.get("player") or {})
        ge_open = bool((payload.get("grand_exchange") or {}).get("open", False))
        bank_open = bool((payload.get("bank") or {}).get("bankOpen", False))

        # For movement legs, use GoToRectPlan’s phase
        if self.phase == "TO_EDGE_BANK":
            sub_payload = dict(payload)
            sub_payload["navTarget"] = {"name": "Edgeville Bank", "rect": self._edge_rect()}
            sub_phase = self._sub_cache["go"].compute_phase(sub_payload, craft_recent)
            if sub_phase == "Arrived":
                self.phase = "CRAFT"
            return self.phase

        if self.phase == "CRAFT":
            # When crafting plan signals done (no more craftable), move to GE
            # We’ll lean on its own compute_phase and treat “DONE” as done.
            craft_phase = self._sub_cache["craft"].compute_phase(payload, craft_recent)
            if craft_phase == "DONE":
                self.phase = "TO_GE"
            return self.phase

        if self.phase == "TO_GE":
            sub_payload = dict(payload)
            sub_payload["navTarget"] = {"name": "Grand Exchange", "rect": self._ge_rect()}
            sub_phase = self._sub_cache["go"].compute_phase(sub_payload, craft_recent)
            if sub_phase == "Arrived":
                self.phase = "GE_TRADE"
            return self.phase

        if self.phase == "GE_TRADE":
            ge_phase = self._sub_cache["ge"].compute_phase(payload, craft_recent)
            if ge_phase == "DONE":
                # After buy/sell, head back to bank for the next craft loop
                self.phase = "TO_EDGE_BANK"
            return self.phase

        # Fallback safety
        return self.phase

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        # Phase computed above; now build steps by delegating to the right child plan
        if self.phase == "TO_EDGE_BANK":
            sub_payload = dict(payload)
            sub_payload["navTarget"] = {"name": "Edgeville Bank", "rect": self._edge_rect()}
            sub_phase = self._sub_cache["go"].compute_phase(sub_payload, False)
            return self._sub_cache["go"].build_action_plan(sub_payload, sub_phase)

        if self.phase == "CRAFT":
            craft_phase = self._sub_cache["craft"].compute_phase(payload, False)
            return self._sub_cache["craft"].build_action_plan(payload, craft_phase)

        if self.phase == "TO_GE":
            sub_payload = dict(payload)
            sub_payload["navTarget"] = {"name": "Grand Exchange", "rect": self._ge_rect()}
            sub_phase = self._sub_cache["go"].compute_phase(sub_payload, False)
            return self._sub_cache["go"].build_action_plan(sub_payload, sub_phase)

        if self.phase == "GE_TRADE":
            ge_phase = self._sub_cache["ge"].compute_phase(payload, False)
            return self._sub_cache["ge"].build_action_plan(payload, ge_phase)

        return {"phase": self.phase, "steps": []}
