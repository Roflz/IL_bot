#!/usr/bin/env python3
"""
Kill Goblins Plan
=================

Super simple combat loop:
- BANK: go to Varrock West bank, equip a scimitar
- GO_TO_GOBLINS: walk to Goblin Village
- FIGHT: attack goblins
"""

import logging
import time
from pathlib import Path
import sys

from actions import bank, inventory, player, combat, combat_interface, tab
from actions import wait_until
from actions.chat import can_continue
from actions.travel import go_to, in_area
from helpers.keyboard import press_spacebar

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.tab import is_tab_open
from helpers.utils import sleep_exponential, exponential_number, random_number


class KillGoblinsPlan(Plan):
    id = "KILL_GOBLINS"
    label = "Kill Goblins"

    def __init__(self):
        self.state = {"phase": "FIGHT", "next_tab_switch_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # ---- Config ----
        self.bank_area = None
        self.goblin_area = "LUMBRIDGE_GOBLINS"
        self.goblin_names = ["Goblin"]

        # Attack level -> best scimitar we can equip (then best we have in bank/inv wins).
        # Requirements (OSRS):
        # - Bronze/Iron: 1
        # - Steel: 5
        # - Black: 10
        # - Mithril: 20
        # - Adamant: 30
        # - Rune: 40
        # - Dragon: 60 (members)
        self.scimitar_by_req = [
            ("Dragon scimitar", 60),
            ("Rune scimitar", 40),
            ("Adamant scimitar", 30),
            ("Mithril scimitar", 20),
            ("Black scimitar", 10),
            ("Steel scimitar", 5),
            ("Iron scimitar", 1),
            ("Bronze scimitar", 1),
        ]

        atk = player.get_skill_level("attack") or 1
        self.scimitar_names = [name for (name, req) in self.scimitar_by_req if int(req) <= int(atk)]
        if not self.scimitar_names:
            self.scimitar_names = ["Bronze scimitar"]
        logging.info(f"[{self.id}] Attack level {atk}; eligible scimitars (best-first): {self.scimitar_names}")

        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[
                # Food for emergency healing in combat
                {"name": "Trout", "quantity": 15},
            ],
            deposit_all=True,
            equip_items={"weapon": self.scimitar_names,
                         "helmet": ["Blue wizard hat"],
                         "body": ["Blue wizard robe"],
                         "legs": ["Blue skirt"],
                         "shield": ["Chronicle"],
                         # "cape": ["cape"],
                         },
        )

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        logging.info(f"[{self.id}] Plan initialized")

    def _maybe_tab_switch(self) -> None:
        """
        Occasionally toggle between Inventory and Skills tabs.
        Interval is sampled using exponential_number between ~1 and ~30 minutes.
        """
        if bank.is_open():
            return

        now = time.time()
        nxt = self.state.get("next_tab_switch_ts")
        if not isinstance(nxt, (int, float)):
            self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))
            return

        if now < float(nxt):
            return

        # Toggle behavior:
        # - If Inventory is open -> open Skills
        # - If Skills is open -> open Inventory
        # - Else -> open Inventory
        if is_tab_open("INVENTORY"):
            tab.open_tab("SKILLS")
        elif is_tab_open("SKILLS"):
            tab.open_tab("INVENTORY")
        else:
            tab.open_tab("INVENTORY")

        self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "BANK")

        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        self._maybe_tab_switch()

        match phase:
            case "BANK":
                return self._handle_bank(ui)
            case "GO_TO_GOBLINS":
                return self._handle_go_to_goblins()
            case "FIGHT":
                return self._handle_fight()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_bank(self, ui) -> int:
        status = self.bank_plan.loop(ui)
        if status == BankPlanSimple.SUCCESS:
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            self.bank_plan.reset()
            self.set_phase("GO_TO_GOBLINS")
            return self.loop_interval_ms

        if status == BankPlanSimple.MISSING_ITEMS:
            missing = getattr(self.bank_plan, "missing_items", None)
            logging.error(f"[{self.id}] Missing items: {missing}")
            return self.loop_interval_ms

        if status == BankPlanSimple.ERROR:
            logging.error(f"[{self.id}] Banking error: {self.bank_plan.get_error_message()}")
            return self.loop_interval_ms

        return status

    def _handle_go_to_goblins(self) -> int:
        if not in_area(self.goblin_area):
            go_to(self.goblin_area)
            return exponential_number(0.8, 4.0, 0.8)
        self.set_phase("FIGHT")
        return self.loop_interval_ms

    def _handle_fight(self) -> int:
        # Emergency eat: if HP is critically low, try to eat food immediately (even in combat).
        try:
            hp = player.get_health()
        except Exception:
            hp = None

        if isinstance(hp, int) and hp <= 3:
            if inventory.has_item("Trout"):
                logging.info(f"[{self.id}] Low HP ({hp}) -> eating Trout")
                inventory.interact("Trout", "Eat")
                return exponential_number(1.2, 2.0, 1.3)

            # No food available: go restock.
            logging.warning(f"[{self.id}] Low HP ({hp}) but no Trout found; returning to bank")
            self.set_phase("BANK")
            return self.loop_interval_ms

        # If already in combat, just wait a bit.
        if player.is_in_combat():
            if can_continue():
                press_spacebar()
                return exponential_number(0.3, 1.5, 1.5)
            return exponential_number(0.3, 6.0, 1.5)

        # Only "remember" to correct combat style sometimes (human-like imperfection).
        # About 30% of loops we check the lowest of Attack/Strength/Defence and switch accordingly.
        if float(random_number(0.0, 1.0, output_type="float")) < 0.30:
            atk = player.get_skill_level("attack") or 1
            str_ = player.get_skill_level("strength") or 1
            def_ = player.get_skill_level("defence") or 1

            # If there's a tie for lowest, prioritize: Strength -> Attack -> Defence
            candidates = [("strength", int(str_)), ("attack", int(atk)), ("defence", int(def_))]
            lowest_val = min(v for (_k, v) in candidates)
            lowest_skill = next(k for (k, v) in candidates if v == lowest_val)

            desired_style = {"attack": 0, "strength": 1, "defence": 3}[lowest_skill]
            cur = combat_interface.current_combat_style()
            if cur != desired_style:
                combat_interface.select_combat_style(desired_style)

        # Try to attack a goblin.
        res = combat.attack_closest(self.goblin_names)
        if res:
            # Give it a moment to register combat
            wait_until(lambda: player.is_in_combat(), max_wait_ms=2500)
        else:
            # No goblins found / reachable; idle a bit and retry.
            return exponential_number(0.5, 1.4, 1.2)
        return self.loop_interval_ms


