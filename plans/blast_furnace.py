#!/usr/bin/env python3
"""
Blast Furnace Plan (simple)
==========================

Very simple loop (iron + coal bag):

- BANK:
  - Open bank chest
  - Ensure only ice gloves are equipped (warn if not)
  - Ensure inventory contains an open coal bag (and nothing else)
  - Withdraw-all Coal
  - Click the coal bag to Fill
  - Withdraw-all Iron ore (fills remaining inventory)
  - Close bank

- SMELT:
  - Click Blast furnace to deposit Iron ore
  - Click coal bag to Empty
  - Click Blast furnace to deposit Coal
  - Click Moulding tray to collect bars
  - Press Space to confirm collection (chatbox continue)
  - Click Bank chest to run back and open it

- BANK_DEPOSIT:
  - Deposit bars only (keep the coal bag in inventory)
  - Repeat

NOTE:
Object names/actions can vary. If your scene uses different names (e.g. Conveyor belt / Bar dispenser),
update the config lists below.
"""

import logging
from pathlib import Path
import sys
import time
import random

from actions import bank, inventory, objects, player, wait_until, equipment, chat
from actions import npc
from actions.chat import can_continue
from helpers.keyboard import press_spacebar
from helpers.utils import exponential_number, sleep_exponential
from helpers.widgets import widget_exists
from helpers.inventory import has_only_items
from helpers.inventory import inv_count
from helpers.vars import get_var

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan


class BlastFurnacePlan(Plan):
    id = "BLAST_FURNACE"
    label = "Smithing: Blast Furnace"

    # RuneLite Var Inspector: VarbitID.BLAST_FURNACE_COFFER = 5357
    BLAST_FURNACE_COFFER_VARBIT = 5357
    # RuneLite Var Inspector: VarbitID.STAMINA_ACTIVE = 25
    STAMINA_ACTIVE_VARBIT = 25
    # RuneLite Var Inspector: VarbitID.STAMINA_DURATION = 24
    STAMINA_DURATION_VARBIT = 24

    def __init__(self):
        self.state = {"phase": "INITIAL_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase

    def _set_done(self, reason: str) -> int:
        self.state["phase"] = "DONE"
        self.state["done_reason"] = str(reason or "unknown")
        logging.warning(f"[{self.id}] DONE: {self.state['done_reason']}")
        return exponential_number(1500, 3500, 1.2)

    def _pay_foreman_if_needed(self) -> int | None:
        """
        Foreman payment guard.
        The fee lasts ~10 minutes. We re-pay when there is <1 minute remaining.
        Skips payment if smithing level is 60+ (free access).

        Returns:
          - int delay to return from the current phase if we took an action
          - None if no action needed (caller should continue)
        """
        # Skip payment if smithing level is 60+
        smith_lvl = player.get_skill_level("smithing")
        if (smith_lvl is not None) and int(smith_lvl) >= 60:
            return None
        
        # Timer: start when we pay. Re-pay when remaining < 60s.
        interval_s = 10 * 60
        repay_when_remaining_le_s = 60
        ts = self.state.get("foreman_paid_ts")
        if ts is not None:
            try:
                elapsed = float(time.monotonic() - float(ts))
            except Exception:
                elapsed = interval_s
        else:
            elapsed = interval_s

        should_pay = (ts is None) or (elapsed >= float(interval_s - repay_when_remaining_le_s))
        if not should_pay:
            return None

        if not inventory.has_item("Coins", min_qty=2500):
            if not inventory.is_empty():
                bank.deposit_inventory()
                wait_until(lambda: inventory.is_empty(), max_wait_ms=1000)
            # If we can't afford the foreman fee, stop the plan.
            try:
                if int(bank.get_item_count("Coins") or 0) < 2500:
                    return self._set_done("Not enough coins to pay foreman (need 2,500).")
            except Exception:
                return self._set_done("Not enough coins to pay foreman (need 2,500).")
            bank.withdraw_item("Coins", withdraw_all=True)
            wait_until(lambda: inventory.has_item("Coins", min_qty=2500), max_wait_ms=1200)
            if not inventory.has_item("Coins", min_qty=2500):
                return self._set_done("Failed to withdraw enough coins to pay foreman (need 2,500).")
            return exponential_number(250, 800, 1.2)
        bank.close_bank()
        wait_until(bank.is_closed, min_wait_ms=exponential_number(350, 1200, 1), max_wait_ms=3000)
        if not npc.click_npc_action_simple_prefer_no_camera("Blast Furnace Foreman", "Pay"):
            return exponential_number(450, 1200, 1.2)
        wait_until(lambda: chat.can_choose_option())
        sleep_exponential(0.35, 1.2, 1)
        coins_before = inv_count("Coins")
        chat.choose_option(1)
        # Verify we actually paid (should consume 2,500 coins)
        if not wait_until(lambda: inv_count("Coins") <= (coins_before - 2500), max_wait_ms=4000):
            return exponential_number(450, 1200, 1.2)
        self.state["foreman_paid"] = True
        self.state["foreman_paid_ts"] = time.monotonic()
        sleep_exponential(0.35, 1.2, 1)
        objects.click_object_closest_by_distance_prefer_no_camera(
            "Bank chest",
            action="Use",
            types=["GAME"],
            exact_match_object=False,
            exact_match_target_and_action=False,
            require_action_on_object=True,
        )
        wait_until(bank.is_open, max_wait_ms=7000)
        return exponential_number(350, 1100, 1.2)

    def coffer_coins(self, timeout: float = 0.35) -> int | None:
        """
        Return the amount of coins currently in the Blast Furnace coffer.
        Reads VarbitID.BLAST_FURNACE_COFFER (5357) via IPC.
        """
        return get_var(self.BLAST_FURNACE_COFFER_VARBIT, timeout=timeout)

    def _maybe_drink_stamina_at_bank(self) -> int | None:
        """
        If stamina is not active, drink 1 dose while the bank is open:
        - Withdraw a stamina potion
        - Drink it (from player inventory while bank is open)
        - Deposit it back
        """
        # Only drink when run energy is low
        run_energy = player.get_run_energy()
        # RuneLite runEnergy is typically 0..10000 (100.00%); treat < 30% as < 3000
        if (run_energy is None) or (int(run_energy) >= 3000):
            return None

        try:
            stamina_active = int(get_var(self.STAMINA_ACTIVE_VARBIT, timeout=0.35) or 0)
        except Exception:
            stamina_active = 0

        # Stamina duration counts down from 20 -> 0 (about 2 minutes total).
        # That implies ~6 seconds per unit. Drink when <= 10 seconds remain.
        try:
            stamina_dur_units = int(get_var(self.STAMINA_DURATION_VARBIT, timeout=0.35) or 0)
        except Exception:
            stamina_dur_units = 0
        dur_unit_seconds = 6
        drink_when_le_seconds = 10
        drink_when_le_units = max(0, (drink_when_le_seconds + dur_unit_seconds - 1) // dur_unit_seconds)  # ceil

        should_drink = (stamina_active != 1) or (stamina_dur_units <= int(drink_when_le_units))
        if not should_drink:
            return None

        if not bank.is_open():
            return None

        # Prefer lowest doses first
        pot_names = [
            "Stamina potion(1)",
            "Stamina potion(2)",
            "Stamina potion(3)",
            "Stamina potion(4)",
        ]

        # Ensure we have a potion in inventory
        inv_pot = None
        for nm in pot_names:
            if inventory.has_item(nm):
                inv_pot = nm
                break

        if inv_pot is None:
            # Withdraw 1 potion from bank
            for nm in pot_names:
                try:
                    if int(bank.get_item_count(nm) or 0) > 0:
                        bank.withdraw_item(nm, withdraw_x=1)
                        wait_until(lambda: inventory.has_any_items(pot_names), max_wait_ms=1200)
                        break
                except Exception:
                    continue

            for nm in pot_names:
                if inventory.has_item(nm):
                    inv_pot = nm
                    break

            if inv_pot is None:
                logging.warning(f"[{self.id}] Stamina not active but no stamina potions found in bank.")
                return None

            return exponential_number(200, 650, 1.2)

        # Drink a dose (menu option should exist even while bank is open)
        if not bank.interact_inventory(inv_pot, "Drink"):
            return exponential_number(200, 650, 1.2)

        # Wait briefly for varbit to flip
        wait_until(lambda: int(get_var(self.STAMINA_ACTIVE_VARBIT, timeout=0.2) or 0) == 1, max_wait_ms=1500)
        sleep_exponential(0.12, 0.35, 1.2)

        # Deposit whatever stamina dose remains
        for nm in pot_names:
            if inventory.has_item(nm):
                bank.deposit_item(nm, deposit_all=True)
                break

        return exponential_number(200, 650, 1.2)

    def ensure_coffer_minimum(
        self,
        *,
        min_coins: int = 5_000,
        top_up_to: int = 25_000,
        bank_prefer: str = "bank chest"
    ) -> bool:
        """
        If the Blast Furnace coffer is below `min_coins`, top it up to `top_up_to` by:
        - withdrawing Coins from the bank
        - using Coins on the coffer
        - typing the amount
        """
        cur = self.coffer_coins()
        if cur is None:
            return False
        if cur >= int(min_coins):
            return True

        # Deposit a flat amount when topping up (simpler/consistent behavior).
        deposit_amt = int(top_up_to)
        if deposit_amt <= 0:
            return True

        from helpers.keyboard import type_text, press_enter
        from helpers.widgets import widget_exists

        if not bank.is_open():
            if (bank_prefer or "").strip().lower() == "bank chest":
                objects.click_object_closest_by_distance_prefer_no_camera(
                    "Bank chest",
                    action="Use",
                    types=["GAME"],
                    exact_match_object=False,
                    exact_match_target_and_action=False,
                    require_action_on_object=True,
                )
            else:
                bank.open_bank(prefer=bank_prefer)
            wait_until(bank.is_open, max_wait_ms=6000)
            if not bank.is_open():
                return False

        # Withdraw coins (use your preferred bank flow)
        try:
            # Deposit inventory before filling coffer to avoid full inventory issues
            if bank.is_open() and not inventory.is_empty():
                bank.deposit_inventory()
                wait_until(lambda: inventory.is_empty(), max_wait_ms=1000)
            available = int(inv_count("Coins") or 0) + int(bank.get_item_count("Coins") or 0)
            if available < int(deposit_amt):
                self._set_done(f"Not enough coins to top up coffer (need {deposit_amt}).")
                return False
        except Exception:
            self._set_done(f"Not enough coins to top up coffer (need {deposit_amt}).")
            return False
        bank.withdraw_item("Coins", withdraw_all=True)
        wait_until(lambda: inventory.has_item("Coins"), max_wait_ms=3000)
        if not inventory.has_item("Coins"):
            self._set_done("Failed to withdraw coins to top up coffer.")
            return False

        bank.close_bank()
        wait_until(bank.is_closed, max_wait_ms=4000)

        objects.click_object_closest_by_distance_prefer_no_camera(
            "Coffer",
            action="Use",
            types=["GAME"],
            exact_match_object=False,
            exact_match_target_and_action=False,
            require_action_on_object=True,
        )
        # Some setups show a chat option prompt; others may go straight to the number input.
        # If options appear, choose option 1.
        wait_until(lambda: chat.can_choose_option(), max_wait_ms=4000)
        if chat.can_choose_option():
            sleep_exponential(0.12, 0.35, 1.2)
            chat.choose_option(1)
            sleep_exponential(0.15, 0.45, 1.2)
            wait_until(lambda: "Deposit how much?" in (chat.get_dialogue_text_raw()[0] or ""), max_wait_ms=4000)

        # Type deposit amount, press enter, and verify the coffer increases.
        coins_before = inv_count("Coins")
        sleep_exponential(0.15, 0.45, 1.2)
        type_text(str(deposit_amt), enter=False)
        sleep_exponential(0.1, 0.35, 1.2)
        press_enter()

        # Inventory coins should drop (usually by `need`), and coffer should increase.
        wait_until(lambda: inv_count("Coins") <= (coins_before - 1), max_wait_ms=4000)
        wait_until(lambda: (self.coffer_coins() or 0) >= int(cur) + 1, max_wait_ms=6000)

        if (bank_prefer or "").strip().lower() == "bank chest":
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Bank chest",
                action="Use",
                types=["GAME"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True,
            )
        else:
            bank.open_bank(prefer=bank_prefer)
        wait_until(bank.is_open, max_wait_ms=6000)
        sleep_exponential(0.15, 0.45, 1.2)

        return (self.coffer_coins() or 0) >= int(min_coins)

    def loop(self, ui) -> int:
        if not player.logged_in():
            player.login()
            return exponential_number(600, 1600, 1.2)

        phase = self.state.get("phase", "BANK")
        match phase:
            case "INITIAL_BANK":
                return self._handle_initial_bank()
            case "BANK":
                return self._handle_bank()
            case "SMELT":
                return self._handle_smelt()
            case "COLLECT_BARS":
                return self._handle_collect_bars()
            case "BANK_DEPOSIT":
                return self._handle_bank_deposit()
            case "DONE":
                # Stay idle; user can stop the plan in the GUI.
                return exponential_number(3000, 7000, 1.2)

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return exponential_number(400, 1100, 1.2)

    def _handle_initial_bank(self) -> int:
        """
        One-time startup banking:
        - bank open
        - ensure coffer has coins
        - pay foreman if smithing < 60 (one-time for now)
        - ensure only ice gloves equipped
        - ensure only coal bag in inventory
        - click Empty on coal bag once (and deposit any coal produced)
        Then transition to normal BANK phase.
        """
        if self.state.get("init_done"):
            self.set_phase("BANK")
            return exponential_number(350, 1100, 1.2)

        if not bank.is_open():
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Bank chest",
                action="Use",
                types=["GAME"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True,
            )
            wait_until(bank.is_open, max_wait_ms=7000)
            return exponential_number(350, 1100, 1.2)

        # Foreman: pay every ~10 minutes (when <1 min remains)
        foreman_delay = self._pay_foreman_if_needed()
        if foreman_delay is not None:
            return foreman_delay

        # coffer guard
        if not self.ensure_coffer_minimum(min_coins=5_000, top_up_to=75_000, bank_prefer="bank chest"):
            return exponential_number(450, 1500, 1.2)

        # Stamina upkeep (after foreman+coffer, before withdrawing ores)
        stamina_delay = self._maybe_drink_stamina_at_bank()
        if stamina_delay is not None:
            return stamina_delay

        # DONE conditions: missing Ice gloves / Coal bag
        if (not equipment.has_equipped("Ice gloves")) and (not inventory.has_item("Ice gloves")):
            if int(bank.get_item_count("Ice gloves") or 0) <= 0:
                return self._set_done("Missing Ice gloves.")

        # equipment guard
        if not equipment.ensure_only_equipped(["Ice gloves"], bank_prefer="bank chest", keep_bank_open=True):
            return exponential_number(450, 1400, 1.2)

        # inventory guard
        if not has_only_items(["Coal bag"]):
            bank.deposit_inventory()
            wait_until(lambda: inventory.is_empty(), max_wait_ms=1000)
            return exponential_number(0, 650, 1.2)

        if not inventory.has_item("Coal bag"):
            if int(bank.get_item_count("Coal bag") or 0) <= 0:
                return self._set_done("Missing Coal bag.")
            bank.withdraw_item("Coal bag", withdraw_x=1)
            wait_until(lambda: inventory.has_item("Coal bag"), max_wait_ms=1000)
            return exponential_number(0, 650, 1.2)

        bank.interact("Coal bag", "Empty")

        self.state["init_done"] = True
        self.set_phase("BANK")
        return exponential_number(250, 900, 1.2)

    def _handle_bank(self) -> int:
        # Open bank (use reusable bank logic; now supports bank chest)
        if not bank.is_open():
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Bank chest",
                action="Use",
                types=["GAME"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True,
            )
            wait_until(bank.is_open, max_wait_ms=7000)
            return exponential_number(350, 1100, 1.2)

        # --- First loop only: prime the furnace with 27 coal ---
        # This happens once, before we start the normal iron+coal-bag cycle.
        self.state["coal_primed"] = True
        if not self.state.get("coal_primed"):
            # Ensure we have room to withdraw 27 coal
            if inventory.get_empty_slots_count() < 2 and not inventory.has_item("Coal"):
                bank.deposit_inventory()
                wait_until(lambda: inventory.is_empty(), max_wait_ms=1000)
                return exponential_number(0, 650, 1.2)

            if not inventory.has_item("Coal"):
                bank.withdraw_item("Coal", withdraw_all=True)
                wait_until(lambda: inventory.has_item("Coal"), max_wait_ms=1000)
                return exponential_number(0, 800, 1.2)

            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=3000)
            self.set_phase("SMELT")
            return exponential_number(250, 900, 1.2)

        # Foreman: re-pay every ~10 minutes (when <1 min remains)
        foreman_delay = self._pay_foreman_if_needed()
        if foreman_delay is not None:
            return foreman_delay

        # Track Blast Furnace coffer coins (VarbitID.BLAST_FURNACE_COFFER = 5357)
        coins = self.coffer_coins()
        if coins is not None:
            logging.info(f"[{self.id}] Coffer coins: {coins}")

        # Top up the coffer if low (tune thresholds as needed)
        if not self.ensure_coffer_minimum(min_coins=5_000, top_up_to=25_000, bank_prefer="bank chest"):
            return exponential_number(450, 1500, 1.2)

        # Stamina upkeep (after foreman+coffer, before withdrawing ores)
        stamina_delay = self._maybe_drink_stamina_at_bank()
        if stamina_delay is not None:
            return stamina_delay

        # DONE conditions: missing Ice gloves / Coal bag / ores
        if (not equipment.has_equipped("Ice gloves")) and (not inventory.has_item("Ice gloves")):
            if int(bank.get_item_count("Ice gloves") or 0) <= 0:
                return self._set_done("Missing Ice gloves.")

        # Enforce ONLY Ice gloves equipped (reusable helper; keeps bank open)
        if not equipment.ensure_only_equipped(["Ice gloves"], bank_prefer="bank chest", keep_bank_open=True):
            return exponential_number(450, 1400, 1.2)

        # Ensure inventory is only Coal bag before withdrawing
        if not has_only_items(["Coal bag", "Iron ore"]):
            bank.deposit_inventory()
            wait_until(lambda: inventory.is_empty(), max_wait_ms=1000)
            return exponential_number(0, 650, 1.2)

        # Ensure coal bag is present
        if not inventory.has_item("Coal bag"):
            if int(bank.get_item_count("Coal bag") or 0) <= 0:
                return self._set_done("Missing Coal bag.")
            bank.withdraw_item("Coal bag", withdraw_x=1)
            wait_until(lambda: inventory.has_item("Coal bag"), max_wait_ms=1000)
            return exponential_number(0, 650, 1.2)

        # Slight randomness: sometimes Fill coal bag before withdrawing iron.
        did_fill = bool(self.state.get("did_fill_coal_bag"))

        if not inventory.has_item("Iron ore"):
            if int(bank.get_item_count("Iron ore") or 0) <= 0:
                return self._set_done("Out of Iron ore.")
            # 35% of the time, fill first (human-ish ordering variance)
            if (not did_fill) and (random.random() < 0.3744):
                if int(bank.get_item_count("Coal") or 0) <= 0:
                    return self._set_done("Out of Coal.")
                # Get coal count before filling to verify the fill succeeds
                coal_before = int(bank.get_item_count("Coal") or 0)
                bank.interact("Coal bag", "Fill")
                # Wait until coal count decreases in bank (verifies fill succeeded)
                if not wait_until(lambda: int(bank.get_item_count("Coal") or 0) < coal_before, max_wait_ms=1000):
                    logging.warning(f"[{self.id}] Coal bag fill may have failed (coal count did not decrease)")
                    return exponential_number(200, 600, 1.2)
                self.state["did_fill_coal_bag"] = True
                return exponential_number(300, 650, 1.2)

            bank.withdraw_item("Iron ore", withdraw_all=True)
            wait_until(lambda: inventory.has_item("Iron ore"), max_wait_ms=1000)
            return exponential_number(0, 650, 1.2)

        # Fill coal bag (once per BANK cycle)
        if not did_fill:
            if int(bank.get_item_count("Coal") or 0) <= 0:
                return self._set_done("Out of Coal.")
            # Get coal count before filling to verify the fill succeeds
            coal_before = int(bank.get_item_count("Coal") or 0)
            bank.interact("Coal bag", "Fill")
            # Wait until coal count decreases in bank (verifies fill succeeded)
            if not wait_until(lambda: int(bank.get_item_count("Coal") or 0) < coal_before, max_wait_ms=1000):
                logging.warning(f"[{self.id}] Coal bag fill may have failed (coal count did not decrease)")
                return exponential_number(200, 600, 1.2)
            self.state["did_fill_coal_bag"] = True
            sleep_exponential(0.3, 0.6, 1)

        # Close bank and move to smelt
        bank.close_bank()
        wait_until(bank.is_closed, max_wait_ms=3000)
        self.state.pop("did_fill_coal_bag", None)
        self.set_phase("SMELT")
        return exponential_number(0, 600, 1.2)

    def _handle_smelt(self) -> int:
        # First loop only: deposit the priming coal and return to bank.
        self.state["coal_primed"] = True
        if (not self.state.get("coal_primed")) and inventory.has_item("Coal"):
            if random.random() < 0.12:
                objects.click_object_closest_by_distance(
                    "Conveyor belt",
                    action="Put-ore-on",
                    types=["GAME"],
                    exact_match_object=False,
                    exact_match_target_and_action=False,
                    require_action_on_object=True,
                )
            else:
                objects.click_object_closest_by_distance_prefer_no_camera(
                    "Conveyor belt",
                    action="Put-ore-on",
                    types=["GAME"],
                    exact_match_object=False,
                    exact_match_target_and_action=False,
                    require_action_on_object=True,
                )
            wait_until(lambda: not inventory.has_item("Coal"), max_wait_ms=10000)
            self.state["coal_primed"] = True
            self.set_phase("BANK")
            return exponential_number(350, 1000, 1.2)

        if inventory.has_item("Iron ore"):
            if random.random() < 0.0321:
                objects.click_object_closest_by_distance(
                    "Conveyor belt",
                    action="Put-ore-on",
                    types=["GAME"],
                    exact_match_object=False,
                    exact_match_target_and_action=False,
                    require_action_on_object=True,
                )
            else:
                objects.click_object_closest_by_distance_prefer_no_camera(
                    "Conveyor belt",
                    action="Put-ore-on",
                    types=["GAME"],
                    exact_match_object=False,
                    exact_match_target_and_action=False,
                    require_action_on_object=True,
                )
            wait_until(lambda: not inventory.has_item("Iron ore"), max_wait_ms=10000)
            return exponential_number(0, 400, 1.2)

        # Empty coal bag
        if not inventory.has_any_items(["Coal", "Iron ore"]):
            if inventory.has_item("Coal bag"):
                inventory.interact("Coal bag", "Empty")
                wait_until(lambda: inventory.has_item("Coal"), max_wait_ms=10000)
                return exponential_number(0, 300, 1.2)

        if inventory.has_item("Coal"):
            # Put coal on the conveyor belt (must have Put-ore-on action)
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Conveyor belt",
                action="Put-ore-on",
                types=["GAME"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True,
            )
            wait_until(lambda: not inventory.has_item("Coal"), max_wait_ms=10000)
            self.set_phase("COLLECT_BARS")
            return exponential_number(0, 400, 1.2)

        return exponential_number(250, 900, 1.2)

    def _handle_collect_bars(self) -> int:
        # Collect bars
        if not inventory.has_item("Steel bar"):
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Bar dispenser",
                action=["Take", "Check"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True
            )
            wait_until(lambda: widget_exists(17694721) or chat.can_continue())
            while not widget_exists(17694721) and not inventory.has_item("Steel bar"):
                if chat.can_continue():
                    press_spacebar()
                elif objects.object_has_action("Bar dispenser", "Take", types=["GAME"], exact_match_object=False):
                    objects.click_object_closest_by_distance_prefer_no_camera(
                        "Bar dispenser",
                        action="Take",
                        exact_match_object=False,
                        exact_match_target_and_action=False,
                    )
                sleep_exponential(0.12, 0.35, 1.2)

            sleep_exponential(0.15, 0.45, 1.2)
            press_spacebar()
            if wait_until(lambda: inventory.has_item("Steel bar"), max_wait_ms=1000):
                self.set_phase("BANK_DEPOSIT")
            return exponential_number(0, 600, 1.1)

        else:
            self.set_phase("BANK_DEPOSIT")
        return exponential_number(0, 600, 1.2)

    def _handle_bank_deposit(self) -> int:
        if not bank.is_open():
            objects.click_object_closest_by_distance_prefer_no_camera(
                "Bank chest",
                action="Use",
                types=["GAME"],
                exact_match_object=False,
                exact_match_target_and_action=False,
                require_action_on_object=True,
            )
            wait_until(bank.is_open, max_wait_ms=10000)
            return exponential_number(300, 900, 1.2)

        # Deposit ALL bars
        if inventory.has_item("Steel bar"):
            bank.deposit_item("Steel bar", deposit_all=True)
            wait_until(lambda: not inventory.has_item("Steel bar"), max_wait_ms=1000)
            return 0

        self.set_phase("BANK")
        return exponential_number(250, 900, 1.2)


