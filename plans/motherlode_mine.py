#!/usr/bin/env python3
"""
Motherlode Mine Plan (scaffold)
===============================

This is a template plan for Motherlode Mine (MLM).

High-level loop:
- GO_TO_MLM: travel to the mine area
- BANK: use the bank chest inside MLM to deposit/withdraw for the loop
- MINE: mine Pay-dirt until inventory is full (or until a threshold)
- DEPOSIT: deposit Pay-dirt into the Hopper
- COLLECT: collect processed ores from the Sack
- Repeat

You MUST fill in:
- `self.bank_area`, `self.mlm_area` (region keys or rect tuples)
- Object names/actions in your scene:
  - ore vein: usually "Ore vein" with action "Mine"
  - hopper: usually "Hopper" with action like "Deposit" / "Use"
  - sack: usually "Sack" with action like "Search" / "Collect"

This is intentionally scaffolded and conservative: it does one action per loop and verifies state.
"""

import logging
import re
import time
import random

from actions import bank, inventory, player, wait_until, objects, equipment, tab
from actions.chat import can_continue
from actions.travel import in_area, go_to, go_to_bank
from actions.tab import is_tab_open
from helpers.keyboard import press_spacebar
from helpers.utils import sleep_exponential, exponential_number
from helpers.runtime_utils import ipc

from .base import Plan
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from services.camera_integration import (
    set_camera_state,
    set_interaction_object,
    set_camera_recording,
    CAMERA_STATE_IDLE_ACTIVITY,
    CAMERA_STATE_OBJECT_INTERACTION,
    CAMERA_STATE_LONG_TRAVEL,
    CAMERA_STATE_AREA_ACTIVITY,
    CAMERA_STATE_PHASE_TRANSITION
)


class MotherlodeMinePlan(Plan):
    id = "MOTHERLODE_MINE"
    label = "Mining: Motherlode Mine"
    description = """Mines Pay-dirt at the Motherlode Mine and processes it through the hopper system. Automatically deposits ore into the hopper and collects processed ores from the sack. Includes banking support for extended mining sessions.

Starting Area: Motherlode Mine
Required Items: Pickaxe"""
    DONE = 0

    # ---- State keys ----
    _S_DEPOSIT_RETRY = "deposit_retry"
    _S_CUR_VEIN_TILE = "cur_vein_tile"
    
    # Water wheel constants
    WATER_WHEEL_RUNNING_ID = 26671
    WATER_WHEEL_NOT_RUNNING_ID = 26672
    BROKEN_STRUT_ID = 26670

    def __init__(self):
        # Start by going to MLM (per your request)
        self.state = {"phase": "BANK", "last_mlm_mining_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        # ---- Configuration ----
        # - mlm_area: the overall MLM working area (where we mine/deposit/collect)
        # - mlm_bank_area: a small rect around the MLM bank chest inside the mine
        self.mlm_area = "MOTHERLODE_MINE"
        self.mlm_bank_area = "MLM_BANK_CHEST"

        # Bank chest config (MLM uses a bank chest, not a banker/booth)
        self.bank_chest_names = ["Bank chest"]
        # Common actions are "Use" or "Bank". We'll try both.
        self.bank_chest_actions = ["Bank", "Use"]

        # Items / object names
        self.paydirt_name = "Pay-dirt"
        self.ore_vein_names = ["Ore vein"]
        self.ore_vein_action = "Mine"

        self.hopper_names = ["Hopper"]
        self.hopper_action = "Deposit"

        self.sack_names = ["Sack"]
        self.sack_action = "Search"

        # Optional: what to keep stocked in inventory each loop.
        # Format: {"name": str, "quantity": int}, quantity=-1 means withdraw all.
        self.required_items = []

        # Pickaxe options in order of preference (best to worst).
        # Format: (pickaxe_name, mining_level_required, attack_level_required)
        # Note: Attack reqs are for wielding; if you don't meet them we will keep the pickaxe in inventory.
        self.pickaxe_options = [
            ("Crystal pickaxe", 71, 70),
            ("Infernal pickaxe", 61, 60),
            ("Dragon pickaxe", 61, 60),
            ("Rune pickaxe", 41, 40),
            ("Adamant pickaxe", 31, 30),
            ("Mithril pickaxe", 21, 20),
            ("Black pickaxe", 11, 10),
            ("Steel pickaxe", 6, 5),
            ("Iron pickaxe", 1, 1),
            ("Bronze pickaxe", 1, 1),
        ]

        # Mine ore veins ONLY within this fixed area (your requested consolidation).
        # Format: (min_x, max_x, min_y, max_y)
        self.ore_vein_area = (3729, 3757, 5645, 5673)
        
        # Calculate area center for camera state (AREA_ACTIVITY)
        min_x, max_x, min_y, max_y = self.ore_vein_area
        self.area_center = {
            "x": (min_x + max_x) // 2,
            "y": (min_y + max_y) // 2
        }
        
        # Configure camera states for MLM
        self.camera_states = {
            CAMERA_STATE_IDLE_ACTIVITY: {
                "zoom_preference": "area_wide",
                "yaw_behavior": "point_to_area_center",
                "movement_frequency": "idle",
                "idle_probability": 0.85,  # 85% chance to skip adjustments - camera should stay stable
                "zoom_range": (550, 650),  # Zoom out more for MLM area
                "pitch_range": (250, 400),  # Medium-high pitch to see character and objects
            },
            CAMERA_STATE_OBJECT_INTERACTION: {
                "zoom_preference": "medium",
                "yaw_behavior": "point_to_object",
                "pitch_behavior": "auto",
                "movement_frequency": "active",
                "idle_probability": 0.0,  # Always adjust for object interactions
                "zoom_range": (450, 550),
                "pitch_range": (300, 500),
            },
            CAMERA_STATE_LONG_TRAVEL: {
                "zoom_preference": "area_wide",
                "yaw_behavior": "follow_path",
                "pitch_behavior": "auto",
                "movement_frequency": "moderate",
                "idle_probability": 0.3,
                "zoom_range": (500, 600),
                "pitch_range": (150, 350),
            },
            CAMERA_STATE_AREA_ACTIVITY: {
                "zoom_preference": "area_wide",
                "yaw_behavior": "point_to_area_center",
                "pitch_behavior": "auto",
                "movement_frequency": "idle",
                "idle_probability": 0.6,
                "zoom_range": (500, 600),
                "pitch_range": (200, 400),
            },
        }

        # Optional: recovery area to walk to if repeated hopper clicks don't register.
        # Defaults to the mining/working area, but you can make this a tight rect around the Hopper.
        self.hopper_area = self.ore_vein_area

        # Mining animation grace period for is_activity_active()
        self.mining_anim_grace_s = 2
        self.state.setdefault(self._S_DEPOSIT_RETRY, 0)

        logging.info(f"[{self.id}] Plan initialized")

        # Optional: if you know the widget ids for the sack counters, fill these in.
        # - `sack_paydirt_widget_id`: current pay-dirt remaining in the sack (stop collecting when it hits 0)
        # - `sack_space_widget_id`: free space/room indicator for the sack (informational; not required)
        # Motherlode HUD widgets (you provided these):
        # - 382.5 MotherlodeHud.PAY_DIRT  -> id 25034757, example text: '87'
        # - 382.6 MotherlodeHud.DEPOSITS  -> id 25034758, example text: 'Space: 21'
        self.sack_paydirt_widget_id: int = 25034757
        self.sack_space_widget_id: int = 25034758
        # When the sack reaches this many pay-dirt, MLM prevents further mining until you collect.
        self.sack_full_pay_dirt = 108

        # Items that count as "ores and such" for the Collect -> Bank loop.
        # If *any* of these are in inventory, we bank them (deposit inventory).
        # Keep this list simple and extend as needed.
        self.collect_bank_item_names = [
            "Coal",
            "Gold ore",
            "Mithril ore",
            "Adamantite ore",
            "Runite ore",
            "Gold nugget",
            "Uncut sapphire",
            "Uncut emerald",
            "Uncut ruby",
            "Uncut diamond",
        ]

    # --- Motherlode HUD helpers ---
    def _sack_paydirt_remaining(self) -> int | None:
        from actions.widgets import get_widget_text
        txt = get_widget_text(self.sack_paydirt_widget_id)
        if not txt:
            return None
        m = re.search(r"(\d+)", txt)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _hopper_space_left(self) -> int | None:
        # "Space: N"
        from actions.widgets import get_widget_text
        txt = get_widget_text(self.sack_space_widget_id)
        if not txt:
            return None
        m = re.search(r"(\d+)", txt)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _has_broken_struts(self) -> bool:
        """
        Check if there are broken struts that need repair.
        Returns True only if BOTH struts are broken (2 or more broken struts with 'Hammer' action).
        """
        try:
            resp = ipc.get_objects("Broken strut", types=["GAME"], radius=26) or {}
            if not resp.get("ok"):
                return False
            
            objs = resp.get("objects", []) or []
            min_x, max_x, min_y, max_y = self.ore_vein_area
            broken_count = 0
            
            for o in objs:
                w = o.get("world", {}) or {}
                ox, oy = w.get("x"), w.get("y")
                op = w.get("p", w.get("plane", 0)) or 0
                if not (isinstance(ox, int) and isinstance(oy, int)):
                    continue
                
                # Check if object is in our area
                if ox < int(min_x) or ox > int(max_x) or oy < int(min_y) or oy > int(max_y):
                    continue
                
                # Check if it has 'Hammer' action
                if objects.object_at_tile_has_action(ox, oy, int(op), "Broken strut", "Hammer", types=["GAME"], exact_match_object=True):
                    broken_count += 1
            
            # Only return True if both struts are broken (2 or more)
            return broken_count >= 2
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking for broken struts: {e}")
            return False

    def _repair_broken_struts(self) -> bool:
        """
        Repair broken struts by hammering them.
        Returns True if repair was successful or no broken struts found.
        """
        try:
            # Check if we have a hammer
            if not inventory.has_item("Hammer"):
                # Need to get hammer from bank
                if not bank.is_open():
                    result = go_to_bank(
                        bank_area=self.mlm_bank_area,
                        prefer="bank chest",
                        prefer_no_camera=True
                    )
                    if not result or not bank.is_open():
                        return False

                sleep_exponential(0.3, 1.1, 1.5)
                # Withdraw hammer
                if not bank.has_item("Hammer"):
                    logging.warning(f"[{self.id}] No hammer in bank!")
                    return False
                
                bank.withdraw_item("Hammer", withdraw_x=1)
                wait_until(lambda: inventory.has_item("Hammer"), max_wait_ms=2500)
                if not inventory.has_item("Hammer"):
                    return False

                sleep_exponential(0.1, 1.1, 1.5)
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            
            # Find and click broken struts
            resp = ipc.get_objects("Broken strut", types=["GAME"], radius=26) or {}
            if not resp.get("ok"):
                return True  # No broken struts found, consider it success
            
            objs = resp.get("objects", []) or []
            min_x, max_x, min_y, max_y = self.ore_vein_area
            repaired_count = 0
            
            for o in objs:
                w = o.get("world", {}) or {}
                ox, oy = w.get("x"), w.get("y")
                op = w.get("p", w.get("plane", 0)) or 0

                # Check if it has 'Hammer' action
                if objects.object_at_tile_has_action(ox, oy, int(op), "Broken strut", "Hammer", types=["GAME"], exact_match_object=True):
                    # Click the broken strut to repair it
                    res = objects.click_object_in_area_action_auto_prefer_no_camera(
                        "Broken strut",
                        area=(ox, ox, oy, oy),
                        prefer_action="Hammer",
                        types=["GAME"],
                        exact_match_object=True,
                        exact_match_target_and_action=True,
                    )
                    if res:
                        repaired_count += 1
                        # Wait a bit for repair animation
                        wait_until(lambda: player.get_player_animation() == "MLM_STRUT_REPAIR" or not objects.object_at_tile_has_action(ox, oy, int(op), "Broken strut", "Hammer", types=["GAME"], exact_match_object=True))
                        wait_until(lambda: not objects.object_at_tile_has_action(ox, oy, int(op), "Broken strut", "Hammer", types=["GAME"], exact_match_object=True), max_wait_ms=30000)
                        sleep_exponential(0, 1.0, 1.2)
            
            # If we repaired at least one, wait a moment and verify
            if repaired_count > 0:
                sleep_exponential(0, 1.1, 1.2)
                # Check if struts are still broken
                return not self._has_broken_struts()
            
            return True  # No broken struts to repair
        except Exception as e:
            logging.warning(f"[{self.id}] Error repairing broken struts: {e}")
            return False

    # --- Pickaxe helpers (reused in BANK and after COLLECT deposits) ---
    def _clear_current_vein(self) -> None:
        old_vein = self.state.get(self._S_CUR_VEIN_TILE)
        self.state[self._S_CUR_VEIN_TILE] = None
        if old_vein:
            logging.info(f"[{self.id}] Cleared tracked vein: {old_vein}")

    def _set_current_vein(self, tile: tuple[int, int, int] | None) -> None:
        if tile:
            old_vein = self.state.get(self._S_CUR_VEIN_TILE)
            new_vein = (int(tile[0]), int(tile[1]), int(tile[2]))
            self.state[self._S_CUR_VEIN_TILE] = new_vein
            if old_vein != new_vein:
                logging.info(f"[{self.id}] Set tracked vein: {new_vein} (was: {old_vein})")
        else:
            self._clear_current_vein()

    def _detect_current_mining_vein(self) -> tuple[int, int, int] | None:
        """
        Detect which ore vein we're currently mining based on player orientation.
        Returns (x, y, plane) of the vein we're facing, or None if not found.
        """
        try:
            resp = ipc.get_player() or {}
            if not resp.get("ok"):
                return None
            pl = resp.get("player") or {}
            if not isinstance(pl, dict):
                return None
                return None
            
            px = pl.get("worldX")
            py = pl.get("worldY")
            pp = pl.get("plane", pl.get("worldP", 0))
            orientation = pl.get("orientation")
            
            if not (isinstance(px, int) and isinstance(py, int) and isinstance(orientation, int)):
                return None
            
            # Get nearby ore veins
            resp = ipc.get_objects(self.ore_vein_names[0], types=["WALL"], radius=3) or {}
            if not resp.get("ok"):
                return None
            
            objs = resp.get("objects", []) or []
            if not objs:
                return None
            
            # Find vein in the direction we're facing
            # Orientation: 0 = south, 512 = west, 1024 = north, 1536 = east
            from constants import ORIENTATION_SOUTH, ORIENTATION_WEST, ORIENTATION_NORTH, ORIENTATION_EAST
            # Calculate expected direction based on orientation
            if orientation < 256 or orientation >= 1792:  # South (0-256 or 1792-2047)
                expected_dir = (0, -1)  # South
            elif 256 <= orientation < 768:  # West
                expected_dir = (-1, 0)  # West
            elif 768 <= orientation < 1280:  # North
                expected_dir = (0, 1)  # North
            else:  # East (1280-1792)
                expected_dir = (1, 0)  # East
            
            # Find adjacent vein in the expected direction
            best_vein = None
            best_dist = 999
            
            for o in objs:
                w = o.get("world", {}) or {}
                ox, oy = w.get("x"), w.get("y")
                op = w.get("p", w.get("plane", 0)) or 0
                
                if not (isinstance(ox, int) and isinstance(oy, int)):
                    continue
                if int(op) != int(pp):
                    continue
                
                # Check if adjacent (distance <= 1)
                dx = abs(ox - px)
                dy = abs(oy - py)
                if dx > 1 or dy > 1:
                    continue
                
                # Check if in the direction we're facing
                dir_x = ox - px
                dir_y = oy - py
                
                # Check if direction matches expected direction
                if (expected_dir[0] == 0 and dir_x == 0 and (dir_y > 0) == (expected_dir[1] > 0)) or \
                   (expected_dir[1] == 0 and dir_y == 0 and (dir_x > 0) == (expected_dir[0] > 0)):
                    dist = dx + dy
                    if dist < best_dist:
                        best_dist = dist
                        best_vein = (int(ox), int(oy), int(op))
            
            return best_vein
        except Exception as e:
            logging.warning(f"[{self.id}] Error detecting current mining vein: {e}")
            return None


    def _get_non_busy_veins(self) -> list[tuple[int, int, int]]:
        """
        Get list of non-busy ore veins in the mining area.
        
        Returns:
            List of (x, y, plane) tuples for non-busy veins
        """
        try:
            obj_resp = ipc.get_objects(self.ore_vein_names[0], types=["WALL"], radius=26) or {}
            objs = obj_resp.get("objects", []) if obj_resp.get("ok") else []
            
            min_x, max_x, min_y, max_y = self.ore_vein_area
            non_busy_veins = []
            
            for o in objs:
                w = o.get("world", {}) or {}
                ox, oy = w.get("x"), w.get("y")
                op = w.get("p", w.get("plane", 0)) or 0
                if not (isinstance(ox, int) and isinstance(oy, int)):
                    continue
                if ox < int(min_x) or ox > int(max_x) or oy < int(min_y) or oy > int(max_y):
                    continue
                
                # Check if vein is busy
                if not self._is_vein_busy(ox, oy, int(op)):
                    non_busy_veins.append((ox, oy, int(op)))
            
            return non_busy_veins
        except Exception as e:
            logging.warning(f"[{self.id}] Error getting non-busy veins: {e}")
            return []

    def _is_vein_busy(self, vein_x: int, vein_y: int, vein_plane: int) -> bool:
        """
        Check if a vein is busy (being mined by another player).
        
        A vein is considered busy if a player is:
        - Adjacent to the vein (distance <= 1)
        - Oriented towards the vein
        - Has animation between 6700-6800
        """
        try:
            players_resp = ipc.get_players() or {}
            if not players_resp.get("ok"):
                return False
            
            players = players_resp.get("players", []) or []
            if not players:
                return False
            
            for p in players:
                # Skip local player
                if p.get("isLocalPlayer"):
                    continue
                
                px = p.get("worldX")
                py = p.get("worldY")
                pp = p.get("plane", p.get("worldP", 0))
                anim = p.get("animation")
                orientation = p.get("orientation")  # May be None if not in IPC response
                
                if not (isinstance(px, int) and isinstance(py, int)):
                    continue
                
                # Check if on same plane
                if int(pp) != int(vein_plane):
                    continue
                
                # Check if adjacent (Chebyshev distance <= 1)
                dx = abs(px - vein_x)
                dy = abs(py - vein_y)
                if dx > 1 or dy > 1:
                    continue
                
                # Check if has mining animation (6700-6800)
                if not isinstance(anim, int) or anim < 6700 or anim > 6800:
                    continue
                
                # Check if oriented towards vein
                # Orientation: 0 = south, 512 = west, 1024 = north, 1536 = east
                if orientation is not None and isinstance(orientation, int):
                    from constants import ORIENTATION_SOUTH, ORIENTATION_WEST, ORIENTATION_NORTH, ORIENTATION_EAST
                    # Calculate direction from player to vein
                    dir_x = vein_x - px
                    dir_y = vein_y - py
                    
                    # Convert direction to expected orientation
                    # 0 = south (dir_y < 0), 512 = west (dir_x < 0), 1024 = north (dir_y > 0), 1536 = east (dir_x > 0)
                    expected_orientation = None
                    if abs(dir_x) > abs(dir_y):
                        # More horizontal
                        expected_orientation = ORIENTATION_EAST if dir_x > 0 else ORIENTATION_WEST
                    else:
                        # More vertical
                        expected_orientation = ORIENTATION_NORTH if dir_y > 0 else ORIENTATION_SOUTH
                    
                    # Check if orientation is roughly towards vein (within 256 degrees, which is 90 degrees)
                    orientation_diff = abs(orientation - expected_orientation)
                    if orientation_diff > 1024:
                        orientation_diff = 2048 - orientation_diff
                    if orientation_diff > 256:
                        continue  # Not oriented towards vein, skip this player
                else:
                    # If orientation is not available, skip orientation check but still check other conditions
                    # This allows the check to work even if orientation data is missing
                    pass
                
                # All conditions met - vein is busy
                return True
            
            return False
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking if vein is busy: {e}")
            return False

    # --- Pickaxe helpers (reused in BANK and after COLLECT deposits) ---
    def _best_pickaxe(self, *, bank_open: bool) -> tuple[str, int, bool] | None:
        """
        Return (name, attack_req, can_equip) for the best pickaxe we have access to.
        Prefers: equipped > inventory > bank (only if bank_open=True).
        """
        mining_level = player.get_skill_level("mining") or 1
        attack_level = player.get_skill_level("attack") or 1

        for nm, mining_req, att_req in self.pickaxe_options:
            if mining_level < int(mining_req):
                continue
            have_it = (
                equipment.has_equipped(nm)
                or inventory.has_item(nm)
                or (bank_open and bank.has_item(nm))
            )
            if not have_it:
                continue
            att_req_i = int(att_req)
            return nm, att_req_i, (attack_level >= att_req_i)
        return None

    def _ensure_best_pickaxe(self, *, bank_open: bool) -> tuple[str, int, bool] | None:
        """
        Ensure the best pickaxe is either equipped or in inventory.
        If it's only in the bank and the bank is open, withdraw it.
        """
        info = self._best_pickaxe(bank_open=bank_open)
        if not info:
            return None
        nm, _att_req, _can_equip = info

        if equipment.has_equipped(nm) or inventory.has_item(nm):
            return info

        if bank_open and bank.has_item(nm):
            bank.withdraw_item(nm, withdraw_x=1)
            wait_until(lambda: inventory.has_item(nm), max_wait_ms=2500)
        return info

    def _equip_pickaxe_if_possible(self, info: tuple[str, int, bool] | None) -> None:
        """
        Best-effort: equip pickaxe if we can. (Not required to mine, but prevents accidental banking.)
        Call this when bank is CLOSED.
        """
        if not info:
            return
        nm, _att_req, can_equip = info
        if not can_equip:
            return
        if equipment.has_equipped(nm):
            return
        if not inventory.has_item(nm):
            return

        inventory.interact(nm, "Wield")
        if not wait_until(lambda: equipment.has_equipped(nm), max_wait_ms=2500):
            inventory.interact(nm, "Wear")
            wait_until(lambda: equipment.has_equipped(nm), max_wait_ms=2500)

    def set_phase(self, phase: str, camera_setup: bool = True):
        # Set camera state based on phase
        if phase == "GO_TO_MLM":
            set_camera_state(CAMERA_STATE_LONG_TRAVEL, self.camera_states.get(CAMERA_STATE_LONG_TRAVEL))
        elif phase == "BANK":
            # Will use OBJECT_INTERACTION when clicking bank chest
            set_camera_state(CAMERA_STATE_OBJECT_INTERACTION, self.camera_states.get(CAMERA_STATE_OBJECT_INTERACTION))
        elif phase == "MINE":
            # Use IDLE_ACTIVITY with area center for mining
            set_camera_state(
                CAMERA_STATE_IDLE_ACTIVITY, 
                self.camera_states.get(CAMERA_STATE_IDLE_ACTIVITY),
                area_center=self.area_center
            )
        elif phase == "DEPOSIT":
            set_camera_state(CAMERA_STATE_OBJECT_INTERACTION, self.camera_states.get(CAMERA_STATE_OBJECT_INTERACTION))
        elif phase == "COLLECT":
            set_camera_state(CAMERA_STATE_OBJECT_INTERACTION, self.camera_states.get(CAMERA_STATE_OBJECT_INTERACTION))
        elif phase == "DONE":
            # Clear camera state for done phase
            from services.camera_integration import clear_camera_state
            clear_camera_state()
        
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "GO_TO_MLM")

        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        match phase:
            case "GO_TO_MLM":
                return self._handle_go_to_mlm()
            case "BANK":
                return self._handle_bank(ui)

            case "MINE":
                return self._handle_mine()
            case "DEPOSIT":
                return self._handle_deposit()
            case "COLLECT":
                return self._handle_collect()
            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_bank(self, ui) -> int:
        # Continuous camera adjustment
        from services.camera_integration import adjust_camera_continuous
        adjust_camera_continuous()
        
        # Travel to bank and open it using go_to_bank()
        if not bank.is_open():
            result = go_to_bank(
                bank_area=self.mlm_bank_area,
                prefer="bank chest",
                prefer_no_camera=True
            )
            if not result or not bank.is_open():
                return exponential_number(600, 2000, 0.8)

        # Ensure we have the best pickaxe for our mining level
        pickaxe_info = self._ensure_best_pickaxe(bank_open=True)
        if not pickaxe_info:
            # No pickaxe available - close bank and stay in BANK phase to retry
            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=3000)
            return exponential_number(600, 2000, 0.8)
        
        pickaxe_name, _att_req, can_equip = pickaxe_info
        
        # Deposit unwanted items (everything except pickaxe and pay-dirt)
        # Pay-dirt cannot be deposited, so we include it in the exception list
        required_items = [pickaxe_name, self.paydirt_name]
        bank.deposit_unwanted_items(required_items, max_unique_for_bulk=3)
        wait_until(lambda: inventory.has_only_items(required_items), max_wait_ms=3000)
        
        # Equip pickaxe if possible (while bank is still open)
        if can_equip and inventory.has_item(pickaxe_name) and not equipment.has_equipped(pickaxe_name):
            inventory.interact(pickaxe_name, "Wield")
            if not wait_until(lambda: equipment.has_equipped(pickaxe_name), max_wait_ms=2500):
                inventory.interact(pickaxe_name, "Wear")
                wait_until(lambda: equipment.has_equipped(pickaxe_name), max_wait_ms=2500)
        
        # Close bank (sometimes click ore vein directly instead of closing normally)
        bank.close_bank(
            object_name=self.ore_vein_names[0],
            object_action=self.ore_vein_action,
            prefer_no_camera=True
        )
        wait_until(bank.is_closed, max_wait_ms=3000)
        
        # Transition to MINE phase
        self.state["last_mlm_mining_ts"] = None
        self.set_phase("MINE")
        return self.loop_interval_ms

    def _handle_go_to_mlm(self) -> int:
        if not in_area(self.mlm_area):
            go_to(self.mlm_area)
            return self.loop_interval_ms
        # Once inside MLM, go bank first to sync inventory for the loop.
        self.set_phase("BANK")
        return self.loop_interval_ms

    def _maybe_tab_switch(self) -> None:
        """
        Occasionally switch to either SKILLS or INVENTORY (human-like tab switching).
        """
        if bank.is_open():
            return

        now = time.time()
        nxt = self.state.get("next_tab_switch_ts")
        if not isinstance(nxt, (int, float)):
            self.state["next_tab_switch_ts"] = now + exponential_number(60.0, 1800.0, 0.5, output_type="float")
            return

        if now < float(nxt):
            return

        if is_tab_open("INVENTORY"):
            tab.open_tab("SKILLS")
        elif is_tab_open("SKILLS"):
            tab.open_tab("INVENTORY")
        else:
            tab.open_tab("INVENTORY")

        self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))

    def _handle_mine(self) -> int:
        # Occasionally switch tabs
        self._maybe_tab_switch()
        
        # Continuous camera adjustment - runs every loop
        from services.camera_integration import adjust_camera_continuous
        adjust_camera_continuous()

        if can_continue():
            self.state["last_mlm_mining_ts"] = None
            press_spacebar()
            return exponential_number(200, 2000, 1)

        # Hard guard: if the sack is full, the game blocks further mining until you collect.
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd >= int(self.sack_full_pay_dirt):
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("COLLECT")
            return self.loop_interval_ms

        # If inventory is full, deposit pay-dirt.
        paydirt_in_inv = inventory.inv_count(self.paydirt_name)
        space_left = self._hopper_space_left()

        # Deposit if:
        # - inventory is full, OR
        # - pay-dirt in inventory >= hopper remaining space (Space: N)
        if inventory.is_full() or (isinstance(space_left, int) and space_left >= 0 and paydirt_in_inv > 0 and paydirt_in_inv >= space_left):
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("DEPOSIT")
            return self.loop_interval_ms

        # Check if already mining using is_activity_active()
        if player.is_activity_active("MLM_MINING", "last_mlm_mining_ts", self.mining_anim_grace_s, self.state):
            # Detect and save current mining vein if not already saved
            cur_vein = self.state.get(self._S_CUR_VEIN_TILE)
            if not cur_vein:
                logging.info(f"[{self.id}] Mining active but no tracked vein - detecting current vein...")
                detected_vein = self._detect_current_mining_vein()
                if detected_vein:
                    logging.info(f"[{self.id}] Detected and set current vein: {detected_vein}")
                    self._set_current_vein(detected_vein)
                    # Update camera interaction object
                    set_interaction_object({"x": detected_vein[0], "y": detected_vein[1], "plane": detected_vein[2]})
                    return 0
                else:
                    logging.info(f"[{self.id}] Could not detect current mining vein (may be too far or orientation unclear)")
                    return 0
            else:
                # Check if our current vein still has 'Mine' action
                if isinstance(cur_vein, (tuple, list)) and len(cur_vein) >= 3:
                    vein_x, vein_y, vein_plane = int(cur_vein[0]), int(cur_vein[1]), int(cur_vein[2])
                    logging.info(f"[{self.id}] Mining active, checking tracked vein at ({vein_x}, {vein_y}, {vein_plane})...")
                    # Check if vein at specific tile still has 'Mine' action
                    if not objects.object_at_tile_has_action(vein_x, vein_y, vein_plane, self.ore_vein_names[0], "Mine", types=["WALL"], exact_match_object=True):
                        # Vein no longer has 'Mine' action - clear it and allow clicking new vein
                        logging.info(f"[{self.id}] Tracked vein at ({vein_x}, {vein_y}, {vein_plane}) no longer has 'Mine' action - clearing and allowing new vein click")
                        self.state["last_mlm_mining_ts"] = None
                        self._clear_current_vein()
                        set_interaction_object(None)  # Clear interaction object
                        # Fall through to click new vein
                    else:
                        # Still mining the same vein - update interaction object
                        set_interaction_object({"x": vein_x, "y": vein_y, "plane": vein_plane})
                        logging.info(f"[{self.id}] Still mining tracked vein at ({vein_x}, {vein_y}, {vein_plane}) - waiting")
                        return 0
                else:
                    logging.warning(f"[{self.id}] Tracked vein has invalid format: {cur_vein} - clearing")
                    self._clear_current_vein()
                    set_interaction_object(None)
            return 0

        # Clear current vein when not mining
        logging.info(f"[{self.id}] Mining NOT active - clearing tracked vein and allowing new vein click")
        self._clear_current_vein()
        set_interaction_object(None)  # Clear interaction object

        # Get non-busy veins and click first one if available
        non_busy_veins = self._get_non_busy_veins()
        if non_busy_veins:
            ox, oy, op = non_busy_veins[0]
            res = objects.click_object_in_area_action_auto_prefer_no_camera(
                self.ore_vein_names[0],
                area=(ox, ox, oy, oy),
                prefer_action=self.ore_vein_action,
                types=["WALL"],
                exact_match_object=True,
                exact_match_target_and_action=True,
            )
            if res:
                # Wait for mining animation to start, then update timestamp and detect vein
                if wait_until(lambda: player.get_player_animation() == "MLM_MINING", max_wait_ms=10000):
                    self.state["last_mlm_mining_ts"] = time.time()
                    # Detect which vein we're mining based on orientation
                    detected_vein = self._detect_current_mining_vein()
                    if detected_vein:
                        self._set_current_vein(detected_vein)
                        set_interaction_object({"x": detected_vein[0], "y": detected_vein[1], "plane": detected_vein[2]})
                    return 0
            return self.loop_interval_ms
        
        # Fallback: click any ore vein in area
        res = objects.click_object_in_area_action_auto_prefer_no_camera(
            self.ore_vein_names[0],
            area=self.ore_vein_area,
            prefer_action=self.ore_vein_action,
            types=["WALL"],
            exact_match_object=True,
            exact_match_target_and_action=True,
        )
        if res:
            # Wait for mining animation to start, then update timestamp and detect vein
            if wait_until(lambda: player.get_player_animation() == "MLM_MINING", max_wait_ms=10000):
                self.state["last_mlm_mining_ts"] = time.time()
                # Detect which vein we're mining based on orientation
                detected_vein = self._detect_current_mining_vein()
                if detected_vein:
                    self._set_current_vein(detected_vein)
                return 0
        return self.loop_interval_ms

    def _handle_deposit(self) -> int:
        """
        Deposit Pay-dirt into the Hopper.
        This is a placeholder: MLM typically uses the hopper to process pay-dirt.
        """
        # Continuous camera adjustment
        from services.camera_integration import adjust_camera_continuous
        adjust_camera_continuous()
        
        # If we somehow have no pay-dirt, go collect (maybe sack is ready) or mine again.
        if not inventory.has_item(self.paydirt_name):
            self.state[self._S_DEPOSIT_RETRY] = 0
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("MINE")
            # self.set_phase("COLLECT")
            return self.loop_interval_ms

        before = inventory.inv_count(self.paydirt_name)

        # Start with direct hopper click.
        res = objects.click_object_in_area_action_auto_prefer_no_camera(
            "Hopper",
            area=self.ore_vein_area,
            prefer_action="Deposit",
            exact_match_object=True,
            exact_match_target_and_action=True,
        )

        # Require pay-dirt to actually decrease to consider this deposit successful.
        # If it doesn't drop, stay in DEPOSIT and retry (bounded).
        deposited = bool(wait_until(lambda: inventory.inv_count("Pay-dirt") < before, max_wait_ms=12000))

        if not deposited:
            retry = int(self.state.get(self._S_DEPOSIT_RETRY) or 0) + 1
            self.state[self._S_DEPOSIT_RETRY] = retry

            # After a few failures, do a simple positional recovery and try again.
            if retry >= 3:
                self.state[self._S_DEPOSIT_RETRY] = 0
                go_to(self.hopper_area)
                return exponential_number(450, 1800, 1.2)

            return exponential_number(220, 1200, 1.25)

        # Successful deposit: reset retry counter.
        self.state[self._S_DEPOSIT_RETRY] = 0

        # Check if water wheels need repair (broken struts present)
        if self._has_broken_struts():
            # Repair broken struts
            if not self._repair_broken_struts():
                # Repair failed, retry next loop
                return exponential_number(400, 1100, 1.2)
            # Repair successful, continue with normal flow

        # If that deposit filled the sack, go collect immediately (mining will be blocked otherwise).
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd >= int(self.sack_full_pay_dirt):
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("COLLECT")
            return self.loop_interval_ms

        # After a successful deposit, re-check hopper space. If it's getting low, go collect from sack.
        # Widget 25034758 example: "Space: 21"
        space_left = self._hopper_space_left()
        if isinstance(space_left, int) and space_left < 27:
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("COLLECT")
            return self.loop_interval_ms

        # Return to mining
        self.state["last_mlm_mining_ts"] = None
        self.set_phase("MINE")
        return self.loop_interval_ms

    def _handle_collect(self) -> int:
        """
        Collect processed ores from the Sack.
        """
        # Continuous camera adjustment
        from services.camera_integration import adjust_camera_continuous
        adjust_camera_continuous()
        
        # Guard 1: Bank ores if we have any
        if inventory.has_any_items(self.collect_bank_item_names):
            if not bank.is_open():
                result = go_to_bank(
                    bank_area=self.mlm_bank_area,
                    prefer="bank chest",
                    prefer_no_camera=True
                )
                if not result or not bank.is_open():
                    return exponential_number(450, 1800, 1.2)
            bank.deposit_inventory()
            wait_until(lambda: not inventory.has_any_items(self.collect_bank_item_names), max_wait_ms=3000)
            sack_pd = self._sack_paydirt_remaining()
            if isinstance(sack_pd, int) and sack_pd <= 0:
                # Sack empty - go to BANK phase to bank and prepare inventory
                self.state["last_mlm_mining_ts"] = None
                self.set_phase("BANK")
                return exponential_number(400, 1100, 1.2)
            # Close bank (sometimes click sack directly instead of closing normally)
            close_result = bank.close_bank(
                object_name=self.sack_names[0],
                object_action=self.sack_action,
                prefer_no_camera=True
            )
            
            if close_result:
                action_taken = close_result.get("action")
                
                if action_taken == "object_click":
                    # Bank was closed by clicking sack directly - wait for bank to close, then items to be collected
                    wait_until(bank.is_closed, max_wait_ms=3000)
                    wait_until(lambda: inventory.has_any_items(self.collect_bank_item_names), max_wait_ms=10000)
                elif action_taken == "ground_click":
                    # Bank was closed by clicking ground to move toward sack - wait for bank to close and movement
                    wait_until(bank.is_closed, max_wait_ms=3000)
                    # Movement will happen, then code will continue to Guard 5 which will click the sack
                else:  # "normal_close"
                    # Bank was closed normally with ESC - wait for bank to close, then code will continue to Guard 5
                    wait_until(bank.is_closed, max_wait_ms=3000)
            else:
                # Bank wasn't open or close failed - just wait a bit
                wait_until(bank.is_closed, max_wait_ms=1000)
            
            # If we clicked the object, we're done here (items should be collected)
            # Otherwise, the code will continue to Guard 5 which will handle clicking the sack
            if close_result and close_result.get("action") == "object_click":
                return exponential_number(0, 1100, 1.2)
            
            # Continue to next guard (will click sack if needed)
            return exponential_number(0, 1100, 1.2)

        # Guard 2: Deposit pay-dirt into hopper if we have it and there's space
        paydirt_in_inv = inventory.inv_count(self.paydirt_name)
        space_left = self._hopper_space_left()
        
        if paydirt_in_inv > 0 and isinstance(space_left, int) and space_left > 0:
            # We have pay-dirt and hopper has space - deposit it
            res = objects.click_object_in_area_action_auto_prefer_no_camera(
                "Hopper",
                area=self.ore_vein_area,
                prefer_action="Deposit",
                exact_match_object=True,
                exact_match_target_and_action=True,
            )
            if res:
                # Wait for pay-dirt to decrease
                before = paydirt_in_inv
                if wait_until(lambda: inventory.inv_count(self.paydirt_name) < before, max_wait_ms=12000):
                    return exponential_number(400, 1100, 1.2)
            return exponential_number(400, 1100, 1.2)

        # Guard 3: Check if sack is empty - transition to BANK
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd <= 0:
            # Sack empty - go to BANK phase to bank and prepare inventory
            self.state["last_mlm_mining_ts"] = None
            self.set_phase("BANK")
            return exponential_number(400, 1100, 1.2)

        # Guard 5: Collect from sack
        before_slots = inventory.get_empty_slots_count()
        res = objects.click_object_closest_by_distance_prefer_no_camera(
            "Sack",
            action="Search",
            types=["GROUND"],
            exact_match_object=False,
            exact_match_target_and_action=False,
        )
        if not res:
            return exponential_number(0.35, 1.1, 1.3)

        # Guard 6: Verify items collected
        if not wait_until(lambda: inventory.get_empty_slots_count() != before_slots, max_wait_ms=6000):
            return exponential_number(0, 1.1, 1.3)
        
        return exponential_number(0.35, 1.1, 1.3)

    def _handle_done(self) -> int:
        sleep_exponential(2.0, 6.0, 1.2)
        return 1500


