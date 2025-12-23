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

from actions import bank, inventory, player, wait_until, objects, equipment
from actions.chat import can_continue
from actions.travel import in_area, go_to
from helpers.keyboard import press_spacebar
from helpers.utils import sleep_exponential, exponential_number
from helpers.runtime_utils import ipc

from .base import Plan
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera


class MotherlodeMinePlan(Plan):
    id = "MOTHERLODE_MINE"
    label = "Mining: Motherlode Mine"
    DONE = 0

    # ---- State keys ----
    _S_LAST_MLM_TS = "last_mlm_mining_ts"
    _S_LAST_MINE_CLICK_TS = "last_mine_click_ts"
    _S_CUR_VEIN_TILE = "current_ore_vein_tile"          # actual (inferred during mining)
    _S_LAST_TARGET_VEIN_TILE = "last_target_ore_vein_tile"  # intended click
    _S_DEPOSIT_RETRY = "deposit_retry"
    _S_MINE_PROGRESS_TS = "mine_progress_ts"
    _S_MINE_LIMBO_COUNT = "mine_limbo_count"

    def __init__(self):
        # Start by going to MLM (per your request)
        self.state = {"phase": "BANK"}
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
        self.mlm_bank_area = (3753, 3761, 5664, 5668)

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

        # Optional: recovery area to walk to if repeated hopper clicks don't register.
        # Defaults to the mining/working area, but you can make this a tight rect around the Hopper.
        self.hopper_area = self.ore_vein_area

        # Mining animation "jitter" compensation:
        # sometimes the player briefly idles between swings while still mining the same vein.
        # If we saw MLM_MINING within this window, treat the player as still mining.
        self.mining_anim_grace_s = 1.4
        self.state.setdefault(self._S_LAST_MLM_TS, None)  # monotonic timestamp
        self.state.setdefault(self._S_LAST_MINE_CLICK_TS, None)
        self.state.setdefault(self._S_CUR_VEIN_TILE, None)
        self.state.setdefault(self._S_LAST_TARGET_VEIN_TILE, None)
        self.state.setdefault(self._S_DEPOSIT_RETRY, 0)
        self.state.setdefault(self._S_MINE_PROGRESS_TS, None)
        self.state.setdefault(self._S_MINE_LIMBO_COUNT, 0)

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

    # --- Generic helpers ---
    def _player_world(self) -> dict | None:
        """
        Return the IPC player dict or None.
        """
        resp = ipc.get_player() or {}
        if not resp.get("ok"):
            return None
        pl = resp.get("player") or {}
        return pl if isinstance(pl, dict) else None

    def _player_tile(self) -> tuple[int, int, int] | None:
        """
        Return (x, y, plane) for the local player, or None.
        """
        pl = self._player_world()
        if not pl:
            return None
        px, py = pl.get("worldX"), pl.get("worldY")
        if not (isinstance(px, int) and isinstance(py, int)):
            return None
        pp = pl.get("plane", pl.get("worldP", 0))
        try:
            p = int(pp) if pp is not None else 0
        except Exception:
            p = 0
        return int(px), int(py), int(p)

    def _open_bank_chest(self, *, max_wait_ms: int = 4000) -> bool:
        """
        Ensure the bank chest interface is open.
        Returns True if bank is open (or became open), else False.
        """
        if bank.is_open():
            return True
        opened = objects.click_object_closest_by_distance_prefer_no_camera(
            "Bank chest",
            action="Use",
            exact_match_object=False,
            exact_match_target_and_action=False,
        )
        if opened:
            wait_until(bank.is_open, max_wait_ms=max_wait_ms)
        return bank.is_open()

    # --- Motherlode HUD helpers (keep these simple and reuse everywhere) ---
    def _widget_text(self, widget_id: int) -> str | None:
        """
        Best-effort widget text fetch via IPC.
        """
        try:
            resp = ipc.get_widget_info(int(widget_id)) or {}
            if not resp.get("ok"):
                return None
            w = resp.get("widget") or {}
            txt = w.get("text")
            return txt if isinstance(txt, str) else None
        except Exception:
            return None

    def _widget_int(self, widget_id: int) -> int | None:
        """
        Extract the first integer from a widget's text (e.g. '87' or 'Space: 21').
        """
        txt = self._widget_text(widget_id)
        if not txt:
            return None
        m = re.search(r"(\d+)", txt)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _sack_paydirt_remaining(self) -> int | None:
        return self._widget_int(self.sack_paydirt_widget_id)

    def _hopper_space_left(self) -> int | None:
        # "Space: N"
        return self._widget_int(self.sack_space_widget_id)

    # --- Mining helpers ---
    @staticmethod
    def _is_other_player_mlm_mining(anim: object) -> bool:
        """
        Treat any 67xx animation as MLM mining for other players (pickaxe-dependent).
        """
        try:
            a = int(anim)
        except Exception:
            return False
        return 6700 <= a < 6800

    @staticmethod
    def _is_local_mlm_mining(anim_val: object) -> bool:
        """
        Local player: accept string aliases ("MLM_MINING"/"MINING") and 67xx ids.
        """
        if anim_val == "MLM_MINING" or anim_val == "MINING":
            return True
        try:
            a = int(anim_val)
        except Exception:
            return False
        return 6700 <= a < 6800

    def _tile_has_exact_action(self, tile: tuple[int, int, int], object_name: str, action_name: str) -> bool:
        """
        True if the specified tile contains an object with EXACT name match and EXACT action match.
        (Avoids confusing "Mine" with "Examine" etc.)
        """
        tx, ty, tp = tile
        resp = ipc.get_object_at_tile(x=int(tx), y=int(ty), plane=int(tp), name=None) or {}
        if not resp.get("ok"):
            return False
        want_obj = (object_name or "").strip().lower()
        want_act = (action_name or "").strip().lower()
        for obj in (resp.get("objects") or []):
            nm = (obj.get("name") or "").strip().lower()
            if nm != want_obj:
                continue
            for a in (obj.get("actions") or []):
                if (a or "").strip().lower() == want_act:
                    return True
        return False

    def _infer_adjacent_ore_vein_tile(self) -> tuple[int, int, int] | None:
        """
        While we're actively mining, infer the actual vein by selecting the closest nearby "Ore vein"
        adjacent to the player's tile. This avoids menu ambiguity when multiple "Mine" entries exist.
        """
        pt = self._player_tile()
        if not pt:
            return None
        px, py, plane = pt

        resp = ipc.get_objects(self.ore_vein_names[0], types=["WALL"], radius=3) or {}
        if not resp.get("ok"):
            return None

        best: tuple[int, int, int] | None = None
        best_d = 999
        for o in (resp.get("objects") or []):
            w = o.get("world") or {}
            ox, oy = w.get("x"), w.get("y")
            op = w.get("p", w.get("plane", 0))
            if not (isinstance(ox, int) and isinstance(oy, int)):
                continue
            try:
                opi = int(op) if op is not None else 0
            except Exception:
                opi = 0
            if opi != plane:
                continue

            dx = abs(int(ox) - int(px))
            dy = abs(int(oy) - int(py))
            if dx > 2 or dy > 2:
                continue
            d = dx + dy
            if d < best_d:
                best_d = d
                best = (int(ox), int(oy), int(opi))
        return best

    def _clear_current_vein(self) -> None:
        self.state[self._S_CUR_VEIN_TILE] = None

    def _set_current_vein(self, tile: tuple[int, int, int] | None) -> None:
        if tile:
            self.state[self._S_CUR_VEIN_TILE] = (int(tile[0]), int(tile[1]), int(tile[2]))

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
        # Ensure we are near the MLM bank chest area first.
        if not in_area(self.mlm_bank_area):
            go_to(self.mlm_bank_area)
            return exponential_number(350, 2500, 0.5)

        # Open bank by clicking the bank chest (NOT actions.bank.open_bank(), which prefers booths/bankers).
        if not self._open_bank_chest(max_wait_ms=4000):
            return exponential_number(600, 2000, 0.8)

        # We are in bank (bank chest).
        # 1) Ensure we have the best pickaxe for our mining level, and equip it if possible.
        pickaxe_info = self._ensure_best_pickaxe(bank_open=True)
        if pickaxe_info:
            nm, _att_req, can_equip = pickaxe_info
            if can_equip and (not equipment.has_equipped(nm)) and inventory.has_item(nm):
                # Close bank before equipping (more reliable than interacting through bank UI).
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
                self._equip_pickaxe_if_possible(pickaxe_info)
                self.set_phase("MINE")
                return self.loop_interval_ms

        # 2) Withdraw any other required items (optional).
        for it in (self.required_items or []):
            if not isinstance(it, dict):
                continue
            nm = it.get("name")
            qty = int(it.get("quantity", 1) or 1)
            if not nm:
                continue
            if qty > 0 and inventory.inv_count(nm) >= qty:
                continue
            if qty == -1:
                bank.withdraw_item(nm, withdraw_all=True)
            else:
                need = max(0, qty - inventory.inv_count(nm))
                for _ in range(need):
                    bank.withdraw_item(nm, withdraw_x=1)

        # Close and continue loop.
        bank.close_bank()
        wait_until(bank.is_closed, max_wait_ms=3000)
        self.set_phase("MINE")
        return self.loop_interval_ms

    def _handle_go_to_mlm(self) -> int:
        if not in_area(self.mlm_area):
            go_to(self.mlm_area)
            return self.loop_interval_ms
        # Once inside MLM, go bank first to sync inventory for the loop.
        self.set_phase("BANK")
        return self.loop_interval_ms

    def _handle_mine(self) -> int:
        if can_continue():
            press_spacebar()
            return exponential_number(200, 2000, 1)

        # Hard guard: if the sack is full, the game blocks further mining until you collect.
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd >= int(self.sack_full_pay_dirt):
            self.set_phase("COLLECT")
            return self.loop_interval_ms

        # If inventory is full, deposit pay-dirt.
        paydirt_in_inv = inventory.inv_count(self.paydirt_name)
        space_left = self._hopper_space_left()

        # Deposit if:
        # - inventory is full, OR
        # - pay-dirt in inventory >= hopper remaining space (Space: N)
        if inventory.is_full() or (isinstance(space_left, int) and space_left >= 0 and paydirt_in_inv > 0 and paydirt_in_inv >= space_left):
            self.set_phase("DEPOSIT")
            return self.loop_interval_ms

        anim = player.get_player_animation()
        now = time.monotonic()
        pl_now = self._player_world() or {}
        is_interacting_now = bool(pl_now.get("isInteracting"))

        # Track "progress" so we can recover from any limbo state.
        # Progress means: we are mining, or we are currently interacting with something.
        if self._is_local_mlm_mining(anim) or is_interacting_now:
            self.state[self._S_MINE_PROGRESS_TS] = now
            self.state[self._S_MINE_LIMBO_COUNT] = 0

        # If already mining, idle a bit (and keep the "last mining" timestamp fresh).
        if self._is_local_mlm_mining(anim):
            self.state[self._S_LAST_MLM_TS] = now
            self._set_current_vein(self._infer_adjacent_ore_vein_tile())
            return exponential_number(150, 6000, 1)

        # Jitter compensation: if the animation briefly drops out but we were mining very recently,
        # don't try to re-click a vein yet.
        last_ts = self.state.get(self._S_LAST_MLM_TS)
        if isinstance(last_ts, (int, float)) and (now - float(last_ts)) < float(self.mining_anim_grace_s):
            return exponential_number(150, 1200, 1.2)

        # Extra guard against spam-clicking: if we recently clicked a vein, don't click again immediately.
        last_click_ts = self.state.get(self._S_LAST_MINE_CLICK_TS)
        if isinstance(last_click_ts, (int, float)) and (now - float(last_click_ts)) < 2.5:
            return exponential_number(180, 1200, 1.2)

        # If we have a current *actual* vein tile remembered (set only while in mining anim),
        # and we're still adjacent + interacting, wait a bit instead of re-clicking.
        cur_tile = self.state.get(self._S_CUR_VEIN_TILE)
        if isinstance(cur_tile, (tuple, list)) and len(cur_tile) == 3:
            try:
                ct = (int(cur_tile[0]), int(cur_tile[1]), int(cur_tile[2]))
                px, py = pl_now.get("worldX"), pl_now.get("worldY")

                if not (isinstance(px, int) and isinstance(py, int)):
                    self._clear_current_vein()
                elif abs(int(px) - ct[0]) > 2 or abs(int(py) - ct[1]) > 2:
                    self._clear_current_vein()
                elif is_interacting_now and self._tile_has_exact_action(ct, self.ore_vein_names[0], self.ore_vein_action):
                    # We are still interacting and this is still a mineable vein; don't spam new clicks.
                    return exponential_number(220, 1600, 1.25)
                else:
                    self._clear_current_vein()
            except Exception:
                self._clear_current_vein()

        # Watchdog: if we're idle/not interacting for too long, force a reset so we can't get stuck.
        prog_ts = self.state.get(self._S_MINE_PROGRESS_TS)
        idle_s = (now - float(prog_ts)) if isinstance(prog_ts, (int, float)) else None
        if (idle_s is not None) and idle_s > 18.0:
            self._clear_current_vein()
            # Allow clicking again immediately (override click cooldown if we were stuck).
            self.state[self._S_LAST_MINE_CLICK_TS] = None
            self.state[self._S_LAST_MLM_TS] = None
            self.state[self._S_MINE_PROGRESS_TS] = now
            cnt = int(self.state.get(self._S_MINE_LIMBO_COUNT) or 0) + 1
            self.state[self._S_MINE_LIMBO_COUNT] = cnt

            # Escalate: on repeated limbo, re-walk into the mining area to re-acquire objects.
            if cnt >= 2:
                go_to(self.ore_vein_area)
                return exponential_number(450, 1800, 1.2)
            return exponential_number(220, 1200, 1.25)

        # Mine ore veins only within the configured area, but try to avoid veins
        # another player is already mining.
        #
        # Heuristic:
        # - Get nearby WALL ore veins from IPC and keep only those inside ore_vein_area.
        # - Get nearby players; if any non-local player is adjacent to a vein AND is in a mining animation,
        #   treat that vein as "busy" and skip it.
        # - Click the nearest non-busy vein; if none found, fall back to the generic in-area click.
        chosen_tile = None
        try:
            obj_resp = ipc.get_objects(self.ore_vein_names[0], types=["WALL"], radius=26) or {}
            objs = obj_resp.get("objects", []) if obj_resp.get("ok") else []

            min_x, max_x, min_y, max_y = self.ore_vein_area
            candidates = []
            for o in objs:
                w = o.get("world", {}) or {}
                ox, oy = w.get("x"), w.get("y")
                if not (isinstance(ox, int) and isinstance(oy, int)):
                    continue
                if ox < int(min_x) or ox > int(max_x) or oy < int(min_y) or oy > int(max_y):
                    continue
                candidates.append(o)

            def _dist(o):
                try:
                    return float(o.get("distance", 9999))
                except Exception:
                    return 9999.0
            candidates.sort(key=_dist)

            players_resp = ipc.get_players() or {}
            players = players_resp.get("players", []) if players_resp.get("ok") else []

            def _is_busy(ox: int, oy: int) -> bool:
                for p in players:
                    if p.get("isLocalPlayer"):
                        continue
                    anim = p.get("animation")
                    if not self._is_other_player_mlm_mining(anim):
                        continue
                    px, py = p.get("worldX"), p.get("worldY")
                    if not (isinstance(px, int) and isinstance(py, int)):
                        continue
                    if abs(int(px) - int(ox)) <= 1 and abs(int(py) - int(oy)) <= 1:
                        return True
                return False

            for o in candidates:
                w = o.get("world", {}) or {}
                ox, oy = int(w.get("x")), int(w.get("y"))
                op = int(w.get("p", w.get("plane", 0)) or 0)
                if _is_busy(ox, oy):
                    continue
                chosen_tile = (ox, oy, op)
                break
        except Exception:
            chosen_tile = None

        if chosen_tile:
            ox, oy, op = chosen_tile
            res = objects.click_object_in_area_action_auto_prefer_no_camera(
                self.ore_vein_names[0],
                area=(ox, ox, oy, oy),
                prefer_action=self.ore_vein_action,
                types=["WALL"],
                exact_match_object=True,
                exact_match_target_and_action=True,
            )
        else:
            res = objects.click_object_in_area_action_auto_prefer_no_camera(
                self.ore_vein_names[0],
                area=self.ore_vein_area,
                prefer_action=self.ore_vein_action,
                types=["WALL"],
                exact_match_object=True,
                exact_match_target_and_action=True,
            )
        if res:
            self.state[self._S_LAST_MINE_CLICK_TS] = now
            if chosen_tile:
                self.state[self._S_LAST_TARGET_VEIN_TILE] = chosen_tile
            # A successful click counts as progress (even if animation starts slightly later).
            self.state[self._S_MINE_PROGRESS_TS] = now
            wait_until(lambda: self._is_local_mlm_mining(player.get_player_animation()), max_wait_ms=7000)
        return self.loop_interval_ms

    def _handle_deposit(self) -> int:
        """
        Deposit Pay-dirt into the Hopper.
        This is a placeholder: MLM typically uses the hopper to process pay-dirt.
        """
        # If we somehow have no pay-dirt, go collect (maybe sack is ready) or mine again.
        if not inventory.has_item(self.paydirt_name):
            self.state[self._S_DEPOSIT_RETRY] = 0
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

        # If that deposit filled the sack, go collect immediately (mining will be blocked otherwise).
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd >= int(self.sack_full_pay_dirt):
            self.set_phase("COLLECT")
            return self.loop_interval_ms

        # After a successful deposit, re-check hopper space. If it's getting low, go collect from sack.
        # Widget 25034758 example: "Space: 21"
        space_left = self._hopper_space_left()
        if isinstance(space_left, int) and space_left < 27:
            self.set_phase("COLLECT")

        # self.set_phase("COLLECT")
        return self.loop_interval_ms

    def _handle_collect(self) -> int:
        """
        Collect processed ores from the Sack.
        """
        # --- Simple Collect logic (per your request) ---
        # 1) If inventory has ores/gems/nuggets, bank them.
        # 2) Else, if sack still has pay-dirt remaining, search the sack.
        # 3) Else (no pay-dirt remaining AND no ores in inventory), go back to mining.

        has_ores = inventory.has_any_items(self.collect_bank_item_names)
        if has_ores:
            if not self._open_bank_chest(max_wait_ms=7000):
                # If bank chest isn't clickable from here, walk to the chest area and try again.
                if not in_area(self.mlm_bank_area):
                    go_to(self.mlm_bank_area)
                return exponential_number(450, 1800, 1.2)

            bank.deposit_inventory()
            wait_until(lambda: not inventory.has_any_items(self.collect_bank_item_names), max_wait_ms=3000)
            sleep_exponential(0.35, 1.5, 1)

            # Only re-withdraw the pickaxe on the FINAL banking pass before we return to mining.
            # (If pay-dirt remaining is 0, we're done collecting and the next phase is MINE.)
            pickaxe_info = None
            sack_pd = self._sack_paydirt_remaining()
            if isinstance(sack_pd, int) and sack_pd <= 0:
                pickaxe_info = self._ensure_best_pickaxe(bank_open=True)

            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=3000)

            # Optional: equip it if we can (prevents accidental banking next time).
            self._equip_pickaxe_if_possible(pickaxe_info)
            return self.loop_interval_ms

        # Determine whether there's still pay-dirt in the sack.
        sack_pd = self._sack_paydirt_remaining()
        if isinstance(sack_pd, int) and sack_pd <= 0:
            self.set_phase("MINE")
            return self.loop_interval_ms

        # Search the sack (we believe pay-dirt remains, or widget is unknown).
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

        got_items = wait_until(lambda: inventory.get_empty_slots_count() != before_slots, max_wait_ms=6000)
        if not got_items:
            # No change after searching: likely empty or already collected, return to mining.
            # (We still rely on the pay-dirt widget as the authoritative source.)
            self.set_phase("MINE")
        return self.loop_interval_ms

    def _handle_done(self) -> int:
        sleep_exponential(2.0, 6.0, 1.2)
        return 1500


