#!/usr/bin/env python3
"""
Guardians of the Rift Plan (scaffolded)
========================================

Guardians of the Rift is a Runecrafting minigame:
- PREPARE: bank and get required items (essence, runes, etc.)
- ENTER_GAME: travel to and enter the rift area
- GAME_LOOP: play the minigame
  - Collect essence fragments
  - Charge barriers with runes
  - Craft runes at altars
  - Defend the Great Guardian
- COLLECT_REWARDS: exit and collect rewards
- Loop back to PREPARE for next game

TODO: Fill in the blanks:
- Bank area/location
- Rift entrance location
- Required items (essence, runes, tools)
- Object names/actions (barriers, altars, portals, etc.)
- Game mechanics (fragment collection, barrier charging, rune crafting)
"""

import logging
from pathlib import Path
import sys

from actions import bank, inventory, objects, player, wait_until
from actions.travel import go_to_tile, go_to
from helpers.utils import exponential_number, sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan


class GuardiansOfTheRiftPlan(Plan):
    id = "GUARDIANS_OF_THE_RIFT"
    label = "Runecrafting: Guardians of the Rift"
    description = """Plays the Guardians of the Rift Runecrafting minigame. Collects essence fragments, charges barriers with runes, crafts runes at altars, and defends the Great Guardian. Includes banking and reward collection phases.

Starting Area: Guardians of the Rift (Temple of the Eye)
Required Items: Runes (for barrier charging), Essence (for rune crafting)"""

    def __init__(self):
        self.state = {"phase": "PREPARE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # ---- Config: Fill these in ----
        # Bank area/location
        self.bank_area = None  # TODO: Set bank area (e.g., "EDGEVILLE_BANK" or specific coordinates)
        
        # Rift entrance location
        self.rift_entrance_tile = (None, None, 0)  # TODO: (x, y, plane) for rift entrance
        self.rift_area = None  # TODO: Area name or rect tuple for the rift game area
        
        # Required items
        self.essence_item = None  # TODO: e.g., "Rune essence", "Pure essence"
        self.rune_items = []  # TODO: List of runes needed for barrier charging (e.g., ["Air rune", "Water rune"])
        self.tool_items = []  # TODO: Any tools needed (e.g., ["Chisel"])
        
        # Object names/actions
        self.barrier_names = ["Barrier"]  # TODO: Adjust if needed
        self.barrier_action = "Charge"  # TODO: May need "Use" or other actions
        
        self.altar_names = []  # TODO: Altar names (e.g., ["Air Altar", "Water Altar"])
        self.altar_action = "Enter"  # TODO: Adjust if needed
        
        self.portal_names = ["Portal"]  # TODO: Portal names for entering/exiting
        self.portal_action = "Enter"  # TODO: Adjust if needed
        
        self.fragment_names = ["Essence fragment"]  # TODO: Fragment object names
        self.fragment_action = "Take"  # TODO: Adjust if needed
        
        # Game state tracking
        self.in_game = False  # TODO: Track if we're currently in an active game
        
        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase

    def loop(self, ui) -> int:
        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        phase = self.state.get("phase", "PREPARE")

        match phase:
            case "PREPARE":
                return self._handle_prepare()
            case "ENTER_GAME":
                return self._handle_enter_game()
            case "GAME_LOOP":
                return self._handle_game_loop()
            case "COLLECT_REWARDS":
                return self._handle_collect_rewards()
            case "DONE":
                logging.info(f"[{self.id}] Plan completed")
                return exponential_number(5000, 10000, 1.2)

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_prepare(self) -> int:
        """
        PREPARE phase: bank and get required items for the minigame.
        TODO: Implement banking logic:
        - Open bank
        - Deposit inventory
        - Withdraw essence, runes, tools
        - Close bank
        - Transition to ENTER_GAME
        """
        # TODO: Implement bank logic
        # Example structure:
        # if not bank.is_open():
        #     bank.open_bank(prefer="bank chest")
        #     wait_until(bank.is_open, max_wait_ms=3000)
        #     return exponential_number(300, 800, 1.2)
        # 
        # # Deposit inventory
        # bank.deposit_inventory()
        # 
        # # Withdraw items
        # if self.essence_item:
        #     bank.withdraw_item(self.essence_item, withdraw_all=True)
        # for rune in self.rune_items:
        #     bank.withdraw_item(rune, withdraw_x=50)  # Adjust quantity as needed
        # for tool in self.tool_items:
        #     bank.withdraw_item(tool, withdraw_x=1)
        # 
        # bank.close_bank()
        # wait_until(bank.is_closed, max_wait_ms=3000)
        # self.set_phase("ENTER_GAME")
        # return exponential_number(300, 800, 1.2)
        
        logging.warning(f"[{self.id}] PREPARE phase not implemented")
        return self.loop_interval_ms

    def _handle_enter_game(self) -> int:
        """
        ENTER_GAME phase: travel to and enter the rift area.
        TODO: Implement entrance logic:
        - Travel to rift entrance
        - Enter the rift (click portal/entrance)
        - Wait until in game area
        - Transition to GAME_LOOP
        """
        x, y, plane = self.rift_entrance_tile
        if x is None or y is None:
            logging.error(f"[{self.id}] Rift entrance tile not configured")
            self.set_phase("DONE")
            return self.loop_interval_ms
        
        # TODO: Implement entrance logic
        # Example structure:
        # px, py = player.get_player_position() or (None, None)
        # if px != x or py != y:
        #     go_to_tile(x, y, plane=plane, arrive_radius=2)
        #     return exponential_number(1.0, 3.0, 1.2)
        # 
        # # Enter the rift
        # # TODO: Click portal/entrance object
        # # TODO: Wait until in game area (check position or game state)
        # 
        # self.in_game = True
        # self.set_phase("GAME_LOOP")
        # return exponential_number(500, 1500, 1.2)
        
        logging.warning(f"[{self.id}] ENTER_GAME phase not implemented")
        return self.loop_interval_ms

    def _handle_game_loop(self) -> int:
        """
        GAME_LOOP phase: play the minigame.
        TODO: Implement game mechanics:
        - Collect essence fragments from the ground
        - Charge barriers with runes when needed
        - Craft runes at altars
        - Defend the Great Guardian
        - Check if game is over (barriers destroyed or game won)
        - Transition to COLLECT_REWARDS when game ends
        """
        # TODO: Implement game loop logic
        # Example structure:
        # # Check if game is over
        # if not self.in_game:
        #     self.set_phase("COLLECT_REWARDS")
        #     return exponential_number(300, 800, 1.2)
        # 
        # # Collect fragments
        # # TODO: Check for essence fragments on ground and collect them
        # 
        # # Charge barriers
        # # TODO: Check barrier health/status
        # # TODO: Charge barriers with appropriate runes
        # 
        # # Craft runes
        # # TODO: Use essence at altars to craft runes
        # # TODO: Use crafted runes to charge barriers
        # 
        # # Check game state
        # # TODO: Determine if game is still active or has ended
        # 
        # return exponential_number(300, 800, 1.2)
        
        logging.warning(f"[{self.id}] GAME_LOOP phase not implemented")
        return self.loop_interval_ms

    def _handle_collect_rewards(self) -> int:
        """
        COLLECT_REWARDS phase: exit the rift and collect rewards.
        TODO: Implement reward collection:
        - Exit the rift area (click portal/exit)
        - Collect rewards (if there's a reward interface)
        - Transition back to PREPARE for next game
        """
        # TODO: Implement reward collection logic
        # Example structure:
        # # Exit the rift
        # # TODO: Click exit portal/object
        # # TODO: Wait until outside rift area
        # 
        # # Collect rewards
        # # TODO: Check for reward interface/widget
        # # TODO: Collect rewards if available
        # 
        # self.in_game = False
        # self.set_phase("PREPARE")
        # return exponential_number(500, 1500, 1.2)
        
        logging.warning(f"[{self.id}] COLLECT_REWARDS phase not implemented")
        return self.loop_interval_ms

