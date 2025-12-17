#!/usr/bin/env python3
"""
Attack NPCs Utility Plan
========================

This is a utility plan that can be used by other plans to attack NPCs at a specific location.
It handles the complete combat cycle: travel, attack, loot, eat food, and return to bank when full.

Usage:
    attack_plan = AttackNpcsPlan(
        target_area="FALADOR_COWS",
        npc_names=["Cow", "Cow calf"],
        loot_item="Cowhide",
        food_item="Trout",
        bank_area="FALADOR_BANK",
        return_when_full=True
    )
    
    # In your plan loop:
    status = attack_plan.loop(ui)
    if status == AttackNpcsPlan.SUCCESS:
        print("Attack session complete!")
    elif status == AttackNpcsPlan.ATTACKING:
        print("Still attacking NPCs...")
    elif status == AttackNpcsPlan.RETURNING_TO_BANK:
        print("Returning to bank...")
"""

import logging
from pathlib import Path
from typing import List

# Add the parent directory to the path for imports
import sys

from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.utils import sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan
from actions import combat, inventory, player, chat
from actions import loot
from actions import wait_until
from actions.travel import go_to, in_area


class AttackNpcsPlan(Plan):
    """Utility plan for attacking NPCs at a specific location."""
    
    id = "ATTACK_NPCS_PLAN"
    label = "Attack NPCs Utility"
    
    # Return status codes
    SUCCESS = 0
    TRAVELING = 1
    ATTACKING = 2
    ERROR = 3
    WAITING = 4
    
    def __init__(self, 
                 target_area: str,
                 npc_names: List[str],
                 loot_item: str = None,
                 food_item: str = "Trout",
                 health_threshold: int = 7):
        """
        Initialize the attack NPCs plan.
        
        Args:
            target_area: Area name to travel to (e.g., "FALADOR_COWS")
            npc_names: List of NPC names to attack (e.g., ["Cow", "Cow calf"])
            loot_item: Item to loot from ground (optional)
            food_item: Food item to eat when low health
            health_threshold: Health threshold for eating food
        """
        self.state = {"phase": "TRAVEL_TO_TARGET"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration
        self.target_area = target_area
        self.npc_names = npc_names
        self.loot_item = loot_item
        self.food_item = food_item
        self.health_threshold = health_threshold
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # State tracking
        self.attack_complete = False
        self.error_message = None
        
        logging.info(f"[{self.id}] Attack NPCs plan initialized")
        logging.info(f"[{self.id}] Target area: {self.target_area}")
        logging.info(f"[{self.id}] NPCs to attack: {self.npc_names}")
        logging.info(f"[{self.id}] Loot item: {self.loot_item}")
        logging.info(f"[{self.id}] Food: {self.food_item}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        if self.attack_complete:
            return self.SUCCESS
        
        if self.error_message:
            return self.ERROR
        
        phase = self.state.get("phase", "TRAVEL_TO_TARGET")
        logging.debug(f"[{self.id}] Current phase: {phase}")

        match(phase):
            case "TRAVEL_TO_TARGET":
                return self._handle_travel_to_target()

            case "ATTACK_NPCS":
                return self._handle_attack_npcs()

            case "COMPLETE":
                logging.info(f"[{self.id}] Attack session completed successfully!")
                self.attack_complete = True
                return self.SUCCESS

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
    def _handle_travel_to_target(self) -> int:
        """Handle traveling to the target area."""
        if not in_area(self.target_area):
            logging.info(f"[{self.id}] Traveling to {self.target_area}...")
            go_to(self.target_area)
            return self.TRAVELING
        else:
            logging.info(f"[{self.id}] Arrived at {self.target_area}")
            self.set_phase("ATTACK_NPCS")
            return self.loop_interval_ms
    
    def _handle_attack_npcs(self) -> int:
        """Handle attacking NPCs and looting."""
        if player.get_run_energy() > 2000 and not player.is_run_on():
            player.toggle_run()
        # Select appropriate combat style for training
        try:
            from actions import select_combat_style_for_training
            select_combat_style_for_training()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not select combat style: {e}")
        
        # Check if inventory is full - if so, just continue attacking (let calling plan handle banking)
        empty_slots = inventory.get_empty_slots_count()
        if empty_slots == 0:
            logging.debug(f"[{self.id}] Inventory full, continuing to attack (banking handled by calling plan)")
        
        # Check health and eat food if needed
        current_health = player.get_health()
        max_health = player.get_skill_level("hitpoints")
        
        if current_health is None or max_health is None:
            logging.warning(f"[{self.id}] Could not get health values - current: {current_health}, max: {max_health}")
            return self.loop_interval_ms
        
        if current_health <= max_health - self.health_threshold:
            logging.info(f"[{self.id}] Low health ({current_health}/{max_health}), eating {self.food_item}")
            inventory.interact(self.food_item, "Eat")
            return self.ATTACKING
        
        # Handle chat dialogues
        if chat.can_continue():
            chat.continue_dialogue()
            return self.ATTACKING
        
        # Wait for combat to finish if in combat
        if player.is_in_combat():
            wait_until(lambda: not player.is_in_combat())
            return self.ATTACKING
        
        # Loot items if specified and there's space
        if self.loot_item and empty_slots > 0:
            loot_count_before = inventory.inv_count(self.loot_item)
            if loot(self.loot_item, radius=4):
                wait_until(lambda: inventory.inv_count(self.loot_item) == loot_count_before + 1, max_wait_ms=3000)
                return self.ATTACKING
        
        # Attack NPCs if not in combat
        if not player.is_in_combat():
            logging.debug(f"[{self.id}] Attacking NPCs: {self.npc_names}")
            combat.attack_closest(self.npc_names)
            return self.ATTACKING
        
        # Small delay to prevent excessive CPU usage
        sleep_exponential(0.8, 1.5, 1.0)
        return self.ATTACKING
    
    
    def is_complete(self) -> bool:
        """Check if the attack session is complete."""
        return self.attack_complete
    
    def get_error_message(self) -> str:
        """Get the current error message."""
        return self.error_message
    
    def reset(self):
        """Reset the plan to initial state."""
        self.state = {"phase": "TRAVEL_TO_TARGET"}
        self.attack_complete = False
        self.error_message = None
        logging.info(f"[{self.id}] Plan reset to initial state")
    
    def set_target_area(self, area: str):
        """Change the target area."""
        self.target_area = area
        logging.info(f"[{self.id}] Target area changed to: {area}")
    
    def set_npc_names(self, npc_names: List[str]):
        """Change the NPC names to attack."""
        self.npc_names = npc_names
        logging.info(f"[{self.id}] NPC names changed to: {npc_names}")


# Helper functions for easy setup
def create_attack_plan(target_area: str,
                      npc_names: List[str],
                      loot_item: str = None,
                      food_item: str = "Trout") -> AttackNpcsPlan:
    """
    Create an attack plan with custom configuration.
    
    Args:
        target_area: Area name to travel to
        npc_names: List of NPC names to attack
        loot_item: Item to loot from ground (optional)
        food_item: Food item to eat when low health
    
    Returns:
        Configured AttackNpcsPlan instance
        
    Example:
        # Attack cows and collect hides
        attack_plan = create_attack_plan(
            target_area="FALADOR_COWS",
            npc_names=["Cow", "Cow calf"],
            loot_item="Cowhide",
            food_item="Trout"
        )
    """
    return AttackNpcsPlan(
        target_area=target_area,
        npc_names=npc_names,
        loot_item=loot_item,
        food_item=food_item
    )


def create_cow_attack_plan() -> AttackNpcsPlan:
    """
    Create a pre-configured plan for attacking cows.
    
    Returns:
        Configured AttackNpcsPlan instance for cows
    """
    return AttackNpcsPlan(
        target_area="FALADOR_COWS",
        npc_names=["Cow", "Cow calf"],
        loot_item="Cowhide",
        food_item="Trout"
    )
