# tutorial_island.py
import time
import random
import logging
from ..helpers.runtime_utils import dispatch

# Action method imports
import ilbot.ui.simple_recorder.actions.travel as trav
import ilbot.ui.simple_recorder.actions.tab as tab
import ilbot.ui.simple_recorder.actions.bank as bank
import ilbot.ui.simple_recorder.actions.chat as chat
import ilbot.ui.simple_recorder.actions.inventory as inventory
import ilbot.ui.simple_recorder.actions.npc as npc
import ilbot.ui.simple_recorder.actions.objects as objects
import ilbot.ui.simple_recorder.actions.player as player
import ilbot.ui.simple_recorder.actions.widgets as widgets
from ..actions import equipment, spellbook

# Specific function imports
from ..actions.timing import wait_until
from ..actions.chat import can_click_continue_widget, click_continue_widget
from ..actions.tutorial import type_tutorial_name, click_tutorial_set_name, click_tutorial_lookup_name
from .base import Plan
from ..helpers import quest
from ..helpers.tab import tab_exists
from ..helpers.widgets import get_widget_text, rect_center_from_widget, get_character_design_widgets, get_character_design_main, get_character_design_button_realtime, get_all_character_design_buttons, widget_exists, character_design_widget_exists, get_widget_info
from ..helpers.utils import press_enter, press_backspace, press_esc, press_spacebar, clean_rs

class TutorialIslandPlan(Plan):
    id = "TUTORIAL_ISLAND"
    label = "Tutorial Island Completion"

    def __init__(self):
        self.state = {"phase": "START_TUTORIAL"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
        setup_camera_optimal()


    def set_phase(self, phase: str, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui):
        phase = self.state.get("phase", "START_TUTORIAL")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        match(phase):
            case "START_TUTORIAL": # WORKING! RUN WITHOUT BREAKPOINTS
                if widget_exists(44498948): # Character creation widget
                    self.set_phase("CHARACTER_CREATION")
                    return
                # Check if tutorial name input is available
                from ..actions.tutorial import is_tutorial_open, get_tutorial_name_input_bounds
                if is_tutorial_open():
                    name_input_bounds = get_tutorial_name_input_bounds()
                    if name_input_bounds:
                        # Check if we need to try a name or if we're waiting for status
                        
                        status_text = get_widget_text(36569101)  # TutorialDisplayname.STATUS
                        name_available = None
                        if status_text:
                            status_lower = clean_rs(status_text.lower())
                            if "is available!" in status_lower:
                                name_available = True
                            else:
                                name_available = False
                        if name_available is True:
                            # Name is available, click SET_NAME button
                            if click_tutorial_set_name():
                                logging.info("[TUTORIAL] SET_NAME button clicked, waiting for window to close...")
                                wait_until(lambda: widget_exists(44498948))
                                return
                            else:
                                logging.info("[TUTORIAL] Failed to click SET_NAME button")
                        else:
                            # No status yet, try initial generated name
                            from ..constants import generate_player_name
                            initial_name = generate_player_name()
                            if get_widget_text(36569100) == "*" or get_widget_text(36569100) == "":
                                type_tutorial_name(initial_name)
                                # Note: type_tutorial_name already presses Enter
                            if wait_until(lambda: initial_name in get_widget_text(36569100)): # text input box for display name
                                logging.info(f"[TUTORIAL] Character name entered: {initial_name}")
                                # Note: type_tutorial_name already pressed Enter, looking for confirmation...
                                # Wait for confirmation message or status update
                                if wait_until(lambda: initial_name in get_widget_text(36569101)):

                                    status_text = get_widget_text(36569101) # confirmatino message text
                                    logging.info(f"[TUTORIAL] Confirmation message received: {status_text}")
                                    
                                    if status_text and "not available" in status_text.lower():
                                        logging.info("[TUTORIAL] Name not available, trying new name...")
                                        # Backspace to clear the current name
                                        current_text = get_widget_text(36569100) or ""
                                        # Click on the name input field to focus it before backspacing
                                        name_input_bounds = get_tutorial_name_input_bounds()
                                        if name_input_bounds:
                                            x, y = rect_center_from_widget({"bounds": name_input_bounds})
                                            if x is not None and y is not None:
                                                # Click to focus the input field
                                                step = {
                                                    "action": "click-name-input-focus",
                                                    "click": {"type": "point", "x": x, "y": y},
                                                    "target": {"domain": "tutorial", "name": "name_input_focus"}
                                                }
                                                dispatch(step)
                                                time.sleep(0.5)  # Wait for focus
                                        # Now backspace to clear the field
                                        for _ in range(len(current_text)):
                                            press_backspace()
                                            time.sleep(0.05)
                                        # Wait a moment for backspacing to complete
                                        time.sleep(0.5)
                                        return
                                    elif status_text and "available" in status_text.lower():
                                        return
                                    else:
                                        logging.info(f"[TUTORIAL] Unknown status message: {status_text}")
                                else:
                                    logging.info("[TUTORIAL] No confirmation message found")
                            else:
                                logging.info("[TUTORIAL] Failed to enter character name")
                    else:
                        logging.info("[TUTORIAL] Name input not visible, waiting...")
                else:
                    npc.chat_with_npc("Gielinor Guide")
                    return
                return

            case "CHARACTER_CREATION": #WORKING! RUN WITHOUT BREAKPOINTS
                if not widget_exists(44498948):
                    self.set_phase("TALK_TO_GUIDE", ui)
                    return

                # Get all character design buttons via real-time IPC
                logging.info("[TUTORIAL] Attempting to get character design buttons...")
                design_buttons = get_all_character_design_buttons()
                logging.info(f"[TUTORIAL] get_all_character_design_buttons() returned: {design_buttons}")
                
                if design_buttons:
                    logging.info(f"[TUTORIAL] Found {len(design_buttons)} character design buttons via IPC")
                    
                    # Customize character appearance
                    from ilbot.ui.simple_recorder.actions.character import customize_character_appearance
                    customize_character_appearance(design_buttons)
                else:
                    logging.info("[TUTORIAL] No character design buttons found via IPC")
                
                # Move to next phase after customization
                if not wait_until(lambda: not widget_exists(44498948)):
                    return
                self.set_phase("TALK_TO_GUIDE", ui)
                return

            case "TALK_TO_GUIDE":
                if chat.dialogue_contains_text("Moving on"):
                    self.set_phase("SURVIVAL_INSTRUCTOR")
                    return
                # Check if we need to click "Click here to continue"
                if can_click_continue_widget():
                    logging.info("[TUTORIAL] Found 'Click here to continue' widget, clicking it...")
                    if click_continue_widget():
                        logging.info("[TUTORIAL] Successfully clicked continue widget")
                        return
                    else:
                        logging.info("[TUTORIAL] Failed to click continue widget")
                        return

                if chat.dialogue_contains_text("flashing spanner icon"):
                    if widget_exists(10747945): # settings menu tab
                        widgets.click_widget(10747945)
                        if not wait_until(lambda: tab.is_tab_open("SETTINGS")):
                            return
                    return

                # Try to find and chat with Gielinor Guide
                if npc.closest_npc_by_name("Gielinor guide"):
                    result = npc.chat_with_npc(
                        "Gielinor guide",
                        options=[
                            "I am an experienced player"
                        ]
                    )
                    return
                return

            case "SURVIVAL_INSTRUCTOR": # need serious improvement in logic here. revisit with walk through.
                if chat.dialogue_contains_text("can't light a fire here"):
                    trav.move_to_random_tile_near_player(3)
                    return
                if chat.can_continue():
                    press_spacebar()
                    return
                if inventory.has_item("Shrimps"):
                    self.set_phase("COOKING_TUTORIAL")
                    return
                if chat.dialogue_contains_text("Cooking"):
                    if objects.object_exists("Fire"):
                        raw_shrimps = inventory.inv_count("Raw shrimps")
                        inventory.use_item_on_object("Raw shrimps", "Fire")
                        if not wait_until(lambda: not player.get_player_animation() == "COOKING_ON_FIRE"):
                            return
                        wait_until(lambda: inventory.inv_count("Raw shrimps") == raw_shrimps - 1)
                        return
                if chat.dialogue_contains_text("Firemaking"):
                    if not player.get_player_animation() == "FIREMAKING":
                        if player.make_fire():
                            return
                        else:
                            logging.info("[TUTORIAL] Failed to make fire")
                            return
                if chat.dialogue_contains_text("Woodcutting"):
                    if not inventory.has_item("logs"):
                        if not player.get_player_animation() == "CHOPPING":
                            objects.click("Tree", "Chop down")
                            wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=3000)
                            if not wait_until(lambda: inventory.has_item("logs")):
                                return
                            return
                if chat.dialogue_contains_text("You've gained some experience"):
                    widgets.click_widget(10747957) # skills tab stone
                    return
                if chat.dialogue_contains_text("You've been given an item"):
                    widgets.click_widget(10747959) # inventory tab stone
                    return
                elif tab.is_tab_open("INVENTORY"):
                    if inventory.has_item("Small fishing net") and not inventory.has_item("Raw shrimps"):
                        if npc.closest_npc_by_name("Fishing spot") and not player.get_player_animation() == "NETTING":
                            npc.click_npc_action("Fishing spot", "Net")
                            if not wait_until(lambda: player.get_player_animation() == "NETTING"):
                                return
                        elif player.get_player_animation() == "NETTING":
                            wait_until(lambda: inventory.has_item("Raw shrimps"))
                            return
                    return
                elif inventory.has_items(["Bronze axe", "Tinderbox"]):
                    if not inventory.has_item("logs"):
                        if not player.get_player_animation() == "CHOPPING":
                            objects.click("Tree", "Chop down")
                            wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=3000)
                            return
                    elif not player.get_player_animation() == "FIREMAKING":
                        if player.make_fire():
                            return
                        else:
                            logging.info("[TUTORIAL] Failed to make fire")
                            return
                    else:
                        # Already firemaking, wait for it to complete
                        xp = player.get_skill_xp("firemaking")
                        wait_until(lambda: not player.get_skill_xp("firemaking") == xp, max_wait_ms=3000)
                        return

                elif npc.closest_npc_by_name("Survival Expert"):
                    npc.chat_with_npc("Survival Expert")
                    return
                return

            case "COOKING_TUTORIAL": # good
                if inventory.has_item("Bread"):
                    self.set_phase("QUEST_TUTORIAL")
                    return
                if not npc.closest_npc_by_name("Master Chef"):
                    trav.go_to("COOKING_TUTORIAL")
                    return
                elif not inventory.has_items({"Pot of flour": 1, "Bucket of water": 1}) and not inventory.has_item("Bread dough"):
                    npc.chat_with_npc("Master Chef")
                    return
                elif not inventory.has_item("bread dough"):
                    inventory.use_item_on_item("Pot of flour", "Bucket of water")
                    wait_until(lambda: inventory.has_item("Bread dough"))
                    return
                elif not inventory.has_item("Bread"):
                    inventory.use_item_on_object("bread dough", "Range")
                    wait_until(lambda: not inventory.has_item("bread dough"))
                    return
                return

            case "QUEST_TUTORIAL":
                if player.get_y() > 9000:
                    self.set_phase("MINING_TUTORIAL")
                    return
                if chat.dialogue_contains_text("caves.") or chat.dialogue_contains_text("ready to move on"):
                    objects.click("Ladder", "Climb-down")
                    wait_until(lambda: player.get_y() > 9000)
                    return
                if chat.dialogue_contains_text("flashing icon"):
                    widgets.click_widget(10747958) # QUESTS tab stone
                    return
                if not npc.closest_npc_by_name("Quest guide"):
                    trav.go_to("QUEST_TUTORIAL")
                    return
                else:
                    npc.chat_with_npc("Quest Guide")
                    return

            case "MINING_TUTORIAL":
                if inventory.has_item("Bronze dagger"):
                    self.set_phase("COMBAT_TUTORIAL")
                    return
                if inventory.has_item("hammer"):
                    if inventory.has_item("bronze bar"):
                        objects.click("Anvil", "Smith")
                        if not wait_until(lambda: widgets.smithing_interface_open()):
                            return
                        widgets.click_widget(20447241) # bronze dagger in smithing interface
                        if not wait_until(lambda: inventory.has_item("Bronze dagger")):
                            return
                        return

                if inventory.has_item("Bronze pickaxe"):
                    if not inventory.has_item("Tin ore") and not inventory.has_item("bronze bar"):
                        objects.click('Tin rocks')
                        wait_until(lambda: inventory.has_item("Tin ore"))
                        return
                    if not inventory.has_item("Copper ore") and not inventory.has_item("bronze bar"):
                        objects.click('Copper rocks')
                        wait_until(lambda: inventory.has_item("Copper ore"))
                        return
                    if inventory.has_items({"Copper ore": 1, "Tin ore": 1}):
                        objects.click("Furnace", "Use")
                        wait_until(lambda: inventory.has_item("Bronze bar"))
                        return
                if not npc.closest_npc_by_name("Mining Instructor"):
                    trav.go_to("MINING_TUTORIAL")
                    return
                else:
                    npc.chat_with_npc("Mining Instructor")
                    return

            case "COMBAT_TUTORIAL":
                if chat.can_continue():
                    press_spacebar()
                    return
                if player.is_in_combat():
                    return
                if player.get_y() < 9000:
                    self.set_phase("BANK_TUTORIAL")
                    return
                if chat.dialogue_contains_text("just talk to the combat instructor"):
                    objects.click("Ladder", "Climb-up")
                    wait_until(lambda: player.get_y() < 9000)
                    return
                if inventory.has_any_items(["Shortbow", "Bronze arrow"]):
                    if not tab.is_tab_open("INVENTORY"):
                        tab.open_tab("INVENTORY")
                        return
                    if not equipment.has_equipped("Shortbow"):
                        inventory.interact("Shortbow", "Wield")
                        return
                    if not equipment.has_equipped("Bronze arrow"):
                        inventory.interact("Bronze arrow", "Wield")
                        return

                if equipment.has_equipped(["Shortbow", "Bronze arrow"]):
                    if not player.is_in_combat():
                        npc.click_npc_action_simple("Giant rat", "Attack")
                        return
                    return
                if chat.dialogue_contains_text("Attacking"):
                    if not player.is_in_combat():
                        npc.click_npc_action("Giant rat", "Attack")
                        return
                    return
                if chat.dialogue_contains_text("Click on the gates to continue."):
                    if not player.is_in_combat():
                        npc.click_npc_action("Giant rat", "Attack")
                        return
                    return
                if chat.dialogue_contains_text("flashing crossed swords"):
                    widgets.click_widget(10747956) # COMBAT tab stone
                    return
                if chat.dialogue_contains_text("Unequipping items") or chat.dialogue_contains_text("bronze sword"):
                    inventory.interact("Bronze sword", "Wield")
                    inventory.interact("Wooden shield", "Wield")
                    wait_until(lambda: equipment.has_equipped(["Bronze sword", "Wooden shield"]))
                    return
                if chat.dialogue_contains_text("Equipping items"):
                    widgets.click_widget(10747960) # EQUIPMENT tab stone
                    wait_until(lambda: tab.is_tab_open("EQUIPMENT"))
                    return
                if chat.dialogue_contains_text("Worn inventory") and not equipment.equipment_interface_open() and not equipment.has_equipment_item("Bronze dagger"):
                    widgets.click_widget(25362434) # view equipment stats button
                    wait_until(lambda: equipment.equipment_interface_open())
                    return
                if chat.dialogue_contains_text("Equipment stats") and equipment.equipment_interface_open():
                    if not equipment.has_equipment_item("Bronze dagger"):
                        equipment.interact("Bronze dagger", "Equip")
                        return
                    else:
                        if equipment.equipment_interface_open():
                            press_esc()
                            return
                if not npc.closest_npc_by_name("Combat Instructor"):
                    trav.go_to("COMBAT_TUTORIAL")
                    return
                else:
                    npc.chat_with_npc("Combat Instructor")
                    return

            case "BANK_TUTORIAL":
                if bank.is_closed():
                    bank.open_bank()
                    return
                elif bank.is_open():
                    bank.close_bank()
                    self.set_phase("POLL_BOOTH")
                    return
                return

            case "POLL_BOOTH":
                if chat.dialogue_contains_text("Moving on"):
                    if widget_exists(20316160):  # Poll booth interface
                        press_esc()
                        return
                    self.set_phase("ACCOUNT_GUIDE")
                    return
                elif not chat.dialogue_contains_text("Voting") and not chat.dialogue_contains_text("booths are found in")\
                        and not chat.dialogue_contains_text("A flag appears"):
                    objects.click("Poll booth", "Use")
                    return
                elif widget_exists(20316160): # Poll booth interface
                    press_esc()
                    return
                else:
                    press_spacebar()
                    return

            case "ACCOUNT_GUIDE": # GOOD AND WORKING
                if widget_exists(20316160) or widget_exists(22609920):  # Poll booth interface
                    press_esc()
                    return
                if chat.dialogue_contains_text("Continue through the next door"):
                    self.set_phase("PRAYER_TUTORIAL")
                    return
                if chat.dialogue_contains_text("Click on the flashing icon to open your Account Management"):
                    widgets.click_widget(10747943)  # ACCOUNT_MANAGEMENT tab stone
                    wait_until(lambda: tab.is_tab_open("ACCOUNT_MANAGEMENT"))
                    return
                if npc.closest_npc_by_name("Account Guide"):
                    npc.chat_with_npc("Account Guide", options=["No thanks"])
                    return
                return

            case "PRAYER_TUTORIAL":
                if chat.dialogue_contains_text("Your final instructor"):
                    self.set_phase("MAGIC_TUTORIAL")
                    return
                if chat.dialogue_contains_text("flashing face"):
                    widgets.click_widget(10747944)  # FRIENDS_LIST tab stone
                    wait_until(lambda: tab.is_tab_open("FRIENDS_LIST"))
                    return
                if chat.dialogue_contains_text("flashing icon"):
                    widgets.click_widget(10747961)  # PRAYER tab stone
                    wait_until(lambda: tab.is_tab_open("PRAYER"))
                    return
                if not npc.closest_npc_by_name("Brother Brace"):
                    trav.go_to("PRAYER_TUTORIAL")
                    return
                else:
                    npc.chat_with_npc("Brother Brace", options=["Nope, I'm ready"])
                    return

            case "MAGIC_TUTORIAL":
                if trav.in_area("LUMBRIDGE_NEW_PLAYER_SPAWN"):
                    self.set_phase("DONE")
                    return
                if inventory.has_items({"Mind rune", "Air rune"}) and not chat.dialogue_contains_text("To the mainland!") and not player.is_in_combat() and not chat.dialogue_is_open() and not chat.get_options() and not chat.can_continue():
                    if not tab.is_tab_open("SPELLBOOK"):
                        tab.open_tab("SPELLBOOK")
                        if not wait_until(lambda: tab.is_tab_open("SPELLBOOK"), max_wait_ms=3000):
                            return
                        spellbook.cast_spell("Wind Strike", "Chicken")
                        return
                    else:
                        spellbook.cast_spell("Wind Strike", "Chicken")
                    return
                if chat.dialogue_contains_text("your final menu"):
                    widgets.click_widget(10747962)  # SPELLBOOK tab stone
                    wait_until(lambda: tab.is_tab_open("SPELLBOOK"))
                    return
                if not npc.closest_npc_by_name("Magic Instructor"):
                    trav.go_to("MAGIC_TUTORIAL")
                    return
                else:
                    npc.chat_with_npc("Magic Instructor", options=[
                        "Yes.",
                        "I'm not planning"
                    ])
                    return

            case "DONE":
                logging.info("[TUTORIAL] Tutorial Island completed!")
                return

