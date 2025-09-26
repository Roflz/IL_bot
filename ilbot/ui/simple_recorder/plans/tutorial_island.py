# tutorial_island.py
import time
import random

# Action method imports
import ilbot.ui.simple_recorder.actions.travel as trav
import ilbot.ui.simple_recorder.actions.tab as tab
import ilbot.ui.simple_recorder.actions.bank as bank
import ilbot.ui.simple_recorder.actions.chat as chat
import ilbot.ui.simple_recorder.actions.combat as combat
import ilbot.ui.simple_recorder.actions.ge as ge
import ilbot.ui.simple_recorder.actions.inventory as inventory
import ilbot.ui.simple_recorder.actions.npc as npc
import ilbot.ui.simple_recorder.actions.objects as objects
import ilbot.ui.simple_recorder.actions.player as player
import ilbot.ui.simple_recorder.actions.timing as timing
import ilbot.ui.simple_recorder.actions.widgets as widgets

# Specific function imports
from ..actions.timing import wait_until
from ..actions.chat import type_tutorial_name, click_tutorial_set_name, click_tutorial_lookup_name, can_click_continue_widget, click_continue_widget
from ..constants import BANK_REGIONS, REGIONS, TUTORIAL_NAME_COMBOS
from .base import Plan
from ..helpers import quest
from ..helpers.widgets import get_widget_text, rect_center_from_widget, get_character_design_widgets, get_character_design_main, get_character_design_button_realtime, get_all_character_design_buttons, widget_exists, character_design_widget_exists, get_widget_info
from ..helpers.utils import press_enter, press_backspace
from ..actions.runtime import emit

class TutorialIslandPlan(Plan):
    id = "TUTORIAL_ISLAND"
    label = "Tutorial Island Completion"

    def __init__(self):
        self.state = {"phase": "START_TUTORIAL"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

    def compute_phase(self, payload, craft_recent):
        return self.state.get("phase", "START_TUTORIAL")

    def set_phase(self, phase: str, ui=None):
        self.state["phase"] = phase
        self.next = phase
        if ui is not None:
            try:
                ui.debug(f"[TUTORIAL] phase → {phase}")
            except Exception:
                pass
        return phase

    def loop(self, ui, payload):
        phase = self.state.get("phase", "START_TUTORIAL")
        
        # Check if tutorial is already completed
        # TODO: Add tutorial completion check

        phase = "SURVIVAL_INSTRUCTOR"
        match(phase):
            case "START_TUTORIAL":
                if widget_exists(44498948):
                    self.set_phase("CHARACTER_CREATION")
                    return
                # Check if tutorial name input is available
                tutorial_data = payload.get("tutorial", {})
                if tutorial_data.get("open", False):
                    name_input = tutorial_data.get("nameInput")
                    if name_input and name_input.get("visible", False):
                        # Check if we need to try a name or if we're waiting for status
                        
                        status_text = get_widget_text(36569101)  # TutorialDisplayname.STATUS
                        name_available = None
                        if status_text:
                            status_lower = status_text.lower()
                            if "great" in status_lower and "available" in status_lower:
                                name_available = True
                            elif "sorry" in status_lower and "not available" in status_lower:
                                name_available = False
                        if name_available is True:
                            # Name is available, click SET_NAME button
                            if click_tutorial_set_name(payload, ui):
                                ui.debug("[TUTORIAL] SET_NAME button clicked, waiting for window to close...")
                                # TODO: Add wait logic for window to close
                                self.set_phase("CHARACTER_CREATION", ui)
                            else:
                                ui.debug("[TUTORIAL] Failed to click SET_NAME button")
                        elif name_available is False:
                            # Name not available, try a different hilarious name
                            # Backspace to clear the current name first
                            current_text = get_widget_text(36569100) or ""
                            # Click on the name input field to focus it before backspacing
                            tutorial_data = payload.get("tutorial", {})
                            name_input = tutorial_data.get("nameInput")
                            if name_input and name_input.get("visible", False):
                                x, y = rect_center_from_widget(name_input)
                                if x is not None and y is not None:
                                    # Click to focus the input field
                                    step = emit({
                                        "action": "click-name-input-focus",
                                        "click": {"type": "point", "x": x, "y": y},
                                        "target": {"domain": "tutorial", "name": "name_input_focus"}
                                    })
                                    ui.dispatch(step)
                                    time.sleep(0.1)  # Wait for focus
                            # Now backspace to clear the field
                            for _ in range(len(current_text)):
                                press_backspace()
                            # Wait a moment for backspacing to complete
                            time.sleep(0.2)
                            # Try a random hilarious combination
                            new_name = random.choice(TUTORIAL_NAME_COMBOS)
                            ui.debug(f"[TUTORIAL] Name not available, trying: {new_name}")
                            if type_tutorial_name(new_name):
                                ui.debug(f"[TUTORIAL] Tried hilarious name: {new_name}")
                                # Note: type_tutorial_name already presses Enter
                            else:
                                ui.debug("[TUTORIAL] Failed to enter new name")
                        else:
                            # No status yet, try initial hilarious name
                            initial_name = random.choice(TUTORIAL_NAME_COMBOS)
                            if get_widget_text(36569100) == "*" or get_widget_text(36569100) == "":
                                type_tutorial_name(initial_name)
                                # Note: type_tutorial_name already presses Enter
                            if wait_until(lambda: initial_name in get_widget_text(36569100)):
                                ui.debug(f"[TUTORIAL] Character name entered: {initial_name}")
                                # Note: type_tutorial_name already pressed Enter, looking for confirmation...
                                # Wait for confirmation message or status update
                                if wait_until(lambda: get_widget_text(36569101) is not None):
                                    status_text = get_widget_text(36569101)
                                    ui.debug(f"[TUTORIAL] Confirmation message received: {status_text}")
                                    
                                    if status_text and "not available" in status_text.lower():
                                        ui.debug("[TUTORIAL] Name not available, trying new name...")
                                        # Backspace to clear the current name
                                        current_text = get_widget_text(36569100) or ""
                                        # Click on the name input field to focus it before backspacing
                                        tutorial_data = payload.get("tutorial", {})
                                        name_input = tutorial_data.get("nameInput")
                                        if name_input and name_input.get("visible", False):
                                            x, y = rect_center_from_widget(name_input)
                                            if x is not None and y is not None:
                                                # Click to focus the input field
                                                step = emit({
                                                    "action": "click-name-input-focus",
                                                    "click": {"type": "point", "x": x, "y": y},
                                                    "target": {"domain": "tutorial", "name": "name_input_focus"}
                                                })
                                                ui.dispatch(step)
                                                time.sleep(0.1)  # Wait for focus
                                        # Now backspace to clear the field
                                        for _ in range(len(current_text)):
                                            press_backspace()
                                        # Wait a moment for backspacing to complete
                                        time.sleep(0.2)
                                        # Try a new hilarious name
                                        new_name = random.choice(TUTORIAL_NAME_COMBOS)
                                        ui.debug(f"[TUTORIAL] Trying new name: {new_name}")
                                        if type_tutorial_name(new_name):
                                            ui.debug("[TUTORIAL] New name typed successfully")
                                            # Note: type_tutorial_name already presses Enter
                                            # Update the initial_name for the next iteration
                                            initial_name = new_name
                                        else:
                                            ui.debug("[TUTORIAL] Failed to enter new name")
                                    elif status_text and "available" in status_text.lower():
                                        ui.debug("[TUTORIAL] Name is available! Clicking SET_NAME button...")
                                        if click_tutorial_set_name(payload, ui):
                                            ui.debug("[TUTORIAL] SET_NAME button clicked successfully")
                                            self.set_phase("CHARACTER_CREATION", ui)
                                        else:
                                            ui.debug("[TUTORIAL] Failed to click SET_NAME button")
                                    else:
                                        ui.debug(f"[TUTORIAL] Unknown status message: {status_text}")
                                else:
                                    ui.debug("[TUTORIAL] No confirmation message found")
                            else:
                                ui.debug("[TUTORIAL] Failed to enter character name")
                    else:
                        ui.debug("[TUTORIAL] Name input not visible, waiting...")
                else:
                    ui.debug("[TUTORIAL] Tutorial interface not open, waiting...")
                return

            case "CHARACTER_CREATION":
                # Check if character creation interface is open via direct IPC
                if not widget_exists(44498948):  # PlayerDesign.MAIN widget
                    ui.debug("[TUTORIAL] Waiting for character creation interface...")
                    return
                
                ui.debug("[TUTORIAL] Character creation interface detected")
                
                # Get all character design buttons via real-time IPC
                ui.debug("[TUTORIAL] Attempting to get character design buttons...")
                design_buttons = get_all_character_design_buttons()
                ui.debug(f"[TUTORIAL] get_all_character_design_buttons() returned: {design_buttons}")
                
                if design_buttons:
                    ui.debug(f"[TUTORIAL] Found {len(design_buttons)} character design buttons via IPC")
                    
                    # Customize character appearance
                    self.customize_character_appearance(design_buttons, ui)
                else:
                    ui.debug("[TUTORIAL] No character design buttons found via IPC")
                
                # Move to next phase after customization
                self.set_phase("TALK_TO_GUIDE", ui)
                return

            case "TALK_TO_GUIDE":
                if "Moving on" in chat.get_dialogue_text_raw():
                    self.set_phase("SURVIVAL_INSTRUCTOR")
                    return

                # Check if we need to click "Click here to continue"
                if can_click_continue_widget():
                    ui.debug("[TUTORIAL] Found 'Click here to continue' widget, clicking it...")
                    if click_continue_widget():
                        ui.debug("[TUTORIAL] Successfully clicked continue widget")
                        time.sleep(0.5)  # Wait for dialogue to advance
                        return
                    else:
                        ui.debug("[TUTORIAL] Failed to click continue widget")

                if widget_exists(10747945): # settings menu tab
                    if not tab.is_tab_open("SETTINGS"):
                        tab.open_tab("SETTINGS")
                        if not wait_until(lambda: tab.is_tab_open("SETTINGS")):
                            return
                    elif npc.closest_npc_by_name("Gielinor guide"):
                        npc.chat_with_npc("Gielinor guide")
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

            case "SURVIVAL_INSTRUCTOR":
                # if not chat.dialogue_is_open():
                npc.chat_with_npc("Survival Expert")
                return

            case "COOKING_TUTORIAL":
                # TODO: Complete cooking tutorial
                self.set_phase("QUEST_TUTORIAL", ui)
                return

            case "QUEST_TUTORIAL":
                # TODO: Complete quest tutorial
                self.set_phase("BANK_TUTORIAL", ui)
                return

            case "BANK_TUTORIAL":
                # TODO: Complete bank tutorial
                self.set_phase("PRAYER_TUTORIAL", ui)
                return

            case "PRAYER_TUTORIAL":
                # TODO: Complete prayer tutorial
                self.set_phase("MAGIC_TUTORIAL", ui)
                return

            case "MAGIC_TUTORIAL":
                # TODO: Complete magic tutorial
                self.set_phase("DONE", ui)
                return

            case "DONE":
                ui.debug("[TUTORIAL] Tutorial Island completed!")
                return

    def customize_character_appearance(self, design_buttons: dict, ui):
        """Customize character appearance by randomly clicking LEFT/RIGHT buttons."""
        ui.debug("[TUTORIAL] Starting character customization...")
        
        import random
        
        # Body part customizations
        body_parts = ["HEAD", "JAW", "TORSO", "ARMS", "HANDS", "LEGS", "FEET"]
        for part in body_parts:
            # Randomly click LEFT or RIGHT 1-10 times for each body part
            num_clicks = random.randint(1, 10)
            direction = random.choice(["LEFT", "RIGHT"])
            button_name = f"{part}_{direction}"
            
            if button_name in design_buttons:
                widget = design_buttons[button_name]
                ui.debug(f"[TUTORIAL] Clicking {button_name} {num_clicks} times")
                
                for click_num in range(num_clicks):
                    if self.click_character_design_button(widget, button_name, ui):
                        ui.debug(f"[TUTORIAL] Click {click_num + 1}/{num_clicks} on {part} {direction}")
                        time.sleep(0.2)  # Wait between clicks
                    else:
                        ui.debug(f"[TUTORIAL] Failed to click {part} {direction}")
                        break
            else:
                ui.debug(f"[TUTORIAL] Button {button_name} not found in design_buttons")
        
        # Color customizations
        color_parts = ["HAIR", "TORSO_COL", "LEGS_COL", "FEET_COL", "SKIN"]
        for part in color_parts:
            # Randomly click LEFT or RIGHT 1-10 times for each color
            num_clicks = random.randint(1, 10)
            direction = random.choice(["LEFT", "RIGHT"])
            button_name = f"{part}_{direction}"
            
            if button_name in design_buttons:
                widget = design_buttons[button_name]
                ui.debug(f"[TUTORIAL] Clicking {button_name} {num_clicks} times")
                
                for click_num in range(num_clicks):
                    if self.click_character_design_button(widget, button_name, ui):
                        ui.debug(f"[TUTORIAL] Click {click_num + 1}/{num_clicks} on {part} color {direction}")
                        time.sleep(0.2)  # Wait between clicks
                    else:
                        ui.debug(f"[TUTORIAL] Failed to click {part} color {direction}")
                        break
            else:
                ui.debug(f"[TUTORIAL] Button {button_name} not found in design_buttons")
        
        ui.debug("[TUTORIAL] Character customization completed!")
        
        # Click the CONFIRM button to finish character creation
        confirm_widget = design_buttons.get("WIDGET_44499018")  # PlayerDesign.CONFIRM
        if confirm_widget:
            ui.debug("[TUTORIAL] Clicking CONFIRM button...")
            if self.click_character_design_button(confirm_widget, "CONFIRM", ui):
                ui.debug("[TUTORIAL] Successfully clicked CONFIRM button")
                time.sleep(1.0)  # Wait for the interface to close
            else:
                ui.debug("[TUTORIAL] Failed to click CONFIRM button")
        else:
            ui.debug("[TUTORIAL] CONFIRM button not found in design_buttons")

    def click_character_design_button(self, widget: dict, button_name: str, ui) -> bool:
        """Click a character design button."""
        if not widget or not widget.get("visible", False):
            ui.debug(f"[TUTORIAL] Button {button_name} not visible")
            return False
        
        # Get click coordinates
        x, y = rect_center_from_widget(widget)
        if x is None or y is None:
            ui.debug(f"[TUTORIAL] Could not get coordinates for {button_name}")
            return False
        
        # Click the button
        step = emit({
            "action": f"click-{button_name.lower()}",
            "click": {"type": "point", "x": x, "y": y},
            "target": {"domain": "character_design", "name": button_name.lower()}
        })
        
        result = ui.dispatch(step)
        if result is None:
            ui.debug(f"[TUTORIAL] Failed to click {button_name}")
            return False
        
        ui.debug(f"[TUTORIAL] Successfully clicked {button_name}")
        return True
