"""
Character creation and customization actions.
"""

import time
import random

from ..helpers.utils import sleep_exponential
from ..helpers.widgets import rect_center_from_widget
from ..helpers.runtime_utils import dispatch
import logging


def customize_character_appearance(design_buttons: dict, plan_id: str = "CHARACTER") -> None:
    """Customize character appearance by randomly clicking LEFT/RIGHT buttons."""
    logging.info(f"[{plan_id}] Starting character customization...")
    
    # Body part customizations
    body_parts = ["HEAD", "JAW", "TORSO", "ARMS", "HANDS", "LEGS", "FEET"]
    for part in body_parts:
        # Randomly click LEFT or RIGHT 1-10 times for each body part
        if part == 'HEAD':
            num_clicks = random.randint(1, 20)
        else:
            num_clicks = random.randint(1, 10)
        direction = "LEFT"
        button_name = f"{part}_{direction}"
        
        if button_name in design_buttons:
            widget = design_buttons[button_name]
            # logging.info(f"[{plan_id}] Clicking {button_name} {num_clicks} times")
            
            for click_num in range(num_clicks):
                if click_character_design_button(widget, button_name, plan_id):
                    sleep_exponential(0.1, 0.3, 1.5)  # Wait between clicks
                else:
                    logging.info(f"[{plan_id}] Failed to click {part} {direction}")
                    break
        else:
            logging.info(f"[{plan_id}] Button {button_name} not found in design_buttons")
    
    # Color customizations
    color_parts = ["HAIR", "TORSO_COL", "LEGS_COL", "FEET_COL", "SKIN"]
    for part in color_parts:
        # Randomly click LEFT or RIGHT 1-10 times for each color
        num_clicks = random.randint(1, 10)
        direction = random.choice(["LEFT", "RIGHT"])
        button_name = f"{part}_{direction}"
        
        if button_name in design_buttons:
            widget = design_buttons[button_name]
            # logging.info(f"[{plan_id}] Clicking {button_name} {num_clicks} times")
            
            for click_num in range(num_clicks):
                if click_character_design_button(widget, button_name, plan_id):
                    # logging.info(f"[{plan_id}] Click {click_num + 1}/{num_clicks} on {part} color {direction}")
                    sleep_exponential(0.1, 0.3, 1.5)  # Wait between clicks
                else:
                    logging.info(f"[{plan_id}] Failed to click {part} color {direction}")
                    break
        else:
            logging.info(f"[{plan_id}] Button {button_name} not found in design_buttons")
    
    logging.info(f"[{plan_id}] Character customization completed!")
    
    # Click the CONFIRM button to finish character creation
    confirm_widget = design_buttons.get("WIDGET_44499018")  # PlayerDesign.CONFIRM
    if confirm_widget:
        logging.info(f"[{plan_id}] Clicking CONFIRM button...")
        if click_character_design_button(confirm_widget, "CONFIRM", plan_id):
            logging.info(f"[{plan_id}] Successfully clicked CONFIRM button")
            sleep_exponential(0.8, 1.5, 1.0)  # Wait for the interface to close
        else:
            logging.info(f"[{plan_id}] Failed to click CONFIRM button")
    else:
        logging.info(f"[{plan_id}] CONFIRM button not found in design_buttons")


def click_character_design_button(widget: dict, button_name: str, plan_id: str = "CHARACTER") -> bool:
    """Click a character design button."""
    if not widget or not widget.get("visible", False):
        logging.info(f"[{plan_id}] Button {button_name} not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(widget)
    if x is None or y is None:
        logging.info(f"[{plan_id}] Could not get coordinates for {button_name}")
        return False
    
    # Click the button
    step = {
        "action": f"click-{button_name.lower()}",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "character_design", "name": button_name.lower()}
    }
    
    result = dispatch(step)
    if result is None:
        logging.info(f"[{plan_id}] Failed to click {button_name}")
        return False

    return True


