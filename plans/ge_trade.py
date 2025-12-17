# ge_trade.py
import logging
import time
import csv
import os

from actions import player, inventory
from actions import find_chat_message, \
    click_chat_message
from actions import wait_until
from actions import other_offer_contains, accept_trade, other_offer_confirmation_contains, \
    accept_trade_confirm, find_player_by_name, trade_with_player, my_offer_contains, offer_all_items
import actions.travel as trav
import actions.bank as bank

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from helpers import move_camera_random, setup_camera_optimal
from helpers.utils import get_world_from_csv, sleep_exponential
from helpers.widgets import widget_exists
from helpers import set_phase_with_camera


class GeTradePlan(Plan):
    id = "GE_TRADE"
    label = "Grand Exchange Trading"

    def __init__(self, role="worker"):
        self.state = {"phase": "GO_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        self.role = role  # "worker" or "mule"

        # Create bank plan for withdrawing coins
        self.bank_plan = BankPlanSimple(
            bank_area=None,  # Use closest bank
            required_items=[],  # Will be set dynamically based on role
            deposit_all=True
        )

        # Load allowed usernames from CSV
        self.allowed_usernames = self._load_allowed_usernames()
        
        # Track chat message snapshots to prevent duplicate trade requests
        # Each snapshot is a tuple of 5 recent messages
        self.chat_snapshots = set()
        
        # Current trading partner
        self.current_trading_partner = None

        # Set up camera immediately during initialization
        setup_camera_optimal()

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def _load_allowed_usernames(self):
        """Load allowed usernames from character_stats.csv"""
        allowed_usernames = set()
        csv_path = "D:\\repos\\bot_runelite_IL\ilbot\\ui\simple_recorder\character_data\character_stats.csv"
        
        try:
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row.get('username'):
                            allowed_usernames.add(row['username'].strip())
                logging.info(f"[{self.id}] Loaded {len(allowed_usernames)} allowed usernames: {allowed_usernames}")
            else:
                logging.warning(f"[{self.id}] Character stats CSV not found at {csv_path}")
        except Exception as e:
            logging.error(f"[{self.id}] Error loading allowed usernames: {e}")
        
        return allowed_usernames

    def _is_username_allowed(self, username):
        """Check if username is in the allowed list"""
        return username in self.allowed_usernames

    def _get_messages_before_trade(self, trade_message, count=5):
        """Get the N messages that came before the trade message"""
        try:
            from actions import get_chatbox_scroll_areas
            messages = get_chatbox_scroll_areas()
            
            # Find the trade message in the chat
            trade_index = -1
            for i, msg in enumerate(messages):
                if trade_message in msg.get("full_text", ""):
                    trade_index = i
                    break
            
            if trade_index == -1:
                logging.warning(f"[{self.id}] Could not find trade message in chat")
                return []
            
            # Get the N messages before the trade message
            end_index = max(0, trade_index + count)
            before_messages = messages[trade_index:end_index]
            
            return [msg.get("full_text", "") for msg in before_messages]
        except Exception as e:
            logging.error(f"[{self.id}] Error getting messages before trade: {e}")
            return []

    def _create_chat_snapshot(self, trade_message):
        """Create a snapshot of the 5 messages before the trade message"""
        before_messages = self._get_messages_before_trade(trade_message, 5)
        
        # Create a tuple of the messages that came before the trade
        snapshot = tuple(before_messages)
        return snapshot

    def _is_duplicate_trade_request(self, trade_message):
        """Check if this trade request matches a previous one based on chat snapshot"""
        current_snapshot = self._create_chat_snapshot(trade_message)
        
        # Check if this exact snapshot was seen before
        for saved_snapshot in self.chat_snapshots:
            if current_snapshot == saved_snapshot:
                logging.info(f"[{self.id}] Found duplicate trade request based on chat snapshot")
                return True
        
        return False

    def _mark_trade_request_processed(self, trade_message):
        """Mark this trade request as processed to prevent duplicates"""
        snapshot = self._create_chat_snapshot(trade_message)
        self.chat_snapshots.add(snapshot)
        logging.info(f"[{self.id}] Marked trade request as processed - snapshot contains {len(snapshot)} messages")

    def _extract_username_from_message(self, message_text):
        """Extract username from 'wishes to trade with you' message"""
        # Message format: "PlayerName wishes to trade with you."
        if "wishes to trade with you" in message_text:
            username = message_text.split(" wishes to trade with you")[0].strip()
            return username
        return None


    def loop(self, ui):
        phase = self.state.get("phase", "GO_TO_GE")
        logged_in = player.logged_in()
        if not logged_in:
            logging.info("Logged out, logging back in.")
            player.login()
            return self.loop_interval_ms

        match (phase):
            case "BANK":
                return self._handle_bank(ui)

            case "GO_TO_GE":
                # Check if we're already at the Grand Exchange
                if trav.in_area("GE"):
                    # Transition to role-based phase
                    self.set_phase("BANK")
                    return
                else:
                    # Use enhanced long-distance travel for GE
                    print(f"[GE_TRADE] Using enhanced travel to reach Grand Exchange as {self.role}...")
                    trav.go_to("GE")
                    return

            case "WORKER":
                mule_world = get_world_from_csv("Batquinn")
                if not player.get_world() == mule_world:
                    player.hop_world(mule_world)
                    return 3000

                if not widget_exists(21954562) and not widget_exists(21889025):
                    if not inventory.has_item("coins"):
                        self.set_phase("DONE")
                        return self.loop_interval_ms

                    if find_player_by_name("Batquinn"):
                        trade_with_player("Batquinn")
                        wait_until(lambda: widget_exists(21954562))
                        return self.loop_interval_ms

                elif widget_exists(21954562): #trade interface
                    if not my_offer_contains("coins"):
                        offer_all_items("coins")

                    else:
                        accept_trade()
                        return 2000

                elif widget_exists(21889025):
                    accept_trade_confirm()
                    return 2000
                
                return self.loop_interval_ms

            case "MULE":
                # Random timer between 90 seconds and 4 minutes (240 seconds)
                import random
                timer_duration = random.randint(90, 240)
                logging.info(f"[GE_TRADE] Starting timer for {timer_duration} seconds")
                
                start_time = time.time()
                while time.time() - start_time < timer_duration:
                    if widget_exists(21954562): #trade interface
                        if other_offer_contains("coins"):
                            accept_trade()
                            return 2000
                        return
                    if widget_exists(21889025): #trade confirm interface
                        if other_offer_confirmation_contains("coins"):
                            accept_trade_confirm()
                            # Reset trading partner after trade completion
                            self.current_trading_partner = None
                            return 2000
                        return

                    message = find_chat_message("wishes to trade with you")
                    if message:
                        # Check if this is a duplicate trade request based on chat snapshot
                        if self._is_duplicate_trade_request(message.get("full_text", "")):
                            logging.info(f"[{self.id}] Ignoring duplicate trade request based on chat snapshot")
                            continue
                        
                        # Extract username from the message
                        username = self._extract_username_from_message(message.get("full_text"))
                        
                        if username:
                            # Check if username is allowed
                            if not self._is_username_allowed(username):
                                logging.info(f"[{self.id}] Ignoring trade request from unauthorized user: {username}")
                                continue
                            
                            # Accept the trade request
                            logging.info(f"[{self.id}] Accepting trade request from authorized user: {username}")
                            self.current_trading_partner = username  # Store the trading partner
                            
                            # Mark this trade request as processed to prevent duplicates
                            self._mark_trade_request_processed(message.get("full_text", ""))
                            
                            click_chat_message("wishes to trade with you")
                            wait_until(lambda: widget_exists(21954562))
                        else:
                            logging.warning(f"[{self.id}] Could not extract username from trade message: {message}")
                    
                    # Small delay to prevent excessive CPU usage
                    sleep_exponential(0.3, 0.8, 1.2)
                
                logging.info(f"[GE_TRADE] Timer completed after {timer_duration} seconds")
                logging.info("Moving camera a random amount")
                move_camera_random()
                return

            case "DONE":
                print("[GE_TRADE] Trading complete!")
                return

        return

    def _handle_bank(self, ui) -> int:
        """Handle banking phase - withdraw coins from bank."""
        # Update bank plan to withdraw coins based on role
        if self.role == "mule":
            self.bank_plan.required_items = [
                {"name": "Coins", "quantity": -1}  # Withdraw all coins
            ]
        else:
            if not trav.travel_to_bank():
                return self.loop_interval_ms
            if not bank.is_open():
                bank.open_bank()
                return self.loop_interval_ms
            current_coins = inventory.inv_count("Coins") + bank.get_item_count("coins")
            coins_to_withdraw = max(0, current_coins - 300000)
            
            logging.info(f"[{self.id}] Worker: Current coins: {current_coins}, withdrawing: {coins_to_withdraw}")
            
            self.bank_plan.required_items = [
                {"name": "Coins", "quantity": coins_to_withdraw}
            ]

        bank_status = self.bank_plan.loop(ui)

        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully - coins withdrawn!")
            if self.role == "worker":
                self.set_phase("WORKER")
            else:  # mule
                self.set_phase("MULE")
            if bank.is_open():
                bank.close_bank()
            return self.loop_interval_ms

        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.warning(f"[{self.id}] No coins found in bank: {error_msg}")
            # Still proceed to GE even without coins
            if self.role == "worker":
                self.set_phase("DONE")
            else:  # mule
                self.set_phase("MULE")
            return self.loop_interval_ms

        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")
            return self.loop_interval_ms

        else:
            # Still working on banking (TRAVELING, BANKING, etc.)
            return bank_status