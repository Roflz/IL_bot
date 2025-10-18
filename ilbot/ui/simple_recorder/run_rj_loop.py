# run_rj_loop.py
# Standalone runner for immediate-mode plans without main_window.py

import json
import threading
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

from ilbot.ui.simple_recorder.helpers.runtime_utils import set_ui, set_ipc, set_action_executor
import logging
from ilbot.ui.simple_recorder.services.action_executor import ActionExecutor
import socket

# Import all available plans
from ilbot.ui.simple_recorder.plans.romeo_and_juliet import RomeoAndJulietPlan
from ilbot.ui.simple_recorder.plans.goblin_diplomacy import GoblinDiplomacyPlan
from ilbot.ui.simple_recorder.plans.tutorial_island import TutorialIslandPlan
from ilbot.ui.simple_recorder.plans.ge_trade import GeTradePlan
from ilbot.ui.simple_recorder.plans.falador_cows import FaladorCowsPlan
from ilbot.ui.simple_recorder.plans.woodcutting import WoodcuttingPlan
from ilbot.ui.simple_recorder.plans.bank_plan import BankPlan

# Plan registry - add new plans here
AVAILABLE_PLANS = {
    "romeo_and_juliet": RomeoAndJulietPlan,
    "goblin_diplomacy": GoblinDiplomacyPlan,
    "tutorial_island": TutorialIslandPlan,
    "ge_trade": GeTradePlan,
    "falador_cows": FaladorCowsPlan,
    "woodcutting": WoodcuttingPlan,
    "bank_plan": BankPlan,
}


def get_plan_class(plan_name: str):
    """Get plan class by name."""
    plan_class = AVAILABLE_PLANS.get(plan_name.lower())
    if plan_class is None:
        available = ", ".join(AVAILABLE_PLANS.keys())
        raise ValueError(f"Unknown plan '{plan_name}'. Available plans: {available}")
    return plan_class


def list_available_plans():
    """List all available plans with their descriptions."""
    print("Available plans:")
    for name, plan_class in AVAILABLE_PLANS.items():
        # Create a temporary instance to get the label
        try:
            temp_plan = plan_class()
            label = getattr(temp_plan, 'label', 'No description')
            print(f"  {name}: {label}")
        except Exception as e:
            print(f"  {name}: Error creating plan - {e}")


def find_available_ipc_port(start_port=17000, max_attempts=10):
    """Find an available IPC port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to create a socket connection to test if port is listening
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)  # 1 second timeout
                result = sock.connect_ex(('localhost', port))
                if result == 0:  # Connection successful, port is listening
                    print(f"Found listening IPC port: {port}")
                    return port
        except Exception:
            continue
    
    return None


class LoopRunner:
    """
    Loop runner for plan execution.
    Provides debug logging and delegates complex actions to ActionExecutor.
    """

    def __init__(self, session_dir: str, port: int, canvas_offset=(0, 0)):
        self.session_dir = Path(session_dir)
        self.canvas_offset = tuple(canvas_offset or (0, 0))
        
        # Create IPC instance directly
        from ilbot.ui.simple_recorder.helpers.ipc import IPCClient
        self.ipc = IPCClient(port=port)

        # Create action executor
        self.action_executor = ActionExecutor(self.ipc, canvas_offset)




def main():
    parser = argparse.ArgumentParser(
        description="Run RuneLite bot plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rj_loop.py romeo_and_juliet
  python run_rj_loop.py goblin_diplomacy --port 17001
  python run_rj_loop.py --list
  python run_rj_loop.py romeo_and_juliet --session-dir "D:\\data\\sessions\\player1\\gamestates\\"
        """
    )
    
    parser.add_argument(
        "plan", 
        nargs="?", 
        help="Plan name to run (use --list to see available plans)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available plans and exit"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None, 
        help="IPC port number (default: auto-detect starting from 17000)"
    )
    parser.add_argument(
        "--session-dir", 
        type=str, 
        default=r"D:\\repos\\bot_runelite_IL\\data\\recording_sessions\\gorillazzz33\\gamestates\\",
        help="Session directory path"
    )
    parser.add_argument(
        "--canvas-offset", 
        type=str, 
        default="0,0", 
        help="Canvas offset as x,y (default: 0,0)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=120, 
        help="Loop interval in milliseconds (default: 120)"
    )
    parser.add_argument(
        "--max-runtime", 
        type=int, 
        default=120, 
        help="Maximum runtime in minutes before auto-stop (default: 120 minutes = 2 hours)"
    )
    
    args = parser.parse_args()
    
    # Handle --list option
    if args.list:
        list_available_plans()
        return
    
    # Require plan name if not listing
    if not args.plan:
        parser.print_help()
        print("\nError: Plan name is required. Use --list to see available plans.")
        sys.exit(1)
    
    # Parse canvas offset
    try:
        offset_parts = args.canvas_offset.split(",")
        canvas_offset = (int(offset_parts[0]), int(offset_parts[1]))
    except (ValueError, IndexError):
        print(f"Error: Invalid canvas offset '{args.canvas_offset}'. Use format 'x,y'")
        sys.exit(1)
    
    # Auto-detect IPC port if not provided
    if args.port is None:
        print("No IPC port specified, auto-detecting available port...")
        detected_port = find_available_ipc_port(start_port=17000, max_attempts=10)
        if detected_port is None:
            print("Error: Could not find any listening IPC ports in range 17000-17009")
            print("Make sure RuneLite is running with the IPC plugin enabled")
            sys.exit(1)
        args.port = detected_port
    else:
        print(f"Using specified IPC port: {args.port}")
    
    # Get plan class
    try:
        plan_class = get_plan_class(args.plan)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create loop runner and plan instances
    loop_runner = LoopRunner(
        session_dir=args.session_dir, 
        port=args.port, 
        canvas_offset=canvas_offset
    )
    set_ui(loop_runner)
    
    # Set the global IPC instance
    set_ipc(loop_runner.ipc)
    
    # Set the global action executor
    set_action_executor(loop_runner.action_executor)
    
    plan = plan_class()
    
    logging.info(f"Starting plan: {plan.label} ({plan.id})")
    logging.info(f"Session dir: {args.session_dir}")
    logging.info(f"IPC port: {args.port}")
    logging.info(f"Canvas offset: {canvas_offset}")
    logging.info(f"Max runtime: {args.max_runtime} minutes")
    
    # Set up auto-stop timer
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=args.max_runtime)
    logging.info(f"Auto-stop scheduled for: {end_time.strftime('%H:%M:%S')}")
    
    try:
        while True:
            # Check if we've exceeded the maximum runtime
            current_time = datetime.now()
            if current_time >= end_time:
                runtime_minutes = (current_time - start_time).total_seconds() / 60
                logging.info(f"Maximum runtime of {args.max_runtime} minutes reached ({runtime_minutes:.1f} minutes elapsed). Stopping...")
                break
            
            # Let the plan decide the wait (ms)
            try:
                delay_ms = plan.loop(loop_runner)
            except Exception as e:
                import traceback
                logging.info(f"[PLAN] error in loop: {e}")
                logging.info(f"[PLAN] error type: {type(e).__name__}")
                logging.info(f"[PLAN] traceback: {traceback.format_exc()}")
                delay_ms = getattr(plan, "loop_interval_ms", args.interval)
            
            # Normalize delay
            try:
                delay_ms = int(delay_ms if delay_ms is not None else plan.loop_interval_ms)
            except Exception:
                delay_ms = getattr(plan, "loop_interval_ms", args.interval)
            delay_ms = max(10, delay_ms)
            
            time.sleep(delay_ms / 1000.0)
    except KeyboardInterrupt:
        import traceback
        logging.info("Stopped by user.")
        logging.info("Interrupted at:")
        logging.info(f"Main thread traceback: {traceback.format_exc()}")
    except Exception as e:
        import traceback
        logging.info(f"Fatal error: {e}")
        logging.info(f"Fatal error type: {type(e).__name__}")
        logging.info(f"Fatal error traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Show final runtime statistics
        final_time = datetime.now()
        total_runtime = (final_time - start_time).total_seconds() / 60
        logging.info(f"Script completed. Total runtime: {total_runtime:.1f} minutes")


if __name__ == "__main__":
    main()
