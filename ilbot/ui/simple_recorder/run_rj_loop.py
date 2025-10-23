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

# Dynamic plan discovery
def discover_plans():
    """Dynamically discover available plans from the plans directory."""
    plans = {}
    plans_dir = Path(__file__).parent / "plans"
    
    # Scan main plans directory
    for plan_file in plans_dir.glob("*.py"):
        if plan_file.name.startswith("__"):
            continue
        
        plan_name = plan_file.stem
        module_name = f"ilbot.ui.simple_recorder.plans.{plan_name}"
        
        try:
            # Import the module dynamically with a fresh import
            import importlib
            import sys
            
            # Remove the module from cache if it exists to force fresh import
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            module = importlib.import_module(module_name)
            
            # Look for a plan class (usually ends with 'Plan')
            plan_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.endswith('Plan') and 
                    attr_name != 'Plan' and
                    hasattr(attr, '__call__')):
                    # Check if this class is actually defined in this module
                    if hasattr(attr, '__module__') and attr.__module__ == module_name:
                        plan_class = attr
                        print(f"[PLAN_DISCOVERY] Found plan class {attr_name} in {plan_name}")
                        break
            
            if plan_class:
                plans[plan_name] = plan_class
                print(f"[PLAN_DISCOVERY] Registered plan: {plan_name} -> {plan_class.__name__}")
            else:
                print(f"[PLAN_DISCOVERY] No plan class found in {plan_name}")
                
        except Exception as e:
            print(f"[PLAN_DISCOVERY] Error importing {plan_name}: {e}")
            continue
    
    # Scan utilities subdirectory
    utilities_dir = plans_dir / "utilities"
    if utilities_dir.exists():
        for plan_file in utilities_dir.glob("*.py"):
            if plan_file.name.startswith("__"):
                continue
            
            plan_name = plan_file.stem
            module_name = f"ilbot.ui.simple_recorder.plans.utilities.{plan_name}"
            
            try:
                import importlib
                import sys
                
                # Remove the module from cache if it exists to force fresh import
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                module = importlib.import_module(module_name)
                
                # Look for a plan class
                plan_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('Plan') and 
                        attr_name != 'Plan' and
                        hasattr(attr, '__call__')):
                        # Check if this class is actually defined in this module
                        if hasattr(attr, '__module__') and attr.__module__ == module_name:
                            plan_class = attr
                            break
                
                if plan_class:
                    plans[plan_name] = plan_class
                    print(f"[PLAN_DISCOVERY] Found utility plan: {plan_name}")
                    
            except Exception as e:
                print(f"[PLAN_DISCOVERY] Error importing utility {plan_name}: {e}")
                continue
    
    return plans

# Discover plans dynamically
AVAILABLE_PLANS = discover_plans()


def get_plan_class(plan_name: str):
    """Get plan class by name."""
    print(f"[DEBUG] Looking for plan: '{plan_name}'")
    print(f"[DEBUG] Available plans: {list(AVAILABLE_PLANS.keys())}")
    
    plan_class = AVAILABLE_PLANS.get(plan_name.lower())
    print(f"[DEBUG] Found plan class: {plan_class}")
    
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


def parse_item_list(item_string):
    """Parse a comma-separated list of items with format name:quantity:bumps:price"""
    items = []
    
    if not item_string.strip():
        return items
    
    for item_str in item_string.split(','):
        item_str = item_str.strip()
        if not item_str:
            continue
            
        parts = item_str.split(':')
        if len(parts) != 4:
            raise ValueError(f"Invalid item format '{item_str}'. Expected 'name:quantity:bumps:price'")
        
        name, quantity_str, bumps_str, price_str = parts
        
        try:
            quantity = int(quantity_str)
            bumps = int(bumps_str)
            price = int(price_str)
        except ValueError as e:
            raise ValueError(f"Invalid number in item '{item_str}': {e}")
        
        items.append({
            "name": name,
            "quantity": quantity,
            "bumps": bumps,
            "set_price": price
        })
    
    return items


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
        default=0, 
        help="Maximum runtime in minutes before auto-stop (0 = no limit)"
    )
    parser.add_argument(
        "--stop-skill", 
        type=str, 
        default="", 
        help="Stop when skill reaches target level (format: skill_name:level, e.g., woodcutting:20)"
    )
    parser.add_argument(
        "--total-level", 
        type=int, 
        default=0, 
        help="Stop when total level reaches target (0 = no limit)"
    )
    parser.add_argument(
        "--stop-item", 
        type=str, 
        default="", 
        help="Stop when item quantity reached (format: item_name:quantity, e.g., logs:100)"
    )
    parser.add_argument(
        "--buy-items", 
        type=str, 
        default="", 
        help="Items to buy (format: name:quantity:bumps:price,name:quantity:bumps:price)"
    )
    parser.add_argument(
        "--sell-items", 
        type=str, 
        default="", 
        help="Items to sell (format: name:quantity:bumps:price,name:quantity:bumps:price)"
    )
    parser.add_argument(
        "--role", 
        type=str, 
        default="worker", 
        choices=["worker", "mule"],
        help="Role for ge_trade plan: 'worker' or 'mule' (default: worker)"
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
    
    print(f"[DEBUG] Creating plan instance with class: {plan_class}")
    
    # Create plan with parameters
    plan_kwargs = {}
    
    # Add GE utility parameters
    if args.plan == "ge":
        if args.buy_items:
            buy_items = parse_item_list(args.buy_items)
            plan_kwargs['items_to_buy'] = buy_items
            print(f"[DEBUG] Buy items: {buy_items}")
        
        if args.sell_items:
            sell_items = parse_item_list(args.sell_items)
            plan_kwargs['items_to_sell'] = sell_items
            print(f"[DEBUG] Sell items: {sell_items}")
    
    # Add role parameter for ge_trade plan
    if args.plan == "ge_trade":
        plan_kwargs['role'] = args.role
        print(f"[DEBUG] Role: {args.role}")
    
    plan = plan_class(**plan_kwargs)
    print(f"[DEBUG] Created plan: {plan.__class__.__name__} with id: {getattr(plan, 'id', 'unknown')}")
    print(f"[DEBUG] Plan parameters: {plan_kwargs}")
    
    logging.info(f"Starting plan: {plan.label} ({plan.id})")
    logging.info(f"Session dir: {args.session_dir}")
    logging.info(f"IPC port: {args.port}")
    logging.info(f"Canvas offset: {canvas_offset}")
    logging.info(f"Max runtime: {args.max_runtime} minutes")
    
    # Parse rules
    skill_name, skill_level = "", 0
    if args.stop_skill:
        try:
            skill_name, skill_level = args.stop_skill.split(":")
            skill_level = int(skill_level)
        except (ValueError, IndexError):
            print(f"Error: Invalid stop-skill format '{args.stop_skill}'. Use 'skill_name:level'")
            sys.exit(1)
    
    item_name, item_quantity = "", 0
    if args.stop_item:
        try:
            item_name, item_quantity = args.stop_item.split(":")
            item_quantity = int(item_quantity)
        except (ValueError, IndexError):
            print(f"Error: Invalid stop-item format '{args.stop_item}'. Use 'item_name:quantity'")
            sys.exit(1)
    
    # Set up rules
    start_time = datetime.now()
    max_minutes = args.max_runtime if args.max_runtime > 0 else 0
    total_level = args.total_level if args.total_level > 0 else 0
    
    # Log rules
    rules_log = []
    if max_minutes > 0:
        rules_log.append(f"Time limit: {max_minutes} minutes")
    if skill_name and skill_level > 0:
        rules_log.append(f"Skill: {skill_name} level {skill_level}")
    if total_level > 0:
        rules_log.append(f"Total level: {total_level}")
    if item_name and item_quantity > 0:
        rules_log.append(f"Item: {item_quantity} {item_name}")
    
    if rules_log:
        logging.info(f"Rules: {', '.join(rules_log)}")
    else:
        logging.info("No rules configured - plan will run indefinitely")
    
    try:
        while True:
            # Check rules every loop
            from ilbot.ui.simple_recorder.helpers.rules import check_rules
            triggered_rule = check_rules(
                start_time=start_time,
                max_minutes=max_minutes,
                skill_name=skill_name,
                skill_level=skill_level,
                total_level=total_level,
                item_name=item_name,
                item_quantity=item_quantity
            )
            
            if triggered_rule:
                runtime_minutes = (datetime.now() - start_time).total_seconds() / 60
                logging.info(f"Rule triggered: {triggered_rule}")
                logging.info(f"Stopping after {runtime_minutes:.1f} minutes")
                break
            
            # Log current phase for GUI detection
            current_phase = getattr(plan, 'state', {}).get('phase', 'unknown')
            logging.info(f"phase: {current_phase}")
            
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
