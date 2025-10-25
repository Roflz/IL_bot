# Runtime utilities - Global instances and utility functions
import time
import logging
from ..services.action_executor import ActionExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# ===== GLOBAL INSTANCES =====

# Global UI instance
_ui = None

def get_ui():
    """Get the global UI instance."""
    return _ui

def set_ui(ui_instance):
    """Set the global UI instance."""
    global _ui
    _ui = ui_instance

# Global IPC instance
_ipc = None

def get_ipc():
    """Get the global IPC instance."""
    return _ipc

def set_ipc(ipc_instance):
    """Set the global IPC instance."""
    global _ipc
    _ipc = ipc_instance

# Global action executor instance
_action_executor = None

def get_action_executor():
    """Get the global action executor instance."""
    return _action_executor

def set_action_executor(executor_instance):
    """Set the global action executor instance."""
    global _action_executor
    _action_executor = executor_instance

# Global rule parameters
_rule_params = None

def get_rule_params():
    """Get the global rule parameters."""
    return _rule_params

def set_rule_params(params):
    """Set the global rule parameters."""
    global _rule_params
    _rule_params = params

# ===== UTILITY FUNCTIONS =====

def dispatch(step):
    """
    Global dispatch function for executing actions.
    
    Usage anywhere:
        from ..helpers.runtime_utils import dispatch
        dispatch({"click": {"type": "point", "x": 100, "y": 200}})
    """
    if _action_executor is None:
        raise RuntimeError("Action executor not initialized. Make sure run_rj_loop.py has been started.")
    
    return _action_executor.dispatch(step)

# ===== MODULE-LEVEL EXPORTS =====

# Create a simple class that acts as a reference to the private variables
class _Ref:
    def __init__(self, getter):
        self._getter = getter
    
    def __getattr__(self, name):
        obj = self._getter()
        if obj is None:
            raise RuntimeError("IPC not initialized. Make sure run_rj_loop.py has been started.")
        return getattr(obj, name)
    
    def __call__(self, *args, **kwargs):
        obj = self._getter()
        if obj is None:
            raise RuntimeError("IPC not initialized. Make sure run_rj_loop.py has been started.")
        return obj(*args, **kwargs)

# Create references that will always get the current value
ipc = _Ref(lambda: _ipc)
ui = _Ref(lambda: _ui)

# Rule params reference
def _get_rule_params():
    if _rule_params is None:
        return None
    return _rule_params
