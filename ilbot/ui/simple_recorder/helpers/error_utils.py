#!/usr/bin/env python3
"""
Error Handling Utilities
========================

This module provides utilities for enhanced error handling with detailed
information including file names, line numbers, stack traces, and context.
"""

import traceback
import logging
from typing import Dict, Any, Optional


def format_error_details(exception: Exception, 
                        context: Optional[Dict[str, Any]] = None,
                        plan_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Format detailed error information for logging and debugging.
    
    Args:
        exception: The exception that occurred
        context: Additional context information (e.g., phase, state)
        plan_id: The plan ID for logging context
    
    Returns:
        Dictionary containing detailed error information
    """
    error_details = {
        "error": str(exception),
        "error_type": type(exception).__name__,
        "traceback": traceback.format_exc(),
        "context": context or {}
    }
    
    # Extract file and line information
    if exception.__traceback__:
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            error_details.update({
                "file": last_frame.filename,
                "line": last_frame.lineno,
                "function": last_frame.name,
                "code": last_frame.line
            })
    
    return error_details


def log_detailed_error(exception: Exception,
                      plan_id: str,
                      phase: Optional[str] = None,
                      additional_context: Optional[Dict[str, Any]] = None):
    """
    Log detailed error information with proper formatting.
    
    Args:
        exception: The exception that occurred
        plan_id: The plan ID for logging context
        phase: The current phase when error occurred
        additional_context: Additional context information
    """
    context = {"phase": phase} if phase else {}
    if additional_context:
        context.update(additional_context)
    
    error_details = format_error_details(exception, context, plan_id)
    
    # Log the main error message
    error_msg = f"[{plan_id}] Error"
    if phase:
        error_msg += f" in phase '{phase}'"
    error_msg += f": {error_details['error_type']}: {error_details['error']}"
    
    logging.error(error_msg)
    
    # Log file and line information
    if "file" in error_details and "line" in error_details:
        logging.error(f"[{plan_id}] File: {error_details['file']}, Line: {error_details['line']}")
        if "function" in error_details:
            logging.error(f"[{plan_id}] Function: {error_details['function']}")
        if "code" in error_details:
            logging.error(f"[{plan_id}] Code: {error_details['code']}")
    
    # Log additional context
    if context:
        logging.error(f"[{plan_id}] Context: {context}")
    
    # Log full traceback
    logging.error(f"[{plan_id}] Full traceback:\n{error_details['traceback']}")


def create_error_message(exception: Exception, 
                        plan_id: str,
                        phase: Optional[str] = None) -> str:
    """
    Create a concise error message for plan state storage.
    
    Args:
        exception: The exception that occurred
        plan_id: The plan ID for context
        phase: The current phase when error occurred
    
    Returns:
        Formatted error message string
    """
    error_details = format_error_details(exception)
    
    error_msg = f"{error_details['error_type']}: {error_details['error']}"
    
    if "file" in error_details and "line" in error_details:
        error_msg += f" (File: {error_details['file']}, Line: {error_details['line']})"
    
    if phase:
        error_msg += f" [Phase: {phase}]"
    
    return error_msg


def safe_execute(func, 
                plan_id: str,
                phase: Optional[str] = None,
                default_return: Any = None,
                log_errors: bool = True) -> Any:
    """
    Safely execute a function with detailed error handling.
    
    Args:
        func: Function to execute
        plan_id: Plan ID for logging context
        phase: Current phase for context
        default_return: Value to return if function fails
        log_errors: Whether to log errors (default: True)
    
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            log_detailed_error(e, plan_id, phase)
        return default_return


# Example usage and testing
if __name__ == "__main__":
    # Test the error handling utilities
    def test_function():
        raise ValueError("This is a test error")
    
    try:
        test_function()
    except Exception as e:
        log_detailed_error(e, "TEST_PLAN", "TEST_PHASE", {"test_param": "test_value"})
        error_msg = create_error_message(e, "TEST_PLAN", "TEST_PHASE")
        print(f"Error message: {error_msg}")
