from helpers.runtime_utils import ipc


def get_var(var_id: int, timeout: float = 0.35) -> int | None:
    """
    Ask the IPC server for a varbit value by ID.
    Returns the integer value, or None on error.
    """
    resp = ipc.get_var(int(var_id), timeout=timeout)
    if not resp or not resp.get("ok"):
        return None
    return resp.get("value")


def achievement_diary_task_completed(varbit_id: int, timeout: float = 0.35) -> bool | None:
    """
    Check if a specific achievement diary task is completed.
    
    Args:
        varbit_id: The varbit ID for the achievement diary task.
                   You can find these using RuneLite's Var Inspector plugin
                   or by looking up the task on the OSRS Wiki.
        timeout: Optional timeout in seconds for the IPC call.
    
    Returns:
        - True if the task is completed (varbit value > 0)
        - False if the task is not completed (varbit value == 0)
        - None if the varbit could not be read or doesn't exist
    
    Example:
        # Check if Varrock Medium diary is completed (varbit ID example)
        if achievement_diary_task_completed(1234):
            print("Varrock Medium diary completed!")
    """
    value = get_var(varbit_id, timeout=timeout)
    if value is None:
        return None
    return bool(value > 0)


def has_blast_furnace_foreman_permission(timeout: float = 0.35) -> bool | None:
    """
    Check if the player has the Blast Furnace Foreman's permission to use
    the Blast Furnace for free (Fremennik Hard diary task completion).
    
    This checks if the Fremennik Hard diary is completed, which grants
    free access to the Blast Furnace.
    
    Args:
        timeout: Optional timeout in seconds for the IPC call.
    
    Returns:
        - True if the Fremennik Hard diary is completed (permission granted)
        - False if the diary is not completed (no free access)
        - None if the varbit could not be read
    
    Varbit ID: 4493 (FREMENNIK_DIARY_HARD_COMPLETE)
    
    Example:
        if has_blast_furnace_foreman_permission():
            print("Free Blast Furnace access available!")
        else:
            print("Need to pay for Blast Furnace access")
    """
    # VarbitID.FREMENNIK_DIARY_HARD_COMPLETE = 4493
    return achievement_diary_task_completed(4493, timeout=timeout)
