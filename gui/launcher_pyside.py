"""
RuneLite Launcher Module (PySide6)
===================================

Handles launching and managing RuneLite instances.
Launch runs net.runelite.client.RuneLite.main() directly (same as IntelliJ Run RuneLite.main()),
with instance setup (credentials, settings) from launch-runelite.ps1.
"""

from PySide6.QtWidgets import QWidget, QMessageBox, QListWidget, QPushButton, QLabel, QSpinBox, QApplication
from PySide6.QtCore import QTimer, QEvent
from typing import List, Optional, Callable, Dict, Set, Any
import subprocess
import threading
import os
import random
import logging
from pathlib import Path

# Valid F2P worlds for random selection when defaultWorld is 0 (same as launch-runelite.ps1)
VALID_WORLDS = [308, 316, 326, 379, 383, 398, 417, 437, 455, 469, 483, 497, 499, 537, 552, 553, 554, 555, 571]


def _kill_process_tree(pid: int, log_callback: Optional[Callable] = None) -> bool:
    """Terminate process and all its children (required on Windows: gradlew spawns java). Returns True if any process was stopped."""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gone, still = psutil.wait_procs(children, timeout=3)
        for p in still:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.terminate()
            parent.wait(3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return True
    except Exception as e:
        if log_callback:
            log_callback(f"Error killing process tree for PID {pid}: {e}", "warning")
        return False


def _get_all_pids_in_tree(pid: int) -> List[int]:
    """Return [pid] plus all descendant PIDs (children recursively). Uses psutil."""
    try:
        import psutil
        pids = [pid]
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                pids.append(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids
    except Exception:
        return [pid]


# Event to post log from background thread to main thread (main window customEvent runs event.callback())
class _LogFromThreadEvent(QEvent):
    log_from_thread = True  # so main window just runs callback without "Processing cleanup event" spam
    def __init__(self, msg: str, level: str, log_callback: Callable):
        super().__init__(QEvent.Type.User)
        self.msg = msg
        self.level = level
        self.log_callback = log_callback
        self.callback = lambda: log_callback(msg, level)


def _post_log_to_main_thread(root: QWidget, msg: str, level: str, log_callback: Callable):
    """Post a log message to be run on the main thread (required when called from background thread)."""
    if root and log_callback:
        QApplication.postEvent(root, _LogFromThreadEvent(msg, level, log_callback))


# Splash uses title "RuneLite Launcher" (SplashScreen.java); main client uses "RuneLite" or "RuneLite - ...". Both are JFrame -> SunAwtFrame, so we distinguish by title.
RUNELITE_SPLASH_TITLE = "RuneLite Launcher"

def _is_main_runelite_window_title(title: str) -> bool:
    """True if this is the main client window title, not the splash (RuneLite Launcher)."""
    if not title or "RuneLite" not in title:
        return False
    # Reject splash: exact match or starts with "RuneLite Launcher"
    t = title.strip()
    if t == RUNELITE_SPLASH_TITLE or t.startswith(RUNELITE_SPLASH_TITLE + " "):
        return False
    return True

# Script approach (launch-runelite.ps1): find window with title "RuneLite" (not "RuneLite Launcher"), wait up to 25s, then ShowWindow(SW_MAXIMIZE)
def _find_runelite_hwnd_for_pids(pids_set: Set[int], debug_log: Optional[Callable[[str], None]] = None):
    """Windows-only: return HWND of first top-level window that is main client (title RuneLite, not splash) owned by one of the PIDs, or None."""
    if os.name != "nt" or not pids_set:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found_hwnd = [None]
        windows_checked = []

        def enum_cb(hwnd, _):
            wpid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            pid = wpid.value
            
            if pid not in pids_set:
                return True
            
            buf = ctypes.create_unicode_buffer(512)
            title = ""
            if user32.GetWindowTextW(hwnd, buf, 512) > 0:
                title = buf.value
            
            # Get window class for debugging
            class_buf = ctypes.create_unicode_buffer(256)
            class_name = ""
            if user32.GetClassNameW(hwnd, class_buf, 256) > 0:
                class_name = class_buf.value
            
            # Check visibility
            is_visible = user32.IsWindowVisible(hwnd) != 0
            
            windows_checked.append({
                'hwnd': hwnd,
                'pid': pid,
                'title': title,
                'class': class_name,
                'visible': is_visible
            })
            
            if title:
                is_main = _is_main_runelite_window_title(title)
                if debug_log:
                    reason = "ACCEPTED (main client)" if is_main else f"REJECTED (splash or other: '{title}')"
                    debug_log(f"[Window Check] PID {pid}, HWND 0x{hwnd:X}, Title: '{title}', Class: {class_name}, Visible: {is_visible} - {reason}")
                
                if is_main:
                    found_hwnd[0] = hwnd
                    return False
            
            return True
        
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        
        if debug_log and not found_hwnd[0] and windows_checked:
            debug_log(f"[Window Check] Checked {len(windows_checked)} window(s) from target PIDs, none matched main client")
            for win in windows_checked:
                if "RuneLite" in win['title']:
                    debug_log(f"  - Found RuneLite-related window: PID {win['pid']}, Title: '{win['title']}', Class: {win['class']}")
        
        return found_hwnd[0]
    except Exception as e:
        if debug_log:
            debug_log(f"[Window Check] Error in _find_runelite_hwnd_for_pids: {e}")
        return None


def _find_runelite_hwnd_any_pid(debug_log: Optional[Callable[[str], None]] = None):
    """Windows-only: return (hwnd, pid) of first top-level window that is main client (title RuneLite, not splash), or (None, None)."""
    if os.name != "nt":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found = [None, None]  # hwnd, pid
        runelite_windows = []  # Track all RuneLite-related windows found

        def enum_cb(hwnd, _):
            wpid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            pid = wpid.value
            
            buf = ctypes.create_unicode_buffer(512)
            title = ""
            if user32.GetWindowTextW(hwnd, buf, 512) > 0:
                title = buf.value
            
            # Get window class for debugging
            class_buf = ctypes.create_unicode_buffer(256)
            class_name = ""
            if user32.GetClassNameW(hwnd, class_buf, 256) > 0:
                class_name = class_buf.value
            
            # Check visibility
            is_visible = user32.IsWindowVisible(hwnd) != 0
            
            if title and "RuneLite" in title:
                is_main = _is_main_runelite_window_title(title)
                runelite_windows.append({
                    'hwnd': hwnd,
                    'pid': pid,
                    'title': title,
                    'class': class_name,
                    'visible': is_visible,
                    'is_main': is_main
                })
                
                if debug_log:
                    reason = "ACCEPTED (main client)" if is_main else f"REJECTED (splash: '{title}')"
                    debug_log(f"[Window Check Any PID] PID {pid}, HWND 0x{hwnd:X}, Title: '{title}', Class: {class_name}, Visible: {is_visible} - {reason}")
                
                if is_main:
                    found[0], found[1] = hwnd, pid
                    return False
            
            return True
        
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        
        if debug_log:
            if found[0] is not None:
                debug_log(f"[Window Check Any PID] Found main client window: HWND 0x{found[0]:X}, PID {found[1]}")
            elif runelite_windows:
                debug_log(f"[Window Check Any PID] Found {len(runelite_windows)} RuneLite-related window(s), but none matched main client:")
                for win in runelite_windows:
                    debug_log(f"  - PID {win['pid']}, HWND 0x{win['hwnd']:X}, Title: '{win['title']}', Class: {win['class']}, Visible: {win['visible']}")
            else:
                debug_log("[Window Check Any PID] No RuneLite-related windows found")
        
        return (found[0], found[1]) if found[0] is not None else (None, None)
    except Exception as e:
        if debug_log:
            debug_log(f"[Window Check Any PID] Error: {e}")
        return None, None


def _wait_for_runelite_window_and_maximize(
    root_pid: int,
    log_callback: Optional[Callable],
    root: Optional[QWidget],
    timeout_sec: int = 25,
    poll_ms: int = 300,
):
    """Poll for a top-level window titled 'RuneLite' owned by root_pid or any of its children (re-queries tree each poll so we pick up Java once it spawns). Run in background thread."""
    import time

    def log(msg: str, level: str = "info"):
        if log_callback and root:
            _post_log_to_main_thread(root, msg, level, log_callback)
    
    # Extended timeout for first-time JDK downloads
    extended_timeout = 900  # 15 minutes for JDK download + build + launch
    log(f"Waiting for RuneLite window (extended timeout: {extended_timeout}s to allow for JDK download if needed)...", "info")

    deadline = time.time() + extended_timeout
    poll_count = 0
    root_create_time = None
    last_log_time = time.time()
    last_debug_log_time = time.time()
    debug_interval = 5.0  # Log debug info every 5 seconds
    
    # Create debug log function that only logs periodically to avoid spam
    def debug_log(msg: str):
        nonlocal last_debug_log_time
        now = time.time()
        if now - last_debug_log_time >= debug_interval:
            log(msg, "info")
            last_debug_log_time = now
    
    while time.time() < deadline:
        # Log progress every 30 seconds
        if time.time() - last_log_time >= 30:
            elapsed = int(time.time() - (deadline - extended_timeout))
            remaining = int(deadline - time.time())
            log(f"Still waiting for RuneLite window... (elapsed: {elapsed}s, remaining: {remaining}s)", "info")
            last_log_time = time.time()
        
        pids_set = set(_get_all_pids_in_tree(root_pid))
        debug_log(f"[Maximize] Poll #{poll_count + 1}: Checking {len(pids_set)} PID(s) in process tree")
        
        hwnd = _find_runelite_hwnd_for_pids(pids_set, debug_log=debug_log)
        if hwnd is None:
            debug_log("[Maximize] No window found in target PIDs, checking all processes...")
            hwnd, win_pid = _find_runelite_hwnd_any_pid(debug_log=debug_log)
            if hwnd is not None and win_pid is not None:
                if win_pid in pids_set:
                    debug_log(f"[Maximize] Found window belongs to target process tree (PID {win_pid})")
                else:
                    try:
                        import psutil
                        if root_create_time is None:
                            try:
                                root_create_time = psutil.Process(root_pid).create_time()
                            except Exception:
                                root_create_time = 0
                        proc = psutil.Process(win_pid)
                        proc_name = proc.name() or ""
                        proc_create_time = proc.create_time()
                        is_java = "java" in proc_name.lower()
                        is_recent = proc_create_time >= root_create_time
                        
                        debug_log(f"[Maximize] Window PID {win_pid} not in target tree. Process: {proc_name}, Created: {proc_create_time:.1f}, Root created: {root_create_time:.1f}, Is Java: {is_java}, Is recent: {is_recent}")
                        
                        if is_recent and is_java:
                            debug_log(f"[Maximize] Accepting window from recent Java process (PID {win_pid})")
                        else:
                            debug_log(f"[Maximize] Rejecting window - process not recent or not Java")
                            hwnd = None
                    except Exception as e:
                        debug_log(f"[Maximize] Error checking process: {e}")
                        hwnd = None
        else:
            debug_log(f"[Maximize] Found window in target process tree: HWND 0x{hwnd:X}")
        
        poll_count += 1
        now = time.time()
        if hwnd is not None:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                SW_MAXIMIZE = 3
                
                # Get final window info before maximizing
                buf = ctypes.create_unicode_buffer(512)
                title = ""
                if user32.GetWindowTextW(hwnd, buf, 512) > 0:
                    title = buf.value
                
                wpid = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
                
                log(f"Maximizing RuneLite window: HWND 0x{hwnd:X}, PID {wpid.value}, Title: '{title}'", "info")
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
                log("Successfully maximized RuneLite window.", "success")
            except Exception as e:
                log(f"Failed to maximize window: {e}", "error")
            return
        time.sleep(poll_ms / 1000.0)

    elapsed = int(time.time() - (deadline - extended_timeout))
    log(f"Timed out waiting for 'RuneLite' window after {elapsed}s. The process may still be downloading JDK or building.", "warning")
    log("Check the Gradle output above for progress. The window may appear shortly.", "info")


# settings.properties template (same content as launch-runelite.ps1); {port} and {world} are substituted
SETTINGS_PROPERTIES_TEMPLATE = """
runelite.ipcinputplugin=true
ipcinput.port={port}
ipcinput.mode=AWT
ipcinput.hoverDelayMs=10
defaultworld.lastWorld={world}
defaultworld.useLastWorld=true
runelite.logouttimerplugin=true
logouttimer.idleTimeout=25
menuentryswapper.swapQuick=true
menuentryswapper.swapGEItemCollect=DEFAULT
menuentryswapper.swapBait=false
menuentryswapper.swapJewelleryBox=false
runelite.menuentryswapperplugin=true
menuentryswapper.npcShiftClickWalkHere=true
menuentryswapper.swapAdmire=true
menuentryswapper.swapHomePortal=HOME
menuentryswapper.shopBuy=OFF
menuentryswapper.swapDepositItems=false
menuentryswapper.swapArdougneCloak=WEAR
menuentryswapper.swapTan=false
menuentryswapper.swapStairsShiftClick=CLIMB
menuentryswapper.swapTrade=true
menuentryswapper.swapBoxTrap=true
menuentryswapper.swapPay=true
menuentryswapper.swapBones=false
menuentryswapper.groundItemShiftClickWalkHere=true
menuentryswapper.swapEssenceMineTeleport=false
menuentryswapper.shiftClickCustomization=true
menuentryswapper.swapTemporossLeave=false
menuentryswapper.swapHerbs=false
menuentryswapper.bankDepositShiftClick=OFF
menuentryswapper.objectLeftClickCustomization=true
menuentryswapper.bankWithdrawShiftClick=OFF
menuentryswapper.swapMorytaniaLegs=WEAR
menuentryswapper.swapHarpoon=false
menuentryswapper.swapBanker=true
menuentryswapper.swapDesertAmulet=WEAR
menuentryswapper.swapPick=false
menuentryswapper.shopSell=OFF
menuentryswapper.swapHelp=true
menuentryswapper.swapAbyssTeleport=true
menuentryswapper.swapAssignment=true
menuentryswapper.objectShiftClickWalkHere=true
menuentryswapper.swapFairyRing=LAST_DESTINATION
menuentryswapper.leftClickCustomization=true
menuentryswapper.swapTravel=true
menuentryswapper.swapExchange=true
menuentryswapper.swapKaramjaGloves=WEAR
menuentryswapper.swapRadasBlessing=EQUIP
menuentryswapper.swapGEAbort=false
menuentryswapper.swapPortalNexus=false
menuentryswapper.swapPrivate=false
menuentryswapper.swapTeleToPoh=false
menuentryswapper.swapBirdhouseEmpty=true
menuentryswapper.swapChase=true
menuentryswapper.swapStairsLeftClick=CLIMB
menuentryswapper.swapTeleportItem=false
menuentryswapper.npcLeftClickCustomization=true
menuentryswapper.removeDeadNpcMenus=false
keyremapping.f10=48\\:0
keyremapping.f12=61\\:0
keyremapping.f11=45\\:0
keyremapping.left=65\\:0
runelite.keyremappingplugin=true
keyremapping.f1=49\\:0
keyremapping.f3=51\\:0
keyremapping.f2=50\\:0
keyremapping.cameraRemap=true
keyremapping.fkeyRemap=false
keyremapping.esc=27\\:0
keyremapping.control=0\\:128
keyremapping.f9=57\\:0
keyremapping.f8=56\\:0
keyremapping.f5=53\\:0
keyremapping.f4=52\\:0
keyremapping.f7=55\\:0
keyremapping.f6=54\\:0
keyremapping.up=87\\:0
keyremapping.space=32\\:0
keyremapping.down=83\\:0
keyremapping.right=68\\:0
"""


class RuneLiteLauncher:
    """Manages RuneLite instance launching and credential management."""
    
    def __init__(self, root: QWidget, config_vars: dict, base_port_var: Optional[QSpinBox],
                 launch_delay_var: Optional[QSpinBox],
                 credentials_listbox: QListWidget, selected_credentials_listbox: QListWidget,
                 selected_credentials: List[str], log_callback: Optional[Callable] = None,
                 instance_count_label: Optional[QLabel] = None,
                 launch_button: Optional[QPushButton] = None,
                 create_instance_tab_callback: Optional[Callable] = None,
                 instance_ports: Optional[Dict] = None,
                 cleanup_instances_callback: Optional[Callable] = None):
        """Initialize RuneLite launcher."""
        self.root = root
        self.config_vars = config_vars
        self.base_port_var = base_port_var
        self.launch_delay_var = launch_delay_var
        self.credentials_listbox = credentials_listbox
        self.selected_credentials_listbox = selected_credentials_listbox
        self.selected_credentials = selected_credentials
        self.log_callback = log_callback or (lambda msg, level='info': None)
        self.instance_count_label = instance_count_label
        self.launch_button = launch_button
        self.create_instance_tab_callback = create_instance_tab_callback
        self.instance_ports = instance_ports if instance_ports is not None else {}
        self.cleanup_instances_callback = cleanup_instances_callback
        self.runelite_process = None
        self.instance_tabs_timer = None
        self._last_launched_pid = None
        self._last_launched_pids = []  # list of {"pid", "port", "instance_index", "inst_home", "cred_name"}
    
    def populate_credentials(self):
        """Populate credentials listbox from credentials directory."""
        # Get credentials directory from config, or use default
        credentials_dir_str = self.config_vars.get("credentialsDir", "")
        if credentials_dir_str:
            credentials_dir = Path(credentials_dir_str)
        else:
            # Default to credentials folder in project root
            credentials_dir = Path(__file__).resolve().parent.parent / "credentials"
        
        self.log_callback(f"Loading credentials from: {credentials_dir}", 'info')
        
        if credentials_dir.exists():
            # Clear existing items
            self.credentials_listbox.clear()
            
            # Add all .properties files
            cred_files = sorted(credentials_dir.glob("*.properties"))
            for cred_file in cred_files:
                self.credentials_listbox.addItem(cred_file.name)
            
            self.log_callback(f"Loaded {len(cred_files)} credential(s)", 'info')
        else:
            self.log_callback(f"Credentials directory not found: {credentials_dir}", 'warning')
    
    def add_credential(self):
        """Add selected credential to launch list."""
        selected_items = self.credentials_listbox.selectedItems()
        for item in selected_items:
            cred_name = item.text()
            if cred_name not in self.selected_credentials:
                self.selected_credentials.append(cred_name)
        self.update_selected_credentials_display()
    
    def remove_credential(self):
        """Remove selected credential from launch list."""
        selected_items = self.selected_credentials_listbox.selectedItems()
        # Get indices and sort in reverse order to maintain indices when removing
        indices = sorted([self.selected_credentials_listbox.row(item) for item in selected_items], reverse=True)
        for index in indices:
            if 0 <= index < len(self.selected_credentials):
                self.selected_credentials.pop(index)
        self.update_selected_credentials_display()
    
    def move_credential_up(self):
        """Move selected credential up in the list."""
        selected_items = self.selected_credentials_listbox.selectedItems()
        if selected_items:
            index = self.selected_credentials_listbox.row(selected_items[0])
            if index > 0:
                item = self.selected_credentials.pop(index)
                self.selected_credentials.insert(index - 1, item)
                self.update_selected_credentials_display()
                # Reselect the moved item
                self.selected_credentials_listbox.setCurrentRow(index - 1)
    
    def move_credential_down(self):
        """Move selected credential down in the list."""
        selected_items = self.selected_credentials_listbox.selectedItems()
        if selected_items:
            index = self.selected_credentials_listbox.row(selected_items[0])
            if index < len(self.selected_credentials) - 1:
                item = self.selected_credentials.pop(index)
                self.selected_credentials.insert(index + 1, item)
                self.update_selected_credentials_display()
                # Reselect the moved item
                self.selected_credentials_listbox.setCurrentRow(index + 1)
    
    def clear_credentials(self):
        """Clear all selected credentials."""
        self.selected_credentials = []
        self.update_selected_credentials_display()

    def update_selected_credentials_display(self):
        """Update the selected credentials display."""
        self.selected_credentials_listbox.clear()
        for i, cred_name in enumerate(self.selected_credentials):
            self.selected_credentials_listbox.addItem(f"{i+1}. {cred_name}")
    
    def launch_runelite(self, save_config_callback: Optional[Callable] = None):
        """Launch RuneLite instances by running net.runelite.client.RuneLite.main() (same as IntelliJ).
        Uses config paths and per-instance setup (credentials, settings) from launch-runelite.ps1."""
        if not self.selected_credentials:
            QMessageBox.warning(self.root, "No Credentials Selected", "Please select at least one credential file.")
            return
        
        instance_count = len(self.selected_credentials)
        if instance_count <= 0:
            QMessageBox.critical(self.root, "Invalid Configuration", "Instance count must be greater than 0.")
            return
        
        if save_config_callback:
            save_config_callback()

        try:
            project_dir_str = self.config_vars.get("projectDir", "")
            base_dir_str = self.config_vars.get("baseDir", "")
            exports_base_str = self.config_vars.get("exportsBase", "")
            credentials_dir_str = self.config_vars.get("credentialsDir", "")
            if not project_dir_str:
                QMessageBox.critical(self.root, "Configuration Error", "Project directory not set. Set it in Setup & Configuration.")
                return
            if not credentials_dir_str:
                QMessageBox.critical(self.root, "Configuration Error", "Credentials directory not set.")
                return
            
            project_dir = Path(project_dir_str)
            base_dir = Path(base_dir_str) if base_dir_str else Path(__file__).resolve().parent.parent / "instances"
            exports_base = Path(exports_base_str) if exports_base_str else Path(__file__).resolve().parent.parent / "exports"
            credentials_dir = Path(credentials_dir_str)
            
            if not project_dir.exists():
                QMessageBox.critical(self.root, "Path Error", f"Project directory not found: {project_dir}")
                return
            if not credentials_dir.exists():
                QMessageBox.critical(self.root, "Path Error", f"Credentials directory not found: {credentials_dir}")
                return
            
            # ============================================================================
            # STEP 1/4: Check and merge latest RuneLite release commit
            # ============================================================================
            self.log_callback("=" * 70, "info")
            self.log_callback("[STEP 1/4] Checking for latest RuneLite release...", "info")
            self.log_callback("=" * 70, "info")
            merge_result = self._check_and_merge_latest_release(project_dir)
            if not merge_result["success"]:
                if merge_result.get("conflict"):
                    QMessageBox.critical(
                        self.root, "Merge Conflict",
                        f"Failed to merge latest RuneLite release due to merge conflicts.\n\n"
                        f"Latest release: {merge_result.get('latest_release', 'unknown')}\n"
                        f"Please resolve conflicts manually in the RuneLite repository and try again.\n\n"
                        f"Repository: {project_dir}"
                    )
                else:
                    QMessageBox.critical(
                        self.root, "Release Check Failed",
                        f"Failed to check/merge latest RuneLite release:\n{merge_result.get('error', 'Unknown error')}"
                    )
                return
            self.log_callback("[STEP 1/4] Release check completed successfully", "success")
            
            # ============================================================================
            # STEP 2/4: Resolve and verify Java toolchain JDK
            # ============================================================================
            self.log_callback("=" * 70, "info")
            self.log_callback("[STEP 2/4] Resolving Java toolchain JDK (this may take several minutes on first run)...", "info")
            self.log_callback("=" * 70, "info")
            toolchain_result = self._resolve_and_log_toolchain_jdk(project_dir)
            if not toolchain_result["success"]:
                QMessageBox.critical(
                    self.root, "Toolchain Resolution Failed",
                    f"Failed to resolve Java toolchain JDK:\n{toolchain_result.get('error', 'Unknown error')}"
                )
                return
            
            # ============================================================================
            # STEP 3/4: Build and run with Gradle
            # ============================================================================
            self.log_callback("=" * 70, "info")
            self.log_callback("[STEP 3/4] Building and launching RuneLite instances...", "info")
            self.log_callback("=" * 70, "info")
            import shutil
            if os.name == "nt":
                gradlew = project_dir / "gradlew.bat"
            else:
                gradlew = project_dir / "gradlew"
            if not gradlew.exists():
                gradlew = shutil.which("gradle")
            if not gradlew or (isinstance(gradlew, Path) and not gradlew.exists()):
                QMessageBox.critical(
                    self.root, "Gradle Required",
                    "Gradle not found: no gradlew or gradlew.bat in project directory, and gradle not in PATH. "
                    "The RuneLite project uses Gradle (same as IntelliJ). Use the project's Gradle wrapper or install Gradle."
                )
                return
            gradlew = str(gradlew) if isinstance(gradlew, Path) else gradlew

            self.log_callback(f"Using Gradle wrapper: {gradlew}", "info")
            self.log_callback(f"Using JDK: {toolchain_result.get('jdk_path', 'Unknown')}", "info")
            self.log_callback(f"JDK Version: {toolchain_result.get('jdk_version', 'Unknown')}", "info")

            default_world = 0  # 0 = random F2P world per instance
            base_port = 17000
            starting_port = self._get_next_available_port(base_port)
            self.log_callback(f"Launching {instance_count} instance(s) starting at port {starting_port} (Run RuneLite.main())", "info")
            
            base_dir.mkdir(parents=True, exist_ok=True)
            exports_base.mkdir(parents=True, exist_ok=True)
            pid_file = base_dir / "runelite-pids.txt"
            self._last_launched_pids = []

            for i in range(instance_count):
                port = starting_port + i
                inst_home = base_dir / f"inst_{i}"
                exp_dir = exports_base / f"inst_{i}"
                inst_home.mkdir(parents=True, exist_ok=True)
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                cred_name = self.selected_credentials[i % len(self.selected_credentials)]
                source_cred = credentials_dir / cred_name
                if not source_cred.exists() and not cred_name.endswith(".properties"):
                    source_cred = credentials_dir / f"{cred_name}.properties"
                config_dir = inst_home / ".runelite"
                config_dir.mkdir(parents=True, exist_ok=True)
                target_cred = config_dir / "credentials.properties"
                if source_cred.exists():
                    import shutil
                    shutil.copy2(source_cred, target_cred)
                    self.log_callback(f"Instance {i}: copied credentials {cred_name}", "info")
                
                profiles2 = config_dir / "profiles2"
                if profiles2.exists():
                    import shutil
                    shutil.rmtree(profiles2)
                
                world = default_world if default_world != 0 else random.choice(VALID_WORLDS)
                settings_file = config_dir / "settings.properties"
                settings_content = SETTINGS_PROPERTIES_TEMPLATE.format(port=port, world=world)
                settings_file.write_text(settings_content.strip(), encoding="utf-8")
                
                inst_home_str = str(inst_home)
                gradle_args = [
                    gradlew,
                    ":client:run",
                    f"-Puser.home={inst_home_str}",
                    f"-Prl.instance={i}",
                ]
                self.log_callback(f"Instance {i}: Starting Gradle build/run process...", "info")
                self.log_callback(f"  Port: {port}", "info")
                self.log_callback(f"  Home: {inst_home}", "info")
                self.log_callback(f"  Command: {' '.join(gradle_args)}", "info")
                self.log_callback(f"  Note: First-time toolchain JDK download may take 5-15 minutes. Please wait...", "warning")
                
                env = os.environ.copy()
                gradle_log = inst_home / "gradle-run.log"
                
                # Use unbuffered output to ensure real-time logging
                proc = subprocess.Popen(
                    gradle_args,
                    cwd=str(project_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # Unbuffered for real-time output
                    env=env,
                )
                with open(pid_file, "a", encoding="ascii") as f:
                    f.write(f"{proc.pid},{i},{port},{inst_home_str}\n")
                self._last_launched_pid = proc.pid
                entry = {
                    "pid": proc.pid,
                    "port": port,
                    "instance_index": i,
                    "inst_home": inst_home_str,
                    "cred_name": cred_name,
                }
                self._last_launched_pids.append(entry)
                self.log_callback(
                    f"Instance #{i}: Gradle process started (PID={proc.pid})", 
                    "info",
                )
                self.log_callback(
                    f"Instance #{i}: Streaming Gradle output (build/download progress will appear below)...",
                    "info",
                )

                def stream_gradle_output(process, instance_idx, log_path, callback):
                    """Read Gradle stdout line by line; post to GUI and append to log file with enhanced logging."""
                    import time
                    last_activity_time = time.time()
                    line_count = 0
                    try:
                        with open(log_path, "w", encoding="utf-8") as log_file:
                            # Log that we're starting to stream
                            callback(f"[Gradle inst_{instance_idx}] Starting to capture Gradle output...", "info")
                            
                            for line in iter(process.stdout.readline, ""):
                                line = line.rstrip()
                                if line:
                                    line_count += 1
                                    last_activity_time = time.time()
                                    log_file.write(line + "\n")
                                    log_file.flush()
                                    
                                    # Filter out repetitive IPC polling messages to reduce log noise
                                    if "[IPC] recv raw: {\"cmd\":\"get_player\"}" in line:
                                        # Skip these repetitive messages - they're normal operation
                                        continue
                                    
                                    # Log every line to GUI with appropriate prefix
                                    # Highlight important messages
                                    if "Downloading" in line or "Download" in line:
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "info")
                                    elif "BUILD" in line.upper() or "Task" in line:
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "info")
                                    elif "error" in line.lower() or "Error" in line or "FAILURE" in line:
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "error")
                                    elif "toolchain" in line.lower() or "JDK" in line or ("java" in line.lower() and "version" in line.lower()):
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "info")
                                    elif "Starting process" in line or "java.exe" in line:
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "info")
                                    elif "[IPC]" in line and "get_player" in line:
                                        # Skip IPC polling messages (already filtered above, but double-check)
                                        continue
                                    else:
                                        callback(f"[Gradle inst_{instance_idx}] {line}", "info")
                                    
                                    # Periodic activity indicator (every 50 lines or 30 seconds)
                                    if line_count % 50 == 0:
                                        callback(f"[Gradle inst_{instance_idx}] Progress: {line_count} lines processed...", "info")
                                
                                # Check for long silence (no output for 60 seconds)
                                if time.time() - last_activity_time > 60:
                                    callback(
                                        f"[Gradle inst_{instance_idx}] ⏳ No output for 60s - process may be downloading JDK or building (this is normal, please wait)...",
                                        "warning"
                                    )
                                    last_activity_time = time.time()
                            
                            callback(f"[Gradle inst_{instance_idx}] Finished capturing output ({line_count} total lines)", "info")
                        process.stdout.close()
                    except Exception as e:
                        callback(f"[Gradle inst_{instance_idx}] Stream error: {e}", "error")
                        import traceback
                        callback(f"[Gradle inst_{instance_idx}] Traceback: {traceback.format_exc()}", "error")
                    finally:
                        ret = process.poll()
                        if ret is not None:
                            if ret == 0:
                                callback(f"[Gradle inst_{instance_idx}] Process completed successfully (exit code {ret})", "success")
                            else:
                                callback(f"[Gradle inst_{instance_idx}] Process ended with error code {ret}", "error")

                threading.Thread(
                    target=stream_gradle_output,
                    args=(proc, i, gradle_log, self.log_callback),
                    daemon=True,
                ).start()

            # ============================================================================
            # STEP 4/4: Wait for RuneLite windows and maximize
            # ============================================================================
            self.log_callback("=" * 70, "info")
            self.log_callback("[STEP 4/4] Waiting for RuneLite windows to appear...", "info")
            self.log_callback("=" * 70, "info")
            
            # Auto-maximize each launched instance's RuneLite window (same as launch-runelite.ps1: wait for "RuneLite" then ShowWindow)
            if os.name == "nt" and self._last_launched_pids:
                self.log_callback("Waiting for RuneLite window(s) to appear (this may take several minutes if JDK is downloading)...", "info")
                for entry in self._last_launched_pids:
                    threading.Thread(
                        target=_wait_for_runelite_window_and_maximize,
                        args=(entry["pid"], self.log_callback, self.root),
                        daemon=True,
                    ).start()

            if self.launch_button:
                self.launch_button.setEnabled(False)
            self.log_callback("=" * 70, "info")
            self.log_callback("All launch steps initiated. Monitor output above for progress.", "success")
            self.log_callback("Note: First-time JDK download can take 5-15 minutes. Be patient!", "warning")
            self.log_callback("=" * 70, "info")
            
            if self.create_instance_tab_callback:
                self.instance_tabs_timer = QTimer(self.root)
                self.instance_tabs_timer.setSingleShot(True)
                self.instance_tabs_timer.timeout.connect(self._create_instance_tabs_wrapper)
                self.instance_tabs_timer.start(2000)
            
            def re_enable():
                if self.launch_button:
                    self.launch_button.setEnabled(True)
            QTimer.singleShot(3000, re_enable)
            
        except Exception as e:
            import traceback
            self.log_callback(f"Error launching RuneLite: {str(e)}", "error")
            self.log_callback(traceback.format_exc(), "error")
            QMessageBox.critical(self.root, "Launch Error", f"Failed to launch RuneLite instances: {str(e)}")
            if self.launch_button:
                self.launch_button.setEnabled(True)
    
    def build_runelite_only(self):
        """Build RuneLite project only (without launching instances). Uses Gradle, same as IntelliJ. Not wired to UI; kept for programmatic use."""
        try:
            project_dir_str = self.config_vars.get("projectDir", "")
            if not project_dir_str:
                QMessageBox.warning(self.root, "Configuration Error", "Project directory not configured. Set it in Setup & Configuration.")
                return
            project_dir_path = Path(project_dir_str)
            if not project_dir_path.exists():
                QMessageBox.critical(self.root, "Path Error", f"Project directory not found: {project_dir_path}")
                return

            import shutil
            if os.name == "nt":
                gradlew = project_dir_path / "gradlew.bat"
            else:
                gradlew = project_dir_path / "gradlew"
            if not gradlew.exists():
                gradlew = shutil.which("gradle")
            if not gradlew or (isinstance(gradlew, Path) and not gradlew.exists()):
                QMessageBox.critical(self.root, "Gradle Required", "Gradle not found: no gradlew or gradlew.bat in project directory, and gradle not in PATH.")
                return
            gradlew = str(gradlew) if isinstance(gradlew, Path) else gradlew

            self.log_callback("Starting RuneLite build with Gradle...", "info")
            self.log_callback(f"Project directory: {project_dir_path}", "info")

            def build_in_thread():
                try:
                    build_process = subprocess.Popen(
                        [gradlew, ":client:classes"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=str(project_dir_path),
                    )
                    while True:
                        output = build_process.stdout.readline()
                        if output == "" and build_process.poll() is not None:
                            break
                        if output:
                            line = output.strip()
                            if line:
                                self.log_callback(f"[Build] {line}", "info")
                    return_code = build_process.wait()
                    if return_code != 0:
                        self.log_callback(f"Gradle build failed with return code {return_code}", "error")
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(0, lambda: QMessageBox.critical(
                            self.root, "Build Failed",
                            f"Gradle build failed with return code {return_code}. Check the log for details."
                        ))
                        return
                    self.log_callback("Gradle build completed successfully", "success")
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: QMessageBox.information(
                        self.root, "Build Complete",
                        "RuneLite build completed successfully (Gradle :client:classes)."
                    ))
                except Exception as e:
                    self.log_callback(f"Error during build: {str(e)}", "error")
                    import traceback
                    self.log_callback(traceback.format_exc(), "error")
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: QMessageBox.critical(
                        self.root, "Build Error", f"Error during build: {str(e)}"
                    ))

            threading.Thread(target=build_in_thread, daemon=True).start()

        except Exception as e:
            self.log_callback(f"Error starting build: {str(e)}", "error")
            QMessageBox.critical(self.root, "Build Error", f"Failed to start build: {str(e)}")
    
    def _create_instance_tabs_wrapper(self):
        """Wrapper to call create_instance_tabs - ensures it runs in main thread."""
        self.log_callback("_create_instance_tabs_wrapper called", 'info')
        self.create_instance_tabs()
    
    def _get_next_available_port(self, start_port: int = 17000) -> int:
        """Find the next available port starting from start_port."""
        # Debug: Check what we're seeing
        if self.instance_ports:
            used_ports = set(self.instance_ports.values())
            self.log_callback(f"_get_next_available_port: instance_ports dict has {len(self.instance_ports)} entries: {dict(self.instance_ports)}", 'info')
            self.log_callback(f"_get_next_available_port: used_ports set: {sorted(used_ports)}", 'info')
        else:
            used_ports = set()
            self.log_callback(f"_get_next_available_port: instance_ports is empty or None", 'info')
        
        port = start_port
        while port in used_ports:
            port += 1
        self.log_callback(f"_get_next_available_port: returning port {port}", 'info')
        return port
    
    def create_instance_tabs(self):
        """Create tabs for each launched RuneLite instance."""
        self.log_callback("create_instance_tabs() method called", 'info')
        
        if not self.create_instance_tab_callback:
            self.log_callback("No create_instance_tab_callback available", 'error')
            return
        
        try:
            self.log_callback("Starting to create instance tabs...", 'info')
            base_port = 17000  # Default since removed from UI
            
            # Debug: Check instance_ports reference
            if self.instance_ports:
                self.log_callback(f"create_instance_tabs: instance_ports dict has {len(self.instance_ports)} entries: {dict(self.instance_ports)}", 'info')
                used_ports = set(self.instance_ports.values())
            else:
                used_ports = set()
                self.log_callback(f"create_instance_tabs: instance_ports is empty or None", 'info')
            
            self.log_callback(f"Currently used ports: {sorted(used_ports)}", 'info')
            self.log_callback(f"Using base port: {base_port}, Credentials: {self.selected_credentials}", 'info')
            
            # Find starting port (next available port from base_port)
            current_port = self._get_next_available_port(base_port)
            self.log_callback(f"Found starting port: {current_port}", 'info')
            
            for i, cred_name in enumerate(self.selected_credentials):
                # Extract username from credential filename (remove .properties)
                username = cred_name.replace('.properties', '')
                
                # Use current_port and increment for next credential
                port = current_port + i
                
                # Make sure this port isn't already taken (in case of concurrent launches)
                while port in used_ports:
                    port += 1
                used_ports.add(port)  # Mark as used for this batch
                
                self.log_callback(f"Creating tab for {username} on port {port}", 'info')
                
                if self.create_instance_tab_callback:
                    try:
                        self.create_instance_tab_callback(username, port)
                        self.log_callback(f"Callback executed for {username}", 'info')
                    except Exception as e:
                        self.log_callback(f"Error in callback for {username}: {str(e)}", 'error')
                        import traceback
                        self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
            
            self.log_callback("Instance tabs creation completed", 'success')
            
        except Exception as e:
            self.log_callback(f"Error creating instance tabs: {str(e)}", 'error')
            import traceback
            self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
    
    def stop_runelite(self, stop_all_callback: Optional[Callable] = None):
        """Stop all RuneLite instances (runs in background thread to avoid blocking UI)."""
        self.log_callback("Stop All Instances button clicked", 'info')
        
        # Run the actual stopping in a background thread to avoid blocking the UI
        def stop_in_thread():
            try:
                # First stop all running plans
                self.log_callback("Stopping all running plans...", 'info')
                if stop_all_callback:
                    try:
                        stop_all_callback()
                        self.log_callback("Called stop_all_callback", 'info')
                    except Exception as e:
                        self.log_callback(f"Error in stop_all_callback: {str(e)}", 'error')
                else:
                    self.log_callback("No stop_all_callback provided", 'warning')
                
                # Then stop RuneLite instances using the PID file
                # PID file is stored at baseDir/runelite-pids.txt (where baseDir comes from config)
                # Try to get baseDir from config, fallback to project_dir/instances
                base_dir = self.config_vars.get("baseDir", "")
                if base_dir:
                    pid_file = Path(base_dir) / "runelite-pids.txt"
                else:
                    # Fallback: use project_dir/instances
                    project_dir = Path(__file__).resolve().parent.parent
                    pid_file = project_dir / "instances" / "runelite-pids.txt"
                self.log_callback(f"Checking for PID file at: {pid_file}", 'info')
                
                if pid_file.exists():
                    try:
                        import psutil
                        stopped_count = 0
                        seen_pids = set()  # Track PIDs we've already processed to avoid duplicates
                        with open(pid_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split(',')
                                if len(parts) >= 1:
                                    try:
                                        pid = int(parts[0])
                                        # Skip if we've already processed this PID (duplicate entries)
                                        if pid in seen_pids:
                                            continue
                                        seen_pids.add(pid)
                                        
                                        # Kill process tree (gradlew + child java); on Windows terminate() does not kill children
                                        if _kill_process_tree(pid, self.log_callback):
                                            stopped_count += 1
                                            self.log_callback(f"Stopped RuneLite instance (tree) PID {pid}", 'info')
                                    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as e:
                                        raw = parts[0] if parts else "?"
                                        self.log_callback(f"Could not stop PID {raw}: {str(e)}", 'warning')
                        
                        # Remove PID file
                        pid_file.unlink()
                        self.log_callback(f"All RuneLite instances stopped ({stopped_count} processes)", 'success')
                    except ImportError:
                        self.log_callback("psutil not available, cannot stop processes", 'warning')
                    except Exception as e:
                        self.log_callback(f"Error stopping processes: {str(e)}", 'error')
                        import traceback
                        self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
                else:
                    self.log_callback(f"No RuneLite instances found to stop (PID file not found at {pid_file})", 'warning')
                
                # Clean up instances from display and dictionaries (even if PID file didn't exist)
                # This ensures UI is cleaned up if instances were stopped manually
                # The issue: QTimer.singleShot called from background thread doesn't work
                # Solution: Use the root widget's event loop to schedule the callback
                self.log_callback("Reached cleanup section", 'info')
                self.log_callback(f"Checking cleanup_instances_callback: hasattr={hasattr(self, 'cleanup_instances_callback')}, callback={getattr(self, 'cleanup_instances_callback', None)}", 'info')
                if hasattr(self, 'cleanup_instances_callback') and self.cleanup_instances_callback:
                    cleanup_callback = self.cleanup_instances_callback
                    def execute_cleanup():
                        try:
                            self.log_callback("About to call cleanup_instances_callback", 'info')
                            cleanup_callback()
                            self.log_callback("Called cleanup_instances_callback successfully", 'info')
                        except Exception as e:
                            self.log_callback(f"Error in cleanup_instances_callback: {str(e)}", 'error')
                            import traceback
                            self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
                    
                    # Schedule the cleanup on the main thread's event loop
                    # We're in a background thread, so we can't create Qt objects with parents here
                    # Solution: Use QApplication.postEvent to post a custom event to the main thread
                    from PySide6.QtCore import QEvent, QTimer
                    from PySide6.QtWidgets import QApplication
                    
                    # Create a custom event class to carry the callback
                    class CleanupEvent(QEvent):
                        def __init__(self, callback):
                            # Use QEvent.Type.User as the base type for custom events
                            super().__init__(QEvent.Type.User)
                            self.callback = callback
                    
                    # Post the event to the root widget (main thread)
                    event = CleanupEvent(execute_cleanup)
                    QApplication.postEvent(self.root, event)
                    self.log_callback("Posted cleanup event to main thread", 'info')
                else:
                    self.log_callback("cleanup_instances_callback not available or None, skipping cleanup", 'warning')
                    
            except Exception as e:
                self.log_callback(f"Error stopping RuneLite instances: {str(e)}", 'error')
                import traceback
                self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
                # Use QTimer to show message box on main thread
                from PySide6.QtCore import QTimer
                def show_error():
                    QMessageBox.critical(self.root, "Stop Error", f"Failed to stop RuneLite instances: {str(e)}")
                QTimer.singleShot(0, show_error)
        
        # Start the stopping operation in a background thread
        stop_thread = threading.Thread(target=stop_in_thread, daemon=True)
        stop_thread.start()
    
    def setup_dependencies(self):
        """Run setup-dependencies.ps1 script."""
        # TODO: Implement
        pass
    
    def _resolve_and_log_toolchain_jdk(self, runelite_repo_path: Path) -> Dict[str, Any]:
        """
        Resolve and log the Java toolchain JDK that will be used.
        Forces toolchain resolution if needed (may trigger download).
        
        Returns:
            dict with keys:
                - success: bool - True if toolchain resolved successfully
                - jdk_path: str - Path to JDK that will be used
                - jdk_version: str - JDK version string
                - error: str - Error message if success is False
        """
        import subprocess
        import shutil
        
        result = {
            "success": False,
            "jdk_path": None,
            "jdk_version": None,
            "error": None
        }
        
        try:
            # Find gradlew
            if os.name == "nt":
                gradlew = runelite_repo_path / "gradlew.bat"
            else:
                gradlew = runelite_repo_path / "gradlew"
            if not gradlew.exists():
                gradlew = shutil.which("gradle")
            if not gradlew or (isinstance(gradlew, Path) and not gradlew.exists()):
                result["error"] = "Gradle wrapper not found"
                return result
            
            gradlew = str(gradlew) if isinstance(gradlew, Path) else gradlew
            
            self.log_callback("[Toolchain] Resolving Java toolchain JDK (this may download JDK 11 if not present)...", "info")
            
            # Run a dry-run of the run task to force toolchain resolution
            # This will download the JDK if needed and show us which one will be used
            resolve_process = subprocess.run(
                [gradlew, ":client:run", "--dry-run", "--info"],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,  # 5 minute timeout for download
            )
            
            resolve_output = resolve_process.stdout
            
            # Look for JDK path in output
            jdk_path = None
            jdk_version = None
            
            # Check for toolchain JDK paths (Gradle stores them in user home)
            gradle_jdks_path = Path.home() / ".gradle" / "jdks"
            if gradle_jdks_path.exists():
                # Find JDK 11 installations
                for jdk_dir in gradle_jdks_path.iterdir():
                    if jdk_dir.is_dir() and "11" in jdk_dir.name:
                        jdk_path = str(jdk_dir)
                        # Try to get version
                        java_exe = jdk_dir / "bin" / "java.exe" if os.name == "nt" else jdk_dir / "bin" / "java"
                        if java_exe.exists():
                            try:
                                version_process = subprocess.run(
                                    [str(java_exe), "-version"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,  # -version outputs to stderr, not stdout
                                    text=True,
                                    timeout=10
                                )
                                # Java -version outputs to stderr, not stdout
                                version_output = version_process.stderr or version_process.stdout
                                # Extract version from output like "openjdk version "11.0.29""
                                for line in version_output.split("\n"):
                                    if "version" in line.lower() and '"' in line:
                                        # Extract version string like "11.0.29"
                                        import re
                                        version_match = re.search(r'"(\d+\.\d+\.\d+[^"]*)"', line)
                                        if version_match:
                                            jdk_version = f"Java {version_match.group(1)}"
                                        else:
                                            jdk_version = line.strip()
                                        break
                            except Exception as e:
                                self.log_callback(f"[Toolchain] Could not get JDK version: {e}", "warning")
                        break
            
            # If no toolchain JDK found, check if system JDK is being used
            if not jdk_path:
                # Check for system JDK references in output
                if "C:\\Program Files\\Java" in resolve_output or "java.home" in resolve_output.lower():
                    self.log_callback("[Toolchain] ⚠️  WARNING: System JDK detected - toolchain may not be configured correctly", "warning")
                    # Try to find system JAVA_HOME
                    java_home = os.environ.get("JAVA_HOME")
                    if java_home:
                        jdk_path = java_home
                        try:
                            java_exe = Path(java_home) / "bin" / "java.exe" if os.name == "nt" else Path(java_home) / "bin" / "java"
                            if java_exe.exists():
                                version_process = subprocess.run(
                                    [str(java_exe), "-version"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,  # -version outputs to stderr
                                    text=True,
                                    timeout=10
                                )
                                # Java -version outputs to stderr, not stdout
                                version_output = version_process.stderr or version_process.stdout
                                # Extract version string
                                import re
                                for line in version_output.split("\n"):
                                    if "version" in line.lower() and '"' in line:
                                        version_match = re.search(r'"(\d+\.\d+\.\d+[^"]*)"', line)
                                        if version_match:
                                            jdk_version = f"Java {version_match.group(1)}"
                                        else:
                                            jdk_version = line.strip()
                                        break
                        except Exception:
                            pass
            
            if jdk_path:
                result["success"] = True
                result["jdk_path"] = jdk_path
                result["jdk_version"] = jdk_version or "Unknown version"
                self.log_callback(f"[Toolchain] ✅ Resolved JDK: {jdk_path}", "success")
                if jdk_version:
                    self.log_callback(f"[Toolchain] JDK Version: {jdk_version}", "info")
            else:
                result["error"] = "Could not determine JDK path from Gradle output"
                self.log_callback(f"[Toolchain] ⚠️  Could not determine JDK path, but continuing anyway...", "warning")
                result["success"] = True  # Continue anyway - Gradle will handle it
                result["jdk_path"] = "Will be resolved by Gradle"
                result["jdk_version"] = "Java 11 (toolchain)"
            
            return result
            
        except subprocess.TimeoutExpired:
            result["error"] = "Toolchain resolution timed out (JDK download may still be in progress)"
            self.log_callback(f"[Toolchain] ⚠️  Resolution timed out, but continuing (JDK download may complete during build)", "warning")
            result["success"] = True  # Continue anyway
            result["jdk_path"] = "Downloading..."
            result["jdk_version"] = "Java 11 (toolchain)"
            return result
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            self.log_callback(f"[Toolchain] ⚠️  Error resolving toolchain: {e}, but continuing anyway...", "warning")
            result["success"] = True  # Continue anyway
            result["jdk_path"] = "Unknown (will be resolved by Gradle)"
            result["jdk_version"] = "Java 11 (toolchain)"
            return result
    
    def _check_and_merge_latest_release(self, runelite_repo_path: Path) -> Dict[str, Any]:
        """
        Check if current RuneLite commit is at or ahead of latest release commit.
        If not, automatically merge the latest release commit.
        
        Returns:
            dict with keys:
                - success: bool - True if check/merge succeeded
                - merged: bool - True if a merge was performed
                - latest_release: str - Latest release tag found (e.g., "runelite-parent-1.12.16")
                - conflict: bool - True if merge conflicts occurred
                - error: str - Error message if success is False
        """
        import subprocess
        import re
        
        result = {
            "success": False,
            "merged": False,
            "latest_release": None,
            "conflict": False,
            "error": None
        }
        
        try:
            # Step 1: Fetch from remote to get latest tags and commits
            self.log_callback("[Release Check] Fetching from remote repository...", "info")
            fetch_process = subprocess.run(
                ["git", "fetch", "--tags", "--prune"],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60
            )
            
            if fetch_process.returncode != 0:
                result["error"] = f"Failed to fetch from remote: {fetch_process.stdout}"
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            self.log_callback("[Release Check] Fetch completed successfully", "info")
            
            # Step 2: Get current HEAD commit hash
            head_process = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            if head_process.returncode != 0:
                result["error"] = f"Failed to get current HEAD: {head_process.stderr}"
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            current_head = head_process.stdout.strip()
            self.log_callback(f"[Release Check] Current HEAD: {current_head[:8]}", "info")
            
            # Step 3: Find all release tags (runelite-parent-X.Y.Z)
            tags_process = subprocess.run(
                ["git", "tag", "--list", "runelite-parent-*"],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            if tags_process.returncode != 0:
                result["error"] = f"Failed to list release tags: {tags_process.stderr}"
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            release_tags = [tag.strip() for tag in tags_process.stdout.strip().split("\n") if tag.strip()]
            
            if not release_tags:
                result["error"] = "No release tags found (runelite-parent-*)"
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            # Step 4: Sort tags by version number to find latest
            def parse_version(tag: str) -> tuple:
                """Extract version tuple from tag like 'runelite-parent-1.12.16' -> (1, 12, 16)"""
                match = re.search(r'runelite-parent-(\d+)\.(\d+)\.(\d+)', tag)
                if match:
                    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                return (0, 0, 0)
            
            release_tags.sort(key=parse_version, reverse=True)
            latest_tag = release_tags[0]
            result["latest_release"] = latest_tag
            
            self.log_callback(f"[Release Check] Latest release tag: {latest_tag}", "info")
            
            # Step 5: Get the commit hash that the latest tag points to
            tag_commit_process = subprocess.run(
                ["git", "rev-parse", latest_tag],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            if tag_commit_process.returncode != 0:
                result["error"] = f"Failed to resolve tag {latest_tag}: {tag_commit_process.stderr}"
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            latest_release_commit = tag_commit_process.stdout.strip()
            self.log_callback(f"[Release Check] Latest release commit: {latest_release_commit[:8]}", "info")
            
            # Step 6: Check if current HEAD is at or ahead of the latest release commit
            merge_base_process = subprocess.run(
                ["git", "merge-base", "--is-ancestor", latest_release_commit, current_head],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            # merge-base --is-ancestor returns 0 if latest_release_commit is an ancestor of current_head
            # (meaning current HEAD includes the release)
            if merge_base_process.returncode == 0:
                self.log_callback(
                    f"[Release Check] Current HEAD is at or ahead of latest release ({latest_tag}). No merge needed.",
                    "success"
                )
                result["success"] = True
                return result
            
            # Step 7: Current HEAD doesn't include the latest release - need to merge
            self.log_callback(
                f"[Release Check] Current HEAD does not include latest release ({latest_tag}). Merging...",
                "warning"
            )
            
            # Step 7a: Clean up Java HotSpot error logs before checking for uncommitted changes
            # These logs are created when JVM crashes and can interfere with merge checks
            import glob
            error_log_pattern = str(runelite_repo_path / "**" / "hs_err_pid*.log")
            error_logs = glob.glob(error_log_pattern, recursive=True)
            if error_logs:
                self.log_callback(
                    f"[Release Check] Cleaning up {len(error_logs)} Java error log file(s) before merge check...",
                    "info"
                )
                for log_file in error_logs:
                    try:
                        Path(log_file).unlink()
                        self.log_callback(f"[Release Check] Removed: {log_file}", "info")
                    except Exception as e:
                        self.log_callback(
                            f"[Release Check] Warning: Could not remove {log_file}: {e}",
                            "warning"
                        )
            
            # Check if there are uncommitted changes (after cleaning up error logs)
            status_process = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            if status_process.returncode == 0 and status_process.stdout.strip():
                result["error"] = (
                    "Repository has uncommitted changes. Please commit or stash changes before merging releases.\n"
                    f"Uncommitted files:\n{status_process.stdout.strip()}"
                )
                self.log_callback(f"[Release Check] Error: {result['error']}", "error")
                return result
            
            # Step 8: Perform the merge
            merge_process = subprocess.run(
                ["git", "merge", "--no-edit", "--no-ff", latest_release_commit],
                cwd=str(runelite_repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120
            )
            
            merge_output = merge_process.stdout
            
            if merge_process.returncode != 0:
                # Check if it's a merge conflict
                if "CONFLICT" in merge_output or "conflict" in merge_output.lower():
                    result["conflict"] = True
                    result["error"] = f"Merge conflicts detected when merging {latest_tag}:\n{merge_output}"
                    self.log_callback(f"[Release Check] Merge conflict detected:\n{merge_output}", "error")
                    
                    # Try to abort the merge to leave repository in a clean state
                    abort_process = subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=str(runelite_repo_path),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=10
                    )
                    if abort_process.returncode == 0:
                        self.log_callback("[Release Check] Merge aborted. Repository left in clean state.", "info")
                    else:
                        self.log_callback(
                            f"[Release Check] Warning: Failed to abort merge: {abort_process.stderr}",
                            "warning"
                        )
                else:
                    result["error"] = f"Merge failed: {merge_output}"
                    self.log_callback(f"[Release Check] Merge failed:\n{merge_output}", "error")
                return result
            
            # Merge succeeded
            result["success"] = True
            result["merged"] = True
            self.log_callback(
                f"[Release Check] Successfully merged latest release ({latest_tag}) into repository.",
                "success"
            )
            self.log_callback(f"[Release Check] Merge output:\n{merge_output}", "info")
            
            return result
            
        except subprocess.TimeoutExpired:
            result["error"] = "Operation timed out (git command took too long)"
            self.log_callback(f"[Release Check] Error: {result['error']}", "error")
            return result
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            self.log_callback(f"[Release Check] Error: {result['error']}", "error")
            import traceback
            self.log_callback(f"[Release Check] Traceback:\n{traceback.format_exc()}", "error")
            return result
