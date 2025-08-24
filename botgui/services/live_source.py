#!/usr/bin/env python3
"""Live data source for bot controller with watchdog-only file watching"""

import json
import time
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
import queue
import logging

LOG = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    LOG.error("watchdog not available - required for live source")


class FileWatcher(FileSystemEventHandler):
    """File system event handler for watchdog"""
    
    def __init__(self, queue: queue.Queue, watch_start_time: int):
        self.queue = queue
        self.watch_start_time = watch_start_time
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            try:
                # Verify the file actually exists and is accessible
                if not os.path.exists(event.src_path):
                    return
                
                # Check if this is a recent file (not an old event)
                filename = os.path.basename(event.src_path)
                try:
                    timestamp_str = filename.replace('.json', '')
                    file_timestamp = int(timestamp_str)
                    
                    # Filter out files that were created before we started watching
                    if file_timestamp < self.watch_start_time:
                        return
                    
                except (ValueError, TypeError):
                    # If we can't parse timestamp, allow it through
                    pass
                
                file_size = os.path.getsize(event.src_path)
                self.queue.put(('created', event.src_path, file_size))
            except OSError:
                # File might have been deleted already, continue
                pass


class LiveSource:
    """Live data source that provides real-time gamestate data through watchdog file watching"""
    
    def __init__(self, dir_path: Path):
        """
        Initialize live data source.
        
        Args:
            dir_path: Directory to watch for gamestate files
            
        Raises:
            RuntimeError: If watchdog is unavailable or cannot start
        """
        # Precondition checks
        if dir_path is None:
            raise ValueError("dir_path cannot be None")
        
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog is required but not available")
        
        self.dir_path = Path(dir_path)
        
        # Validate directory exists and is readable
        if not self.dir_path.exists():
            # Try to create the directory if it doesn't exist
            try:
                self.dir_path.mkdir(parents=True, exist_ok=True)
                LOG.info(f"Created gamestate directory: {self.dir_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to create gamestate directory {self.dir_path}: {e}")
        
        if not self.dir_path.is_dir():
            raise RuntimeError(f"gamestate dir missing: {self.dir_path}")
        if not os.access(self.dir_path, os.R_OK):
            raise RuntimeError(f"gamestate dir not readable: {self.dir_path}")
        
        # Log the directory we're watching
        LOG.info("live_source: watching dir: %s", os.path.abspath(self.dir_path))
        
        # Track when we started watching (to filter out old events)
        self._watch_start_time = int(time.time() * 1000)  # milliseconds
        LOG.info("live_source: Watch started at timestamp: %d", self._watch_start_time)
        
        # Setup file watching
        self.file_queue = queue.Queue()
        self.observer = None
        self.watching = False
        
        # Setup watchdog watcher
        self._setup_watcher()
        
        LOG.info("live_source: Watchdog watcher setup successful")
    
    def _setup_watcher(self):
        """Setup watchdog file system observer"""
        try:
            print("🔍 DEBUG: _setup_watcher called")
            if not WATCHDOG_AVAILABLE:
                print("❌ ERROR: watchdog not available")
                raise RuntimeError("watchdog not available")
            
            print("🔍 DEBUG: watchdog is available")
            
            # Clear any existing observer and its state
            if hasattr(self, 'observer') and self.observer:
                print("🔍 DEBUG: Stopping existing observer...")
                try:
                    self.observer.unschedule_all()
                    print("🔍 DEBUG: Observer unscheduled")
                    self.observer.stop()
                    print("🔍 DEBUG: Observer stop called")
                    
                    # SAFER: Use a shorter timeout and handle hanging joins
                    print("🔍 DEBUG: Waiting for observer to join (timeout 0.5s)...")
                    try:
                        self.observer.join(timeout=0.5)  # Reduced timeout to prevent hanging
                        print("🔍 DEBUG: Observer join completed")
                    except Exception as join_error:
                        print(f"🔍 DEBUG: Observer join failed: {join_error}")
                    
                    if self.observer.is_alive():
                        LOG.warning("Observer thread did not stop cleanly, forcing cleanup")
                        print("🔍 DEBUG: Observer still alive after join, forcing cleanup")
                        # Force cleanup without waiting
                        try:
                            self.observer.unschedule_all()
                        except:
                            pass
                except Exception as e:
                    LOG.warning("Failed to stop existing observer: %s", e)
                    print(f"🔍 DEBUG: Warning - failed to stop existing observer: {e}")
            else:
                print("🔍 DEBUG: No existing observer to stop")
            
            # Clear the file queue completely
            print("🔍 DEBUG: Clearing file queue...")
            while not self.file_queue.empty():
                try:
                    self.file_queue.get_nowait()
                except:
                    pass
            print("🔍 DEBUG: File queue cleared")
            
            # Create a completely new observer instance
            print("🔍 DEBUG: Creating new observer...")
            self.observer = Observer()
            print("🔍 DEBUG: Observer created")
            
            print("🔍 DEBUG: Creating event handler...")
            event_handler = FileWatcher(self.file_queue, self._watch_start_time)
            print("🔍 DEBUG: Event handler created")
            
            print(f"🔍 DEBUG: Scheduling observer for directory: {self.dir_path}")
            self.observer.schedule(event_handler, str(self.dir_path), recursive=False)
            print("🔍 DEBUG: Observer scheduled")
            
            print("🔍 DEBUG: Starting observer...")
            self.observer.start()
            print("🔍 DEBUG: Observer started")
            
            self.watching = True
            print("🔍 DEBUG: Watching flag set to True")
            
            LOG.info("Watchdog watcher completely reset and started fresh")
            print("🔍 DEBUG: _setup_watcher completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in _setup_watcher: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _is_file_stable(self, file_path: str, min_delay_ms: int = 50) -> bool:
        """
        Check if file size is stable by reading size multiple times with delay.
        
        Args:
            file_path: Path to the file to check
            min_delay_ms: Minimum delay between reads in milliseconds
            
        Returns:
            True if file size is stable for 2 consecutive checks, False otherwise
        """
        try:
            # First read
            size1 = os.path.getsize(file_path)
            
            # Wait for specified delay
            time.sleep(min_delay_ms / 1000.0)
            
            # Second read
            size2 = os.path.getsize(file_path)
            
            # Wait a bit more for third read to ensure stability
            time.sleep(min_delay_ms / 1000.0)
            
            # Third read
            size3 = os.path.getsize(file_path)
            
            # File is stable if size doesn't change for 2 consecutive checks
            return size1 == size2 and size2 == size3
        except OSError as e:
            LOG.debug("File stability check failed for %s: %s", file_path, e)
            return False
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """
        Check if a file should be ignored (hidden, temp, or zero size).
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file should be ignored, False otherwise
        """
        try:
            # Check for hidden/temp files
            filename = os.path.basename(file_path)
            if filename.startswith('.') or filename.startswith('~') or filename.endswith('.tmp'):
                return True
            
            # Check for zero size files
            if os.path.getsize(file_path) == 0:
                return True
                
            return False
        except OSError:
            # If we can't check the file, ignore it
            return True
    
    def _scan_latest(self) -> Optional[Dict[str, Any]]:
        """
        Scan directory for the latest file by numeric filename.
        
        Returns:
            Dictionary with latest file info or None if no files found
        """
        try:
            # Scan directory for all JSON files
            json_files = list(self.dir_path.glob("*.json"))
            if not json_files:
                return None
            
            # Enumerate files with metadata for latest detection
            file_candidates = []
            for file_path in json_files:
                try:
                    # Skip files that should be ignored
                    if self._should_ignore_file(str(file_path)):
                        continue
                    
                    # Get filename stem and check if numeric
                    stem = file_path.stem
                    try:
                        name_numeric = int(stem)
                        is_numeric = True
                    except ValueError:
                        name_numeric = None
                        is_numeric = False
                    
                    # Get modification time
                    mtime = file_path.stat().st_mtime
                    
                    file_candidates.append({
                        'path': str(file_path),
                        'name_numeric': name_numeric,
                        'mtime': mtime,
                        'is_numeric': is_numeric
                    })
                except OSError as e:
                    LOG.debug("File metadata check failed for %s: %s", file_path, e)
                    continue
            
            if not file_candidates:
                return None
            
            # Choose latest by name_numeric (if numeric), with mtime as tie-breaker
            numeric_candidates = [f for f in file_candidates if f['is_numeric']]
            if numeric_candidates:
                # Sort by name_numeric (descending), then by mtime (descending) as tie-breaker
                numeric_candidates.sort(key=lambda f: (f['name_numeric'], f['mtime']), reverse=True)
                latest = numeric_candidates[0]
            else:
                # Fallback to mtime sorting if no numeric filenames
                file_candidates.sort(key=lambda f: f['mtime'], reverse=True)
                latest = file_candidates[0]
            
            return latest
                
        except Exception as e:
            LOG.exception("Error scanning for latest file")
            return None
    
    def _next_file_blocking(self, last_path: Optional[str] = None, timeout_seconds: float = 1.0) -> str:
        """
        Get the next gamestate file from the queue, blocking until one appears.
        
        Args:
            last_path: Path of last seen file (to avoid re-processing)
            timeout_seconds: Maximum time to wait for a file (default 1 second)
            
        Returns:
            Path to the new file
            
        Raises:
            TimeoutError: If no file appears within timeout
        """
        start_time = time.time()
        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timeout waiting for gamestate file after {timeout_seconds} seconds")
                
            try:
                event_type, file_path, file_size = self.file_queue.get(timeout=0.5)  # Shorter timeout for more responsive checking
                
                if event_type == 'created' and file_path != last_path:
                    # Additional safety check: verify this is a recent file by checking timestamp
                    try:
                        filename = os.path.basename(file_path)
                        timestamp_str = filename.replace('.json', '')
                        file_timestamp = int(timestamp_str)
                        
                        # Filter out files that were created before we started watching
                        if file_timestamp < self._watch_start_time:
                            LOG.debug("Ignoring old file event: %s (created=%d, watch_start=%d)", 
                                    filename, file_timestamp, self._watch_start_time)
                            continue
                            
                    except (ValueError, TypeError):
                        # If we can't parse timestamp, allow it through
                        pass
                    
                    # CRITICAL: Check if file actually exists before trying to access it
                    if not os.path.exists(file_path):
                        LOG.debug("Ignoring event for non-existent file: %s", file_path)
                        continue
                    
                    # Check if file should be ignored (hidden, temp, zero size)
                    if self._should_ignore_file(file_path):
                        LOG.debug("Ignoring file that should be ignored: %s", file_path)
                        continue
                    
                    # Check if file is stable (debounce partial writes)
                    if not self._is_file_stable(file_path):
                        # File not stable, try one more time after a short delay
                        time.sleep(0.1)
                        if not self._is_file_stable(file_path):
                            LOG.debug("File still not stable after retry: %s", file_path)
                            continue
                    
                    return file_path
                            
            except queue.Empty:
                # Timeout occurred - this is normal, not an error
                # Just continue the loop to wait for the next file
                continue
            except Exception as e:
                # Don't hide other exceptions - let them propagate so we can see what's wrong
                LOG.error("Exception in watcher loop: %s", e, exc_info=True)
                raise
    
    def wait_for_next_gamestate(self, last_seen: Optional[str] = None, timeout_seconds: float = 1.0) -> str:
        """
        Wait for the next gamestate file to appear and become stable.
        
        Args:
            last_seen: Path of last seen file (to avoid re-processing)
            timeout_seconds: Maximum time to wait for a new file (default 1 second)
            
        Returns:
            Path to the new stable gamestate file
            
        Note:
            Skips stale files and continues watching (non-fatal)
        """
        try:
            # Only print initial messages once per session to avoid spam
            if not hasattr(self, '_initial_messages_printed'):
                print(f"🔍 DEBUG [live_source.py:400] wait_for_next_gamestate called - last_seen: {last_seen}, timeout: {timeout_seconds}s")
                
                # Validate directory exists
                if not self.dir_path.exists():
                    error_msg = f"gamestate dir missing: {self.dir_path}"
                    print(f"❌ ERROR [live_source.py:405] {error_msg}")
                    raise RuntimeError(error_msg)
                
                # Validate watchdog is watching
                if not self.watching:
                    error_msg = "watchdog not watching"
                    print(f"❌ ERROR [live_source.py:410] {error_msg}")
                    raise RuntimeError(error_msg)
                
                print("🔍 DEBUG [live_source.py:415] Directory and watchdog validation passed")
                self._initial_messages_printed = True
            else:
                # Still validate but don't print messages
                if not self.dir_path.exists():
                    raise RuntimeError(f"gamestate dir missing: {self.dir_path}")
                if not self.watching:
                    raise RuntimeError("watchdog not watching")
            
        except Exception as e:
            print(f"❌ ERROR [live_source.py:420] in wait_for_next_gamestate validation: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        start_time = time.time()
        loop_start_time = time.time()
        last_summary_time = time.time()
        loop_count = 0
        
        # Only print main loop message once per session to avoid spam
        if not hasattr(self, '_main_loop_message_printed'):
            print(f"🔍 DEBUG [live_source.py:425] Starting main loop at {start_time}")
            self._main_loop_message_printed = True
        
        while True:
            try:
                loop_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                loop_elapsed = current_time - loop_start_time
                
                # Check timeout
                if elapsed > timeout_seconds:
                    timeout_msg = f"Timeout waiting for gamestate file after {timeout_seconds} seconds"
                    print(f"🔍 DEBUG [live_source.py:435] {timeout_msg}")
                    raise TimeoutError(timeout_msg)
                
                # Summary update every 15 seconds instead of every iteration
                if current_time - last_summary_time >= 15.0:
                    print(f"🔍 DEBUG [live_source.py:440] Loop status - iterations: {loop_count}, elapsed: {elapsed:.1f}s, timeout: {timeout_seconds}s")
                    last_summary_time = current_time
                
                # Get next file
                file_path = self._next_file_blocking(last_seen)
                
                # Only log file details on first iteration or when something changes
                if loop_count == 1 or (hasattr(self, '_last_logged_file') and self._last_logged_file != file_path):
                    print(f"🔍 DEBUG [live_source.py:450] Got file path: {file_path}")
                    self._last_logged_file = file_path
                
                # Check if this file is the latest
                latest = self._scan_latest()
                if latest is None:
                    if loop_count == 1:
                        print("🔍 DEBUG [live_source.py:455] No files found, continuing...")
                    # No files found, continue
                    last_seen = file_path
                    continue
                
                # Only log latest file info on first iteration or when it changes
                if loop_count == 1 or (hasattr(self, '_last_logged_latest') and self._last_logged_latest != latest):
                    print(f"🔍 DEBUG [live_source.py:460] Latest file info: {latest}")
                    self._last_logged_latest = latest
                
                # Parse candidate filename
                try:
                    candidate_stem = Path(file_path).stem
                    candidate_numeric = int(candidate_stem)
                    if loop_count == 1:
                        print(f"🔍 DEBUG [live_source.py:470] Candidate filename: {candidate_stem} -> numeric: {candidate_numeric}")
                except ValueError as ve:
                    print(f"🔍 DEBUG [live_source.py:475] Non-numeric filename '{candidate_stem}', allowing through")
                    # Non-numeric filename, allow it through
                    return file_path
                
                # Check if candidate is the latest
                if latest['is_numeric'] is not None:
                    latest_numeric = latest['name_numeric']
                    
                    if candidate_numeric < latest_numeric:
                        if loop_count == 1:
                            print(f"🔍 DEBUG [live_source.py:485] Candidate {candidate_numeric} is stale, skipping...")
                        last_seen = file_path
                        continue  # DO NOT RAISE - skip stale and continue
                    else:
                        print(f"🔍 DEBUG [live_source.py:490] Candidate {candidate_numeric} is current, returning")
                        return file_path
                else:
                    print("🔍 DEBUG [live_source.py:495] Latest file is non-numeric, allowing candidate through")
                    # Latest file is non-numeric, allow candidate through
                    return file_path
                    
            except TimeoutError:
                # Timeout is normal when waiting for files - just re-raise it
                raise
            except Exception as e:
                print(f"❌ CRITICAL ERROR [live_source.py:500] in wait_for_next_gamestate main loop: {e}")
                import traceback
                traceback.print_exc()
                
                # CRASH IMMEDIATELY - no fallback behavior for data processing errors
                # This ensures we don't hide critical failures with misleading behavior
                raise RuntimeError(f"Data processing failed in wait_for_next_gamestate: {e}") from e
    
    def load_json(self, path: str) -> Dict[str, Any]:
        """
        Load and parse JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Parsed JSON data as dictionary with source metadata attached
            
        Raises:
            ValueError: If JSON is invalid
            RuntimeError: If file cannot be read
        """
        try:
            print(f"🔍 DEBUG [live_source.py:510] load_json called for path: {path}")
            
            # Precondition checks
            if not path:
                error_msg = "path cannot be empty"
                print(f"❌ ERROR [live_source.py:515] {error_msg}")
                raise ValueError(error_msg)
                
            if not os.path.exists(path):
                error_msg = f"file does not exist: {path}"
                print(f"❌ ERROR [live_source.py:520] {error_msg}")
                raise RuntimeError(error_msg)
                
            if not os.access(path, os.R_OK):
                error_msg = f"file not readable: {path}"
                print(f"❌ ERROR [live_source.py:525] {error_msg}")
                raise RuntimeError(error_msg)
            
            print("🔍 DEBUG [live_source.py:530] File validation passed")
            
            # Check file size
            file_size = os.path.getsize(path)
            print(f"🔍 DEBUG [live_source.py:535] File size: {file_size} bytes")
            
            if file_size == 0:
                error_msg = f"file is empty: {path}"
                print(f"❌ ERROR [live_source.py:540] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Load JSON with timeout protection
            print("🔍 DEBUG [live_source.py:545] About to open and parse JSON file...")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"🔍 DEBUG [live_source.py:550] JSON parsed successfully - keys: {list(data.keys()) if data else 'None'}")
            
            try:
                st = os.stat(path)
                mtime = st.st_mtime
            except Exception:
                mtime = None
                
            # Attach the path and file times so downstream can log them
            try:
                data["_source_path"] = str(path)
                data["_source_mtime"] = mtime
                # If filename is like 1755684135702.json, capture the numeric stem
                try:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    data["_source_name_numeric"] = int(stem)
                except Exception:
                    data["_source_name_numeric"] = None
            except Exception:
                LOG.exception("live_source: failed to attach source metadata")
                raise
            
            return data
            
        except json.JSONDecodeError as e:
            LOG.exception("invalid JSON: %s", path)
            raise ValueError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            LOG.exception("Failed to load JSON from %s", path)
            raise RuntimeError(f"Cannot read file {path}: {e}")
    
    def get_recent_gamestates(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent gamestates.
        
        Args:
            count: Number of recent gamestates to return
            
        Returns:
            List of recent gamestate dictionaries, most recent first
        """
        try:
            # Get all JSON files in the directory
            json_files = list(self.dir_path.glob("*.json"))
            if not json_files:
                return []
            
            # Filter out ignored files
            valid_files = [f for f in json_files if not self._should_ignore_file(str(f))]
            if not valid_files:
                return []
            
            # Sort by filename (timestamp) in descending order
            valid_files.sort(key=lambda x: x.name, reverse=True)
            
            # Load the most recent files
            recent_gamestates = []
            for file_path in valid_files[:count]:
                try:
                    gamestate = self.load_json(str(file_path))
                    if gamestate:
                        recent_gamestates.append(gamestate)
                except Exception as e:
                    LOG.debug(f"Failed to load gamestate from {file_path}: {e}")
                    continue
            
            return recent_gamestates
            
        except Exception as e:
            LOG.error(f"Error getting recent gamestates: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the live source and cleanup resources"""
        if hasattr(self, 'observer') and self.observer and self.watching:
            try:
                self.observer.unschedule_all()
                self.observer.stop()
                self.observer.join(timeout=1.0)
                if self.observer.is_alive():
                    LOG.warning("Observer thread did not stop cleanly during shutdown")
            except Exception as e:
                LOG.warning("Error during observer shutdown: %s", e)
            finally:
                self.observer = None
                self.watching = False
                
        # Clear the queue completely
        if hasattr(self, 'file_queue'):
            while not self.file_queue.empty():
                try:
                    self.file_queue.get_nowait()
                except:
                    pass
        
        LOG.info("Live source watcher completely stopped and cleaned up")
    
    def update_directory(self, new_dir_path: Path):
        """Update the directory being watched"""
        try:
            print(f"🔍 DEBUG: LiveSource update_directory called: {self.dir_path} -> {new_dir_path}")
            LOG.info(f"Updating live source directory from {self.dir_path} to {new_dir_path}")
            
            # Stop current watcher
            print("🔍 DEBUG: Stopping current watcher...")
            self.shutdown()
            print("🔍 DEBUG: Current watcher stopped")
            
            # Update directory path
            self.dir_path = Path(new_dir_path)
            print(f"🔍 DEBUG: Directory path updated to: {self.dir_path}")
            
            # Validate and create directory if needed
            print("🔍 DEBUG: Validating new directory...")
            if not self.dir_path.exists():
                try:
                    print("🔍 DEBUG: Creating new directory...")
                    self.dir_path.mkdir(parents=True, exist_ok=True)
                    LOG.info(f"Created new gamestate directory: {self.dir_path}")
                    print("🔍 DEBUG: New directory created successfully")
                except Exception as e:
                    print(f"❌ ERROR creating directory: {e}")
                    raise RuntimeError(f"Failed to create new gamestate directory {self.dir_path}: {e}")
            
            if not self.dir_path.is_dir():
                print("❌ ERROR: New path is not a directory")
                raise RuntimeError(f"New gamestate dir is not a directory: {self.dir_path}")
            if not os.access(self.dir_path, os.R_OK):
                print("❌ ERROR: New directory not readable")
                raise RuntimeError(f"New gamestate dir not readable: {self.dir_path}")
            
            print("🔍 DEBUG: Directory validation passed")
            
            # Reset watch start time
            self._watch_start_time = int(time.time() * 1000)
            print(f"🔍 DEBUG: Watch start time reset to: {self._watch_start_time}")
            
            # Setup new watcher
            print("🔍 DEBUG: Setting up new watcher...")
            self._setup_watcher()
            print("🔍 DEBUG: New watcher setup completed")
            
            LOG.info(f"Live source successfully updated to watch: {self.dir_path}")
            print("🔍 DEBUG: LiveSource update_directory completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in LiveSource update_directory: {e}")
            import traceback
            traceback.print_exc()
            LOG.error(f"Failed to update live source directory: {e}")
            raise
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.shutdown()
        except Exception as e:
            LOG.debug("Error during cleanup: %s", e)
            # Ignore errors during cleanup

    def get_watchdog_status(self) -> Dict[str, Any]:
        """
        Get current watchdog status information.
        
        Returns:
            Dictionary with watchdog status details
        """
        try:
            status = {
                'watching': self.watching,
                'directory': str(self.dir_path) if self.dir_path else None,
                'directory_absolute': str(self.dir_path.resolve()) if self.dir_path and self.dir_path.exists() else None,
                'watchdog_available': WATCHDOG_AVAILABLE,
                'observer_active': False,
                'observer_thread_alive': False,
                'files_in_directory': 0,
                'last_watch_start': self._watch_start_time
            }
            
            # Check observer status
            if hasattr(self, 'observer') and self.observer:
                status['observer_active'] = True
                if hasattr(self.observer, 'is_alive'):
                    status['observer_thread_alive'] = self.observer.is_alive()
            
            # Count files in directory
            if self.dir_path and self.dir_path.exists():
                try:
                    json_files = list(self.dir_path.glob("*.json"))
                    # Filter out ignored files
                    valid_files = [f for f in json_files if not self._should_ignore_file(str(f))]
                    status['files_in_directory'] = len(valid_files)
                except Exception:
                    status['files_in_directory'] = 0
            
            return status
            
        except Exception as e:
            LOG.error(f"Error getting watchdog status: {e}")
            return {
                'watching': False,
                'directory': None,
                'directory_absolute': None,
                'watchdog_available': WATCHDOG_AVAILABLE,
                'observer_active': False,
                'observer_thread_alive': False,
                'files_in_directory': 0,
                'last_watch_start': None,
                'error': str(e)
            }
    
    def get_current_directory_info(self) -> str:
        """
        Get a human-readable string describing the current directory being watched.
        
        Returns:
            Formatted string with directory information
        """
        try:
            if not self.dir_path:
                return "No directory set"
            
            if not self.dir_path.exists():
                return f"Directory does not exist: {self.dir_path}"
            
            # Get relative path from current working directory
            try:
                cwd = Path.cwd()
                relative_path = self.dir_path.relative_to(cwd)
                display_path = str(relative_path)
            except ValueError:
                # If not relative to cwd, show absolute path
                display_path = str(self.dir_path.resolve())
            
            # Count files
            json_files = list(self.dir_path.glob("*.json"))
            valid_files = [f for f in json_files if not self._should_ignore_file(str(f))]
            
            status = "🟢 Active" if self.watching else "🔴 Inactive"
            
            return f"📁 {display_path}\n{status} | 📄 {len(valid_files)} files"
            
        except Exception as e:
            LOG.error(f"Error getting directory info: {e}")
            return f"Error: {e}"
