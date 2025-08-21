#!/usr/bin/env python3
"""Live data source for bot controller with watchdog-only file watching"""

import json
import time
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
from queue import Queue
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
    
    def __init__(self, queue: Queue, watch_start_time: int):
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
            raise RuntimeError(f"gamestate dir missing: {self.dir_path}")
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
        self.file_queue = Queue()
        self.observer = None
        self.watching = False
        
        # Setup watchdog watcher
        self._setup_watcher()
        
        LOG.info("live_source: Watchdog watcher setup successful")
    
    def _setup_watcher(self):
        """Setup watchdog file system observer"""
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog not available")
        
        # Clear any existing observer and its state
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.unschedule_all()
                self.observer.stop()
                self.observer.join(timeout=1.0)
                if self.observer.is_alive():
                    LOG.warning("Observer thread did not stop cleanly, forcing cleanup")
            except Exception as e:
                LOG.warning("Failed to stop existing observer: %s", e)
        
        # Clear the file queue completely
        while not self.file_queue.empty():
            try:
                self.file_queue.get_nowait()
            except:
                pass
        
        # Create a fresh queue to ensure no old events persist
        self.file_queue = Queue()
        
        # Create a completely new observer instance
        self.observer = Observer()
        event_handler = FileWatcher(self.file_queue, self._watch_start_time)
        self.observer.schedule(event_handler, str(self.dir_path), recursive=False)
        self.observer.start()
        self.watching = True
        
        LOG.info("Watchdog watcher completely reset and started fresh")
    
    def _is_file_stable(self, file_path: str, min_delay_ms: int = 20) -> bool:
        """
        Check if file size is stable by reading size twice with delay.
        
        Args:
            file_path: Path to the file to check
            min_delay_ms: Minimum delay between reads in milliseconds
            
        Returns:
            True if file size is stable, False otherwise
        """
        try:
            # First read
            size1 = os.path.getsize(file_path)
            
            # Wait for specified delay
            time.sleep(min_delay_ms / 1000.0)
            
            # Second read
            size2 = os.path.getsize(file_path)
            
            return size1 == size2
        except OSError as e:
            LOG.debug("File stability check failed for %s: %s", file_path, e)
            return False
    
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
    
    def _next_file_blocking(self, last_path: Optional[str]) -> str:
        """
        Wait for the next file event and return the file path.
        
        Args:
            last_path: Path of last seen file (to avoid re-processing)
            
        Returns:
            Path to the new file
        """
        while True:
            try:
                event_type, file_path, file_size = self.file_queue.get(timeout=1.0)
                
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
                    
                    # Check if file is stable
                    if not self._is_file_stable(file_path):
                        # File not stable, try one more time after a short delay
                        time.sleep(0.1)
                        if not self._is_file_stable(file_path):
                            continue
                    
                    return file_path
                            
            except Exception as e:
                # Don't hide exceptions - let them propagate so we can see what's wrong
                LOG.error("Exception in watcher loop: %s", e, exc_info=True)
                raise
    
    def wait_for_next_gamestate(self, last_seen: Optional[str] = None) -> str:
        """
        Wait for the next gamestate file to appear and become stable.
        
        Args:
            last_seen: Path of last seen file (to avoid re-processing)
            
        Returns:
            Path to the new stable gamestate file
            
        Note:
            Skips stale files and continues watching (non-fatal)
        """
        if not self.dir_path.exists():
            raise RuntimeError(f"gamestate dir missing: {self.dir_path}")
        
        if not self.watching:
            raise RuntimeError("watchdog not watching")
        
        while True:
            # Get next file
            file_path = self._next_file_blocking(last_seen)
            
            # Check if this file is the latest
            latest = self._scan_latest()
            if latest is None:
                # No files found, continue
                last_seen = file_path
                continue
            
            # Parse candidate filename
            try:
                candidate_stem = Path(file_path).stem
                candidate_numeric = int(candidate_stem)
            except ValueError:
                # Non-numeric filename, allow it through
                return file_path
            
            # Check if candidate is the latest
            if latest['is_numeric'] is not None:
                if candidate_numeric < latest['name_numeric']:
                    last_seen = file_path
                    continue  # DO NOT RAISE - skip stale and continue
                else:
                    return file_path
            else:
                # Latest file is non-numeric, allow candidate through
                return file_path
    
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
        # Precondition checks
        if not path:
            raise ValueError("path cannot be empty")
        if not os.path.exists(path):
            raise RuntimeError(f"file does not exist: {path}")
        if not os.access(path, os.R_OK):
            raise RuntimeError(f"file not readable: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
            
            # Sort by filename (timestamp) in descending order
            json_files.sort(key=lambda x: x.name, reverse=True)
            
            # Load the most recent files
            recent_gamestates = []
            for file_path in json_files[:count]:
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
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.shutdown()
        except Exception as e:
            LOG.debug("Error during cleanup: %s", e)
            # Ignore errors during cleanup
