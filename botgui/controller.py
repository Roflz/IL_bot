#!/usr/bin/env python3
"""Main controller for the Bot Controller GUI"""

import threading
import time
import logging
import json
import glob
import datetime
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import tkinter as tk
from tkinter import ttk
import queue
import numpy as np

# Import services
from .services.live_source import LiveSource
from .services.feature_pipeline import FeaturePipeline
from .services.predictor import PredictorService
from .services.mapping_service import MappingService
from .services.window_finder import WindowFinder
from .services.actions_service import ActionsService

# Import views
from .ui.views.live_features_view import LiveFeaturesView
from .ui.views.predictions_view import PredictionsView
from .ui.views.logs_view import LogsView
from .ui.views.live_view import LiveView

# Import utilities
from .util.queues import FeatureUpdateQueue, PredictionUpdateQueue, MessageDispatcher
from .state import UIState, RuntimeState

LOG = logging.getLogger(__name__)


class BotController:
    """Main controller that orchestrates all services and views"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bot Controller GUI")

        # Initialize state
        self.ui_state = UIState()
        self.runtime_state = RuntimeState()

        # Initialize services
        self._init_services()

        # Initialize views
        self.views = {}
        self._init_views()

        # Initialize queues and dispatcher
        self._init_queues()

        # Initialize worker threads
        self.workers = {}
        self._init_workers()

        # Initialize queues and threading control
        self.gs_queue = queue.Queue()
        # More headroom so the feature thread never blocks while UI catches up
        self.ui_queue = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        
        # Frame sequence tracking for heartbeat logging
        self._frame_seq = 0
        
        # Schema set flag
        self._schema_set = False
        
        # Window tracking for change detection
        self._last_window = None  # keep previous for changed_mask
        self._feature_schema_set = False

        # Start UI update loop
        self._start_ui_loop()

        LOG.info("Bot Controller initialized")

    def _init_services(self):
        """Initialize all services"""
        try:
            # Mapping service
            self.mapping_service = MappingService(self.ui_state.data_root)

            # Actions service
            self.actions_service = ActionsService(self)
            
            # Feature pipeline (needs actions service for synchronization)
            self.feature_pipeline = FeaturePipeline(self.ui_state.data_root, self.actions_service)

            # Predictor service
            self.predictor_service = PredictorService()

            # Window finder
            self.window_finder = WindowFinder()

            # Live source (use watchdog for instant file detection)
            # Use default gamestates directory during initialization, will be updated when session starts
            gamestates_dir = Path("data/gamestates")
            try:
                self.live_source = LiveSource(dir_path=gamestates_dir)
                LOG.info(f"Live source initialized for {gamestates_dir}")
            except RuntimeError as e:
                LOG.warning(f"Live source not available: {e}")
                LOG.info("GUI will run without live gamestate monitoring")
                self.live_source = None
            
            # Store gamestate files for later loading after views are bound
            self.initial_gamestate_files = []
            gamestates_dir = Path("data/gamestates")
            if gamestates_dir.exists():
                gamestate_files = list(gamestates_dir.glob("*.json"))
                # Sort by filename timestamp (newest first)
                gamestate_files.sort(key=lambda f: int(f.stem), reverse=True)
                self.initial_gamestate_files = gamestate_files[:10]  # Keep up to 10 files
                LOG.info(f"Found {len(self.initial_gamestate_files)} initial gamestate files to load")
            else:
                LOG.warning(f"Gamestates directory does not exist: {gamestates_dir}")

            LOG.info("All services initialized successfully")

        except Exception as e:
            LOG.exception("Failed to initialize services")
            raise

    def _init_views(self):
        """Initialize all views"""
        try:
            # Views are created by the main window, not here
            # Just initialize empty dictionary
            self.views = {}
            
            LOG.info("Views dictionary initialized")

        except Exception as e:
            LOG.exception("Failed to initialize views")
            raise

    def bind_views(self, live_view, logs_view, features_view, predictions_view):
        """Bind view references for direct access"""
        try:
            print("🔍 DEBUG [controller.py:145] bind_views called")
            print(f"🔍 DEBUG [controller.py:150] Binding views - live_view: {type(live_view)}, features_view: {type(features_view)}")
            
            self.live_view = live_view
            self.logs_view = logs_view
            self.features_view = features_view
            self.live_features_view = features_view  # Add reference for _pump_ui
            self.predictions_view = predictions_view
            
            print("🔍 DEBUG [controller.py:155] View references set successfully")
            
            # Also keep the views dictionary for backward compatibility
            self.views = {
                'live': live_view,
                'logs': logs_view,
                'live_features': features_view,
                'predictions': predictions_view
            }
            
            print("🔍 DEBUG [controller.py:165] Views dictionary created successfully")
            LOG.info("Views bound to controller")
            
            # Now load initial gamestates since views are available
            print("🔍 DEBUG [controller.py:170] About to load initial gamestates...")
            self._load_initial_gamestates()
            print("🔍 DEBUG [controller.py:175] Initial gamestates loaded successfully")
            
        except Exception as e:
            print(f"❌ ERROR [controller.py:180] in bind_views: {e}")
            import traceback
            traceback.print_exc()
            # CRASH IMMEDIATELY - view binding failures are critical
            raise RuntimeError(f"Failed to bind views: {e}") from e
    
    def _load_initial_gamestates(self):
        """Load initial gamestates from bot1 folder to populate the table"""
        if not hasattr(self, 'initial_gamestate_files') or not self.initial_gamestate_files:
            LOG.info("No initial gamestate files to load")
            return
        
        LOG.info("Loading initial gamestates from bot1 folder...")
        try:
            gamestates_loaded = 0
            
            # Load up to 10 gamestates
            for i, gamestate_file in enumerate(self.initial_gamestate_files):
                try:
                    LOG.info(f"Loading gamestate {i+1}/{len(self.initial_gamestate_files)} from {gamestate_file.name}")
                    with open(gamestate_file, 'r') as f:
                        gamestate = json.load(f)
                    
                    # Add metadata
                    gamestate['_source'] = 'file_polling'
                    gamestate['_file_timestamp'] = int(gamestate_file.stem)
                    gamestate['_file_path'] = str(gamestate_file)
                    
                    # Process the gamestate
                    window, changed_mask, feature_names, feature_groups = self.feature_pipeline.push(gamestate)
                    gamestates_loaded += 1
                    
                    LOG.info(f"Successfully loaded gamestate {i+1} with {len(gamestate)} keys")
                    
                except Exception as e:
                    LOG.warning(f"Failed to load gamestate {gamestate_file.name}: {e}")
                    continue
            
            LOG.info(f"Loaded {gamestates_loaded} initial gamestates")
            
            # Update the live features view with the loaded data
            if hasattr(self, 'features_view') and self.features_view:
                # UI must be on main thread
                # Non-blocking put (drop the stale one if the queue is full)
                try:
                    self.ui_queue.put_nowait(("table_update", (window, changed_mask, feature_names, feature_groups)))
                except queue.Full:
                    try:
                        _ = self.ui_queue.get_nowait()  # drop oldest
                    except queue.Empty:
                        pass
                    self.ui_queue.put_nowait(("table_update", (window, changed_mask, feature_names, feature_groups)))
                LOG.info("Updated live features view with initial data")
                # IMPORTANT: ensure the UI pump wakes up to process it
                try:
                    self._schedule_ui_pump()
                except Exception as e:
                    LOG.warning(f"Could not schedule UI pump: {e}")
                
        except Exception as e:
            LOG.exception("Failed to load initial gamestates")
            # Don't raise here - this is not critical for startup

    def _init_queues(self):
        """Initialize queues and message dispatcher"""
        try:
            # Create simple queues for now
            import queue
            self.feature_queue = queue.Queue(maxsize=100)
            self.prediction_queue = queue.Queue(maxsize=100)

            # Create dispatcher
            self.dispatcher = MessageDispatcher()
            self.dispatcher.register_queue("features", self.feature_queue)
            self.dispatcher.register_queue("predictions", self.prediction_queue)

            # Start dispatcher
            self.dispatcher.start()

            LOG.info("Queues and dispatcher initialized successfully")

        except Exception as e:
            LOG.exception("Failed to initialize queues")
            raise

    def _init_workers(self):
        """Initialize worker threads"""
        try:
            # Note: Live source worker removed - was dead code with 600ms throttling

            # Prediction worker
            self.workers['predictor'] = threading.Thread(
                target=self._predictor_worker,
                daemon=True,
                name="PredictorWorker"
            )

            # DON'T auto-start workers - they depend on live source being active

            LOG.info("Worker threads initialized successfully")

        except Exception as e:
            LOG.exception("Failed to initialize workers")
            raise

    def _predictor_worker(self):
        """Worker thread for model predictions"""
        LOG.info("Predictor worker started")
        
        while True:
            try:
                if not self.runtime_state.predictions_enabled:
                    time.sleep(1.0)
                    continue

                # Check if buffers are warm
                buffer_status = self.feature_pipeline.get_buffer_status()
                if not buffer_status.get('is_warm', False):
                    time.sleep(0.1)
                    continue

                # Get latest data
                if self.feature_pipeline.window is not None:
                    features = self.feature_pipeline.window
                    # Note: actions removed in simplified version
                    actions = []

                    # Run prediction
                    prediction = self.predictor_service.predict(features, actions)
                    if prediction is not None:
                        # Queue prediction update
                        self.prediction_queue.put({
                            'type': 'prediction',
                            'data': prediction,
                            'timestamp': time.time()
                        })

                time.sleep(0.6)  # 600ms interval

            except Exception as e:
                LOG.exception("Error in predictor worker")
                time.sleep(1.0)

    def _start_ui_loop(self):
        """Start the UI update loop"""
        def ui_tick():
            try:
                # Process message queues
                self.dispatcher.process_queues()

                # Update status displays
                self._update_status_displays()

                # Schedule next tick
                self.root.after(100, ui_tick)

            except Exception as e:
                LOG.exception("Error in UI tick")
                # Continue ticking even if there's an error
                self.root.after(100, ui_tick)

        # Start the first tick
        self.root.after(100, ui_tick)

    def _update_status_displays(self):
        """Update status displays in views"""
        try:
            # Update buffer status
            buffer_status = self.feature_pipeline.get_buffer_status()
            
            # Update live features summary
            if 'live_features' in self.views:
                self.views['live_features']._update_summary()

            # Update predictions status
            if 'predictions' in self.views:
                self.views['predictions']._update_status()

        except Exception as e:
            LOG.exception("Error updating status displays")

    # Public API methods

    def load_model(self, model_path: Path) -> bool:
        """Load a trained model"""
        try:
            success = self.predictor_service.load_model(model_path)
            if success:
                self.runtime_state.model_loaded = True
                self.runtime_state.model_path = model_path
                if 'logs' in self.views:
                    self.views['logs'].log_success(f"Model loaded: {model_path}")
                LOG.info(f"Model loaded successfully: {model_path}")
            else:
                if 'logs' in self.views:
                    self.views['logs'].log_error(f"Failed to load model: {model_path}")
                LOG.error(f"Failed to load model: {model_path}")
            
            return success

        except Exception as e:
            error_msg = f"Error loading model: {e}"
            if 'logs' in self.views:
                self.views['logs'].log_error(error_msg)
            LOG.exception("Error loading model")
            return False

    def enable_predictions(self, enabled: bool):
        """Enable or disable predictions"""
        try:
            self.predictor_service.enable_predictions(enabled)
            self.runtime_state.predictions_enabled = enabled
            
            status = "enabled" if enabled else "disabled"
            if 'logs' in self.views:
                self.views['logs'].log_info(f"Predictions {status}")
            LOG.info(f"Predictions {status}")

        except Exception as e:
            error_msg = f"Failed to toggle predictions: {e}"
            if 'logs' in self.views:
                self.views['logs'].log_error(error_msg)
            LOG.exception("Failed to toggle predictions")

    def clear_buffers(self):
        """Clear all feature and action buffers"""
        try:
            self.feature_pipeline.clear_buffers()
            if 'logs' in self.views:
                self.views['logs'].log_info("Buffers cleared")
            LOG.info("Buffers cleared")

        except Exception as e:
            error_msg = f"Failed to clear buffers: {e}"
            if 'logs' in self.views:
                self.views['logs'].log_error(error_msg)
            LOG.exception("Failed to clear buffers")

    def get_view(self, view_name: str):
        """Get a view by name"""
        return self.views.get(view_name)

    def start_live_mode(self):
        """Start live mode by starting watcher and feature threads, then UI pump"""
        try:
            print("🔍 DEBUG [controller.py:390] Controller start_live_mode called")
            LOG.info("Starting live mode...")
            
            # Check if live source is available
            if not self.live_source:
                LOG.error("Cannot start live mode: live source not available")
                raise RuntimeError("Live source not available. Please ensure gamestates directory exists.")
            
            # Ensure we're not already running
            if hasattr(self, '_watcher_thread') and self._watcher_thread.is_alive():
                LOG.warning("Watcher thread already running, stopping first...")
                self.stop_live_mode()
            
            if hasattr(self, '_feature_thread') and self._feature_thread.is_alive():
                LOG.warning("Feature thread already running, stopping first...")
                self.stop_live_mode()
            
            # Reset stop event
            self._stop.clear()
            
            # Reset feature pipeline to ensure clean state
            if hasattr(self, 'feature_pipeline'):
                LOG.debug("Resetting feature pipeline for clean start...")
                self.feature_pipeline.clear_buffers()
            
            # Reset schema flag
            self._schema_set = False
            self._feature_schema_set = False
            
            # Start actions recording
            self.actions_service.start_recording()
            
            # Start watcher thread
            self._watcher_thread = threading.Thread(
                target=self._watcher_worker,
                daemon=True,
                name="WatcherThread"
            )
            self._watcher_thread.start()
            LOG.debug("controller: watcher thread started")
            
            # Start feature thread
            self._feature_thread = threading.Thread(
                target=self._feature_worker,
                daemon=True,
                name="FeatureThread"
            )
            self._feature_thread.start()
            LOG.debug("controller: feature thread started")
            
            # Start UI pump scheduling
            print("🔍 DEBUG [controller.py:450] About to start UI pump scheduling...")
            try:
                self._schedule_ui_pump()
                print("🔍 DEBUG [controller.py:455] UI pump scheduling started")
            except Exception as pump_error:
                print(f"❌ ERROR [controller.py:458] Failed to start UI pump scheduling: {pump_error}")
                import traceback
                traceback.print_exc()
                # CRASH IMMEDIATELY - UI pump scheduling failures are critical
                raise RuntimeError(f"Failed to start UI pump scheduling: {pump_error}") from pump_error
            
            # Give threads a moment to initialize
            time.sleep(0.1)
            
            LOG.info("Live mode started successfully")
            print("🔍 DEBUG [controller.py:465] Live mode completed successfully")
            
        except Exception as e:
            LOG.exception("Failed to start live mode")
            raise

    def stop_live_mode(self):
        """Stop live mode by setting stop event and waiting for threads to finish"""
        try:
            self._stop.set()
            
            # Wait for threads to finish
            if hasattr(self, '_watcher_thread') and self._watcher_thread.is_alive():
                self._watcher_thread.join(timeout=2.0)
                if self._watcher_thread.is_alive():
                    LOG.warning("Watcher thread did not stop cleanly")
            
            if hasattr(self, '_feature_thread') and self._feature_thread.is_alive():
                self._feature_thread.join(timeout=2.0)
                if self._feature_thread.is_alive():
                    LOG.warning("Feature thread did not stop cleanly")
            
            # Clear queues to remove stale data
            while not self.gs_queue.empty():
                try:
                    self.gs_queue.get_nowait()
                except:
                    pass
            
            while not self.ui_queue.empty():
                try:
                    self.ui_queue.get_nowait()
                except:
                    pass
            
            # Stop actions recording
            self.actions_service.stop_recording()
            
            # Save final data for the last 10 timesteps
            try:
                self.feature_pipeline.save_final_data()
                LOG.info("Successfully saved final data for sample buttons")
            except Exception as e:
                LOG.error(f"Failed to save final data: {e}")
            
            # Reset stop event for next start
            self._stop.clear()
            
        except Exception as e:
            LOG.exception("Failed to stop live mode")
            raise

    def _pump_ui(self):
        """UI pump that processes UI queue messages"""
        try:
            print("🔍 DEBUG [controller.py:505] _pump_ui called")
            
            # Validate root window is available
            if not hasattr(self, 'root') or not self.root:
                error_msg = "Root window not available in _pump_ui"
                print(f"❌ ERROR [controller.py:510] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Check if we have a message
            try:
                message = self.ui_queue.get_nowait()
                print(f"🔍 DEBUG [controller.py:515] UI pump processing message: {type(message)}")
            except queue.Empty:
                # No message, don't schedule next pump
                print("🔍 DEBUG [controller.py:520] No message in queue, stopping pump")
                self._ui_pump_scheduled = False
                return
            else:
                try:
                    # Expect exactly ("table_update", (window, changed_mask, feature_names, feature_groups))
                    if not (isinstance(message, tuple) and len(message) == 2):
                        raise ValueError(f"UI message must be a 2-tuple; got {type(message).__name__} len={len(message) if isinstance(message, tuple) else 'n/a'}")

                    kind, payload = message  # <— no slicing
                    print(f"🔍 DEBUG [controller.py:530] Message kind: {kind}, payload type: {type(payload)}")

                    if kind != "table_update":
                        raise ValueError(f"Unknown UI message kind: {kind}")

                    if not (isinstance(payload, tuple) and len(payload) == 4):
                        raise ValueError("table_update payload must be a 4-tuple (window, changed_mask, feature_names, feature_groups)")

                    window, changed_mask, feature_names, feature_groups = payload
                    print(f"🔍 DEBUG [controller.py:540] Payload unpacked - window shape: {window.shape}, changed_mask shape: {changed_mask.shape}")
                    
                    # Validate shapes
                    if window.shape != (10, 128):
                        raise RuntimeError(f"Window must have shape (10,128), got {window.shape}")
                    if changed_mask.shape != (10, 128):
                        raise RuntimeError(f"Changed mask must have shape (10,128), got {changed_mask.shape}")
                    
                    # First update: set schema *before* any cell writes
                    if not self._feature_schema_set:
                        print("🔍 DEBUG [controller.py:550] Setting feature schema...")
                        if len(feature_names) != 128 or len(feature_groups) != 128:
                            raise ValueError(
                                f"schema size mismatch: names={len(feature_names)}, groups={len(feature_groups)}, expected 128"
                            )
                        self.live_features_view.set_schema(feature_names, feature_groups)
                        self._feature_schema_set = True
                        LOG.info("UI: schema set (128 names / 128 groups)")
                        print("🔍 DEBUG [controller.py:560] Feature schema set successfully")
                    
                    # Push the window into the view
                    print("🔍 DEBUG [controller.py:565] About to update view from window...")
                    
                    # Check if live_features_view is available
                    if not hasattr(self, 'live_features_view') or not self.live_features_view:
                        error_msg = "live_features_view not available for update"
                        print(f"❌ ERROR [controller.py:568] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    # Use the view's handler: it updates the table AND saves features.csv if recording.
                    self.live_features_view._handle_feature_update(
                        window, changed_mask, feature_names, feature_groups
                    )
                    print("🔍 DEBUG [controller.py:575] View update + save handled successfully")
                    
                    # Check if there are more messages before scheduling next pump
                    try:
                        # Peek at the queue to see if there are more messages
                        self.ui_queue.get_nowait()
                        # Put it back since we just peeked
                        self.ui_queue.put(("table_update", (window, changed_mask, feature_names, feature_groups)))
                        # There are more messages, schedule next pump
                        print("🔍 DEBUG [controller.py:580] More messages in queue, scheduling next UI pump in 30ms")
                        self.root.after(30, self._pump_ui)
                    except queue.Empty:
                        # No more messages, stop the pump
                        print("🔍 DEBUG [controller.py:585] No more messages, stopping UI pump")
                        self._ui_pump_scheduled = False
                    
                except Exception as e:
                    print(f"❌ ERROR [controller.py:590] in UI pump message processing: {e}")
                    import traceback
                    traceback.print_exc()
                    LOG.exception("UI apply failed")
                    raise
                finally:
                    # Reset scheduling flag when done processing
                    self._ui_pump_scheduled = False
        except Exception as e:
            print(f"❌ ERROR [controller.py:590] in UI pump main loop: {e}")
            import traceback
            traceback.print_exc()
            LOG.exception("UI pump failed")
            # CRASH IMMEDIATELY - UI pump failures are critical
            raise RuntimeError(f"UI pump failed: {e}") from e
    
    def _schedule_ui_pump(self):
        """Schedule the next UI pump if not already scheduled"""
        try:
            print("🔍 DEBUG [controller.py:600] _schedule_ui_pump called")
            
            if not hasattr(self, '_ui_pump_scheduled'):
                self._ui_pump_scheduled = False
            
            if not self._ui_pump_scheduled:
                print("🔍 DEBUG [controller.py:610] Scheduling UI pump...")
                self._ui_pump_scheduled = True
                self.root.after(1, self._pump_ui)
                print("🔍 DEBUG [controller.py:615] UI pump scheduled successfully")
            else:
                print("🔍 DEBUG [controller.py:620] UI pump already scheduled")
                
        except Exception as e:
            print(f"❌ ERROR [controller.py:625] in _schedule_ui_pump: {e}")
            import traceback
            traceback.print_exc()
            # CRASH IMMEDIATELY - UI pump scheduling failures are critical
            raise RuntimeError(f"Failed to schedule UI pump: {e}") from e
    
    def _watcher_worker(self):
        """Worker thread for watching gamestate files"""
        # Precondition checks
        if not hasattr(self, 'live_source') or not self.live_source:
            raise RuntimeError("live_source not initialized or not available")
        if not hasattr(self, 'gs_queue'):
            raise RuntimeError("gs_queue not initialized")
        
        last = None
        print("🔍 DEBUG: Watcher worker main loop starting...")
        while not self._stop.is_set():
            try:
                # Check if there are any existing files first
                if last is None:
                    print("🔍 DEBUG: First run - checking for existing files...")
                    # On first run, check if there are any existing files
                    existing_files = self.live_source.get_recent_gamestates(count=1)
                    if existing_files:
                        print(f"🔍 DEBUG: Found {len(existing_files)} existing files")
                        # Process existing files first
                        for gs in existing_files:
                            if gs.get("_source_path") != last:
                                print(f"🔍 DEBUG: Processing existing gamestate: {gs.get('_source_path')}")
                                self.gs_queue.put(gs)
                                last = gs.get("_source_path")
                                LOG.info(f"Processed existing gamestate: {last}")
                        # Continue to next iteration to process the file we just found
                        continue
                    else:
                        print("🔍 DEBUG: No existing files found, transitioning to normal watching mode...")
                        # No existing files, set last to a special value that will never match any real path
                        # This ensures we don't loop back to check existing files again
                        last = "NO_EXISTING_FILES_CHECKED"
                        print("🔍 DEBUG: Transitioned to normal watching mode - waiting for new files...")
                        # Small delay before starting normal watching
                        time.sleep(0.1)
                        continue
                
                frame_start = time.time()
                
                # Time detect→load
                detect_start = time.time()
                try:
                    # Only print waiting message once in a while to avoid spam
                    if not hasattr(self, '_last_waiting_message') or (time.time() - self._last_waiting_message) > 10.0:
                        print("🔍 DEBUG: Waiting for next gamestate...")
                        self._last_waiting_message = time.time()
                    
                    path = self.live_source.wait_for_next_gamestate(last, timeout_seconds=2.0)  # 2 second timeout
                    detect_time = (time.time() - detect_start) * 1000
                    print(f"🔍 DEBUG: New gamestate detected: {path}")
                    
                    # Time load→extract
                    load_start = time.time()
                    print("🔍 DEBUG: Loading gamestate JSON...")
                    gs = self.live_source.load_json(path)                  # may raise
                    load_time = (time.time() - load_start) * 1000
                    print(f"🔍 DEBUG: Gamestate loaded successfully, size: {len(str(gs)) if gs else 'unknown'}")
                except TimeoutError:
                    # No new files, continue waiting - only print timeout message occasionally
                    if not hasattr(self, '_last_timeout_message') or (time.time() - self._last_timeout_message) > 15.0:
                        print("🔍 DEBUG: Timeout waiting for new gamestate, continuing...")
                        self._last_timeout_message = time.time()
                    time.sleep(0.1)
                    continue
                
                # In watcher loop when a new file arrives, remember the source path
                try:
                    self._last_gs_path = gs.get("_source_path")
                    print(f"🔍 DEBUG: Set last gamestate path: {self._last_gs_path}")
                except Exception:
                    self._last_gs_path = None
                    print("🔍 DEBUG: Failed to set last gamestate path")
                
                print("🔍 DEBUG: Putting gamestate in queue...")
                self.gs_queue.put(gs)
                print("🔍 DEBUG: Gamestate queued successfully")
                
                # Store timing info for feature worker
                gs['_timing'] = {'detect': detect_time, 'load': load_time}
                print(f"🔍 DEBUG: Timing info stored - detect: {detect_time:.1f}ms, load: {load_time:.1f}ms")
                
                # *** CRUCIAL: remember what we just processed ***
                last = path
                print(f"🔍 DEBUG: Updated last processed path: {last}")
                
            except Exception as e:
                # CRITICAL ERROR - crash immediately, no fallback behavior
                print(f"❌ CRITICAL ERROR [controller.py:650] in watcher worker loop: {e}")
                import traceback
                traceback.print_exc()
                LOG.exception("Critical error in watcher worker loop: %s", e)
                
                # CRASH IMMEDIATELY - data processing errors must not be hidden
                # This ensures we see exactly what's wrong instead of misleading behavior
                raise RuntimeError(f"Data processing failed in watcher worker: {e}") from e

    def _feature_worker(self):
        """Worker thread for processing features"""
        # Precondition checks
        if not hasattr(self, 'gs_queue'):
            raise RuntimeError("gs_queue not initialized")
        if not hasattr(self, 'ui_queue'):
            raise RuntimeError("ui_queue not initialized")
        if not hasattr(self, 'feature_pipeline'):
            raise RuntimeError("feature_pipeline not initialized")
        
        try:
            LOG.debug("controller: feature thread started")
            
            while not self._stop.is_set():
                try:
                    print("🔍 DEBUG: Feature worker waiting for gamestate...")
                    gs = self.gs_queue.get()           # blocks
                    print(f"🔍 DEBUG: Got gamestate from queue, timestamp: {gs.get('timestamp', 'unknown')}")
                    
                    # Push into pipeline (extract + buffer + warm window)
                    print("🔍 DEBUG: About to push gamestate to feature pipeline...")
                    window, changed_mask, feature_names, feature_groups = self.feature_pipeline.push(gs)
                    print(f"🔍 DEBUG: Feature pipeline returned - window: {type(window)}, changed_mask: {type(changed_mask)}")
                    
                    # Check if window data is actually changing
                    if window is not None and window.shape[0] >= 2:
                        # Check if timestamps are different
                        if window[0, 127] == window[1, 127]:
                            LOG.warning("T0 and T1 have identical timestamps!")
                        if window[0, 127] == window[-1, 127]:
                            LOG.warning("T0 and T9 have identical timestamps!")
                    
                    # Also build synchronized action window for this gamestate timestamp
                    try:
                        if hasattr(self, 'actions_service') and self.actions_service:
                            ts = gs.get('timestamp')
                            if ts:
                                window_start = ts - 600
                                window_end = ts
                                actions = self.actions_service.get_actions_in_window(window_start, window_end)
                                if actions is not None:
                                    self.feature_pipeline.record_action_window_from_actions(actions)
                    except Exception:
                        pass
                    
                    # DEBUG: Log pipeline state after processing
                    if window is None:
                        LOG.error("feature worker: window is None after extract")
                        print("❌ ERROR: Window is None after feature extraction")
                    else:
                        print(f"🔍 DEBUG: Window shape: {window.shape}, changed_mask shape: {changed_mask.shape}")
                        # Use the pipeline's computed change mask
                        # Feature names / groups must come from pipeline (or static list if you prefer)
                        feature_names = self.feature_pipeline.feature_names  # len 128
                        feature_groups = self.feature_pipeline.feature_groups  # len 128
                        print(f"🔍 DEBUG: Feature names length: {len(feature_names)}, groups length: {len(feature_groups)}")

                        # POST the full 4-tuple payload the UI expects
                        print("🔍 DEBUG: About to put message in UI queue...")
                        # Non-blocking put (drop the stale one if the queue is full)
                        try:
                            self.ui_queue.put_nowait((
                                "table_update",
                                (window, changed_mask, feature_names, feature_groups)
                            ))
                        except queue.Full:
                            try:
                                _ = self.ui_queue.get_nowait()  # drop oldest
                            except queue.Empty:
                                pass
                            self.ui_queue.put_nowait((
                                "table_update",
                                (window, changed_mask, feature_names, feature_groups)
                            ))
                        print("🔍 DEBUG: Message put in UI queue successfully")
                        # IMPORTANT: ensure the UI pump wakes up to process it
                        try:
                            self._schedule_ui_pump()
                        except Exception as e:
                            LOG.warning(f"Could not schedule UI pump: {e}")
                        
                except Exception as e:
                    # CRITICAL ERROR - crash immediately, no fallback behavior
                    print(f"❌ CRITICAL ERROR [controller.py:750] in feature worker loop: {e}")
                    import traceback
                    traceback.print_exc()
                    LOG.exception("Critical error processing gamestate in feature worker: %s", e)
                    
                    # CRASH IMMEDIATELY - data processing errors must not be hidden
                    # This ensures we see exactly what's wrong instead of misleading behavior
                    raise RuntimeError(f"Data processing failed in feature worker: {e}") from e
                    
        except Exception as e:
            LOG.exception("Fatal error in feature worker")
            raise

    def shutdown(self):
        """Shutdown the controller and all services"""
        try:
            LOG.info("Shutting down Bot Controller")

            # Stop live mode
            if self.runtime_state.live_source_active:
                self.stop_live_mode()

            # Stop dispatcher
            if hasattr(self, 'dispatcher'):
                self.dispatcher.stop()

            # Clear buffers
            if hasattr(self, 'feature_pipeline'):
                self.feature_pipeline.clear_buffers()

            LOG.info("Bot Controller shutdown complete")

        except Exception as e:
            LOG.exception("Error during shutdown")
    
    def get_action_features(self) -> List[List[float]]:
        """Get last 10 action frames aligned to the gamestate timeline for display."""
        try:
            if hasattr(self, "feature_pipeline"):
                frames = self.feature_pipeline.get_last_action_windows(10)
                # Pad to 10 so the table always has T0..T9
                if len(frames) < 10:
                    frames = frames + ([[0.0]] * (10 - len(frames)))
                return frames
        except Exception:
            pass
        return [[0.0]] * 10
    
    def on_feature_data(self, feature_data):
        """Handle feature data updates for views that need them"""
        # This method can be overridden by views that need feature data
        pass
    
    def bind_live_features_view(self, view):
        """Bind a live features view to receive feature updates"""
        try:
            self.live_features_view = view
            LOG.info("Live features view bound to controller")
        except Exception as e:
            LOG.error(f"Failed to bind live features view: {e}")
    
    def start_live_mode_for_recorder(self):
        """Start live mode for recorder view"""
        try:
            self.start_live_mode()
        except Exception as e:
            LOG.error(f"Failed to start live mode for recorder: {e}")
            raise
    
    def stop_live_mode_for_recorder(self):
        """Stop live mode for recorder view"""
        try:
            self.stop_live_mode()
        except Exception as e:
            LOG.error(f"Failed to stop live mode for recorder: {e}")
            raise
    
    def update_gamestates_directory(self, session_timestamp: str):
        """Update the gamestates directory to point to a specific recording session"""
        try:
            print("🔍 DEBUG: update_gamestates_directory called")
            
            # Use pathlib for Windows-safe path construction
            new_gamestates_dir = Path("data") / "recording_sessions" / session_timestamp / "gamestates"
            new_gamestates_dir = new_gamestates_dir.resolve()  # Normalize and resolve any symlinks
            print(f"🔍 WATCHING GAMESTATES DIRECTORY: {new_gamestates_dir}")
            LOG.info(f"Updating gamestates directory to: {new_gamestates_dir}")
            
            # Update the initial gamestate files
            print("🔍 DEBUG: Checking for existing gamestate files...")
            if new_gamestates_dir.exists():
                gamestate_files = list(new_gamestates_dir.glob("*.json"))
                gamestate_files.sort(key=lambda f: int(f.stem), reverse=True)
                self.initial_gamestate_files = gamestate_files[:10]
                LOG.info(f"Updated initial gamestate files: {len(self.initial_gamestate_files)} files")
                print(f"🔍 DEBUG: Found {len(self.initial_gamestate_files)} initial gamestate files")
            else:
                LOG.warning(f"New gamestates directory does not exist: {new_gamestates_dir}")
                print("🔍 DEBUG: New gamestates directory does not exist yet")
            
            # CRITICAL: Update the live source to watch the new directory
            print("🔍 DEBUG: Checking live source availability...")
            if hasattr(self, 'live_source') and self.live_source:
                print("🔍 DEBUG: Live source available, updating directory...")
                try:
                    # Use the new update_directory method
                    print("🔍 DEBUG: Calling live_source.update_directory...")
                    self.live_source.update_directory(new_gamestates_dir)
                    LOG.info(f"Live source updated to watch: {new_gamestates_dir}")
                    print("🔍 DEBUG: Live source directory updated successfully")
                    
                    # If live mode is active, restart the watcher thread
                    print("🔍 DEBUG: Checking if watcher thread is active...")
                    if hasattr(self, '_watcher_thread') and self._watcher_thread.is_alive():
                        LOG.info("Restarting watcher thread for new directory...")
                        print("🔍 DEBUG: Restarting watcher thread...")
                        self._stop.set()
                        print("🔍 DEBUG: Stop event set, waiting for thread to finish...")
                        self._watcher_thread.join(timeout=2.0)
                        print("🔍 DEBUG: Watcher thread joined")
                        
                        # Start new watcher thread
                        print("🔍 DEBUG: Starting new watcher thread...")
                        self._stop.clear()
                        self._watcher_thread = threading.Thread(
                            target=self._watcher_worker,
                            daemon=True,
                            name="WatcherThread"
                        )
                        self._watcher_thread.start()
                        LOG.info("Watcher thread restarted for new directory")
                        print("🔍 DEBUG: New watcher thread started successfully")
                    else:
                        print("🔍 DEBUG: No active watcher thread to restart")
                        
                except Exception as e:
                    print(f"❌ ERROR updating live source directory: {e}")
                    import traceback
                    traceback.print_exc()
                    LOG.error(f"Failed to update live source directory: {e}")
                    raise
            else:
                LOG.warning("Live source not available, cannot update directory")
                print("🔍 DEBUG: Live source not available")
            
            # Optionally load a couple recent files to prime the table
            try:
                self._load_initial_gamestates()
                # _load_initial_gamestates() already enqueues a table_update and schedules the pump
            except Exception as e:
                LOG.warning(f"Could not load initial gamestates after directory update: {e}")
                
            print("🔍 DEBUG: update_gamestates_directory completed successfully")
                
        except Exception as e:
            print(f"❌ ERROR in update_gamestates_directory: {e}")
            import traceback
            traceback.print_exc()
            LOG.error(f"Failed to update gamestates directory: {e}")
            raise

    def get_watchdog_status(self) -> Dict[str, Any]:
        """Get current watchdog status and directory information"""
        try:
            if not hasattr(self, 'live_source') or not self.live_source:
                return {
                    'available': False,
                    'error': 'Live source not available'
                }
            
            # Get status from live source
            status = self.live_source.get_watchdog_status()
            status['available'] = True
            
            # Add controller-specific info
            if hasattr(self, '_watcher_thread'):
                status['watcher_thread_alive'] = self._watcher_thread.is_alive()
            else:
                status['watcher_thread_alive'] = False
            
            if hasattr(self, '_feature_thread'):
                status['feature_thread_alive'] = self._feature_thread.is_alive()
            else:
                status['feature_thread_alive'] = False
            
            status['live_mode_active'] = hasattr(self, '_watcher_thread') and self._watcher_thread.is_alive()
            
            return status
            
        except Exception as e:
            LOG.error(f"Error getting watchdog status: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def get_current_directory_display(self) -> str:
        """Get a formatted string showing current directory and watchdog status"""
        try:
            if not hasattr(self, 'live_source') or not self.live_source:
                return "🔴 Live source not available"
            
            return self.live_source.get_current_directory_info()
            
        except Exception as e:
            LOG.error(f"Error getting directory display: {e}")
            return f"🔴 Error: {e}"
