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
from queue import Queue, Empty, Full
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
        self.gs_queue = Queue()
        self.ui_queue = Queue(maxsize=1)
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

            # Feature pipeline
            self.feature_pipeline = FeaturePipeline(self.ui_state.data_root)

            # Predictor service
            self.predictor_service = PredictorService()

            # Window finder
            self.window_finder = WindowFinder()

            # Actions service
            self.actions_service = ActionsService(self)

            # Live source (use watchdog for instant file detection)
            gamestates_dir = Path(f"data/{self.ui_state.bot_mode}/gamestates")
            try:
                self.live_source = LiveSource(dir_path=gamestates_dir)
                LOG.info(f"Live source initialized for {gamestates_dir}")
            except RuntimeError as e:
                LOG.warning(f"Live source not available: {e}")
                LOG.info("GUI will run without live gamestate monitoring")
                self.live_source = None
            
            # Store gamestate files for later loading after views are bound
            self.initial_gamestate_files = []
            gamestates_dir = Path(f"data/{self.ui_state.bot_mode}/gamestates")
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
        self.live_view = live_view
        self.logs_view = logs_view
        self.features_view = features_view
        self.live_features_view = features_view  # Add reference for _pump_ui
        self.predictions_view = predictions_view
        
        # Also keep the views dictionary for backward compatibility
        self.views = {
            'live': live_view,
            'logs': logs_view,
            'live_features': features_view,
            'predictions': predictions_view
        }
        
        LOG.info("Views bound to controller")
        
        # Now load initial gamestates since views are available
        self._load_initial_gamestates()
    
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
                self.ui_queue.put(("table_update", (window, changed_mask, feature_names, feature_groups)))
                self._schedule_ui_pump()
                LOG.info("Updated live features view with initial data")
                
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
            
            # Start UI pump
            self._pump_ui()
            
            LOG.info("Live mode started successfully")
            
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
            
            # Reset stop event for next start
            self._stop.clear()
            
        except Exception as e:
            LOG.exception("Failed to stop live mode")
            raise

    def _pump_ui(self):
        """UI pump that processes UI queue every 30ms"""
        try:
            message = self.ui_queue.get_nowait()
        except Empty:
            pass
        else:
            try:
                # Expect exactly ("table_update", (window, changed_mask, feature_names, feature_groups))
                if not (isinstance(message, tuple) and len(message) == 2):
                    raise ValueError(f"UI message must be a 2-tuple; got {type(message).__name__} len={len(message) if isinstance(message, tuple) else 'n/a'}")

                kind, payload = message  # <— no slicing

                if kind != "table_update":
                    raise ValueError(f"Unknown UI message kind: {kind}")

                if not (isinstance(payload, tuple) and len(payload) == 4):
                    raise ValueError("table_update payload must be a 4-tuple (window, changed_mask, feature_names, feature_groups)")

                window, changed_mask, feature_names, feature_groups = payload
                
                # Validate shapes
                if window.shape != (10, 128):
                    raise RuntimeError(f"Window must have shape (10,128), got {window.shape}")
                if changed_mask.shape != (10, 128):
                    raise RuntimeError(f"Changed mask must have shape (10,128), got {changed_mask.shape}")
                
                # First update: set schema *before* any cell writes
                if not self._feature_schema_set:
                    if len(feature_names) != 128 or len(feature_groups) != 128:
                        raise ValueError(
                            f"schema size mismatch: names={len(feature_names)}, groups={len(feature_groups)}, expected 128"
                        )
                    self.live_features_view.set_schema(feature_names, feature_groups)
                    self._feature_schema_set = True
                    LOG.info("UI: schema set (128 names / 128 groups)")
                
                # Push the window into the view
                self.live_features_view.update_from_window(window, changed_mask)
                

                
            except Exception as e:
                LOG.exception("UI apply failed")
                raise
        self.root.after(30, self._pump_ui)
    
    def _schedule_ui_pump(self):
        """Schedule the next UI pump if not already scheduled"""
        if not hasattr(self, '_ui_pump_scheduled'):
            self._ui_pump_scheduled = False
        
        if not self._ui_pump_scheduled:
            self._ui_pump_scheduled = True
            self.root.after(1, self._pump_ui)
    
    def _watcher_worker(self):
        """Worker thread for watching gamestate files"""
        # Precondition checks
        if not hasattr(self, 'live_source') or not self.live_source:
            raise RuntimeError("live_source not initialized or not available")
        if not hasattr(self, 'gs_queue'):
            raise RuntimeError("gs_queue not initialized")
        
        last = None
        while not self._stop.is_set():
            frame_start = time.time()
            
            # Time detect→load
            detect_start = time.time()
            path = self.live_source.wait_for_next_gamestate(last)  # may raise
            detect_time = (time.time() - detect_start) * 1000
            
            # Time load→extract
            load_start = time.time()
            gs = self.live_source.load_json(path)                  # may raise
            load_time = (time.time() - load_start) * 1000
            
            # In watcher loop when a new file arrives, remember the source path
            try:
                self._last_gs_path = gs.get("_source_path")
            except Exception:
                self._last_gs_path = None
            
            self.gs_queue.put(gs)
            # Store timing info for feature worker
            gs['_timing'] = {'detect': detect_time, 'load': load_time}
            
            # *** CRUCIAL: remember what we just processed ***
            last = path

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
                gs = self.gs_queue.get()           # blocks
                
                # Push into pipeline (extract + buffer + warm window)
                window, changed_mask, feature_names, feature_groups = self.feature_pipeline.push(gs)
                
                # DEBUG: Log pipeline state after processing
                if window is None:
                    LOG.error("feature worker: window is None after extract")
                else:
                    # Use the pipeline's computed change mask
                    # Feature names / groups must come from pipeline (or static list if you prefer)
                    feature_names = self.feature_pipeline.feature_names  # len 128
                    feature_groups = self.feature_pipeline.feature_groups  # len 128

                    # POST the full 4-tuple payload the UI expects
                    self.ui_queue.put((
                        "table_update",
                        (window, changed_mask, feature_names, feature_groups)
                    ))
                    
                

                    
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
    
    def get_action_features(self) -> List[float]:
        """Get current action features for display"""
        if hasattr(self, 'actions_service'):
            return self.actions_service.get_action_features()
        return [0.0] * 8
