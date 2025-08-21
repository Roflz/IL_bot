#!/usr/bin/env python3
"""Thread-safe queues and dispatcher for GUI updates"""

import queue
import time
import logging
from typing import Any, Callable, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CoalescedQueue:
    """Queue that coalesces similar messages to prevent UI spam"""
    
    def __init__(self, maxsize: int = 100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.last_message = None
        self.last_message_time = 0
        self.coalesce_interval = 0.1  # 100ms
    
    def put(self, message: Any, coalesce_key: Optional[str] = None):
        """Put a message in the queue, optionally coalescing similar messages"""
        current_time = time.time()
        
        if coalesce_key and self.last_message == coalesce_key:
            if current_time - self.last_message_time < self.coalesce_interval:
                # Skip duplicate message
                return
        
        try:
            self.queue.put_nowait(message)
            self.last_message = coalesce_key
            self.last_message_time = current_time
        except queue.Full:
            logger.warning("Queue full, dropping message")
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Get a message from the queue"""
        return self.queue.get(block, timeout)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()


class MessageDispatcher:
    """Dispatches messages from queues to handlers"""
    
    def __init__(self):
        self.handlers = defaultdict(list)
        self.queues = {}
        self.running = False
    
    def register_queue(self, name: str, queue_obj: CoalescedQueue):
        """Register a named queue"""
        self.queues[name] = queue_obj
    
    def register_handler(self, queue_name: str, handler: Callable[[Any], None]):
        """Register a handler for a specific queue"""
        self.handlers[queue_name].append(handler)
    
    def start(self):
        """Start the dispatcher"""
        self.running = True
        logger.info("Message dispatcher started")
    
    def stop(self):
        """Stop the dispatcher"""
        self.running = False
        logger.info("Message dispatcher stopped")

    def process_queues(self):
        """Process all registered queues"""
        if not self.running:
            return

        for queue_name, queue_obj in self.queues.items():
            # Check if queue has updates (works with both CoalescedQueue and UpdateQueue)
            if hasattr(queue_obj, 'has_updates'):
                has_data = queue_obj.has_updates()
            elif hasattr(queue_obj, 'empty'):
                has_data = not queue_obj.empty()
            else:
                has_data = False
                
            if has_data:
                try:
                    # Get message from queue
                    if hasattr(queue_obj, 'get_update'):
                        message = queue_obj.get_update()
                    elif hasattr(queue_obj, 'get'):
                        message = queue_obj.get()
                    else:
                        continue
                        
                    self._dispatch_message(queue_name, message)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message from {queue_name}: {e}")
    
    def _dispatch_message(self, queue_name: str, message: Any):
        """Dispatch a message to all registered handlers"""
        handlers = self.handlers.get(queue_name, [])
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Error in handler for {queue_name}: {e}")


class UpdateQueue:
    """Specialized queue for UI updates"""
    
    def __init__(self, name: str, maxsize: int = 50):
        self.name = name
        self.queue = CoalescedQueue(maxsize)
        self.last_update_time = 0
        self.update_interval = 0.1  # 100ms minimum between updates
    
    def put_update(self, update_data: Any, coalesce_key: Optional[str] = None):
        """Put an update in the queue"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.queue.put(update_data, coalesce_key)
        self.last_update_time = current_time
    
    def get_update(self) -> Any:
        """Get an update from the queue"""
        return self.queue.get()
    
    def has_updates(self) -> bool:
        """Check if there are updates available"""
        return not self.queue.empty()
    
    def empty(self) -> bool:
        """Check if queue is empty (alias for not has_updates)"""
        return not self.has_updates()
    
    def clear(self):
        """Clear all pending updates"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class FeatureUpdateQueue(UpdateQueue):
    """Queue specifically for feature updates"""
    
    def __init__(self):
        super().__init__("features")
    
    def put_feature_update(self, features: Any, timestamp: float):
        """Put a feature update in the queue"""
        self.put_update({
            'type': 'features',
            'data': features,
            'timestamp': timestamp
        }, coalesce_key='features')
    
    def put_action_update(self, actions: Any, timestamp: float):
        """Put an action update in the queue"""
        self.put_update({
            'type': 'actions',
            'data': actions,
            'timestamp': timestamp
        }, coalesce_key='actions')


class PredictionUpdateQueue(UpdateQueue):
    """Queue specifically for prediction updates"""
    
    def __init__(self):
        super().__init__("predictions")
    
    def put_prediction(self, prediction: Any, timestamp: float):
        """Put a prediction in the queue"""
        self.put_update({
            'type': 'prediction',
            'data': prediction,
            'timestamp': timestamp
        }, coalesce_key='prediction')
    
    def put_status_update(self, status: dict):
        """Put a status update in the queue"""
        self.put_update({
            'type': 'status',
            'data': status
        }, coalesce_key='status')
