"""
Message Bus System
Event-driven communication system for components
"""

from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict
import threading
import time
import uuid
from loguru import logger

from SLM.core.singleton import Singleton


class MessageBus(Singleton):
    """
    Singleton event-driven message bus for component communication
    """

    def __init__(self):
        """
        Initialize the message bus
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: List[Dict[str, Any]] = []
        self._processing = False
        self._lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self._initialized = True

    def subscribe(self, message_type: str, handler: Callable):
        """
        Subscribe to a message type

        Args:
            message_type: Type of message to subscribe to
            handler: Function to call when message is received
        """
        with self._lock:
            if handler not in self._subscribers[message_type]:
                self._subscribers[message_type].append(handler)
                logger.debug(f"Subscribed {getattr(handler, '__name__', 'handler')} to {message_type}")

    def unsubscribe(self, message_type: str, handler: Callable):
        """
        Unsubscribe from a message type

        Args:
            message_type: Type of message to unsubscribe from
            handler: Handler function to remove
        """
        with self._lock:
            if handler in self._subscribers[message_type]:
                self._subscribers[message_type].remove(handler)
                logger.debug(f"Unsubscribed {getattr(handler, '__name__', 'handler')} from {message_type}")

    def publish(self, message_type: str, **kwargs):
        """
        Publish a message synchronously

        Args:
            message_type: Type of message
            **kwargs: Message data
        """
        message = {
            'id': str(uuid.uuid4()),
            'type': message_type,
            'timestamp': time.time(),
            'data': kwargs
        }

        with self._lock:
            if self._processing:
                # Queue message for background processing
                self._message_queue.append(message)
            else:
                # Process immediately
                self._process_message(message)

    def start_processing(self):
        """
        Start background message processing thread
        """
        if self._processing:
            return

        self._processing = True

        def process_queue():
            while self._processing:
                with self._lock:
                    if self._message_queue:
                        message = self._message_queue.pop(0)
                        self._process_message(message)
                time.sleep(0.01)  # Small delay to prevent busy waiting

        self._processing_thread = threading.Thread(target=process_queue, daemon=True)
        self._processing_thread.start()
        logger.info("Message bus background processing started")

    def stop_processing(self):
        """
        Stop background message processing
        """
        self._processing = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        logger.info("Message bus background processing stopped")

    def _process_message(self, message: Dict[str, Any]):
        """
        Process a single message synchronously

        Args:
            message: Message to process
        """
        message_type = message['type']
        kwargs = message['data']
        handlers = self._subscribers.get(message_type, [])
        
        if handlers:
            logger.debug(f"Processing message: {message_type} with {len(handlers)} handlers")

        for handler in handlers:
            logger.debug(f"  -> Calling handler: {getattr(handler, '__name__', 'unknown')} for message {message_type}")
            try:
                handler(message_type, **kwargs)
                logger.debug(f"  <- Finished handler: {getattr(handler, '__name__', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing message {message_type} with handler {getattr(handler, '__name__', 'unknown')}: {e}")

    def clear_queue(self):
        """
        Clear the message queue
        """
        with self._lock:
            self._message_queue.clear()

    def get_subscriber_count(self, message_type: str) -> int:
        """
        Get the number of subscribers for a message type

        Args:
            message_type: Message type to check

        Returns:
            Number of subscribers
        """
        with self._lock:
            return len(self._subscribers.get(message_type, []))

    def get_message_types(self) -> List[str]:
        """
        Get all registered message types

        Returns:
            List of message types
        """
        with self._lock:
            return list(self._subscribers.keys())

    def get_queue_length(self) -> int:
        """
        Get the current queue length

        Returns:
            Number of messages in queue
        """
        with self._lock:
            return len(self._message_queue)

    def has_subscribers(self, message_type: str) -> bool:
        """
        Check if a message type has subscribers

        Args:
            message_type: Message type to check

        Returns:
            True if has subscribers
        """
        with self._lock:
            return len(self._subscribers.get(message_type, [])) > 0

    def wait_for_message(self, message_type: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Wait for a specific message type

        Args:
            message_type: Message type to wait for
            timeout: Timeout in seconds

        Returns:
            Message if received, None if timeout
        """
        start_time = time.time()
        received_message = None
        message_event = threading.Event()

        def message_handler(msg_type, **kwargs):
            nonlocal received_message
            if msg_type == message_type:
                received_message = {
                    'id': str(uuid.uuid4()),
                    'type': msg_type,
                    'timestamp': time.time(),
                    'data': kwargs
                }
                message_event.set()

        # Subscribe to the message type
        self.subscribe(message_type, message_handler)

        try:
            # Wait for message or timeout
            message_event.wait(timeout=timeout)
        finally:
            # Unsubscribe
            self.unsubscribe(message_type, message_handler)

        return received_message

    def publish_and_wait(self, message_type: str, response_type: str, timeout: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Publish a message and wait for a specific response type

        Args:
            message_type: Type of message to publish
            response_type: Type of response to wait for
            timeout: Timeout in seconds
            **kwargs: Message data

        Returns:
            Response message if received, None if timeout
        """
        # Set up listener for response first
        response_event = threading.Event()
        response_data = {'message': None}

        def response_handler(msg_type, **response_kwargs):
            if msg_type == response_type:
                response_data['message'] = {
                    'id': str(uuid.uuid4()),
                    'type': msg_type,
                    'timestamp': time.time(),
                    'data': response_kwargs
                }
                response_event.set()

        self.subscribe(response_type, response_handler)

        try:
            # Publish the original message
            self.publish(message_type, **kwargs)
            
            # Wait for response
            if response_event.wait(timeout=timeout):
                return response_data['message']
            return None
        finally:
            self.unsubscribe(response_type, response_handler)

    def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all subscribers

        Args:
            message: Message to broadcast
        """
        with self._lock:
            for message_type in self._subscribers:
                handlers = self._subscribers[message_type]
                for handler in handlers:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting message to {message_type}: {e}")

    def __repr__(self):
        return f"MessageBus(subscribers={len(self._subscribers)}, queue={len(self._message_queue)})"

    def __str__(self):
        return f"MessageBus with {len(self._subscribers)} message types and {len(self._message_queue)} queued messages"
