"""
Base interface and abstract classes for market data feeds.
Provides standardized interface for all data providers (Polygon, CCXT, etc.).
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Union

import structlog
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

from ..models.schemas import (
    MarketDataType, Tick, Quote, Candle, OrderBook, 
    Exchange, DataFeedStats
)
from ...shared.utils.logging import get_logger, LogEvents
from ...shared.utils.metrics import TradingMetrics


class FeedStatus(str, Enum):
    """Feed connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class DataType(str, Enum):
    """Supported data types for subscription."""
    TICKS = "ticks"
    QUOTES = "quotes"
    CANDLES = "candles"
    ORDER_BOOK = "order_book"
    NEWS = "news"
    ALL = "all"


class FeedConfig(Dict[str, Any]):
    """Configuration dictionary for feed initialization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default values
        self.setdefault("max_reconnect_attempts", 10)
        self.setdefault("reconnect_delay", 1.0)
        self.setdefault("heartbeat_interval", 30)
        self.setdefault("message_timeout", 10)
        self.setdefault("rate_limit", 100)  # messages per second
        self.setdefault("enable_compression", True)


class FeedError(Exception):
    """Base exception for feed-related errors."""
    pass


class ConnectionError(FeedError):
    """Feed connection error."""
    pass


class AuthenticationError(FeedError):
    """Feed authentication error."""
    pass


class RateLimitError(FeedError):
    """Feed rate limit exceeded error."""
    pass


class BaseDataFeed(ABC):
    """
    Abstract base class for all market data feeds.
    Provides common functionality and standardized interface.
    """
    
    def __init__(
        self,
        name: str,
        exchange: Exchange,
        config: FeedConfig,
        metrics: Optional[TradingMetrics] = None
    ):
        self.name = name
        self.exchange = exchange
        self.config = config
        self.metrics = metrics
        self.logger = get_logger(f"feed_{name}")
        
        # Connection state
        self.status = FeedStatus.DISCONNECTED
        self.connected_at: Optional[float] = None
        self.reconnect_count = 0
        self.last_message_time: Optional[float] = None
        
        # Subscription tracking
        self.subscriptions: Dict[str, Set[DataType]] = {}  # symbol -> data_types
        self.message_handlers: Dict[DataType, Callable] = {}
        
        # Statistics
        self.stats = DataFeedStats(
            feed_name=name,
            symbol="",  # Will be updated per symbol
            exchange=exchange
        )
        
        # Internal state
        self._stop_event = asyncio.Event()
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data feed."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data feed."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to symbols and data types."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from symbols and data types."""
        pass
    
    @abstractmethod
    async def _process_message(self, message: Dict[str, Any]) -> Optional[MarketDataType]:
        """Process raw message into typed data model."""
        pass
    
    async def start(self) -> None:
        """Start the data feed with automatic reconnection."""
        self.logger.info("Starting data feed", feed=self.name, exchange=self.exchange.value)
        
        try:
            await self.connect()
            self.status = FeedStatus.CONNECTED
            self.connected_at = time.time()
            
            # Start background tasks
            self._connection_task = asyncio.create_task(self._connection_monitor())
            if self.config.get("heartbeat_interval", 0) > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            self.logger.info(
                "Data feed started successfully",
                feed=self.name,
                status=self.status.value
            )
            
        except Exception as e:
            self.status = FeedStatus.ERROR
            self.logger.error("Failed to start data feed", feed=self.name, error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the data feed and cleanup resources."""
        self.logger.info("Stopping data feed", feed=self.name)
        
        # Signal stop to all tasks
        self._stop_event.set()
        
        # Cancel background tasks
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        
        # Disconnect from feed
        try:
            await self.disconnect()
        except Exception as e:
            self.logger.warning("Error during disconnect", error=str(e))
        
        self.status = FeedStatus.STOPPED
        self.logger.info("Data feed stopped", feed=self.name)
    
    def register_handler(self, data_type: DataType, handler: Callable[[MarketDataType], None]) -> None:
        """Register message handler for specific data type."""
        self.message_handlers[data_type] = handler
        self.logger.debug("Handler registered", feed=self.name, data_type=data_type.value)
    
    async def _connection_monitor(self) -> None:
        """Monitor connection and handle automatic reconnection."""
        while not self._stop_event.is_set():
            try:
                if self.status == FeedStatus.CONNECTED:
                    # Check if we're still receiving messages
                    if self.last_message_time:
                        silence_duration = time.time() - self.last_message_time
                        timeout = self.config.get("message_timeout", 10)
                        
                        if silence_duration > timeout:
                            self.logger.warning(
                                "Message timeout detected",
                                feed=self.name,
                                silence_duration=silence_duration
                            )
                            await self._attempt_reconnect()
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in connection monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _heartbeat_monitor(self) -> None:
        """Send periodic heartbeat/ping messages."""
        interval = self.config.get("heartbeat_interval", 30)
        
        while not self._stop_event.is_set():
            try:
                if self.status == FeedStatus.CONNECTED:
                    await self._send_heartbeat()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in heartbeat monitor", error=str(e))
                await asyncio.sleep(interval)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError))
    )
    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self.status = FeedStatus.RECONNECTING
        self.reconnect_count += 1
        
        self.logger.info(
            "Attempting reconnection",
            feed=self.name,
            attempt=self.reconnect_count
        )
        
        try:
            # Disconnect existing connection
            await self.disconnect()
            await asyncio.sleep(self.config.get("reconnect_delay", 1.0))
            
            # Reconnect
            await self.connect()
            
            # Resubscribe to all symbols
            if self.subscriptions:
                for symbol, data_types in self.subscriptions.items():
                    await self.subscribe([symbol], list(data_types))
            
            self.status = FeedStatus.CONNECTED
            self.connected_at = time.time()
            
            self.logger.info(
                "Reconnection successful",
                feed=self.name,
                attempt=self.reconnect_count
            )
            
        except Exception as e:
            self.status = FeedStatus.ERROR
            self.logger.error(
                "Reconnection failed",
                feed=self.name,
                attempt=self.reconnect_count,
                error=str(e)
            )
            raise
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat/ping message - override in subclasses."""
        pass
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        rate_limit = self.config.get("rate_limit", 100)
        
        # Reset counter every second
        if current_time - self._rate_limit_window_start >= 1.0:
            self._request_count = 0
            self._rate_limit_window_start = current_time
        
        # Check if we've exceeded the rate limit
        if self._request_count >= rate_limit:
            sleep_time = 1.0 - (current_time - self._rate_limit_window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()
        
        self._request_count += 1
    
    def _update_stats(self, message_type: str, processing_time: float = 0.0) -> None:
        """Update feed statistics."""
        self.last_message_time = time.time()
        
        if message_type == "tick":
            self.stats.ticks_received += 1
        elif message_type == "quote":
            self.stats.quotes_received += 1
        elif message_type == "candle":
            self.stats.candles_received += 1
        
        # Update latency metrics
        if processing_time > 0:
            if self.stats.avg_latency_ms == 0:
                self.stats.avg_latency_ms = processing_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.stats.avg_latency_ms = (alpha * processing_time + 
                                           (1 - alpha) * self.stats.avg_latency_ms)
            
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, processing_time)
        
        self.stats.connected = (self.status == FeedStatus.CONNECTED)
        self.stats.last_message_time = self.last_message_time
        self.stats.reconnect_count = self.reconnect_count
    
    async def _handle_message(self, raw_message: Dict[str, Any]) -> None:
        """Handle incoming message and dispatch to appropriate handler."""
        try:
            start_time = time.perf_counter()
            
            # Process message into typed data model
            processed_data = await self._process_message(raw_message)
            if not processed_data:
                return
            
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            
            if self.metrics:
                self.metrics.record_data_received(
                    processed_data.symbol,
                    processed_data.exchange.value,
                    1
                )
                
                # Calculate latency from message timestamp
                if hasattr(processed_data, 'timestamp'):
                    latency = time.time() - processed_data.timestamp.timestamp()
                    self.metrics.record_data_latency(
                        processed_data.symbol,
                        processed_data.exchange.value,
                        latency
                    )
            
            # Determine message type for stats
            message_type = type(processed_data).__name__.lower()
            self._update_stats(message_type, processing_time)
            
            # Dispatch to registered handlers
            data_type = self._get_data_type_from_message(processed_data)
            if data_type in self.message_handlers:
                handler = self.message_handlers[data_type]
                await self._safe_handle(handler, processed_data)
            
            self.logger.debug(
                "Message processed",
                feed=self.name,
                symbol=processed_data.symbol,
                type=message_type,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.stats.error_count += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = time.time()
            
            self.logger.error(
                "Error processing message",
                feed=self.name,
                error=str(e),
                raw_message=raw_message
            )
    
    def _get_data_type_from_message(self, message: MarketDataType) -> DataType:
        """Determine data type from message instance."""
        if isinstance(message, Tick):
            return DataType.TICKS
        elif isinstance(message, Quote):
            return DataType.QUOTES
        elif isinstance(message, Candle):
            return DataType.CANDLES
        elif isinstance(message, OrderBook):
            return DataType.ORDER_BOOK
        else:
            return DataType.ALL
    
    async def _safe_handle(self, handler: Callable, data: MarketDataType) -> None:
        """Safely execute handler with error isolation."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            self.logger.error(
                "Handler execution failed",
                feed=self.name,
                handler=handler.__name__,
                error=str(e)
            )
    
    def get_stats(self) -> DataFeedStats:
        """Get current feed statistics."""
        stats = self.stats.copy()
        stats.connected = (self.status == FeedStatus.CONNECTED)
        return stats
    
    def is_connected(self) -> bool:
        """Check if feed is currently connected."""
        return self.status == FeedStatus.CONNECTED
    
    def get_subscriptions(self) -> Dict[str, Set[DataType]]:
        """Get current symbol subscriptions."""
        return self.subscriptions.copy()
    
    @asynccontextmanager
    async def managed_connection(self) -> AsyncGenerator['BaseDataFeed', None]:
        """Context manager for automatic connection lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


# Utility functions for feed management
async def test_feed_connection(feed: BaseDataFeed, timeout: float = 10.0) -> bool:
    """Test if a feed can connect successfully."""
    try:
        async with asyncio.timeout(timeout):
            await feed.start()
            await asyncio.sleep(1)  # Brief connection test
            return feed.is_connected()
    except Exception as e:
        feed.logger.error("Connection test failed", error=str(e))
        return False
    finally:
        try:
            await feed.stop()
        except:
            pass


def create_feed_config(**kwargs) -> FeedConfig:
    """Create feed configuration with defaults."""
    return FeedConfig(**kwargs)