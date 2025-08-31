"""
FastAPI application for data ingestion service.
Provides REST API, WebSocket endpoints, and feed management.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_config, DataIngestConfig, TradingMode
from .feeds.base import BaseDataFeed, DataType
from .feeds.polygon import create_polygon_feed
from .feeds.ccxt_wrapper import create_binance_feed, create_coinbase_feed
from .feeds.mock_exchange import create_mock_feed
from .models.schemas import (
    SymbolRequest, HealthCheckResponse, MetricsResponse, 
    DataFeedStats, MarketDataType, ProcessingResult
)
from ..shared.utils.logging import setup_logging, get_logger, set_trace_context, generate_trace_id
from ..shared.utils.metrics import TradingMetrics, MetricsMiddleware
from ..shared.messaging.kafka.consumer import (
    TradingKafkaProducer, KafkaConfig, MarketDataMessage, KafkaTopics
)


class FeedManager:
    """Manages multiple data feeds and their lifecycles."""
    
    def __init__(self, config: DataIngestConfig, metrics: TradingMetrics):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("feed_manager")
        
        # Active feeds
        self.feeds: Dict[str, BaseDataFeed] = {}
        self.feed_stats: Dict[str, DataFeedStats] = {}
        
        # Message processing
        self.kafka_producer: Optional[TradingKafkaProducer] = None
        self.message_buffer: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # WebSocket connections for real-time data
        self.websocket_connections: List[WebSocket] = []
        
        # Processing statistics
        self.processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "start_time": time.time()
        }
    
    async def initialize(self) -> None:
        """Initialize feed manager and create feeds."""
        try:
            # Initialize Kafka producer
            kafka_config = KafkaConfig(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                compression_type=self.config.kafka_compression,
                batch_size=self.config.kafka_batch_size
            )
            
            self.kafka_producer = TradingKafkaProducer(kafka_config, self.config.service_name)
            await self.kafka_producer.start()
            
            # Create and configure feeds based on config
            await self._create_feeds()
            
            # Start message processing loop
            asyncio.create_task(self._process_message_queue())
            
            self.logger.info(
                "Feed manager initialized",
                total_feeds=len(self.feeds),
                kafka_servers=self.config.kafka_bootstrap_servers
            )
            
        except Exception as e:
            self.logger.error("Feed manager initialization failed", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all feeds and cleanup resources."""
        self.logger.info("Shutting down feed manager")
        
        # Stop all feeds
        for feed_name, feed in self.feeds.items():
            try:
                await feed.stop()
                self.logger.info("Feed stopped", feed=feed_name)
            except Exception as e:
                self.logger.error("Error stopping feed", feed=feed_name, error=str(e))
        
        # Close Kafka producer
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        # Close WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.close()
            except:
                pass
        
        self.logger.info("Feed manager shutdown complete")
    
    async def _create_feeds(self) -> None:
        """Create and configure data feeds based on configuration."""
        
        # Create Polygon feed if API key provided
        if self.config.polygon_api_key and self.config.polygon_api_key != "demo":
            polygon_feed = create_polygon_feed(
                api_key=self.config.polygon_api_key,
                rate_limit=self.config.rate_limit_requests_per_second
            )
            
            polygon_feed.register_handler(DataType.TICKS, self._handle_tick_data)
            polygon_feed.register_handler(DataType.QUOTES, self._handle_quote_data)
            polygon_feed.register_handler(DataType.CANDLES, self._handle_candle_data)
            
            self.feeds["polygon"] = polygon_feed
        
        # Create CCXT feeds for crypto exchanges
        if "binance" in self.config.ccxt_exchanges:
            binance_feed = create_binance_feed(sandbox=self.config.ccxt_sandbox)
            binance_feed.register_handler(DataType.TICKS, self._handle_tick_data)
            binance_feed.register_handler(DataType.QUOTES, self._handle_quote_data)
            binance_feed.register_handler(DataType.ORDER_BOOK, self._handle_orderbook_data)
            
            self.feeds["binance"] = binance_feed
        
        if "coinbase" in self.config.ccxt_exchanges:
            coinbase_feed = create_coinbase_feed(sandbox=self.config.ccxt_sandbox)
            coinbase_feed.register_handler(DataType.TICKS, self._handle_tick_data)
            coinbase_feed.register_handler(DataType.QUOTES, self._handle_quote_data)
            
            self.feeds["coinbase"] = coinbase_feed
        
        # Always create mock feed for testing
        if self.config.enable_mock_feeds or self.config.trading_mode == TradingMode.TEST:
            mock_feed = create_mock_feed(
                tick_interval=0.1,
                quote_interval=1.0,
                enable_trends=True,
                data_file=self.config.mock_data_file
            )
            
            mock_feed.register_handler(DataType.TICKS, self._handle_tick_data)
            mock_feed.register_handler(DataType.QUOTES, self._handle_quote_data)
            mock_feed.register_handler(DataType.CANDLES, self._handle_candle_data)
            mock_feed.register_handler(DataType.ORDER_BOOK, self._handle_orderbook_data)
            
            self.feeds["mock"] = mock_feed
    
    async def start_feeds(self) -> None:
        """Start all configured feeds."""
        for feed_name, feed in self.feeds.items():
            try:
                await feed.start()
                self.logger.info("Feed started", feed=feed_name, status=feed.status.value)
            except Exception as e:
                self.logger.error("Failed to start feed", feed=feed_name, error=str(e))
    
    async def subscribe_to_symbols(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to symbols across all feeds."""
        success_count = 0
        
        for feed_name, feed in self.feeds.items():
            if feed.is_connected():
                try:
                    # Filter symbols based on feed type
                    feed_symbols = self._filter_symbols_for_feed(feed_name, symbols)
                    
                    if feed_symbols:
                        success = await feed.subscribe(feed_symbols, data_types)
                        if success:
                            success_count += 1
                            self.logger.info(
                                "Subscription successful",
                                feed=feed_name,
                                symbols=feed_symbols,
                                data_types=[dt.value for dt in data_types]
                            )
                        else:
                            self.logger.warning("Subscription failed", feed=feed_name)
                            
                except Exception as e:
                    self.logger.error("Subscription error", feed=feed_name, error=str(e))
        
        return success_count > 0
    
    def _filter_symbols_for_feed(self, feed_name: str, symbols: List[str]) -> List[str]:
        """Filter symbols based on feed capabilities."""
        if feed_name == "polygon":
            # Polygon handles stocks, options, forex
            return [s for s in symbols if not self._is_crypto_symbol(s)]
        elif feed_name in ["binance", "coinbase"]:
            # Crypto exchanges handle crypto pairs
            return [s for s in symbols if self._is_crypto_symbol(s)]
        else:
            # Mock feed handles everything
            return symbols
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency pair."""
        crypto_indicators = ["-USD", "USDT", "USDC", "BTC", "ETH"]
        return any(indicator in symbol.upper() for indicator in crypto_indicators)
    
    async def _handle_tick_data(self, tick_data: Dict[str, Any]) -> None:
        """Handle incoming tick data."""
        try:
            # Add to processing queue
            await self.message_buffer.put(("tick", tick_data))
            
            # Update metrics
            if self.metrics:
                self.metrics.record_data_received(
                    tick_data.get("symbol", ""),
                    tick_data.get("exchange", ""),
                    1
                )
                
        except Exception as e:
            self.logger.error("Tick data handling failed", error=str(e))
    
    async def _handle_quote_data(self, quote_data: Dict[str, Any]) -> None:
        """Handle incoming quote data."""
        try:
            await self.message_buffer.put(("quote", quote_data))
            
            if self.metrics:
                self.metrics.record_data_received(
                    quote_data.get("symbol", ""),
                    quote_data.get("exchange", ""),
                    1
                )
                
        except Exception as e:
            self.logger.error("Quote data handling failed", error=str(e))
    
    async def _handle_candle_data(self, candle_data: Dict[str, Any]) -> None:
        """Handle incoming candle data."""
        try:
            await self.message_buffer.put(("candle", candle_data))
            
            if self.metrics:
                self.metrics.record_data_received(
                    candle_data.get("symbol", ""),
                    candle_data.get("exchange", ""),
                    1
                )
                
        except Exception as e:
            self.logger.error("Candle data handling failed", error=str(e))
    
    async def _handle_orderbook_data(self, orderbook_data: Dict[str, Any]) -> None:
        """Handle incoming order book data."""
        try:
            await self.message_buffer.put(("orderbook", orderbook_data))
            
            if self.metrics:
                self.metrics.record_data_received(
                    orderbook_data.get("symbol", ""),
                    orderbook_data.get("exchange", ""),
                    1
                )
                
        except Exception as e:
            self.logger.error("Order book data handling failed", error=str(e))
    
    async def _process_message_queue(self) -> None:
        """Process messages from the queue and forward to Kafka/WebSockets."""
        batch_size = self.config.batch_size
        flush_interval = self.config.flush_interval_seconds
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Collect messages for batching
                try:
                    message_type, message_data = await asyncio.wait_for(
                        self.message_buffer.get(),
                        timeout=0.1
                    )
                    batch.append((message_type, message_data))
                except asyncio.TimeoutError:
                    pass
                
                # Flush batch if size or time threshold reached
                current_time = time.time()
                should_flush = (
                    len(batch) >= batch_size or
                    (batch and current_time - last_flush >= flush_interval)
                )
                
                if should_flush and batch:
                    await self._process_message_batch(batch)
                    batch.clear()
                    last_flush = current_time
                
            except Exception as e:
                self.logger.error("Message queue processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_message_batch(self, batch: List[tuple]) -> None:
        """Process a batch of messages."""
        try:
            # Send to Kafka
            kafka_tasks = []
            
            for message_type, message_data in batch:
                if self.kafka_producer:
                    # Create Kafka message
                    kafka_message = MarketDataMessage(
                        timestamp=time.time(),
                        source_service=self.config.service_name,
                        message_type=message_type,
                        symbol=message_data.get("symbol", ""),
                        exchange=message_data.get("exchange", ""),
                        price=float(message_data.get("price", 0)),
                        size=float(message_data.get("size", 0))
                    )
                    
                    # Select appropriate topic
                    topic = KafkaTopics.MARKET_DATA_TICKS
                    if message_type == "candle":
                        topic = KafkaTopics.MARKET_DATA_CANDLES
                    elif message_type == "orderbook":
                        topic = KafkaTopics.MARKET_DATA_ORDERBOOK
                    
                    task = asyncio.create_task(
                        self.kafka_producer.send_message(topic, kafka_message)
                    )
                    kafka_tasks.append(task)
            
            # Wait for Kafka sends to complete
            if kafka_tasks:
                results = await asyncio.gather(*kafka_tasks, return_exceptions=True)
                successful_sends = sum(1 for r in results if r is True)
                failed_sends = len(results) - successful_sends
                
                self.processing_stats["messages_processed"] += successful_sends
                self.processing_stats["messages_failed"] += failed_sends
            
            # Send to WebSocket clients
            await self._broadcast_to_websockets(batch)
            
        except Exception as e:
            self.logger.error("Batch processing failed", error=str(e))
            self.processing_stats["messages_failed"] += len(batch)
    
    async def _broadcast_to_websockets(self, batch: List[tuple]) -> None:
        """Broadcast messages to connected WebSocket clients."""
        if not self.websocket_connections:
            return
        
        # Prepare messages for WebSocket transmission
        ws_messages = []
        for message_type, message_data in batch:
            ws_message = {
                "type": message_type,
                "data": message_data,
                "timestamp": time.time()
            }
            ws_messages.append(ws_message)
        
        # Send to all connected clients
        disconnected_clients = []
        
        for ws in self.websocket_connections:
            try:
                await ws.send_json({"messages": ws_messages})
            except Exception:
                disconnected_clients.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected_clients:
            self.websocket_connections.remove(ws)
    
    def get_feed_stats(self) -> List[DataFeedStats]:
        """Get statistics for all feeds."""
        stats = []
        
        for feed_name, feed in self.feeds.items():
            try:
                feed_stats = feed.get_stats()
                feed_stats.feed_name = feed_name
                stats.append(feed_stats)
            except Exception as e:
                self.logger.error("Error getting feed stats", feed=feed_name, error=str(e))
        
        return stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get message processing statistics."""
        current_time = time.time()
        uptime = current_time - self.processing_stats["start_time"]
        
        total_messages = (
            self.processing_stats["messages_processed"] + 
            self.processing_stats["messages_failed"]
        )
        
        messages_per_second = total_messages / max(uptime, 1)
        
        return {
            "uptime_seconds": uptime,
            "messages_processed": self.processing_stats["messages_processed"],
            "messages_failed": self.processing_stats["messages_failed"],
            "messages_per_second": round(messages_per_second, 2),
            "success_rate": (
                self.processing_stats["messages_processed"] / max(total_messages, 1) * 100
            ),
            "queue_size": self.message_buffer.qsize()
        }


# Global feed manager instance
feed_manager: Optional[FeedManager] = None


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global feed_manager
    
    config = get_config()
    logger = get_logger("app_lifecycle")
    
    try:
        # Initialize metrics
        metrics = TradingMetrics(config.service_name)
        
        # Create and initialize feed manager
        feed_manager = FeedManager(config, metrics)
        await feed_manager.initialize()
        
        # Start feeds
        await feed_manager.start_feeds()
        
        # Subscribe to configured symbols
        if config.symbols:
            symbols = list(config.symbols)
            data_types = [DataType.TICKS, DataType.QUOTES]
            
            if config.trading_mode == TradingMode.TEST:
                data_types.extend([DataType.CANDLES, DataType.ORDER_BOOK])
            
            await feed_manager.subscribe_to_symbols(symbols, data_types)
        
        logger.info("Application startup complete")
        
        yield
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    finally:
        # Cleanup
        if feed_manager:
            await feed_manager.shutdown()
        
        logger.info("Application shutdown complete")


# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    
    # Initialize logging
    setup_logging(config.service_name, config.log_level.value)
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Trading Data Ingestion Service",
        description="High-frequency market data ingestion and distribution",
        version=config.service_version,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Health check and monitoring"},
            {"name": "feeds", "description": "Data feed management"},
            {"name": "websocket", "description": "Real-time data streaming"},
        ]
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    if config.enable_compression:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add metrics middleware if enabled
    if config.enable_metrics:
        metrics = TradingMetrics(config.service_name)
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    
    return app


# Create app instance
app = create_app()
config = get_config()


# Dependency injection
def get_feed_manager() -> FeedManager:
    """Get feed manager dependency."""
    if feed_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return feed_manager


# API Endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check(fm: FeedManager = Depends(get_feed_manager)):
    """Health check endpoint with detailed service status."""
    stats = fm.get_processing_stats()
    feed_stats = fm.get_feed_stats()
    
    connected_feeds = sum(1 for feed in feed_stats if feed.connected)
    total_feeds = len(feed_stats)
    
    # Determine service status
    if total_feeds == 0:
        status = "starting"
    elif connected_feeds == 0:
        status = "degraded"
    elif connected_feeds < total_feeds:
        status = "partial"
    else:
        status = "healthy"
    
    return HealthCheckResponse(
        status=status,
        timestamp=time.time(),
        version=config.service_version,
        uptime_seconds=stats["uptime_seconds"],
        feeds_connected=connected_feeds,
        total_feeds=total_feeds,
        last_data_received=None  # TODO: Track last message time
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["health"])
async def get_metrics(fm: FeedManager = Depends(get_feed_manager)):
    """Get detailed service metrics."""
    processing_stats = fm.get_processing_stats()
    feed_stats = fm.get_feed_stats()
    
    return MetricsResponse(
        service=config.service_name,
        timestamp=time.time(),
        feeds=feed_stats,
        total_messages=processing_stats["messages_processed"],
        messages_per_second=processing_stats["messages_per_second"],
        avg_processing_time_ms=0.0  # TODO: Track processing time
    )


@app.post("/feeds/subscribe", tags=["feeds"])
async def subscribe_to_symbol(
    request: SymbolRequest,
    fm: FeedManager = Depends(get_feed_manager)
):
    """Subscribe to market data for a symbol."""
    trace_id = generate_trace_id()
    set_trace_context(trace_id)
    
    try:
        # Convert string data types to enum
        data_types = [DataType(dt) for dt in request.data_types]
        
        success = await fm.subscribe_to_symbols([request.symbol], data_types)
        
        if success:
            return {"success": True, "message": f"Subscribed to {request.symbol}"}
        else:
            raise HTTPException(status_code=400, detail="Subscription failed")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription error: {e}")


@app.get("/feeds/status", tags=["feeds"])
async def get_feed_status(fm: FeedManager = Depends(get_feed_manager)):
    """Get status of all data feeds."""
    feed_stats = fm.get_feed_stats()
    
    return {
        "feeds": [
            {
                "name": stats.feed_name,
                "connected": stats.connected,
                "ticks_received": stats.ticks_received,
                "quotes_received": stats.quotes_received,
                "last_message_time": stats.last_message_time,
                "error_count": stats.error_count
            }
            for stats in feed_stats
        ],
        "total_feeds": len(feed_stats),
        "connected_feeds": sum(1 for stats in feed_stats if stats.connected)
    }


@app.websocket("/ws/market-data")
async def websocket_market_data(
    websocket: WebSocket,
    fm: FeedManager = Depends(get_feed_manager)
):
    """WebSocket endpoint for real-time market data streaming."""
    await websocket.accept()
    fm.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            message = await websocket.receive_text()
            
            # Handle client commands (subscribe/unsubscribe)
            try:
                command = json.loads(message)
                if command.get("action") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except:
                pass  # Ignore invalid messages
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in fm.websocket_connections:
            fm.websocket_connections.remove(websocket)


@app.get("/debug/stats", tags=["debug"])
async def get_debug_stats(fm: FeedManager = Depends(get_feed_manager)):
    """Get detailed debugging statistics (only in development)."""
    if not config.enable_debug_endpoints:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "processing_stats": fm.get_processing_stats(),
        "feed_subscriptions": {
            name: feed.get_subscriptions()
            for name, feed in fm.feeds.items()
        },
        "websocket_connections": len(fm.websocket_connections),
        "message_queue_size": fm.message_buffer.qsize(),
        "config": {
            "trading_mode": config.trading_mode.value,
            "symbols": list(config.symbols),
            "exchanges": list(config.exchanges),
            "enable_mock_feeds": config.enable_mock_feeds
        }
    }


if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "data_ingest.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.value.lower(),
        workers=1 if config.reload else config.workers
    )