"""
High-performance Kafka consumer for trading system events.
Handles market data streams, order events, and risk notifications with fault tolerance.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type, Union

import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel, ValidationError

from ..utils.logging import get_logger, set_trace_context


@dataclass
class KafkaConfig:
    """Kafka connection configuration."""
    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_context: Optional[str] = None
    
    # Consumer settings
    group_id: str = "trading_system"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000  # 5 minutes
    session_timeout_ms: int = 10000
    heartbeat_interval_ms: int = 3000
    
    # Producer settings
    compression_type: str = "snappy"
    batch_size: int = 16384
    linger_ms: int = 1
    max_request_size: int = 1048576
    
    # Retry settings
    retry_backoff_ms: int = 100
    request_timeout_ms: int = 30000
    
    # Performance tuning
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500


class BaseKafkaMessage(BaseModel):
    """Base class for all Kafka messages."""
    timestamp: float
    trace_id: Optional[str] = None
    source_service: str
    message_type: str
    version: str = "1.0"


class MarketDataMessage(BaseKafkaMessage):
    """Market data event message."""
    symbol: str
    exchange: str
    price: float
    size: float
    side: Optional[str] = None
    sequence_number: Optional[int] = None


class OrderEventMessage(BaseKafkaMessage):
    """Order execution event message."""
    client_order_id: str
    strategy_id: str
    symbol: str
    exchange: str
    event_type: str  # SUBMITTED, FILLED, CANCELLED, REJECTED
    order_data: Dict[str, Any]


class RiskEventMessage(BaseKafkaMessage):
    """Risk management event message."""
    strategy_id: str
    event_type: str  # LIMIT_BREACH, POSITION_SIZED, PORTFOLIO_REBALANCED
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_data: Dict[str, Any]


class SignalMessage(BaseKafkaMessage):
    """Trading signal message."""
    strategy_id: str
    symbol: str
    side: str  # BUY, SELL
    confidence: float
    signal_data: Dict[str, Any]


# Topic configuration
class KafkaTopics:
    """Standardized Kafka topic names."""
    MARKET_DATA_TICKS = "market.data.ticks"
    MARKET_DATA_CANDLES = "market.data.candles" 
    MARKET_DATA_ORDERBOOK = "market.data.orderbook"
    
    TRADING_SIGNALS = "trading.signals"
    TRADING_ORDERS = "trading.orders"
    TRADING_FILLS = "trading.fills"
    TRADING_POSITIONS = "trading.positions"
    
    RISK_EVENTS = "risk.events"
    RISK_ALERTS = "risk.alerts"
    
    ML_PREDICTIONS = "ml.predictions"
    ML_FEATURES = "ml.features"
    ML_MODEL_UPDATES = "ml.model.updates"
    
    SYSTEM_HEALTH = "system.health"
    AUDIT_EVENTS = "audit.events"


class KafkaMessageHandler:
    """Base class for handling specific message types."""
    
    def __init__(self, message_class: Type[BaseKafkaMessage]):
        self.message_class = message_class
        self.logger = get_logger(f"kafka_handler_{message_class.__name__}")
    
    async def handle_message(self, message: BaseKafkaMessage) -> bool:
        """Process a single message. Return True if successful."""
        raise NotImplementedError("Subclasses must implement handle_message")
    
    def deserialize_message(self, raw_data: bytes) -> Optional[BaseKafkaMessage]:
        """Convert raw bytes to typed message object."""
        try:
            data = json.loads(raw_data.decode('utf-8'))
            return self.message_class(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error("Message deserialization failed", error=str(e), raw_data=raw_data[:100])
            return None


class TradingKafkaConsumer:
    """High-performance async Kafka consumer for trading events."""
    
    def __init__(self, config: KafkaConfig, service_name: str):
        self.config = config
        self.service_name = service_name
        self.logger = get_logger(f"kafka_consumer_{service_name}")
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.handlers: Dict[str, KafkaMessageHandler] = {}
        self.running = False
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'last_message_time': None
        }
    
    def register_handler(self, topic: str, handler: KafkaMessageHandler) -> None:
        """Register message handler for specific topic."""
        self.handlers[topic] = handler
        self.logger.info("Handler registered", topic=topic, handler=handler.__class__.__name__)
    
    async def start(self, topics: List[str]) -> None:
        """Initialize and start consuming messages."""
        try:
            self.consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=f"{self.config.group_id}_{self.service_name}",
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                max_poll_records=self.config.max_poll_records,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                fetch_min_bytes=self.config.fetch_min_bytes,
                fetch_max_wait_ms=self.config.fetch_max_wait_ms,
                security_protocol=self.config.security_protocol
            )
            
            await self.consumer.start()
            self.running = True
            self.logger.info("Kafka consumer started", topics=topics, group_id=self.consumer._group_id)
            
        except Exception as e:
            self.logger.error("Failed to start Kafka consumer", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop consuming and cleanup resources."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            self.logger.info("Kafka consumer stopped")
    
    async def consume_messages(self) -> None:
        """Main consumption loop with error handling and retries."""
        retry_count = 0
        max_retries = 5
        
        while self.running:
            try:
                # Fetch batch of messages with timeout
                msg_batch = await asyncio.wait_for(
                    self.consumer.getmany(timeout_ms=1000),
                    timeout=2.0
                )
                
                if not msg_batch:
                    continue
                
                # Process messages in parallel batches
                tasks = []
                for topic_partition, messages in msg_batch.items():
                    for message in messages:
                        task = asyncio.create_task(
                            self._process_single_message(message, topic_partition.topic)
                        )
                        tasks.append(task)
                
                # Wait for all messages in batch to complete
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    successful = sum(1 for r in results if r is True)
                    failed = len(results) - successful
                    
                    self.stats['messages_processed'] += successful
                    self.stats['messages_failed'] += failed
                    self.stats['last_message_time'] = time.time()
                    
                    # Commit offsets after successful processing
                    if not self.config.enable_auto_commit:
                        await self.consumer.commit()
                
                retry_count = 0  # Reset retry count on successful batch
                
            except asyncio.TimeoutError:
                # Normal timeout - continue consuming
                continue
                
            except KafkaError as e:
                retry_count += 1
                self.logger.error(
                    "Kafka error during consumption", 
                    error=str(e), 
                    retry_count=retry_count
                )
                
                if retry_count >= max_retries:
                    self.logger.critical("Max retries exceeded, stopping consumer")
                    break
                
                # Exponential backoff
                await asyncio.sleep(min(2 ** retry_count, 30))
                
            except Exception as e:
                self.logger.error("Unexpected error in consume loop", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_single_message(self, message, topic: str) -> bool:
        """Process individual message with handler dispatch."""
        try:
            # Set trace context from message if available
            message_data = json.loads(message.value.decode('utf-8'))
            if trace_id := message_data.get('trace_id'):
                set_trace_context(trace_id)
            
            # Find appropriate handler
            handler = self.handlers.get(topic)
            if not handler:
                self.logger.warning("No handler registered for topic", topic=topic)
                return False
            
            # Deserialize and process message
            parsed_message = handler.deserialize_message(message.value)
            if not parsed_message:
                return False
            
            success = await handler.handle_message(parsed_message)
            
            if success:
                self.logger.debug(
                    "Message processed successfully",
                    topic=topic,
                    offset=message.offset,
                    partition=message.partition
                )
            else:
                self.logger.warning(
                    "Message processing failed", 
                    topic=topic,
                    offset=message.offset
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Error processing message",
                topic=topic, 
                offset=message.offset,
                error=str(e)
            )
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            'running': self.running,
            'messages_processed': self.stats['messages_processed'],
            'messages_failed': self.stats['messages_failed'],
            'last_message_time': self.stats['last_message_time'],
            'success_rate': (
                self.stats['messages_processed'] / 
                max(1, self.stats['messages_processed'] + self.stats['messages_failed'])
            ) * 100
        }


class TradingKafkaProducer:
    """High-performance async Kafka producer for trading events."""
    
    def __init__(self, config: KafkaConfig, service_name: str):
        self.config = config
        self.service_name = service_name
        self.logger = get_logger(f"kafka_producer_{service_name}")
        self.producer: Optional[AIOKafkaProducer] = None
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0
        }
    
    async def start(self) -> None:
        """Initialize Kafka producer connection."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                compression_type=self.config.compression_type,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                max_request_size=self.config.max_request_size,
                retry_backoff_ms=self.config.retry_backoff_ms,
                request_timeout_ms=self.config.request_timeout_ms,
                security_protocol=self.config.security_protocol
            )
            
            await self.producer.start()
            self.logger.info("Kafka producer started")
            
        except Exception as e:
            self.logger.error("Failed to start Kafka producer", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop producer and flush pending messages."""
        if self.producer:
            await self.producer.stop()
            self.logger.info("Kafka producer stopped")
    
    async def send_message(self, topic: str, message: BaseKafkaMessage, 
                          key: Optional[str] = None) -> bool:
        """Send typed message to Kafka topic."""
        try:
            if not self.producer:
                raise RuntimeError("Producer not started")
            
            # Serialize message to JSON bytes
            message_dict = asdict(message) if hasattr(message, '__dataclass_fields__') else message.dict()
            message_bytes = json.dumps(message_dict).encode('utf-8')
            key_bytes = key.encode('utf-8') if key else None
            
            # Send with metadata
            await self.producer.send_and_wait(
                topic, 
                value=message_bytes, 
                key=key_bytes
            )
            
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += len(message_bytes)
            
            self.logger.debug(
                "Message sent successfully", 
                topic=topic, 
                key=key,
                message_type=message.message_type
            )
            
            return True
            
        except Exception as e:
            self.stats['messages_failed'] += 1
            self.logger.error(
                "Failed to send message", 
                topic=topic,
                key=key,
                error=str(e)
            )
            return False
    
    async def send_batch(self, topic: str, messages: List[BaseKafkaMessage], 
                        key_fn: Optional[Callable = None) -> int:
        """Send batch of messages efficiently."""
        if not self.producer:
            raise RuntimeError("Producer not started")
        
        successful = 0
        tasks = []
        
        for message in messages:
            key = key_fn(message) if key_fn else None
            task = asyncio.create_task(self.send_message(topic, message, key))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)
        
        self.logger.info(
            "Batch send completed",
            topic=topic,
            total=len(messages),
            successful=successful,
            failed=len(messages) - successful
        )
        
        return successful
    
    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            'messages_sent': self.stats['messages_sent'],
            'messages_failed': self.stats['messages_failed'],
            'bytes_sent': self.stats['bytes_sent'],
            'success_rate': (
                self.stats['messages_sent'] / 
                max(1, self.stats['messages_sent'] + self.stats['messages_failed'])
            ) * 100
        }


@asynccontextmanager
async def kafka_consumer_context(config: KafkaConfig, service_name: str, 
                                topics: List[str]) -> AsyncGenerator[TradingKafkaConsumer, None]:
    """Context manager for Kafka consumer lifecycle."""
    consumer = TradingKafkaConsumer(config, service_name)
    try:
        await consumer.start(topics)
        yield consumer
    finally:
        await consumer.stop()


@asynccontextmanager 
async def kafka_producer_context(config: KafkaConfig, service_name: str) -> AsyncGenerator[TradingKafkaProducer, None]:
    """Context manager for Kafka producer lifecycle."""
    producer = TradingKafkaProducer(config, service_name)
    try:
        await producer.start()
        yield producer
    finally:
        await producer.stop()


# Example message handlers for different trading events
class MarketDataHandler(KafkaMessageHandler):
    """Handler for processing market data events."""
    
    def __init__(self):
        super().__init__(MarketDataMessage)
        self.processed_count = 0
    
    async def handle_message(self, message: MarketDataMessage) -> bool:
        """Process market data tick/candle updates."""
        try:
            # Store in TimescaleDB, update Redis cache, etc.
            self.logger.debug(
                "Processing market data",
                symbol=message.symbol,
                exchange=message.exchange,
                price=message.price
            )
            
            # TODO: Implement actual market data storage
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.logger.error("Market data processing failed", error=str(e))
            return False


class OrderEventHandler(KafkaMessageHandler):
    """Handler for processing order execution events."""
    
    def __init__(self):
        super().__init__(OrderEventMessage)
    
    async def handle_message(self, message: OrderEventMessage) -> bool:
        """Process order state changes and updates."""
        try:
            self.logger.info(
                "Processing order event",
                client_order_id=message.client_order_id,
                event_type=message.event_type,
                strategy=message.strategy_id
            )
            
            # TODO: Update order status in database, notify risk engine
            return True
            
        except Exception as e:
            self.logger.error("Order event processing failed", error=str(e))
            return False


class RiskEventHandler(KafkaMessageHandler):
    """Handler for processing risk management events."""
    
    def __init__(self):
        super().__init__(RiskEventMessage)
    
    async def handle_message(self, message: RiskEventMessage) -> bool:
        """Process risk alerts and limit breaches."""
        try:
            self.logger.warning(
                "Processing risk event",
                event_type=message.event_type,
                severity=message.severity,
                strategy=message.strategy_id
            )
            
            # TODO: Trigger alerts, update risk limits, emergency stops
            if message.severity == "CRITICAL":
                # Implement emergency procedures
                pass
            
            return True
            
        except Exception as e:
            self.logger.error("Risk event processing failed", error=str(e))
            return False


# Example usage and testing
async def example_usage():
    """Demonstrate Kafka consumer/producer usage."""
    config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        group_id="trading_system_test"
    )
    
    # Setup consumer with handlers
    async with kafka_consumer_context(config, "test_service", [KafkaTopics.MARKET_DATA_TICKS]) as consumer:
        # Register handlers
        consumer.register_handler(KafkaTopics.MARKET_DATA_TICKS, MarketDataHandler())
        
        # Setup producer
        async with kafka_producer_context(config, "test_service") as producer:
            # Send test message
            test_message = MarketDataMessage(
                timestamp=time.time(),
                source_service="test",
                message_type="tick",
                symbol="AAPL",
                exchange="NASDAQ",
                price=150.25,
                size=100.0
            )
            
            await producer.send_message(KafkaTopics.MARKET_DATA_TICKS, test_message, key="AAPL")
            
            # Consume for a short time
            consume_task = asyncio.create_task(consumer.consume_messages())
            await asyncio.sleep(5)
            consume_task.cancel()
            
            print("Consumer stats:", consumer.get_stats())
            print("Producer stats:", producer.get_stats())


if __name__ == "__main__":
    asyncio.run(example_usage())