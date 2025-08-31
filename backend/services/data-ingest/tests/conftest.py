"""
Test configuration and fixtures for data ingestion service.
Provides reusable test components, mocks, and database setup.
"""

import asyncio
import json
import os
import tempfile
import time
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pandas as pd

from data_ingest.app import create_app
from data_ingest.config import DataIngestConfig, TradingMode, LogLevel
from data_ingest.feeds.base import BaseDataFeed, DataType, FeedConfig
from data_ingest.feeds.mock_exchange import MockExchangeFeed, create_mock_feed
from data_ingest.models.schemas import Tick, Quote, Candle, OrderBook, Exchange, Side
from ..shared.utils.logging import setup_logging


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> DataIngestConfig:
    """Create test configuration."""
    return DataIngestConfig(
        environment="test",
        service_name="data-ingest-test",
        trading_mode=TradingMode.TEST,
        log_level=LogLevel.DEBUG,
        
        # Test database URLs
        postgres_url="postgresql://test_user:test_pass@localhost:5432/test_db",
        redis_url="redis://localhost:6379/15",
        
        # Test Kafka settings
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="test_group",
        
        # Enable mock feeds for testing
        enable_mock_feeds=True,
        enable_debug_endpoints=True,
        
        # Test symbols
        symbols={"AAPL", "GOOGL", "BTCUSD"},
        exchanges={"NASDAQ", "MOCK"},
        
        # Performance settings for tests
        batch_size=10,
        flush_interval_seconds=1,
        max_queue_size=100,
        
        # Disable external services
        polygon_api_key="test_key",
        ccxt_sandbox=True
    )


@pytest.fixture
def app(test_config):
    """Create FastAPI test application."""
    with patch('data_ingest.app.get_config', return_value=test_config):
        return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_message = AsyncMock(return_value=True)
    producer.send_batch = AsyncMock(return_value=3)
    producer.get_stats = MagicMock(return_value={
        'messages_sent': 100,
        'messages_failed': 2,
        'success_rate': 98.0
    })
    return producer


@pytest.fixture
def sample_tick_data() -> Dict[str, Any]:
    """Sample tick data for testing."""
    return {
        "timestamp": time.time(),
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "price": 150.25,
        "size": 100.0,
        "side": "B",
        "trade_id": "test_trade_123",
        "sequence_number": 1001
    }


@pytest.fixture
def sample_quote_data() -> Dict[str, Any]:
    """Sample quote data for testing."""
    return {
        "timestamp": time.time(),
        "symbol": "AAPL", 
        "exchange": "NASDAQ",
        "bid_price": 150.20,
        "bid_size": 500.0,
        "ask_price": 150.30,
        "ask_size": 300.0
    }


@pytest.fixture
def sample_candle_data() -> Dict[str, Any]:
    """Sample candle data for testing."""
    return {
        "timestamp": time.time(),
        "symbol": "AAPL",
        "exchange": "NASDAQ", 
        "timeframe": "1m",
        "open_price": 150.00,
        "high_price": 150.50,
        "low_price": 149.75,
        "close_price": 150.25,
        "volume": 10000.0,
        "trade_count": 45,
        "vwap": 150.15
    }


@pytest.fixture
def sample_orderbook_data() -> Dict[str, Any]:
    """Sample order book data for testing."""
    return {
        "timestamp": time.time(),
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "bids": [
            {"price": 150.20, "size": 100.0},
            {"price": 150.19, "size": 200.0},
            {"price": 150.18, "size": 150.0}
        ],
        "asks": [
            {"price": 150.21, "size": 75.0},
            {"price": 150.22, "size": 125.0}, 
            {"price": 150.23, "size": 90.0}
        ],
        "sequence_number": 5001
    }


@pytest.fixture
def market_data_samples() -> Dict[str, List[Dict[str, Any]]]:
    """Collection of market data samples for batch testing."""
    ticks = []
    quotes = []
    candles = []
    
    base_time = time.time()
    
    # Generate 10 samples of each type
    for i in range(10):
        timestamp = base_time + i
        price = 150.0 + (i * 0.01)
        
        ticks.append({
            "timestamp": timestamp,
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "price": price,
            "size": 100.0 + i,
            "side": "B" if i % 2 == 0 else "S",
            "trade_id": f"trade_{i}"
        })
        
        quotes.append({
            "timestamp": timestamp,
            "symbol": "AAPL",
            "exchange": "NASDAQ", 
            "bid_price": price - 0.01,
            "bid_size": 500.0,
            "ask_price": price + 0.01,
            "ask_size": 300.0
        })
        
        candles.append({
            "timestamp": timestamp,
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "timeframe": "1m", 
            "open_price": price,
            "high_price": price + 0.05,
            "low_price": price - 0.03,
            "close_price": price + 0.02,
            "volume": 1000.0 + (i * 100)
        })
    
    return {
        "ticks": ticks,
        "quotes": quotes, 
        "candles": candles
    }


@pytest.fixture
def historical_data_file():
    """Create temporary CSV file with historical data."""
    data = {
        'timestamp': [time.time() - (i * 60) for i in range(100, 0, -1)],
        'symbol': ['AAPL'] * 100,
        'open': [150.0 + (i * 0.01) for i in range(100)],
        'high': [150.0 + (i * 0.01) + 0.05 for i in range(100)],
        'low': [150.0 + (i * 0.01) - 0.03 for i in range(100)],
        'close': [150.0 + (i * 0.01) + 0.02 for i in range(100)],
        'volume': [1000 + (i * 10) for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


class MockDataFeed(BaseDataFeed):
    """Mock data feed for testing."""
    
    def __init__(self, name: str = "test_feed", **kwargs):
        config = FeedConfig(**kwargs)
        super().__init__(name, Exchange.MOCK, config)
        self.connected = False
        self.subscribed_symbols: Dict[str, List[DataType]] = {}
        
        # Mock statistics
        self.mock_stats = {
            "ticks_received": 0,
            "quotes_received": 0,
            "candles_received": 0,
            "connected": False,
            "error_count": 0
        }
    
    async def connect(self) -> None:
        """Mock connection."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        self.mock_stats["connected"] = True
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False
        self.mock_stats["connected"] = False
        self.subscribed_symbols.clear()
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Mock subscription."""
        if not self.connected:
            return False
        
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols[symbol] = []
            self.subscribed_symbols[symbol].extend(data_types)
        
        return True
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Mock unsubscription."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                for dt in data_types:
                    if dt in self.subscribed_symbols[symbol]:
                        self.subscribed_symbols[symbol].remove(dt)
                
                if not self.subscribed_symbols[symbol]:
                    del self.subscribed_symbols[symbol]
        
        return True
    
    async def _process_message(self, message: Dict[str, Any]) -> Optional[Any]:
        """Mock message processing."""
        return message
    
    async def send_test_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Send test data through the feed."""
        self.mock_stats[f"{data_type}_received"] += 1
        await self._handle_message(data)
    
    def get_mock_stats(self) -> Dict[str, Any]:
        """Get mock statistics."""
        return self.mock_stats.copy()


@pytest_asyncio.fixture
async def mock_feed():
    """Create and start mock data feed."""
    feed = MockDataFeed()
    await feed.start()
    yield feed
    await feed.stop()


@pytest.fixture
def websocket_mock():
    """Mock WebSocket connection."""
    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_text = AsyncMock()
    return mock_ws


@pytest.fixture
def database_mock():
    """Mock database operations."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetch_one = AsyncMock(return_value=None)
    return mock_db


class TestDataValidator:
    """Utility class for validating test data."""
    
    @staticmethod
    def validate_tick(tick: Dict[str, Any]) -> bool:
        """Validate tick data structure."""
        required_fields = ["timestamp", "symbol", "exchange", "price", "size"]
        return all(field in tick for field in required_fields)
    
    @staticmethod
    def validate_quote(quote: Dict[str, Any]) -> bool:
        """Validate quote data structure."""
        required_fields = ["timestamp", "symbol", "exchange", "bid_price", "ask_price"]
        return all(field in quote for field in required_fields)
    
    @staticmethod
    def validate_candle(candle: Dict[str, Any]) -> bool:
        """Validate candle data structure."""
        required_fields = [
            "timestamp", "symbol", "exchange", "timeframe",
            "open_price", "high_price", "low_price", "close_price", "volume"
        ]
        return all(field in candle for field in required_fields)
    
    @staticmethod
    def validate_ohlc_constraints(candle: Dict[str, Any]) -> bool:
        """Validate OHLC price constraints."""
        o, h, l, c = candle["open_price"], candle["high_price"], candle["low_price"], candle["close_price"]
        return (
            h >= max(o, c) and
            l <= min(o, c) and
            h >= l
        )


@pytest.fixture
def data_validator():
    """Data validation utility."""
    return TestDataValidator()


@pytest_asyncio.fixture
async def feed_with_data(mock_feed, sample_tick_data, sample_quote_data):
    """Feed with pre-loaded test data."""
    
    # Subscribe to data types
    await mock_feed.subscribe(["AAPL"], [DataType.TICKS, DataType.QUOTES])
    
    # Send test data
    await mock_feed.send_test_data("ticks", sample_tick_data)
    await mock_feed.send_test_data("quotes", sample_quote_data)
    
    yield mock_feed


class PerformanceTester:
    """Performance testing utility."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.measurements = []
    
    def start(self):
        """Start timing measurement."""
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing measurement."""
        self.end_time = time.perf_counter()
        return self.duration
    
    @property
    def duration(self) -> float:
        """Get measurement duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def measure(self, operation_name: str):
        """Context manager for measuring operations."""
        class MeasurementContext:
            def __init__(self, tester, name):
                self.tester = tester
                self.name = name
                self.start = None
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.perf_counter() - self.start
                self.tester.measurements.append({
                    'operation': self.name,
                    'duration': duration,
                    'timestamp': time.time()
                })
        
        return MeasurementContext(self, operation_name)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.measurements:
            return {}
        
        durations = [m['duration'] for m in self.measurements]
        return {
            'count': len(durations),
            'total': sum(durations),
            'average': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations)
        }


@pytest.fixture
def performance_tester():
    """Performance testing utility."""
    return PerformanceTester()


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=["AAPL", "GOOGL", "MSFT", "BTCUSD", "ETHUSD"])
def test_symbol(request):
    """Parametrized test symbols."""
    return request.param


@pytest.fixture(params=[DataType.TICKS, DataType.QUOTES, DataType.CANDLES, DataType.ORDER_BOOK])
def data_type(request):
    """Parametrized data types."""
    return request.param


@pytest.fixture(params=[Exchange.POLYGON, Exchange.BINANCE, Exchange.MOCK])
def exchange(request):
    """Parametrized exchanges."""
    return request.param


# Async test decorators
def async_test(func):
    """Decorator for async test functions."""
    return pytest.mark.asyncio(func)


def integration_test(func):
    """Decorator for integration tests."""
    return pytest.mark.integration(func)


def slow_test(func):
    """Decorator for slow tests."""
    return pytest.mark.slow(func)


def requires_external(service: str):
    """Decorator for tests requiring external services."""
    def decorator(func):
        return pytest.mark.skipif(
            not os.getenv(f"TEST_{service.upper()}", False),
            reason=f"Set TEST_{service.upper()}=1 to run {service} tests"
        )(func)
    return decorator


# Test data generators
def generate_tick_sequence(symbol: str, count: int, start_price: float = 100.0) -> List[Dict[str, Any]]:
    """Generate sequence of tick data."""
    ticks = []
    price = start_price
    
    for i in range(count):
        # Random walk price
        price += (0.5 - hash(f"{symbol}_{i}") % 100 / 100) * 0.01
        
        ticks.append({
            "timestamp": time.time() + i,
            "symbol": symbol,
            "exchange": "MOCK",
            "price": round(price, 2),
            "size": 100 + (i % 50),
            "side": "B" if i % 2 == 0 else "S",
            "trade_id": f"trade_{i}"
        })
    
    return ticks


def generate_market_session(symbols: List[str], duration_minutes: int = 60) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a full market session of data."""
    session_data = {}
    
    for symbol in symbols:
        ticks = generate_tick_sequence(symbol, duration_minutes * 10)  # 10 ticks per minute
        session_data[symbol] = ticks
    
    return session_data


# Cleanup utilities
@pytest.fixture(scope="function", autouse=True)
def cleanup_logs():
    """Cleanup log handlers after each test."""
    yield
    # Clear any test-specific log handlers
    import logging
    logging.getLogger().handlers.clear()


@pytest.fixture(scope="session", autouse=True)  
def setup_test_logging():
    """Setup logging for test session."""
    setup_logging("test", "DEBUG", enable_trace=False)
    yield


# Mock external services
@pytest.fixture
def mock_polygon_api():
    """Mock Polygon.io API responses."""
    responses = {
        "/v2/aggs/ticker/AAPL/range/1/minute/2023-01-01/2023-01-02": {
            "ticker": "AAPL",
            "status": "OK", 
            "results": [
                {
                    "t": 1672531200000,
                    "o": 150.0,
                    "h": 150.5,
                    "l": 149.8,
                    "c": 150.2,
                    "v": 1000
                }
            ]
        }
    }
    
    def mock_get(url):
        mock_response = MagicMock()
        
        for path, response_data in responses.items():
            if path in url:
                mock_response.json.return_value = response_data
                mock_response.status_code = 200
                break
        else:
            mock_response.status_code = 404
        
        return mock_response
    
    with patch('aiohttp.ClientSession.get', return_value=mock_get):
        yield responses


# Error injection utilities  
class ErrorInjector:
    """Utility for injecting errors during tests."""
    
    def __init__(self):
        self.error_patterns = {}
        self.call_count = {}
    
    def inject_error_after(self, method_name: str, calls: int, error: Exception):
        """Inject error after N successful calls."""
        self.error_patterns[method_name] = (calls, error)
        self.call_count[method_name] = 0
    
    def should_raise_error(self, method_name: str) -> Optional[Exception]:
        """Check if error should be raised."""
        if method_name not in self.error_patterns:
            return None
        
        self.call_count[method_name] += 1
        calls_needed, error = self.error_patterns[method_name]
        
        if self.call_count[method_name] > calls_needed:
            return error
        
        return None


@pytest.fixture
def error_injector():
    """Error injection utility for fault tolerance testing."""
    return ErrorInjector()