"""
Comprehensive tests for data feed implementations.
Tests connection handling, data processing, error recovery, and performance.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

import pytest
import websockets
from aioresponses import aioresponses

from data_ingest.feeds.base import BaseDataFeed, DataType, FeedStatus, ConnectionError
from data_ingest.feeds.polygon import PolygonFeed, PolygonConfig, create_polygon_feed
from data_ingest.feeds.ccxt_wrapper import CCXTFeed, CCXTConfig, create_binance_feed
from data_ingest.feeds.mock_exchange import MockExchangeFeed, create_mock_feed
from data_ingest.models.schemas import Tick, Quote, Candle, OrderBook, Exchange, Side


class TestBaseDataFeed:
    """Test base data feed functionality."""
    
    @pytest.mark.asyncio
    async def test_feed_lifecycle(self, mock_feed):
        """Test feed connection lifecycle."""
        # Initial state
        assert mock_feed.status == FeedStatus.CONNECTED
        assert mock_feed.is_connected()
        
        # Test disconnection
        await mock_feed.stop()
        assert mock_feed.status == FeedStatus.STOPPED
        assert not mock_feed.is_connected()
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, mock_feed):
        """Test symbol subscription and unsubscription."""
        symbols = ["AAPL", "GOOGL"]
        data_types = [DataType.TICKS, DataType.QUOTES]
        
        # Test subscription
        success = await mock_feed.subscribe(symbols, data_types)
        assert success
        
        subscriptions = mock_feed.get_subscriptions()
        for symbol in symbols:
            assert symbol in subscriptions
            assert set(data_types).issubset(subscriptions[symbol])
        
        # Test partial unsubscription
        await mock_feed.unsubscribe(["AAPL"], [DataType.TICKS])
        subscriptions = mock_feed.get_subscriptions()
        assert DataType.QUOTES in subscriptions["AAPL"]
        assert DataType.TICKS not in subscriptions["AAPL"]
        
        # Test full unsubscription
        await mock_feed.unsubscribe(["AAPL"], [DataType.QUOTES])
        subscriptions = mock_feed.get_subscriptions()
        assert "AAPL" not in subscriptions
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, mock_feed):
        """Test message handler registration and dispatch."""
        handler_calls = []
        
        def test_handler(data):
            handler_calls.append(data)
        
        # Register handler
        mock_feed.register_handler(DataType.TICKS, test_handler)
        
        # Send test data
        test_data = {"symbol": "AAPL", "price": 150.0}
        await mock_feed.send_test_data("ticks", test_data)
        
        # Verify handler was called
        assert len(handler_calls) == 1
        assert handler_calls[0] == test_data
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, mock_feed, market_data_samples):
        """Test feed statistics tracking."""
        # Send various data types
        for tick in market_data_samples["ticks"][:3]:
            await mock_feed.send_test_data("ticks", tick)
        
        for quote in market_data_samples["quotes"][:2]:
            await mock_feed.send_test_data("quotes", quote)
        
        # Check statistics
        stats = mock_feed.get_mock_stats()
        assert stats["ticks_received"] == 3
        assert stats["quotes_received"] == 2
        assert stats["connected"] == True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_feed, error_injector):
        """Test error handling and recovery."""
        # Inject error after 2 successful calls
        error_injector.inject_error_after("send_test_data", 2, ConnectionError("Test error"))
        
        # This should work
        await mock_feed.send_test_data("ticks", {"symbol": "AAPL"})
        await mock_feed.send_test_data("ticks", {"symbol": "AAPL"})
        
        # This should handle error gracefully
        try:
            await mock_feed.send_test_data("ticks", {"symbol": "AAPL"})
        except ConnectionError:
            pass  # Expected
        
        # Feed should still be connected after error
        assert mock_feed.is_connected()


class TestPolygonFeed:
    """Test Polygon.io feed implementation."""
    
    @pytest.fixture
    def polygon_config(self):
        return PolygonConfig(
            api_key="test_api_key",
            rate_limit=100,
            heartbeat_interval=30
        )
    
    @pytest.fixture
    def polygon_feed(self, polygon_config):
        return PolygonFeed(polygon_config)
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_websocket_connection(self, mock_connect, polygon_feed):
        """Test WebSocket connection establishment."""
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        # Mock authentication response
        auth_response = json.dumps([{"status": "auth_success", "message": "authenticated"}])
        mock_ws.recv.return_value = auth_response
        
        # Test connection
        await polygon_feed.connect()
        
        # Verify connection was established
        assert polygon_feed.status == FeedStatus.CONNECTED
        assert polygon_feed.authenticated == True
        
        # Verify authentication message was sent
        mock_ws.send.assert_called()
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["action"] == "auth"
        assert sent_message["params"] == "test_api_key"
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_subscription(self, mock_connect, polygon_feed):
        """Test symbol subscription."""
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        # Mock authentication
        auth_response = json.dumps([{"status": "auth_success"}])
        mock_ws.recv.return_value = auth_response
        
        await polygon_feed.connect()
        
        # Test subscription
        success = await polygon_feed.subscribe(["AAPL"], [DataType.TICKS, DataType.QUOTES])
        assert success
        
        # Verify subscription message was sent
        calls = mock_ws.send.call_args_list
        subscription_call = calls[-1]  # Last call should be subscription
        sent_message = json.loads(subscription_call[0][0])
        assert sent_message["action"] == "subscribe"
        assert "T.*" in sent_message["params"]  # Trades subscription
        assert "Q.*" in sent_message["params"]  # Quotes subscription
    
    def test_trade_message_parsing(self, polygon_feed):
        """Test parsing of Polygon trade messages."""
        trade_message = {
            "ev": "T",
            "sym": "AAPL",
            "p": 150.25,
            "s": 100,
            "t": 1609459200000,  # Timestamp in milliseconds
            "c": ["@"],
            "i": "12345",
            "q": 1001
        }
        
        tick = polygon_feed._parse_trade_message(trade_message)
        
        assert tick.symbol == "AAPL"
        assert tick.price == 150.25
        assert tick.size == 100.0
        assert tick.trade_id == "12345"
        assert tick.sequence_number == 1001
        assert tick.exchange == Exchange.POLYGON
    
    def test_quote_message_parsing(self, polygon_feed):
        """Test parsing of Polygon quote messages."""
        quote_message = {
            "ev": "Q",
            "sym": "AAPL",
            "bp": 150.20,
            "bs": 500,
            "ap": 150.30,
            "as": 300,
            "t": 1609459200000
        }
        
        quote = polygon_feed._parse_quote_message(quote_message)
        
        assert quote.symbol == "AAPL"
        assert quote.bid_price == 150.20
        assert quote.bid_size == 500.0
        assert quote.ask_price == 150.30
        assert quote.ask_size == 300.0
        assert quote.exchange == Exchange.POLYGON
    
    def test_candle_message_parsing(self, polygon_feed):
        """Test parsing of Polygon aggregate messages."""
        agg_message = {
            "ev": "AM",
            "sym": "AAPL",
            "o": 150.00,
            "h": 150.50,
            "l": 149.75,
            "c": 150.25,
            "v": 10000,
            "s": 1609459200000,  # Start timestamp
            "n": 45,
            "vw": 150.15
        }
        
        candle = polygon_feed._parse_candle_message(agg_message)
        
        assert candle.symbol == "AAPL"
        assert candle.open_price == 150.00
        assert candle.high_price == 150.50
        assert candle.low_price == 149.75
        assert candle.close_price == 150.25
        assert candle.volume == 10000.0
        assert candle.trade_count == 45
        assert candle.vwap == 150.15
        assert candle.timeframe == "1m"
    
    @pytest.mark.asyncio
    @aioresponses()
    async def test_historical_data_request(self, mock_aiohttp, polygon_feed):
        """Test historical data REST API request."""
        # Mock HTTP session
        polygon_feed.http_session = AsyncMock()
        
        # Mock API response
        api_response = {
            "ticker": "AAPL",
            "status": "OK",
            "resultsCount": 2,
            "results": [
                {
                    "t": 1609459200000,
                    "o": 150.0,
                    "h": 150.5,
                    "l": 149.8,
                    "c": 150.2,
                    "v": 1000
                },
                {
                    "t": 1609459260000,
                    "o": 150.2,
                    "h": 150.6,
                    "l": 149.9,
                    "c": 150.3,
                    "v": 1200
                }
            ]
        }
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status.return_value = None
        
        polygon_feed.http_session.get.return_value.__aenter__.return_value = mock_response
        
        # Test historical data request
        data = await polygon_feed.get_historical_data(
            "AAPL",
            "candles",
            "2021-01-01",
            "2021-01-02"
        )
        
        assert len(data) == 2
        assert data[0]["o"] == 150.0
        assert data[1]["c"] == 150.3
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, polygon_feed):
        """Test handling of connection failures."""
        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            with pytest.raises(ConnectionError):
                await polygon_feed.connect()
            
            assert polygon_feed.status == FeedStatus.ERROR
            assert not polygon_feed.authenticated


class TestCCXTFeed:
    """Test CCXT cryptocurrency feed implementation."""
    
    @pytest.fixture
    def ccxt_config(self):
        return CCXTConfig(
            exchange_id="binance",
            sandbox=True,
            api_key="test_key",
            secret="test_secret"
        )
    
    @pytest.fixture
    def ccxt_feed(self, ccxt_config):
        return CCXTFeed(ccxt_config)
    
    @pytest.mark.asyncio
    @patch('ccxt.pro.binance')
    async def test_exchange_initialization(self, mock_binance_class, ccxt_feed):
        """Test CCXT exchange initialization."""
        # Mock exchange instance
        mock_exchange = AsyncMock()
        mock_exchange.load_markets.return_value = {
            "BTC/USDT": {"id": "BTCUSDT", "symbol": "BTC/USDT"},
            "ETH/USDT": {"id": "ETHUSDT", "symbol": "ETH/USDT"}
        }
        mock_binance_class.return_value = mock_exchange
        
        # Test connection
        await ccxt_feed.connect()
        
        # Verify exchange was initialized correctly
        assert ccxt_feed.exchange_instance is not None
        assert ccxt_feed.status == FeedStatus.CONNECTED
        assert len(ccxt_feed.markets) == 2
        
        # Verify markets were loaded
        mock_exchange.load_markets.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('ccxt.pro.binance')
    async def test_symbol_normalization(self, mock_binance_class, ccxt_feed):
        """Test symbol format normalization."""
        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets.return_value = {
            "BTC/USDT": {"id": "BTCUSDT", "symbol": "BTC/USDT"}
        }
        mock_binance_class.return_value = mock_exchange
        
        await ccxt_feed.connect()
        
        # Test symbol normalization
        assert ccxt_feed._normalize_symbol("BTC-USDT") == "BTC/USDT"
        assert ccxt_feed._normalize_symbol("BTC_USDT") == "BTC/USDT"
        assert ccxt_feed._normalize_symbol("BTC/USDT") == "BTC/USDT"
    
    @pytest.mark.asyncio
    @patch('ccxt.pro.binance')
    async def test_trade_subscription(self, mock_binance_class, ccxt_feed):
        """Test trade data subscription."""
        # Mock exchange with watch_trades
        mock_exchange = AsyncMock()
        mock_exchange.load_markets.return_value = {"BTC/USDT": {"symbol": "BTC/USDT"}}
        
        # Mock trade data
        mock_trades = [
            {
                "id": "123",
                "symbol": "BTC/USDT",
                "amount": 0.1,
                "price": 50000.0,
                "side": "buy",
                "timestamp": 1609459200000
            }
        ]
        mock_exchange.watch_trades.return_value = mock_trades
        mock_binance_class.return_value = mock_exchange
        
        await ccxt_feed.connect()
        
        # Test subscription
        success = await ccxt_feed.subscribe(["BTC-USDT"], [DataType.TICKS])
        assert success
        
        # Verify trade subscription was set up
        assert "BTC/USDT" in ccxt_feed.trades_subscriptions
    
    def test_trade_conversion(self, ccxt_feed):
        """Test conversion of CCXT trade to Tick."""
        ccxt_trade = {
            "id": "123456",
            "symbol": "BTC/USDT",
            "amount": 0.1,
            "price": 50000.0,
            "side": "buy",
            "timestamp": 1609459200000
        }
        
        tick = ccxt_feed._convert_trade_to_tick("BTC/USDT", ccxt_trade)
        
        assert tick.symbol == "BTC-USDT"  # Normalized format
        assert tick.price == 50000.0
        assert tick.size == 0.1
        assert tick.side == Side.BUY
        assert tick.trade_id == "123456"
        assert tick.exchange == ccxt_feed.exchange
    
    def test_ticker_conversion(self, ccxt_feed):
        """Test conversion of CCXT ticker to Quote."""
        ccxt_ticker = {
            "symbol": "BTC/USDT",
            "bid": 49950.0,
            "ask": 50050.0,
            "bidVolume": 1.5,
            "askVolume": 2.0,
            "timestamp": 1609459200000
        }
        
        quote = ccxt_feed._convert_ticker_to_quote("BTC/USDT", ccxt_ticker)
        
        assert quote.symbol == "BTC-USDT"
        assert quote.bid_price == 49950.0
        assert quote.ask_price == 50050.0
        assert quote.bid_size == 1.5
        assert quote.ask_size == 2.0
    
    def test_orderbook_conversion(self, ccxt_feed):
        """Test conversion of CCXT order book to OrderBook."""
        ccxt_orderbook = {
            "symbol": "BTC/USDT",
            "bids": [
                [49950.0, 1.5],
                [49940.0, 2.0],
                [49930.0, 1.0]
            ],
            "asks": [
                [50050.0, 2.0],
                [50060.0, 1.5],
                [50070.0, 1.0]
            ],
            "timestamp": 1609459200000,
            "nonce": 12345
        }
        
        orderbook = ccxt_feed._convert_orderbook("BTC/USDT", ccxt_orderbook)
        
        assert orderbook.symbol == "BTC-USDT"
        assert len(orderbook.bids) <= 10  # Limited to top 10 levels
        assert len(orderbook.asks) <= 10
        assert orderbook.bids[0].price == 49950.0
        assert orderbook.asks[0].price == 50050.0
        assert orderbook.sequence_number == 12345


class TestMockExchangeFeed:
    """Test mock exchange feed for development and testing."""
    
    @pytest.mark.asyncio
    async def test_mock_feed_creation(self):
        """Test mock feed creation and initialization."""
        feed = create_mock_feed(
            tick_interval=0.1,
            enable_trends=True
        )
        
        assert feed.name == "mock_exchange"
        assert feed.exchange == Exchange.MOCK
        assert feed.config["tick_interval"] == 0.1
        assert feed.config["enable_trends"] == True
    
    @pytest.mark.asyncio
    async def test_mock_data_generation(self):
        """Test realistic data generation."""
        feed = create_mock_feed(tick_interval=0.01)  # Fast generation for testing
        
        received_data = []
        
        def data_handler(data):
            received_data.append(data)
        
        # Register handlers
        feed.register_handler(DataType.TICKS, data_handler)
        feed.register_handler(DataType.QUOTES, data_handler)
        
        # Start feed and subscribe
        await feed.start()
        await feed.subscribe(["AAPL"], [DataType.TICKS, DataType.QUOTES])
        
        # Let it generate data for a short time
        await asyncio.sleep(0.1)
        
        await feed.stop()
        
        # Verify data was generated
        assert len(received_data) > 0
        
        # Check data structure
        tick_data = [d for d in received_data if 'side' in d]
        quote_data = [d for d in received_data if 'bid_price' in d]
        
        assert len(tick_data) > 0
        assert len(quote_data) > 0
    
    @pytest.mark.asyncio
    async def test_price_shock_injection(self):
        """Test price shock injection functionality."""
        feed = create_mock_feed()
        await feed.start()
        await feed.subscribe(["AAPL"], [DataType.TICKS])
        
        # Get initial price
        generator = feed.generators.get("AAPL")
        initial_price = generator.current_price
        
        # Inject 10% price shock
        feed.inject_price_shock("AAPL", 10.0)
        
        # Verify price changed
        new_price = generator.current_price
        price_change_pct = ((new_price - initial_price) / initial_price) * 100
        
        assert abs(price_change_pct - 10.0) < 0.1  # Allow small tolerance
        
        await feed.stop()
    
    @pytest.mark.asyncio
    async def test_historical_data_replay(self, historical_data_file):
        """Test historical data replay functionality."""
        feed = create_mock_feed(data_file=historical_data_file)
        
        # Should have loaded historical data
        assert "AAPL" in feed.historical_data
        assert len(feed.historical_data["AAPL"]) == 100
        
        await feed.start()
        await feed.stop()
    
    def test_data_validator_integration(self, data_validator, sample_tick_data, sample_quote_data, sample_candle_data):
        """Test data validation utilities."""
        # Test valid data
        assert data_validator.validate_tick(sample_tick_data)
        assert data_validator.validate_quote(sample_quote_data)
        assert data_validator.validate_candle(sample_candle_data)
        assert data_validator.validate_ohlc_constraints(sample_candle_data)
        
        # Test invalid data
        invalid_tick = sample_tick_data.copy()
        del invalid_tick["price"]
        assert not data_validator.validate_tick(invalid_tick)
        
        invalid_candle = sample_candle_data.copy()
        invalid_candle["high_price"] = 100.0  # Lower than other prices
        invalid_candle["low_price"] = 200.0   # Higher than other prices
        assert not data_validator.validate_ohlc_constraints(invalid_candle)


@pytest.mark.integration
class TestFeedIntegration:
    """Integration tests for feed interactions."""
    
    @pytest.mark.asyncio
    async def test_multiple_feed_coordination(self):
        """Test coordination between multiple feeds."""
        # Create multiple feeds
        mock_feed = create_mock_feed(tick_interval=0.01)
        
        received_messages = []
        
        def message_handler(data):
            received_messages.append((data.get("symbol"), type(data)))
        
        # Set up feeds
        await mock_feed.start()
        mock_feed.register_handler(DataType.TICKS, message_handler)
        
        # Subscribe to same symbols on different feeds
        await mock_feed.subscribe(["AAPL", "GOOGL"], [DataType.TICKS])
        
        # Let feeds generate data
        await asyncio.sleep(0.05)
        
        await mock_feed.stop()
        
        # Verify both symbols received data
        symbols = set(msg[0] for msg in received_messages)
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
    
    @pytest.mark.asyncio
    async def test_feed_failover_simulation(self):
        """Test feed failover and recovery."""
        feed = create_mock_feed()
        
        connection_states = []
        
        # Monitor connection state changes
        original_connect = feed.connect
        original_disconnect = feed.disconnect
        
        async def monitored_connect():
            connection_states.append("connecting")
            await original_connect()
            connection_states.append("connected")
        
        async def monitored_disconnect():
            connection_states.append("disconnecting")
            await original_disconnect()
            connection_states.append("disconnected")
        
        feed.connect = monitored_connect
        feed.disconnect = monitored_disconnect
        
        # Test lifecycle
        await feed.start()
        await feed.stop()
        
        # Verify state transitions
        assert "connecting" in connection_states
        assert "connected" in connection_states
        assert "disconnecting" in connection_states
        assert "disconnected" in connection_states


@pytest.mark.slow
class TestFeedPerformance:
    """Performance tests for feed implementations."""
    
    @pytest.mark.asyncio
    async def test_high_frequency_data_processing(self, performance_tester):
        """Test processing performance under high data rates."""
        feed = create_mock_feed(tick_interval=0.001)  # Very fast ticks
        
        message_count = 0
        
        def counting_handler(data):
            nonlocal message_count
            message_count += 1
        
        feed.register_handler(DataType.TICKS, counting_handler)
        
        with performance_tester.measure("high_frequency_processing"):
            await feed.start()
            await feed.subscribe(["TESTPERF"], [DataType.TICKS])
            
            # Run for 1 second
            await asyncio.sleep(1.0)
            
            await feed.stop()
        
        stats = performance_tester.get_stats()
        
        # Verify performance metrics
        assert message_count > 500  # Should process many messages per second
        assert stats['total'] < 2.0  # Should complete within 2 seconds including setup
        
        print(f"Processed {message_count} messages in {stats['total']:.3f} seconds")
        print(f"Rate: {message_count / stats['total']:.0f} messages/second")
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        feed = create_mock_feed(tick_interval=0.01)
        await feed.start()
        await feed.subscribe(["MEMTEST"], [DataType.TICKS, DataType.QUOTES])
        
        # Run for several seconds
        await asyncio.sleep(3.0)
        
        current_memory = process.memory_info().rss
        memory_growth = current_memory - initial_memory
        
        await feed.stop()
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.1f} MB"


@pytest.mark.requires_external("polygon")
class TestPolygonIntegration:
    """Integration tests requiring actual Polygon API."""
    
    @pytest.mark.asyncio
    async def test_real_polygon_connection(self):
        """Test connection to real Polygon API."""
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key or api_key == "demo":
            pytest.skip("Real Polygon API key required")
        
        feed = create_polygon_feed(api_key=api_key)
        
        try:
            await feed.start()
            assert feed.is_connected()
            
            # Test basic subscription
            success = await feed.subscribe(["AAPL"], [DataType.TICKS])
            assert success
            
            # Wait briefly for data
            await asyncio.sleep(2)
            
        finally:
            await feed.stop()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow and not integration"  # Skip slow and integration tests by default
    ])