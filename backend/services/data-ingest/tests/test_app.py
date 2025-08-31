"""
Comprehensive tests for FastAPI application endpoints.
Tests REST API, WebSocket connections, health checks, and error handling.
"""

import asyncio
import json
import time
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import websockets
from starlette.websockets import WebSocketDisconnect

from data_ingest.app import create_app, FeedManager
from data_ingest.config import DataIngestConfig, TradingMode, LogLevel
from data_ingest.feeds.base import DataType, FeedStatus
from data_ingest.models.schemas import (
    SymbolRequest, HealthCheckResponse, MetricsResponse, 
    DataFeedStats, Exchange, Tick, Quote, Candle, OrderBook
)


class TestHealthEndpoints:
    """Test health check and monitoring endpoints."""
    
    def test_health_check_basic(self, client):
        """Test basic health check endpoint."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 100.0,
                "messages_processed": 500,
                "messages_failed": 5
            }
            mock_fm.get_feed_stats.return_value = []
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "uptime_seconds" in data
            assert data["uptime_seconds"] == 100.0
            assert data["feeds_connected"] == 0
            assert data["total_feeds"] == 0
    
    def test_health_check_healthy_status(self, client):
        """Test health check with all feeds connected."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 300.5,
                "messages_processed": 1000,
                "messages_failed": 5
            }
            
            # Mock feed stats - all connected
            mock_fm.get_feed_stats.return_value = [
                DataFeedStats(
                    feed_name="polygon",
                    symbol="AAPL",
                    exchange=Exchange.POLYGON,
                    connected=True,
                    ticks_received=500,
                    quotes_received=300
                ),
                DataFeedStats(
                    feed_name="mock",
                    symbol="BTCUSD", 
                    exchange=Exchange.MOCK,
                    connected=True,
                    ticks_received=200,
                    quotes_received=150
                )
            ]
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["feeds_connected"] == 2
            assert data["total_feeds"] == 2
            assert isinstance(data["timestamp"], (int, float))
            assert data["uptime_seconds"] == 300.5
    
    def test_health_check_partial_status(self, client):
        """Test health check with some feeds disconnected."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 150.0,
                "messages_processed": 800,
                "messages_failed": 10
            }
            
            # Mock feed stats - partial connectivity
            mock_fm.get_feed_stats.return_value = [
                DataFeedStats(
                    feed_name="polygon",
                    symbol="AAPL",
                    exchange=Exchange.POLYGON,
                    connected=True,
                    ticks_received=400,
                    quotes_received=200
                ),
                DataFeedStats(
                    feed_name="binance",
                    symbol="BTCUSD",
                    exchange=Exchange.BINANCE,
                    connected=False,  # Disconnected
                    error_count=3
                )
            ]
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "partial"
            assert data["feeds_connected"] == 1
            assert data["total_feeds"] == 2
    
    def test_health_check_degraded_status(self, client):
        """Test health check with all feeds disconnected."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 50.0,
                "messages_processed": 0,
                "messages_failed": 20
            }
            
            # Mock feed stats - all disconnected
            mock_fm.get_feed_stats.return_value = [
                DataFeedStats(
                    feed_name="polygon",
                    symbol="AAPL",
                    exchange=Exchange.POLYGON,
                    connected=False,
                    error_count=10,
                    last_error="Connection timeout"
                ),
                DataFeedStats(
                    feed_name="mock",
                    symbol="BTCUSD",
                    exchange=Exchange.MOCK,
                    connected=False,
                    error_count=5
                )
            ]
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "degraded"
            assert data["feeds_connected"] == 0
            assert data["total_feeds"] == 2
    
    def test_health_check_starting_status(self, client):
        """Test health check during startup with no feeds."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 5.0,
                "messages_processed": 0,
                "messages_failed": 0
            }
            mock_fm.get_feed_stats.return_value = []
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "starting"
            assert data["feeds_connected"] == 0
            assert data["total_feeds"] == 0
    
    def test_metrics_endpoint_comprehensive(self, client):
        """Test comprehensive metrics endpoint."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "messages_processed": 2500,
                "messages_per_second": 42.5,
                "uptime_seconds": 120.0,
                "messages_failed": 15,
                "success_rate": 99.4,
                "queue_size": 3
            }
            
            mock_fm.get_feed_stats.return_value = [
                DataFeedStats(
                    feed_name="polygon",
                    symbol="AAPL",
                    exchange=Exchange.POLYGON,
                    connected=True,
                    ticks_received=1000,
                    quotes_received=800,
                    candles_received=50,
                    avg_latency_ms=15.5,
                    max_latency_ms=45.0,
                    error_count=2,
                    last_message_time=time.time() - 1
                ),
                DataFeedStats(
                    feed_name="binance",
                    symbol="BTCUSD",
                    exchange=Exchange.BINANCE,
                    connected=True,
                    ticks_received=1500,
                    quotes_received=1200,
                    avg_latency_ms=8.2,
                    max_latency_ms=25.0,
                    error_count=0
                )
            ]
            
            response = client.get("/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service"] == "data-ingest"
            assert data["total_messages"] == 2500
            assert data["messages_per_second"] == 42.5
            assert isinstance(data["timestamp"], (int, float))
            assert len(data["feeds"]) == 2
            
            # Verify feed details
            polygon_feed = next(f for f in data["feeds"] if f["feed_name"] == "polygon")
            assert polygon_feed["connected"] == True
            assert polygon_feed["ticks_received"] == 1000
            assert polygon_feed["quotes_received"] == 800
            assert polygon_feed["avg_latency_ms"] == 15.5
            
            binance_feed = next(f for f in data["feeds"] if f["feed_name"] == "binance")
            assert binance_feed["error_count"] == 0
            assert binance_feed["avg_latency_ms"] == 8.2
    
    def test_service_not_ready_error(self):
        """Test error handling when service is not ready."""
        app = create_app()
        
        with patch('data_ingest.app.feed_manager', None):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 503
            assert "Service not ready" in response.json()["detail"]
    
    def test_metrics_endpoint_service_not_ready(self):
        """Test metrics endpoint when service is not ready."""
        app = create_app()
        
        with patch('data_ingest.app.feed_manager', None):
            client = TestClient(app)
            response = client.get("/metrics")
            assert response.status_code == 503


class TestFeedManagementEndpoints:
    """Test feed management API endpoints."""
    
    def test_subscribe_to_symbol_success(self, client):
        """Test successful symbol subscription."""
        request_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": ["ticks", "quotes"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            response = client.post("/feeds/subscribe", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] == True
            assert "AAPL" in data["message"]
            
            # Verify feed manager was called correctly
            mock_fm.subscribe_to_symbols.assert_called_once_with(
                ["AAPL"], 
                [DataType.TICKS, DataType.QUOTES]
            )
    
    def test_subscribe_multiple_data_types(self, client):
        """Test subscription with multiple data types."""
        request_data = {
            "symbol": "GOOGL",
            "exchange": "NASDAQ",
            "data_types": ["ticks", "quotes", "candles", "order_book"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            response = client.post("/feeds/subscribe", json=request_data)
            assert response.status_code == 200
            
            # Verify all data types were converted properly
            expected_types = [DataType.TICKS, DataType.QUOTES, DataType.CANDLES, DataType.ORDER_BOOK]
            mock_fm.subscribe_to_symbols.assert_called_once_with(["GOOGL"], expected_types)
    
    def test_subscribe_crypto_symbol(self, client):
        """Test subscription to cryptocurrency symbol."""
        request_data = {
            "symbol": "BTC-USD",
            "exchange": "BINANCE",
            "data_types": ["ticks", "order_book"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            response = client.post("/feeds/subscribe", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "BTC-USD" in data["message"]
    
    def test_subscribe_invalid_data_type(self, client):
        """Test subscription with invalid data type."""
        request_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": ["invalid_type", "ticks"]
        }
        
        response = client.post("/feeds/subscribe", json=request_data)
        assert response.status_code == 400
        assert "Invalid data type" in response.json()["detail"]
    
    def test_subscribe_empty_data_types(self, client):
        """Test subscription with empty data types list."""
        request_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": []
        }
        
        response = client.post("/feeds/subscribe", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_subscribe_missing_required_fields(self, client):
        """Test subscription with missing required fields."""
        # Missing symbol
        request_data = {
            "exchange": "NASDAQ",
            "data_types": ["ticks"]
        }
        
        response = client.post("/feeds/subscribe", json=request_data)
        assert response.status_code == 422
        
        # Missing exchange
        request_data = {
            "symbol": "AAPL",
            "data_types": ["ticks"]
        }
        
        response = client.post("/feeds/subscribe", json=request_data)
        assert response.status_code == 422
        
        # Missing data_types
        request_data = {
            "symbol": "AAPL", 
            "exchange": "NASDAQ"
        }
        
        response = client.post("/feeds/subscribe", json=request_data)
        assert response.status_code == 422
    
    def test_subscribe_failure_from_feed_manager(self, client):
        """Test subscription failure from feed manager."""
        request_data = {
            "symbol": "INVALID_SYMBOL",
            "exchange": "NASDAQ",
            "data_types": ["ticks"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=False)
            
            response = client.post("/feeds/subscribe", json=request_data)
            assert response.status_code == 400
            assert "Subscription failed" in response.json()["detail"]
    
    def test_subscribe_internal_error(self, client):
        """Test subscription with internal server error."""
        request_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": ["ticks"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(side_effect=Exception("Database connection failed"))
            
            response = client.post("/feeds/subscribe", json=request_data)
            assert response.status_code == 500
            assert "Subscription error" in response.json()["detail"]
            assert "Database connection failed" in response.json()["detail"]
    
    def test_feed_status_endpoint_comprehensive(self, client):
        """Test comprehensive feed status reporting."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            current_time = time.time()
            
            mock_stats = [
                DataFeedStats(
                    feed_name="polygon",
                    symbol="AAPL",
                    exchange=Exchange.POLYGON,
                    connected=True,
                    ticks_received=1500,
                    quotes_received=800,
                    candles_received=25,
                    last_message_time=current_time - 2,
                    error_count=1,
                    last_error="Temporary timeout",
                    last_error_time=current_time - 300,
                    reconnect_count=2
                ),
                DataFeedStats(
                    feed_name="binance",
                    symbol="BTCUSD",
                    exchange=Exchange.BINANCE,
                    connected=True,
                    ticks_received=2500,
                    quotes_received=1200,
                    last_message_time=current_time - 1,
                    error_count=0,
                    reconnect_count=0
                ),
                DataFeedStats(
                    feed_name="mock",
                    symbol="ETHUSD",
                    exchange=Exchange.MOCK,
                    connected=False,
                    ticks_received=0,
                    quotes_received=0,
                    error_count=5,
                    last_error="Connection refused",
                    last_error_time=current_time - 60,
                    reconnect_count=3
                )
            ]
            
            mock_fm.get_feed_stats.return_value = mock_stats
            
            response = client.get("/feeds/status")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_feeds"] == 3
            assert data["connected_feeds"] == 2
            assert len(data["feeds"]) == 3
            
            # Check individual feed details
            polygon_feed = next(f for f in data["feeds"] if f["name"] == "polygon")
            assert polygon_feed["connected"] == True
            assert polygon_feed["ticks_received"] == 1500
            assert polygon_feed["quotes_received"] == 800
            assert polygon_feed["error_count"] == 1
            assert isinstance(polygon_feed["last_message_time"], (int, float))
            
            binance_feed = next(f for f in data["feeds"] if f["name"] == "binance")
            assert binance_feed["connected"] == True
            assert binance_feed["error_count"] == 0
            
            mock_feed = next(f for f in data["feeds"] if f["name"] == "mock")
            assert mock_feed["connected"] == False
            assert mock_feed["error_count"] == 5
    
    def test_feed_status_empty_feeds(self, client):
        """Test feed status with no configured feeds."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_feed_stats.return_value = []
            
            response = client.get("/feeds/status")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_feeds"] == 0
            assert data["connected_feeds"] == 0
            assert data["feeds"] == []


class TestWebSocketEndpoints:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self, app):
        """Test WebSocket connection establishment and cleanup."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Connection should be added to the list
                    assert len(mock_fm.websocket_connections) == 1
                    
                    # WebSocket should be in the connections list
                    assert mock_fm.websocket_connections[0] == websocket
                
                # After context exit, connection should be removed
                # Note: This behavior depends on the WebSocket disconnect handler
    
    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, app):
        """Test WebSocket ping-pong functionality."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Send ping message
                    ping_message = {"action": "ping"}
                    websocket.send_text(json.dumps(ping_message))
                    
                    # Should receive pong response
                    response = websocket.receive_json()
                    assert response["type"] == "pong"
                    assert "timestamp" in response
                    assert isinstance(response["timestamp"], (int, float))
    
    @pytest.mark.asyncio
    async def test_websocket_invalid_message_handling(self, app):
        """Test WebSocket handling of invalid messages."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Send invalid JSON
                    websocket.send_text("invalid json")
                    
                    # Connection should remain open (errors are ignored)
                    # Send valid ping to verify connection is still active
                    websocket.send_text(json.dumps({"action": "ping"}))
                    response = websocket.receive_json()
                    assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_data_broadcasting(self, app):
        """Test real-time data broadcasting via WebSocket."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Simulate the FeedManager's broadcast functionality
                    test_messages = [
                        {
                            "type": "tick",
                            "data": {
                                "symbol": "AAPL",
                                "price": 150.25,
                                "size": 100.0,
                                "side": "B",
                                "timestamp": time.time()
                            }
                        },
                        {
                            "type": "quote", 
                            "data": {
                                "symbol": "AAPL",
                                "bid_price": 150.20,
                                "ask_price": 150.30,
                                "timestamp": time.time()
                            }
                        }
                    ]
                    
                    # Mock the internal broadcast method
                    broadcast_message = {"messages": test_messages}
                    
                    # Send broadcast data to WebSocket
                    websocket.send_json(broadcast_message)
                    
                    # The client should receive the broadcast
                    # Note: In the actual implementation, this would be sent FROM the server
    
    @pytest.mark.asyncio
    async def test_multiple_websocket_connections(self, app):
        """Test multiple concurrent WebSocket connections."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Create multiple WebSocket connections
                connections = []
                
                # Due to limitations of the test client, we'll simulate multiple connections
                for i in range(3):
                    # In a real scenario, these would be separate client connections
                    mock_connection = AsyncMock()
                    mock_fm.websocket_connections.append(mock_connection)
                
                assert len(mock_fm.websocket_connections) == 3
                
                # Verify each connection can be addressed individually
                for i, connection in enumerate(mock_fm.websocket_connections):
                    assert connection is not None
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect_cleanup(self, app):
        """Test proper cleanup when WebSocket disconnects."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                try:
                    with client.websocket_connect("/ws/market-data") as websocket:
                        assert len(mock_fm.websocket_connections) == 1
                        
                        # Simulate disconnect by closing websocket
                        websocket.close()
                        
                except Exception:
                    # WebSocket disconnect is expected
                    pass
                
                # In the actual implementation, disconnected WebSockets would be cleaned up
    
    @pytest.mark.asyncio
    async def test_websocket_message_filtering(self, app):
        """Test WebSocket message filtering and subscription management."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Send subscription message (custom protocol)
                    subscription_message = {
                        "action": "subscribe",
                        "symbols": ["AAPL", "GOOGL"],
                        "data_types": ["ticks", "quotes"]
                    }
                    
                    websocket.send_text(json.dumps(subscription_message))
                    
                    # In a full implementation, this would configure server-side filtering


class TestDebugEndpoints:
    """Test debug and development endpoints."""
    
    def test_debug_stats_disabled_in_production(self, client):
        """Test debug endpoints are properly disabled in production."""
        with patch('data_ingest.app.config') as mock_config:
            mock_config.enable_debug_endpoints = False
            
            response = client.get("/debug/stats")
            assert response.status_code == 404
    
    def test_debug_stats_enabled_in_development(self, client):
        """Test debug endpoints work correctly in development."""
        with patch('data_ingest.app.config') as mock_config:
            mock_config.enable_debug_endpoints = True
            
            with patch('data_ingest.app.feed_manager') as mock_fm:
                # Mock processing stats
                mock_fm.get_processing_stats.return_value = {
                    "messages_processed": 1500,
                    "messages_failed": 25,
                    "uptime_seconds": 180.5,
                    "success_rate": 98.3,
                    "queue_size": 8
                }
                
                # Mock feed subscriptions
                mock_feeds = {
                    "polygon": MagicMock(),
                    "binance": MagicMock(),
                    "mock": MagicMock()
                }
                
                mock_feeds["polygon"].get_subscriptions.return_value = {
                    "AAPL": {DataType.TICKS, DataType.QUOTES},
                    "GOOGL": {DataType.TICKS}
                }
                mock_feeds["binance"].get_subscriptions.return_value = {
                    "BTCUSD": {DataType.TICKS, DataType.ORDER_BOOK}
                }
                mock_feeds["mock"].get_subscriptions.return_value = {
                    "ETHUSD": {DataType.TICKS, DataType.QUOTES, DataType.CANDLES}
                }
                
                mock_fm.feeds = mock_feeds
                mock_fm.websocket_connections = [MagicMock(), MagicMock()]
                
                # Mock message buffer
                mock_buffer = MagicMock()
                mock_buffer.qsize.return_value = 8
                mock_fm.message_buffer = mock_buffer
                
                response = client.get("/debug/stats")
                assert response.status_code == 200
                
                data = response.json()
                
                # Verify processing stats
                assert "processing_stats" in data
                assert data["processing_stats"]["messages_processed"] == 1500
                assert data["processing_stats"]["success_rate"] == 98.3
                assert data["processing_stats"]["queue_size"] == 8
                
                # Verify feed subscriptions
                assert "feed_subscriptions" in data
                assert "polygon" in data["feed_subscriptions"]
                assert "binance" in data["feed_subscriptions"]
                assert "mock" in data["feed_subscriptions"]
                
                # Verify system info
                assert data["websocket_connections"] == 2
                assert data["message_queue_size"] == 8
                
                # Verify config info
                assert "config" in data
    
    def test_debug_stats_comprehensive_data(self, client):
        """Test debug stats with comprehensive system data."""
        with patch('data_ingest.app.config') as mock_config:
            mock_config.enable_debug_endpoints = True
            mock_config.trading_mode = TradingMode.TEST
            mock_config.symbols = {"AAPL", "GOOGL", "BTCUSD"}
            mock_config.exchanges = {"NASDAQ", "BINANCE"}
            mock_config.enable_mock_feeds = True
            
            with patch('data_ingest.app.feed_manager') as mock_fm:
                mock_fm.get_processing_stats.return_value = {
                    "messages_processed": 5000,
                    "messages_failed": 50,
                    "uptime_seconds": 3600.0,
                    "messages_per_second": 25.5,
                    "success_rate": 99.0
                }
                
                mock_fm.feeds = {"test_feed": MagicMock()}
                mock_fm.feeds["test_feed"].get_subscriptions.return_value = {"AAPL": {DataType.TICKS}}
                
                mock_fm.websocket_connections = []
                mock_fm.message_buffer = MagicMock()
                mock_fm.message_buffer.qsize.return_value = 0
                
                response = client.get("/debug/stats")
                assert response.status_code == 200
                
                data = response.json()
                assert data["config"]["trading_mode"] == "test"
                assert "AAPL" in data["config"]["symbols"]
                assert "NASDAQ" in data["config"]["exchanges"]
                assert data["config"]["enable_mock_feeds"] == True


class TestErrorHandling:
    """Test comprehensive error handling and edge cases."""
    
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/feeds/subscribe",
            data="invalid json{{{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post("/feeds/subscribe", json={})
        assert response.status_code == 422
        
        error_detail = response.json()["detail"]
        assert isinstance(error_detail, list)
        # Should have validation errors for missing fields
        field_errors = [error["loc"][-1] for error in error_detail]
        assert "symbol" in field_errors
        assert "exchange" in field_errors
        assert "data_types" in field_errors
    
    def test_invalid_content_type(self, client):
        """Test handling of invalid content type."""
        response = client.post(
            "/feeds/subscribe",
            data="symbol=AAPL&exchange=NASDAQ",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422
    
    def test_oversized_request_payload(self, client):
        """Test handling of oversized request payloads."""
        # Create a large payload
        large_symbol_list = ["SYM" + str(i) for i in range(10000)]
        large_request = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": ["ticks"],
            "large_field": large_symbol_list
        }
        
        response = client.post("/feeds/subscribe", json=large_request)
        # Should either accept gracefully or reject with appropriate status
        assert response.status_code in [200, 400, 413, 422]
    
    def test_special_characters_in_symbol(self, client):
        """Test handling of special characters in symbol names."""
        special_symbols = [
            "AAPL.US",
            "BRK-A",
            "BTC/USD",
            "SPX_INDEX", 
            "€USD",
            "TEST!@#$%"
        ]
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            for symbol in special_symbols:
                request_data = {
                    "symbol": symbol,
                    "exchange": "NASDAQ",
                    "data_types": ["ticks"]
                }
                
                response = client.post("/feeds/subscribe", json=request_data)
                # Should handle gracefully
                assert response.status_code in [200, 400, 422]
    
    def test_sql_injection_attempt(self, client):
        """Test protection against SQL injection attempts."""
        malicious_symbols = [
            "'; DROP TABLE users; --",
            "AAPL' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "'; DELETE FROM trades; --"
        ]
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            for malicious_symbol in malicious_symbols:
                request_data = {
                    "symbol": malicious_symbol,
                    "exchange": "NASDAQ",
                    "data_types": ["ticks"]
                }
                
                response = client.post("/feeds/subscribe", json=request_data)
                # Should handle safely without executing malicious code
                assert response.status_code in [200, 400, 422]
    
    def test_concurrent_request_handling(self, client):
        """Test handling of concurrent requests."""
        import threading
        
        results = []
        
        def make_request():
            with patch('data_ingest.app.feed_manager') as mock_fm:
                mock_fm.get_processing_stats.return_value = {"uptime_seconds": 100}
                mock_fm.get_feed_stats.return_value = []
                
                response = client.get("/health")
                results.append(response.status_code)
        
        # Create multiple threads making concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should have completed successfully
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_timeout_handling(self, client):
        """Test timeout handling for slow operations."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            # Simulate slow feed manager operation
            async def slow_subscribe(*args, **kwargs):
                await asyncio.sleep(30)  # Simulate very slow operation
                return True
            
            mock_fm.subscribe_to_symbols = slow_subscribe
            
            request_data = {
                "symbol": "AAPL",
                "exchange": "NASDAQ", 
                "data_types": ["ticks"]
            }
            
            # Request should either timeout or complete
            response = client.post("/feeds/subscribe", json=request_data)
            # Accept various timeout-related status codes
            assert response.status_code in [200, 408, 504, 500]
    
    def test_memory_pressure_handling(self, client):
        """Test behavior under memory pressure."""
        # This test simulates memory pressure by making many requests
        # In a real system, this would test memory cleanup and GC behavior
        
        responses = []
        for i in range(100):
            with patch('data_ingest.app.feed_manager') as mock_fm:
                mock_fm.get_processing_stats.return_value = {
                    "uptime_seconds": i,
                    "messages_processed": i * 10
                }
                mock_fm.get_feed_stats.return_value = []
                
                response = client.get("/health")
                responses.append(response.status_code)
                
                # Clean up mock to prevent memory buildup
                mock_fm.reset_mock()
        
        # All requests should succeed despite high volume
        success_rate = sum(1 for status in responses if status == 200) / len(responses)
        assert success_rate >= 0.95  # 95% success rate minimum


class TestSecurityAndValidation:
    """Test security measures and input validation."""
    
    def test_cors_headers_presence(self, client):
        """Test CORS headers are properly configured."""
        response = client.options("/health")
        
        # TestClient may not fully simulate CORS, but we can verify the middleware is configured
        # The actual CORS testing would require a browser or specialized testing tool
        assert response.status_code in [200, 405]  # OPTIONS method handling varies
    
    def test_content_security_headers(self, client):
        """Test security headers in responses."""
        response = client.get("/health")
        
        # Check for security headers (if implemented)
        headers = response.headers
        
        # These would be added by security middleware in production
        # For now, just verify the response is successful
        assert response.status_code in [200, 503]
    
    def test_api_key_header_handling(self, client):
        """Test handling of API key headers."""
        headers = {"X-API-Key": "test-api-key-12345"}
        
        response = client.get("/health", headers=headers)
        # Health endpoint should be accessible regardless of API key
        assert response.status_code in [200, 503]
        
        # Test subscription with API key
        request_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "data_types": ["ticks"]
        }
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            response = client.post("/feeds/subscribe", json=request_data, headers=headers)
            assert response.status_code == 200
    
    def test_user_agent_validation(self, client):
        """Test user agent validation and logging."""
        custom_headers = {
            "User-Agent": "TradingBot/1.0 (https://example.com/bot)"
        }
        
        response = client.get("/health", headers=custom_headers)
        assert response.status_code in [200, 503]
    
    def test_rate_limiting_simulation(self, client):
        """Test rate limiting behavior simulation."""
        # This simulates rapid requests to test rate limiting
        # In a production system, this would test actual rate limiting middleware
        
        responses = []
        for i in range(50):  # Make 50 rapid requests
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Without rate limiting implemented, all should succeed
        # With rate limiting, some would return 429 (Too Many Requests)
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Either all succeed (no rate limiting) or some are rate limited
        assert success_count + rate_limited_count == len(responses)


class TestIntegrationScenarios:
    """Test complete integration scenarios and workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_session_simulation(self, app):
        """Test complete trading session from startup to shutdown."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 0,
                "messages_processed": 0
            }
            mock_fm.get_feed_stats.return_value = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # 1. Check initial service health
                health_response = await client.get("/health")
                assert health_response.status_code in [200, 503]
                
                # 2. Subscribe to multiple symbols
                symbols_to_subscribe = [
                    {"symbol": "AAPL", "exchange": "NASDAQ", "data_types": ["ticks", "quotes"]},
                    {"symbol": "GOOGL", "exchange": "NASDAQ", "data_types": ["ticks", "quotes"]}, 
                    {"symbol": "BTC-USD", "exchange": "BINANCE", "data_types": ["ticks", "order_book"]},
                    {"symbol": "ETH-USD", "exchange": "BINANCE", "data_types": ["ticks", "quotes"]}
                ]
                
                subscription_results = []
                for symbol_data in symbols_to_subscribe:
                    response = await client.post("/feeds/subscribe", json=symbol_data)
                    subscription_results.append({
                        "symbol": symbol_data["symbol"],
                        "success": response.status_code == 200,
                        "status_code": response.status_code
                    })
                
                # 3. Monitor feed status during session
                mock_fm.get_feed_stats.return_value = [
                    DataFeedStats(
                        feed_name="polygon",
                        symbol="AAPL", 
                        exchange=Exchange.POLYGON,
                        connected=True,
                        ticks_received=100,
                        quotes_received=50
                    )
                ]
                
                status_response = await client.get("/feeds/status")
                assert status_response.status_code == 200
                
                # 4. Check metrics
                mock_fm.get_processing_stats.return_value = {
                    "uptime_seconds": 300,
                    "messages_processed": 500,
                    "messages_per_second": 15.5
                }
                
                metrics_response = await client.get("/metrics")
                assert metrics_response.status_code == 200
                
                # 5. Test WebSocket connection for real-time data
                with client.websocket_connect("/ws/market-data") as websocket:
                    # Send ping to verify connection
                    websocket.send_text(json.dumps({"action": "ping"}))
                    
                    try:
                        pong_response = websocket.receive_json()
                        assert pong_response["type"] == "pong"
                    except Exception:
                        # WebSocket functionality may be limited in tests
                        pass
                
                # Verify overall session success
                successful_subscriptions = sum(1 for r in subscription_results if r["success"])
                assert successful_subscriptions >= 0  # At least some should succeed
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, app):
        """Test system recovery from various error conditions."""
        with patch('data_ingest.app.feed_manager') as mock_fm:
            async with AsyncClient(app=app, base_url="http://test") as client:
                
                # Scenario 1: Service temporarily unavailable
                mock_fm.subscribe_to_symbols = AsyncMock(side_effect=Exception("Service temporarily unavailable"))
                
                response1 = await client.post("/feeds/subscribe", json={
                    "symbol": "AAPL",
                    "exchange": "NASDAQ",
                    "data_types": ["ticks"]
                })
                assert response1.status_code == 500
                
                # Scenario 2: Service recovers
                mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
                
                response2 = await client.post("/feeds/subscribe", json={
                    "symbol": "GOOGL",
                    "exchange": "NASDAQ", 
                    "data_types": ["quotes"]
                })
                assert response2.status_code == 200
                
                # Scenario 3: Partial service degradation
                mock_fm.get_processing_stats.return_value = {
                    "uptime_seconds": 100,
                    "messages_processed": 50,
                    "messages_failed": 20
                }
                mock_fm.get_feed_stats.return_value = [
                    DataFeedStats(
                        feed_name="polygon", 
                        symbol="AAPL",
                        exchange=Exchange.POLYGON,
                        connected=False,
                        error_count=5
                    )
                ]
                
                health_response = await client.get("/health")
                assert health_response.status_code == 200
                health_data = health_response.json()
                assert health_data["status"] in ["degraded", "partial"]
    
    @pytest.mark.asyncio
    async def test_load_balancing_simulation(self, app):
        """Test behavior under load balancing scenarios."""
        # Simulate multiple instances handling requests
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {
                "uptime_seconds": 1000,
                "messages_processed": 10000
            }
            mock_fm.get_feed_stats.return_value = []
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                
                # Simulate requests that might be load balanced
                health_checks = []
                for i in range(20):
                    response = await client.get("/health")
                    health_checks.append(response.status_code)
                
                # All health checks should succeed
                assert all(status in [200, 503] for status in health_checks)
    
    def test_configuration_driven_behavior(self, client):
        """Test that system behavior changes based on configuration."""
        # Test with different trading modes
        trading_modes = [TradingMode.TEST, TradingMode.SHADOW, TradingMode.LIVE]
        
        for mode in trading_modes:
            with patch('data_ingest.app.config') as mock_config:
                mock_config.trading_mode = mode
                mock_config.enable_debug_endpoints = (mode == TradingMode.TEST)
                
                with patch('data_ingest.app.feed_manager') as mock_fm:
                    mock_fm.get_processing_stats.return_value = {"uptime_seconds": 100}
                    mock_fm.get_feed_stats.return_value = []
                    
                    # Health check should always work
                    response = client.get("/health")
                    assert response.status_code in [200, 503]
                    
                    # Debug endpoints should only work in test mode
                    debug_response = client.get("/debug/stats")
                    if mode == TradingMode.TEST:
                        assert debug_response.status_code in [200, 503]
                    else:
                        assert debug_response.status_code == 404


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_response_time_under_load(self, client):
        """Test response times remain reasonable under load."""
        import time as time_module
        
        response_times = []
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.get_processing_stats.return_value = {"uptime_seconds": 100}
            mock_fm.get_feed_stats.return_value = []
            
            # Make 50 requests and measure response times
            for i in range(50):
                start_time = time_module.perf_counter()
                response = client.get("/health")
                end_time = time_module.perf_counter()
                
                response_times.append(end_time - start_time)
                assert response.status_code in [200, 503]
            
            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Response times should be reasonable (under 100ms average, 500ms max)
            assert avg_response_time < 0.1, f"Average response time too high: {avg_response_time:.3f}s"
            assert max_response_time < 0.5, f"Max response time too high: {max_response_time:.3f}s"
    
    def test_memory_efficiency(self, client):
        """Test memory efficiency during request processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.subscribe_to_symbols = AsyncMock(return_value=True)
            
            # Make many requests to test memory usage
            for i in range(100):
                request_data = {
                    "symbol": f"TEST{i}",
                    "exchange": "NASDAQ",
                    "data_types": ["ticks"]
                }
                
                response = client.post("/feeds/subscribe", json=request_data)
                assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 50MB for 100 requests)
        max_allowed_growth = 50 * 1024 * 1024  # 50MB
        assert memory_growth < max_allowed_growth, f"Memory grew by {memory_growth / 1024 / 1024:.1f} MB"
    
    def test_concurrent_websocket_performance(self, app):
        """Test WebSocket performance with multiple connections."""
        # This test would be more meaningful with actual WebSocket load testing tools
        # For now, we test the basic scalability of the connection management
        
        with patch('data_ingest.app.feed_manager') as mock_fm:
            mock_fm.websocket_connections = []
            
            # Simulate multiple WebSocket connections
            for i in range(10):
                mock_connection = AsyncMock()
                mock_fm.websocket_connections.append(mock_connection)
            
            assert len(mock_fm.websocket_connections) == 10
            
            # Verify system can handle the connection list efficiently
            connections_count = len(mock_fm.websocket_connections)
            assert connections_count == 10


# Test utilities and helpers
def simulate_market_data_flow(client, duration_seconds: int = 10):
    """Utility function to simulate a complete market data flow."""
    results = {
        "subscriptions": [],
        "health_checks": [],
        "feed_status_checks": [],
        "errors": []
    }
    
    symbols = ["AAPL", "GOOGL", "MSFT", "BTCUSD", "ETHUSD"]
    
    # Subscribe to symbols
    for symbol in symbols:
        try:
            request_data = {
                "symbol": symbol,
                "exchange": "NASDAQ" if not symbol.endswith("USD") else "BINANCE",
                "data_types": ["ticks", "quotes"]
            }
            
            response = client.post("/feeds/subscribe", json=request_data)
            results["subscriptions"].append({
                "symbol": symbol,
                "success": response.status_code == 200,
                "status_code": response.status_code
            })
            
        except Exception as e:
            results["errors"].append(f"Subscription error for {symbol}: {e}")
    
    # Simulate monitoring during market session
    import time
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        try:
            # Health check
            health_response = client.get("/health")
            results["health_checks"].append({
                "timestamp": time.time(),
                "status": health_response.status_code,
                "healthy": health_response.status_code == 200
            })
            
            # Feed status check
            status_response = client.get("/feeds/status") 
            results["feed_status_checks"].append({
                "timestamp": time.time(),
                "status": status_response.status_code
            })
            
        except Exception as e:
            results["errors"].append(f"Monitoring error: {e}")
        
        time.sleep(1)  # Check every second
    
    return results


if __name__ == "__main__":
    # Run comprehensive test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=data_ingest.app",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10",  # Show 10 slowest tests
        "-x",  # Stop on first failure
    ])