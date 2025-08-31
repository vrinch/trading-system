"""
Polygon.io market data feed implementation.
Provides real-time and historical data for stocks, options, forex, and crypto.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlencode

import aiohttp
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseDataFeed, DataType, FeedConfig, FeedStatus, ConnectionError, AuthenticationError
from ..models.schemas import Tick, Quote, Candle, OrderBook, Exchange, Side
from ...shared.utils.logging import get_logger, LogEvents


class PolygonConfig(FeedConfig):
    """Polygon.io specific configuration."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.update({
            "api_key": api_key,
            "base_url": "https://api.polygon.io",
            "websocket_url": "wss://socket.polygon.io",
            "max_symbols_per_connection": 100,
            "subscription_buffer_size": 1000,
            "enable_options": True,
            "enable_forex": True,
            "enable_crypto": True,
        })


class PolygonFeed(BaseDataFeed):
    """
    Polygon.io WebSocket and REST API feed implementation.
    Supports stocks, options, forex, and cryptocurrency data.
    """
    
    def __init__(self, config: PolygonConfig, **kwargs):
        super().__init__("polygon", Exchange.POLYGON, config, **kwargs)
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.websocket_url = config["websocket_url"]
        
        # Connection objects
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Polygon-specific state
        self.authenticated = False
        self.subscription_buffer: List[Dict[str, Any]] = []
        
        # Market type mapping
        self.market_type_map = {
            "stocks": "A",      # Aggregates (minute bars)
            "trades": "T",      # Trades
            "quotes": "Q",      # Quotes
            "options": "O",     # Options
            "forex": "C",       # Forex
            "crypto": "X"       # Crypto
        }
    
    async def connect(self) -> None:
        """Establish WebSocket connection to Polygon.io."""
        try:
            self.logger.info("Connecting to Polygon WebSocket", url=self.websocket_url)
            
            # Create HTTP session for REST API calls
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                f"{self.websocket_url}/stocks",
                ping_interval=20,
                ping_timeout=10,
                max_size=2**23,  # 8MB buffer for large messages
                compression=None if not self.config.get("enable_compression", True) else "deflate"
            )
            
            # Authenticate
            await self._authenticate()
            
            # Start message processing loop
            asyncio.create_task(self._message_loop())
            
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to Polygon WebSocket successfully")
            
        except Exception as e:
            self.status = FeedStatus.ERROR
            self.logger.error("Failed to connect to Polygon", error=str(e))
            raise ConnectionError(f"Polygon connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close WebSocket and HTTP session."""
        self.authenticated = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.warning("Error closing WebSocket", error=str(e))
            finally:
                self.websocket = None
        
        if self.http_session:
            try:
                await self.http_session.close()
            except Exception as e:
                self.logger.warning("Error closing HTTP session", error=str(e))
            finally:
                self.http_session = None
        
        self.status = FeedStatus.DISCONNECTED
        self.logger.info("Disconnected from Polygon")
    
    async def _authenticate(self) -> None:
        """Authenticate WebSocket connection."""
        auth_message = {
            "action": "auth",
            "params": self.api_key
        }
        
        await self.websocket.send(json.dumps(auth_message))
        
        # Wait for authentication response
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
            auth_response = json.loads(response)
            
            if auth_response[0].get("status") == "auth_success":
                self.authenticated = True
                self.logger.info("Polygon authentication successful")
            else:
                raise AuthenticationError(f"Authentication failed: {auth_response}")
                
        except asyncio.TimeoutError:
            raise AuthenticationError("Authentication timeout")
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to symbols and data types."""
        if not self.authenticated:
            self.logger.error("Cannot subscribe - not authenticated")
            return False
        
        try:
            # Map data types to Polygon subscription types
            subscription_actions = []
            
            for data_type in data_types:
                if data_type == DataType.TICKS:
                    subscription_actions.append("T.*")  # All trades
                elif data_type == DataType.QUOTES:
                    subscription_actions.append("Q.*")  # All quotes
                elif data_type == DataType.CANDLES:
                    subscription_actions.append("A.*")  # Minute aggregates
                elif data_type == DataType.ORDER_BOOK:
                    subscription_actions.append("L2.*")  # Level 2 data
            
            # Build subscription message
            subscription_message = {
                "action": "subscribe",
                "params": ",".join([f"{action}" for action in subscription_actions])
            }
            
            await self.websocket.send(json.dumps(subscription_message))
            
            # Update local subscription tracking
            for symbol in symbols:
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = set()
                self.subscriptions[symbol].update(data_types)
            
            self.logger.info(
                "Subscribed to symbols",
                symbols=symbols,
                data_types=[dt.value for dt in data_types]
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Subscription failed", error=str(e))
            return False
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from symbols and data types."""
        try:
            # Build unsubscription message
            unsubscribe_actions = []
            
            for data_type in data_types:
                if data_type == DataType.TICKS:
                    unsubscribe_actions.append("T.*")
                elif data_type == DataType.QUOTES:
                    unsubscribe_actions.append("Q.*")
                elif data_type == DataType.CANDLES:
                    unsubscribe_actions.append("A.*")
            
            unsubscribe_message = {
                "action": "unsubscribe",
                "params": ",".join(unsubscribe_actions)
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            
            # Update local subscription tracking
            for symbol in symbols:
                if symbol in self.subscriptions:
                    self.subscriptions[symbol] -= set(data_types)
                    if not self.subscriptions[symbol]:
                        del self.subscriptions[symbol]
            
            self.logger.info("Unsubscribed from symbols", symbols=symbols)
            return True
            
        except Exception as e:
            self.logger.error("Unsubscription failed", error=str(e))
            return False
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.status == FeedStatus.CONNECTED and self.websocket:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=self.config.get("message_timeout", 30)
                )
                
                # Parse and handle message
                if message:
                    await self._handle_raw_message(message)
                    
            except asyncio.TimeoutError:
                self.logger.warning("Message timeout - connection may be stale")
                break
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                break
            except Exception as e:
                self.logger.error("Error in message loop", error=str(e))
                await asyncio.sleep(1)
    
    async def _handle_raw_message(self, raw_message: str) -> None:
        """Process raw WebSocket message."""
        try:
            messages = json.loads(raw_message)
            
            # Handle array of messages
            if isinstance(messages, list):
                for message in messages:
                    await self._handle_message(message)
            else:
                await self._handle_message(messages)
                
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse message JSON", error=str(e))
        except Exception as e:
            self.logger.error("Error handling raw message", error=str(e))
    
    async def _process_message(self, message: Dict[str, Any]) -> Optional[Any]:
        """Convert Polygon message to standardized format."""
        try:
            event_type = message.get("ev")  # Event type
            
            if event_type == "T":  # Trade
                return self._parse_trade_message(message)
            elif event_type == "Q":  # Quote
                return self._parse_quote_message(message)
            elif event_type == "A" or event_type == "AM":  # Aggregate/minute bar
                return self._parse_candle_message(message)
            elif event_type == "status":
                self._handle_status_message(message)
                return None
            else:
                self.logger.debug("Unhandled message type", event_type=event_type)
                return None
                
        except Exception as e:
            self.logger.error("Message processing failed", error=str(e), message=message)
            return None
    
    def _parse_trade_message(self, message: Dict[str, Any]) -> Tick:
        """Parse trade message into Tick object."""
        return Tick(
            timestamp=message.get("t", time.time() * 1000) / 1000,  # Convert ms to seconds
            symbol=message.get("sym", "").replace(".", "-"),  # Normalize symbol format
            exchange=Exchange.POLYGON,
            price=message.get("p", 0.0),
            size=message.get("s", 0.0),
            side=self._parse_side(message.get("c", [])),
            conditions=message.get("c", []),
            trade_id=str(message.get("i", "")),
            sequence_number=message.get("q", None)
        )
    
    def _parse_quote_message(self, message: Dict[str, Any]) -> Quote:
        """Parse quote message into Quote object."""
        return Quote(
            timestamp=message.get("t", time.time() * 1000) / 1000,
            symbol=message.get("sym", "").replace(".", "-"),
            exchange=Exchange.POLYGON,
            bid_price=message.get("bp", 0.0),
            bid_size=message.get("bs", 0.0),
            ask_price=message.get("ap", 0.0),
            ask_size=message.get("as", 0.0)
        )
    
    def _parse_candle_message(self, message: Dict[str, Any]) -> Candle:
        """Parse aggregate/minute bar into Candle object."""
        return Candle(
            timestamp=message.get("s", time.time() * 1000) / 1000,  # Start time
            symbol=message.get("sym", "").replace(".", "-"),
            exchange=Exchange.POLYGON,
            timeframe="1m",  # Polygon sends minute bars
            open_price=message.get("o", 0.0),
            high_price=message.get("h", 0.0),
            low_price=message.get("l", 0.0),
            close_price=message.get("c", 0.0),
            volume=message.get("v", 0.0),
            trade_count=message.get("n", None),
            vwap=message.get("vw", None)
        )
    
    def _parse_side(self, conditions: List[str]) -> Optional[Side]:
        """Determine trade side from conditions."""
        # Polygon doesn't provide explicit side, attempt to infer
        if not conditions:
            return Side.UNKNOWN
        
        # This is simplified - real implementation would need condition mapping
        return Side.UNKNOWN
    
    def _handle_status_message(self, message: Dict[str, Any]) -> None:
        """Handle status/control messages."""
        status = message.get("status")
        msg = message.get("message", "")
        
        if status == "connected":
            self.logger.info("Polygon connection confirmed", message=msg)
        elif status == "auth_success":
            self.authenticated = True
            self.logger.info("Polygon authentication confirmed")
        elif status == "auth_failed":
            self.logger.error("Polygon authentication failed", message=msg)
        else:
            self.logger.debug("Polygon status message", status=status, message=msg)
    
    async def _send_heartbeat(self) -> None:
        """Send ping to keep connection alive."""
        if self.websocket and self.authenticated:
            try:
                await self.websocket.ping()
                self.logger.debug("Sent heartbeat ping")
            except Exception as e:
                self.logger.error("Heartbeat failed", error=str(e))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def get_historical_data(
        self,
        symbol: str,
        data_type: str = "trades",
        from_date: str = None,
        to_date: str = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Fetch historical data via REST API."""
        if not self.http_session:
            raise ConnectionError("HTTP session not initialized")
        
        # Build API endpoint based on data type
        if data_type == "trades":
            endpoint = f"/v3/trades/{symbol}"
        elif data_type == "quotes":
            endpoint = f"/v3/quotes/{symbol}"
        elif data_type == "candles":
            endpoint = f"/v2/aggs/ticker/{symbol}/range/1/minute/{from_date}/{to_date}"
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Build query parameters
        params = {
            "apikey": self.api_key,
            "limit": limit
        }
        
        if from_date and data_type != "candles":
            params["timestamp.gte"] = from_date
        if to_date and data_type != "candles":
            params["timestamp.lte"] = to_date
        
        url = f"{self.base_url}{endpoint}?" + urlencode(params)
        
        try:
            async with self.http_session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                self.logger.info(
                    "Historical data fetched",
                    symbol=symbol,
                    data_type=data_type,
                    records=data.get("resultsCount", 0)
                )
                
                return data.get("results", [])
                
        except aiohttp.ClientError as e:
            self.logger.error("Historical data request failed", error=str(e))
            raise
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        if not self.http_session:
            raise ConnectionError("HTTP session not initialized")
        
        url = f"{self.base_url}/v1/marketstatus/now?apikey={self.api_key}"
        
        try:
            async with self.http_session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error("Market status request failed", error=str(e))
            raise
    
    async def get_symbol_details(self, symbol: str) -> Dict[str, Any]:
        """Get symbol details and metadata."""
        if not self.http_session:
            raise ConnectionError("HTTP session not initialized")
        
        url = f"{self.base_url}/v3/reference/tickers/{symbol}?apikey={self.api_key}"
        
        try:
            async with self.http_session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("results", {})
        except aiohttp.ClientError as e:
            self.logger.error("Symbol details request failed", error=str(e))
            raise


# Factory function for easy instantiation
def create_polygon_feed(api_key: str, **config_kwargs) -> PolygonFeed:
    """Create configured Polygon feed instance."""
    config = PolygonConfig(api_key=api_key, **config_kwargs)
    return PolygonFeed(config)


# Example usage and testing
async def test_polygon_connection():
    """Test Polygon connection and basic functionality."""
    import os
    
    api_key = os.getenv("POLYGON_API_KEY", "demo")
    feed = create_polygon_feed(api_key)
    
    try:
        # Test connection
        await feed.start()
        
        if feed.is_connected():
            print("✓ Polygon connection successful")
            
            # Test subscription
            success = await feed.subscribe(["AAPL"], [DataType.TICKS, DataType.QUOTES])
            if success:
                print("✓ Subscription successful")
                
                # Listen for a few seconds
                await asyncio.sleep(5)
                
                stats = feed.get_stats()
                print(f"✓ Stats: {stats.ticks_received} ticks, {stats.quotes_received} quotes")
            
        else:
            print("✗ Connection failed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    finally:
        await feed.stop()


if __name__ == "__main__":
    asyncio.run(test_polygon_connection())