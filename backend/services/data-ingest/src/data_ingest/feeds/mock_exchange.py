"""
Mock exchange feed for testing and development.
Generates realistic market data patterns without external dependencies.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal
from pathlib import Path

import numpy as np

from .base import BaseDataFeed, DataType, FeedConfig, FeedStatus
from ..models.schemas import Tick, Quote, Candle, OrderBook, OrderBookLevel, Exchange, Side
from ...shared.utils.logging import get_logger


class MockConfig(FeedConfig):
    """Mock feed configuration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update({
            "tick_interval": 0.1,  # seconds between ticks
            "quote_interval": 0.5,  # seconds between quotes
            "candle_interval": 60,  # seconds between candles
            "orderbook_interval": 1,  # seconds between orderbook updates
            "price_volatility": 0.01,  # daily volatility (1%)
            "spread_bps": 5,  # bid-ask spread in basis points
            "enable_trends": True,  # enable trending price movements
            "enable_noise": True,  # add random noise
            "data_file": kwargs.get("data_file"),  # optional CSV data file
            "replay_speed": 1.0,  # playback speed multiplier
        })


class MockDataGenerator:
    """Generates realistic market data using various models."""
    
    def __init__(self, symbol: str, base_price: float = 100.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        
        # Market microstructure parameters
        self.bid_price = base_price * 0.9995
        self.ask_price = base_price * 1.0005
        self.volume = 1000
        
        # Time series parameters
        self.trend = 0.0  # Current trend direction
        self.volatility = 0.01  # Current volatility
        self.last_update = time.time()
        
        # Order book state
        self.bid_levels = self._generate_order_book_side(self.bid_price, -1)
        self.ask_levels = self._generate_order_book_side(self.ask_price, 1)
        
        # Random seed for deterministic testing
        self.rng = np.random.RandomState(hash(symbol) % 2**32)
    
    def _generate_order_book_side(self, base_price: float, direction: int, levels: int = 10) -> List[Dict[str, float]]:
        """Generate order book levels for one side."""
        book_side = []
        
        for i in range(levels):
            # Price levels with increasing spacing
            price_offset = (i + 1) * 0.0001 * abs(direction)
            price = base_price - (price_offset if direction < 0 else -price_offset)
            
            # Size decreases with distance from mid
            size = self.rng.exponential(100) * (1 / (i + 1))
            
            book_side.append({"price": price, "size": size})
        
        return book_side
    
    def update_prices(self, dt: float) -> None:
        """Update prices using geometric Brownian motion with mean reversion."""
        # Mean reversion component
        mean_reversion_rate = 0.1
        long_term_mean = self.base_price
        mean_reversion = mean_reversion_rate * (long_term_mean - self.current_price) * dt
        
        # Brownian motion component
        brownian_motion = self.volatility * self.rng.normal(0, np.sqrt(dt))
        
        # Trending component
        if self.rng.random() < 0.01:  # Trend changes 1% of the time
            self.trend = self.rng.normal(0, 0.001)
        
        trend_component = self.trend * dt
        
        # Update price
        price_change = mean_reversion + brownian_motion + trend_component
        self.current_price = max(0.01, self.current_price + price_change)
        
        # Update bid/ask with realistic spread
        spread = self.current_price * 0.0005  # 5 bps spread
        self.bid_price = self.current_price - spread / 2
        self.ask_price = self.current_price + spread / 2
        
        # Update order book levels
        self.bid_levels = self._generate_order_book_side(self.bid_price, -1)
        self.ask_levels = self._generate_order_book_side(self.ask_price, 1)
        
        self.last_update = time.time()
    
    def generate_tick(self) -> Tick:
        """Generate a realistic trade tick."""
        # Determine if trade is buy or sell (slight buy bias)
        is_buy = self.rng.random() > 0.48
        
        # Price based on side and some randomness
        if is_buy:
            price = self.ask_price + self.rng.uniform(-0.0001, 0.0001) * self.current_price
            side = Side.BUY
        else:
            price = self.bid_price + self.rng.uniform(-0.0001, 0.0001) * self.current_price
            side = Side.SELL
        
        # Generate realistic trade size
        size = max(1, self.rng.exponential(50))
        
        return Tick(
            timestamp=datetime.now(),
            symbol=self.symbol,
            exchange=Exchange.MOCK,
            price=Decimal(str(round(price, 4))),
            size=Decimal(str(round(size, 2))),
            side=side,
            trade_id=f"mock_{int(time.time() * 1000)}"
        )
    
    def generate_quote(self) -> Quote:
        """Generate bid/ask quote."""
        return Quote(
            timestamp=datetime.now(),
            symbol=self.symbol,
            exchange=Exchange.MOCK,
            bid_price=Decimal(str(round(self.bid_price, 4))),
            bid_size=Decimal(str(round(self.bid_levels[0]["size"], 2))),
            ask_price=Decimal(str(round(self.ask_price, 4))),
            ask_size=Decimal(str(round(self.ask_levels[0]["size"], 2)))
        )
    
    def generate_order_book(self) -> OrderBook:
        """Generate full order book snapshot."""
        bids = [
            OrderBookLevel(
                price=Decimal(str(round(level["price"], 4))),
                size=Decimal(str(round(level["size"], 2)))
            )
            for level in self.bid_levels
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(round(level["price"], 4))),
                size=Decimal(str(round(level["size"], 2)))
            )
            for level in self.ask_levels
        ]
        
        return OrderBook(
            timestamp=datetime.now(),
            symbol=self.symbol,
            exchange=Exchange.MOCK,
            bids=bids,
            asks=asks,
            sequence_number=int(time.time() * 1000)
        )
    
    def generate_candle(self, timeframe: str, duration_seconds: int) -> Candle:
        """Generate OHLCV candle for time period."""
        # Simple candle based on current price and some randomness
        close_price = self.current_price
        
        # Generate realistic OHLC
        price_range = self.current_price * 0.002  # 0.2% range
        high_price = close_price + self.rng.uniform(0, price_range)
        low_price = close_price - self.rng.uniform(0, price_range)
        open_price = close_price + self.rng.uniform(-price_range/2, price_range/2)
        
        # Ensure OHLC constraints
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume based on volatility
        base_volume = 1000
        volume = max(1, self.rng.exponential(base_volume))
        
        return Candle(
            timestamp=datetime.now(),
            symbol=self.symbol,
            exchange=Exchange.MOCK,
            timeframe=timeframe,
            open_price=Decimal(str(round(open_price, 4))),
            high_price=Decimal(str(round(high_price, 4))),
            low_price=Decimal(str(round(low_price, 4))),
            close_price=Decimal(str(round(close_price, 4))),
            volume=Decimal(str(round(volume, 2)))
        )


class MockExchangeFeed(BaseDataFeed):
    """
    Mock exchange feed that generates realistic market data.
    Perfect for testing, development, and backtesting.
    """
    
    def __init__(self, config: MockConfig, **kwargs):
        super().__init__("mock_exchange", Exchange.MOCK, config, **kwargs)
        
        # Data generators for each subscribed symbol
        self.generators: Dict[str, MockDataGenerator] = {}
        
        # Background tasks for data generation
        self.generation_tasks: Set[asyncio.Task] = set()
        
        # Historical data replay
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.replay_index: Dict[str, int] = {}
        
        # Load historical data if specified
        if self.config.get("data_file"):
            self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical data from CSV file."""
        data_file = Path(self.config["data_file"])
        
        if data_file.exists() and data_file.suffix == '.csv':
            try:
                import pandas as pd
                
                df = pd.read_csv(data_file)
                
                # Group by symbol if present
                if 'symbol' in df.columns:
                    for symbol in df['symbol'].unique():
                        symbol_data = df[df['symbol'] == symbol].to_dict('records')
                        self.historical_data[symbol] = symbol_data
                        self.replay_index[symbol] = 0
                else:
                    # Single symbol file
                    symbol = data_file.stem.upper()
                    self.historical_data[symbol] = df.to_dict('records')
                    self.replay_index[symbol] = 0
                
                self.logger.info(
                    "Historical data loaded",
                    file=str(data_file),
                    symbols=list(self.historical_data.keys()),
                    total_records=sum(len(data) for data in self.historical_data.values())
                )
                
            except Exception as e:
                self.logger.error("Failed to load historical data", error=str(e))
    
    async def connect(self) -> None:
        """Mock connection - always succeeds instantly."""
        self.status = FeedStatus.CONNECTED
        self.logger.info("Mock exchange connected")
    
    async def disconnect(self) -> None:
        """Stop all generation tasks."""
        # Cancel all background tasks
        for task in self.generation_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.generation_tasks:
            await asyncio.gather(*self.generation_tasks, return_exceptions=True)
        
        self.generation_tasks.clear()
        self.generators.clear()
        
        self.status = FeedStatus.DISCONNECTED
        self.logger.info("Mock exchange disconnected")
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Start generating data for subscribed symbols."""
        try:
            for symbol in symbols:
                # Create data generator if not exists
                if symbol not in self.generators:
                    # Use historical data as base price if available
                    base_price = 100.0
                    if symbol in self.historical_data and self.historical_data[symbol]:
                        base_price = float(self.historical_data[symbol][0].get('close', 100.0))
                    
                    self.generators[symbol] = MockDataGenerator(symbol, base_price)
                
                # Update subscriptions
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = set()
                self.subscriptions[symbol].update(data_types)
                
                # Start generation tasks for each data type
                for data_type in data_types:
                    task_name = f"{symbol}_{data_type.value}"
                    
                    # Check if task already running
                    if any(task.get_name() == task_name for task in self.generation_tasks):
                        continue
                    
                    if data_type == DataType.TICKS:
                        task = asyncio.create_task(
                            self._generate_ticks(symbol),
                            name=task_name
                        )
                    elif data_type == DataType.QUOTES:
                        task = asyncio.create_task(
                            self._generate_quotes(symbol),
                            name=task_name
                        )
                    elif data_type == DataType.CANDLES:
                        task = asyncio.create_task(
                            self._generate_candles(symbol),
                            name=task_name
                        )
                    elif data_type == DataType.ORDER_BOOK:
                        task = asyncio.create_task(
                            self._generate_order_books(symbol),
                            name=task_name
                        )
                    else:
                        continue
                    
                    self.generation_tasks.add(task)
            
            self.logger.info(
                "Mock subscriptions started",
                symbols=symbols,
                data_types=[dt.value for dt in data_types],
                active_tasks=len(self.generation_tasks)
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Mock subscription failed", error=str(e))
            return False
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Stop generating data for unsubscribed symbols."""
        try:
            for symbol in symbols:
                # Update subscriptions
                if symbol in self.subscriptions:
                    self.subscriptions[symbol] -= set(data_types)
                    if not self.subscriptions[symbol]:
                        del self.subscriptions[symbol]
                
                # Cancel relevant tasks
                for data_type in data_types:
                    task_name = f"{symbol}_{data_type.value}"
                    
                    for task in list(self.generation_tasks):
                        if task.get_name() == task_name:
                            task.cancel()
                            self.generation_tasks.remove(task)
            
            return True
            
        except Exception as e:
            self.logger.error("Mock unsubscription failed", error=str(e))
            return False
    
    async def _generate_ticks(self, symbol: str) -> None:
        """Generate tick data for symbol."""
        interval = self.config.get("tick_interval", 0.1)
        generator = self.generators[symbol]
        
        while symbol in self.subscriptions and DataType.TICKS in self.subscriptions[symbol]:
            try:
                # Update price model
                generator.update_prices(interval)
                
                # Generate tick
                tick = generator.generate_tick()
                
                # Send to handlers
                await self._handle_message(tick.__dict__)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Tick generation error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _generate_quotes(self, symbol: str) -> None:
        """Generate quote data for symbol."""
        interval = self.config.get("quote_interval", 0.5)
        generator = self.generators[symbol]
        
        while symbol in self.subscriptions and DataType.QUOTES in self.subscriptions[symbol]:
            try:
                # Update price model
                generator.update_prices(interval)
                
                # Generate quote
                quote = generator.generate_quote()
                
                # Send to handlers
                await self._handle_message(quote.__dict__)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Quote generation error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _generate_candles(self, symbol: str) -> None:
        """Generate candle data for symbol."""
        interval = self.config.get("candle_interval", 60)
        generator = self.generators[symbol]
        
        while symbol in self.subscriptions and DataType.CANDLES in self.subscriptions[symbol]:
            try:
                # Update price model
                generator.update_prices(interval)
                
                # Generate candle
                candle = generator.generate_candle("1m", interval)
                
                # Send to handlers
                await self._handle_message(candle.__dict__)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Candle generation error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _generate_order_books(self, symbol: str) -> None:
        """Generate order book data for symbol."""
        interval = self.config.get("orderbook_interval", 1)
        generator = self.generators[symbol]
        
        while symbol in self.subscriptions and DataType.ORDER_BOOK in self.subscriptions[symbol]:
            try:
                # Update price model
                generator.update_prices(interval)
                
                # Generate order book
                order_book = generator.generate_order_book()
                
                # Send to handlers
                await self._handle_message(order_book.__dict__)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Order book generation error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Dict[str, Any]) -> Optional[Any]:
        """Process generated message - already in correct format."""
        return message
    
    def set_base_price(self, symbol: str, price: float) -> None:
        """Set base price for a symbol generator."""
        if symbol in self.generators:
            self.generators[symbol].base_price = price
            self.generators[symbol].current_price = price
            self.logger.info("Base price updated", symbol=symbol, price=price)
    
    def set_volatility(self, symbol: str, volatility: float) -> None:
        """Set volatility for a symbol generator."""
        if symbol in self.generators:
            self.generators[symbol].volatility = volatility
            self.logger.info("Volatility updated", symbol=symbol, volatility=volatility)
    
    def inject_price_shock(self, symbol: str, shock_pct: float) -> None:
        """Inject a sudden price movement."""
        if symbol in self.generators:
            generator = self.generators[symbol]
            shock_amount = generator.current_price * (shock_pct / 100)
            generator.current_price += shock_amount
            self.logger.info("Price shock injected", symbol=symbol, shock_pct=shock_pct)


# Factory function
def create_mock_feed(**config_kwargs) -> MockExchangeFeed:
    """Create mock exchange feed with configuration."""
    config = MockConfig(**config_kwargs)
    return MockExchangeFeed(config)


# Example usage and testing
async def test_mock_feed():
    """Test mock feed functionality."""
    
    # Create mock feed
    mock_feed = create_mock_feed(
        tick_interval=0.1,
        quote_interval=0.5,
        enable_trends=True
    )
    
    # Message counter
    message_count = {"ticks": 0, "quotes": 0}
    
    def tick_handler(tick_data):
        message_count["ticks"] += 1
        if message_count["ticks"] <= 3:
            print(f"Tick: {tick_data['symbol']} @ {tick_data['price']} x {tick_data['size']}")
    
    def quote_handler(quote_data):
        message_count["quotes"] += 1
        if message_count["quotes"] <= 3:
            print(f"Quote: {quote_data['symbol']} bid={quote_data['bid_price']} ask={quote_data['ask_price']}")
    
    try:
        # Register handlers
        mock_feed.register_handler(DataType.TICKS, tick_handler)
        mock_feed.register_handler(DataType.QUOTES, quote_handler)
        
        # Start feed
        await mock_feed.start()
        
        if mock_feed.is_connected():
            print("✓ Mock feed connected")
            
            # Subscribe to data
            success = await mock_feed.subscribe(["AAPL", "BTCUSD"], [DataType.TICKS, DataType.QUOTES])
            if success:
                print("✓ Mock subscriptions successful")
                
                # Let it run for a few seconds
                await asyncio.sleep(3)
                
                # Inject a price shock
                mock_feed.inject_price_shock("AAPL", 5.0)  # 5% shock
                await asyncio.sleep(2)
                
                stats = mock_feed.get_stats()
                print(f"✓ Generated {message_count['ticks']} ticks, {message_count['quotes']} quotes")
        
    except Exception as e:
        print(f"✗ Mock feed test failed: {e}")
    finally:
        await mock_feed.stop()


if __name__ == "__main__":
    asyncio.run(test_mock_feed())