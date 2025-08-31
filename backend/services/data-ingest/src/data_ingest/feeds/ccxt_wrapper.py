"""
CCXT wrapper for cryptocurrency exchange data feeds.
Provides unified interface for multiple crypto exchanges.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal

import ccxt.pro as ccxtpro
import ccxt
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseDataFeed, DataType, FeedConfig, FeedStatus, ConnectionError
from ..models.schemas import Tick, Quote, Candle, OrderBook, OrderBookLevel, Exchange, Side
from ...shared.utils.logging import get_logger


class CCXTConfig(FeedConfig):
    """CCXT-specific configuration."""
    
    def __init__(self, exchange_id: str, **kwargs):
        super().__init__(**kwargs)
        self.update({
            "exchange_id": exchange_id,
            "sandbox": True,  # Use sandbox by default
            "api_key": kwargs.get("api_key", ""),
            "secret": kwargs.get("secret", ""),
            "password": kwargs.get("password", ""),
            "uid": kwargs.get("uid", ""),
            "options": kwargs.get("options", {}),
            "enable_rate_limit": True,
            "timeout": 30000,  # 30 seconds
            "rateLimit": 1000,  # milliseconds between requests
        })


class CCXTFeed(BaseDataFeed):
    """
    CCXT Pro WebSocket feed for cryptocurrency exchanges.
    Supports Binance, Coinbase, Kraken, and other major exchanges.
    """
    
    def __init__(self, config: CCXTConfig, **kwargs):
        # Map exchange IDs to our enum
        exchange_mapping = {
            "binance": Exchange.BINANCE,
            "coinbase": Exchange.COINBASE,
            "coinbasepro": Exchange.COINBASE,
            "kraken": Exchange.KRAKEN,
        }
        
        exchange_enum = exchange_mapping.get(
            config["exchange_id"].lower(), 
            Exchange.BINANCE  # Default fallback
        )
        
        super().__init__(f"ccxt_{config['exchange_id']}", exchange_enum, config, **kwargs)
        
        self.exchange_id = config["exchange_id"]
        self.exchange_instance: Optional[ccxtpro.Exchange] = None
        
        # Symbol mapping and market info
        self.markets: Dict[str, Any] = {}
        self.symbol_map: Dict[str, str] = {}  # normalized -> exchange format
        
        # Subscription tracking per data type
        self.ticker_subscriptions: Set[str] = set()
        self.trades_subscriptions: Set[str] = set()
        self.orderbook_subscriptions: Set[str] = set()
        self.ohlcv_subscriptions: Set[str] = set()
    
    async def connect(self) -> None:
        """Initialize CCXT exchange instance and load markets."""
        try:
            # Create exchange instance
            exchange_class = getattr(ccxtpro, self.exchange_id)
            
            self.exchange_instance = exchange_class({
                'apiKey': self.config.get('api_key', ''),
                'secret': self.config.get('secret', ''),
                'password': self.config.get('password', ''),
                'uid': self.config.get('uid', ''),
                'sandbox': self.config.get('sandbox', True),
                'enableRateLimit': self.config.get('enable_rate_limit', True),
                'timeout': self.config.get('timeout', 30000),
                'rateLimit': self.config.get('rateLimit', 1000),
                'options': self.config.get('options', {}),
            })
            
            # Load markets for symbol mapping
            self.markets = await self.exchange_instance.load_markets()
            self._build_symbol_map()
            
            self.status = FeedStatus.CONNECTED
            self.logger.info(
                "CCXT exchange connected",
                exchange=self.exchange_id,
                markets_loaded=len(self.markets),
                sandbox=self.config.get('sandbox', True)
            )
            
        except Exception as e:
            self.status = FeedStatus.ERROR
            self.logger.error("CCXT connection failed", error=str(e))
            raise ConnectionError(f"CCXT connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close CCXT exchange connections."""
        if self.exchange_instance:
            try:
                await self.exchange_instance.close()
            except Exception as e:
                self.logger.warning("Error closing CCXT exchange", error=str(e))
            finally:
                self.exchange_instance = None
        
        self.status = FeedStatus.DISCONNECTED
        self.logger.info("CCXT exchange disconnected")
    
    def _build_symbol_map(self) -> None:
        """Build mapping between normalized symbols and exchange symbols."""
        for symbol, market in self.markets.items():
            # Normalize symbol (e.g., BTC/USD -> BTC-USD)
            normalized = symbol.replace('/', '-')
            self.symbol_map[normalized] = symbol
            
            # Also map base/quote parts
            if '/' in symbol:
                base, quote = symbol.split('/')
                alt_format = f"{base}_{quote}"
                self.symbol_map[alt_format] = symbol
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to exchange format."""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        
        # Try common conversions
        if '-' in symbol:
            return symbol.replace('-', '/')
        elif '_' in symbol:
            return symbol.replace('_', '/')
        
        return symbol
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to symbols and data types using CCXT Pro."""
        if not self.exchange_instance:
            return False
        
        try:
            for symbol in symbols:
                exchange_symbol = self._normalize_symbol(symbol)
                
                # Update subscription tracking
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = set()
                self.subscriptions[symbol].update(data_types)
                
                # Start appropriate subscription loops
                for data_type in data_types:
                    if data_type == DataType.TICKS and exchange_symbol not in self.trades_subscriptions:
                        self.trades_subscriptions.add(exchange_symbol)
                        asyncio.create_task(self._trades_loop(exchange_symbol))
                    
                    elif data_type == DataType.QUOTES and exchange_symbol not in self.ticker_subscriptions:
                        self.ticker_subscriptions.add(exchange_symbol)
                        asyncio.create_task(self._ticker_loop(exchange_symbol))
                    
                    elif data_type == DataType.ORDER_BOOK and exchange_symbol not in self.orderbook_subscriptions:
                        self.orderbook_subscriptions.add(exchange_symbol)
                        asyncio.create_task(self._orderbook_loop(exchange_symbol))
                    
                    elif data_type == DataType.CANDLES and exchange_symbol not in self.ohlcv_subscriptions:
                        self.ohlcv_subscriptions.add(exchange_symbol)
                        asyncio.create_task(self._ohlcv_loop(exchange_symbol, '1m'))
            
            self.logger.info("CCXT subscriptions started", symbols=symbols, data_types=[dt.value for dt in data_types])
            return True
            
        except Exception as e:
            self.logger.error("CCXT subscription failed", error=str(e))
            return False
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from symbols and data types."""
        try:
            for symbol in symbols:
                exchange_symbol = self._normalize_symbol(symbol)
                
                # Remove from subscription tracking
                if symbol in self.subscriptions:
                    self.subscriptions[symbol] -= set(data_types)
                    if not self.subscriptions[symbol]:
                        del self.subscriptions[symbol]
                
                # Remove from specific subscription sets
                for data_type in data_types:
                    if data_type == DataType.TICKS:
                        self.trades_subscriptions.discard(exchange_symbol)
                    elif data_type == DataType.QUOTES:
                        self.ticker_subscriptions.discard(exchange_symbol)
                    elif data_type == DataType.ORDER_BOOK:
                        self.orderbook_subscriptions.discard(exchange_symbol)
                    elif data_type == DataType.CANDLES:
                        self.ohlcv_subscriptions.discard(exchange_symbol)
            
            return True
            
        except Exception as e:
            self.logger.error("CCXT unsubscription failed", error=str(e))
            return False
    
    async def _trades_loop(self, symbol: str) -> None:
        """Watch trades for a specific symbol."""
        while symbol in self.trades_subscriptions and self.status == FeedStatus.CONNECTED:
            try:
                trades = await self.exchange_instance.watch_trades(symbol)
                
                for trade in trades:
                    tick = self._convert_trade_to_tick(symbol, trade)
                    if tick:
                        await self._handle_message(tick.__dict__)
                        
            except Exception as e:
                self.logger.error("Trades loop error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _ticker_loop(self, symbol: str) -> None:
        """Watch ticker/quote updates for a specific symbol."""
        while symbol in self.ticker_subscriptions and self.status == FeedStatus.CONNECTED:
            try:
                ticker = await self.exchange_instance.watch_ticker(symbol)
                
                quote = self._convert_ticker_to_quote(symbol, ticker)
                if quote:
                    await self._handle_message(quote.__dict__)
                    
            except Exception as e:
                self.logger.error("Ticker loop error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _orderbook_loop(self, symbol: str) -> None:
        """Watch order book updates for a specific symbol."""
        while symbol in self.orderbook_subscriptions and self.status == FeedStatus.CONNECTED:
            try:
                orderbook = await self.exchange_instance.watch_order_book(symbol)
                
                ob = self._convert_orderbook(symbol, orderbook)
                if ob:
                    await self._handle_message(ob.__dict__)
                    
            except Exception as e:
                self.logger.error("Order book loop error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _ohlcv_loop(self, symbol: str, timeframe: str) -> None:
        """Watch OHLCV candles for a specific symbol."""
        while symbol in self.ohlcv_subscriptions and self.status == FeedStatus.CONNECTED:
            try:
                ohlcv = await self.exchange_instance.watch_ohlcv(symbol, timeframe)
                
                if ohlcv:
                    # Convert latest candle
                    latest_candle = ohlcv[-1]
                    candle = self._convert_ohlcv_to_candle(symbol, latest_candle, timeframe)
                    if candle:
                        await self._handle_message(candle.__dict__)
                        
            except Exception as e:
                self.logger.error("OHLCV loop error", symbol=symbol, error=str(e))
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Dict[str, Any]) -> Optional[Any]:
        """Process message - already converted in the loops."""
        # Messages are already converted in the specific loops
        # This method handles the converted objects
        return message
    
    def _convert_trade_to_tick(self, symbol: str, trade: Dict[str, Any]) -> Optional[Tick]:
        """Convert CCXT trade to Tick object."""
        try:
            return Tick(
                timestamp=trade.get('timestamp', time.time() * 1000) / 1000,
                symbol=symbol.replace('/', '-'),
                exchange=self.exchange,
                price=Decimal(str(trade.get('price', 0))),
                size=Decimal(str(trade.get('amount', 0))),
                side=Side.BUY if trade.get('side') == 'buy' else Side.SELL,
                trade_id=str(trade.get('id', '')),
                conditions=None
            )
        except Exception as e:
            self.logger.error("Trade conversion failed", error=str(e), trade=trade)
            return None
    
    def _convert_ticker_to_quote(self, symbol: str, ticker: Dict[str, Any]) -> Optional[Quote]:
        """Convert CCXT ticker to Quote object."""
        try:
            return Quote(
                timestamp=ticker.get('timestamp', time.time() * 1000) / 1000,
                symbol=symbol.replace('/', '-'),
                exchange=self.exchange,
                bid_price=Decimal(str(ticker.get('bid', 0))),
                bid_size=Decimal(str(ticker.get('bidVolume', 0))),
                ask_price=Decimal(str(ticker.get('ask', 0))),
                ask_size=Decimal(str(ticker.get('askVolume', 0)))
            )
        except Exception as e:
            self.logger.error("Ticker conversion failed", error=str(e), ticker=ticker)
            return None
    
    def _convert_orderbook(self, symbol: str, orderbook: Dict[str, Any]) -> Optional[OrderBook]:
        """Convert CCXT order book to OrderBook object."""
        try:
            bids = [
                OrderBookLevel(price=Decimal(str(bid[0])), size=Decimal(str(bid[1])))
                for bid in orderbook.get('bids', [])[:10]  # Top 10 levels
            ]
            
            asks = [
                OrderBookLevel(price=Decimal(str(ask[0])), size=Decimal(str(ask[1])))
                for ask in orderbook.get('asks', [])[:10]  # Top 10 levels
            ]
            
            return OrderBook(
                timestamp=orderbook.get('timestamp', time.time() * 1000) / 1000,
                symbol=symbol.replace('/', '-'),
                exchange=self.exchange,
                bids=bids,
                asks=asks,
                sequence_number=orderbook.get('nonce')
            )
        except Exception as e:
            self.logger.error("Order book conversion failed", error=str(e))
            return None
    
    def _convert_ohlcv_to_candle(self, symbol: str, ohlcv: List, timeframe: str) -> Optional[Candle]:
        """Convert CCXT OHLCV array to Candle object."""
        try:
            if len(ohlcv) < 6:
                return None
            
            return Candle(
                timestamp=ohlcv[0] / 1000,  # Convert ms to seconds
                symbol=symbol.replace('/', '-'),
                exchange=self.exchange,
                timeframe=timeframe,
                open_price=Decimal(str(ohlcv[1])),
                high_price=Decimal(str(ohlcv[2])),
                low_price=Decimal(str(ohlcv[3])),
                close_price=Decimal(str(ohlcv[4])),
                volume=Decimal(str(ohlcv[5])) if ohlcv[5] else Decimal('0')
            )
        except Exception as e:
            self.logger.error("OHLCV conversion failed", error=str(e), ohlcv=ohlcv)
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = '1m',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """Fetch historical OHLCV data."""
        if not self.exchange_instance:
            raise ConnectionError("Exchange not connected")
        
        try:
            exchange_symbol = self._normalize_symbol(symbol)
            ohlcv_data = await self.exchange_instance.fetch_ohlcv(
                exchange_symbol, 
                timeframe, 
                since, 
                limit
            )
            
            candles = []
            for ohlcv in ohlcv_data:
                candle = self._convert_ohlcv_to_candle(exchange_symbol, ohlcv, timeframe)
                if candle:
                    candles.append(candle)
            
            self.logger.info(
                "Historical candles fetched",
                symbol=symbol,
                timeframe=timeframe,
                count=len(candles)
            )
            
            return candles
            
        except Exception as e:
            self.logger.error("Historical candles fetch failed", error=str(e))
            raise
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and capabilities."""
        if not self.exchange_instance:
            raise ConnectionError("Exchange not connected")
        
        return {
            'id': self.exchange_instance.id,
            'name': self.exchange_instance.name,
            'countries': getattr(self.exchange_instance, 'countries', []),
            'rateLimit': self.exchange_instance.rateLimit,
            'has': self.exchange_instance.has,
            'timeframes': getattr(self.exchange_instance, 'timeframes', {}),
            'markets_count': len(self.markets),
            'sandbox': self.config.get('sandbox', True)
        }


# Factory functions for different exchanges
def create_binance_feed(api_key: str = "", secret: str = "", sandbox: bool = True, **kwargs) -> CCXTFeed:
    """Create Binance CCXT feed."""
    config = CCXTConfig(
        exchange_id="binance",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
        **kwargs
    )
    return CCXTFeed(config)


def create_coinbase_feed(api_key: str = "", secret: str = "", password: str = "", sandbox: bool = True, **kwargs) -> CCXTFeed:
    """Create Coinbase Pro CCXT feed."""
    config = CCXTConfig(
        exchange_id="coinbasepro",
        api_key=api_key,
        secret=secret,
        password=password,
        sandbox=sandbox,
        **kwargs
    )
    return CCXTFeed(config)


def create_kraken_feed(api_key: str = "", secret: str = "", **kwargs) -> CCXTFeed:
    """Create Kraken CCXT feed."""
    config = CCXTConfig(
        exchange_id="kraken",
        api_key=api_key,
        secret=secret,
        sandbox=False,  # Kraken doesn't have sandbox
        **kwargs
    )
    return CCXTFeed(config)


# Example usage and testing
async def test_ccxt_feeds():
    """Test CCXT feeds with multiple exchanges."""
    
    # Test Binance
    binance_feed = create_binance_feed(sandbox=True)
    
    try:
        await binance_feed.start()
        
        if binance_feed.is_connected():
            print("✓ Binance connection successful")
            
            # Test subscription
            success = await binance_feed.subscribe(["BTC-USDT"], [DataType.TICKS, DataType.ORDER_BOOK])
            if success:
                print("✓ Binance subscription successful")
                
                # Listen for data
                await asyncio.sleep(5)
                
                stats = binance_feed.get_stats()
                print(f"✓ Binance stats: {stats.ticks_received} ticks")
        
    except Exception as e:
        print(f"✗ Binance test failed: {e}")
    finally:
        await binance_feed.stop()


if __name__ == "__main__":
    asyncio.run(test_ccxt_feeds())