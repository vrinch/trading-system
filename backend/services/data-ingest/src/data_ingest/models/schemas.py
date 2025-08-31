"""
Pydantic data models for market data ingestion.
Provides type safety, validation, and serialization for all market data types.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
import numpy as np


class Side(str, Enum):
    """Order/trade side enumeration."""
    BUY = "B"
    SELL = "S"
    UNKNOWN = "U"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class AssetClass(str, Enum):
    """Asset class classification."""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"


class Exchange(str, Enum):
    """Supported exchanges."""
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    POLYGON = "POLYGON"
    MOCK = "MOCK"


class BaseMarketData(BaseModel):
    """Base class for all market data with common fields."""
    timestamp: datetime = Field(..., description="Event timestamp in UTC")
    symbol: str = Field(..., description="Trading symbol", min_length=1, max_length=32)
    exchange: Exchange = Field(..., description="Exchange identifier")
    
    @validator("timestamp", pre=True)
    def parse_timestamp(cls, v):
        """Convert various timestamp formats to datetime."""
        if isinstance(v, (int, float)):
            # Unix timestamp (seconds or milliseconds)
            if v > 1e10:  # Milliseconds
                v = v / 1000
            return datetime.fromtimestamp(v, tz=timezone.utc)
        elif isinstance(v, str):
            # ISO format string
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        elif isinstance(v, datetime):
            # Ensure UTC timezone
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v
    
    @validator("symbol")
    def normalize_symbol(cls, v):
        """Normalize symbol format."""
        return v.upper().strip()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
        use_enum_values = True


class Tick(BaseMarketData):
    """Individual trade tick data."""
    price: Decimal = Field(..., description="Trade price", gt=0)
    size: Decimal = Field(..., description="Trade quantity", gt=0)
    side: Optional[Side] = Field(None, description="Trade side")
    conditions: Optional[List[str]] = Field(None, description="Trade conditions")
    trade_id: Optional[str] = Field(None, description="Unique trade identifier")
    sequence_number: Optional[int] = Field(None, description="Sequence number for ordering")
    
    @validator("price", "size", pre=True)
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, str):
            return Decimal(v)
        return Decimal(str(v))


class Quote(BaseMarketData):
    """Bid/ask quote data."""
    bid_price: Decimal = Field(..., description="Best bid price", gt=0)
    bid_size: Decimal = Field(..., description="Best bid quantity", gt=0)
    ask_price: Decimal = Field(..., description="Best ask price", gt=0)
    ask_size: Decimal = Field(..., description="Best ask quantity", gt=0)
    spread: Optional[Decimal] = Field(None, description="Bid-ask spread")
    
    @validator("bid_price", "bid_size", "ask_price", "ask_size", pre=True)
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal."""
        return Decimal(str(v))
    
    @validator("spread", pre=True, always=True)
    def calculate_spread(cls, v, values):
        """Auto-calculate spread if not provided."""
        if v is None and "ask_price" in values and "bid_price" in values:
            return values["ask_price"] - values["bid_price"]
        return v


class Candle(BaseMarketData):
    """OHLCV candle data."""
    timeframe: str = Field(..., description="Timeframe (1m, 5m, 1h, 1d)")
    open_price: Decimal = Field(..., description="Opening price", gt=0)
    high_price: Decimal = Field(..., description="Highest price", gt=0)
    low_price: Decimal = Field(..., description="Lowest price", gt=0)
    close_price: Decimal = Field(..., description="Closing price", gt=0)
    volume: Decimal = Field(..., description="Trading volume", ge=0)
    trade_count: Optional[int] = Field(None, description="Number of trades", ge=0)
    vwap: Optional[Decimal] = Field(None, description="Volume weighted average price")
    
    @validator("open_price", "high_price", "low_price", "close_price", "volume", "vwap", pre=True)
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal."""
        if v is None:
            return v
        return Decimal(str(v))
    
    @validator("high_price")
    def validate_high_price(cls, v, values):
        """Ensure high >= open, close, low."""
        for price_field in ["open_price", "close_price", "low_price"]:
            if price_field in values and v < values[price_field]:
                raise ValueError(f"high_price must be >= {price_field}")
        return v
    
    @validator("low_price") 
    def validate_low_price(cls, v, values):
        """Ensure low <= open, close, high."""
        for price_field in ["open_price", "close_price", "high_price"]:
            if price_field in values and v > values[price_field]:
                raise ValueError(f"low_price must be <= {price_field}")
        return v


class OrderBookLevel(BaseModel):
    """Single order book level (price and size)."""
    price: Decimal = Field(..., description="Price level", gt=0)
    size: Decimal = Field(..., description="Quantity at price level", ge=0)
    
    @validator("price", "size", pre=True)
    def convert_to_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class OrderBook(BaseMarketData):
    """Level 2 order book snapshot."""
    bids: List[OrderBookLevel] = Field(..., description="Bid levels (highest first)")
    asks: List[OrderBookLevel] = Field(..., description="Ask levels (lowest first)")
    sequence_number: Optional[int] = Field(None, description="Sequence number")
    
    @validator("bids")
    def validate_bids_order(cls, v):
        """Ensure bids are sorted by price descending."""
        if len(v) > 1:
            prices = [level.price for level in v]
            if prices != sorted(prices, reverse=True):
                raise ValueError("Bids must be sorted by price descending")
        return v
    
    @validator("asks")
    def validate_asks_order(cls, v):
        """Ensure asks are sorted by price ascending."""
        if len(v) > 1:
            prices = [level.price for level in v]
            if prices != sorted(prices):
                raise ValueError("Asks must be sorted by price ascending")
        return v
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return None
    
    def get_midpoint(self) -> Optional[Decimal]:
        """Calculate midpoint price."""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return None


class NewsItem(BaseModel):
    """Financial news item for sentiment analysis."""
    id: str = Field(..., description="Unique news identifier")
    headline: str = Field(..., description="News headline")
    summary: Optional[str] = Field(None, description="News summary")
    content: Optional[str] = Field(None, description="Full news content")
    source: str = Field(..., description="News source")
    published_at: datetime = Field(..., description="Publication timestamp")
    symbols: List[str] = Field(default_factory=list, description="Related symbols")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score (-1 to 1)")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0 to 1)")
    
    @validator("sentiment_score")
    def validate_sentiment_score(cls, v):
        """Ensure sentiment score is in valid range."""
        if v is not None and not -1.0 <= v <= 1.0:
            raise ValueError("Sentiment score must be between -1 and 1")
        return v
    
    @validator("relevance_score")
    def validate_relevance_score(cls, v):
        """Ensure relevance score is in valid range."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Relevance score must be between 0 and 1")
        return v


class MarketStatus(BaseModel):
    """Market status information."""
    market: str = Field(..., description="Market identifier")
    status: str = Field(..., description="Market status (open/closed/pre/after)")
    next_open: Optional[datetime] = Field(None, description="Next market open time")
    next_close: Optional[datetime] = Field(None, description="Next market close time")
    timezone: str = Field(default="UTC", description="Market timezone")


class DataFeedStats(BaseModel):
    """Statistics for data feed monitoring."""
    feed_name: str = Field(..., description="Data feed identifier")
    symbol: str = Field(..., description="Symbol")
    exchange: Exchange = Field(..., description="Exchange")
    
    # Message counts
    ticks_received: int = Field(default=0, description="Total ticks received", ge=0)
    quotes_received: int = Field(default=0, description="Total quotes received", ge=0)
    candles_received: int = Field(default=0, description="Total candles received", ge=0)
    
    # Latency metrics
    avg_latency_ms: float = Field(default=0.0, description="Average latency in ms", ge=0)
    max_latency_ms: float = Field(default=0.0, description="Max latency in ms", ge=0)
    
    # Connection status
    connected: bool = Field(default=False, description="Connection status")
    last_message_time: Optional[datetime] = Field(None, description="Last message timestamp")
    reconnect_count: int = Field(default=0, description="Reconnection attempts", ge=0)
    
    # Error tracking
    error_count: int = Field(default=0, description="Total errors", ge=0)
    last_error: Optional[str] = Field(None, description="Last error message")
    last_error_time: Optional[datetime] = Field(None, description="Last error timestamp")


class ProcessingResult(BaseModel):
    """Result of data processing operation."""
    success: bool = Field(..., description="Processing success status")
    records_processed: int = Field(..., description="Number of records processed", ge=0)
    records_failed: int = Field(default=0, description="Number of failed records", ge=0)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0)
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        total = self.records_processed + self.records_failed
        if total == 0:
            return 0.0
        return (self.records_processed / total) * 100.0


# Union type for all market data types
MarketDataType = Union[Tick, Quote, Candle, OrderBook, NewsItem]


# Request/Response models for API endpoints
class SymbolRequest(BaseModel):
    """Request to subscribe/unsubscribe to symbol."""
    symbol: str = Field(..., description="Trading symbol")
    exchange: Exchange = Field(..., description="Exchange")
    data_types: List[str] = Field(..., description="Data types to subscribe")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")
    feeds_connected: int = Field(..., description="Number of connected feeds")
    total_feeds: int = Field(..., description="Total configured feeds")
    last_data_received: Optional[datetime] = Field(None, description="Last data timestamp")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    feeds: List[DataFeedStats] = Field(..., description="Feed statistics")
    total_messages: int = Field(..., description="Total messages processed")
    messages_per_second: float = Field(..., description="Current message rate")
    avg_processing_time_ms: float = Field(..., description="Average processing time")