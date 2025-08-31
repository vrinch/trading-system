"""
Configuration management for data ingestion service.
Handles environment variables, validation, and service settings.
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set

from pydantic import Field, HttpUrl, PostgresDsn, RedisDsn, validator
from pydantic_settings import BaseSettings


class TradingMode(str, Enum):
    """Trading execution modes with safety controls."""
    TEST = "test"        # Paper trading with mock data
    SHADOW = "shadow"    # Live data, no real trades
    LIVE = "live"        # Real trading with actual money


class LogLevel(str, Enum):
    """Logging levels for service configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataIngestConfig(BaseSettings):
    """Configuration settings for data ingestion service."""
    
    # Service identification
    service_name: str = "data-ingest"
    service_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    trading_mode: TradingMode = Field(default=TradingMode.TEST, env="TRADING_MODE")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Database connections
    postgres_url: PostgresDsn = Field(
        default="postgresql://trading_user:trading_pass@localhost:5432/trading_db",
        env="POSTGRES_URL"
    )
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0", 
        env="REDIS_URL"
    )
    
    # Database pool settings for high performance
    db_pool_min_size: int = Field(default=5, env="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=20, env="DB_POOL_MAX_SIZE")
    db_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # Kafka configuration
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_group_id: str = Field(default="data_ingest", env="KAFKA_GROUP_ID")
    kafka_batch_size: int = Field(default=16384, env="KAFKA_BATCH_SIZE")
    kafka_compression: str = Field(default="snappy", env="KAFKA_COMPRESSION")
    
    # Market data provider configurations
    polygon_api_key: str = Field(default="demo", env="POLYGON_API_KEY")
    polygon_base_url: str = Field(
        default="https://api.polygon.io", 
        env="POLYGON_BASE_URL"
    )
    polygon_websocket_url: str = Field(
        default="wss://socket.polygon.io", 
        env="POLYGON_WEBSOCKET_URL"
    )
    
    # CCXT exchange configurations
    ccxt_exchanges: List[str] = Field(
        default=["binance", "coinbase", "kraken"], 
        env="CCXT_EXCHANGES"
    )
    ccxt_sandbox: bool = Field(default=True, env="CCXT_SANDBOX")
    
    # Rate limiting configuration
    rate_limit_requests_per_second: int = Field(default=100, env="RATE_LIMIT_RPS")
    rate_limit_burst: int = Field(default=200, env="RATE_LIMIT_BURST")
    max_concurrent_requests: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")
    
    # Data processing settings
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    flush_interval_seconds: int = Field(default=1, env="FLUSH_INTERVAL_SECONDS")
    max_queue_size: int = Field(default=10000, env="MAX_QUEUE_SIZE")
    
    # Symbols and exchanges to monitor
    symbols: Set[str] = Field(
        default={"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC-USD", "ETH-USD"},
        env="SYMBOLS"
    )
    exchanges: Set[str] = Field(
        default={"NASDAQ", "NYSE", "CRYPTO"},
        env="EXCHANGES"
    )
    
    # Data retention and cleanup
    data_retention_days: int = Field(default=30, env="DATA_RETENTION_DAYS")
    cleanup_interval_hours: int = Field(default=24, env="CLEANUP_INTERVAL_HOURS")
    
    # Monitoring and observability
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")
    
    # Security settings
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Performance tuning
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    
    # Circuit breaker settings for fault tolerance
    circuit_breaker_failure_threshold: int = Field(default=5, env="CB_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=60, env="CB_RECOVERY_TIMEOUT")
    circuit_breaker_expected_exception: str = Field(default="Exception", env="CB_EXPECTED_EXCEPTION")
    
    # Retry configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay_seconds: float = Field(default=1.0, env="RETRY_DELAY_SECONDS")
    retry_exponential_base: float = Field(default=2.0, env="RETRY_EXPONENTIAL_BASE")
    
    # Webhook configurations for real-time alerts
    webhook_url: Optional[HttpUrl] = Field(default=None, env="WEBHOOK_URL")
    webhook_timeout: int = Field(default=10, env="WEBHOOK_TIMEOUT")
    
    # Development and testing flags
    enable_mock_feeds: bool = Field(default=False, env="ENABLE_MOCK_FEEDS")
    mock_data_file: Optional[str] = Field(default=None, env="MOCK_DATA_FILE")
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")
    
    @validator("symbols", pre=True)
    def parse_symbols(cls, v):
        """Parse comma-separated symbol list from environment."""
        if isinstance(v, str):
            return set(s.strip().upper() for s in v.split(",") if s.strip())
        return v
    
    @validator("exchanges", pre=True) 
    def parse_exchanges(cls, v):
        """Parse comma-separated exchange list from environment."""
        if isinstance(v, str):
            return set(e.strip().upper() for e in v.split(",") if e.strip())
        return v
    
    @validator("ccxt_exchanges", pre=True)
    def parse_ccxt_exchanges(cls, v):
        """Parse comma-separated CCXT exchange list."""
        if isinstance(v, str):
            return [e.strip().lower() for e in v.split(",") if e.strip()]
        return v
    
    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @validator("trading_mode")
    def validate_trading_mode(cls, v):
        """Ensure trading mode is valid and log warning for live mode."""
        if v == TradingMode.LIVE:
            import logging
            logging.getLogger(__name__).warning(
                "LIVE TRADING MODE ENABLED - Real money at risk!"
            )
        return v
    
    def get_database_url(self) -> str:
        """Get formatted database URL for SQLAlchemy."""
        return str(self.postgres_url)
    
    def get_redis_url(self) -> str:
        """Get formatted Redis URL."""
        return str(self.redis_url)
    
    def get_kafka_config(self) -> Dict[str, str]:
        """Get Kafka configuration dictionary."""
        return {
            "bootstrap_servers": self.kafka_bootstrap_servers,
            "group_id": self.kafka_group_id,
            "compression_type": self.kafka_compression,
            "batch_size": str(self.kafka_batch_size),
        }
    
    def get_exchange_symbols(self, exchange: str) -> Set[str]:
        """Get symbols for specific exchange with filtering."""
        if exchange.upper() == "CRYPTO":
            return {s for s in self.symbols if "-USD" in s or "USDT" in s}
        else:
            return {s for s in self.symbols if "-" not in s}
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.trading_mode == TradingMode.LIVE
    
    class Config:
        """Pydantic model configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Custom validation for environment variables
        @classmethod
        def prepare_field(cls, field):
            if 'env' in field.field_info.extra:
                field.field_info.extra['env'] = field.field_info.extra['env'].upper()
            return field


# Cache configuration instance for performance
@lru_cache()
def get_config() -> DataIngestConfig:
    """Get cached configuration instance."""
    return DataIngestConfig()


# Environment-specific configuration profiles
class DevelopmentConfig(DataIngestConfig):
    """Development environment configuration."""
    environment: str = "development"
    log_level: LogLevel = LogLevel.DEBUG
    reload: bool = True
    enable_debug_endpoints: bool = True
    enable_mock_feeds: bool = True


class ProductionConfig(DataIngestConfig):
    """Production environment configuration."""
    environment: str = "production"
    log_level: LogLevel = LogLevel.INFO
    reload: bool = False
    enable_debug_endpoints: bool = False
    enable_mock_feeds: bool = False
    workers: int = 4


class TestConfig(DataIngestConfig):
    """Test environment configuration."""
    environment: str = "test"
    log_level: LogLevel = LogLevel.DEBUG
    trading_mode: TradingMode = TradingMode.TEST
    enable_mock_feeds: bool = True
    postgres_url: str = "postgresql://test_user:test_pass@localhost:5432/test_db"
    redis_url: str = "redis://localhost:6379/15"  # Use test database


def get_config_by_env(env: Optional[str] = None) -> DataIngestConfig:
    """Get configuration based on environment."""
    env = env or os.getenv("ENVIRONMENT", "development")
    
    if env.lower() in ("production", "prod"):
        return ProductionConfig()
    elif env.lower() == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()


# Global configuration instance
config = get_config()