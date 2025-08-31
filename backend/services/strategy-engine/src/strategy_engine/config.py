"""
Configuration management for strategy engine service.
Handles trading parameters, risk limits, ML settings, and backtesting configuration.
"""

import os
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Union

from pydantic import Field, HttpUrl, PostgresDsn, RedisDsn, validator
from pydantic_settings import BaseSettings


class TradingMode(str, Enum):
    """Trading execution modes with safety controls."""
    TEST = "test"        # Paper trading with mock data
    SHADOW = "shadow"    # Live data, no real trades, validation only
    LIVE = "live"        # Real trading with actual money


class StrategyType(str, Enum):
    """Supported strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    ML_BASED = "ml_based"
    MULTI_FACTOR = "multi_factor"
    OPTIONS_STRATEGY = "options_strategy"


class RiskModel(str, Enum):
    """Risk management models."""
    KELLY = "kelly"
    FIXED_FRACTIONAL = "fixed_fractional"
    VAR_BASED = "var_based"
    VOLATILITY_TARGET = "volatility_target"
    MAX_DRAWDOWN = "max_drawdown"


class BacktestEngine(str, Enum):
    """Backtesting engine options."""
    VECTORIZED = "vectorized"      # Fast pandas-based backtesting
    EVENT_DRIVEN = "event_driven"  # Realistic order-by-order simulation
    MONTE_CARLO = "monte_carlo"    # Statistical scenario analysis


class FeatureSet(str, Enum):
    """Feature engineering sets."""
    TECHNICAL = "technical"        # TA-Lib indicators
    FUNDAMENTAL = "fundamental"    # Financial ratios
    MACRO = "macro"               # Economic indicators
    SENTIMENT = "sentiment"       # News/social sentiment
    MICROSTRUCTURE = "microstructure"  # Order book features
    ALL = "all"


class StrategyEngineConfig(BaseSettings):
    """Configuration settings for strategy engine service."""
    
    # Service identification
    service_name: str = "strategy-engine"
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
        default="redis://localhost:6379/1", 
        env="REDIS_URL"
    )
    
    # Database pool settings
    db_pool_min_size: int = Field(default=5, env="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=20, env="DB_POOL_MAX_SIZE")
    db_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # Message queues
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_group_id: str = Field(default="strategy_engine", env="KAFKA_GROUP_ID")
    
    # Strategy configuration
    active_strategies: List[str] = Field(
        default=["momentum_v1", "mean_reversion_v1"], 
        env="ACTIVE_STRATEGIES"
    )
    strategy_config_path: str = Field(
        default="./configs/strategies/", 
        env="STRATEGY_CONFIG_PATH"
    )
    
    # Universe definition
    symbols: Set[str] = Field(
        default={"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC-USD", "ETH-USD"},
        env="SYMBOLS"
    )
    asset_classes: Set[str] = Field(
        default={"equity", "crypto", "forex"},
        env="ASSET_CLASSES"
    )
    
    # Risk management parameters
    max_portfolio_leverage: float = Field(default=2.0, env="MAX_PORTFOLIO_LEVERAGE")
    max_position_size_pct: float = Field(default=10.0, env="MAX_POSITION_SIZE_PCT")
    max_daily_loss_pct: float = Field(default=5.0, env="MAX_DAILY_LOSS_PCT")
    max_drawdown_pct: float = Field(default=15.0, env="MAX_DRAWDOWN_PCT")
    var_confidence_level: float = Field(default=0.95, env="VAR_CONFIDENCE_LEVEL")
    
    # Position sizing
    risk_model: RiskModel = Field(default=RiskModel.KELLY, env="RISK_MODEL")
    kelly_fractional: float = Field(default=0.25, env="KELLY_FRACTIONAL")
    volatility_target: float = Field(default=0.15, env="VOLATILITY_TARGET")
    
    # Signal processing
    signal_decay_half_life: int = Field(default=300, env="SIGNAL_DECAY_HALF_LIFE")  # seconds
    min_signal_strength: float = Field(default=0.1, env="MIN_SIGNAL_STRENGTH")
    max_positions_per_strategy: int = Field(default=20, env="MAX_POSITIONS_PER_STRATEGY")
    
    # Backtesting configuration
    backtest_engine: BacktestEngine = Field(default=BacktestEngine.EVENT_DRIVEN, env="BACKTEST_ENGINE")
    backtest_start_date: str = Field(default="2020-01-01", env="BACKTEST_START_DATE")
    backtest_end_date: str = Field(default="2024-01-01", env="BACKTEST_END_DATE")
    initial_capital: float = Field(default=1000000.0, env="INITIAL_CAPITAL")
    
    # Transaction costs and market impact
    commission_rate: float = Field(default=0.0001, env="COMMISSION_RATE")  # 1 bp
    spread_cost: float = Field(default=0.0002, env="SPREAD_COST")  # 2 bp
    market_impact_model: str = Field(default="linear", env="MARKET_IMPACT_MODEL")
    slippage_model: str = Field(default="constant", env="SLIPPAGE_MODEL")
    
    # Feature engineering
    feature_sets: List[FeatureSet] = Field(
        default=[FeatureSet.TECHNICAL, FeatureSet.SENTIMENT],
        env="FEATURE_SETS"
    )
    lookback_periods: List[int] = Field(
        default=[5, 10, 20, 50, 100, 200],
        env="LOOKBACK_PERIODS"
    )
    technical_indicators: List[str] = Field(
        default=["sma", "ema", "rsi", "macd", "bollinger", "atr", "vwap"],
        env="TECHNICAL_INDICATORS"
    )
    
    # ML Configuration
    enable_ml_strategies: bool = Field(default=True, env="ENABLE_ML_STRATEGIES")
    ml_model_path: str = Field(default="./models/", env="ML_MODEL_PATH")
    feature_store_path: str = Field(default="./features/", env="FEATURE_STORE_PATH")
    ml_prediction_horizon: int = Field(default=60, env="ML_PREDICTION_HORIZON")  # minutes
    
    # Model training
    retrain_frequency: str = Field(default="daily", env="RETRAIN_FREQUENCY")  # daily, weekly, monthly
    validation_split: float = Field(default=0.2, env="VALIDATION_SPLIT")
    early_stopping_patience: int = Field(default=10, env="EARLY_STOPPING_PATIENCE")
    max_training_samples: int = Field(default=100000, env="MAX_TRAINING_SAMPLES")
    
    # Model selection and ensemble
    model_types: List[str] = Field(
        default=["lightgbm", "xgboost", "lstm"],
        env="MODEL_TYPES"
    )
    ensemble_method: str = Field(default="weighted_average", env="ENSEMBLE_METHOD")
    model_selection_metric: str = Field(default="sharpe_ratio", env="MODEL_SELECTION_METRIC")
    
    # Performance tracking
    benchmark_symbols: List[str] = Field(
        default=["SPY", "QQQ", "BTC-USD"],
        env="BENCHMARK_SYMBOLS"
    )
    performance_attribution: bool = Field(default=True, env="PERFORMANCE_ATTRIBUTION")
    risk_attribution: bool = Field(default=True, env="RISK_ATTRIBUTION")
    
    # Real-time processing
    signal_processing_interval: int = Field(default=5, env="SIGNAL_PROCESSING_INTERVAL")  # seconds
    portfolio_update_interval: int = Field(default=10, env="PORTFOLIO_UPDATE_INTERVAL")
    risk_check_interval: int = Field(default=1, env="RISK_CHECK_INTERVAL")
    
    # Data sources
    market_data_sources: List[str] = Field(
        default=["data-ingest"], 
        env="MARKET_DATA_SOURCES"
    )
    fundamental_data_source: Optional[str] = Field(default=None, env="FUNDAMENTAL_DATA_SOURCE")
    news_data_source: Optional[str] = Field(default=None, env="NEWS_DATA_SOURCE")
    
    # Caching and performance
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    enable_feature_caching: bool = Field(default=True, env="ENABLE_FEATURE_CACHING")
    parallel_processing: bool = Field(default=True, env="PARALLEL_PROCESSING")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    
    # Monitoring and alerts
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9091, env="METRICS_PORT")
    alert_on_drawdown_pct: float = Field(default=5.0, env="ALERT_ON_DRAWDOWN_PCT")
    alert_on_var_breach: bool = Field(default=True, env="ALERT_ON_VAR_BREACH")
    
    # Strategy-specific parameters
    momentum_lookback: int = Field(default=20, env="MOMENTUM_LOOKBACK")
    momentum_threshold: float = Field(default=0.02, env="MOMENTUM_THRESHOLD")
    
    mean_reversion_zscore_threshold: float = Field(default=2.0, env="MEAN_REVERSION_ZSCORE_THRESHOLD")
    mean_reversion_lookback: int = Field(default=50, env="MEAN_REVERSION_LOOKBACK")
    
    pairs_correlation_threshold: float = Field(default=0.8, env="PAIRS_CORRELATION_THRESHOLD")
    pairs_zscore_threshold: float = Field(default=2.0, env="PAIRS_ZSCORE_THRESHOLD")
    
    # Options strategies (if applicable)
    enable_options_strategies: bool = Field(default=False, env="ENABLE_OPTIONS_STRATEGIES")
    max_options_delta: float = Field(default=0.3, env="MAX_OPTIONS_DELTA")
    min_options_days_to_expiry: int = Field(default=30, env="MIN_OPTIONS_DAYS_TO_EXPIRY")
    
    # Execution settings
    execution_service_url: HttpUrl = Field(
        default="http://localhost:8003", 
        env="EXECUTION_SERVICE_URL"
    )
    order_timeout_seconds: int = Field(default=30, env="ORDER_TIMEOUT_SECONDS")
    max_order_retries: int = Field(default=3, env="MAX_ORDER_RETRIES")
    
    # Development and testing
    enable_debug_mode: bool = Field(default=False, env="ENABLE_DEBUG_MODE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    save_debug_data: bool = Field(default=False, env="SAVE_DEBUG_DATA")
    
    # Research and experimentation
    enable_research_mode: bool = Field(default=False, env="ENABLE_RESEARCH_MODE")
    experiment_tracking: bool = Field(default=True, env="EXPERIMENT_TRACKING")
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns.db", env="MLFLOW_TRACKING_URI")
    
    @validator("symbols", pre=True)
    def parse_symbols(cls, v):
        """Parse comma-separated symbol list from environment."""
        if isinstance(v, str):
            return set(s.strip().upper() for s in v.split(",") if s.strip())
        return v
    
    @validator("active_strategies", pre=True)
    def parse_strategies(cls, v):
        """Parse comma-separated strategy list."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v
    
    @validator("feature_sets", pre=True)
    def parse_feature_sets(cls, v):
        """Parse comma-separated feature sets."""
        if isinstance(v, str):
            feature_strings = [s.strip() for s in v.split(",") if s.strip()]
            return [FeatureSet(fs) for fs in feature_strings]
        return v
    
    @validator("lookback_periods", pre=True)
    def parse_lookback_periods(cls, v):
        """Parse comma-separated lookback periods."""
        if isinstance(v, str):
            return [int(period.strip()) for period in v.split(",") if period.strip()]
        return v
    
    @validator("technical_indicators", pre=True)
    def parse_technical_indicators(cls, v):
        """Parse comma-separated technical indicators."""
        if isinstance(v, str):
            return [ind.strip().lower() for ind in v.split(",") if ind.strip()]
        return v
    
    @validator("trading_mode")
    def validate_trading_mode(cls, v):
        """Validate trading mode and issue warnings."""
        if v == TradingMode.LIVE:
            import logging
            logging.getLogger(__name__).warning(
                "LIVE TRADING MODE ENABLED - Real money at risk!"
            )
        return v
    
    @validator("max_portfolio_leverage")
    def validate_leverage(cls, v):
        """Validate leverage limits."""
        if v > 5.0:
            raise ValueError("Portfolio leverage cannot exceed 5.0x")
        return v
    
    @validator("max_position_size_pct")
    def validate_position_size(cls, v):
        """Validate position size limits."""
        if not 0 < v <= 50:
            raise ValueError("Position size must be between 0 and 50%")
        return v
    
    @validator("var_confidence_level")
    def validate_var_level(cls, v):
        """Validate VaR confidence level."""
        if not 0.8 <= v <= 0.99:
            raise ValueError("VaR confidence level must be between 0.8 and 0.99")
        return v
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get configuration for specific strategy."""
        strategy_configs = {
            "momentum_v1": {
                "lookback": self.momentum_lookback,
                "threshold": self.momentum_threshold,
                "max_positions": self.max_positions_per_strategy
            },
            "mean_reversion_v1": {
                "zscore_threshold": self.mean_reversion_zscore_threshold,
                "lookback": self.mean_reversion_lookback,
                "max_positions": self.max_positions_per_strategy
            },
            "pairs_trading_v1": {
                "correlation_threshold": self.pairs_correlation_threshold,
                "zscore_threshold": self.pairs_zscore_threshold,
                "max_positions": self.max_positions_per_strategy // 2  # Pairs use 2 positions
            }
        }
        
        return strategy_configs.get(strategy_name, {})
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get comprehensive risk limits dictionary."""
        return {
            "max_portfolio_leverage": self.max_portfolio_leverage,
            "max_position_size_pct": self.max_position_size_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "var_confidence_level": self.var_confidence_level,
            "alert_on_drawdown_pct": self.alert_on_drawdown_pct
        }
    
    def get_backtest_config(self) -> Dict:
        """Get backtesting configuration."""
        return {
            "engine": self.backtest_engine.value,
            "start_date": self.backtest_start_date,
            "end_date": self.backtest_end_date,
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "spread_cost": self.spread_cost,
            "market_impact_model": self.market_impact_model,
            "slippage_model": self.slippage_model
        }
    
    def get_ml_config(self) -> Dict:
        """Get ML configuration."""
        return {
            "enabled": self.enable_ml_strategies,
            "model_path": self.ml_model_path,
            "feature_store_path": self.feature_store_path,
            "prediction_horizon": self.ml_prediction_horizon,
            "retrain_frequency": self.retrain_frequency,
            "validation_split": self.validation_split,
            "model_types": self.model_types,
            "ensemble_method": self.ensemble_method,
            "selection_metric": self.model_selection_metric
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.trading_mode == TradingMode.LIVE
    
    def get_symbol_universe(self, asset_class: Optional[str] = None) -> Set[str]:
        """Get symbol universe filtered by asset class."""
        if not asset_class:
            return self.symbols
        
        if asset_class == "equity":
            return {s for s in self.symbols if not ("-" in s or "USD" in s)}
        elif asset_class == "crypto":
            return {s for s in self.symbols if "-USD" in s or "USD" in s}
        elif asset_class == "forex":
            return {s for s in self.symbols if "/" in s}
        else:
            return set()
    
    class Config:
        """Pydantic model configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Cache configuration instance for performance
@lru_cache()
def get_config() -> StrategyEngineConfig:
    """Get cached configuration instance."""
    return StrategyEngineConfig()


# Environment-specific configuration profiles
class DevelopmentConfig(StrategyEngineConfig):
    """Development environment configuration."""
    environment: str = "development"
    log_level: str = "DEBUG"
    reload: bool = True
    enable_debug_mode: bool = True
    enable_research_mode: bool = True
    save_debug_data: bool = True


class ProductionConfig(StrategyEngineConfig):
    """Production environment configuration."""
    environment: str = "production"
    log_level: str = "INFO"
    reload: bool = False
    enable_debug_mode: bool = False
    enable_research_mode: bool = False
    save_debug_data: bool = False
    workers: int = 4
    parallel_processing: bool = True


class TestConfig(StrategyEngineConfig):
    """Test environment configuration."""
    environment: str = "test"
    log_level: str = "DEBUG"
    trading_mode: TradingMode = TradingMode.TEST
    postgres_url: str = "postgresql://test_user:test_pass@localhost:5432/test_db"
    redis_url: str = "redis://localhost:6379/15"
    initial_capital: float = 100000.0
    enable_ml_strategies: bool = False


def get_config_by_env(env: Optional[str] = None) -> StrategyEngineConfig:
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


# Validation functions for runtime checks
def validate_strategy_config(strategy_name: str, config_dict: Dict) -> bool:
    """Validate strategy-specific configuration."""
    required_fields = {
        "momentum_v1": ["lookback", "threshold"],
        "mean_reversion_v1": ["zscore_threshold", "lookback"],
        "pairs_trading_v1": ["correlation_threshold", "zscore_threshold"]
    }
    
    if strategy_name in required_fields:
        return all(field in config_dict for field in required_fields[strategy_name])
    
    return True


def validate_risk_parameters(config: StrategyEngineConfig) -> List[str]:
    """Validate risk management parameters and return any issues."""
    issues = []
    
    if config.max_portfolio_leverage > 3.0:
        issues.append(f"High portfolio leverage: {config.max_portfolio_leverage}x")
    
    if config.max_position_size_pct > 20.0:
        issues.append(f"High position size limit: {config.max_position_size_pct}%")
    
    if config.max_daily_loss_pct > 10.0:
        issues.append(f"High daily loss limit: {config.max_daily_loss_pct}%")
    
    if config.is_live_trading() and config.initial_capital < 10000:
        issues.append("Low initial capital for live trading")
    
    return issues