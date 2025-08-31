"""
Base strategy interface and abstract classes.
Provides framework for implementing trading strategies with consistent API.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np
from decimal import Decimal

from ..models.schemas import Signal, Position, SignalType, Side, AssetClass
from ..config import StrategyEngineConfig
from ...shared.utils.logging import get_logger, LogEvents, AuditLogger
from ...shared.utils.metrics import TradingMetrics


class StrategyState(str, Enum):
    """Strategy execution states."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class StrategyMetrics:
    """Strategy performance metrics tracking."""
    
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.signals_generated = 0
        self.signals_executed = 0
        self.positions_opened = 0
        self.positions_closed = 0
        self.total_pnl = Decimal('0')
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.start_time = time.time()
    
    def update_signal_metrics(self, signal: Signal, executed: bool = False):
        """Update signal-related metrics."""
        self.signals_generated += 1
        if executed:
            self.signals_executed += 1
    
    def update_position_metrics(self, position: Position):
        """Update position-related metrics."""
        if position.status == "OPEN":
            self.positions_opened += 1
        elif position.status == "CLOSED":
            self.positions_closed += 1
            self.total_pnl += position.realized_pnl
    
    def calculate_performance_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns_array) - 1
        annual_return = np.mean(returns_array) * 252  # Assuming daily returns
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate calculation
        winning_trades = len([r for r in returns if r > 0])
        win_rate = winning_trades / len(returns) if returns else 0
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(abs(max_drawdown)),
            'win_rate': float(win_rate),
            'total_trades': len(returns),
            'winning_trades': winning_trades
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime = time.time() - self.start_time
        
        return {
            'strategy_id': self.strategy_id,
            'uptime_seconds': uptime,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'execution_rate': (
                self.signals_executed / max(1, self.signals_generated)
            ),
            'positions_opened': self.positions_opened,
            'positions_closed': self.positions_closed,
            'total_pnl': float(self.total_pnl),
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(
        self,
        strategy_id: str,
        config: StrategyEngineConfig,
        metrics: Optional[TradingMetrics] = None
    ):
        self.strategy_id = strategy_id
        self.config = config
        self.metrics = metrics
        self.logger = get_logger(f"strategy_{strategy_id}")
        self.audit_logger = AuditLogger(f"strategy_{strategy_id}")
        
        # Strategy state
        self.state = StrategyState.INACTIVE
        self.universe = set()  # Symbols this strategy trades
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.strategy_metrics = StrategyMetrics(strategy_id)
        self.returns_history: List[float] = []
        
        # Internal state
        self.last_update_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None
        self.error_count = 0
        
        # Configuration
        self.max_positions = config.get_strategy_config(strategy_id).get(
            'max_positions', config.max_positions_per_strategy
        )
        
        # Risk management
        self.position_size_limit = config.max_position_size_pct / 100.0
        self.daily_loss_limit = config.max_daily_loss_pct / 100.0
        
        # Data storage
        self.market_data_buffer: Dict[str, pd.DataFrame] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata and description."""
        pass
    
    @abstractmethod
    async def initialize(self, universe: Set[str]) -> bool:
        """Initialize strategy with symbol universe."""
        pass
    
    @abstractmethod
    async def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame],
        current_time: datetime
    ) -> List[Signal]:
        """Generate trading signals based on market data."""
        pass
    
    @abstractmethod
    async def update_positions(self, positions: Dict[str, Position]) -> None:
        """Update strategy with current positions."""
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Any]:
        """Return required market data specifications."""
        pass
    
    async def start(self, universe: Set[str]) -> bool:
        """Start the strategy with given universe."""
        try:
            self.state = StrategyState.INITIALIZING
            self.logger.info("Initializing strategy", strategy_id=self.strategy_id)
            
            # Initialize strategy-specific components
            success = await self.initialize(universe)
            if not success:
                self.state = StrategyState.ERROR
                return False
            
            self.universe = universe
            self.state = StrategyState.ACTIVE
            
            self.audit_logger.log_order_event(
                "STRATEGY_STARTED",
                {"strategy_id": self.strategy_id, "universe": list(universe)}
            )
            
            self.logger.info(
                "Strategy started successfully",
                strategy_id=self.strategy_id,
                universe_size=len(universe)
            )
            
            return True
            
        except Exception as e:
            self.state = StrategyState.ERROR
            self.error_count += 1
            self.logger.error(
                "Strategy initialization failed",
                strategy_id=self.strategy_id,
                error=str(e)
            )
            return False
    
    async def stop(self) -> None:
        """Stop the strategy and cleanup resources."""
        self.logger.info("Stopping strategy", strategy_id=self.strategy_id)
        
        self.state = StrategyState.SHUTDOWN
        
        # Clear internal state
        self.market_data_buffer.clear()
        self.feature_cache.clear()
        
        self.audit_logger.log_order_event(
            "STRATEGY_STOPPED",
            {"strategy_id": self.strategy_id}
        )
        
        self.logger.info("Strategy stopped", strategy_id=self.strategy_id)
    
    async def pause(self) -> None:
        """Pause strategy execution."""
        if self.state == StrategyState.ACTIVE:
            self.state = StrategyState.PAUSED
            self.logger.info("Strategy paused", strategy_id=self.strategy_id)
    
    async def resume(self) -> None:
        """Resume strategy execution."""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.ACTIVE
            self.logger.info("Strategy resumed", strategy_id=self.strategy_id)
    
    async def process_market_data(
        self, 
        market_data: Dict[str, pd.DataFrame],
        current_time: datetime
    ) -> List[Signal]:
        """Main processing loop - generate signals from market data."""
        if self.state != StrategyState.ACTIVE:
            return []
        
        try:
            # Update internal market data buffer
            self._update_market_data_buffer(market_data)
            
            # Generate signals
            signals = await self.generate_signals(market_data, current_time)
            
            # Validate and filter signals
            validated_signals = self._validate_signals(signals)
            
            # Update metrics
            for signal in validated_signals:
                self.strategy_metrics.update_signal_metrics(signal)
                if self.metrics:
                    self.metrics.record_signal_generated(
                        self.strategy_id,
                        signal.symbol,
                        signal.side.value
                    )
            
            self.last_update_time = current_time
            if validated_signals:
                self.last_signal_time = current_time
            
            return validated_signals
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(
                "Signal generation failed",
                strategy_id=self.strategy_id,
                error=str(e)
            )
            
            if self.error_count > 5:  # Error threshold
                self.state = StrategyState.ERROR
            
            return []
    
    def _update_market_data_buffer(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update internal market data buffer with size limits."""
        max_buffer_size = 1000  # Maximum rows per symbol
        
        for symbol, data in market_data.items():
            if symbol in self.market_data_buffer:
                # Append new data and limit size
                combined = pd.concat([self.market_data_buffer[symbol], data])
                self.market_data_buffer[symbol] = combined.tail(max_buffer_size)
            else:
                self.market_data_buffer[symbol] = data.tail(max_buffer_size)
    
    def _validate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Validate signals against strategy constraints."""
        validated = []
        
        for signal in signals:
            # Check signal strength threshold
            if signal.strength < self.config.min_signal_strength:
                self.logger.debug(
                    "Signal below strength threshold",
                    symbol=signal.symbol,
                    strength=signal.strength
                )
                continue
            
            # Check if symbol is in universe
            if signal.symbol not in self.universe:
                self.logger.warning(
                    "Signal for symbol not in universe",
                    symbol=signal.symbol
                )
                continue
            
            # Check position limits
            if self._would_exceed_position_limits(signal):
                self.logger.warning(
                    "Signal would exceed position limits",
                    symbol=signal.symbol
                )
                continue
            
            # Check signal validity period
            if signal.valid_until and signal.valid_until < datetime.now():
                self.logger.debug(
                    "Signal expired",
                    symbol=signal.symbol,
                    valid_until=signal.valid_until
                )
                continue
            
            validated.append(signal)
        
        return validated
    
    def _would_exceed_position_limits(self, signal: Signal) -> bool:
        """Check if signal would cause position limit violations."""
        # Check maximum number of positions
        if len(self.positions) >= self.max_positions:
            # Only allow closing signals
            if signal.signal_type != SignalType.EXIT:
                return True
        
        # Check position size limits (placeholder - would need portfolio context)
        # This would typically be handled by the portfolio manager
        
        return False
    
    def calculate_position_size(
        self, 
        signal: Signal, 
        portfolio_value: float,
        volatility: Optional[float] = None
    ) -> Decimal:
        """Calculate appropriate position size based on risk model."""
        if self.config.risk_model == "fixed_fractional":
            return Decimal(str(portfolio_value * self.position_size_limit))
        
        elif self.config.risk_model == "kelly":
            # Simplified Kelly criterion
            if volatility and volatility > 0:
                kelly_fraction = min(
                    signal.strength / volatility,
                    self.config.kelly_fractional
                )
                return Decimal(str(portfolio_value * kelly_fraction))
        
        elif self.config.risk_model == "volatility_target":
            # Volatility targeting
            if volatility and volatility > 0:
                target_vol = self.config.volatility_target
                position_vol = target_vol / volatility
                return Decimal(str(portfolio_value * position_vol))
        
        # Default fallback
        return Decimal(str(portfolio_value * 0.02))  # 2% position size
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = self.strategy_metrics.get_summary()
        
        # Add performance calculations
        if self.returns_history:
            performance_metrics = self.strategy_metrics.calculate_performance_metrics(
                self.returns_history
            )
            base_metrics.update(performance_metrics)
        
        # Add current state information
        base_metrics.update({
            'state': self.state.value,
            'universe_size': len(self.universe),
            'active_positions': len(self.positions),
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'error_count': self.error_count
        })
        
        return base_metrics
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions managed by this strategy."""
        return self.positions.copy()
    
    def is_active(self) -> bool:
        """Check if strategy is active."""
        return self.state == StrategyState.ACTIVE
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get strategy health status."""
        is_healthy = (
            self.state == StrategyState.ACTIVE and
            self.error_count < 5 and
            (not self.last_update_time or 
             (datetime.now() - self.last_update_time).seconds < 300)
        )
        
        return {
            'strategy_id': self.strategy_id,
            'healthy': is_healthy,
            'state': self.state.value,
            'error_count': self.error_count,
            'last_update_age_seconds': (
                (datetime.now() - self.last_update_time).seconds 
                if self.last_update_time else None
            )
        }


class QuantitativeStrategy(BaseStrategy):
    """Base class for quantitative strategies using mathematical models."""
    
    def __init__(self, strategy_id: str, config: StrategyEngineConfig, **kwargs):
        super().__init__(strategy_id, config, **kwargs)
        
        # Quantitative parameters
        self.lookback_periods = config.lookback_periods
        self.rebalance_frequency = getattr(config, 'rebalance_frequency', 'daily')
        
        # Statistical thresholds
        self.significance_level = 0.05
        
        # Technical analysis indicators
        self.indicators_config = {
            'sma_periods': [10, 20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for quantitative analysis."""
        indicators = {}
        
        if 'close' not in data.columns:
            return indicators
        
        try:
            import talib as ta
            
            close_prices = data['close'].values
            high_prices = data.get('high', data['close']).values
            low_prices = data.get('low', data['close']).values
            volume = data.get('volume', pd.Series([0] * len(data))).values
            
            # Moving averages
            for period in self.indicators_config['sma_periods']:
                if len(close_prices) >= period:
                    indicators[f'sma_{period}'] = pd.Series(
                        ta.SMA(close_prices, timeperiod=period),
                        index=data.index
                    )
            
            for period in self.indicators_config['ema_periods']:
                if len(close_prices) >= period:
                    indicators[f'ema_{period}'] = pd.Series(
                        ta.EMA(close_prices, timeperiod=period),
                        index=data.index
                    )
            
            # RSI
            if len(close_prices) >= self.indicators_config['rsi_period']:
                indicators['rsi'] = pd.Series(
                    ta.RSI(close_prices, timeperiod=self.indicators_config['rsi_period']),
                    index=data.index
                )
            
            # Bollinger Bands
            if len(close_prices) >= self.indicators_config['bollinger_period']:
                bb_upper, bb_middle, bb_lower = ta.BBANDS(
                    close_prices,
                    timeperiod=self.indicators_config['bollinger_period'],
                    nbdevup=self.indicators_config['bollinger_std'],
                    nbdevdn=self.indicators_config['bollinger_std']
                )
                indicators['bb_upper'] = pd.Series(bb_upper, index=data.index)
                indicators['bb_middle'] = pd.Series(bb_middle, index=data.index)
                indicators['bb_lower'] = pd.Series(bb_lower, index=data.index)
            
            # MACD
            if len(close_prices) >= max(self.indicators_config['macd_fast'], self.indicators_config['macd_slow']):
                macd, macd_signal, macd_hist = ta.MACD(
                    close_prices,
                    fastperiod=self.indicators_config['macd_fast'],
                    slowperiod=self.indicators_config['macd_slow'],
                    signalperiod=self.indicators_config['macd_signal']
                )
                indicators['macd'] = pd.Series(macd, index=data.index)
                indicators['macd_signal'] = pd.Series(macd_signal, index=data.index)
                indicators['macd_histogram'] = pd.Series(macd_hist, index=data.index)
            
            # Volume indicators if volume data available
            if volume.sum() > 0:
                indicators['volume_sma'] = pd.Series(
                    ta.SMA(volume, timeperiod=20),
                    index=data.index
                )
                
        except ImportError:
            self.logger.warning("TA-Lib not available, using simple indicators")
            # Fallback to simple pandas calculations
            indicators.update(self._calculate_simple_indicators(data))
        
        return indicators
    
    def _calculate_simple_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate simple indicators using pandas when TA-Lib is not available."""
        indicators = {}
        
        if 'close' not in data.columns:
            return indicators
        
        # Simple moving averages
        for period in [10, 20, 50]:
            if len(data) >= period:
                indicators[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        # Simple RSI calculation
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        return indicators
    
    def calculate_statistical_features(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Calculate statistical features for quantitative analysis."""
        if len(data) < window:
            return {}
        
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close'].tail(window)
            
            # Basic statistics
            features['mean_return'] = close_prices.pct_change().mean()
            features['volatility'] = close_prices.pct_change().std()
            features['skewness'] = close_prices.pct_change().skew()
            features['kurtosis'] = close_prices.pct_change().kurtosis()
            
            # Price momentum
            features['momentum_5d'] = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) >= 6 else 0
            features['momentum_10d'] = (close_prices.iloc[-1] / close_prices.iloc[-11] - 1) if len(close_prices) >= 11 else 0
            
            # Volatility measures
            features['realized_vol'] = close_prices.pct_change().std() * np.sqrt(252)
            features['vol_of_vol'] = close_prices.pct_change().rolling(5).std().std() if len(close_prices) >= 5 else 0
        
        if 'volume' in data.columns:
            volume = data['volume'].tail(window)
            features['avg_volume'] = volume.mean()
            features['volume_trend'] = np.corrcoef(range(len(volume)), volume)[0, 1] if len(volume) > 1 else 0
        
        return features
    
    def perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on price data."""
        results = {}
        
        if len(data) < 30 or 'close' not in data.columns:
            return results
        
        returns = data['close'].pct_change().dropna()
        
        try:
            from scipy import stats
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            results['normality_test'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > self.significance_level
            }
            
            # Stationarity test (simplified)
            # In production, would use ADF test from statsmodels
            results['mean_reversion_score'] = float(1 - abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]))
            
        except ImportError:
            self.logger.warning("scipy not available for statistical tests")
        
        return results


class MLStrategy(BaseStrategy):
    """Base class for machine learning based strategies."""
    
    def __init__(self, strategy_id: str, config: StrategyEngineConfig, **kwargs):
        super().__init__(strategy_id, config, **kwargs)
        
        # ML configuration
        self.model_path = config.ml_model_path
        self.feature_store_path = config.feature_store_path
        self.prediction_horizon = config.ml_prediction_horizon
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.feature_pipeline = None
        self.last_prediction_time: Optional[datetime] = None
        
        # Feature configuration
        self.feature_sets = config.feature_sets
        self.lookback_for_features = max(config.lookback_periods)
    
    async def load_model(self, model_name: str, model_path: str) -> bool:
        """Load ML model from disk."""
        try:
            import joblib
            
            model = joblib.load(model_path)
            self.models[model_name] = model
            
            self.logger.info(
                "Model loaded successfully",
                model_name=model_name,
                model_path=model_path
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to load model",
                model_name=model_name,
                error=str(e)
            )
            return False
    
    async def generate_ml_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, float]:
        """Generate ML features for prediction."""
        features = {}
        
        if symbol not in market_data:
            return features
        
        data = market_data[symbol]
        
        if len(data) < self.lookback_for_features:
            return features
        
        try:
            # Technical features
            if 'technical' in [fs.value for fs in self.feature_sets]:
                tech_features = self._generate_technical_features(data)
                features.update(tech_features)
            
            # Statistical features
            stat_features = self._generate_statistical_features(data)
            features.update(stat_features)
            
            # Market microstructure features (if available)
            if 'microstructure' in [fs.value for fs in self.feature_sets]:
                micro_features = self._generate_microstructure_features(data)
                features.update(micro_features)
            
        except Exception as e:
            self.logger.error(
                "Feature generation failed",
                symbol=symbol,
                error=str(e)
            )
        
        return features
    
    def _generate_technical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate technical analysis features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        # Price-based features
        close = data['close']
        features['price_change_1d'] = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0
        features['price_change_5d'] = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
        
        # Moving average ratios
        if len(close) >= 20:
            ma_20 = close.rolling(20).mean()
            features['price_to_ma20'] = close.iloc[-1] / ma_20.iloc[-1] - 1
        
        if len(close) >= 50:
            ma_50 = close.rolling(50).mean()
            features['ma20_to_ma50'] = ma_20.iloc[-1] / ma_50.iloc[-1] - 1 if len(close) >= 20 else 0
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1]
            features['rsi_normalized'] = (rsi.iloc[-1] - 50) / 50  # Normalize around 50
        
        return features
    
    def _generate_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate statistical features."""
        features = {}
        
        if 'close' not in data.columns or len(data) < 20:
            return features
        
        returns = data['close'].pct_change().dropna()
        
        # Return statistics
        features['return_mean'] = returns.mean()
        features['return_std'] = returns.std()
        features['return_skew'] = returns.skew()
        features['return_kurt'] = returns.kurtosis()
        
        # Rolling statistics
        features['vol_5d'] = returns.tail(5).std()
        features['vol_20d'] = returns.tail(20).std()
        features['vol_ratio'] = features['vol_5d'] / features['vol_20d'] if features['vol_20d'] > 0 else 1
        
        # Autocorrelation
        if len(returns) > 1:
            features['autocorr_1'] = returns.autocorr(lag=1)
        
        return features
    
    def _generate_microstructure_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate market microstructure features."""
        features = {}
        
        # Bid-ask spread features (if available)
        if 'bid' in data.columns and 'ask' in data.columns:
            spread = data['ask'] - data['bid']
            mid_price = (data['bid'] + data['ask']) / 2
            
            features['spread_bps'] = (spread / mid_price * 10000).mean()  # Average spread in bps
            features['spread_volatility'] = (spread / mid_price).std()
        
        # Volume features
        if 'volume' in data.columns:
            volume = data['volume']
            features['volume_mean'] = volume.mean()
            features['volume_std'] = volume.std()
            
            if len(volume) >= 20:
                vol_ma_20 = volume.rolling(20).mean()
                features['volume_to_ma'] = volume.iloc[-1] / vol_ma_20.iloc[-1]
        
        return features
    
    async def make_prediction(
        self,
        symbol: str,
        features: Dict[str, float],
        model_name: str = 'default'
    ) -> Optional[Dict[str, float]]:
        """Make ML prediction using loaded model."""
        if model_name not in self.models:
            return None
        
        try:
            model = self.models[model_name]
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, model)
            
            # Make prediction
            prediction = model.predict_proba(feature_vector.reshape(1, -1))[0]
            
            # Convert to signal format
            result = {
                'probability_up': float(prediction[1]) if len(prediction) > 1 else 0.5,
                'probability_down': float(prediction[0]) if len(prediction) > 1 else 0.5,
                'confidence': float(abs(prediction[1] - prediction[0])) if len(prediction) > 1 else 0,
                'prediction_time': datetime.now().isoformat()
            }
            
            self.last_prediction_time = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(
                "ML prediction failed",
                symbol=symbol,
                model_name=model_name,
                error=str(e)
            )
            return None
    
    def _prepare_feature_vector(self, features: Dict[str, float], model) -> np.ndarray:
        """Prepare feature vector for model prediction."""
        # Get expected features from model (if available)
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            # Fallback to alphabetical order
            expected_features = sorted(features.keys())
        
        # Create feature vector
        feature_vector = np.array([
            features.get(feature_name, 0.0) 
            for feature_name in expected_features
        ])
        
        return feature_vector


# Factory function for creating strategies
def create_strategy(
    strategy_type: str,
    strategy_id: str,
    config: StrategyEngineConfig,
    **kwargs
) -> Optional[BaseStrategy]:
    """Factory function to create strategy instances."""
    
    strategy_classes = {
        'quantitative': QuantitativeStrategy,
        'ml_based': MLStrategy,
        # Additional strategy types would be registered here
    }
    
    if strategy_type not in strategy_classes:
        return None
    
    try:
        strategy_class = strategy_classes[strategy_type]
        return strategy_class(strategy_id, config, **kwargs)
    except Exception as e:
        logger = get_logger("strategy_factory")
        logger.error(
            "Failed to create strategy",
            strategy_type=strategy_type,
            strategy_id=strategy_id,
            error=str(e)
        )
        return None


# Utility functions for strategy management
async def validate_strategy_config(strategy_config: Dict[str, Any]) -> List[str]:
    """Validate strategy configuration and return any issues."""
    issues = []
    
    required_fields = ['strategy_id', 'strategy_type', 'universe']
    for field in required_fields:
        if field not in strategy_config:
            issues.append(f"Missing required field: {field}")
    
    # Validate strategy type
    if 'strategy_type' in strategy_config:
        valid_types = ['quantitative', 'ml_based']
        if strategy_config['strategy_type'] not in valid_types:
            issues.append(f"Invalid strategy type: {strategy_config['strategy_type']}")
    
    # Validate universe
    if 'universe' in strategy_config:
        if not isinstance(strategy_config['universe'], (list, set)):
            issues.append("Universe must be a list or set of symbols")
        elif len(strategy_config['universe']) == 0:
            issues.append("Universe cannot be empty")
    
    return issues


def calculate_portfolio_metrics(positions: List[Position]) -> Dict[str, float]:
    """Calculate portfolio-level metrics from positions."""
    if not positions:
        return {}
    
    total_value = sum(float(pos.cost_basis) for pos in positions)
    total_pnl = sum(float(pos.total_pnl or 0) for pos in positions)
    
    long_exposure = sum(
        float(pos.cost_basis) for pos in positions 
        if pos.side in ['BUY', 'LONG']
    )
    short_exposure = sum(
        float(pos.cost_basis) for pos in positions 
        if pos.side in ['SELL', 'SHORT']
    )
    
    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'position_count': len(positions),
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'net_exposure': long_exposure - short_exposure,
        'gross_exposure': long_exposure + short_exposure,
        'return_pct': (total_pnl / total_value * 100) if total_value > 0 else 0
    }
        