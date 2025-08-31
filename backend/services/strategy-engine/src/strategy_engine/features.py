"""
Advanced feature engineering pipeline for trading strategies.
Generates technical, fundamental, macro, and alternative data features.
"""

import asyncio
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass

from .config import StrategyEngineConfig, FeatureSet
from .models.schemas import AssetClass
from ..shared.utils.logging import get_logger, LogEvents
from ..shared.utils.metrics import TradingMetrics


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    lookback_periods: List[int]
    technical_indicators: List[str]
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Technical analysis parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    
    # Statistical parameters
    volatility_windows: List[int] = None
    correlation_window: int = 60
    z_score_window: int = 252
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]


class TechnicalIndicators:
    """Technical analysis indicator calculations."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("technical_indicators")
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all configured technical indicators."""
        indicators = {}
        
        if len(data) == 0 or 'close' not in data.columns:
            return indicators
        
        try:
            # Try using TA-Lib if available
            indicators.update(self._calculate_talib_indicators(data))
        except ImportError:
            self.logger.warning("TA-Lib not available, using pandas implementations")
            indicators.update(self._calculate_pandas_indicators(data))
        
        return indicators
    
    def _calculate_talib_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators using TA-Lib library."""
        import talib as ta
        
        indicators = {}
        
        # Extract OHLCV data
        open_prices = data.get('open', data['close']).values.astype(float)
        high_prices = data.get('high', data['close']).values.astype(float)
        low_prices = data.get('low', data['close']).values.astype(float)
        close_prices = data['close'].values.astype(float)
        volume = data.get('volume', pd.Series([1] * len(data))).values.astype(float)
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close_prices) >= period:
                indicators[f'sma_{period}'] = pd.Series(
                    ta.SMA(close_prices, timeperiod=period), index=data.index
                )
                indicators[f'ema_{period}'] = pd.Series(
                    ta.EMA(close_prices, timeperiod=period), index=data.index
                )
        
        # RSI
        if len(close_prices) >= self.config.rsi_period:
            indicators['rsi'] = pd.Series(
                ta.RSI(close_prices, timeperiod=self.config.rsi_period), index=data.index
            )
        
        # MACD
        if len(close_prices) >= max(self.config.macd_fast, self.config.macd_slow):
            macd, macd_signal, macd_hist = ta.MACD(
                close_prices,
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            indicators['macd'] = pd.Series(macd, index=data.index)
            indicators['macd_signal'] = pd.Series(macd_signal, index=data.index)
            indicators['macd_histogram'] = pd.Series(macd_hist, index=data.index)
        
        # Bollinger Bands
        if len(close_prices) >= self.config.bollinger_period:
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                close_prices,
                timeperiod=self.config.bollinger_period,
                nbdevup=self.config.bollinger_std,
                nbdevdn=self.config.bollinger_std
            )
            indicators['bb_upper'] = pd.Series(bb_upper, index=data.index)
            indicators['bb_middle'] = pd.Series(bb_middle, index=data.index)
            indicators['bb_lower'] = pd.Series(bb_lower, index=data.index)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (pd.Series(close_prices, index=data.index) - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # ATR (Average True Range)
        if len(close_prices) >= self.config.atr_period:
            indicators['atr'] = pd.Series(
                ta.ATR(high_prices, low_prices, close_prices, timeperiod=self.config.atr_period),
                index=data.index
            )
        
        # Stochastic Oscillator
        if len(close_prices) >= 14:
            slowk, slowd = ta.STOCH(high_prices, low_prices, close_prices)
            indicators['stoch_k'] = pd.Series(slowk, index=data.index)
            indicators['stoch_d'] = pd.Series(slowd, index=data.index)
        
        # Williams %R
        if len(close_prices) >= 14:
            indicators['williams_r'] = pd.Series(
                ta.WILLR(high_prices, low_prices, close_prices), index=data.index
            )
        
        # CCI (Commodity Channel Index)
        if len(close_prices) >= 20:
            indicators['cci'] = pd.Series(
                ta.CCI(high_prices, low_prices, close_prices), index=data.index
            )
        
        # Volume indicators
        if volume.sum() > 0:
            # OBV (On Balance Volume)
            indicators['obv'] = pd.Series(
                ta.OBV(close_prices, volume), index=data.index
            )
            
            # VWAP approximation using TA-Lib
            if 'high' in data.columns and 'low' in data.columns:
                typical_price = (high_prices + low_prices + close_prices) / 3
                indicators['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
                indicators['vwap'] = pd.Series(indicators['vwap'], index=data.index)
        
        # Momentum indicators
        if len(close_prices) >= 10:
            indicators['momentum'] = pd.Series(
                ta.MOM(close_prices, timeperiod=10), index=data.index
            )
            indicators['roc'] = pd.Series(
                ta.ROC(close_prices, timeperiod=10), index=data.index
            )
        
        return indicators
    
    def _calculate_pandas_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Fallback indicator calculations using pandas."""
        indicators = {}
        close = data['close']
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close) >= period:
                indicators[f'sma_{period}'] = close.rolling(window=period).mean()
                indicators[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # RSI calculation
        if len(close) >= self.config.rsi_period:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        if len(close) >= max(self.config.macd_fast, self.config.macd_slow):
            ema_fast = close.ewm(span=self.config.macd_fast).mean()
            ema_slow = close.ewm(span=self.config.macd_slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=self.config.macd_signal).mean()
            
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = macd - signal
        
        # Bollinger Bands
        if len(close) >= self.config.bollinger_period:
            sma = close.rolling(window=self.config.bollinger_period).mean()
            std = close.rolling(window=self.config.bollinger_period).std()
            
            indicators['bb_upper'] = sma + (std * self.config.bollinger_std)
            indicators['bb_middle'] = sma
            indicators['bb_lower'] = sma - (std * self.config.bollinger_std)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators


class StatisticalFeatures:
    """Statistical feature calculations."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("statistical_features")
    
    def calculate_return_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate return-based statistical features."""
        features = {}
        
        if 'close' not in data.columns or len(data) < 2:
            return features
        
        # Calculate returns
        returns = data['close'].pct_change()
        log_returns = np.log(data['close'] / data['close'].shift(1))
        
        # Return statistics for different windows
        for window in [5, 10, 20, 60]:
            if len(returns) >= window:
                features[f'return_mean_{window}d'] = returns.rolling(window).mean()
                features[f'return_std_{window}d'] = returns.rolling(window).std()
                features[f'return_skew_{window}d'] = returns.rolling(window).skew()
                features[f'return_kurt_{window}d'] = returns.rolling(window).kurtosis()
                
                # Volatility (annualized)
                features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Momentum features
        for period in [1, 5, 10, 20, 60]:
            if len(data) > period:
                features[f'momentum_{period}d'] = data['close'] / data['close'].shift(period) - 1
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            if len(returns) > lag:
                features[f'autocorr_{lag}'] = returns.rolling(60).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )
        
        return features
    
    def calculate_price_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price-level statistical features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        # Z-scores relative to historical means
        for window in [20, 60, 252]:
            if len(close) >= window:
                rolling_mean = close.rolling(window).mean()
                rolling_std = close.rolling(window).std()
                features[f'zscore_{window}d'] = (close - rolling_mean) / rolling_std
        
        # Percentile ranks
        for window in [20, 60, 252]:
            if len(close) >= window:
                features[f'percentile_rank_{window}d'] = close.rolling(window).rank(pct=True)
        
        # Distance from highs/lows
        for window in [20, 60, 252]:
            if len(close) >= window:
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                features[f'dist_from_high_{window}d'] = (rolling_max - close) / rolling_max
                features[f'dist_from_low_{window}d'] = (close - rolling_min) / rolling_min
        
        return features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based features."""
        features = {}
        
        if 'volume' not in data.columns or data['volume'].sum() == 0:
            return features
        
        volume = data['volume']
        close = data['close']
        
        # Volume statistics
        for window in [5, 20, 60]:
            if len(volume) >= window:
                features[f'volume_mean_{window}d'] = volume.rolling(window).mean()
                features[f'volume_std_{window}d'] = volume.rolling(window).std()
                features[f'volume_ratio_{window}d'] = volume / volume.rolling(window).mean()
        
        # Price-volume relationship
        if len(data) >= 20:
            # Volume-weighted returns
            returns = close.pct_change()
            features['volume_weighted_return'] = (returns * volume).rolling(20).sum() / volume.rolling(20).sum()
            
            # Price-volume correlation
            features['price_volume_corr'] = close.rolling(60).corr(volume)
        
        # On-Balance Volume approximation
        if len(data) >= 2:
            price_change = close.diff()
            volume_direction = np.where(price_change > 0, volume, 
                                     np.where(price_change < 0, -volume, 0))
            features['obv_approx'] = pd.Series(volume_direction, index=data.index).cumsum()
        
        return features


class MarketMicrostructureFeatures:
    """Market microstructure and order book features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("microstructure_features")
    
    def calculate_spread_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate bid-ask spread features."""
        features = {}
        
        if not all(col in data.columns for col in ['bid', 'ask']):
            return features
        
        # Basic spread metrics
        spread = data['ask'] - data['bid']
        mid_price = (data['bid'] + data['ask']) / 2
        
        features['spread_absolute'] = spread
        features['spread_bps'] = spread / mid_price * 10000
        features['spread_percentage'] = spread / mid_price * 100
        
        # Spread statistics
        for window in [20, 60]:
            if len(spread) >= window:
                features[f'spread_mean_{window}'] = features['spread_bps'].rolling(window).mean()
                features[f'spread_std_{window}'] = features['spread_bps'].rolling(window).std()
                features[f'spread_percentile_{window}'] = features['spread_bps'].rolling(window).rank(pct=True)
        
        return features
    
    def calculate_orderbook_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate order book imbalance features."""
        features = {}
        
        # Check for bid/ask size data
        if not all(col in data.columns for col in ['bid_size', 'ask_size']):
            return features
        
        bid_size = data['bid_size']
        ask_size = data['ask_size']
        
        # Order book imbalance
        total_size = bid_size + ask_size
        features['order_imbalance'] = (bid_size - ask_size) / total_size
        
        # Imbalance statistics
        for window in [20, 60]:
            if len(features['order_imbalance']) >= window:
                features[f'imbalance_mean_{window}'] = features['order_imbalance'].rolling(window).mean()
                features[f'imbalance_std_{window}'] = features['order_imbalance'].rolling(window).std()
        
        return features
    
    def calculate_trade_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trade-level features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        # Trade intensity (simplified using volume as proxy)
        if 'volume' in data.columns:
            volume = data['volume']
            
            # Trade rate (volume per time period)
            for window in [5, 20, 60]:
                if len(volume) >= window:
                    features[f'trade_rate_{window}'] = volume.rolling(window).sum()
            
            # Average trade size (simplified)
            if 'trade_count' in data.columns:
                trade_count = data['trade_count']
                avg_trade_size = volume / trade_count
                features['avg_trade_size'] = avg_trade_size
        
        return features


class SentimentFeatures:
    """Sentiment and alternative data features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("sentiment_features")
    
    def calculate_news_sentiment_features(self, news_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate news sentiment features."""
        features = {}
        
        if news_data.empty or 'sentiment_score' not in news_data.columns:
            return features
        
        # Aggregate sentiment scores
        daily_sentiment = news_data.groupby(news_data.index.date)['sentiment_score'].mean()
        features['news_sentiment'] = daily_sentiment
        
        # Sentiment momentum
        for window in [5, 20]:
            if len(daily_sentiment) >= window:
                features[f'sentiment_ma_{window}'] = daily_sentiment.rolling(window).mean()
                features[f'sentiment_std_{window}'] = daily_sentiment.rolling(window).std()
        
        return features
    
    def calculate_social_sentiment_features(self, social_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate social media sentiment features."""
        features = {}
        
        if social_data.empty:
            return features
        
        # Volume of mentions
        if 'mention_count' in social_data.columns:
            daily_mentions = social_data.groupby(social_data.index.date)['mention_count'].sum()
            features['social_volume'] = daily_mentions
        
        # Average sentiment
        if 'sentiment_score' in social_data.columns:
            daily_sentiment = social_data.groupby(social_data.index.date)['sentiment_score'].mean()
            features['social_sentiment'] = daily_sentiment
        
        return features


class MacroeconomicFeatures:
    """Macroeconomic and fundamental features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("macro_features")
    
    def calculate_interest_rate_features(self, rates_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate interest rate environment features."""
        features = {}
        
        if rates_data.empty:
            return features
        
        # Yield curve features
        if 'treasury_10y' in rates_data.columns and 'treasury_2y' in rates_data.columns:
            features['yield_curve_slope'] = rates_data['treasury_10y'] - rates_data['treasury_2y']
        
        # Rate changes
        for rate_type in ['fed_funds', 'treasury_10y', 'treasury_2y']:
            if rate_type in rates_data.columns:
                features[f'{rate_type}_change'] = rates_data[rate_type].diff()
        
        return features
    
    def calculate_market_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Calculate market regime and cross-asset features."""
        features = {}
        
        # VIX-like volatility measure
        if 'SPY' in market_data and 'close' in market_data['SPY'].columns:
            spy_data = market_data['SPY']
            spy_returns = spy_data['close'].pct_change()
            
            # Realized volatility as VIX proxy
            features['market_volatility'] = spy_returns.rolling(20).std() * np.sqrt(252) * 100
            
            # Market stress indicator
            features['market_stress'] = (features['market_volatility'] > features['market_volatility'].rolling(252).quantile(0.8)).astype(float)
        
        # Cross-asset correlations
        if len(market_data) >= 2:
            symbols = list(market_data.keys())[:5]  # Limit for performance
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    if ('close' in market_data[symbol1].columns and 
                        'close' in market_data[symbol2].columns):
                        
                        returns1 = market_data[symbol1]['close'].pct_change()
                        returns2 = market_data[symbol2]['close'].pct_change()
                        
                        correlation = returns1.rolling(60).corr(returns2)
                        features[f'corr_{symbol1}_{symbol2}'] = correlation
        
        return features


class FeatureEngineeringPipeline:
    """Main feature engineering pipeline orchestrator."""
    
    def __init__(self, config: StrategyEngineConfig, metrics: Optional[TradingMetrics] = None):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("feature_pipeline")
        
        # Initialize feature configuration
        self.feature_config = FeatureConfig(
            lookback_periods=config.lookback_periods,
            technical_indicators=config.technical_indicators,
            enable_caching=config.enable_feature_caching,
            cache_ttl_seconds=config.cache_ttl_seconds
        )
        
        # Initialize feature calculators
        self.technical = TechnicalIndicators(self.feature_config)
        self.statistical = StatisticalFeatures(self.feature_config)
        self.microstructure = MarketMicrostructureFeatures(self.feature_config)
        self.sentiment = SentimentFeatures(self.feature_config)
        self.macro = MacroeconomicFeatures(self.feature_config)
        
        # Feature cache
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
    
    async def generate_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        symbol: str,
        feature_sets: Optional[List[FeatureSet]] = None,
        additional_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """Generate comprehensive feature set for a symbol."""
        
        if feature_sets is None:
            feature_sets = self.config.feature_sets
        
                if symbol not in market_data:
            self.logger.warning(f"No market data available for symbol {symbol}")
            return {}
        
        data = market_data[symbol]
        
        # Check cache first
        cache_key = f"{symbol}_{hash(str(feature_sets))}"
        if self._is_cache_valid(cache_key):
            return self.feature_cache[cache_key]
        
        start_time = time.time()
        features = {}
        
        try:
            # Technical features
            if FeatureSet.TECHNICAL in feature_sets:
                tech_features = await self._generate_technical_features(data)
                features.update(tech_features)
            
            # Statistical features
            stat_features = await self._generate_statistical_features(data)
            features.update(stat_features)
            
            # Market microstructure features
            if FeatureSet.MICROSTRUCTURE in feature_sets:
                micro_features = await self._generate_microstructure_features(data)
                features.update(micro_features)
            
            # Sentiment features
            if FeatureSet.SENTIMENT in feature_sets and additional_data:
                sentiment_features = await self._generate_sentiment_features(additional_data)
                features.update(sentiment_features)
            
            # Macro features
            if FeatureSet.MACRO in feature_sets and additional_data:
                macro_features = await self._generate_macro_features(additional_data, market_data)
                features.update(macro_features)
            
            # Fundamental features
            if FeatureSet.FUNDAMENTAL in feature_sets and additional_data:
                fundamental_features = await self._generate_fundamental_features(
                    symbol, additional_data
                )
                features.update(fundamental_features)
            
            # Clean and validate features
            features = self._clean_features(features)
            
            # Cache results
            if self.feature_config.enable_caching:
                self.feature_cache[cache_key] = features
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Log performance metrics
            processing_time = time.time() - start_time
            self.logger.debug(
                "Features generated",
                symbol=symbol,
                feature_count=len(features),
                processing_time_ms=processing_time * 1000
            )
            
            if self.metrics:
                self.metrics.feature_computation_duration.labels(
                    service=self.config.service_name,
                    symbol=symbol
                ).observe(processing_time)
            
            return features
            
        except Exception as e:
            self.logger.error(
                "Feature generation failed",
                symbol=symbol,
                error=str(e)
            )
            return {}
    
    async def _generate_technical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate technical analysis features."""
        features = {}
        
        # Calculate technical indicators
        indicators = self.technical.calculate_all_indicators(data)
        
        # Convert to latest values
        for name, series in indicators.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'tech_{name}'] = float(series.iloc[-1])
        
        # Calculate technical patterns and signals
        if 'close' in data.columns and len(data) >= 20:
            close = data['close']
            
            # Moving average crossovers
            if 'sma_10' in indicators and 'sma_20' in indicators:
                ma_cross = indicators['sma_10'].iloc[-1] / indicators['sma_20'].iloc[-1] - 1
                features['tech_ma_cross_10_20'] = float(ma_cross)
            
            # Bollinger Band position
            if all(k in indicators for k in ['bb_upper', 'bb_lower']):
                bb_pos = ((close.iloc[-1] - indicators['bb_lower'].iloc[-1]) / 
                         (indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]))
                features['tech_bb_position'] = float(bb_pos)
            
            # RSI signals
            if 'rsi' in indicators:
                rsi_val = indicators['rsi'].iloc[-1]
                features['tech_rsi_overbought'] = float(rsi_val > 70)
                features['tech_rsi_oversold'] = float(rsi_val < 30)
            
            # MACD signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_cross = indicators['macd'].iloc[-1] - indicators['macd_signal'].iloc[-1]
                features['tech_macd_cross'] = float(macd_cross)
        
        return features
    
    async def _generate_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate statistical features."""
        features = {}
        
        # Return-based features
        return_features = self.statistical.calculate_return_features(data)
        for name, series in return_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'stat_{name}'] = float(series.iloc[-1])
        
        # Price-based features
        price_features = self.statistical.calculate_price_features(data)
        for name, series in price_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'stat_{name}'] = float(series.iloc[-1])
        
        # Volume features
        volume_features = self.statistical.calculate_volume_features(data)
        for name, series in volume_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'stat_{name}'] = float(series.iloc[-1])
        
        return features
    
    async def _generate_microstructure_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate market microstructure features."""
        features = {}
        
        # Spread features
        spread_features = self.microstructure.calculate_spread_features(data)
        for name, series in spread_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'micro_{name}'] = float(series.iloc[-1])
        
        # Order book features
        orderbook_features = self.microstructure.calculate_orderbook_features(data)
        for name, series in orderbook_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'micro_{name}'] = float(series.iloc[-1])
        
        # Trade features
        trade_features = self.microstructure.calculate_trade_features(data)
        for name, series in trade_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'micro_{name}'] = float(series.iloc[-1])
        
        return features
    
    async def _generate_sentiment_features(self, additional_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate sentiment-based features."""
        features = {}
        
        # News sentiment
        if 'news' in additional_data:
            news_features = self.sentiment.calculate_news_sentiment_features(
                additional_data['news']
            )
            for name, series in news_features.items():
                if not series.empty and not pd.isna(series.iloc[-1]):
                    features[f'sentiment_{name}'] = float(series.iloc[-1])
        
        # Social sentiment
        if 'social' in additional_data:
            social_features = self.sentiment.calculate_social_sentiment_features(
                additional_data['social']
            )
            for name, series in social_features.items():
                if not series.empty and not pd.isna(series.iloc[-1]):
                    features[f'sentiment_{name}'] = float(series.iloc[-1])
        
        return features
    
    async def _generate_macro_features(
        self, 
        additional_data: Dict[str, pd.DataFrame],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate macroeconomic features."""
        features = {}
        
        # Interest rate features
        if 'rates' in additional_data:
            rate_features = self.macro.calculate_interest_rate_features(
                additional_data['rates']
            )
            for name, series in rate_features.items():
                if not series.empty and not pd.isna(series.iloc[-1]):
                    features[f'macro_{name}'] = float(series.iloc[-1])
        
        # Market regime features
        regime_features = self.macro.calculate_market_regime_features(market_data)
        for name, series in regime_features.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                features[f'macro_{name}'] = float(series.iloc[-1])
        
        return features
    
    async def _generate_fundamental_features(
        self, 
        symbol: str, 
        additional_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate fundamental analysis features."""
        features = {}
        
        if 'fundamentals' not in additional_data:
            return features
        
        fundamental_data = additional_data['fundamentals']
        
        # Financial ratios
        if not fundamental_data.empty:
            for ratio in ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'roa']:
                if ratio in fundamental_data.columns:
                    latest_value = fundamental_data[ratio].iloc[-1]
                    if not pd.isna(latest_value):
                        features[f'fundamental_{ratio}'] = float(latest_value)
        
        # Earnings features
        if 'earnings' in additional_data:
            earnings_data = additional_data['earnings']
            if not earnings_data.empty:
                # Earnings surprise
                if 'actual_eps' in earnings_data.columns and 'expected_eps' in earnings_data.columns:
                    actual = earnings_data['actual_eps'].iloc[-1]
                    expected = earnings_data['expected_eps'].iloc[-1]
                    if not pd.isna(actual) and not pd.isna(expected) and expected != 0:
                        surprise = (actual - expected) / expected
                        features['fundamental_earnings_surprise'] = float(surprise)
        
        return features
    
    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Clean and validate feature values."""
        cleaned_features = {}
        
        for name, value in features.items():
            # Check for invalid values
            if pd.isna(value) or np.isinf(value):
                continue
            
            # Clip extreme values to prevent outliers
            if abs(value) > 1e6:
                value = np.sign(value) * 1e6
            
            # Round to reasonable precision
            cleaned_features[name] = round(float(value), 6)
        
        return cleaned_features
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached features are still valid."""
        if not self.feature_config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return cache_age < self.feature_config.cache_ttl_seconds
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        self.feature_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Feature cache cleared")
    
    async def batch_generate_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        feature_sets: Optional[List[FeatureSet]] = None,
        additional_data: Optional[Dict[str, pd.DataFrame]] = None,
        max_workers: int = 4
    ) -> Dict[str, Dict[str, float]]:
        """Generate features for multiple symbols in parallel."""
        
        if max_workers > 1:
            # Parallel processing
            semaphore = asyncio.Semaphore(max_workers)
            
            async def generate_with_semaphore(symbol):
                async with semaphore:
                    return await self.generate_features(
                        market_data, symbol, feature_sets, additional_data
                    )
            
            tasks = [generate_with_semaphore(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            feature_dict = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, dict):
                    feature_dict[symbol] = result
                else:
                    self.logger.error(f"Feature generation failed for {symbol}: {result}")
                    feature_dict[symbol] = {}
            
            return feature_dict
        
        else:
            # Sequential processing
            feature_dict = {}
            for symbol in symbols:
                features = await self.generate_features(
                    market_data, symbol, feature_sets, additional_data
                )
                feature_dict[symbol] = features
            
            return feature_dict
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        importance_dict = {}
        
        try:
            # LightGBM/XGBoost style
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
            
            # Sklearn style
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # Take first class for binary
                
                for name, coef in zip(feature_names, coefficients):
                    importance_dict[name] = float(abs(coef))
            
            # Tree-based models
            elif hasattr(model, 'get_score'):
                # XGBoost native
                scores = model.get_score(importance_type='weight')
                for name in feature_names:
                    importance_dict[name] = float(scores.get(name, 0))
        
        except Exception as e:
            self.logger.error(f"Failed to extract feature importance: {e}")
        
        return importance_dict
    
    def select_features(
        self, 
        features: Dict[str, float], 
        importance_scores: Dict[str, float],
        top_k: int = 50
    ) -> Dict[str, float]:
        """Select top-k most important features."""
        
        if not importance_scores:
            # If no importance scores, return all features
            return features
        
        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top-k
        selected_names = [name for name, _ in sorted_features[:top_k]]
        
        # Filter original features
        selected_features = {
            name: value for name, value in features.items() 
            if name in selected_names
        }
        
        self.logger.debug(f"Selected {len(selected_features)} features from {len(features)}")
        
        return selected_features
    
    def get_feature_statistics(self, feature_history: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for features over time."""
        if not feature_history:
            return {}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(feature_history)
        
        statistics = {}
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                statistics[column] = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'median': float(df[column].median()),
                    'skew': float(df[column].skew()),
                    'kurtosis': float(df[column].kurtosis())
                }
        
        return statistics


# Utility functions
def create_feature_pipeline(config: StrategyEngineConfig, metrics: Optional[TradingMetrics] = None) -> FeatureEngineeringPipeline:
    """Factory function to create feature engineering pipeline."""
    return FeatureEngineeringPipeline(config, metrics)


def validate_features(features: Dict[str, float], expected_features: Optional[List[str]] = None) -> List[str]:
    """Validate feature dictionary and return any issues."""
    issues = []
    
    # Check for empty features
    if not features:
        issues.append("No features generated")
        return issues
    
    # Check for invalid values
    for name, value in features.items():
        if pd.isna(value):
            issues.append(f"NaN value in feature: {name}")
        elif np.isinf(value):
            issues.append(f"Infinite value in feature: {name}")
        elif not isinstance(value, (int, float)):
            issues.append(f"Non-numeric value in feature: {name}")
    
    # Check for expected features
    if expected_features:
        missing_features = set(expected_features) - set(features.keys())
        if missing_features:
            issues.append(f"Missing expected features: {list(missing_features)}")
    
    return issues


def normalize_features(
    features: Dict[str, float], 
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """Normalize features using z-score normalization."""
    if not feature_stats:
        return features
    
    normalized = {}
    
    for name, value in features.items():
        if name in feature_stats:
            stats = feature_stats[name]
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            
            if std > 0:
                normalized_value = (value - mean) / std
                normalized[name] = normalized_value
            else:
                normalized[name] = value
        else:
            normalized[name] = value
    
    return normalized


def calculate_feature_correlation_matrix(feature_history: List[Dict[str, float]]) -> pd.DataFrame:
    """Calculate correlation matrix for features."""
    if not feature_history:
        return pd.DataFrame()
    
    df = pd.DataFrame(feature_history)
    
    # Only include numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) < 2:
        return pd.DataFrame()
    
    correlation_matrix = df[numeric_columns].corr()
    return correlation_matrix


async def test_feature_pipeline():
    """Test function for feature engineering pipeline."""
    # Create test configuration
    config = StrategyEngineConfig()
    
    # Create pipeline
    pipeline = create_feature_pipeline(config)
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    market_data = {'AAPL': sample_data}
    
    # Generate features
    features = await pipeline.generate_features(
        market_data, 
        'AAPL', 
        [FeatureSet.TECHNICAL, FeatureSet.ALL]
    )
    
    print(f"Generated {len(features)} features:")
    for name, value in sorted(features.items()):
        print(f"  {name}: {value:.6f}")
    
    # Validate features
    issues = validate_features(features)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All features passed validation")


if __name__ == "__main__":
    asyncio.run(test_feature_pipeline())