"""
Comprehensive metrics collection for algorithmic trading system.
Provides Prometheus integration, custom trading metrics, and performance monitoring.
"""

import time
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from prometheus_client import (
    CollectorRegistry, Counter, Gauge, Histogram, Info, Summary,
    generate_latest, push_to_gateway, start_http_server
)


class MetricLabels:
    """Standardized metric labels for trading system."""
    
    SERVICE = "service"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    EXCHANGE = "exchange"
    SIDE = "side"
    ORDER_TYPE = "order_type"
    STATUS = "status"
    MODEL_NAME = "model"
    ASSET_CLASS = "asset_class"
    TIMEFRAME = "timeframe"


class TradingMetrics:
    """Comprehensive metrics collector for trading operations."""
    
    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.service_name = service_name
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize all trading-specific metrics."""
        
        # System metrics
        self.service_info = Info(
            'trading_service_info',
            'Service information',
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'trading_service_uptime_seconds',
            'Service uptime in seconds',
            [MetricLabels.SERVICE],
            registry=self.registry
        )
        
        # Data ingestion metrics
        self.data_received_total = Counter(
            'trading_data_received_total',
            'Total data points received',
            [MetricLabels.SERVICE, MetricLabels.SYMBOL, MetricLabels.EXCHANGE],
            registry=self.registry
        )
        
        self.data_processing_duration = Histogram(
            'trading_data_processing_duration_seconds',
            'Time spent processing market data',
            [MetricLabels.SERVICE, MetricLabels.SYMBOL],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.feed_connections = Gauge(
            'trading_feed_connections_active',
            'Active feed connections',
            [MetricLabels.SERVICE, MetricLabels.EXCHANGE],
            registry=self.registry
        )
        
        self.data_latency_seconds = Histogram(
            'trading_data_latency_seconds',
            'Market data latency from exchange timestamp',
            [MetricLabels.SERVICE, MetricLabels.SYMBOL, MetricLabels.EXCHANGE],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry
        )
        
        # Strategy metrics
        self.signals_generated_total = Counter(
            'trading_signals_generated_total',
            'Total trading signals generated',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL, MetricLabels.SIDE],
            registry=self.registry
        )
        
        self.strategy_performance_ratio = Gauge(
            'trading_strategy_performance_ratio',
            'Strategy performance metrics',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, 'metric_type'],
            registry=self.registry
        )
        
        self.backtest_duration_seconds = Histogram(
            'trading_backtest_duration_seconds',
            'Backtesting execution time',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY],
            registry=self.registry
        )
        
        # Execution metrics
        self.orders_submitted_total = Counter(
            'trading_orders_submitted_total',
            'Total orders submitted',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL, 
             MetricLabels.SIDE, MetricLabels.ORDER_TYPE, MetricLabels.EXCHANGE],
            registry=self.registry
        )
        
        self.orders_filled_total = Counter(
            'trading_orders_filled_total',
            'Total orders filled',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL, 
             MetricLabels.SIDE, MetricLabels.EXCHANGE],
            registry=self.registry
        )
        
        self.order_execution_duration = Histogram(
            'trading_order_execution_duration_seconds',
            'Time from order submission to fill',
            [MetricLabels.SERVICE, MetricLabels.EXCHANGE, MetricLabels.ORDER_TYPE],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )
        
        self.fill_price_slippage = Histogram(
            'trading_fill_price_slippage_bps',
            'Price slippage in basis points',
            [MetricLabels.SERVICE, MetricLabels.SYMBOL, MetricLabels.EXCHANGE],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
            registry=self.registry
        )
        
        # Position and portfolio metrics
        self.positions_active = Gauge(
            'trading_positions_active',
            'Currently active positions',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL],
            registry=self.registry
        )
        
        self.portfolio_value_usd = Gauge(
            'trading_portfolio_value_usd',
            'Total portfolio value in USD',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY],
            registry=self.registry
        )
        
        self.unrealized_pnl_usd = Gauge(
            'trading_unrealized_pnl_usd',
            'Unrealized PnL in USD',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL],
            registry=self.registry
        )
        
        self.realized_pnl_usd = Counter(
            'trading_realized_pnl_usd',
            'Cumulative realized PnL in USD',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, MetricLabels.SYMBOL],
            registry=self.registry
        )
        
        # Risk metrics
        self.risk_limit_breaches_total = Counter(
            'trading_risk_limit_breaches_total',
            'Total risk limit breaches',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY, 'limit_type'],
            registry=self.registry
        )
        
        self.leverage_ratio = Gauge(
            'trading_leverage_ratio',
            'Current leverage ratio',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY],
            registry=self.registry
        )
        
        self.var_95_usd = Gauge(
            'trading_var_95_usd',
            'Value at Risk 95% confidence in USD',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY],
            registry=self.registry
        )
        
        self.max_drawdown_pct = Gauge(
            'trading_max_drawdown_percent',
            'Maximum drawdown percentage',
            [MetricLabels.SERVICE, MetricLabels.STRATEGY],
            registry=self.registry
        )
        
        # ML metrics
        self.model_predictions_total = Counter(
            'trading_model_predictions_total',
            'Total ML model predictions',
            [MetricLabels.SERVICE, MetricLabels.MODEL_NAME, MetricLabels.SYMBOL],
            registry=self.registry
        )
        
        self.model_accuracy_score = Gauge(
            'trading_model_accuracy_score',
            'Model accuracy score',
            [MetricLabels.SERVICE, MetricLabels.MODEL_NAME],
            registry=self.registry
        )
        
        self.model_training_duration_seconds = Histogram(
            'trading_model_training_duration_seconds',
            'Model training time',
            [MetricLabels.SERVICE, MetricLabels.MODEL_NAME],
            registry=self.registry
        )
        
        self.feature_computation_duration = Histogram(
            'trading_feature_computation_duration_seconds',
            'Feature computation time',
            [MetricLabels.SERVICE, MetricLabels.SYMBOL],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        # API and system performance
        self.http_requests_total = Counter(
            'trading_http_requests_total',
            'Total HTTP requests',
            [MetricLabels.SERVICE, 'method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'trading_http_request_duration_seconds',
            'HTTP request duration',
            [MetricLabels.SERVICE, 'method', 'endpoint'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'trading_memory_usage_bytes',
            'Memory usage in bytes',
            [MetricLabels.SERVICE],
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'trading_cpu_usage_percent',
            'CPU usage percentage',
            [MetricLabels.SERVICE],
            registry=self.registry
        )
    
    def record_data_received(self, symbol: str, exchange: str, count: int = 1) -> None:
        """Record market data received."""
        self.data_received_total.labels(
            service=self.service_name, symbol=symbol, exchange=exchange
        ).inc(count)
    
    def record_data_latency(self, symbol: str, exchange: str, latency_seconds: float) -> None:
        """Record market data latency."""
        self.data_latency_seconds.labels(
            service=self.service_name, symbol=symbol, exchange=exchange
        ).observe(latency_seconds)
    
    def record_signal_generated(self, strategy: str, symbol: str, side: str) -> None:
        """Record trading signal generation."""
        self.signals_generated_total.labels(
            service=self.service_name, strategy=strategy, symbol=symbol, side=side
        ).inc()
    
    def record_order_submitted(self, strategy: str, symbol: str, side: str, 
                             order_type: str, exchange: str) -> None:
        """Record order submission."""
        self.orders_submitted_total.labels(
            service=self.service_name, strategy=strategy, symbol=symbol,
            side=side, order_type=order_type, exchange=exchange
        ).inc()
    
    def record_order_filled(self, strategy: str, symbol: str, side: str, 
                          exchange: str, execution_time: float) -> None:
        """Record order fill."""
        self.orders_filled_total.labels(
            service=self.service_name, strategy=strategy, symbol=symbol,
            side=side, exchange=exchange
        ).inc()
        
        self.order_execution_duration.labels(
            service=self.service_name, exchange=exchange, order_type="MARKET"
        ).observe(execution_time)
    
    def record_slippage(self, symbol: str, exchange: str, slippage_bps: float) -> None:
        """Record price slippage in basis points."""
        self.fill_price_slippage.labels(
            service=self.service_name, symbol=symbol, exchange=exchange
        ).observe(slippage_bps)
    
    def update_portfolio_metrics(self, strategy: str, portfolio_value: float, 
                               unrealized_pnl: float, leverage: float) -> None:
        """Update portfolio-level metrics."""
        self.portfolio_value_usd.labels(
            service=self.service_name, strategy=strategy
        ).set(portfolio_value)
        
        self.leverage_ratio.labels(
            service=self.service_name, strategy=strategy
        ).set(leverage)
    
    def record_risk_breach(self, strategy: str, limit_type: str) -> None:
        """Record risk limit breach."""
        self.risk_limit_breaches_total.labels(
            service=self.service_name, strategy=strategy, limit_type=limit_type
        ).inc()
    
    def record_model_prediction(self, model_name: str, symbol: str, accuracy: Optional[float] = None) -> None:
        """Record ML model prediction."""
        self.model_predictions_total.labels(
            service=self.service_name, model_name=model_name, symbol=symbol
        ).inc()
        
        if accuracy is not None:
            self.model_accuracy_score.labels(
                service=self.service_name, model_name=model_name
            ).set(accuracy)
    
    def update_system_metrics(self, memory_bytes: int, cpu_percent: float) -> None:
        """Update system resource metrics."""
        self.memory_usage_bytes.labels(service=self.service_name).set(memory_bytes)
        self.cpu_usage_percent.labels(service=self.service_name).set(cpu_percent)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""
    
    def __init__(self, metrics: TradingMetrics):
        self.metrics = metrics
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record HTTP metrics
        self.metrics.http_requests_total.labels(
            service=self.metrics.service_name,
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        self.metrics.http_request_duration_seconds.labels(
            service=self.metrics.service_name,
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response


def track_performance(metrics: TradingMetrics, operation: str):
    """Decorator to track operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # You can add custom performance tracking here
                pass
        return wrapper
    return decorator


def track_async_performance(metrics: TradingMetrics, operation: str):
    """Decorator to track async operation performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # You can add custom performance tracking here
                pass
        return wrapper
    return decorator


class SystemResourceMonitor:
    """Monitor system resources and update metrics."""
    
    def __init__(self, metrics: TradingMetrics):
        self.metrics = metrics
        self._start_time = time.time()
    
    def update_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_usage_bytes.labels(
                service=self.metrics.service_name
            ).set(memory_info.rss)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics.cpu_usage_percent.labels(
                service=self.metrics.service_name
            ).set(cpu_percent)
            
            # Uptime
            uptime = time.time() - self._start_time
            self.metrics.uptime_seconds.labels(
                service=self.metrics.service_name
            ).set(uptime)
            
        except ImportError:
            # psutil not available
            pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize metrics for a service
    metrics = TradingMetrics("data-ingest")
    
    # Set service info
    metrics.service_info.info({
        'version': '1.0.0',
        'build': 'abc123',
        'environment': 'development'
    })
    
    # Simulate some trading activities
    metrics.record_data_received("AAPL", "NASDAQ", 100)
    metrics.record_data_latency("AAPL", "NASDAQ", 0.015)
    
    metrics.record_signal_generated("momentum_v1", "AAPL", "BUY")
    metrics.record_order_submitted("momentum_v1", "AAPL", "BUY", "MARKET", "NASDAQ")
    metrics.record_order_filled("momentum_v1", "AAPL", "BUY", "NASDAQ", 0.125)
    
    metrics.record_slippage("AAPL", "NASDAQ", 2.5)
    
    metrics.update_portfolio_metrics("momentum_v1", 100000.0, 1250.0, 1.5)
    
    metrics.record_model_prediction("lightgbm_v1", "AAPL", 0.85)
    
    # Print metrics
    print("Generated Metrics:")
    print(metrics.get_metrics())
    
    # Start HTTP server for metrics endpoint (for testing)
    # start_http_server(8000, registry=metrics.registry)
    # print("Metrics server started on port 8000")