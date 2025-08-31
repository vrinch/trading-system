"""
Production-grade structured logging for trading system.
Provides trace correlation, performance metrics, and audit trails.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import structlog
from pythonjsonlogger import jsonlogger

# Context variables for distributed tracing
trace_id_ctx: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
strategy_id_ctx: ContextVar[Optional[str]] = ContextVar('strategy_id', default=None)


class TradingLogFormatter(jsonlogger.JsonFormatter):
    """High-performance JSON formatter for trading logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        log_record['level'] = record.levelname
        log_record['component'] = getattr(record, 'component', 'unknown')
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add tracing context
        if trace_id := trace_id_ctx.get():
            log_record['trace_id'] = trace_id
        if user_id := user_id_ctx.get():
            log_record['user_id'] = user_id
        if strategy_id := strategy_id_ctx.get():
            log_record['strategy_id'] = strategy_id
            
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        if hasattr(record, 'memory_mb'):
            log_record['memory_mb'] = record.memory_mb


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add memory usage for critical operations
        if hasattr(record, 'operation') and record.operation in ['order_execution', 'risk_check', 'ml_prediction']:
            try:
                import psutil
                process = psutil.Process()
                record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
            except ImportError:
                pass
        return True


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    enable_trace: bool = True,
    enable_performance: bool = True
) -> structlog.BoundLogger:
    """
    Setup structured logging for trading services.
    
    Args:
        service_name: Name of the service (e.g., 'data-ingest', 'execution')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_trace: Enable distributed tracing
        enable_performance: Enable performance metrics
    
    Returns:
        Configured structured logger
    """
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout,
        format='%(message)s'
    )
    
    # Setup JSON formatter
    formatter = TradingLogFormatter(
        fmt='%(timestamp)s %(level)s %(component)s %(message)s',
        static_fields={'service': service_name}
    )
    
    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Add performance filter if enabled
    if enable_performance:
        handler.addFilter(PerformanceFilter())
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger().bind(component=service_name)


def get_logger(component: str) -> structlog.BoundLogger:
    """Get a component-specific logger."""
    return structlog.get_logger().bind(component=component)


def set_trace_context(trace_id: str, user_id: Optional[str] = None, strategy_id: Optional[str] = None) -> None:
    """Set tracing context for current request/operation."""
    trace_id_ctx.set(trace_id)
    if user_id:
        user_id_ctx.set(user_id)
    if strategy_id:
        strategy_id_ctx.set(strategy_id)


def generate_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


def clear_trace_context() -> None:
    """Clear tracing context."""
    trace_id_ctx.set(None)
    user_id_ctx.set(None)
    strategy_id_ctx.set(None)


class LogExecutionTime:
    """Context manager for logging execution time of operations."""
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug("Operation started", operation=self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        
        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation,
                duration_ms=duration_ms,
                **self.context
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error=str(exc_val),
                **self.context
            )


class AuditLogger:
    """Specialized logger for audit trails and compliance."""
    
    def __init__(self, service_name: str):
        self.logger = get_logger(f"{service_name}-audit")
    
    def log_order_event(self, event_type: str, order_data: Dict[str, Any], **context) -> None:
        """Log order-related events for audit trail."""
        self.logger.info(
            "Order event",
            event_type=event_type,
            client_order_id=order_data.get('client_order_id'),
            symbol=order_data.get('symbol'),
            side=order_data.get('side'),
            quantity=float(order_data.get('quantity', 0)),
            price=float(order_data.get('price', 0)) if order_data.get('price') else None,
            order_type=order_data.get('order_type'),
            **context
        )
    
    def log_risk_event(self, event_type: str, risk_data: Dict[str, Any], **context) -> None:
        """Log risk management events."""
        self.logger.warning(
            "Risk event",
            event_type=event_type,
            risk_type=risk_data.get('risk_type'),
            current_value=float(risk_data.get('current_value', 0)),
            limit_value=float(risk_data.get('limit_value', 0)),
            breach_severity=risk_data.get('breach_severity', 'low'),
            **context
        )
    
    def log_ml_event(self, event_type: str, model_data: Dict[str, Any], **context) -> None:
        """Log ML model events."""
        self.logger.info(
            "ML event",
            event_type=event_type,
            model_name=model_data.get('model_name'),
            model_version=model_data.get('model_version'),
            accuracy_score=float(model_data.get('accuracy_score', 0)) if model_data.get('accuracy_score') else None,
            **context
        )


# Trading-specific log event types
class LogEvents:
    """Constants for standardized log event types."""
    
    # Data ingestion
    FEED_CONNECTED = "feed_connected"
    FEED_DISCONNECTED = "feed_disconnected"
    DATA_RECEIVED = "data_received"
    DATA_PROCESSED = "data_processed"
    DATA_ERROR = "data_error"
    
    # Strategy engine
    SIGNAL_GENERATED = "signal_generated"
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"
    STRATEGY_ERROR = "strategy_error"
    
    # Execution
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk management
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_SIZED = "position_sized"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    
    # ML pipeline
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_PROMOTED = "model_promoted"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    PREDICTION_GENERATED = "prediction_generated"


# Performance monitoring decorators
def log_performance(operation: str):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with LogExecutionTime(logger, operation, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def log_async_performance(operation: str):
    """Decorator for async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with LogExecutionTime(logger, operation, function=func.__name__):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage patterns for trading system
if __name__ == "__main__":
    # Setup logging for a service
    logger = setup_logging("data-ingest", "DEBUG")
    
    # Set tracing context
    trace_id = generate_trace_id()
    set_trace_context(trace_id, strategy_id="momentum_v1")
    
    # Log various events
    logger.info("Service starting", version="1.0.0", mode="test")
    
    # Use performance logging
    @log_performance("market_data_processing")
    def process_market_data(symbol: str, data: Dict[str, Any]):
        logger.debug("Processing market data", symbol=symbol, records=len(data))
        time.sleep(0.1)  # Simulate processing
        return {"processed": True, "symbol": symbol}
    
    # Use audit logger
    audit = AuditLogger("data-ingest")
    audit.log_order_event("ORDER_RECEIVED", {
        "client_order_id": "test_001",
        "symbol": "AAPL",
        "side": "B",
        "quantity": "100",
        "order_type": "MARKET"
    })
    
    # Test performance logging
    result = process_market_data("AAPL", {"tick": 1})
    logger.info("Processing result", result=result)
    
    # Clear context
    clear_trace_context()