"""
Main entry point for the data ingestion service.
Handles CLI arguments, configuration, and service startup.
"""

import asyncio
import os
import sys
import signal
import argparse
from typing import Optional

import click
import uvicorn
from prometheus_client import start_http_server

from .app import create_app
from .config import get_config_by_env, TradingMode, LogLevel
from .feeds.polygon import test_polygon_connection
from .feeds.ccxt_wrapper import test_ccxt_feeds
from .feeds.mock_exchange import test_mock_feed
from ..shared.utils.logging import setup_logging, get_logger


def setup_signal_handlers(logger):
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        # Uvicorn will handle the graceful shutdown
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@click.group()
@click.option('--env', '-e', default=None, help='Environment (development, test, production)')
@click.option('--log-level', '-l', default=None, help='Logging level')
@click.option('--config-file', '-c', default=None, help='Configuration file path')
@click.pass_context
def cli(ctx, env, log_level, config_file):
    """Data Ingestion Service CLI."""
    ctx.ensure_object(dict)
    ctx.obj['env'] = env
    ctx.obj['log_level'] = log_level
    ctx.obj['config_file'] = config_file


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', type=int, default=None, help='Port to bind to')
@click.option('--workers', type=int, default=None, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--trading-mode', type=click.Choice(['test', 'shadow', 'live']), 
              default=None, help='Trading mode')
@click.pass_context
def serve(ctx, host, port, workers, reload, trading_mode):
    """Start the data ingestion service."""
    
    # Get configuration
    config = get_config_by_env(ctx.obj.get('env'))
    
    # Override config with CLI arguments
    if host:
        config.host = host
    if port:
        config.port = port
    if workers:
        config.workers = workers
    if reload is not None:
        config.reload = reload
    if trading_mode:
        config.trading_mode = TradingMode(trading_mode)
    if ctx.obj.get('log_level'):
        config.log_level = LogLevel(ctx.obj['log_level'].upper())
    
    # Setup logging
    logger = setup_logging(config.service_name, config.log_level.value)
    
    # Setup signal handlers
    setup_signal_handlers(logger)
    
    # Log startup information
    logger.info(
        "Starting data ingestion service",
        version=config.service_version,
        environment=config.environment,
        trading_mode=config.trading_mode.value,
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.reload
    )
    
    # Warning for live trading mode
    if config.trading_mode == TradingMode.LIVE:
        logger.warning(
            "🚨 LIVE TRADING MODE ENABLED 🚨",
            message="Real money is at risk! Ensure proper risk controls are in place."
        )
        
        # Require explicit confirmation for live mode
        if not os.getenv("CONFIRM_LIVE_TRADING"):
            logger.error(
                "Live trading mode requires confirmation",
                message="Set CONFIRM_LIVE_TRADING=true environment variable"
            )
            sys.exit(1)
    
    # Start metrics server if enabled
    if config.enable_metrics:
        try:
            start_http_server(config.metrics_port)
            logger.info("Metrics server started", port=config.metrics_port)
        except Exception as e:
            logger.warning("Failed to start metrics server", error=str(e))
    
    # Create and run application
    try:
        app = create_app()
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            workers=1 if config.reload else config.workers,
            reload=config.reload,
            log_level=config.log_level.value.lower(),
            access_log=True,
            use_colors=config.environment != "production",
            loop="asyncio"
        )
        
        # Start server
        server = uvicorn.Server(uvicorn_config)
        server.run()
        
    except Exception as e:
        logger.error("Service startup failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option('--provider', type=click.Choice(['polygon', 'ccxt', 'mock']), 
              default='mock', help='Data provider to test')
@click.option('--symbol', default='AAPL', help='Symbol to test')
@click.option('--duration', type=int, default=10, help='Test duration in seconds')
@click.pass_context
def test(ctx, provider, symbol, duration):
    """Test data feed connections."""
    
    config = get_config_by_env(ctx.obj.get('env'))
    logger = setup_logging(config.service_name, LogLevel.DEBUG.value)
    
    logger.info(
        "Testing data feed",
        provider=provider,
        symbol=symbol,
        duration=duration
    )
    
    async def run_test():
        try:
            if provider == 'polygon':
                await test_polygon_connection()
            elif provider == 'ccxt':
                await test_ccxt_feeds()
            elif provider == 'mock':
                await test_mock_feed()
            else:
                logger.error("Unknown provider", provider=provider)
                return False
            
            logger.info("Test completed successfully")
            return True
            
        except Exception as e:
            logger.error("Test failed", error=str(e))
            return False
    
    # Run test
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--symbols', default='AAPL,GOOGL,MSFT', help='Comma-separated symbols')
@click.option('--data-types', default='ticks,quotes', help='Comma-separated data types')
@click.option('--duration', type=int, default=60, help='Collection duration in seconds')
@click.option('--output', default=None, help='Output file path')
@click.pass_context
def collect(ctx, symbols, data_types, duration, output):
    """Collect market data for analysis."""
    
    config = get_config_by_env(ctx.obj.get('env'))
    logger = setup_logging(config.service_name, LogLevel.INFO.value)
    
    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    data_type_list = [dt.strip().lower() for dt in data_types.split(',')]
    
    logger.info(
        "Starting data collection",
        symbols=symbol_list,
        data_types=data_type_list,
        duration=duration,
        output=output
    )
    
    # TODO: Implement data collection logic
    logger.info("Data collection feature coming soon!")


@cli.command()
@click.pass_context
def config_check(ctx):
    """Validate configuration and show current settings."""
    
    config = get_config_by_env(ctx.obj.get('env'))
    logger = setup_logging(config.service_name, LogLevel.INFO.value)
    
    logger.info("Configuration validation started")
    
    # Validate configuration
    issues = []
    
    # Check required settings
    if config.trading_mode == TradingMode.LIVE and not config.polygon_api_key:
        issues.append("Live trading requires Polygon API key")
    
    if not config.symbols:
        issues.append("No symbols configured for monitoring")
    
    if not config.kafka_bootstrap_servers:
        issues.append("Kafka bootstrap servers not configured")
    
    # Check connections
    try:
        # Test database connection
        from sqlalchemy import create_engine
        engine = create_engine(config.get_database_url())
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✓ Database connection successful")
    except Exception as e:
        issues.append(f"Database connection failed: {e}")
    
    try:
        # Test Redis connection
        import redis
        r = redis.from_url(config.get_redis_url())
        r.ping()
        logger.info("✓ Redis connection successful")
    except Exception as e:
        issues.append(f"Redis connection failed: {e}")
    
    # Print configuration summary
    click.echo("\n" + "="*50)
    click.echo("CONFIGURATION SUMMARY")
    click.echo("="*50)
    click.echo(f"Service Name: {config.service_name}")
    click.echo(f"Version: {config.service_version}")
    click.echo(f"Environment: {config.environment}")
    click.echo(f"Trading Mode: {config.trading_mode.value}")
    click.echo(f"Log Level: {config.log_level.value}")
    click.echo(f"Host:Port: {config.host}:{config.port}")
    click.echo(f"Workers: {config.workers}")
    click.echo(f"Symbols: {', '.join(config.symbols)}")
    click.echo(f"Exchanges: {', '.join(config.exchanges)}")
    click.echo(f"Enable Mock Feeds: {config.enable_mock_feeds}")
    click.echo(f"Polygon API Key: {'***' if config.polygon_api_key != 'demo' else 'demo'}")
    click.echo(f"CCXT Exchanges: {', '.join(config.ccxt_exchanges)}")
    click.echo(f"Kafka Servers: {config.kafka_bootstrap_servers}")
    click.echo(f"Database URL: {config.get_database_url()}")
    click.echo(f"Redis URL: {config.get_redis_url()}")
    
    # Print validation results
    click.echo("\n" + "="*50)
    click.echo("VALIDATION RESULTS")
    click.echo("="*50)
    
    if not issues:
        click.echo("✓ Configuration is valid")
        logger.info("Configuration validation passed")
    else:
        click.echo("✗ Configuration issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")
        logger.error("Configuration validation failed", issues=issues)
        sys.exit(1)


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
@click.pass_context
def show_config(ctx, output_format):
    """Show current configuration."""
    
    config = get_config_by_env(ctx.obj.get('env'))
    
    if output_format == 'json':
        import json
        config_dict = config.dict()
        click.echo(json.dumps(config_dict, indent=2, default=str))
    else:
        import yaml
        config_dict = config.dict()
        click.echo(yaml.dump(config_dict, default_flow_style=False))


def main():
    """Main entry point."""
    # Ensure we're in the right directory
    if __name__ == '__main__':
        cli()
    else:
        # Called as module
        cli(standalone_mode=False)


if __name__ == '__main__':
    main()