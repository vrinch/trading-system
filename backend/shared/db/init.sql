-- TimescaleDB initialization for algorithmic trading system
-- Optimized for high-frequency time-series data with proper indexing

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schemas for logical separation
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS audit;

-- Market data tables optimized for high-frequency ingestion
CREATE TABLE market_data.ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    side CHAR(1) CHECK (side IN ('B', 'S', 'U')), -- Buy/Sell/Unknown
    conditions TEXT[],
    sequence_number BIGINT,
    trade_id VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data.ticks', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create composite index for fast symbol+time queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ticks_symbol_timestamp 
ON market_data.ticks (symbol, timestamp DESC);

-- Create index for exchange queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ticks_exchange 
ON market_data.ticks (exchange, timestamp DESC);

-- OHLCV candles table
CREATE TABLE market_data.candles (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    timeframe VARCHAR(8) NOT NULL, -- 1m, 5m, 1h, 1d
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    trade_count INTEGER,
    vwap DECIMAL(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, exchange, timeframe)
);

SELECT create_hypertable('market_data.candles', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candles_symbol_timeframe 
ON market_data.candles (symbol, timeframe, timestamp DESC);

-- Level 2 order book data
CREATE TABLE market_data.order_book (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    side CHAR(1) CHECK (side IN ('B', 'S')),
    price DECIMAL(20,8) NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    level INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('market_data.order_book', 'timestamp',
    chunk_time_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orderbook_symbol_side 
ON market_data.order_book (symbol, side, timestamp DESC);

-- Trading positions table
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    side CHAR(1) CHECK (side IN ('L', 'S')), -- Long/Short
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    commission DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(16) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CLOSING')),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_strategy_symbol 
ON trading.positions (strategy_id, symbol, status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_opened_at 
ON trading.positions (opened_at DESC);

-- Orders table with execution tracking
CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_order_id VARCHAR(64) UNIQUE NOT NULL,
    strategy_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    order_type VARCHAR(16) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'ICEBERG', 'TWAP', 'VWAP')),
    side CHAR(1) CHECK (side IN ('B', 'S')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    avg_fill_price DECIMAL(20,8),
    status VARCHAR(16) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED')),
    time_in_force VARCHAR(8) DEFAULT 'GTC' CHECK (time_in_force IN ('GTC', 'IOC', 'FOK', 'DAY')),
    broker_order_id VARCHAR(64),
    commission DECIMAL(20,8) DEFAULT 0,
    reject_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_strategy_symbol 
ON trading.orders (strategy_id, symbol, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_status_created 
ON trading.orders (status, created_at DESC);

-- Order fills for granular execution tracking
CREATE TABLE trading.fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    fill_id VARCHAR(64),
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    side CHAR(1) CHECK (side IN ('B', 'S')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0,
    liquidity CHAR(1) CHECK (liquidity IN ('M', 'T')), -- Maker/Taker
    filled_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fills_order_id 
ON trading.fills (order_id, filled_at DESC);

-- Risk metrics and limits
CREATE TABLE risk.portfolio_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(64) NOT NULL,
    total_equity DECIMAL(20,8) NOT NULL,
    available_margin DECIMAL(20,8) NOT NULL,
    used_margin DECIMAL(20,8) NOT NULL,
    unrealized_pnl DECIMAL(20,8) NOT NULL,
    realized_pnl DECIMAL(20,8) NOT NULL,
    daily_pnl DECIMAL(20,8) NOT NULL,
    max_drawdown DECIMAL(20,8) NOT NULL,
    var_95 DECIMAL(20,8), -- Value at Risk 95%
    sharpe_ratio DECIMAL(10,6),
    leverage_ratio DECIMAL(10,4),
    position_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('risk.portfolio_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_metrics_strategy 
ON risk.portfolio_metrics (strategy_id, timestamp DESC);

-- Risk limits and controls
CREATE TABLE risk.limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(64) NOT NULL,
    limit_type VARCHAR(32) NOT NULL, -- MAX_LEVERAGE, MAX_POSITION_SIZE, DAILY_LOSS, etc.
    symbol VARCHAR(32), -- NULL for portfolio-wide limits
    value DECIMAL(20,8) NOT NULL,
    current_value DECIMAL(20,8) DEFAULT 0,
    breach_count INTEGER DEFAULT 0,
    last_breach_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_limits_strategy_type_symbol 
ON risk.limits (strategy_id, limit_type, COALESCE(symbol, ''));

-- ML features store
CREATE TABLE ml.features (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    feature_set VARCHAR(64) NOT NULL,
    features JSONB NOT NULL,
    target DECIMAL(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('ml.features', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_symbol_set 
ON ml.features (symbol, feature_set, timestamp DESC);

-- ML model registry
CREATE TABLE ml.models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL,
    model_type VARCHAR(32) NOT NULL, -- LIGHTGBM, LSTM, etc.
    target_symbol VARCHAR(32),
    hyperparameters JSONB,
    metrics JSONB, -- validation metrics
    artifact_path TEXT, -- S3/MinIO path
    training_start TIMESTAMPTZ NOT NULL,
    training_end TIMESTAMPTZ NOT NULL,
    status VARCHAR(16) DEFAULT 'TRAINING' CHECK (status IN ('TRAINING', 'VALIDATING', 'SHADOW', 'LIVE', 'ARCHIVED')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    UNIQUE(name, version)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_models_name_status 
ON ml.models (name, status, created_at DESC);

-- Audit trail for compliance
CREATE TABLE audit.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(32) NOT NULL, -- ORDER_PLACED, RISK_BREACH, MODEL_SWITCH, etc.
    user_id VARCHAR(64),
    strategy_id VARCHAR(64),
    component VARCHAR(32) NOT NULL, -- execution, risk, ml, etc.
    message TEXT NOT NULL,
    details JSONB,
    severity VARCHAR(16) DEFAULT 'INFO' CHECK (severity IN ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')),
    trace_id VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('audit.events', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_type_timestamp 
ON audit.events (event_type, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_strategy_timestamp 
ON audit.events (strategy_id, timestamp DESC) WHERE strategy_id IS NOT NULL;

-- Continuous aggregates for performance optimization
CREATE MATERIALIZED VIEW market_data.candles_1h
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', timestamp) AS timestamp,
       symbol,
       exchange,
       first(price, timestamp) AS open_price,
       max(price) AS high_price,
       min(price) AS low_price,
       last(price, timestamp) AS close_price,
       sum(size) AS volume,
       count(*) AS trade_count,
       avg(price) AS vwap
FROM market_data.ticks
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY time_bucket('1 hour', timestamp), symbol, exchange
WITH NO DATA;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('market_data.candles_1h',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Data retention policies
SELECT add_retention_policy('market_data.ticks', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('market_data.order_book', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('audit.events', INTERVAL '1 year', if_not_exists => TRUE);

-- Create users and permissions
CREATE USER data_ingest_user WITH PASSWORD 'ingest_pass_2024';
CREATE USER strategy_user WITH PASSWORD 'strategy_pass_2024';
CREATE USER execution_user WITH PASSWORD 'execution_pass_2024';
CREATE USER ml_user WITH PASSWORD 'ml_pass_2024';
CREATE USER monitoring_user WITH PASSWORD 'monitoring_pass_2024';

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA market_data TO data_ingest_user;
GRANT INSERT, SELECT ON ALL TABLES IN SCHEMA market_data TO data_ingest_user;

GRANT USAGE ON SCHEMA market_data, trading, risk TO strategy_user;
GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO strategy_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO strategy_user;
GRANT INSERT, SELECT, UPDATE ON ALL TABLES IN SCHEMA risk TO strategy_user;

GRANT USAGE ON SCHEMA trading, risk, audit TO execution_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO execution_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO execution_user;
GRANT INSERT ON ALL TABLES IN SCHEMA audit TO execution_user;

GRANT USAGE ON SCHEMA market_data, ml TO ml_user;
GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO ml_user;

GRANT USAGE ON ALL SCHEMAS TO monitoring_user;
GRANT SELECT ON ALL TABLES IN ALL SCHEMAS TO monitoring_user;

-- Performance optimizations
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb,pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Enable query planning insights
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET pg_stat_statements.track = all;

-- Trigger for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_limits_updated_at BEFORE UPDATE ON risk.limits
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

-- Vacuum and analyze for optimal performance
VACUUM ANALYZE;

-- Insert initial test data
INSERT INTO risk.limits (strategy_id, limit_type, value) VALUES
('momentum_v1', 'MAX_LEVERAGE', 3.0),
('momentum_v1', 'DAILY_LOSS', 1000.0),
('momentum_v1', 'MAX_POSITION_SIZE', 10000.0),
('mean_reversion_v1', 'MAX_LEVERAGE', 2.0),
('mean_reversion_v1', 'DAILY_LOSS', 500.0);

-- Create notification functions for real-time alerts
CREATE OR REPLACE FUNCTION notify_risk_breach()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.current_value > NEW.value THEN
        PERFORM pg_notify('risk_breach', 
            json_build_object(
                'strategy_id', NEW.strategy_id,
                'limit_type', NEW.limit_type,
                'symbol', NEW.symbol,
                'current_value', NEW.current_value,
                'limit_value', NEW.value,
                'timestamp', NOW()
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER risk_breach_notification 
    AFTER UPDATE ON risk.limits
    FOR EACH ROW EXECUTE PROCEDURE notify_risk_breach();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Trading system database initialized successfully';
    RAISE NOTICE 'TimescaleDB version: %', (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb');
    RAISE NOTICE 'Database is ready for high-frequency trading operations';
END $$;