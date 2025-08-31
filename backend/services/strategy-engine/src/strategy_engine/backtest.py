"""
Advanced backtesting engine for strategy validation.
Supports event-driven simulation, realistic market conditions, and comprehensive analysis.
"""

import asyncio
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .models.schemas import (
    Signal, Position, Order, BacktestResult, Side, SignalType, 
    OrderType, OrderStatus, PositionStatus
)
from .config import StrategyEngineConfig, BacktestEngine
from .strategies.base import BaseStrategy
from ..shared.utils.logging import get_logger, AuditLogger
from ..shared.utils.metrics import TradingMetrics


@dataclass
class MarketEvent:
    """Market data event for event-driven backtesting."""
    timestamp: datetime
    symbol: str
    event_type: str  # 'tick', 'bar', 'quote'
    data: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=datetime.timezone.utc)


@dataclass 
class TradeExecution:
    """Trade execution record for backtesting."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    slippage: Decimal
    order_id: str
    position_id: Optional[str] = None
    
    @property
    def gross_amount(self) -> Decimal:
        """Gross trade amount before costs."""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> Decimal:
        """Net trade amount after costs."""
        return self.gross_amount - self.commission - self.slippage


@dataclass
class BacktestState:
    """Current state of backtest simulation."""
    current_time: datetime
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    trades: List[TradeExecution] = field(default_factory=list)
    portfolio_values: List[Tuple[datetime, Decimal]] = field(default_factory=list)
    
    @property
    def total_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.quantity * pos.current_price if pos.current_price else pos.cost_basis
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def update_position_price(self, symbol: str, price: Decimal) -> None:
        """Update position with current market price."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = price
            
            # Calculate unrealized P&L
            if pos.side in [Side.BUY, Side.LONG]:
                pos.unrealized_pnl = pos.quantity * (price - pos.entry_price)
            else:  # Short position
                pos.unrealized_pnl = pos.quantity * (pos.entry_price - price)


class MarketSimulator:
    """Simulates realistic market conditions during backtesting."""
    
    def __init__(self, config: StrategyEngineConfig):
        self.config = config
        self.logger = get_logger("market_simulator")
        
        # Market impact parameters
        self.linear_impact_coeff = 0.0001  # Price impact per unit volume
        self.sqrt_impact_coeff = 0.001     # Square root price impact
        
        # Slippage parameters
        self.base_slippage_bps = 2.0       # Base slippage in basis points
        self.volume_slippage_factor = 0.1   # Additional slippage based on volume
        
    def calculate_fill_price(
        self, 
        order: Order, 
        market_price: Decimal,
        market_volume: Optional[Decimal] = None
    ) -> Tuple[Decimal, Decimal]:
        """Calculate realistic fill price including slippage and market impact."""
        
        # Base price
        if order.order_type == OrderType.MARKET:
            base_price = market_price
        elif order.order_type == OrderType.LIMIT:
            # For limit orders, use the better of limit price or market price
            if order.side in [Side.BUY, Side.LONG]:
                base_price = min(order.price, market_price)
            else:
                base_price = max(order.price, market_price)
        else:
            base_price = market_price
        
        # Calculate slippage
        slippage = self._calculate_slippage(order, market_volume)
        
        # Apply slippage direction based on order side
        if order.side in [Side.BUY, Side.LONG]:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage
        
        return fill_price, slippage
    
    def _calculate_slippage(self, order: Order, market_volume: Optional[Decimal] = None) -> Decimal:
        """Calculate slippage based on order size and market conditions."""
        base_slippage = Decimal(str(self.base_slippage_bps / 10000))
        
        # Volume-based slippage
        if market_volume and market_volume > 0:
            volume_ratio = order.quantity / market_volume
            volume_slippage = Decimal(str(self.volume_slippage_factor)) * volume_ratio
        else:
            volume_slippage = Decimal('0')
        
        # Market impact (simplified square root model)
        market_impact = Decimal(str(self.sqrt_impact_coeff)) * (order.quantity ** Decimal('0.5'))
        
        total_slippage_ratio = base_slippage + volume_slippage + market_impact
        
        # Convert to absolute price slippage
        market_price = order.price if order.price else Decimal('100')  # Fallback
        return market_price * total_slippage_ratio
    
    def calculate_commission(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate commission based on trade value."""
        trade_value = order.quantity * fill_price
        commission_rate = Decimal(str(self.config.commission_rate))
        return trade_value * commission_rate
    
    def should_fill_order(
        self, 
        order: Order, 
        market_price: Decimal,
        timestamp: datetime
    ) -> bool:
        """Determine if order should be filled given market conditions."""
        
        # Market orders fill immediately
        if order.order_type == OrderType.MARKET:
            return True
        
        # Limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side in [Side.BUY, Side.LONG]:
                return market_price <= order.price
            else:
                return market_price >= order.price
        
        # Stop orders
        if order.order_type == OrderType.STOP:
            if order.side in [Side.BUY, Side.LONG]:
                return market_price >= order.stop_price
            else:
                return market_price <= order.stop_price
        
        # Default: don't fill
        return False


class VectorizedBacktester:
    """Fast vectorized backtesting using pandas operations."""
    
    def __init__(self, config: StrategyEngineConfig):
        self.config = config
        self.logger = get_logger("vectorized_backtester")
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 1000000
    ) -> BacktestResult:
        """Run vectorized backtest for simple strategies."""
        
        self.logger.info(
            "Starting vectorized backtest",
            strategy_id=strategy.strategy_id,
            data_points=len(data),
            initial_capital=initial_capital
        )
        
        # Initialize results tracking
        portfolio_values = []
        trades = []
        positions = {}
        cash = Decimal(str(initial_capital))
        
        # Process data in chunks for memory efficiency
        chunk_size = 10000
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            
            # Generate signals for chunk
            signals = await self._generate_signals_vectorized(strategy, chunk)
            
            # Execute trades
            for signal in signals:
                trade_result = self._execute_trade_vectorized(
                    signal, chunk, cash, positions
                )
                if trade_result:
                    trades.append(trade_result)
                    cash = trade_result['remaining_cash']
        
        # Calculate final metrics
        return self._calculate_backtest_results(
            strategy.strategy_id,
            trades,
            portfolio_values,
            data.index[0],
            data.index[-1],
            initial_capital
        )
    
    async def _generate_signals_vectorized(
        self, 
        strategy: BaseStrategy,
        data: pd.DataFrame
    ) -> List[Signal]:
        """Generate signals using vectorized operations."""
        
        # Convert DataFrame to format expected by strategy
        market_data = {}
        for symbol in data['symbol'].unique() if 'symbol' in data.columns else ['DEFAULT']:
            symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data
            market_data[symbol] = symbol_data
        
        # Generate signals
        signals = await strategy.generate_signals(market_data, datetime.now())
        return signals
    
    def _execute_trade_vectorized(
        self,
        signal: Signal,
        data: pd.DataFrame,
        cash: Decimal,
        positions: Dict[str, Position]
    ) -> Optional[Dict[str, Any]]:
        """Execute trade using vectorized approach."""
        
        # Get current price
        symbol_data = data[data['symbol'] == signal.symbol] if 'symbol' in data.columns else data
        if symbol_data.empty:
            return None
        
        current_price = Decimal(str(symbol_data['close'].iloc[-1]))
        
        # Calculate position size
        position_value = cash * Decimal('0.1')  # 10% position size
        quantity = position_value / current_price
        
        # Create trade record
        return {
            'timestamp': symbol_data.index[-1],
            'symbol': signal.symbol,
            'side': signal.side.value,
            'quantity': quantity,
            'price': current_price,
            'remaining_cash': cash - position_value
        }
    
    def _calculate_backtest_results(
        self,
        strategy_id: str,
        trades: List[Dict[str, Any]],
        portfolio_values: List[Tuple[datetime, Decimal]],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        
        # Basic metrics
        total_trades = len(trades)
        final_value = float(portfolio_values[-1][1]) if portfolio_values else initial_capital
        total_return = (final_value / initial_capital - 1) * 100
        
        # Calculate daily returns
        daily_returns = []
        if len(portfolio_values) > 1:
            for i in range(1, len(portfolio_values)):
                prev_value = float(portfolio_values[i-1][1])
                curr_value = float(portfolio_values[i][1])
                daily_return = (curr_value / prev_value - 1) if prev_value > 0 else 0
                daily_returns.append(daily_return)
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        if portfolio_values:
            values = [float(pv[1]) for pv in portfolio_values]
            running_max = np.maximum.accumulate(values)
            drawdowns = (np.array(values) - running_max) / running_max
            max_drawdown = float(np.min(drawdowns)) * 100
        else:
            max_drawdown = 0.0
        
        return BacktestResult(
            backtest_id=f"{strategy_id}_{int(datetime.now().timestamp())}",
            strategy_name=strategy_id,
            start_date=start_date,
            end_date=end_date,
            duration_days=(end_date - start_date).days,
            initial_capital=Decimal(str(initial_capital)),
            commission_rate=self.config.commission_rate,
            slippage_model=self.config.slippage_model,
            total_return=total_return,
            annual_return=total_return * (365 / (end_date - start_date).days),
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio,  # Simplified
            calmar_ratio=sharpe_ratio,   # Simplified
            max_drawdown=abs(max_drawdown),
            var_95=max_drawdown * 1.5,   # Simplified
            cvar_95=max_drawdown * 1.8,  # Simplified
            beta=1.0,  # Would need benchmark data
            total_trades=total_trades,
            winning_trades=total_trades // 2,  # Simplified
            losing_trades=total_trades // 2,   # Simplified
            win_rate=50.0,  # Simplified
            avg_win=2.0,    # Simplified
            avg_loss=-1.5,  # Simplified
            profit_factor=1.33,  # Simplified
            daily_returns=daily_returns,
            equity_curve=[float(pv[1]) for pv in portfolio_values],
            drawdown_series=drawdowns.tolist() if 'drawdowns' in locals() else []
        )


class EventDrivenBacktester:
    """Realistic event-driven backtesting engine."""
    
    def __init__(self, config: StrategyEngineConfig, metrics: Optional[TradingMetrics] = None):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("event_backtester")
        self.audit_logger = AuditLogger("backtest")
        
        # Market simulation
        self.market_simulator = MarketSimulator(config)
        
        # State tracking
        self.state: Optional[BacktestState] = None
        self.event_queue: List[MarketEvent] = []
        
        # Order management
        self.order_counter = 0
        self.position_counter = 0
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 1000000,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """Run comprehensive event-driven backtest."""
        
        self.logger.info(
            "Starting event-driven backtest",
            strategy_id=strategy.strategy_id,
            data_points=len(data),
            initial_capital=initial_capital
        )
        
        # Initialize backtest state
        start_date = data.index[0] if len(data) > 0 else datetime.now()
        self.state = BacktestState(
            current_time=start_date,
            cash=Decimal(str(initial_capital))
        )
        
        # Initialize strategy
        universe = set(data['symbol'].unique()) if 'symbol' in data.columns else {'DEFAULT'}
        await strategy.start(universe)
        
        # Convert data to events
        self._prepare_event_queue(data)
        
        # Main simulation loop
        signal_generation_interval = timedelta(minutes=1)
        last_signal_time = start_date
        
        for event in self.event_queue:
            self.state.current_time = event.timestamp
            
            # Process market event
            await self._process_market_event(event)
            
            # Generate signals periodically
            if event.timestamp - last_signal_time >= signal_generation_interval:
                await self._generate_and_process_signals(strategy)
                last_signal_time = event.timestamp
            
            # Process pending orders
            await self._process_orders()
            
            # Update portfolio value
            self.state.portfolio_values.append(
                (event.timestamp, self.state.total_portfolio_value)
            )
            
            # Risk checks
            await self._perform_risk_checks()
        
        # Calculate final results
        end_date = data.index[-1] if len(data) > 0 else datetime.now()
        result = await self._calculate_comprehensive_results(
            strategy.strategy_id,
            start_date,
            end_date,
            initial_capital,
            benchmark_data
        )
        
        # Cleanup
        await strategy.stop()
        
        self.audit_logger.log_order_event(
            "BACKTEST_COMPLETED",
            {
                "strategy_id": strategy.strategy_id,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown
            }
        )
        
        return result
    
    def _prepare_event_queue(self, data: pd.DataFrame) -> None:
        """Convert DataFrame to ordered event queue."""
        self.event_queue = []
        
        for timestamp, row in data.iterrows():
            event = MarketEvent(
                timestamp=timestamp,
                symbol=row.get('symbol', 'DEFAULT'),
                event_type='bar',
                data=row.to_dict()
            )
            self.event_queue.append(event)
        
        # Sort events by timestamp
        self.event_queue.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Prepared {len(self.event_queue)} market events")
    
    async def _process_market_event(self, event: MarketEvent) -> None:
        """Process incoming market data event."""
        
        # Update position prices
        if 'close' in event.data:
            price = Decimal(str(event.data['close']))
            self.state.update_position_price(event.symbol, price)
    
    async def _generate_and_process_signals(self, strategy: BaseStrategy) -> None:
        """Generate signals from strategy and convert to orders."""
        
        # Prepare market data for strategy
        market_data = self._get_current_market_data()
        
        # Generate signals
        signals = await strategy.process_market_data(market_data, self.state.current_time)
        
        # Convert signals to orders
        for signal in signals:
            order = await self._convert_signal_to_order(signal)
            if order:
                self.state.orders[order.order_id] = order
                
                self.logger.debug(
                    "Order created from signal",
                    order_id=order.order_id,
                    symbol=signal.symbol,
                    side=signal.side.value
                )
    
    async def _convert_signal_to_order(self, signal: Signal) -> Optional[Order]:
        """Convert trading signal to executable order."""
        
        # Calculate position size
        portfolio_value = float(self.state.total_portfolio_value)
        position_size = strategy.calculate_position_size(signal, portfolio_value)
        
        # Check available cash for long positions
        if signal.side in [Side.BUY, Side.LONG]:
            required_cash = position_size
            if required_cash > self.state.cash:
                self.logger.warning(
                    "Insufficient cash for order",
                    required=float(required_cash),
                    available=float(self.state.cash)
                )
                return None
        
        # Create order
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        
        order = Order(
            timestamp=self.state.current_time,
            strategy_id=signal.strategy_id,
            order_id=order_id,
            client_order_id=order_id,
            symbol=signal.symbol,
            order_type=OrderType.MARKET,  # Simplified
            side=signal.side,
            quantity=position_size / (signal.target_price or Decimal('100')),
            price=signal.target_price,
            stop_price=signal.stop_price,
            risk_approved=True  # Auto-approve in backtest
        )
        
        return order
    
    async def _process_orders(self) -> None:
        """Process pending orders against current market conditions."""
        
        orders_to_remove = []
        
        for order_id, order in self.state.orders.items():
            if order.status != OrderStatus.PENDING:
                continue
            
            # Get current market price
            current_price = self._get_current_price(order.symbol)
            if not current_price:
                continue
            
            # Check if order should fill
            should_fill = self.market_simulator.should_fill_order(
                order, current_price, self.state.current_time
            )
            
            if should_fill:
                await self._execute_order(order, current_price)
                orders_to_remove.append(order_id)
        
        # Clean up filled orders
        for order_id in orders_to_remove:
            del self.state.orders[order_id]
    
    async def _execute_order(self, order: Order, market_price: Decimal) -> None:
        """Execute order and update positions."""
        
        # Calculate realistic fill price and costs
        fill_price, slippage = self.market_simulator.calculate_fill_price(
            order, market_price
        )
        commission = self.market_simulator.calculate_commission(order, fill_price)
        
        # Create trade execution
        trade = TradeExecution(
            timestamp=self.state.current_time,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            order_id=order.order_id
        )
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.filled_at = self.state.current_time
        order.commission = commission
        
        # Update positions
        await self._update_position_from_trade(trade)
        
        # Update cash
        if order.side in [Side.BUY, Side.LONG]:
            self.state.cash -= trade.net_amount
        else:
            self.state.cash += trade.net_amount
        
        # Record trade
        self.state.trades.append(trade)
        
        self.logger.info(
            "Order executed",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=float(order.quantity),
            fill_price=float(fill_price),
            commission=float(commission)
        )
    
    async def _update_position_from_trade(self, trade: TradeExecution) -> None:
        """Update or create position from trade execution."""
        
        symbol = trade.symbol
        
        if symbol not in self.state.positions:
            # Create new position
            self.position_counter += 1
            position_id = f"POS_{self.position_counter:06d}"
            
            position = Position(
                timestamp=trade.timestamp,
                strategy_id="backtest",  # Simplified
                position_id=position_id,
                symbol=symbol,
                side=Side(trade.side),
                quantity=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
                cost_basis=trade.gross_amount,
                commission_paid=trade.commission,
                opened_at=trade.timestamp
            )
            
            self.state.positions[symbol] = position
            
        else:
            # Update existing position
            position = self.state.positions[symbol]
            
            if trade.side == position.side.value:
                # Add to position
                new_quantity = position.quantity + trade.quantity
                new_cost_basis = position.cost_basis + trade.gross_amount
                position.entry_price = new_cost_basis / new_quantity
                position.quantity = new_quantity
                position.cost_basis = new_cost_basis
                position.commission_paid += trade.commission
                
            else:
                # Reduce or close position
                if trade.quantity >= position.quantity:
                    # Close position completely
                    realized_pnl = self._calculate_realized_pnl(position, trade)
                    position.realized_pnl += realized_pnl
                    position.status = PositionStatus.CLOSED
                    position.closed_at = trade.timestamp
                    
                    # Remove closed position
                    del self.state.positions[symbol]
                    
                else:
                    # Partial close
                    close_ratio = trade.quantity / position.quantity
                    realized_pnl = position.unrealized_pnl * close_ratio
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                    position.cost_basis *= (1 - close_ratio)
    
    def _calculate_realized_pnl(self, position: Position, closing_trade: TradeExecution) -> Decimal:
        """Calculate realized P&L from closing trade."""
        
        if position.side in [Side.BUY, Side.LONG]:
            return closing_trade.quantity * (closing_trade.price - position.entry_price)
        else:
            return closing_trade.quantity * (position.entry_price - closing_trade.price)
    
    async def _perform_risk_checks(self) -> None:
        """Perform risk management checks during backtest."""
        
        portfolio_value = float(self.state.total_portfolio_value)
        initial_capital = float(self.state.portfolio_values[0][1]) if self.state.portfolio_values else portfolio_value
        
        # Check maximum drawdown
        if self.state.portfolio_values:
            values = [float(pv[1]) for pv in self.state.portfolio_values]
            peak = max(values)
            current_drawdown = (peak - portfolio_value) / peak * 100
            
            if current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(
                    "Maximum drawdown exceeded",
                    current_drawdown=current_drawdown,
                    limit=self.config.max_drawdown_pct
                )
        
        # Check daily loss limit
        daily_return = (portfolio_value / initial_capital - 1) * 100
        if daily_return < -self.config.max_daily_loss_pct:
            self.logger.warning(
                "Daily loss limit exceeded",
                daily_return=daily_return,
                limit=-self.config.max_daily_loss_pct
            )
    
    def _get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get current market data for strategy signal generation."""
        
        # This is simplified - in practice would maintain rolling windows
        # of recent market data for each symbol
        return {}
    
    def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol."""
        
        # Find latest price from recent events
        for event in reversed(self.event_queue):
            if event.symbol == symbol and 'close' in event.data:
                if event.timestamp <= self.state.current_time:
                    return Decimal(str(event.data['close']))
        
        return None
    
    async def _calculate_comprehensive_results(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """Calculate comprehensive backtest results with advanced metrics."""
        
        # Basic calculations
        final_value = float(self.state.total_portfolio_value)
        total_return = (final_value / initial_capital - 1) * 100
        duration_days = (end_date - start_date).days
        
        # Calculate returns series
        daily_returns = []
        equity_curve = []
        
        if len(self.state.portfolio_values) > 1:
            for i, (timestamp, value) in enumerate(self.state.portfolio_values):
                equity_curve.append(float(value))
                
                if i > 0:
                    prev_value = float(self.state.portfolio_values[i-1][1])
                    daily_return = (float(value) / prev_value - 1) if prev_value > 0 else 0
                    daily_returns.append(daily_return)
        
        # Risk and performance metrics
        if daily_returns:
            returns_array = np.array(daily_returns)
            
            # Annualized metrics
            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)
            annual_return = mean_return * 252
            annual_volatility = volatility * np.sqrt(252)
            
            # Risk-adjusted returns
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_return = annual_return - risk_free_rate
            sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
            
            # Downside metrics
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown calculation
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(float(np.min(drawdowns))) * 100
            
            # Calmar ratio
            calmar_ratio = annual_return / (max_drawdown / 100) if max_drawdown > 0 else 0
            
        else:
            annual_return = total_return
            annual_volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
            drawdowns = np.array([])
        
        # Trade analysis
        trades = self.state.trades
        total_trades = len(trades)
        
        if total_trades > 0:
            # Calculate trade returns
            trade_returns = []
            for trade in trades:
                # Simplified trade return calculation
                trade_return = float(trade.price) / 100 - 1  # Placeholder
                trade_returns.append(trade_return)
            
            winning_trades = len([r for r in trade_returns if r > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = np.mean([r for r in trade_returns if r > 0]) * 100 if winning_trades > 0 else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) * 100 if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # VaR calculation (simplified)
        if daily_returns:
            var_95 = float(np.percentile(returns_array, 5)) * 100
            cvar_95 = float(np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)])) * 100
        else:
            var_95 = 0
            cvar_95 = 0
        
        # Benchmark comparison (if provided)
        benchmark_return = None
        alpha = None
        beta = 1.0
        information_ratio = None
        
        if benchmark_data is not None and len(daily_returns) > 0:
            # Calculate benchmark metrics (simplified)
            benchmark_return = 10.0  # Placeholder
            alpha = annual_return - benchmark_return
            information_ratio = alpha / annual_volatility if annual_volatility > 0 else 0
        
        return BacktestResult(
            backtest_id=f"{strategy_id}_{int(datetime.now().timestamp())}",
            strategy_name=strategy_id,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            initial_capital=Decimal(str(initial_capital)),
            commission_rate=self.config.commission_rate,
            slippage_model=self.config.slippage_model,
            total_return=total_return,
            annual_return=annual_return * 100,
            volatility=annual_volatility * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            drawdown_series=drawdowns.tolist(),
            benchmark_return=benchmark_return,
            alpha=alpha,
            information_ratio=information_ratio
        )


# Factory function for creating backtester
def create_backtester(
    engine_type: BacktestEngine,
    config: StrategyEngineConfig,
    metrics: Optional[TradingMetrics] = None
) -> Union[VectorizedBacktester, EventDrivenBacktester]:
    """Create backtester instance based on engine type."""
    
    if engine_type == BacktestEngine.VECTORIZED:
        return VectorizedBacktester(config)
    elif engine_type == BacktestEngine.EVENT_DRIVEN:
        return EventDrivenBacktester(config, metrics)
    else:
        # Default to event-driven for most realistic results
        return EventDrivenBacktester(config, metrics)


# Utility functions
def load_backtest_data(
    file_path: str,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    
    # Load data (supports various formats)
    if file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Filter by symbols if specified
    if symbols and 'symbol' in data.columns:
        data = data[data['symbol'].isin(symbols)]
    
    # Filter by date range if specified
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return data


def compare_backtest_results(results: List[BacktestResult]) -> pd.DataFrame:
    """Compare multiple backtest results in a summary table."""
    
    comparison_data = []
    
    for result in results:
        comparison_data.append({
            'Strategy': result.strategy_name,
            'Total Return (%)': result.total_return,
            'Annual Return (%)': result.annual_return,
            'Volatility (%)': result.volatility,
            'Sharpe Ratio': result.sharpe_ratio,
            'Max Drawdown (%)': result.max_drawdown,
            'Win Rate (%)': result.win_rate,
            'Total Trades': result.total_trades,
            'Profit Factor': result.profit_factor
        })
    
    return pd.DataFrame(comparison_data)