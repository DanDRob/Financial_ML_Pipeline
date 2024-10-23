"""
Trade execution and order management system.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

import numpy as np
import pandas as pd


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    # Execution parameters
    max_order_size: float = 0.1
    min_order_size: float = 0.01
    order_chunk_size: float = 0.05
    max_slippage: float = 0.002

    # Trading schedule
    market_open: str = "09:30:00"
    market_close: str = "16:00:00"
    trading_days: List[int] = None  # 0=Monday, 6=Sunday

    # Cost parameters
    commission_rate: float = 0.001
    min_commission: float = 1.0
    market_impact_factor: float = 0.1

    # Execution algorithms
    default_algo: str = "VWAP"
    use_dynamic_algo: bool = True
    participation_rate: float = 0.1


class Order:
    """Order representation class."""

    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        quantity: float,
        direction: int,  # 1 for buy, -1 for sell
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        algo: Optional[str] = None
    ):
        """Initialize order."""
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.algo = algo

        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_price = 0.0
        self.commission = 0.0
        self.submit_time = None
        self.fill_time = None

        self.child_orders = []
        self.parent_order = None


class ExecutionManager:
    """Execution and order management system."""

    def __init__(self, config: ExecutionConfig):
        """
        Initialize execution manager.

        Args:
            config: Execution configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.orders = {}  # Order ID to Order mapping
        self.positions = {}  # Symbol to Position mapping
        self.fills = []  # List of fills
        self.pending_orders = []  # List of pending orders

        # Initialize execution algorithms
        self.execution_algos = {
            'VWAP': self._vwap_execution,
            'TWAP': self._twap_execution,
            'POV': self._pov_execution,
            'ADAPTIVE': self._adaptive_execution
        }

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        direction: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        algo: Optional[str] = None
    ) -> Order:
        """
        Submit a new order.

        Args:
            symbol: Asset symbol
            quantity: Order quantity
            order_type: Type of order
            direction: Order direction (1=buy, -1=sell)
            limit_price: Optional limit price
            stop_price: Optional stop price
            algo: Optional execution algorithm

        Returns:
            Submitted order
        """
        try:
            # Validate order parameters
            self._validate_order_parameters(
                symbol, quantity, order_type, direction,
                limit_price, stop_price
            )

            # Create order
            order = Order(
                symbol=symbol,
                order_type=order_type,
                quantity=quantity,
                direction=direction,
                limit_price=limit_price,
                stop_price=stop_price,
                algo=algo or self.config.default_algo
            )

            # Split into child orders if necessary
            if quantity > self.config.order_chunk_size:
                child_orders = self._split_order(order)
                order.child_orders = child_orders
                for child in child_orders:
                    child.parent_order = order

            # Store order
            self.orders[order.order_id] = order

            # Submit order for execution
            if order.child_orders:
                for child in order.child_orders:
                    self._submit_for_execution(child)
            else:
                self._submit_for_execution(order)

            return order

        except Exception as e:
            self.logger.error(f"Order submission failed: {str(e)}")
            raise

    def _validate_order_parameters(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        direction: int,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> None:
        """Validate order parameters."""
        if quantity < self.config.min_order_size:
            raise ValueError(f"Order size {quantity} below minimum {
                             self.config.min_order_size}")

        if quantity > self.config.max_order_size:
            raise ValueError(f"Order size {quantity} above maximum {
                             self.config.max_order_size}")

        if direction not in [-1, 1]:
            raise ValueError(f"Invalid direction {direction}")

        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("Limit price required for limit order")

        if order_type == OrderType.STOP and stop_price is None:
            raise ValueError("Stop price required for stop order")

    def _split_order(self, order: Order) -> List[Order]:
        """Split large order into smaller chunks."""
        try:
            chunk_size = self.config.order_chunk_size
            n_chunks = int(np.ceil(order.quantity / chunk_size))

            child_orders = []
            remaining_qty = order.quantity

            for i in range(n_chunks):
                child_qty = min(chunk_size, remaining_qty)

                child_order = Order(
                    symbol=order.symbol,
                    order_type=order.order_type,
                    quantity=child_qty,
                    direction=order.direction,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    algo=order.algo
                )

                child_orders.append(child_order)
                remaining_qty -= child_qty

            return child_orders

        except Exception as e:
            self.logger.error(f"Order splitting failed: {str(e)}")
            raise

    def _submit_for_execution(self, order: Order) -> None:
        """Submit order for execution."""
        try:
            order.submit_time = datetime.now()
            order.status = OrderStatus.SUBMITTED

            # Add to pending orders
            self.pending_orders.append(order)

            # Execute based on algorithm
            if order.algo in self.execution_algos:
                self.execution_algos[order.algo](order)
            else:
                self._default_execution(order)

        except Exception as e:
            self.logger.error(f"Order execution submission failed: {str(e)}")
            order.status = OrderStatus.REJECTED
            raise

    def _vwap_execution(self, order: Order) -> None:
        """Execute order using VWAP algorithm."""
        try:
            # Get volume profile for scheduling
            volume_profile = self._get_volume_profile(order.symbol)

            # Calculate target participation
            participation = self.config.participation_rate
            if self.config.use_dynamic_algo:
                participation = self._calculate_dynamic_participation(
                    order.symbol,
                    order.quantity
                )

            # Schedule order
            schedule = self._generate_vwap_schedule(
                order.quantity,
                volume_profile,
                participation
            )

            # Create child orders based on schedule
            for time_slot, qty in schedule.items():
                child = Order(
                    symbol=order.symbol,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    direction=order.direction,
                    algo=None
                )
                order.child_orders.append(child)
                child.parent_order = order

                # Schedule child order execution
                self._schedule_order(child, time_slot)

        except Exception as e:
            self.logger.error(f"VWAP execution failed: {str(e)}")
            raise

    def _twap_execution(self, order: Order) -> None:
        """Execute order using TWAP algorithm."""
        try:
            # Calculate time intervals
            start_time = datetime.strptime(
                self.config.market_open, "%H:%M:%S").time()
            end_time = datetime.strptime(
                self.config.market_close, "%H:%M:%S").time()
            trading_minutes = (end_time.hour - start_time.hour) * \
                60 + (end_time.minute - start_time.minute)

            # Calculate interval size
            n_intervals = int(trading_minutes / 5)  # 5-minute intervals
            qty_per_interval = order.quantity / n_intervals

            # Create schedule
            current_time = datetime.now().time()
            remaining_intervals = n_intervals

            if current_time > start_time:
                elapsed_intervals = int((current_time.hour - start_time.hour) * 12 +
                                        (current_time.minute - start_time.minute) / 5)
                remaining_intervals -= elapsed_intervals

            qty_per_interval = order.quantity / remaining_intervals

            # Create and schedule child orders
            for i in range(remaining_intervals):
                child = Order(
                    symbol=order.symbol,
                    order_type=OrderType.MARKET,
                    quantity=qty_per_interval,
                    direction=order.direction,
                    algo=None
                )
                order.child_orders.append(child)
                child.parent_order = order

                # Schedule execution
                execution_time = datetime.now().replace(
                    hour=current_time.hour,
                    minute=current_time.minute + (i * 5)
                )
                self._schedule_order(child, execution_time)

        except Exception as e:
            self.logger.error(f"TWAP execution failed: {str(e)}")
            raise

    def _pov_execution(self, order: Order) -> None:
        """Execute order using Percentage of Volume algorithm."""
        try:
            # Get real-time volume data
            volume_data = self._get_realtime_volume(order.symbol)

            # Calculate target participation
            participation = self.config.participation_rate
            if self.config.use_dynamic_algo:
                participation = self._calculate_dynamic_participation(
                    order.symbol,
                    order.quantity,
                    volume_data
                )

            # Monitor volume and execute
            while order.filled_quantity < order.quantity:
                current_volume = self._get_current_volume(order.symbol)
                target_qty = current_volume * participation

                # Create child order
                remaining_qty = order.quantity - order.filled_quantity
                child_qty = min(target_qty, remaining_qty)

                if child_qty >= self.config.min_order_size:
                    child = Order(
                        symbol=order.symbol,
                        order_type=OrderType.MARKET,
                        quantity=child_qty,
                        direction=order.direction,
                        algo=None
                    )
                    order.child_orders.append(child)
                    child.parent_order = order

                    # Execute immediately
                    self._execute_market_order(child)

        except Exception as e:
            self.logger.error(f"POV execution failed: {str(e)}")
            raise

    def _adaptive_execution(self, order: Order) -> None:
        """Execute order using adaptive algorithm."""
        try:
            # Get market state
            volatility = self._calculate_volatility(order.symbol)
            spread = self._calculate_spread(order.symbol)
            market_impact = self._estimate_market_impact(
                order.symbol,
                order.quantity
            )

            # Choose execution strategy based on market conditions
            if volatility > 0.02:  # High volatility
                if spread > 0.0005:  # Wide spread
                    self._vwap_execution(order)
                else:
                    self._pov_execution(order)
            else:  # Low volatility
                if market_impact > 0.001:  # High market impact
                    self._twap_execution(order)
                else:
                    self._default_execution(order)

        except Exception as e:
            self.logger.error(f"Adaptive execution failed: {str(e)}")
            raise

    def _default_execution(self, order: Order) -> None:
        """Default market order execution."""
        try:
            # Calculate expected market impact
            market_impact = self._estimate_market_impact(
                order.symbol,
                order.quantity
            )

            # Execute if impact is acceptable
            if market_impact <= self.config.max_slippage:
                self._execute_market_order(order)
            else:
                # Split into smaller orders
                child_orders = self._split_order(order)
                order.child_orders = child_orders

                for child in child_orders:
                    child.parent_order = order
                    self._execute_market_order(child)

        except Exception as e:
            self.logger.error(f"Default execution failed: {str(e)}")
            raise

    def _execute_market_order(self, order: Order) -> None:
        """Execute market order."""
        try:
            # Get current market price
            current_price = self._get_market_price(order.symbol)

            # Calculate execution price with slippage
            slippage = self._calculate_slippage(
                order.symbol,
                order.quantity,
                order.direction
            )
            execution_price = current_price * (1 + slippage * order.direction)

            # Calculate commission
            commission = max(
                self.config.min_commission,
                order.quantity * execution_price * self.config.commission_rate
            )

            # Record fill
            fill = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'price': execution_price,
                'direction': order.direction,
                'commission': commission,
                'timestamp': datetime.now()
            }
            self.fills.append(fill)

            # Update order status
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            order.commission = commission
            order.status = OrderStatus.FILLED
            order.fill_time = datetime.now()

            # Update position
            self._update_position(
                order.symbol,
                order.quantity * order.direction,
                execution_price
            )

            # Remove from pending orders
            if order in self.pending_orders:
                self.pending_orders.remove(order)

            self.logger.info(f"Order {order.order_id} executed: {
                             order.quantity} {order.symbol} @ {execution_price}")

        except Exception as e:
            self.logger.error(f"Market order execution failed: {str(e)}")
            raise

    def _update_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """Update position after execution."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'average_price': 0,
                'cost_basis': 0
            }

        position = self.positions[symbol]
        old_quantity = position['quantity']
        old_cost = position['cost_basis']

        # Update position
        new_quantity = old_quantity + quantity
        new_cost = old_cost + (quantity * price)

        if new_quantity != 0:
            position['average_price'] = new_cost / new_quantity

        position['quantity'] = new_quantity
        position['cost_basis'] = new_cost

    def _estimate_market_impact(
        self,
        symbol: str,
        quantity: float
    ) -> float:
        """Estimate market impact of order."""
        try:
            # Get average daily volume
            adv = self._get_average_daily_volume(symbol)

            # Calculate market impact using square root model
            market_impact = (
                self.config.market_impact_factor *
                np.sqrt(quantity / adv)
            )

            return market_impact

        except Exception as e:
            self.logger.error(f"Market impact estimation failed: {str(e)}")
            return float('inf')

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")

            order = self.orders[order_id]

            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                raise ValueError(f"Order {order_id} cannot be cancelled")

            # Cancel child orders
            for child in order.child_orders:
                if child.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    child.status = OrderStatus.CANCELLED

            order.status = OrderStatus.CANCELLED

            # Remove from pending orders
            if order in self.pending_orders:
                self.pending_orders.remove(order)

            return True

        except Exception as e:
            self.logger.error(f"Order cancellation failed: {str(e)}")
            return False

    def get_order_status(self, order_id: str) -> Dict:
        """Get current order status and details."""
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")

            order = self.orders[order_id]

            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': order.status,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'average_price': order.average_price,
                'commission': order.commission,
                'submit_time': order.submit_time,
                'fill_time': order.fill_time
            }

        except Exception as e:
            self.logger.error(f"Order status retrieval failed: {str(e)}")
            raise

    def get_position(self, symbol: str) -> Dict:
        """Get current position for symbol."""
        return self.positions.get(symbol, {
            'quantity': 0,
            'average_price': 0,
            'cost_basis': 0
        })

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        return self.positions

    def get_fills(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get fill history with optional filters."""
        fills = self.fills

        if symbol:
            fills = [f for f in fills if f['symbol'] == symbol]

        if start_time:
            fills = [f for f in fills if f['timestamp'] >= start_time]

        if end_time:
            fills = [f for f in fills if f['timestamp'] <= end_time]

        return fills
