# MT5 Order Manager
# Handles execution of trades

import MetaTrader5 as mt5
import logging

logger = logging.getLogger(__name__)

class MT5OrderManager:
    def __init__(self):
        pass
        
    def get_positions(self, symbol: str = None):
        """Get current open positions"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
            
        if positions is None:
            return []
            
        # Convert to list of dicts for easier handling
        return [p._asdict() for p in positions]
        
    def close_position(self, ticket: int):
        """Close a specific position by ticket"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found")
            return False
            
        pos = position[0]
        symbol = pos.symbol
        volume = pos.volume
        type_ = pos.type # 0=Buy, 1=Sell
        
        # Close request is opposite order
        # Buy (0) -> Close with Sell (1)
        # Sell (1) -> Close with Buy (0)
        action_type = mt5.ORDER_TYPE_SELL if type_ == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if action_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "Python Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment} ({result.retcode})")
            return False
            
        logger.info(f"Closed position {ticket} for {symbol}")
        return True

    def open_order(self, symbol: str, direction: str, volume: float, stop_loss: float, take_profit: float, magic: int = 0):
        """
        Open a market order.
        direction: 'LONG' or 'SHORT'
        """
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Symbol {symbol} not available")
            return None
            
        if direction == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            # Validate SL/TP
            if stop_loss >= price: 
                logger.warning("Long SL must be below price")
                # stop_loss = 0.0 
            if take_profit <= price and take_profit > 0:
                logger.warning("Long TP must be above price")
                
        elif direction == 'SHORT':
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            if stop_loss <= price and stop_loss > 0:
                logger.warning("Short SL must be above price")
            if take_profit >= price:
                logger.warning("Short TP must be below price")
        else:
            return None
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": 20,
            "magic": magic,
            "comment": "Python Algo",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} ({result.retcode})")
            return None
            
        logger.info(f"Opened {direction} on {symbol} @ {price}, Vol: {volume}, Ticket: {result.order}")
        return result.order

    def modify_sl(self, ticket: int, stop_loss: float):
        """Update Stop Loss for a position"""
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return False
        pos = pos_list[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": float(stop_loss),
            "tp": pos.tp, # Keep existing TP
            "magic": pos.magic
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Common error: SL too close to price
            logger.warning(f"Modify SL failed: {result.comment}")
            return False
            
        return True
