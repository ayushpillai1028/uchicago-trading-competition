import tkinter as tk
from typing import Optional
from utcxchange.utcxchangelib.xchange_client import XChangeClient as utc_bot_pb2 # type: ignore
import asyncio
import logging
from collections import defaultdict
from typing import Optional
from enum import Enum
import grpc
import math
import numpy as np
from scipy.stats import norm
import time
import argparse

STOCK_SYMBOLS = ['APT', 'DLR', 'MKJ', 'AKAV']
# STOCK_SYMBOLS = ['MKJ']
MAX_ORDER_SIZE = 40
MAX_OPEN_ORDERS = 50
MAX_OUTSTANDING_VOLUME = 120
MAX_ABSOLUTE_POSITION = 200

class Side(Enum):
    BUY = 1
    SELL = 2

class MarketMakerBot(utc_bot_pb2):
    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.fading_coefficient = 0.02    # Coefficient for fading based on position (GUI Parameter)
        self.stock_symbols = STOCK_SYMBOLS  
        self.etf_conversion_fee = 5
        self.orders_placed = {} 
        self.apt_news_price = None
        self.dlr_news_price = 5000
        self.apt_market_price = None
        self.akav_market_price = None
        self.dlr_market_price = None
        self.mkj_market_price = None
        self.akim_market_price = None
        self.apt_trade_value = None
        self.akav_trade_value = None
        self.current_news_num = 0
        self.earnings = []
        self.current_news_event_num = 0
        self.last_transactions = {"APT": [], "DLR": [], "MKJ": [], "AKAV": [], 'AKIM': []}
        self.trade_task = None


    async def bot_handle_book_update(self, symbol: str) -> None:
        """
        Anytime the client gets a message, it will update the books with new bid or ask
        # TODO: Fill in subclassed bot.
        :return:
        """
        pass

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        """
        adds last transactions to dictionary to use for calculating weighted fair value
        """
        self.last_transactions[symbol].append((price, qty))
        if len(self.last_transactions[symbol]) > 5:
            self.last_transactions[symbol].pop()
            
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        """
        Handle_order_fill will get all the info for that order_id and update the xchange client.
        NEED TO CREATE OPEN_ORDER DATA CLASS?
        # TODO: Fill in subclassed bot.
        :param order_id: Order id corresponding to fill
        :param qty: Amount filled
        :param price: Price filled at
        :return:
        """
        pass
    
    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        """
        handle_order_rejected calls this function then pops the order_id from open_orders
        # TODO: Fill in subclassed bot.
        :param order_id: order id corresponding to the one in open_orders
        :param reason: reason for rejection from the exchange
        """
        # print("order rejected because of ", reason)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        """
        Function for the user to fill in if they want to have any additional actions upon receiving
        a CancelOrderResponse.
        WHAT IS A CANCEL ORDER RESPONSE
        # TODO: Fill in subclassed bot
        :param order_id: Order ID requested to cancel
        :param success: Bool representing if the order was cancelled
        :param error:   Error in cancelling the order (if applicable)
        :return:
        """
        # if success:
        #     order_price, symb, side, quantity = self.orders_placed[order_id]
        #     print(f"Limit Order ID {order_id} cancelled, {quantity} unfilled")

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        """
        handle_swap records swap of stocks to ETF and vice-versa for the exchange
        # TODO: Fill in subclassed bot
        :param swap:    Name of the swap
        :param qty:     How many to Swap
        :param success: Swap executed succesfully
        :return:
        """
        pass

    async def bot_handle_news(self, news_release: dict):
        """
        used to handle news events for DLR and APT; 
        MKJ also has news events but they are unstructured and mostly gibberish
        """
        
        print(f"News: {news_release}")
        news_type = news_release['kind']
        news_package = news_release['new_data']
        if news_type == 'structured':

            # calculate apt fair value using constant P/E ration and earnings, also store the time of news event
            if news_package['structured_subtype'] == 'earnings':
                if news_package['asset'] == 'APT':
                    eps = news_package['value']
                    self.apt_news_price = 10 * eps
                    self.apt_news_timestamp = time.time()

            # calculate dlr news value using log-normal growth model to calculate probability of reaching 100,000 signatures
            elif news_package['structured_subtype'] == 'petition':
                alpha = 1.0630449594499
                sigma = 0.006
                log_alpha = np.log(alpha)
                threshold = 100000
                scaling_factor = 100
                self.earnings.append(news_package['new_signatures'])
                # print(self.earnings)

                cumulative_signatures = news_package['cumulative']
                self.current_news_event_num += 1
                total_news_events = 50
                remaining_news_events = total_news_events - self.current_news_event_num

                mean_final = np.log(cumulative_signatures) + remaining_news_events * log_alpha
                std_final = np.sqrt(remaining_news_events) * sigma

                p_success = 1 - norm.cdf(np.log(threshold), loc=mean_final, scale=std_final)
                news_value = scaling_factor * 100 * p_success  # scaling_factor * 100 gives 10000 when p_success = 1

                self.dlr_news_price = news_value
                print("Probability of success (reaching 100,000 signatures): {:.4f}".format(p_success))


    def get_best_bid_ask(self, symbol: str) -> Optional[tuple]:
        """
        returns best and best asks from the order books (filters through the order book to remove any outlier bid and asks from hitter bots)
        """
        book = self.order_books[symbol]
        if book.bids and book.asks:
            # Filter bids and asks to include only those with non-zero volume
            valid_bids = {price: vol for price, vol in book.bids.items() if vol > 0}
            valid_asks = {price: vol for price, vol in book.asks.items() if vol > 0}
            
            # Ensure we have valid entries on both sides
            if valid_bids and valid_asks:
                if len(self.last_transactions[symbol]) > 1:
                    total_price = 0
                    reference_price = 0
                    for price, quantity in self.last_transactions[symbol]:
                        total_price += price
                    reference_price = total_price / len(self.last_transactions[symbol])
                    if symbol == "AKAV":
                        acceptable_bid_range = {price: vol for price, vol in valid_bids.items() if price <= 1.50 * reference_price}
                        acceptable_ask_range = {price: vol for price, vol in valid_asks.items() if price >= 0.50 * reference_price}
                    else:
                        acceptable_bid_range = {price: vol for price, vol in valid_bids.items() if price <= 1.50 * reference_price}
                        acceptable_ask_range = {price: vol for price, vol in valid_asks.items() if price >= 0.50 * reference_price}

                    # get best bid below best ask
                    if len(acceptable_bid_range) == 0 or len(acceptable_ask_range) == 0:
                        return None
                    best_bid = max(acceptable_bid_range.keys())   # Highest bid with non-zero volume
                    best_ask = min(acceptable_ask_range.keys())   # Lowest ask with non-zero volume
        
                    if best_bid >= best_ask:
                        # Narrow down the acceptable bid range to prices <= best_ask.
                        filtered_bids = {price: vol for price, vol in acceptable_bid_range.items() if price < best_ask}
                        if not filtered_bids:
                            return None
                        best_bid = max(filtered_bids.keys())

                    return best_bid, best_ask

    
    def compute_weighted_avg(self, symbol:str) -> int:
        """
        sets market price for each stock/etf with (best bid + best ask) / 2
        """
        best = self.get_best_bid_ask(symbol)
        if not best:
            return None
        best_bid, best_ask = best
        if not best_bid or not best_ask:
            return None
        measure = (best_bid + best_ask) / 2
        if symbol == "APT":
            self.apt_market_price = measure
        if symbol == 'AKAV':
            print(f"spread: {best_bid}, {best_ask}")
            self.akav_market_price = measure
        if symbol == 'DLR':
            self.dlr_market_price = measure
        if symbol == 'MKJ':
            self.mkj_market_price = measure
        if symbol == 'AKIM':
            self.akim_market_price = measure
        return (measure, best_bid, best_ask)
    
    def compute_fair_value_with_news(self, symbol: str) -> Optional[int]:
        """
        returns the best value to create bid/ask spread around based on market value and news value 
        """
        # for APT, uses a log decay function news value (as earnings gets old, we don't want to use it to compute fair val)
        if symbol == 'APT' and self.apt_news_price is not None:
            weights = self.compute_weighted_avg(symbol)
            if weights is None:
                return None
            weighted_measure = weights[0]
            if weighted_measure is None:
                return None
            news_value = self.apt_news_price
            if hasattr(self, 'apt_news_timestamp'):
                time_elapsed = time.time() - self.apt_news_timestamp - 0.5
            else:
                time_elapsed = 0
            k = math.log(3) / 3
            news_weight = 0.9 * np.exp(-k * time_elapsed) + 0.1
            news_weight = min(news_weight, 1)
            if news_weight <= 0.15:
                news_weight = 0
            alpha = 1 - news_weight
            alpha = min(alpha, 1.0)

            final_value = alpha * weighted_measure + (1 - alpha) * news_value
            # print(f"Blended value: {final_value} with alpha (bayesian weight): {alpha:.2f} and news weight: {1 - alpha:.2f}")
            self.apt_trade_value = final_value
            return int(round(final_value))
        
        # DLR fair value fully based on news price
        elif symbol == 'DLR' and self.dlr_news_price is not None:
            self.compute_weighted_avg(symbol)
            return int(round(self.dlr_news_price))
        
        # MKJ fair value fully based on market price (wasn't very useful)
        elif symbol == 'MKJ':
            weights = self.compute_weighted_avg(symbol)
            if self.mkj_market_price is not None:
                print (self.mkj_market_price)
                return int(round(self.mkj_market_price))
            else:
                return None
        
        # sets AKAV etf value as the sum of all three stock values, weight slightly more in favor of DLR
        elif symbol == "AKAV":
            print("ON AKAV")
            weights = self.compute_weighted_avg(symbol)
            if self.apt_trade_value and self.dlr_news_price and self.mkj_market_price:
                stock_val = self.apt_trade_value + self.dlr_news_price + self.mkj_market_price
                self.akav_trade_value = stock_val + 0.05 * self.positions['DLR']
                return int(round(self.akav_trade_value))
            else:
                print("shoot missing akav market price")
                return None
        
        # didn't end up trading
        elif symbol == "AKIM":
            weights = self.compute_weighted_avg(symbol)
            return int(round(self.akim_market_price)) 
        
        return None
    
    def apply_fading(self, fair_value: int, symbol: str) -> int:
        """
        Applies a fading adjustment to the fair value based on the current position.
        Shifts the fair value away from extreme positions for risk management
        """
        # Assumes that self.positions is maintained by the base class.
        position = self.positions[symbol]
        fading_adjustment = int(-self.fading_coefficient * position)
        return fair_value + fading_adjustment
    
    def adjust_edge_based_on_competitors(self, symbol: str, fair_value: int) -> tuple:
        """
        Determines the optimal bid/ask edge based on competitor orders and the slack parameter.
        If a competitor is found, we "penny" their price by 1.
        Returns a tuple (bid_edge, ask_edge).
        """
        min_edge = 1 #(GUI Parameter)
        max_edge = 200 #(GUI Parameter)
        
        book = self.order_books[symbol]
        competitor_bid = max(book.bids.keys()) if book.bids else None
        competitor_ask = min(book.asks.keys()) if book.asks else None
        
        # Bid side: if competitor exists, penny it by adding $0.01.
        if competitor_bid is not None:
            target_bid = competitor_bid + 1
            bid_edge = fair_value - target_bid
        else:
            bid_edge = max_edge
        
        # Ask side: if competitor exists, penny it by subtracting $0.01.
        if competitor_ask is not None:
            target_ask = competitor_ask - 1
            ask_edge = target_ask - fair_value
        else:
            ask_edge = max_edge
        
        # Clamp edges within our defined range.
        bid_edge = max(min(bid_edge, max_edge), min_edge)
        ask_edge = max(min(ask_edge, max_edge), min_edge)
        
        return bid_edge, ask_edge
    
    def integrate_signals(self, symbol: str) -> tuple:
        """
        returns bid and ask prices using fair value and bid/ask edges
        """
        fair_value = self.compute_fair_value_with_news(symbol)
        if fair_value is None:
            return None
        # didn't use fading in competition
        # faded_value = self.apply_fading(fair_value, symbol)
        bid_edge, ask_edge = self.adjust_edge_based_on_competitors(symbol, fair_value)
        bid_price = fair_value - bid_edge
        ask_price = fair_value + ask_edge

        if bid_price <= 0:
            logging.warning("Computed bid_price <= 0 for symbol %s. Skipping order placement.", symbol)
            return None
        
        return bid_price, ask_price
    
    async def get_outstanding_volume(self, symbol: str) -> int:
        """
        Return the total unfilled volume for orders on the given symbol.
        """
        volume = 0
        for order_id, order_info in self.open_orders.items():
            if order_info[0].symbol == symbol:
                volume += order_info[1]
        return volume
    

    async def cancel_least_competitive_order(self, symbol: str) -> None:
        """
        Cancels the worst order for the given symbol b/c limit for open order count
        For sell orders: order with the highest placed price;
        For buy orders: the one with the lowest placed price.
        """
        # Build lists of orders for the symbol
        ask_orders = []
        bid_orders = []
        for order_id, order_info in self.open_orders.items():
            if order_info[0].symbol == symbol:
                placed_price = self.orders_placed[order_id][0]
                if order_info[0].side == 2:
                    ask_orders.append((order_id, placed_price))
                elif order_info[0].side == 1:
                    bid_orders.append((order_id, placed_price))
       
        # Cancel the worst order(s)
        if ask_orders:
            worst_ask_id, worst_ask_price = max(ask_orders, key=lambda x: x[1])
            try:
                await self.cancel_order(worst_ask_id)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.INTERNAL:
                    logging.info("Internal error during cancellation for order %s: %s", worst_ask_id, e.details())
                else:
                    logging.exception("Error cancelling ask order %s: %s", worst_ask_id, e)
            except asyncio.InvalidStateError as e:
                if "RPC already finished" in str(e):
                    logging.info("RPC already finished for ask order %s", worst_ask_id)
                else:
                    logging.exception("Error cancelling ask order %s: %s", worst_ask_id, e)
            except Exception as e:
                logging.exception("Error cancelling ask order %s: %s", worst_ask_id, e)

        if bid_orders:
            worst_bid_id, worst_bid_price = min(bid_orders, key=lambda x: x[1])
            try:
                await self.cancel_order(worst_bid_id)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.INTERNAL:
                    logging.info("Internal error during cancellation for order %s: %s", worst_bid_id, e.details())
                else:
                    logging.exception("Error cancelling bid order %s: %s", worst_bid_id, e)
            except asyncio.InvalidStateError as e:
                if "RPC already finished" in str(e):
                    logging.info("RPC already finished for bid order %s", worst_bid_id)
                else:
                    logging.exception("Error cancelling bid order %s: %s", worst_bid_id, e)
            except Exception as e:
                logging.exception("Error cancelling bid order %s: %s", worst_bid_id, e)

    async def cancel_bad_orders(self, symbol, bid_price, ask_price) -> None:
        """
        cancels any bids greater than current bid price and any asks less than current ask price
        """
        for order_id, order_info in list(self.open_orders.items()):
            if order_info[0].symbol == symbol:
                if order_info[0].side == 1 and self.orders_placed[order_id][0] > bid_price:
                    try:
                        await self.cancel_order(order_id)
                    except Exception as e:
                        logging.exception("Error cancelling bid order %s: %s", order_id, e)
                if order_info[0].side == 2 and self.orders_placed[order_id][0] < ask_price:
                    try:
                        await self.cancel_order(order_id)
                    except Exception as e:
                        logging.exception("Error cancelling bid order %s: %s", order_id, e)

    async def placing_orders(self, symbol, bid_price, ask_price) -> None:
        """Places orders for each stock (max order size of 40)"""
        if symbol == "DLR":
            # position taken is based on probability DLR hits 100,000
            if self.dlr_news_price is not None:
                if self.dlr_news_price > 5000:
                    # Calculate the target (positive) position based on the news price.
                    max_bid_position = int(((self.dlr_news_price - 5000) / 10000) * 400)
                    # Compute a base quantity and cap it at 40 per trade.
                    base_trade_qty = int(((self.dlr_news_price - 5000) / 10000) * 400)
                    trade_qty = min(base_trade_qty, 40)
                    tolerance = 5  # Acceptable deviation from the target position.
                    target_position = max_bid_position

                    if abs(self.positions[symbol] - target_position) > tolerance:
                        current_position = self.positions[symbol]
                        # If the current position is lower than the target, buy more.
                        if current_position < target_position:
                            needed = target_position - current_position
                            order_qty = min(trade_qty, needed, 40)
                            # logging.info("Current position %d < target %d: placing BUY order for %d", 
                            #             current_position, target_position, order_qty)
                            b_id = await self.place_order(symbol, order_qty, side="buy", px=int(bid_price))
                            self.orders_placed[b_id] = (bid_price, symbol, Side.BUY.value, order_qty)
                        # If the current position is above the target (overshoot), sell to reduce the position.
                        elif current_position > target_position:
                            excess = current_position - target_position
                            order_qty = min(excess, 40)
                            # logging.info("Current position %d > target %d: placing SELL order for %d", 
                            #             current_position, target_position, order_qty)
                            a_id = await self.place_order(symbol, order_qty, side="sell", px=int(ask_price))
                            self.orders_placed[a_id] = (ask_price, symbol, Side.SELL.value, order_qty)

                elif self.dlr_news_price < 5000:
                    max_ask_position = int(((5000 - self.dlr_news_price) / 10000) * 400)
                    target_position = -max_ask_position  # The desired negative position.
                    
                    # Determine a base trade quantity from the news price; then cap it at 40.
                    base_trade_qty = int(((5000 - self.dlr_news_price) / 10000) * 400)
                    trade_qty = min(base_trade_qty, 40)
                    tolerance = 5  # Acceptable deviation from the target position.
                    
                    # Adjust the position until it is within the tolerance of the target.
                    if abs(self.positions[symbol] - target_position) > tolerance:
                        current_position = self.positions[symbol]
                        if current_position > target_position:
                            available_qty = current_position - target_position
                            order_qty = min(trade_qty, available_qty, 40)
                            logging.info("Current position %d > target %d: placing SELL order for %d", 
                                        current_position, target_position, order_qty)
                            a_id = await self.place_order(symbol, order_qty, side="sell", px=int(ask_price))
                            self.orders_placed[a_id] = (ask_price, symbol, Side.SELL.value, order_qty)
                        
                        elif current_position < target_position:
                            # Position is too low (more negative than desired): place a BUY order to correct the overshoot.
                            available_qty = target_position - current_position
                            order_qty = min(trade_qty, available_qty, 40)
                            logging.info("Current position %d < target %d: placing BUY order for %d", 
                                        current_position, target_position, order_qty)
                            b_id = await self.place_order(symbol, order_qty, side="buy", px=int(bid_price))
                            self.orders_placed[b_id] = (bid_price, symbol, Side.BUY.value, order_qty)
                        

        if symbol == "APT":
            # if market price is within 5 of fair value then don't trade
            if hasattr(self, 'apt_news_timestamp'):
                time_elapsed = time.time() - self.apt_news_timestamp - 0.5
                if self.apt_news_price is not None and self.apt_market_price is not None and self.apt_trade_value is not None:
                    if self.apt_market_price > self.apt_trade_value + 5:
                        ask_trade_quantity = 40
                        # print(f"ask: {ask_trade_quantity}")
                        a_id = await self.place_order(symbol, ask_trade_quantity, side = "sell", px=int(ask_price))
                        self.orders_placed[a_id] = (bid_price, symbol, 2, ask_trade_quantity)
                        self.compute_fair_value_with_news(symbol)

                    if self.apt_market_price + 5 < self.apt_trade_value:
                        bid_trade_quantity = 40
                        # print(f"bid: {bid_trade_quantity}")
                        b_id = await self.place_order(symbol, bid_trade_quantity, side = "buy", px=int(bid_price))
                        self.orders_placed[b_id] = (bid_price, symbol, 1, bid_trade_quantity)
                        self.compute_fair_value_with_news(symbol)

                    else:
                        if self.positions[symbol] > 0 and time_elapsed > 5:
                            # print ("test ask")
                            ask_trade_quantity = min(abs(self.positions[symbol]), 40)
                            a_id = await self.place_order(symbol, ask_trade_quantity, side = "sell", px=int(ask_price))
                            self.orders_placed[a_id] = (bid_price, symbol, 2, ask_trade_quantity)
                        if self.positions[symbol] < 0 and time_elapsed > 5:
                            bid_trade_quantity = min(abs(self.positions[symbol]), 40)
                            # print("test bid")
                            b_id = await self.place_order(symbol, bid_trade_quantity, side = "buy", px=int(bid_price))
                            self.orders_placed[b_id] = (bid_price, symbol, 1, bid_trade_quantity)
        
        # decided not to trade MKJ because news was gibberish
        # if symbol == 'MKJ':
        #     b_id = await self.place_order(symbol, qty = 10, side='buy', px=int(bid_price))
        #     a_id = await self.place_order(symbol, qty=10, side='sell', px=int(ask_price))
        #     print (bid_price, ask_price)
        #     self.orders_placed[b_id] = (bid_price, symbol, 1, 10)
        #     self.orders_placed[a_id] = (ask_price, symbol, 2, 10)

        if symbol == "AKAV":
            await self.akav_arbitrage(bid_price, ask_price)

        # decided not to trade AKIM because prices reset every 'day' --> hard to judge fair value
        # if symbol == 'AKIM':
        #     await self.akim_arbitrage(bid_price, ask_price)


    async def akav_arbitrage(self, bid_price, ask_price):
        """
        finds and trades on arbitrage opportunities if underlying stocks or etf are mispriced 
        """
        if self.akav_market_price is None:
            return None
        if self.akav_trade_value is None:
            return None
        if self.akav_trade_value > 11000 or self.akav_trade_value < 3000:
            print('akav trade val too high or low')
            return
        print(self.akav_market_price, self.akav_trade_value)
        print(self.apt_market_price, self.dlr_news_price, self.mkj_market_price )

        if self.akav_market_price + self.etf_conversion_fee + 5 < self.akav_trade_value:
            min_pos = min((200 + self.positions['DLR']), (200 + self.positions['APT']), (200 + self.positions['MKJ']))
            qty_buy = min((self.akav_trade_value - self.akav_market_price), 40, abs(min_pos))
            await self.place_swap_order("toAKAV", (int(round(qty_buy))))
            await asyncio.sleep(0.1)

        elif self.akav_market_price > self.akav_trade_value + self.etf_conversion_fee + 5:
            min_pos = min((200 - self.positions['DLR']), (200 - self.positions['APT']), (200 - self.positions['MKJ']))
            qty_sell = min((self.akav_market_price - self.akav_trade_value), 40, abs(min_pos))
            await self.place_swap_order("fromAKAV", (int(round(qty_sell))))
            await asyncio.sleep(0.1)

    async def trade(self):
        """loop for trading all stocks"""
        await asyncio.sleep(2)
        while True:

            for symbol in self.stock_symbols:
                try:
                    pkg = self.integrate_signals(symbol)
                    if pkg is not None:
                        bid_price, ask_price = pkg
                        #logging.info("Trading signal for %s: bid=%s, ask=%s", symbol, bid_price, ask_price)
                        # Quantity is a (GUI Parameter for now)
                        # Check outstanding volume for the symbol.
                        current_volume = await self.get_outstanding_volume(symbol)
                        # If outstanding volume (or count) exceeds our limits, cancel orders until we're below threshold.
                        if current_volume + 60 > MAX_OUTSTANDING_VOLUME or len({
                            oid for oid, info in self.open_orders.items() if info[0].symbol == symbol
                        }) >= MAX_OPEN_ORDERS - 20:
                            print(f"current volume {current_volume}")
                            await self.cancel_least_competitive_order(symbol)
                            current_volume = await self.get_outstanding_volume(symbol)

                        await self.cancel_bad_orders(symbol, bid_price, ask_price)
                        await self.placing_orders(symbol, bid_price, ask_price)

                except Exception as e:
                    logging.exception("Error placing orders for %s: %s", symbol, e) 
            await asyncio.sleep(0.2)

    async def start(self, user_interface):
        self.trade_task = asyncio.create_task(self.trade())

        # This is where Phoenixhood will be launched if desired. There is no need to change these
        # lines, you can either remove the if or delete the whole thing depending on your purposes.
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())

        await self.connect()    


async def main(user_interface):
    """connecting to the exchange"""
    SERVER = "3.138.154.148:3333"    # e.g. "localhost:50051"
    USERNAME = "chicago1"
    PASSWORD = "ck3)PoYyf6"

    while True:
        bot = None
        try:
            # Instantiate your bot.
            bot = MarketMakerBot(SERVER, USERNAME, PASSWORD)
            # Start the bot (which connects and begins processing)
            await bot.start(user_interface)
            # Block indefinitely to keep the event loop alive.
            await asyncio.Event().wait()
        except grpc.aio.AioRpcError as e:
            logging.error("gRPC error occurred: %s", e)
            # Optionally, add any reconnection cleanup here.
            await asyncio.sleep(1)
        except Exception as e:
            logging.exception("Unexpected exception occurred: %s. Restarting bot...", e)
            await asyncio.sleep(1)
        finally:
            # Clean up: If a trade task exists, cancel it.
            if bot is not None and hasattr(bot, "trade_task") and bot.trade_task is not None:
                bot.trade_task.cancel()
                try:
                    await bot.trade_task
                except asyncio.CancelledError:
                    logging.info("Trade task cancelled successfully.")
        # Wait briefly before restarting the bot.
        await asyncio.sleep(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that connects client to exchange, runs algorithmic trading logic, and optionally deploys Pheonixhood"
    )

    parser.add_argument("--pheonixhood", required=False, type=bool, help="Starts pheonixhood APT if true")
    args = parser.parse_args()
    user_interface = args.pheonixhood

    # 4. Run the async function in an event loop
    asyncio.run(main(user_interface))