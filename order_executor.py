"""
Order Executor

Provides helper functions to format orders (entry/TP/SL) and place them on Binance Spot.
This module intentionally focuses on generating order payloads and executing signed REST requests
for Binance. It is written to be safe: by default it only returns the payload. To actually place an
order you must provide API credentials and opt-in to execution.

Notes:
- `TradeDecision.position_size` is used as the `quantity` for Binance orders. Make sure position_size
  is in base asset units (e.g. BTC for BTCUSDT). If your position_size is in quote currency, convert
  it to base units before calling place_order.
- This module supports `test=True` which calls Binance's `order/test` endpoint (no trade executed).

"""
from __future__ import annotations

import time
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional
import urllib.parse

BINANCE_API_BASE = "https://api.binance.com"
BINANCE_TESTNET_BASE = "https://testnet.binance.vision"


def _sign_params(params: Dict[str, Any], api_secret: str) -> str:
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature


def format_order_payload(decision: Any, order_type: str = 'MARKET') -> Dict[str, Any]:
    """
    Create a simple Binance order payload from a TradeDecision object.

    Args:
        decision: TradeDecision (from tryd_fixed) or similar object/dict with fields:
                  - symbol (e.g. 'BTC/USDT')
                  - action (TradingAction enum or 'BUY'/'SELL')
                  - position_size (float): quantity in base asset units
                  - entry_price: suggested entry price (float)
                  - take_profit, stop_loss (optional floats)

        order_type: 'MARKET' or 'LIMIT' (default MARKET)

    Returns:
        dict containing payload ready to be signed and sent to Binance
    """
    symbol = decision.symbol.replace('/', '') if hasattr(decision, 'symbol') else decision['symbol'].replace('/', '')
    side = decision.action if isinstance(decision.action, str) else decision.action.value
    quantity = float(decision.position_size)

    payload = {
        'symbol': symbol,
        'side': side,
        'type': order_type,
        'quantity': round(quantity, 8),
        'timestamp': int(time.time() * 1000)
    }

    if order_type == 'LIMIT':
        # Use entry_price when provided
        entry_price = getattr(decision, 'entry_price', None) or decision.get('entry_price') if isinstance(decision, dict) else None
        if entry_price:
            payload['price'] = str(round(float(entry_price), 8))
            payload['timeInForce'] = 'GTC'

    # Attach SL/TP in the payload for reference (these are not native fields for a single Binance order)
    payload['meta_take_profit'] = getattr(decision, 'take_profit', None) or (decision.get('take_profit') if isinstance(decision, dict) else None)
    payload['meta_stop_loss'] = getattr(decision, 'stop_loss', None) or (decision.get('stop_loss') if isinstance(decision, dict) else None)

    return payload


def place_order_binance(api_key: str, api_secret: str, payload: Dict[str, Any], test: bool = True, use_testnet: bool = False) -> Dict[str, Any]:
    """
    Place an order on Binance (spot). By default `test=True` so no actual order is executed.

    Args:
        api_key, api_secret: Binance credentials
        payload: dict from format_order_payload
        test: if True use /api/v3/order/test
        use_testnet: if True use Binance testnet base url

    Returns:
        dict with Binance response (or raises requests.HTTPError)
    """
    base = BINANCE_TESTNET_BASE if use_testnet else BINANCE_API_BASE
    endpoint = '/api/v3/order/test' if test else '/api/v3/order'

    params = {k: v for k, v in payload.items() if k not in ['meta_take_profit', 'meta_stop_loss']}
    params['timestamp'] = int(time.time() * 1000)

    signature = _sign_params(params, api_secret)
    params['signature'] = signature

    headers = {
        'X-MBX-APIKEY': api_key,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    url = base + endpoint
    resp = requests.post(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()

    return resp.json()


def build_order_and_optional_oco(decision: Any) -> Dict[str, Any]:
    """
    Build a compact order instruction dict that includes an order payload and TP/SL details.
    This is intended to be returned to n8n; n8n can then call `place_order_binance` with credentials.
    """
    payload = format_order_payload(decision)

    return {
        'order_payload': payload,
        'take_profit': payload.get('meta_take_profit'),
        'stop_loss': payload.get('meta_stop_loss'),
        'note': 'Provide API credentials to n8n node to execute. By default use test=True to avoid real trades.'
    }
