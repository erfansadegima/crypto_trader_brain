"""
═══════════════════════════════════════════════════════════════════════════
N8N API WRAPPER & INTEGRATION MODULE
═══════════════════════════════════════════════════════════════════════════
برای ارتباط آسان‌تر با n8n و سایر Webhooks
"""

import json
import logging
from typing import Dict, Optional, Any, List
from dataclasses import asdict
from datetime import datetime
from enum import Enum
import traceback

from tryd_fixed import (
    CryptoTradingBrain,
    TradeDecision,
    TradingAction,
    RiskParameters
)
from order_executor import build_order_and_optional_oco, place_order_binance


class WebhookResponseStatus(Enum):
    """وضعیت پاسخ Webhook"""
    SUCCESS = "success"
    HOLD = "hold"
    ERROR = "error"
    INVALID_INPUT = "invalid_input"


class N8NIntegration:
    """
    کلاس انتگریشن n8n
    مدیریت Webhooks و API calls از n8n
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("N8NIntegration")
        self.brain = CryptoTradingBrain()
    
    def validate_webhook_input(self, data: Dict) -> tuple[bool, str]:
        """
        اعتبارسنجی ورودی Webhook
        
        Args:
            data: دیکشنری ورودی از n8n
        
        Returns:
            (is_valid, error_message)
        """
        
        # Required fields
        required_fields = ['symbol']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate symbol format
        symbol = data.get('symbol', '')
        if '/' not in symbol:
            return False, f"Invalid symbol format. Expected 'BTC/USDT', got '{symbol}'"
        
        # Validate timeframe if provided
        if 'timeframe' in data:
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            if data['timeframe'] not in valid_timeframes:
                return False, f"Invalid timeframe: {data['timeframe']}"
        
        # Validate numeric fields
        numeric_fields = ['equity', 'risk_per_trade', 'fear_greed_index']
        for field in numeric_fields:
            if field in data:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    return False, f"Invalid {field}: must be numeric"
        
        return True, "Valid input"
    
    def process_webhook(self, webhook_data: Dict) -> Dict:
        """
        پردازش درخواست Webhook از n8n
        
        Args:
            webhook_data: داده‌های ورودی
        
        Returns:
            دیکشنری خروجی
        """
        
        try:
            self.logger.info(f"📨 Processing webhook: {webhook_data.get('symbol', 'UNKNOWN')}")
            
            # Validate input
            is_valid, error_msg = self.validate_webhook_input(webhook_data)
            if not is_valid:
                return {
                    "status": WebhookResponseStatus.INVALID_INPUT.value,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract parameters
            symbol = webhook_data.get('symbol')
            timeframe = webhook_data.get('timeframe', '1h')
            use_mtf = webhook_data.get('use_mtf', True)
            fear_greed = webhook_data.get('fear_greed_index')
            
            # Update risk parameters if provided
            if 'equity' in webhook_data:
                self.brain.risk_manager.params.equity = float(webhook_data['equity'])
            
            if 'risk_per_trade' in webhook_data:
                self.brain.risk_manager.params.risk_per_trade = float(webhook_data['risk_per_trade'])
            
            if 'max_leverage' in webhook_data:
                self.brain.risk_manager.params.max_leverage = float(webhook_data['max_leverage'])
            
            # Analyze and decide
            decision = self.brain.analyze_and_decide(
                symbol=symbol,
                timeframe=timeframe,
                use_mtf=use_mtf,
                fear_greed=fear_greed
            )

            # If user requested execution and provided credentials, attempt to place order
            execute = bool(webhook_data.get('execute', False))
            exchange_creds = webhook_data.get('exchange_credentials', {})

            # Format response and optionally execute
            if decision:
                # Convert decision to order instruction
                order_instruction = build_order_and_optional_oco(decision)

                if execute and exchange_creds:
                    # Currently supports Binance spot only via exchange='binance'
                    if exchange_creds.get('exchange', '').lower() == 'binance':
                        try:
                            api_key = exchange_creds.get('api_key')
                            api_secret = exchange_creds.get('api_secret')
                            use_testnet = bool(exchange_creds.get('testnet', True))
                            # By default keep test=True to avoid real orders unless explicitly disabled
                            test_flag = not bool(exchange_creds.get('execute_live', False))

                            resp = place_order_binance(
                                api_key=api_key,
                                api_secret=api_secret,
                                payload=order_instruction['order_payload'],
                                test=test_flag,
                                use_testnet=use_testnet
                            )

                            response = self._format_decision_response(decision)
                            response['order_execution'] = resp
                            response['status'] = WebhookResponseStatus.SUCCESS.value

                        except Exception as e:
                            response = self._format_decision_response(decision)
                            response['status'] = WebhookResponseStatus.ERROR.value
                            response['order_error'] = str(e)
                            response['traceback'] = traceback.format_exc()

                    else:
                        # Unsupported exchange for direct execution yet — return order instruction
                        response = self._format_decision_response(decision)
                        response['order_instruction'] = order_instruction
                        response['status'] = WebhookResponseStatus.SUCCESS.value
                else:
                    # No execution requested — return order instruction for n8n to handle
                    response = self._format_decision_response(decision)
                    response['order_instruction'] = order_instruction
                    response['status'] = WebhookResponseStatus.SUCCESS.value

            else:
                response = {
                    'status': WebhookResponseStatus.HOLD.value,
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No strong signal detected',
                    'timestamp': datetime.now().isoformat()
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error processing webhook: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            return {
                'status': WebhookResponseStatus.ERROR.value,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_decision_response(self, decision: TradeDecision) -> Dict:
        """فرمت‌کردن پاسخ تصمیم معامله"""
        
        return {
            'symbol': decision.symbol,
            'action': decision.action.value,
            'entry_price': round(decision.entry_price, 8),
            'take_profit': round(decision.take_profit, 8) if decision.take_profit else None,
            'stop_loss': round(decision.stop_loss, 8) if decision.stop_loss else None,
            'position_size': round(decision.position_size, 8),
            'confidence': round(decision.confidence, 4),
            'strategy': decision.strategy,
            'risk_reward_ratio': round(decision.risk_reward_ratio, 2),
            'reasons': decision.reasons,
            'indicators': decision.indicators,
            'timestamp': decision.timestamp.isoformat()
        }
    
    def get_status(self) -> Dict:
        """دریافت وضعیت سیستم"""
        
        return {
            'status': 'operational',
            'version': '2.0.0',
            'modules': {
                'data_ingestor': True,
                'preprocessor': True,
                'feature_engineer': True,
                'pattern_recognizer': True,
                'signal_generator': True,
                'decision_engine': True,
                'risk_manager': True,
                'trade_executor': True
            },
            'strategies': [s.name for s in self.brain.signal_generator.strategies],
            'timestamp': datetime.now().isoformat()
        }


class WebhookServer:
    """
    سرور Webhook برای n8n
    می‌توان این را در یک Flask/FastAPI استفاده کرد
    """
    
    def __init__(self, port: int = 5000, logger: logging.Logger = None):
        self.port = port
        self.logger = logger or logging.getLogger("WebhookServer")
        self.integration = N8NIntegration(logger)
    
    def handle_webhook(self, payload: Dict) -> Dict:
        """
        هاندلر Webhook
        
        برای استفاده در Flask:
        ```python
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        server = WebhookServer()
        
        @app.route('/webhook/trade', methods=['POST'])
        def webhook():
            response = server.handle_webhook(request.json)
            return jsonify(response)
        
        if __name__ == '__main__':
            app.run(port=5000)
        ```
        """
        
        self.logger.info(f"Received webhook payload: {payload}")
        
        response = self.integration.process_webhook(payload)
        
        self.logger.info(f"Webhook response: {response}")
        
        return response
    
    def handle_status_request(self) -> Dict:
        """هاندلر درخواست وضعیت"""
        return self.integration.get_status()


class BatchAnalyzer:
    """
    تحلیل دسته‌ای برای چندین نماد در یک درخواست
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("BatchAnalyzer")
        self.integration = N8NIntegration(logger)
    
    def analyze_multiple_symbols(self, symbols: List[str], 
                                 timeframe: str = '1h',
                                 use_mtf: bool = False) -> Dict:
        """
        تحلیل چندین نماد به صورت دسته‌ای
        
        Args:
            symbols: لیست نمادها (مثل ['BTC/USDT', 'ETH/USDT'])
            timeframe: تایم‌فریم مشترک
            use_mtf: استفاده از تحلیل چند تایم‌فریمی
        
        Returns:
            دیکشنری حاوی تصمیمات برای هر نماد
        """
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(symbols),
            'decisions': []
        }
        
        for symbol in symbols:
            try:
                webhook_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'use_mtf': use_mtf
                }
                
                response = self.integration.process_webhook(webhook_data)
                results['decisions'].append(response)
                
                self.logger.info(f"✅ Analyzed {symbol}")
                
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {symbol}: {str(e)}")
                results['decisions'].append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_portfolio_summary(self, symbols: List[str]) -> Dict:
        """
        خلاصه پورتفولیو برای چندین نماد
        """
        
        self.logger.info(f"Analyzing portfolio of {len(symbols)} symbols...")
        
        batch_results = self.analyze_multiple_symbols(symbols, use_mtf=True)
        
        # Calculate summary metrics
        successful = [d for d in batch_results['decisions'] 
                     if d.get('status') == 'success']
        hold = [d for d in batch_results['decisions'] 
               if d.get('status') == 'hold']
        errors = [d for d in batch_results['decisions'] 
                 if d.get('status') == 'error']
        
        buy_signals = [d for d in successful if d.get('action') == 'BUY']
        sell_signals = [d for d in successful if d.get('action') == 'SELL']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_size': len(symbols),
            'analysis_results': {
                'successful': len(successful),
                'hold': len(hold),
                'errors': len(errors)
            },
            'signals': {
                'buy': len(buy_signals),
                'sell': len(sell_signals),
                'hold': len(hold)
            },
            'buy_opportunities': [
                {
                    'symbol': d['symbol'],
                    'confidence': d.get('confidence', 0),
                    'entry_price': d.get('entry_price'),
                    'take_profit': d.get('take_profit'),
                    'stop_loss': d.get('stop_loss')
                }
                for d in buy_signals[:5]  # Top 5
            ],
            'sell_opportunities': [
                {
                    'symbol': d['symbol'],
                    'confidence': d.get('confidence', 0),
                    'entry_price': d.get('entry_price'),
                    'take_profit': d.get('take_profit'),
                    'stop_loss': d.get('stop_loss')
                }
                for d in sell_signals[:5]
            ]
        }
        
        return summary


class ConfigManager:
    """
    مدیریت پیکربندی برای n8n Workflows
    """
    
    def __init__(self):
        self.presets = {
            'conservative': {
                'risk_per_trade': 0.005,
                'max_leverage': 1.0,
                'max_position_size': 0.05,
                'rr_ratio': 3.0
            },
            'moderate': {
                'risk_per_trade': 0.01,
                'max_leverage': 2.0,
                'max_position_size': 0.1,
                'rr_ratio': 2.0
            },
            'aggressive': {
                'risk_per_trade': 0.02,
                'max_leverage': 5.0,
                'max_position_size': 0.2,
                'rr_ratio': 1.5
            },
            'scalping': {
                'risk_per_trade': 0.005,
                'max_leverage': 3.0,
                'max_position_size': 0.08,
                'rr_ratio': 1.0
            }
        }
    
    def get_preset(self, preset_name: str) -> Optional[Dict]:
        """دریافت یک preset پیکربندی"""
        return self.presets.get(preset_name.lower())
    
    def list_presets(self) -> List[str]:
        """لیست دسترس‌پذیر Presets"""
        return list(self.presets.keys())
    
    def validate_config(self, config: Dict) -> tuple[bool, str]:
        """اعتبارسنجی پیکربندی"""
        
        if 'risk_per_trade' in config:
            if not (0 < config['risk_per_trade'] < 1):
                return False, "risk_per_trade must be between 0 and 1"
        
        if 'max_leverage' in config:
            if config['max_leverage'] < 1:
                return False, "max_leverage must be >= 1"
        
        if 'max_position_size' in config:
            if not (0 < config['max_position_size'] <= 1):
                return False, "max_position_size must be between 0 and 1"
        
        if 'rr_ratio' in config:
            if config['rr_ratio'] < 0.5:
                return False, "rr_ratio must be >= 0.5"
        
        return True, "Config is valid"


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE FOR N8N
# ═══════════════════════════════════════════════════════════════════════════

"""
### N8N Workflow Example - Code Node

```javascript
// Initialize integration
const integration = require('./n8n_integration.py');
const n8n_handler = new integration.N8NIntegration();

// Get input from previous node
const webhook_data = $json;

// Process
const response = n8n_handler.process_webhook(webhook_data);

// Return to next node
return [{json: response}];
```

### Using with Webhooks

1. Add Webhook trigger in n8n
2. Configure to send JSON payload like:
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "use_mtf": true,
  "equity": 10000,
  "risk_per_trade": 0.01
}
```

3. Add HTTP Request node or Code node to call this module
4. Process response and send to Telegram/Email/Trading Bot

### Batch Analysis Example

```python
from n8n_integration import BatchAnalyzer

analyzer = BatchAnalyzer()
symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
results = analyzer.get_portfolio_summary(symbols)
```
"""


if __name__ == "__main__":
    # Test example
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    integration = N8NIntegration()
    
    # Test webhook
    test_payload = {
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'use_mtf': False,
        'equity': 5000,
        'risk_per_trade': 0.01
    }
    
    print("\n" + "="*70)
    print("Testing N8N Integration")
    print("="*70)
    
    response = integration.process_webhook(test_payload)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    
    print("\n" + "="*70)
    print("System Status")
    print("="*70)
    
    status = integration.get_status()
    print(json.dumps(status, indent=2))
