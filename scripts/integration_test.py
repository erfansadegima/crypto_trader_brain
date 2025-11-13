import json
from n8n_integration import N8NIntegration

if __name__ == '__main__':
    integration = N8NIntegration()

    payload1 = {
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'use_mtf': False,
        'execute': False
    }

    print('\n=== TEST 1: order_instruction (no execution) ===')
    resp1 = integration.process_webhook(payload1)
    print(json.dumps(resp1, indent=2, ensure_ascii=False))

    payload2 = {
        'symbol': 'ETH/USDT',
        'timeframe': '15m',
        'use_mtf': False,
        'execute': True,
        'exchange_credentials': {}
    }

    print('\n=== TEST 2: execute True but no creds (should return order_instruction) ===')
    resp2 = integration.process_webhook(payload2)
    print(json.dumps(resp2, indent=2, ensure_ascii=False))
