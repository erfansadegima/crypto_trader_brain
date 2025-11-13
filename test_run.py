from tryd_fixed import CryptoTradingBrain
import json

if __name__ == "__main__":
    # نمونه ورودی تستی
    test_input = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "limit": 200,
        "risk_per_trade": 0.02,
        "leverage": 3,
        "mode": "paper",
        "current_equity": 5000,
        "fear_greed_index": 45
    }

    # ساخت نمونه از مغز تریدر
    brain = CryptoTradingBrain()

    # پردازش ورودی و دریافت خروجی
    output = brain.process_n8n_webhook(test_input)

    # نمایش خروجی نهایی
    print("\n========= TRADE DECISION =========")
    print(json.dumps(output, indent=4))
    print("==================================\n")
