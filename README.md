# 🧠 Crypto Trading Brain v2.0

**Advanced Automated Trading System with n8n Integration**

سیستم تریدینگ اتوماتیک حرفه‌ای برای کریپتو‌ارزها با انتگریشن n8n

---

## 📋 فهرست مطالب | Table of Contents

- [ویژگی‌ها](#-ویژگی‌ها)
- [معماری سیستم](#-معماری-سیستم)
- [نصب و راه‌اندازی](#-نصب-و-راه‌اندازی)
- [استفاده](#-استفاده)
- [انتگریشن n8n](#-انتگریشن-n8n)
- [استراتژی‌های معاملاتی](#-استراتژی‌های-معاملاتی)
- [مدیریت ریسک](#-مدیریت-ریسک)
- [تست‌ها](#-تست‌ها)

---

## 🚀 ویژگی‌ها

### تحلیل تکنیکال پیشرفته
- ✅ **30+ اندیکاتور تکنیکال**: RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, CCI, OBV, MFI و بیشتر
- ✅ **شناسایی الگوهای کندلی**: Engulfing, Hammer, Shooting Star, Doji, Morning Star, Evening Star
- ✅ **شناسایی سطوح حمایت و مقاومت**: تشخیص خودکار سطوح کلیدی
- ✅ **تحلیل چند تایم‌فریمی**: تایید سیگنال‌ها در چندین timeframe

### استراتژی‌های هوشمند
- 🤖 **Strategy_Tech**: ترکیب RSI + MACD + EMA + Bollinger Bands + Volume
- 🤖 **Strategy_Trend**: تشخیص روند بر اساس EMA alignment و ADX
- 🤖 **Strategy_Breakout**: شناسایی شکست‌های حمایت/مقاومت
- 🤖 **Strategy_RiskSentiment**: تحلیل احساسات بازار و Fear & Greed Index

### مدیریت ریسک پیشرفته
- 💰 **محاسبه خودکار حجم پوزیشن**: بر اساس ریسک تعریف شده
- 💰 **حد ضرر و حد سود**: محاسبه اتوماتیک بر اساس ATR
- 💰 **مدیریت Drawdown**: کنترل حداکثر افت سرمایه
- 💰 **مدیریت Leverage**: محدودیت کنترل‌شده اهرم

### صرافی‌های پشتیبانی
- 📊 **Binance**: مکمل‌ترین پشتیبانی
- 📊 **KuCoin**: API integration کامل
- 📊 **Coinbase Pro**: راه‌اندازی آماده
- 📊 **سایر**: داده مصنوعی برای تست

### انتگریشن n8n
- 🔄 **Webhook Integration**: درخواست‌های Webhook آماده
- 🔄 **Real-time Processing**: پردازش سریع و آنی
- 🔄 **Batch Analysis**: تحلیل دسته‌ای برای چندین نماد
- 🔄 **Portfolio Management**: مدیریت پورتفولیو

---

## 🏗️ معماری سیستم

```
┌─────────────────────────────────────────────────────────────┐
│                    n8n Webhook / API                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              N8NIntegration / WebhookServer                 │
│                   (Input Validation)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            CryptoTradingBrain (Main Controller)             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Data Ingestor (Binance, KuCoin, Coinbase)       │  │
│  │    └─> Fetch OHLCV data from API                    │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 2. Preprocessor                                      │  │
│  │    └─> Clean, validate, and normalize data          │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 3. Feature Engineer (30+ Indicators)                │  │
│  │    ├─> Moving Averages (SMA, EMA, WMA)             │  │
│  │    ├─> RSI, MACD, Bollinger Bands                  │  │
│  │    ├─> ATR, Stochastic, ADX, CCI                  │  │
│  │    └─> Volume & Momentum indicators                │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 4. Pattern Recognizer                              │  │
│  │    ├─> Candle patterns (Engulfing, Hammer, etc)   │  │
│  │    └─> Chart patterns (Support/Resistance)         │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 5. Signal Generator (4 Strategies)                 │  │
│  │    ├─> Strategy_Tech (Technical Combo)             │  │
│  │    ├─> Strategy_Trend (Trend Following)            │  │
│  │    ├─> Strategy_Breakout (Pattern Breaking)        │  │
│  │    └─> Strategy_RiskSentiment (Sentiment)          │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 6. Decision Engine (Weighted Voting)               │  │
│  │    └─> Combine signals from all strategies         │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 7. Risk Manager                                     │  │
│  │    ├─> Calculate position size                      │  │
│  │    ├─> Set stop loss & take profit                │  │
│  │    └─> Validate risk constraints                   │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ 8. Trade Executor                                   │  │
│  │    └─> Generate final trade decision (JSON)        │  │
│  └──────────────────────┬───────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│         JSON Trade Decision (to n8n / Trading Bot)          │
│                                                             │
│  {                                                          │
│    "symbol": "BTC/USDT",                                  │
│    "action": "BUY",                                        │
│    "entry_price": 50000.0,                               │
│    "take_profit": 52000.0,                               │
│    "stop_loss": 48000.0,                                 │
│    "position_size": 0.5,                                 │
│    "confidence": 0.85,                                   │
│    "strategy": "TechnicalCombo",                         │
│    "risk_reward_ratio": 2.0                              │
│  }                                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 نصب و راه‌اندازی

### نیازمندی‌ها

- Python 3.8+
- pip / conda

### مراحل نصب

```bash
# 1. Clone یا دانلود پروژه
cd crypto_trader_brain

# 2. نصب وابستگی‌ها
pip install -r requirements.txt

# 3. تست نصب
python -m pytest test_crypto_brain.py -v
```

### تنظیمات محیط

```bash
# ایجاد فایل .env (اختیاری)
cat > .env << EOF
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
LOG_LEVEL=INFO
EOF
```

---

## 💻 استفاده

### استفاده ساده (Python)

```python
from tryd_fixed import CryptoTradingBrain

# ایجاد مغز تریدر
brain = CryptoTradingBrain()

# تحلیل و تصمیم‌گیری
decision = brain.analyze_and_decide(
    symbol="BTC/USDT",
    timeframe="1h",
    use_mtf=True,
    fear_greed=45
)

# دریافت نتیجه
if decision:
    print(decision.to_json())
```

### استفاده از خط فرمان

```bash
# تحلیل یک نماد
python main.py --symbol BTC/USDT --timeframe 1h --mode live

# بک‌تست
python main.py --symbol BTC/USDT --timeframe 1h --mode backtest

# تحلیل چند تایم‌فریمی
python main.py --symbol ETH/USDT --mtf --equity 10000

# verbose logging
python main.py --symbol ADA/USDT --verbose
```

### استفاده از CLI

```python
from main import TradingSystemCLI

cli = TradingSystemCLI()

# تحلیل یک نماد
cli.analyze_single_symbol("BTC/USDT", "1h")

# تحلیل پورتفولیو
cli.analyze_portfolio(["BTC/USDT", "ETH/USDT", "ADA/USDT"])

# تست webhook
cli.run_webhook_test()

# نمایش وضعیت سیستم
cli.show_system_status()
```

---

## 🔌 انتگریشن n8n

### روش 1: Webhook Server

```python
from n8n_integration import WebhookServer

# راه‌اندازی سرور
server = WebhookServer(port=5000)

# در Flask:
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/trade', methods=['POST'])
def webhook():
    response = server.handle_webhook(request.json)
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)
```

### روش 2: n8n Code Node

```javascript
// در n8n Code Node

const integration = require('./n8n_integration.py');
const handler = new integration.N8NIntegration();

const webhook_data = {
    symbol: $json.symbol || "BTC/USDT",
    timeframe: $json.timeframe || "1h",
    use_mtf: true,
    equity: $json.equity || 10000,
    risk_per_trade: $json.risk_per_trade || 0.01
};

const response = handler.process_webhook(webhook_data);
return [{json: response}];
```

### Webhook Payload

```json
{
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "use_mtf": true,
    "equity": 10000,
    "risk_per_trade": 0.01,
    "max_leverage": 3,
    "fear_greed_index": 45
}
```

### Response Format

```json
{
    "status": "success",
    "symbol": "BTC/USDT",
    "action": "BUY",
    "entry_price": 50000.12,
    "take_profit": 52000.50,
    "stop_loss": 48000.75,
    "position_size": 0.25,
    "confidence": 0.85,
    "strategy": "TechnicalCombo",
    "risk_reward_ratio": 2.0,
    "reasons": [
        "RSI oversold (28.5)",
        "MACD bullish crossover",
        "Price above EMA20 & EMA50"
    ],
    "timestamp": "2025-11-13T10:30:45"
}
```

---

## 📊 استراتژی‌های معاملاتی

### 1. Strategy_Tech - ترکیب تکنیکال

**مؤلفه‌ها:**
- RSI (14 دوره): Oversold < 30, Overbought > 70
- MACD: Bullish/Bearish Crossovers
- EMA: Trend alignment (EMA9 > EMA20 > EMA50)
- Bollinger Bands: Price breakouts
- Volume: Confirmation

**وزن:** 1.2x

### 2. Strategy_Trend - تشخیص روند

**مؤلفه‌ها:**
- EMA Alignment: 9 > 20 > 50 > 100 > 200
- SMA Position: Price > SMA50 > SMA200
- ADX Strength: > 25 for strong trends
- ROC Momentum: Positive/Negative

**وزن:** 1.0x

### 3. Strategy_Breakout - شکست‌ها

**مؤلفه‌ها:**
- Candle Patterns: Engulfing, Hammer, Morning Star
- Bollinger Band Breakouts: With volume confirmation
- ATR Volatility: > 3% for high volatility moves
- Support/Resistance: Breakout above/below levels

**وزن:** 0.9x

### 4. Strategy_RiskSentiment - احساسات بازار

**مؤلفه‌ها:**
- Fear & Greed Index: Extreme values (< 25, > 75)
- Money Flow Index: < 20 oversold, > 80 overbought
- OBV Divergence: Price/volume divergence
- Buy/Sell Ratio: Volume-weighted analysis

**وزن:** 0.8x

---

## 💰 مدیریت ریسک

### پارامترهای قابل تنظیم

```python
risk_params = RiskParameters(
    equity=10000.0,                    # سرمایه اولیه
    risk_per_trade=0.01,              # 1% ریسک به ازای هر معامله
    max_position_size=0.1,            # حداکثر 10% پوزیشن
    max_leverage=3.0,                 # حداکثر 3x اهرم
    rr_ratio=2.0,                     # نسبت ریسک:ریوارد 1:2
    atr_multiplier_sl=1.5,            # Stop Loss = Entry - (1.5 * ATR)
    atr_multiplier_tp=3.0,            # Take Profit = Entry + (3 * ATR)
    max_daily_loss=0.05,              # حداکثر 5% ضرر روزانه
    max_drawdown=0.10                 # حداکثر 10% drawdown
)
```

### Presets پیکربندی

```python
from n8n_integration import ConfigManager

config_mgr = ConfigManager()

# Conservative
conservative = config_mgr.get_preset('conservative')

# Moderate (پیش‌فرض)
moderate = config_mgr.get_preset('moderate')

# Aggressive
aggressive = config_mgr.get_preset('aggressive')

# Scalping
scalping = config_mgr.get_preset('scalping')
```

---

## 🧪 تست‌ها

### اجرای تمام تست‌ها

```bash
# اجرای کامل
pytest test_crypto_brain.py -v

# با Coverage report
pytest test_crypto_brain.py --cov=. --cov-report=html

# تست خاصی
pytest test_crypto_brain.py::TestFeatureEngineer::test_rsi_calculation -v
```

### تست Webhook n8n

```python
from n8n_integration import N8NIntegration

integration = N8NIntegration()

# تست ورودی معتبر
response = integration.process_webhook({
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'equity': 10000
})

# تست ورودی نامعتبر
response = integration.process_webhook({
    'invalid': 'data'
})
```

---

## 📈 نمونه Output

```json
{
  "status": "success",
  "symbol": "BTC/USDT",
  "action": "BUY",
  "entry_price": 51234.56,
  "take_profit": 53421.78,
  "stop_loss": 49847.34,
  "position_size": 0.3256,
  "confidence": 0.8534,
  "strategy": "TechnicalCombo+TrendFollowing",
  "risk_reward_ratio": 2.23,
  "reasons": [
    "RSI oversold (24.5)",
    "MACD bullish crossover",
    "Price above EMA20 & EMA50",
    "Breakout above upper Bollinger Band with volume"
  ],
  "indicators": {
    "rsi": 24.5,
    "macd": 0.234,
    "ema_50": 50234.12,
    "ema_200": 48932.45,
    "atr": 567.89,
    "volume_ratio": 1.75
  },
  "timestamp": "2025-11-13T14:30:25.123456"
}
```

---

## 📋 نقشه راه | Roadmap

- [ ] Real-time WebSocket connections
- [ ] Historical backtesting dashboard
- [ ] Advanced portfolio analysis
- [ ] Machine learning prediction models
- [ ] Telegram/Discord notifications
- [ ] Database integration (PostgreSQL)
- [ ] GUI Dashboard (web-based)
- [ ] Docker containerization
- [ ] AWS Lambda deployment

---

## 🔒 امنیت

### نکات مهم

- ⚠️ **API Keys**: هرگز نمی‌اید API keys شخصی را در کد قرار دهید
- ⚠️ **Backtesting**: نتایج گذشته تضمین عملکرد آینده نیستند
- ⚠️ **Live Trading**: تست شامل در محیط demo/paper قبل از حساب واقعی
- ⚠️ **Risk Management**: همیشه stop losses و position sizing را قیمت بگذارید

---

## 📝 License

MIT License - [مطالعه بیشتر](LICENSE)

---

## 👥 Contributing

مشارکت‌ها پذیرفته می‌شوند! لطفاً:

1. Fork کنید
2. Feature branch بسازید (`git checkout -b feature/AmazingFeature`)
3. Commit کنید (`git commit -m 'Add AmazingFeature'`)
4. Push کنید (`git push origin feature/AmazingFeature`)
5. Pull Request بسازید

---

## 📞 تماس و پشتیبانی

- 📧 Email: [Erfansadegima@gmail.com]
- 💬 Telegram: [http://a7000i.t.me]
- 🐛 Issues: [GitHub Issues]

---

## 📚 منابع و مراجع

- [Binance API Documentation](https://binance-docs.github.io/)
- [KuCoin API Docs](https://docs.kucoin.com/)
- [n8n Documentation](https://docs.n8n.io/)
- [Technical Analysis Guide](https://www.investopedia.com/)

---

**ایجاد شده با ❤️ برای تریدرهای حرفه‌ای**

**Version:** 2.0.0  
**Last Updated:** November 13, 2025
