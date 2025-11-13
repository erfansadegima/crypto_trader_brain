"""
═══════════════════════════════════════════════════════════════════════════
    CRYPTO TRADING BRAIN - ADVANCED AUTOMATED TRADING SYSTEM
═══════════════════════════════════════════════════════════════════════════
نسخه: 2.0.0
توسعه‌دهنده: سیستم هوش مصنوعی معاملاتی
توضیحات: سیستم کامل تحلیل تکنیکال و تصمیم‌گیری خودکار برای معاملات کریپتو
سازگار با: n8n, Binance, KuCoin, Coinbase و سایر صرافی‌ها
═══════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import traceback

# External libraries
import numpy as np
import pandas as pd
import requests
from collections import deque, defaultdict
import hashlib
import time
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: LOGGING & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class LoggerSetup:
    """مدیریت لاگینگ حرفه‌ای با سطوح مختلف"""
    
    @staticmethod
    def setup_logger(name: str = "CryptoTradingBrain", level: int = logging.INFO) -> logging.Logger:
        """راه‌اندازی logger با فرمت دلخواه"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger

# Initialize global logger
LOGGER = LoggerSetup.setup_logger()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ENUMS & DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class TradingAction(Enum):
    """اقدامات معاملاتی ممکن"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class TimeFrame(Enum):
    """تایم‌فریم‌های معاملاتی"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class TrendDirection(Enum):
    """جهت روند"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class CandlePattern(Enum):
    """الگوهای کندل استیک"""
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    DOJI = "DOJI"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    NONE = "NONE"

@dataclass
class MarketData:
    """ساختار داده‌های بازار"""
    timestamp: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    symbol: str
    timeframe: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """تبدیل به DataFrame"""
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.timestamp, unit='ms'),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })

@dataclass
class TechnicalIndicators:
    """مجموعه اندیکاتورهای تکنیکال"""
    rsi: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_7: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_9: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    ema_200: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None
    sma_200: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    atr: Optional[float] = None
    atr_percentage: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    adx: Optional[float] = None
    cci: Optional[float] = None
    williams_r: Optional[float] = None
    obv: Optional[float] = None
    mfi: Optional[float] = None
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None
    vwap: Optional[float] = None
    
@dataclass
class Signal:
    """سیگنال معاملاتی"""
    action: TradingAction
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    reasons: List[str] = field(default_factory=list)
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class TradeDecision:
    """تصمیم نهایی معامله"""
    symbol: str
    action: TradingAction
    entry_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float = 0.0
    confidence: float = 0.0
    strategy: str = ""
    risk_reward_ratio: float = 0.0
    reasons: List[str] = field(default_factory=list)
    indicators: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """تبدیل به JSON برای n8n"""
        output = {
            "symbol": self.symbol,
            "action": self.action.value,
            "entry_price": round(self.entry_price, 8),
            "take_profit": round(self.take_profit, 8) if self.take_profit else None,
            "stop_loss": round(self.stop_loss, 8) if self.stop_loss else None,
            "position_size": round(self.position_size, 8),
            "confidence": round(self.confidence, 4),
            "strategy": self.strategy,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
            "indicators": self.indicators
        }
        return json.dumps(output, indent=2, ensure_ascii=False)

@dataclass
class RiskParameters:
    """پارامترهای مدیریت ریسک"""
    equity: float = 10000.0
    risk_per_trade: float = 0.01  # 1% of equity
    max_position_size: float = 0.1  # 10% of equity
    max_leverage: float = 3.0
    rr_ratio: float = 2.0  # Risk/Reward ratio
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 3.0
    max_daily_loss: float = 0.05  # 5%
    max_drawdown: float = 0.10  # 10%
    trailing_stop: bool = False
    trailing_stop_percentage: float = 0.02

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA INGESTION MODULE
# ═══════════════════════════════════════════════════════════════════════════

class DataIngestor:
    """
    ماژول دریافت داده از APIهای مختلف
    پشتیبانی از Binance, KuCoin, Coinbase و سایر صرافی‌ها
    """
    
    EXCHANGE_ENDPOINTS = {
        'binance': 'https://api.binance.com/api/v3/klines',
        'kucoin': 'https://api.kucoin.com/api/v1/market/candles',
        'coinbase': 'https://api.pro.coinbase.com/products/{symbol}/candles'
    }
    
    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
    }
    
    def __init__(self, exchange: str = 'binance', logger: logging.Logger = LOGGER):
        self.exchange = exchange.lower()
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoTradingBrain/2.0'})
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[MarketData]:
        """
        دریافت داده‌های OHLCV از صرافی
        
        Args:
            symbol: نماد ارز (مثل BTC/USDT)
            timeframe: بازه زمانی
            limit: تعداد کندل
            
        Returns:
            MarketData object یا None در صورت خطا
        """
        try:
            self.logger.info(f"📥 Fetching {limit} candles for {symbol} ({timeframe}) from {self.exchange}")
            
            if self.exchange == 'binance':
                data = self._fetch_binance(symbol, timeframe, limit)
            elif self.exchange == 'kucoin':
                data = self._fetch_kucoin(symbol, timeframe, limit)
            elif self.exchange == 'coinbase':
                data = self._fetch_coinbase(symbol, timeframe, limit)
            else:
                # Fallback: generate synthetic data for testing
                self.logger.warning(f"Exchange {self.exchange} not supported, generating synthetic data")
                data = self._generate_synthetic_data(symbol, timeframe, limit)
            
            if data:
                self.logger.info(f"✅ Successfully fetched {len(data.close)} candles")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _fetch_binance(self, symbol: str, timeframe: str, limit: int) -> Optional[MarketData]:
        """دریافت از Binance"""
        try:
            # Convert symbol format: BTC/USDT -> BTCUSDT
            binance_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': binance_symbol,
                'interval': self.TIMEFRAME_MAP.get(timeframe, '1h'),
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            response = self.session.get(
                self.EXCHANGE_ENDPOINTS['binance'],
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            # Parse Binance data format
            timestamps = [int(candle[0]) for candle in data]
            opens = [float(candle[1]) for candle in data]
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            closes = [float(candle[4]) for candle in data]
            volumes = [float(candle[5]) for candle in data]
            
            return MarketData(
                timestamp=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                volume=volumes,
                symbol=symbol,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"Binance fetch error: {str(e)}")
            return None
    
    def _fetch_kucoin(self, symbol: str, timeframe: str, limit: int) -> Optional[MarketData]:
        """دریافت از KuCoin"""
        try:
            # KuCoin uses different timeframe format
            tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                     '1h': '1hour', '4h': '4hour', '1d': '1day', '1w': '1week'}
            
            kucoin_symbol = symbol.replace('/', '-')
            
            params = {
                'symbol': kucoin_symbol,
                'type': tf_map.get(timeframe, '1hour'),
                'limit': min(limit, 1500)
            }
            
            response = self.session.get(
                self.EXCHANGE_ENDPOINTS['kucoin'],
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result['code'] != '200000' or not result['data']:
                return None
            
            data = result['data']
            
            # KuCoin format: [timestamp, open, close, high, low, volume, turnover]
            timestamps = [int(candle[0]) * 1000 for candle in reversed(data)]
            opens = [float(candle[1]) for candle in reversed(data)]
            closes = [float(candle[2]) for candle in reversed(data)]
            highs = [float(candle[3]) for candle in reversed(data)]
            lows = [float(candle[4]) for candle in reversed(data)]
            volumes = [float(candle[5]) for candle in reversed(data)]
            
            return MarketData(
                timestamp=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                volume=volumes,
                symbol=symbol,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"KuCoin fetch error: {str(e)}")
            return None
    
    def _fetch_coinbase(self, symbol: str, timeframe: str, limit: int) -> Optional[MarketData]:
        """دریافت از Coinbase Pro"""
        try:
            # Coinbase uses granularity in seconds
            granularity_map = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400
            }
            
            coinbase_symbol = symbol.replace('/', '-')
            granularity = granularity_map.get(timeframe, 3600)
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=granularity * limit)
            
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': granularity
            }
            
            url = self.EXCHANGE_ENDPOINTS['coinbase'].format(symbol=coinbase_symbol)
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            # Coinbase format: [timestamp, low, high, open, close, volume]
            data = sorted(data, key=lambda x: x[0])  # Sort by timestamp
            
            timestamps = [int(candle[0]) * 1000 for candle in data]
            lows = [float(candle[1]) for candle in data]
            highs = [float(candle[2]) for candle in data]
            opens = [float(candle[3]) for candle in data]
            closes = [float(candle[4]) for candle in data]
            volumes = [float(candle[5]) for candle in data]
            
            return MarketData(
                timestamp=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                volume=volumes,
                symbol=symbol,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"Coinbase fetch error: {str(e)}")
            return None
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str, limit: int) -> MarketData:
        """
        تولید داده‌های مصنوعی برای تست
        """
        np.random.seed(42)
        
        current_time = int(datetime.now().timestamp() * 1000)
        tf_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
        interval_ms = tf_seconds.get(timeframe, 3600) * 1000
        
        timestamps = [current_time - (limit - i) * interval_ms for i in range(limit)]
        
        # Generate realistic price movement using GBM
        base_price = 40000.0
        returns = np.random.normal(0.0001, 0.02, limit)
        prices = base_price * np.exp(np.cumsum(returns))
        
        closes = prices.tolist()
        opens = [closes[i-1] if i > 0 else closes[0] * 0.999 for i in range(limit)]
        highs = [max(opens[i], closes[i]) * (1 + abs(np.random.normal(0, 0.005))) for i in range(limit)]
        lows = [min(opens[i], closes[i]) * (1 - abs(np.random.normal(0, 0.005))) for i in range(limit)]
        volumes = [np.random.uniform(100, 1000) for _ in range(limit)]
        
        return MarketData(
            timestamp=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
            symbol=symbol,
            timeframe=timeframe
        )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA PREPROCESSING MODULE
# ═══════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    ماژول پیش‌پردازش داده‌ها
    شامل: پاکسازی، نرمال‌سازی، پر کردن داده‌های ناقص
    """
    
    def __init__(self, logger: logging.Logger = LOGGER):
        self.logger = logger
    
    def process(self, market_data: MarketData) -> pd.DataFrame:
        """
        پردازش کامل داده‌های بازار
        
        Returns:
            DataFrame تمیز و آماده
        """
        try:
            self.logger.info("🔧 Preprocessing market data...")
            
            # Convert to DataFrame
            df = market_data.to_dataframe()
            
            # Remove duplicates
            df = self._remove_duplicates(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Validate data integrity
            df = self._validate_data(df)
            
            # Add derived columns
            df = self._add_derived_columns(df)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"✅ Preprocessing complete: {len(df)} valid candles")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing error: {str(e)}")
            raise
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """حذف داده‌های تکراری"""
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        after = len(df)
        
        if before != after:
            self.logger.warning(f"Removed {before - after} duplicate candles")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """مدیریت داده‌های ناقص"""
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            
            # Forward fill for OHLCV
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
            
            # Fill volume with 0
            df['volume'] = df['volume'].fillna(0)
            
            # Drop remaining NaN rows
            df = df.dropna()
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """اعتبارسنجی داده‌ها"""
        
        # Ensure high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.sum() > 0:
            self.logger.warning(f"Fixing {invalid_hl.sum()} candles where high < low")
            df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
        
        # Ensure open/close within high/low
        df['open'] = df[['open', 'low']].max(axis=1)
        df['open'] = df[['open', 'high']].min(axis=1)
        df['close'] = df[['close', 'low']].max(axis=1)
        df['close'] = df[['close', 'high']].min(axis=1)
        
        # Remove zero or negative prices
        invalid_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if invalid_prices.sum() > 0:
            self.logger.warning(f"Removing {invalid_prices.sum()} candles with invalid prices")
            df = df[~invalid_prices]
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """اضافه کردن ستون‌های مشتق شده"""
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        
        # True Range
        df['tr'] = df[['high', 'low']].apply(
            lambda x: x['high'] - x['low'], axis=1
        )
        
        # Candle body
        df['body'] = abs(df['close'] - df['open'])
        
        # Upper/Lower shadow
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Candle direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE ENGINEERING MODULE
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    ماژول محاسبه اندیکاتورهای تکنیکال و ویژگی‌های پیشرفته
    """
    
    def __init__(self, logger: logging.Logger = LOGGER):
        self.logger = logger
    
    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه تمام اندیکاتورها"""
        try:
            self.logger.info("📊 Computing technical indicators...")
            
            df = df.copy()

            # Ensure basic derived columns exist
            if 'is_bullish' not in df.columns:
                df['is_bullish'] = (df['close'] > df['open']).astype(int)

            if 'typical_price' not in df.columns:
                df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Moving Averages
            df = self._compute_moving_averages(df)
            
            # RSI
            df = self._compute_rsi(df)
            
            # MACD
            df = self._compute_macd(df)
            
            # Bollinger Bands
            df = self._compute_bollinger_bands(df)
            
            # ATR
            df = self._compute_atr(df)
            
            # Stochastic
            df = self._compute_stochastic(df)
            
            # ADX
            df = self._compute_adx(df)
            
            # CCI
            df = self._compute_cci(df)
            
            # Williams %R
            df = self._compute_williams_r(df)
            
            # OBV
            df = self._compute_obv(df)
            
            # MFI
            df = self._compute_mfi(df)
            
            # VWAP
            df = self._compute_vwap(df)
            
            # Volume indicators
            df = self._compute_volume_indicators(df)
            
            # Momentum indicators
            df = self._compute_momentum_indicators(df)
            
            # Volatility clusters
            df = self._compute_volatility_clusters(df)
            
            self.logger.info(f"✅ Computed {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error computing indicators: {str(e)}")
            raise
    
    def _compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه میانگین‌های متحرک"""
        
        # Simple Moving Averages
        for period in [7, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        
        # Exponential Moving Averages
        for period in [9, 12, 20, 26, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Weighted Moving Average
        df['wma_20'] = df['close'].rolling(window=20).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True
        )
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """محاسبه RSI (Relative Strength Index)"""
        
        if periods is None:
            periods = [7, 14, 21]
        
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """محاسبه MACD"""
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _compute_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """محاسبه باندهای بولینگر"""
        
        df['bb_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=period, min_periods=1).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Avoid division by zero
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_width'] = ((bb_range) / (df['bb_middle'] + 1e-10)) * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه Average True Range"""
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period, min_periods=1).mean()
        df['atr_percentage'] = (df['atr'] / df['close']) * 100
        
        return df
    
    def _compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """محاسبه Stochastic Oscillator"""
        
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=1).mean()
        
        return df
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه Average Directional Index"""
        
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = df['atr'] if 'atr' in df.columns else df['high'] - df['low']
        
        pos_di = 100 * (pos_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))
        neg_di = 100 * (neg_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        df['adx'] = dx.rolling(window=period, min_periods=1).mean()
        df['pos_di'] = pos_di
        df['neg_di'] = neg_di
        
        return df
    
    def _compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """محاسبه Commodity Channel Index"""
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period, min_periods=1).mean()
        mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        return df
    
    def _compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه Williams %R"""
        
        high_max = df['high'].rolling(window=period, min_periods=1).max()
        low_min = df['low'].rolling(window=period, min_periods=1).min()
        
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
        
        return df
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه On-Balance Volume"""
        
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df
    
    def _compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه Money Flow Index"""
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
        
        mfi_ratio = positive_mf / (negative_mf + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def _compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه Volume Weighted Average Price"""
        
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _compute_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای حجمی"""
        
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # Buy/Sell volume estimation
        df['buy_volume'] = df['volume'] * df['is_bullish']
        df['sell_volume'] = df['volume'] * (1 - df['is_bullish'])
        
        df['buy_sell_ratio'] = (
            df['buy_volume'].rolling(window=10).sum() / 
            (df['sell_volume'].rolling(window=10).sum() + 1e-10)
        )
        
        return df
    
    def _compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای مومنتوم"""
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   (df['close'].shift(period) + 1e-10)) * 100
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Price velocity
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
        
        # Acceleration
        df['price_acceleration'] = df['price_velocity'].diff()
        
        return df
    
    def _compute_volatility_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """شناسایی خوشه‌های نوسان"""
        
        # Historical volatility
        df['hist_volatility_10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        df['hist_volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Volatility ratio
        df['volatility_ratio'] = df['hist_volatility_10'] / (df['hist_volatility_20'] + 1e-10)
        
        # High volatility flag
        df['high_volatility'] = (df['volatility_ratio'] > 1.5).astype(int)
        
        return df
    
    def get_latest_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """استخراج آخرین مقادیر اندیکاتورها"""
        
        latest = df.iloc[-1]
        
        return TechnicalIndicators(
            rsi=latest.get('rsi_14'),
            rsi_14=latest.get('rsi_14'),
            rsi_7=latest.get('rsi_7'),
            macd=latest.get('macd'),
            macd_signal=latest.get('macd_signal'),
            macd_histogram=latest.get('macd_histogram'),
            ema_9=latest.get('ema_9'),
            ema_20=latest.get('ema_20'),
            ema_50=latest.get('ema_50'),
            ema_100=latest.get('ema_100'),
            ema_200=latest.get('ema_200'),
            sma_20=latest.get('sma_20'),
            sma_50=latest.get('sma_50'),
            sma_100=latest.get('sma_100'),
            sma_200=latest.get('sma_200'),
            bb_upper=latest.get('bb_upper'),
            bb_middle=latest.get('bb_middle'),
            bb_lower=latest.get('bb_lower'),
            bb_width=latest.get('bb_width'),
            atr=latest.get('atr'),
            atr_percentage=latest.get('atr_percentage'),
            stoch_k=latest.get('stoch_k'),
            stoch_d=latest.get('stoch_d'),
            adx=latest.get('adx'),
            cci=latest.get('cci'),
            williams_r=latest.get('williams_r'),
            obv=latest.get('obv'),
            mfi=latest.get('mfi'),
            volume_sma=latest.get('volume_sma_20'),
            volume_ratio=latest.get('volume_ratio'),
            vwap=latest.get('vwap')
        )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: PATTERN RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════

class PatternRecognizer:
    """
    شناسایی الگوهای کندل استیک و الگوهای نموداری
    """
    
    def __init__(self, logger: logging.Logger = LOGGER):
        self.logger = logger
    
    def identify_candle_patterns(self, df: pd.DataFrame, lookback: int = 3) -> List[Dict]:
        """
        شناسایی الگوهای کندلی
        
        Returns:
            لیست الگوهای شناسایی شده
        """
        patterns = []
        
        if len(df) < lookback:
            return patterns
        
        recent = df.tail(lookback)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Engulfing patterns
        if self._is_bullish_engulfing(prev, latest):
            patterns.append({
                'pattern': CandlePattern.ENGULFING_BULLISH,
                'strength': 0.8,
                'description': 'Bullish Engulfing - Strong reversal signal'
            })
        
        if self._is_bearish_engulfing(prev, latest):
            patterns.append({
                'pattern': CandlePattern.ENGULFING_BEARISH,
                'strength': 0.8,
                'description': 'Bearish Engulfing - Strong reversal signal'
            })
        
        # Hammer
        if self._is_hammer(latest):
            patterns.append({
                'pattern': CandlePattern.HAMMER,
                'strength': 0.7,
                'description': 'Hammer - Potential bullish reversal'
            })
        
        # Shooting Star
        if self._is_shooting_star(latest):
            patterns.append({
                'pattern': CandlePattern.SHOOTING_STAR,
                'strength': 0.7,
                'description': 'Shooting Star - Potential bearish reversal'
            })
        
        # Doji
        if self._is_doji(latest):
            patterns.append({
                'pattern': CandlePattern.DOJI,
                'strength': 0.5,
                'description': 'Doji - Indecision, potential reversal'
            })
        
        # Morning Star (3-candle pattern)
        if len(recent) >= 3:
            if self._is_morning_star(recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]):
                patterns.append({
                    'pattern': CandlePattern.MORNING_STAR,
                    'strength': 0.85,
                    'description': 'Morning Star - Strong bullish reversal'
                })
        
        # Evening Star (3-candle pattern)
        if len(recent) >= 3:
            if self._is_evening_star(recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]):
                patterns.append({
                    'pattern': CandlePattern.EVENING_STAR,
                    'strength': 0.85,
                    'description': 'Evening Star - Strong bearish reversal'
                })
        
        return patterns
    
    def _is_bullish_engulfing(self, prev: pd.Series, current: pd.Series) -> bool:
        """بررسی الگوی Bullish Engulfing"""
        return (
            prev['close'] < prev['open'] and  # Previous candle bearish
            current['close'] > current['open'] and  # Current candle bullish
            current['open'] < prev['close'] and  # Opens below previous close
            current['close'] > prev['open']  # Closes above previous open
        )
    
    def _is_bearish_engulfing(self, prev: pd.Series, current: pd.Series) -> bool:
        """بررسی الگوی Bearish Engulfing"""
        return (
            prev['close'] > prev['open'] and  # Previous candle bullish
            current['close'] < current['open'] and  # Current candle bearish
            current['open'] > prev['close'] and  # Opens above previous close
            current['close'] < prev['open']  # Closes below previous open
        )
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """بررسی الگوی Hammer"""
        body = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        return (
            lower_shadow > 2 * body and
            upper_shadow < body * 0.3 and
            body > 0
        )
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """بررسی الگوی Shooting Star"""
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        
        return (
            upper_shadow > 2 * body and
            lower_shadow < body * 0.3 and
            body > 0
        )
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """بررسی الگوی Doji"""
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        
        return body < full_range * 0.1
    
    def _is_morning_star(self, first: pd.Series, second: pd.Series, third: pd.Series) -> bool:
        """بررسی الگوی Morning Star"""
        return (
            first['close'] < first['open'] and  # First bearish
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # Small body
            third['close'] > third['open'] and  # Third bullish
            third['close'] > (first['open'] + first['close']) / 2  # Closes above midpoint of first
        )
    
    def _is_evening_star(self, first: pd.Series, second: pd.Series, third: pd.Series) -> bool:
        """بررسی الگوی Evening Star"""
        return (
            first['close'] > first['open'] and  # First bullish
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # Small body
            third['close'] < third['open'] and  # Third bearish
            third['close'] < (first['open'] + first['close']) / 2  # Closes below midpoint of first
        )
    
    def identify_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """شناسایی الگوهای نموداری (Support/Resistance, Triangles, etc.)"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        # Support and Resistance levels
        support_resistance = self._find_support_resistance(df)
        patterns.extend(support_resistance)
        
        # Trend lines
        trend = self._identify_trend(df)
        if trend:
            patterns.append(trend)
        
        return patterns
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> List[Dict]:
        """شناسایی سطوح حمایت و مقاومت"""
        levels = []
        
        recent = df.tail(100)
        current_price = df.iloc[-1]['close']
        
        # Find local minima (support)
        for i in range(window, len(recent) - window):
            if recent.iloc[i]['low'] == recent.iloc[i - window:i + window]['low'].min():
                support_level = recent.iloc[i]['low']
                distance = abs(current_price - support_level) / current_price
                
                if distance < 0.05:  # Within 5%
                    levels.append({
                        'type': 'support',
                        'level': support_level,
                        'distance_pct': distance * 100,
                        'strength': 0.7
                    })
        
        # Find local maxima (resistance)
        for i in range(window, len(recent) - window):
            if recent.iloc[i]['high'] == recent.iloc[i - window:i + window]['high'].max():
                resistance_level = recent.iloc[i]['high']
                distance = abs(current_price - resistance_level) / current_price
                
                if distance < 0.05:  # Within 5%
                    levels.append({
                        'type': 'resistance',
                        'level': resistance_level,
                        'distance_pct': distance * 100,
                        'strength': 0.7
                    })
        
        return levels
    
    def _identify_trend(self, df: pd.DataFrame, period: int = 50) -> Optional[Dict]:
        """شناسایی روند کلی"""
        recent = df.tail(period)
        
        if len(recent) < period:
            return None
        
        # Linear regression for trend
        x = np.arange(len(recent))
        y = recent['close'].values
        
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Normalize slope
        avg_price = y.mean()
        slope_pct = (slope / avg_price) * 100
        
        if slope_pct > 0.1:
            return {
                'type': 'trend',
                'direction': TrendDirection.BULLISH,
                'strength': min(abs(slope_pct) / 2, 1.0),
                'slope_pct': slope_pct
            }
        elif slope_pct < -0.1:
            return {
                'type': 'trend',
                'direction': TrendDirection.BEARISH,
                'strength': min(abs(slope_pct) / 2, 1.0),
                'slope_pct': slope_pct
            }
        else:
            return {
                'type': 'trend',
                'direction': TrendDirection.SIDEWAYS,
                'strength': 0.5,
                'slope_pct': slope_pct
            }

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: TRADING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class TradingStrategy(ABC):
    """کلاس پایه برای استراتژی‌های معاملاتی"""
    
    def __init__(self, name: str, weight: float = 1.0, logger: logging.Logger = LOGGER):
        self.name = name
        self.weight = weight
        self.logger = logger
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Signal:
        """تولید سیگنال معاملاتی"""
        pass
    
    def _calculate_confidence(self, conditions: List[bool], weights: List[float] = None) -> float:
        """محاسبه میزان اطمینان بر اساس شرایط"""
        if not conditions:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(conditions)
        
        total_weight = sum(w for c, w in zip(conditions, weights) if c)
        max_weight = sum(weights)
        
        return total_weight / max_weight if max_weight > 0 else 0.0


class Strategy_Tech(TradingStrategy):
    """
    استراتژی تکنیکال ترکیبی
    ترکیب RSI + MACD + EMA + Bollinger Bands + Volume
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="TechnicalCombo", **kwargs)
    
    def generate_signal(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Signal:
        """تولید سیگنال بر اساس ترکیب اندیکاتورها"""
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        reasons = []
        buy_conditions = []
        sell_conditions = []
        weights = []
        
        # RSI Analysis
        if indicators.rsi_14 is not None:
            if indicators.rsi_14 < 30:
                buy_conditions.append(True)
                weights.append(1.0)
                reasons.append(f"RSI oversold ({indicators.rsi_14:.1f})")
            elif indicators.rsi_14 > 70:
                sell_conditions.append(True)
                weights.append(1.0)
                reasons.append(f"RSI overbought ({indicators.rsi_14:.1f})")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.3)
        
        # MACD Analysis
        if indicators.macd is not None and indicators.macd_signal is not None:
            # Get previous MACD values
            prev_macd = df.iloc[-2]['macd'] if len(df) > 1 and 'macd' in df.columns else indicators.macd
            prev_signal = df.iloc[-2]['macd_signal'] if len(df) > 1 and 'macd_signal' in df.columns else indicators.macd_signal
            
            macd_cross_up = (
                indicators.macd > indicators.macd_signal and
                prev_macd <= prev_signal
            )
            macd_cross_down = (
                indicators.macd < indicators.macd_signal and
                prev_macd >= prev_signal
            )
            
            if macd_cross_up:
                buy_conditions.append(True)
                weights.append(1.2)
                reasons.append("MACD bullish crossover")
            elif macd_cross_down:
                sell_conditions.append(True)
                weights.append(1.2)
                reasons.append("MACD bearish crossover")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.3)
        
        # EMA Trend Analysis
        if indicators.ema_20 is not None and indicators.ema_50 is not None:
            price = latest['close']
            
            if price > indicators.ema_20 > indicators.ema_50:
                buy_conditions.append(True)
                weights.append(0.8)
                reasons.append("Price above EMA20 & EMA50")
            elif price < indicators.ema_20 < indicators.ema_50:
                sell_conditions.append(True)
                weights.append(0.8)
                reasons.append("Price below EMA20 & EMA50")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.3)
        
        # Bollinger Bands
        if indicators.bb_lower is not None and indicators.bb_upper is not None:
            price = latest['close']
            
            if price < indicators.bb_lower:
                buy_conditions.append(True)
                weights.append(0.9)
                reasons.append("Price below lower Bollinger Band")
            elif price > indicators.bb_upper:
                sell_conditions.append(True)
                weights.append(0.9)
                reasons.append("Price above upper Bollinger Band")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.2)
        
        # Volume Confirmation
        if indicators.volume_ratio is not None:
            if indicators.volume_ratio > 1.5:
                if latest['close'] > latest['open']:
                    buy_conditions.append(True)
                    weights.append(0.7)
                    reasons.append(f"High volume bullish candle (ratio: {indicators.volume_ratio:.2f})")
                else:
                    sell_conditions.append(True)
                    weights.append(0.7)
                    reasons.append(f"High volume bearish candle (ratio: {indicators.volume_ratio:.2f})")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.2)
        
        # Decision Logic
        buy_confidence = self._calculate_confidence(buy_conditions, weights)
        sell_confidence = self._calculate_confidence(sell_conditions, weights)
        
        if buy_confidence > 0.6 and buy_confidence > sell_confidence:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        elif sell_confidence > 0.6 and sell_confidence > buy_confidence:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name=self.name,
                reasons=["No strong signal"],
                weight=self.weight
            )


class Strategy_Trend(TradingStrategy):
    """
    استراتژی تشخیص روند
    استفاده از میانگین‌های متحرک تطبیقی و ADX
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="TrendFollowing", **kwargs)
    
    def generate_signal(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Signal:
        """تولید سیگنال بر اساس روند"""
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        reasons = []
        buy_conditions = []
        sell_conditions = []
        weights = []
        
        # ADX Trend Strength
        strong_trend = indicators.adx is not None and indicators.adx > 25
        
        if strong_trend:
            reasons.append(f"Strong trend (ADX: {indicators.adx:.1f})")
        
        # Multi-timeframe EMA alignment
        if all([indicators.ema_9, indicators.ema_20, indicators.ema_50, indicators.ema_100]):
            bullish_alignment = (
                indicators.ema_9 > indicators.ema_20 > 
                indicators.ema_50 > indicators.ema_100
            )
            bearish_alignment = (
                indicators.ema_9 < indicators.ema_20 < 
                indicators.ema_50 < indicators.ema_100
            )
            
            if bullish_alignment:
                buy_conditions.append(True)
                weights.append(1.5 if strong_trend else 1.0)
                reasons.append("Bullish EMA alignment")
            elif bearish_alignment:
                sell_conditions.append(True)
                weights.append(1.5 if strong_trend else 1.0)
                reasons.append("Bearish EMA alignment")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.3)
        
        # Price position relative to SMAs
        if indicators.sma_50 and indicators.sma_200:
            price = latest['close']
            
            if price > indicators.sma_50 > indicators.sma_200:
                buy_conditions.append(True)
                weights.append(1.0)
                reasons.append("Price above SMA50 and SMA200")
            elif price < indicators.sma_50 < indicators.sma_200:
                sell_conditions.append(True)
                weights.append(1.0)
                reasons.append("Price below SMA50 and SMA200")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.2)
        
        # Momentum confirmation
        if 'roc_10' in df.columns:
            roc_10 = latest.get('roc_10', 0) if isinstance(latest, dict) else df.iloc[-1]['roc_10']
            if roc_10 > 2:
                buy_conditions.append(True)
                weights.append(0.7)
                reasons.append(f"Positive momentum (ROC: {roc_10:.2f}%)")
            elif roc_10 < -2:
                sell_conditions.append(True)
                weights.append(0.7)
                reasons.append(f"Negative momentum (ROC: {roc_10:.2f}%)")
        
        # Decision
        buy_confidence = self._calculate_confidence(buy_conditions, weights)
        sell_confidence = self._calculate_confidence(sell_conditions, weights)
        
        if buy_confidence > 0.65:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        elif sell_confidence > 0.65:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name=self.name,
                reasons=["Trend not strong enough"],
                weight=self.weight
            )


class Strategy_Breakout(TradingStrategy):
    """
    استراتژی شکست سطوح
    شناسایی شکست‌های حمایت/مقاومت و الگوهای کندلی
    """
    
    def __init__(self, pattern_recognizer: PatternRecognizer, **kwargs):
        super().__init__(name="BreakoutPattern", **kwargs)
        self.pattern_recognizer = pattern_recognizer
    
    def generate_signal(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Signal:
        """تولید سیگنال بر اساس شکست‌ها"""
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        reasons = []
        buy_conditions = []
        sell_conditions = []
        weights = []
        
        # Candle Patterns
        candle_patterns = self.pattern_recognizer.identify_candle_patterns(df)
        
        for pattern in candle_patterns:
            if pattern['pattern'] in [CandlePattern.ENGULFING_BULLISH, 
                                     CandlePattern.HAMMER, 
                                     CandlePattern.MORNING_STAR]:
                buy_conditions.append(True)
                weights.append(pattern['strength'])
                reasons.append(pattern['description'])
            
            elif pattern['pattern'] in [CandlePattern.ENGULFING_BEARISH, 
                                       CandlePattern.SHOOTING_STAR, 
                                       CandlePattern.EVENING_STAR]:
                sell_conditions.append(True)
                weights.append(pattern['strength'])
                reasons.append(pattern['description'])
        
        # Bollinger Band Breakout
        if indicators.bb_upper and indicators.bb_lower:
            price = latest['close']
            prev_price = prev['close'] if len(df) > 1 else price
            
            prev_bb_upper = df.iloc[-2]['bb_upper'] if len(df) > 1 and 'bb_upper' in df.columns else indicators.bb_upper
            prev_bb_lower = df.iloc[-2]['bb_lower'] if len(df) > 1 and 'bb_lower' in df.columns else indicators.bb_lower
            
            # Breakout above upper band
            if prev_price <= prev_bb_upper and price > indicators.bb_upper:
                if indicators.volume_ratio and indicators.volume_ratio > 1.3:
                    buy_conditions.append(True)
                    weights.append(0.9)
                    reasons.append("Breakout above upper Bollinger Band with volume")
            
            # Breakdown below lower band
            elif prev_price >= prev_bb_lower and price < indicators.bb_lower:
                if indicators.volume_ratio and indicators.volume_ratio > 1.3:
                    sell_conditions.append(True)
                    weights.append(0.9)
                    reasons.append("Breakdown below lower Bollinger Band with volume")
        
        # ATR Volatility Breakout
        if indicators.atr_percentage:
            if indicators.atr_percentage > 3.0:  # High volatility
                price_change = (latest['close'] - prev['close']) / prev['close']
                
                if price_change > 0.02:  # 2% move up
                    buy_conditions.append(True)
                    weights.append(0.7)
                    reasons.append(f"High volatility breakout upward (ATR: {indicators.atr_percentage:.2f}%)")
                elif price_change < -0.02:  # 2% move down
                    sell_conditions.append(True)
                    weights.append(0.7)
                    reasons.append(f"High volatility breakdown (ATR: {indicators.atr_percentage:.2f}%)")
        
        # Support/Resistance Breakout
        chart_patterns = self.pattern_recognizer.identify_chart_patterns(df)
        
        for pattern in chart_patterns:
            if pattern['type'] == 'support':
                if latest['close'] < pattern['level'] and prev['close'] >= pattern['level']:
                    sell_conditions.append(True)
                    weights.append(pattern['strength'])
                    reasons.append(f"Breakdown below support level {pattern['level']:.2f}")
            
            elif pattern['type'] == 'resistance':
                if latest['close'] > pattern['level'] and prev['close'] <= pattern['level']:
                    buy_conditions.append(True)
                    weights.append(pattern['strength'])
                    reasons.append(f"Breakout above resistance level {pattern['level']:.2f}")
        
        # Decision
        buy_confidence = self._calculate_confidence(buy_conditions, weights)
        sell_confidence = self._calculate_confidence(sell_conditions, weights)
        
        if buy_confidence > 0.6:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        elif sell_confidence > 0.6:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name=self.name,
                reasons=["No breakout detected"],
                weight=self.weight
            )


class Strategy_RiskSentiment(TradingStrategy):
    """
    استراتژی مبتنی بر ریسک و احساسات بازار
    استفاده از Fear & Greed Index و فیلترهای حجمی
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="RiskSentiment", **kwargs)
    
    def generate_signal(self, df: pd.DataFrame, indicators: TechnicalIndicators, 
                       fear_greed: Optional[int] = None) -> Signal:
        """
        تولید سیگنال بر اساس احساسات بازار
        
        Args:
            fear_greed: شاخص Fear & Greed (0-100)
        """
        
        latest = df.iloc[-1]
        
        reasons = []
        buy_conditions = []
        sell_conditions = []
        weights = []
        
        # Fear & Greed Analysis
        if fear_greed is not None:
            if fear_greed < 25:  # Extreme Fear
                buy_conditions.append(True)
                weights.append(1.2)
                reasons.append(f"Extreme Fear in market ({fear_greed}) - contrarian buy")
            elif fear_greed > 75:  # Extreme Greed
                sell_conditions.append(True)
                weights.append(1.2)
                reasons.append(f"Extreme Greed in market ({fear_greed}) - contrarian sell")
            else:
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(0.3)
        
        # Volume Profile Analysis
        if indicators.volume_ratio:
            if indicators.volume_ratio < 0.5:  # Very low volume
                # Low volume = avoid trading
                reasons.append(f"Low volume warning (ratio: {indicators.volume_ratio:.2f})")
                # Add negative weight
                buy_conditions.append(False)
                sell_conditions.append(False)
                weights.append(-0.5)
            elif indicators.volume_ratio > 2.0:  # High volume
                reasons.append(f"High volume confirmation (ratio: {indicators.volume_ratio:.2f})")
                # Don't add condition, just note it
        
        # Money Flow Index (buying/selling pressure)
        if indicators.mfi:
            if indicators.mfi < 20:  # Oversold with money flow
                buy_conditions.append(True)
                weights.append(0.9)
                reasons.append(f"MFI oversold ({indicators.mfi:.1f})")
            elif indicators.mfi > 80:  # Overbought with money flow
                sell_conditions.append(True)
                weights.append(0.9)
                reasons.append(f"MFI overbought ({indicators.mfi:.1f})")
        
        # OBV Divergence
        if 'obv' in df.columns and 'obv_ema' in df.columns:
            obv_trend = df.iloc[-1]['obv'] > df.iloc[-1]['obv_ema']
            price_trend = df.iloc[-1]['close'] > df.iloc[-10]['close'] if len(df) >= 10 else True
            
            if obv_trend and not price_trend:
                buy_conditions.append(True)
                weights.append(0.8)
                reasons.append("Bullish OBV divergence")
            elif not obv_trend and price_trend:
                sell_conditions.append(True)
                weights.append(0.8)
                reasons.append("Bearish OBV divergence")
        
        # Buy/Sell Ratio
        if 'buy_sell_ratio' in df.columns:
            buy_sell_ratio = df.iloc[-1]['buy_sell_ratio']
            if buy_sell_ratio > 1.5:
                buy_conditions.append(True)
                weights.append(0.6)
                reasons.append(f"Strong buying pressure (ratio: {buy_sell_ratio:.2f})")
            elif buy_sell_ratio < 0.67:
                sell_conditions.append(True)
                weights.append(0.6)
                reasons.append(f"Strong selling pressure (ratio: {buy_sell_ratio:.2f})")
        
        # Volatility Filter
        if 'high_volatility' in df.columns and df.iloc[-1]['high_volatility'] == 1:
            reasons.append("⚠️ High volatility detected - reduce position size")
        
        # Decision
        buy_confidence = self._calculate_confidence(buy_conditions, weights)
        sell_confidence = self._calculate_confidence(sell_conditions, weights)
        
        # Apply volume filter - reduce confidence if volume too low
        if indicators.volume_ratio and indicators.volume_ratio < 0.7:
            buy_confidence *= 0.7
            sell_confidence *= 0.7
        
        if buy_confidence > 0.55:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        elif sell_confidence > 0.55:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_confidence,
                strategy_name=self.name,
                reasons=reasons,
                weight=self.weight
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name=self.name,
                reasons=reasons if reasons else ["No clear sentiment signal"],
                weight=self.weight
            )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: SIGNAL GENERATOR & DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """
    ماژول تولید سیگنال‌ها از استراتژی‌های مختلف
    """
    
    def __init__(self, logger: logging.Logger = LOGGER):
        self.logger = logger
        self.strategies: List[TradingStrategy] = []
        self.pattern_recognizer = PatternRecognizer(logger=logger)
    
    def add_strategy(self, strategy: TradingStrategy):
        """افزودن استراتژی به لیست"""
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.name} (weight: {strategy.weight})")
    
    def initialize_default_strategies(self):
        """راه‌اندازی استراتژی‌های پیش‌فرض"""
        self.add_strategy(Strategy_Tech(weight=1.2))
        self.add_strategy(Strategy_Trend(weight=1.0))
        self.add_strategy(Strategy_Breakout(pattern_recognizer=self.pattern_recognizer, weight=0.9))
        self.add_strategy(Strategy_RiskSentiment(weight=0.8))
    
    def generate_signals(self, df: pd.DataFrame, indicators: TechnicalIndicators, 
                        fear_greed: Optional[int] = None) -> List[Signal]:
        """
        تولید سیگنال‌ها از تمام استراتژی‌ها
        """
        signals = []
        
        for strategy in self.strategies:
            try:
                if isinstance(strategy, Strategy_RiskSentiment):
                    signal = strategy.generate_signal(df, indicators, fear_greed)
                else:
                    signal = strategy.generate_signal(df, indicators)
                
                signals.append(signal)
                
                self.logger.info(
                    f"📊 {strategy.name}: {signal.action.value} "
                    f"(confidence: {signal.confidence:.2f})"
                )
                
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy.name}: {str(e)}")
                continue
        
        return signals


class DecisionEngine:
    """
    موتور تصمیم‌گیری نهایی
    ترکیب سیگنال‌های مختلف با وزن‌دهی
    """
    
    def __init__(self, logger: logging.Logger = LOGGER):
        self.logger = logger
    
    def make_decision(self, signals: List[Signal], method: str = "weighted_vote") -> Signal:
        """
        تصمیم‌گیری نهایی بر اساس سیگنال‌ها
        
        Args:
            signals: لیست سیگنال‌ها از استراتژی‌های مختلف
            method: روش ترکیب ("weighted_vote", "average", "majority")
        
        Returns:
            سیگنال نهایی
        """
        
        if not signals:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name="NoSignals",
                reasons=["No signals generated"]
            )
        
        if method == "weighted_vote":
            return self._weighted_vote(signals)
        elif method == "average":
            return self._average_consensus(signals)
        elif method == "majority":
            return self._majority_vote(signals)
        else:
            return self._weighted_vote(signals)
    
    def _weighted_vote(self, signals: List[Signal]) -> Signal:
        """رای‌گیری وزن‌دار"""
        
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        
        all_reasons = []
        active_strategies = []
        
        for signal in signals:
            weighted_confidence = signal.confidence * signal.weight
            
            if signal.action == TradingAction.BUY:
                buy_score += weighted_confidence
                all_reasons.extend(signal.reasons)
                active_strategies.append(signal.strategy_name)
            elif signal.action == TradingAction.SELL:
                sell_score += weighted_confidence
                all_reasons.extend(signal.reasons)
                active_strategies.append(signal.strategy_name)
            else:
                hold_score += signal.weight * 0.5
        
        total_weight = sum(s.weight for s in signals)
        
        # Normalize scores
        buy_confidence = buy_score / total_weight if total_weight > 0 else 0
        sell_confidence = sell_score / total_weight if total_weight > 0 else 0
        
        # Decision threshold
        threshold = 0.5
        
        if buy_confidence > threshold and buy_confidence > sell_confidence:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_confidence,
                strategy_name="+".join(active_strategies),
                reasons=all_reasons,
                weight=1.0
            )
        elif sell_confidence > threshold and sell_confidence > buy_confidence:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_confidence,
                strategy_name="+".join(active_strategies),
                reasons=all_reasons,
                weight=1.0
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=max(buy_confidence, sell_confidence),
                strategy_name="Consensus",
                reasons=["Signals not strong enough"] + all_reasons,
                weight=1.0
            )
    
    def _average_consensus(self, signals: List[Signal]) -> Signal:
        """میانگین‌گیری از سیگنال‌ها"""
        
        buy_signals = [s for s in signals if s.action == TradingAction.BUY]
        sell_signals = [s for s in signals if s.action == TradingAction.SELL]
        
        buy_avg = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
        sell_avg = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
        
        all_reasons = []
        for s in signals:
            all_reasons.extend(s.reasons)
        
        if buy_avg > 0.6 and buy_avg > sell_avg:
            return Signal(
                action=TradingAction.BUY,
                confidence=buy_avg,
                strategy_name="AverageConsensus",
                reasons=all_reasons
            )
        elif sell_avg > 0.6 and sell_avg > buy_avg:
            return Signal(
                action=TradingAction.SELL,
                confidence=sell_avg,
                strategy_name="AverageConsensus",
                reasons=all_reasons
            )
        else:
            return Signal(
                action=TradingAction.HOLD,
                confidence=0.0,
                strategy_name="AverageConsensus",
                reasons=all_reasons
            )
    
    def _majority_vote(self, signals: List[Signal]) -> Signal:
        """رای اکثریت ساده"""
        
        votes = {
            TradingAction.BUY: 0,
            TradingAction.SELL: 0,
            TradingAction.HOLD: 0
        }
        
        for signal in signals:
            votes[signal.action] += 1
        
        majority_action = max(votes, key=votes.get)
        
        relevant_signals = [s for s in signals if s.action == majority_action]
        avg_confidence = np.mean([s.confidence for s in relevant_signals]) if relevant_signals else 0
        
        all_reasons = []
        for s in relevant_signals:
            all_reasons.extend(s.reasons)
        
        return Signal(
            action=majority_action,
            confidence=avg_confidence,
            strategy_name="MajorityVote",
            reasons=all_reasons
        )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: RISK MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    ماژول مدیریت ریسک و محاسبه حجم پوزیشن
    """
    
    def __init__(self, risk_params: RiskParameters, logger: logging.Logger = LOGGER):
        self.params = risk_params
        self.logger = logger
        self.trade_history: List[Dict] = []
        self.current_drawdown = 0.0
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                action: TradingAction) -> float:
        """
        محاسبه حجم پوزیشن بر اساس ریسک
        
        Returns:
            حجم پوزیشن (به واحد ارز پایه)
        """
        
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        # Calculate risk per unit
        if action == TradingAction.BUY:
            risk_per_unit = abs(entry_price - stop_loss)
        else:  # SELL
            risk_per_unit = abs(stop_loss - entry_price)
        
        if risk_per_unit == 0:
            return 0.0
        
        # Risk amount in currency
        risk_amount = self.params.equity * self.params.risk_per_trade
        
        # Position size
        position_size = risk_amount / risk_per_unit
        
        # Apply max position size limit
        max_position = self.params.equity * self.params.max_position_size / entry_price
        position_size = min(position_size, max_position)
        
        # Apply leverage
        position_size *= self.params.max_leverage
        
        self.logger.info(
            f"💰 Position sizing: Risk=${risk_amount:.2f}, "
            f"Risk/unit=${risk_per_unit:.2f}, Size={position_size:.6f}"
        )
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                           action: TradingAction) -> float:
        """
        محاسبه حد ضرر بر اساس ATR
        """
        
        if action == TradingAction.BUY:
            stop_loss = entry_price - (atr * self.params.atr_multiplier_sl)
        else:  # SELL
            stop_loss = entry_price + (atr * self.params.atr_multiplier_sl)
        
        return max(stop_loss, 0)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             action: TradingAction) -> float:
        """
        محاسبه حد سود بر اساس نسبت ریسک به ریوارد
        """
        
        risk = abs(entry_price - stop_loss)
        reward = risk * self.params.rr_ratio
        
        if action == TradingAction.BUY:
            take_profit = entry_price + reward
        else:  # SELL
            take_profit = entry_price - reward
        
        return max(take_profit, 0)
    
    def check_risk_limits(self, signal: Signal) -> Tuple[bool, str]:
        """
        بررسی محدودیت‌های ریسک
        
        Returns:
            (allowed, reason)
        """
        
        # Check drawdown limit
        if self.current_drawdown > self.params.max_drawdown:
            return False, f"Max drawdown exceeded ({self.current_drawdown:.2%})"
        
        # Check daily loss limit
        daily_loss = self._calculate_daily_loss()
        if daily_loss > self.params.max_daily_loss:
            return False, f"Daily loss limit exceeded ({daily_loss:.2%})"
        
        # Check minimum confidence
        if signal.confidence < 0.5:
            return False, f"Confidence too low ({signal.confidence:.2f})"
        
        return True, "Risk check passed"
    
    def _calculate_daily_loss(self) -> float:
        """محاسبه ضرر امروز"""
        today = datetime.now().date()
        
        daily_trades = [
            t for t in self.trade_history
            if t.get('timestamp', datetime.now()).date() == today
        ]
        
        if not daily_trades:
            return 0.0
        
        total_pnl = sum(t.get('pnl', 0) for t in daily_trades)
        return abs(total_pnl / self.params.equity) if total_pnl < 0 else 0.0
    
    def update_equity(self, new_equity: float):
        """به‌روزرسانی سرمایه"""
        old_equity = self.params.equity
        self.params.equity = new_equity
        
        # Update drawdown
        if new_equity < old_equity:
            drawdown = (old_equity - new_equity) / old_equity
            self.current_drawdown = max(self.current_drawdown, drawdown)
        else:
            self.current_drawdown = 0.0
    
    def log_trade(self, trade: Dict):
        """ثبت معامله"""
        trade['timestamp'] = datetime.now()
        self.trade_history.append(trade)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: TRADE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════

class TradeExecutor:
    """
    ماژول اجرای معاملات و تولید خروجی JSON
    """
    
    def __init__(self, risk_manager: RiskManager, logger: logging.Logger = LOGGER):
        self.risk_manager = risk_manager
        self.logger = logger
    
    def execute(self, symbol: str, signal: Signal, current_price: float, 
                atr: float) -> Optional[TradeDecision]:
        """
        اجرای معامله و تولید تصمیم نهایی
        
        Returns:
            TradeDecision object یا None
        """
        
        try:
            # Check if action is actionable
            if signal.action == TradingAction.HOLD:
                self.logger.info("🔵 Decision: HOLD - No trade executed")
                return None
            
            # Risk check
            allowed, reason = self.risk_manager.check_risk_limits(signal)
            if not allowed:
                self.logger.warning(f"⛔ Trade rejected: {reason}")
                return None
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss(
                current_price, atr, signal.action
            )
            
            take_profit = self.risk_manager.calculate_take_profit(
                current_price, stop_loss, signal.action
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                current_price, stop_loss, signal.action
            )
            
            if position_size <= 0:
                self.logger.warning("⛔ Position size is zero or negative")
                return None
            
            # Calculate risk/reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Create decision
            decision = TradeDecision(
                symbol=symbol,
                action=signal.action,
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_size=position_size,
                confidence=signal.confidence,
                strategy=signal.strategy_name,
                risk_reward_ratio=rr_ratio,
                reasons=signal.reasons
            )
            
            # Log trade
            self.risk_manager.log_trade({
                'symbol': symbol,
                'action': signal.action.value,
                'entry_price': current_price,
                'position_size': position_size,
                'confidence': signal.confidence
            })
            
            self.logger.info(
                f"✅ Trade Decision: {signal.action.value} {symbol} @ {current_price:.2f}\n"
                f"   Position: {position_size:.6f}\n"
                f"   TP: {take_profit:.2f} | SL: {stop_loss:.2f}\n"
                f"   R:R = 1:{rr_ratio:.2f} | Confidence: {signal.confidence:.2%}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: MULTI-TIMEFRAME ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """
    تحلیل چند تایم‌فریمی برای تایید سیگنال‌ها
    """
    
    def __init__(self, data_ingestor: DataIngestor, feature_engineer: FeatureEngineer,
                 logger: logging.Logger = LOGGER):
        self.data_ingestor = data_ingestor
        self.feature_engineer = feature_engineer
        self.logger = logger
    
    def analyze_multiple_timeframes(self, symbol: str, 
                                   timeframes: List[str] = ['15m', '1h', '4h']) -> Dict:
        """
        تحلیل در چند تایم‌فریم
        
        Returns:
            دیکشنری حاوی اطلاعات هر تایم‌فریم
        """
        
        results = {}
        
        for tf in timeframes:
            try:
                self.logger.info(f"🔍 Analyzing {symbol} on {tf} timeframe...")
                
                # Fetch data
                market_data = self.data_ingestor.fetch_ohlcv(symbol, tf, limit=200)
                
                if not market_data:
                    continue
                
                # Preprocess
                preprocessor = Preprocessor(self.logger)
                df = preprocessor.process(market_data)
                
                # Compute indicators
                df = self.feature_engineer.compute_all_indicators(df)
                indicators = self.feature_engineer.get_latest_indicators(df)
                
                # Determine trend
                trend = self._determine_trend(df, indicators)
                
                results[tf] = {
                    'trend': trend,
                    'rsi': indicators.rsi_14,
                    'macd_histogram': indicators.macd_histogram,
                    'price': df.iloc[-1]['close'],
                    'ema_50': indicators.ema_50,
                    'ema_200': indicators.ema_200
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {tf}: {str(e)}")
                continue
        
        return results
    
    def _determine_trend(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> TrendDirection:
        """تعیین روند کلی"""
        
        latest = df.iloc[-1]
        price = latest['close']
        
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA alignment
        if indicators.ema_50 and indicators.ema_200:
            if price > indicators.ema_50 > indicators.ema_200:
                bullish_signals += 2
            elif price < indicators.ema_50 < indicators.ema_200:
                bearish_signals += 2
        
        # MACD
        if indicators.macd_histogram:
            if indicators.macd_histogram > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # ADX strength
        if indicators.adx and indicators.adx > 25:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            return TrendDirection.BULLISH
        elif bearish_signals > bullish_signals + 1:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def get_timeframe_consensus(self, mtf_results: Dict) -> Tuple[TrendDirection, float]:
        """
        محاسبه اجماع روند در تایم‌فریم‌های مختلف
        
        Returns:
            (روند غالب, درصد اطمینان)
        """
        
        if not mtf_results:
            return TrendDirection.UNKNOWN, 0.0
        
        trends = [data['trend'] for data in mtf_results.values()]
        
        bullish_count = trends.count(TrendDirection.BULLISH)
        bearish_count = trends.count(TrendDirection.BEARISH)
        total = len(trends)
        
        if bullish_count > bearish_count:
            return TrendDirection.BULLISH, bullish_count / total
        elif bearish_count > bullish_count:
            return TrendDirection.BEARISH, bearish_count / total
        else:
            return TrendDirection.SIDEWAYS, 0.5

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    ماژول بک‌تست استراتژی‌ها روی داده‌های تاریخی
    """
    
    def __init__(self, initial_capital: float = 10000, logger: logging.Logger = LOGGER):
        self.initial_capital = initial_capital
        self.logger = logger
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
    
    def run_backtest(self, df: pd.DataFrame, signal_generator: SignalGenerator,
                    decision_engine: DecisionEngine, risk_manager: RiskManager) -> Dict:
        """
        اجرای بک‌تست
        
        Returns:
            نتایج بک‌تست
        """
        
        self.logger.info("🔬 Starting backtest...")
        
        equity = self.initial_capital
        self.equity_curve = [equity]
        
        # Iterate through candles
        for i in range(100, len(df)):  # Start after warmup period
            window = df.iloc[:i+1]
            
            # Get indicators
            feature_engineer = FeatureEngineer(self.logger)
            df_features = feature_engineer.compute_all_indicators(window)
            indicators = feature_engineer.get_latest_indicators(df_features)
            
            # Generate signals
            signals = signal_generator.generate_signals(window, indicators)
            
            # Make decision
            final_signal = decision_engine.make_decision(signals)
            
            if final_signal.action in [TradingAction.BUY, TradingAction.SELL]:
                # Simulate trade
                entry_price = window.iloc[-1]['close']
                atr = indicators.atr if indicators.atr else entry_price * 0.02
                
                stop_loss = risk_manager.calculate_stop_loss(
                    entry_price, atr, final_signal.action
                )
                take_profit = risk_manager.calculate_take_profit(
                    entry_price, stop_loss, final_signal.action
                )
                
                position_size = risk_manager.calculate_position_size(
                    entry_price, stop_loss, final_signal.action
                )
                
                # Simulate outcome (simplified)
                future_prices = df.iloc[i+1:i+20]['close'] if i+20 < len(df) else df.iloc[i+1:]['close']
                
                if len(future_prices) > 0:
                    if final_signal.action == TradingAction.BUY:
                        hit_tp = (future_prices >= take_profit).any()
                        hit_sl = (future_prices <= stop_loss).any()
                    else:
                        hit_tp = (future_prices <= take_profit).any()
                        hit_sl = (future_prices >= stop_loss).any()
                    
                    if hit_tp:
                        pnl = abs(take_profit - entry_price) * position_size
                        equity += pnl
                        outcome = 'WIN'
                    elif hit_sl:
                        pnl = -abs(entry_price - stop_loss) * position_size
                        equity += pnl
                        outcome = 'LOSS'
                    else:
                        pnl = 0
                        outcome = 'OPEN'
                    
                    self.trades.append({
                        'entry_price': entry_price,
                        'action': final_signal.action.value,
                        'pnl': pnl,
                        'outcome': outcome,
                        'equity': equity
                    })
                    
                    self.equity_curve.append(equity)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """محاسبه معیارهای عملکرد"""
        
        if not self.trades:
            return {}
        
        wins = [t for t in self.trades if t['outcome'] == 'WIN']
        losses = [t for t in self.trades if t['outcome'] == 'LOSS']
        
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        
        returns = [(self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1] 
                   for i in range(1, len(self.equity_curve))]
        
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown()
        
        metrics = {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_equity': final_equity,
            'return_pct': ((final_equity - self.initial_capital) / self.initial_capital) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        self.logger.info(f"\n{'='*50}\n📈 BACKTEST RESULTS\n{'='*50}")
        self.logger.info(f"Total Trades: {total_trades}")
        self.logger.info(f"Win Rate: {win_rate:.2%}")
        self.logger.info(f"Total P/L: ${total_pnl:.2f}")
        self.logger.info(f"Return: {metrics['return_pct']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """محاسبه حداکثر افت سرمایه"""
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN TRADING BRAIN
# ═══════════════════════════════════════════════════════════════════════════

class CryptoTradingBrain:
    """
    کلاس اصلی - مغز تریدر
    یکپارچه‌سازی تمام ماژول‌ها
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = LOGGER):
        """
        راه‌اندازی سیستم معاملاتی
        
        Args:
            config: تنظیمات سیستم
        """
        self.logger = logger
        self.config = config or self._default_config()
        
        self.logger.info("🧠 Initializing Crypto Trading Brain...")
        
        # Initialize modules
        self.data_ingestor = DataIngestor(
            exchange=self.config.get('exchange', 'binance'),
            logger=self.logger
        )
        
        self.preprocessor = Preprocessor(logger=self.logger)
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.pattern_recognizer = PatternRecognizer(logger=self.logger)
        
        self.signal_generator = SignalGenerator(logger=self.logger)
        self.signal_generator.initialize_default_strategies()
        
        self.decision_engine = DecisionEngine(logger=self.logger)
        
        # Risk parameters
        risk_params = RiskParameters(
            equity=self.config.get('equity', 10000),
            risk_per_trade=self.config.get('risk_per_trade', 0.01),
            max_position_size=self.config.get('max_position_size', 0.1),
            max_leverage=self.config.get('max_leverage', 3.0),
            rr_ratio=self.config.get('rr_ratio', 2.0)
        )
        
        self.risk_manager = RiskManager(risk_params, logger=self.logger)
        self.trade_executor = TradeExecutor(self.risk_manager, logger=self.logger)
        
        self.mtf_analyzer = MultiTimeframeAnalyzer(
            self.data_ingestor,
            self.feature_engineer,
            logger=self.logger
        )
        
        self.logger.info("✅ Trading Brain initialized successfully!")
    
    def _default_config(self) -> Dict:
        """تنظیمات پیش‌فرض"""
        return {
            'exchange': 'binance',
            'equity': 10000,
            'risk_per_trade': 0.01,
            'max_position_size': 0.1,
            'max_leverage': 3.0,
            'rr_ratio': 2.0,
            'timeframe': '1h',
            'limit': 500
        }
    
    def analyze_and_decide(self, symbol: str, timeframe: str = None, 
                          use_mtf: bool = True, fear_greed: Optional[int] = None) -> Optional[TradeDecision]:
        """
        تحلیل کامل و تصمیم‌گیری
        
        Args:
            symbol: نماد ارز (e.g., "BTC/USDT")
            timeframe: تایم‌فریم اصلی
            use_mtf: استفاده از تحلیل چند تایم‌فریمی
            fear_greed: شاخص Fear & Greed
        
        Returns:
            TradeDecision یا None
        """
        
        try:
            self.logger.info(f"\n{'='*70}\n🎯 ANALYZING {symbol}\n{'='*70}")
            
            timeframe = timeframe or self.config.get('timeframe', '1h')
            
            # 1. Fetch market data
            market_data = self.data_ingestor.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=self.config.get('limit', 500)
            )
            
            if not market_data:
                self.logger.error("Failed to fetch market data")
                return None
            
            # 2. Preprocess
            df = self.preprocessor.process(market_data)
            
            # 3. Compute indicators
            df = self.feature_engineer.compute_all_indicators(df)
            indicators = self.feature_engineer.get_latest_indicators(df)
            
            # 4. Multi-timeframe analysis (optional)
            if use_mtf:
                mtf_results = self.mtf_analyzer.analyze_multiple_timeframes(symbol)
                mtf_trend, mtf_confidence = self.mtf_analyzer.get_timeframe_consensus(mtf_results)
                
                self.logger.info(f"📊 Multi-Timeframe Consensus: {mtf_trend.value} ({mtf_confidence:.0%})")
            
            # 5. Generate signals from all strategies
            signals = self.signal_generator.generate_signals(df, indicators, fear_greed)
            
            # 6. Make final decision
            final_signal = self.decision_engine.make_decision(signals, method="weighted_vote")
            
            self.logger.info(
                f"\n{'─'*70}\n"
                f"📋 FINAL SIGNAL: {final_signal.action.value}\n"
                f"   Confidence: {final_signal.confidence:.2%}\n"
                f"   Strategy: {final_signal.strategy_name}\n"
                f"{'─'*70}"
            )
            
            # 7. Execute trade (if actionable)
            if final_signal.action != TradingAction.HOLD:
                current_price = df.iloc[-1]['close']
                atr = indicators.atr if indicators.atr else current_price * 0.02
                
                decision = self.trade_executor.execute(
                    symbol=symbol,
                    signal=final_signal,
                    current_price=current_price,
                    atr=atr
                )
                
                # Add indicators to decision
                if decision:
                    decision.indicators = {
                        'rsi': indicators.rsi_14,
                        'macd': indicators.macd,
                        'ema_50': indicators.ema_50,
                        'ema_200': indicators.ema_200,
                        'atr': indicators.atr,
                        'volume_ratio': indicators.volume_ratio
                    }
                
                return decision
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error in analysis: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def run_backtest(self, symbol: str, timeframe: str = '1h') -> Dict:
        """
        اجرای بک‌تست
        """
        self.logger.info(f"🔬 Starting backtest for {symbol} on {timeframe}")
        
        # Fetch more data for backtest
        market_data = self.data_ingestor.fetch_ohlcv(symbol, timeframe, limit=1000)
        
        if not market_data:
            return {}
        
        df = self.preprocessor.process(market_data)
        df = self.feature_engineer.compute_all_indicators(df)
        
        backtester = Backtester(
            initial_capital=self.config.get('equity', 10000),
            logger=self.logger
        )
        
        results = backtester.run_backtest(
            df,
            self.signal_generator,
            self.decision_engine,
            self.risk_manager
        )
        
        return results
    
    def process_n8n_webhook(self, webhook_data: Dict) -> str:
        """
        پردازش داده‌های ورودی از n8n
        
        Args:
            webhook_data: دیکشنری حاوی پارامترهای ورودی
        
        Returns:
            JSON string خروجی
        """
        
        try:
            # Extract parameters
            symbol = webhook_data.get('symbol', 'BTC/USDT')
            timeframe = webhook_data.get('timeframe', '1h')
            use_mtf = webhook_data.get('use_mtf', True)
            fear_greed = webhook_data.get('fear_greed')
            
            # Update config if provided
            if 'equity' in webhook_data:
                self.risk_manager.params.equity = float(webhook_data['equity'])
            
            if 'risk_per_trade' in webhook_data:
                self.risk_manager.params.risk_per_trade = float(webhook_data['risk_per_trade'])
            
            # Analyze and decide
            decision = self.analyze_and_decide(
                symbol=symbol,
                timeframe=timeframe,
                use_mtf=use_mtf,
                fear_greed=fear_greed
            )
            
            if decision:
                return decision.to_json()
            else:
                # Return HOLD signal
                hold_decision = TradeDecision(
                    symbol=symbol,
                    action=TradingAction.HOLD,
                    entry_price=0.0,
                    confidence=0.0,
                    strategy="NoAction",
                    reasons=["No strong signal detected"]
                )
                return hold_decision.to_json()
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "failed"
            }
            return json.dumps(error_response, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: CLI & MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def parse_arguments():
    """پارس کردن آرگومان‌های خط فرمان"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🧠 Crypto Trading Brain - Advanced Automated Trading System"
    )
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading symbol (e.g., BTC/USDT)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d)')
    
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange name (binance, kucoin, coinbase)')
    
    parser.add_argument('--equity', type=float, default=10000,
                       help='Initial capital/equity')
    
    parser.add_argument('--risk', type=float, default=0.01,
                       help='Risk per trade (0.01 = 1%)')
    
    parser.add_argument('--mode', type=str, default='live',
                       choices=['live', 'backtest'],
                       help='Operation mode')
    
    parser.add_argument('--mtf', action='store_true',
                       help='Enable multi-timeframe analysis')
    
    parser.add_argument('--fear-greed', type=int, default=None,
                       help='Fear & Greed Index (0-100)')
    
    parser.add_argument('--json-input', type=str, default=None,
                       help='Path to JSON input file (for n8n simulation)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """
    تابع اصلی اجرا
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║      🧠 CRYPTO TRADING BRAIN v2.0                               ║
    ║      Advanced Automated Trading System                          ║
    ║                                                                  ║
    ║      Developed for professional crypto trading                  ║
    ║      Compatible with n8n, Binance, KuCoin, Coinbase            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    args = parse_arguments()
    
    # Setup logger level
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # Configuration
    config = {
        'exchange': args.exchange,
        'equity': args.equity,
        'risk_per_trade': args.risk,
        'timeframe': args.timeframe
    }
    
    # Initialize Trading Brain
    brain = CryptoTradingBrain(config=config)
    
    # Mode: Backtest or Live
    if args.mode == 'backtest':
        LOGGER.info("📊 Running in BACKTEST mode")
        results = brain.run_backtest(symbol=args.symbol, timeframe=args.timeframe)
        
        if results:
            print(f"\n{'='*70}")
            print("📈 BACKTEST SUMMARY")
            print(f"{'='*70}")
            for key, value in results.items():
                print(f"{key:20s}: {value}")
            print(f"{'='*70}\n")
    
    else:
        LOGGER.info("🔴 Running in LIVE mode")
        
        # Check if JSON input provided (n8n simulation)
        if args.json_input:
            try:
                with open(args.json_input, 'r') as f:
                    webhook_data = json.load(f)
                
                output = brain.process_n8n_webhook(webhook_data)
                print("\n" + "="*70)
                print("📤 OUTPUT FOR N8N:")
                print("="*70)
                print(output)
                print("="*70 + "\n")
                
            except Exception as e:
                LOGGER.error(f"Error reading JSON input: {str(e)}")
        
        else:
            # Standard live analysis
            decision = brain.analyze_and_decide(
                symbol=args.symbol,
                timeframe=args.timeframe,
                use_mtf=args.mtf,
                fear_greed=args.fear_greed
            )
            
            if decision:
                print("\n" + "="*70)
                print("📤 TRADE DECISION (JSON OUTPUT):")
                print("="*70)
                print(decision.to_json())
                print("="*70 + "\n")
            else:
                print("\n⚪ No actionable decision at this time.\n")
    
    LOGGER.info("✅ Execution completed!")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15: N8N INTEGRATION EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════
📱 INTEGRATION WITH N8N - USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════

### Example 1: Direct Function Call (in n8n Code node)

```python
from main import CryptoTradingBrain

# Input from previous node
input_data = $json

# Initialize
brain = CryptoTradingBrain()

# Process
output = brain.process_n8n_webhook(input_data)

# Return to next node
return [{"json": json.loads(output)}]
"""