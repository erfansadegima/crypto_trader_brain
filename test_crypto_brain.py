"""
═══════════════════════════════════════════════════════════════════════════
UNIT TESTS FOR CRYPTO TRADING BRAIN
═══════════════════════════════════════════════════════════════════════════
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tryd_fixed import (
    CryptoTradingBrain,
    DataIngestor,
    Preprocessor,
    FeatureEngineer,
    PatternRecognizer,
    SignalGenerator,
    DecisionEngine,
    RiskManager,
    TradeExecutor,
    TradingAction,
    TrendDirection,
    RiskParameters,
    MarketData,
    TechnicalIndicators,
    Signal
)


class TestMarketData:
    """تست داده‌های بازار"""
    
    def test_market_data_creation(self):
        """تست ایجاد داده‌های بازار"""
        data = MarketData(
            timestamp=[1000000, 2000000],
            open=[100.0, 101.0],
            high=[102.0, 103.0],
            low=[99.0, 100.0],
            close=[101.0, 102.0],
            volume=[1000, 2000],
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert data.symbol == "BTC/USDT"
        assert len(data.close) == 2
        assert data.timeframe == "1h"
    
    def test_market_data_to_dataframe(self):
        """تست تبدیل به DataFrame"""
        data = MarketData(
            timestamp=[1000000000, 1000003600],
            open=[100.0, 101.0],
            high=[102.0, 103.0],
            low=[99.0, 100.0],
            close=[101.0, 102.0],
            volume=[1000, 2000],
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        df = data.to_dataframe()
        
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert 'close' in df.columns
        assert df['close'].iloc[0] == 101.0


class TestPreprocessor:
    """تست پیش‌پردازش داده‌ها"""
    
    @pytest.fixture
    def sample_market_data(self):
        """نمونه داده‌های بازار"""
        return MarketData(
            timestamp=[1000000000 + i*3600 for i in range(100)],
            open=[100.0 + i*0.1 for i in range(100)],
            high=[102.0 + i*0.1 for i in range(100)],
            low=[99.0 + i*0.1 for i in range(100)],
            close=[101.0 + i*0.1 for i in range(100)],
            volume=[1000 + i*10 for i in range(100)],
            symbol="BTC/USDT",
            timeframe="1h"
        )
    
    def test_preprocessor_init(self):
        """تست ایجاد پردازشگر"""
        preprocessor = Preprocessor()
        assert preprocessor is not None
    
    def test_remove_duplicates(self, sample_market_data):
        """تست حذف داده‌های تکراری"""
        preprocessor = Preprocessor()
        df = sample_market_data.to_dataframe()
        
        # Add duplicate
        df = pd.concat([df, df.iloc[-1:]], ignore_index=True)
        
        cleaned = preprocessor._remove_duplicates(df)
        assert len(cleaned) == len(df) - 1
    
    def test_full_processing(self, sample_market_data):
        """تست پیش‌پردازش کامل"""
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_market_data)
        
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'typical_price' in df.columns
        assert df['close'].isna().sum() == 0


class TestFeatureEngineer:
    """تست محاسبه اندیکاتورها"""
    
    @pytest.fixture
    def sample_df(self):
        """نمونه DataFrame"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices - np.abs(np.random.randn(100) * 0.2),
            'high': prices + np.abs(np.random.randn(100) * 0.3),
            'low': prices - np.abs(np.random.randn(100) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100),
            'typical_price': prices
        })
    
    def test_feature_engineer_init(self):
        """تست ایجاد Feature Engineer"""
        engineer = FeatureEngineer()
        assert engineer is not None
    
    def test_moving_averages(self, sample_df):
        """تست محاسبه میانگین‌های متحرک"""
        engineer = FeatureEngineer()
        df = engineer._compute_moving_averages(sample_df)
        
        assert 'sma_20' in df.columns
        assert 'ema_9' in df.columns
        assert df['sma_20'].isna().sum() < 20
    
    def test_rsi_calculation(self, sample_df):
        """تست محاسبه RSI"""
        engineer = FeatureEngineer()
        df = engineer._compute_rsi(sample_df)
        
        assert 'rsi_14' in df.columns
        assert df['rsi_14'].isna().sum() < 15
        assert (df['rsi_14'] >= 0).all() or df['rsi_14'].isna().any()
        assert (df['rsi_14'] <= 100).all() or df['rsi_14'].isna().any()
    
    def test_macd_calculation(self, sample_df):
        """تست محاسبه MACD"""
        engineer = FeatureEngineer()
        df = engineer._compute_macd(sample_df)
        
        assert 'macd' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_histogram' in df.columns
    
    def test_all_indicators(self, sample_df):
        """تست محاسبه تمام اندیکاتورها"""
        engineer = FeatureEngineer()
        df = engineer.compute_all_indicators(sample_df)
        
        # Check major indicators exist
        assert 'rsi_14' in df.columns
        assert 'macd' in df.columns
        assert 'bb_upper' in df.columns
        assert 'atr' in df.columns
        assert len(df) > 0


class TestPatternRecognizer:
    """تست شناسایی الگوها"""
    
    @pytest.fixture
    def bullish_engulfing_df(self):
        """نمونه داده برای Bullish Engulfing"""
        dates = pd.date_range('2023-01-01', periods=5, freq='1h')
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': [105.0, 100.0, 95.0, 98.0, 105.0],
            'high': [106.0, 102.0, 96.0, 110.0, 108.0],
            'low': [104.0, 98.0, 94.0, 97.0, 104.0],
            'close': [104.0, 99.0, 95.0, 108.0, 107.0],
            'volume': [1000, 1000, 1000, 2000, 1500]
        })
    
    def test_pattern_recognizer_init(self):
        """تست ایجاد Pattern Recognizer"""
        recognizer = PatternRecognizer()
        assert recognizer is not None
    
    def test_identify_candle_patterns(self, bullish_engulfing_df):
        """تست شناسایی الگوهای کندلی"""
        recognizer = PatternRecognizer()
        patterns = recognizer.identify_candle_patterns(bullish_engulfing_df)
        
        assert isinstance(patterns, list)


class TestRiskManager:
    """تست مدیریت ریسک"""
    
    @pytest.fixture
    def risk_manager(self):
        """ایجاد Risk Manager"""
        params = RiskParameters(
            equity=10000,
            risk_per_trade=0.01,
            max_position_size=0.1,
            max_leverage=3.0
        )
        return RiskManager(params)
    
    def test_position_size_calculation(self, risk_manager):
        """تست محاسبه حجم پوزیشن"""
        size = risk_manager.calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            action=TradingAction.BUY
        )
        
        assert size > 0
        assert size < 10000  # Within max position limit
    
    def test_stop_loss_calculation(self, risk_manager):
        """تست محاسبه حد ضرر"""
        sl = risk_manager.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            action=TradingAction.BUY
        )
        
        assert sl < 100.0
        assert sl > 0
    
    def test_take_profit_calculation(self, risk_manager):
        """تست محاسبه حد سود"""
        tp = risk_manager.calculate_take_profit(
            entry_price=100.0,
            stop_loss=95.0,
            action=TradingAction.BUY
        )
        
        assert tp > 100.0
    
    def test_risk_check(self, risk_manager):
        """تست بررسی ریسک"""
        signal = Signal(
            action=TradingAction.BUY,
            confidence=0.8,
            strategy_name="Test"
        )
        
        allowed, reason = risk_manager.check_risk_limits(signal)
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)


class TestSignalGeneration:
    """تست تولید سیگنال"""
    
    @pytest.fixture
    def sample_df_with_indicators(self):
        """نمونه DataFrame با اندیکاتورها"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices - np.abs(np.random.randn(100) * 0.2),
            'high': prices + np.abs(np.random.randn(100) * 0.3),
            'low': prices - np.abs(np.random.randn(100) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100),
            'typical_price': prices,
            'is_bullish': (np.random.rand(100) > 0.5).astype(int)
        })
        
        engineer = FeatureEngineer()
        return engineer.compute_all_indicators(df)
    
    def test_signal_generator_init(self):
        """تست ایجاد Signal Generator"""
        generator = SignalGenerator()
        assert generator is not None
    
    def test_signal_generation(self, sample_df_with_indicators):
        """تست تولید سیگنال"""
        generator = SignalGenerator()
        generator.initialize_default_strategies()
        
        engineer = FeatureEngineer()
        indicators = engineer.get_latest_indicators(sample_df_with_indicators)
        
        signals = generator.generate_signals(
            sample_df_with_indicators,
            indicators
        )
        
        assert isinstance(signals, list)
        assert len(signals) > 0
        assert all(isinstance(s, Signal) for s in signals)


class TestDecisionEngine:
    """تست موتور تصمیم‌گیری"""
    
    def test_decision_engine_init(self):
        """تست ایجاد Decision Engine"""
        engine = DecisionEngine()
        assert engine is not None
    
    def test_weighted_vote(self):
        """تست رای‌گیری وزن‌دار"""
        engine = DecisionEngine()
        
        signals = [
            Signal(
                action=TradingAction.BUY,
                confidence=0.8,
                strategy_name="Strategy1",
                weight=1.0
            ),
            Signal(
                action=TradingAction.BUY,
                confidence=0.7,
                strategy_name="Strategy2",
                weight=1.0
            ),
            Signal(
                action=TradingAction.SELL,
                confidence=0.3,
                strategy_name="Strategy3",
                weight=0.8
            )
        ]
        
        decision = engine.make_decision(signals, method="weighted_vote")
        
        assert decision.action in [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD]
        assert 0 <= decision.confidence <= 1.0


class TestCryptoTradingBrain:
    """تست مغز تریدر"""
    
    def test_brain_initialization(self):
        """تست ایجاد Trading Brain"""
        brain = CryptoTradingBrain()
        
        assert brain is not None
        assert brain.data_ingestor is not None
        assert brain.feature_engineer is not None
        assert brain.signal_generator is not None
        assert brain.decision_engine is not None
        assert brain.risk_manager is not None
    
    def test_default_config(self):
        """تست تنظیمات پیش‌فرض"""
        brain = CryptoTradingBrain()
        config = brain._default_config()
        
        assert 'exchange' in config
        assert 'equity' in config
        assert 'risk_per_trade' in config
    
    def test_custom_config(self):
        """تست تنظیمات سفارشی"""
        custom_config = {
            'equity': 50000,
            'risk_per_trade': 0.02,
            'exchange': 'kucoin'
        }
        
        brain = CryptoTradingBrain(config=custom_config)
        
        assert brain.config['equity'] == 50000
        assert brain.config['risk_per_trade'] == 0.02
        assert brain.config['exchange'] == 'kucoin'
    
    def test_n8n_webhook_processing(self):
        """تست پردازش Webhook n8n"""
        brain = CryptoTradingBrain()
        
        webhook_data = {
            'symbol': 'ETH/USDT',
            'timeframe': '15m',
            'use_mtf': False
        }
        
        output = brain.process_n8n_webhook(webhook_data)
        
        assert isinstance(output, str)
        # Output should be valid JSON
        import json
        parsed = json.loads(output)
        assert 'symbol' in parsed or 'error' in parsed


class TestIntegration:
    """تست یکپارچگی"""
    
    def test_full_pipeline(self):
        """تست لوله کامل"""
        # This is a simplified test without actual API calls
        brain = CryptoTradingBrain(config={'equity': 5000})
        
        # Verify all modules are properly initialized
        assert brain.signal_generator.strategies
        assert len(brain.signal_generator.strategies) > 0
    
    def test_json_output_format(self):
        """تست فرمت خروجی JSON"""
        from tryd_fixed import TradeDecision
        
        decision = TradeDecision(
            symbol="BTC/USDT",
            action=TradingAction.BUY,
            entry_price=50000.0,
            take_profit=52000.0,
            stop_loss=48000.0,
            position_size=0.5,
            confidence=0.85,
            strategy="TestStrategy"
        )
        
        json_output = decision.to_json()
        
        import json
        parsed = json.loads(json_output)
        
        assert parsed['symbol'] == "BTC/USDT"
        assert parsed['action'] == "BUY"
        assert parsed['entry_price'] == 50000.0


# ═══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
