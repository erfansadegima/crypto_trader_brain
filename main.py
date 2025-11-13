#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
🧠 CRYPTO TRADING BRAIN v2.0
Advanced Automated Trading System with n8n Integration
═══════════════════════════════════════════════════════════════════════════

یک سیستم تریدینگ اتوماتیک حرفه‌ای برای کریپتو‌ارزها
سازگار با n8n، Binance، KuCoin، Coinbase و بیشتر
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tryd_fixed import CryptoTradingBrain, TradingAction
from n8n_integration import N8NIntegration, BatchAnalyzer, ConfigManager


def setup_logging(log_file: str = "crypto_trading.log", level: str = "INFO"):
    """راه‌اندازی لاگینگ"""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger


class TradingSystemCLI:
    """رابط خط فرمان برای سیستم تریدینگ"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.brain = CryptoTradingBrain()
        self.n8n_integration = N8NIntegration(logger=self.logger)
        self.batch_analyzer = BatchAnalyzer(logger=self.logger)
        self.config_manager = ConfigManager()
    
    def print_welcome(self):
        """نمایش پیام خوش‌آمدگویی"""
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                                                                  ║
        ║      🧠 CRYPTO TRADING BRAIN v2.0                               ║
        ║                                                                  ║
        ║      Advanced Automated Trading System                          ║
        ║      Compatible with n8n, Binance, KuCoin, Coinbase            ║
        ║                                                                  ║
        ║      📊 Multi-Strategy Analysis                                 ║
        ║      🤖 Intelligent Risk Management                             ║
        ║      🔄 Real-time Signal Generation                             ║
        ║      📱 n8n Webhook Integration                                 ║
        ║                                                                  ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
    
    def analyze_single_symbol(self, symbol: str, timeframe: str = "1h"):
        """تحلیل یک نماد"""
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing {symbol} on {timeframe} timeframe")
        print(f"{'='*70}\n")
        
        decision = self.brain.analyze_and_decide(
            symbol=symbol,
            timeframe=timeframe,
            use_mtf=True
        )
        
        if decision:
            print(decision.to_json())
        else:
            print("No actionable decision at this time.")
    
    def analyze_portfolio(self, symbols: list):
        """تحلیل پورتفولیو"""
        print(f"\n{'='*70}")
        print(f"📊 Analyzing Portfolio ({len(symbols)} symbols)")
        print(f"{'='*70}\n")
        
        summary = self.batch_analyzer.get_portfolio_summary(symbols)
        
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    def run_webhook_test(self, payload: dict = None):
        """تست Webhook n8n"""
        print(f"\n{'='*70}")
        print(f"📨 Testing n8n Webhook")
        print(f"{'='*70}\n")
        
        if payload is None:
            payload = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "use_mtf": True,
                "equity": 5000,
                "risk_per_trade": 0.01
            }
        
        print(f"📥 Input Payload:")
        print(json.dumps(payload, indent=2))
        
        response = self.n8n_integration.process_webhook(payload)
        
        print(f"\n📤 Output Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    
    def show_config_presets(self):
        """نمایش Presets پیکربندی"""
        print(f"\n{'='*70}")
        print(f"⚙️  Configuration Presets")
        print(f"{'='*70}\n")
        
        presets = self.config_manager.list_presets()
        
        for preset_name in presets:
            config = self.config_manager.get_preset(preset_name)
            print(f"📋 {preset_name.upper()}")
            print(f"   Risk per Trade:  {config['risk_per_trade']*100}%")
            print(f"   Max Leverage:    {config['max_leverage']}x")
            print(f"   Position Size:   {config['max_position_size']*100}%")
            print(f"   R:R Ratio:       1:{config['rr_ratio']}")
            print()
    
    def show_system_status(self):
        """نمایش وضعیت سیستم"""
        print(f"\n{'='*70}")
        print(f"🔧 System Status")
        print(f"{'='*70}\n")
        
        status = self.n8n_integration.get_status()
        
        print(f"Version:     {status['version']}")
        print(f"Status:      {status['status'].upper()}")
        print(f"\nModules:")
        
        for module, active in status['modules'].items():
            status_icon = "✅" if active else "❌"
            print(f"  {status_icon} {module}")
        
        print(f"\nActive Strategies:")
        for strategy in status['strategies']:
            print(f"  • {strategy}")


def main():
    """تابع اصلی"""
    
    cli = TradingSystemCLI()
    cli.print_welcome()
    
    # Example: Analyze single symbol
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Symbol Analysis")
    print("="*70)
    cli.analyze_single_symbol("BTC/USDT", "1h")
    
    # Example: Show config presets
    print("\n")
    cli.show_config_presets()
    
    # Example: Test webhook
    print("\n")
    cli.run_webhook_test()
    
    # Example: Show system status
    print("\n")
    cli.show_system_status()


if __name__ == "__main__":
    main()
