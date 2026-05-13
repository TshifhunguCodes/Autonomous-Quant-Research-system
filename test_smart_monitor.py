"""
Test script for Smart Monitor components
"""
import sys
import traceback

def test_imports():
    """Test all imports"""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import (
            TradePerformanceTracker,
            TradeQualityScorer,
            AdaptiveFilter,
            SimpleTradeLearner,
            SmartMonitor,
            get_smart_monitor
        )
        print("✅ All smart_monitor imports successful")
    except Exception as e:
        errors.append(f"Import error: {e}")
        print(f"❌ Import error: {e}")
    
    return errors

def test_instantiation():
    """Test class instantiation"""
    print("\n" + "=" * 50)
    print("Testing instantiation...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import TradePerformanceTracker
        tracker = TradePerformanceTracker()
        print("✅ TradePerformanceTracker instantiated")
    except Exception as e:
        errors.append(f"TradePerformanceTracker error: {e}")
        print(f"❌ TradePerformanceTracker error: {e}")
    
    try:
        from smart_monitor import TradeQualityScorer
        scorer = TradeQualityScorer()
        print("✅ TradeQualityScorer instantiated")
    except Exception as e:
        errors.append(f"TradeQualityScorer error: {e}")
        print(f"❌ TradeQualityScorer error: {e}")
    
    try:
        from smart_monitor import AdaptiveFilter
        adaptive = AdaptiveFilter()
        print("✅ AdaptiveFilter instantiated")
    except Exception as e:
        errors.append(f"AdaptiveFilter error: {e}")
        print(f"❌ AdaptiveFilter error: {e}")
    
    try:
        from smart_monitor import SimpleTradeLearner
        learner = SimpleTradeLearner()
        print("✅ SimpleTradeLearner instantiated")
    except Exception as e:
        errors.append(f"SimpleTradeLearner error: {e}")
        print(f"❌ SimpleTradeLearner error: {e}")
    
    try:
        from smart_monitor import SmartMonitor
        monitor = SmartMonitor()
        print("✅ SmartMonitor instantiated")
    except Exception as e:
        errors.append(f"SmartMonitor error: {e}")
        print(f"❌ SmartMonitor error: {e}")
    
    return errors

def test_execution_gate_import():
    """Test execution gate import"""
    print("\n" + "=" * 50)
    print("Testing ExecutionGate import...")
    print("=" * 50)
    
    errors = []
    
    try:
        from strategy.execution_gate import ExecutionGate
        print("✅ ExecutionGate import successful")
    except Exception as e:
        errors.append(f"ExecutionGate import error: {e}")
        print(f"❌ ExecutionGate import error: {e}")
        traceback.print_exc()
    
    return errors

def test_quality_scoring():
    """Test quality scoring with sample data"""
    print("\n" + "=" * 50)
    print("Testing quality scoring...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import TradeQualityScorer
        scorer = TradeQualityScorer()
        
        test_signal = {
            'time': '2026-05-08 10:00:00',
            'confirm_score': 70,
            'market_regime': 'TREND_UP',
            'htf_bias': 'BULLISH',
            'confirmed_signal': 'buy',
            'rsi14': 55,
            'volume': 1.2,
            'volume_avg_20': 1.0,
            'spread': 10,
            'fvg_bullish': True,
            'order_block': True,
            'liquidity_sweep': False,
            'bos': True,
            'choch': False,
            'major_support': True,
            'lifecycle_state': 'TREND_HEALTHY',
            'continuation_strength': 75
        }
        
        score = scorer.score_trade(test_signal)
        tier = scorer.get_quality_tier(score)
        print(f"✅ Quality score test: {score:.1f} ({tier})")
        
        # Test edge cases
        test_signal['confirm_score'] = 30
        score2 = scorer.score_trade(test_signal)
        print(f"✅ Low score test: {score2:.1f} ({scorer.get_quality_tier(score2)})")
        
        test_signal['confirm_score'] = 95
        score3 = scorer.score_trade(test_signal)
        print(f"✅ High score test: {score3:.1f} ({scorer.get_quality_tier(score3)})")
        
    except Exception as e:
        errors.append(f"Quality scoring error: {e}")
        print(f"❌ Quality scoring error: {e}")
        traceback.print_exc()
    
    return errors

def test_adaptive_filter():
    """Test adaptive filter"""
    print("\n" + "=" * 50)
    print("Testing adaptive filter...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import AdaptiveFilter
        adaptive = AdaptiveFilter()
        
        test_signal = {
            'time': '2026-05-08 10:00:00',
            'spread': 15,
            'market_regime': 'TREND_UP',
            'smart_quality_score': 70
        }
        
        allow, reason, lot_mult = adaptive.get_adaptive_filters(test_signal)
        print(f"✅ Adaptive filter test: allow={allow}, reason={reason}, lot_mult={lot_mult}")
        
    except Exception as e:
        errors.append(f"Adaptive filter error: {e}")
        print(f"❌ Adaptive filter error: {e}")
        traceback.print_exc()
    
    return errors

def test_ml_learner():
    """Test ML learner"""
    print("\n" + "=" * 50)
    print("Testing ML learner...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import SimpleTradeLearner
        learner = SimpleTradeLearner()
        
        test_signal = {
            'time': '2026-05-08 10:00:00',
            'confirm_score': 70,
            'market_regime': 'TREND_UP',
            'htf_bias': 'BULLISH',
            'confirmed_signal': 'buy',
            'rsi14': 55,
            'volume': 1.2,
            'volume_avg_20': 1.0,
            'spread': 10,
            'fvg_bullish': True,
            'order_block': True,
            'liquidity_sweep': False,
            'bos': True,
            'choch': False,
            'major_support': True,
            'lifecycle_state': 'TREND_HEALTHY'
        }
        
        prob = learner.predict_success_probability(test_signal)
        print(f"✅ ML prediction test: probability={prob:.2f}")
        
        allow, prob2, reason = learner.should_allow_trade(test_signal)
        print(f"✅ ML trade decision: allow={allow}, p={prob2:.2f}, reason={reason}")
        
    except Exception as e:
        errors.append(f"ML learner error: {e}")
        print(f"❌ ML learner error: {e}")
        traceback.print_exc()
    
    return errors

def test_smart_monitor_integration():
    """Test full SmartMonitor integration"""
    print("\n" + "=" * 50)
    print("Testing SmartMonitor integration...")
    print("=" * 50)
    
    errors = []
    
    try:
        from smart_monitor import get_smart_monitor
        
        monitor = get_smart_monitor()
        
        test_signal = {
            'time': '2026-05-08 10:00:00',
            'confirm_score': 70,
            'market_regime': 'TREND_UP',
            'htf_bias': 'BULLISH',
            'confirmed_signal': 'buy',
            'rsi14': 55,
            'volume': 1.2,
            'volume_avg_20': 1.0,
            'spread': 10,
            'fvg_bullish': True,
            'order_block': True,
            'liquidity_sweep': False,
            'bos': True,
            'choch': False,
            'major_support': True,
            'lifecycle_state': 'TREND_HEALTHY',
            'continuation_strength': 75
        }
        
        allow, quality, lot_mult, reason, tier = monitor.evaluate_signal(test_signal, "FLOW_EXP")
        print(f"✅ SmartMonitor evaluation: allow={allow}, quality={quality:.0f}, tier={tier}, lot_mult={lot_mult:.2f}")
        print(f"   Reason: {reason}")
        
    except Exception as e:
        errors.append(f"SmartMonitor integration error: {e}")
        print(f"❌ SmartMonitor integration error: {e}")
        traceback.print_exc()
    
    return errors

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SMART MONITOR TEST SUITE")
    print("=" * 60)
    
    all_errors = []
    
    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_instantiation())
    all_errors.extend(test_execution_gate_import())
    all_errors.extend(test_quality_scoring())
    all_errors.extend(test_adaptive_filter())
    all_errors.extend(test_ml_learner())
    all_errors.extend(test_smart_monitor_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all_errors:
        print(f"❌ {len(all_errors)} ERRORS FOUND:")
        for error in all_errors:
            print(f"  - {error}")
    else:
        print("✅ ALL TESTS PASSED - No errors found!")
    
    return len(all_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)