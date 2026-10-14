#!/usr/bin/env python3
"""
Mock test script to verify the orchestrator implementation without external APIs
Run from the project root: python tests/test_orchestrator_mock.py
"""

import asyncio
import sys
import os

# Add backend to path
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path[:0] = [os.path.join(ROOT, 'backend'), ROOT]

from backend.orchestrator.utils.schema import AnalyzeRequest, TradeFeatures, NewsFeatures, WeatherFeatures, PoliticalFeatures, GSCPIFeatures
from backend.orchestrator.agents.normalizer_agent import normalize_all
from backend.orchestrator.agents.reporter_agent import concise_from_contrib, comprehensive, label_from_score
from backend.models.tgn_model import tgn
from datetime import datetime, timezone

async def test_orchestrator_mock():
    """Test the orchestrator with mock data (no external APIs)"""
    print("Testing Supply Chain Risk Orchestrator (Mock Mode)...")
    
    # Create mock agent outputs
    trade = TradeFeatures(
        inventory_days=45.0,
        past_delay_days=8.0,
        edges=[]
    )
    
    news = NewsFeatures(
        news_vol_7d=12,
        neg_tone_frac_3d=0.35,
        strike_flag_7d=0,
        sources=["https://example.com/news1", "https://example.com/news2"]
    )
    
    weather = WeatherFeatures(
        weather_anomaly_7d=1,
        details={"provider": "mock", "anomaly_detected": True}
    )
    
    political = PoliticalFeatures(
        sanction_flag=0,
        political_risk_score=0.25,
        notes="Low political risk for this route"
    )
    
    gscpi = GSCPIFeatures(
        global_risk=0.18,
        timestamp="2024-01"
    )
    
    try:
        print("Processing mock data through normalizer and TGN...")
        
        # Normalize features
        now = datetime.now(timezone.utc)
        norm = normalize_all(now.isoformat(), trade, news, weather, political, gscpi)
        
        # Run TGN prediction
        risk_score, contrib = tgn.predict(norm.features)
        label = label_from_score(risk_score)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Risk Score: {risk_score:.3f}")
        print(f"Risk Label: {label}")
        print(f"Created At: {now.isoformat()}")
        
        print("\n=== NORMALIZED FEATURES ===")
        for feature, value in norm.features.items():
            print(f"- {feature}: {value:.3f}")
        
        print("\n=== RISK CONTRIBUTIONS ===")
        for factor, contribution in contrib.items():
            print(f"- {factor}: {contribution:.3f}")
        
        # Generate reports
        concise = concise_from_contrib(contrib, risk_score)
        comp = comprehensive(contrib)
        
        print("\n=== RISK FACTORS ===")
        for factor in concise:
            print(f"- {factor.name}: {factor.level} ({factor.percent:.1f}%) - {factor.impact}")
        
        print("\n=== MITIGATION STRATEGIES ===")
        for strategy, description in comp.mitigation_strategies.items():
            print(f"- {strategy}: {description}")
        
        print("\nSUCCESS: Mock test completed successfully!")
        print("The orchestrator components are working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: Mock test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("This test uses mock data to verify the orchestrator without external APIs.")
    print()
    
    success = asyncio.run(test_orchestrator_mock())
    sys.exit(0 if success else 1)
