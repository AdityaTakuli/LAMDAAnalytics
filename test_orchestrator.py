#!/usr/bin/env python3
"""
Verify the LangGraph orchestrator entry point.
Run from project root: python test_orchestrator.py
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.orchestrator.utils.schema import AnalyzeRequest
from backend.orchestrator.orchestrator import run_analysis

async def test_orchestrator():
    """Test LangGraph run_analysis with a simple request"""
    print("Testing LangGraph Supply Chain Risk Network...")
    
    # Create a test request
    request = AnalyzeRequest(
        component_type="Semiconductor",
        seller_location="Hsinchu, Taiwan",
        import_location="Los Angeles, USA",
        seller_name="TSMC",
        additional_factors={}
    )
    
    try:
        print(f"Analyzing: {request.component_type} from {request.seller_location} to {request.import_location}")
        print("This may take 30-60 seconds as it calls external APIs...")
        
        # Run the analysis
        result = await run_analysis(request)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Request ID: {result.request_id}")
        print(f"Risk Score: {result.tgn_result.risk_score:.3f}")
        print(f"Risk Label: {result.tgn_result.risk_label}")
        print(f"Created At: {result.created_at}")
        
        print("\n=== RISK FACTORS ===")
        for factor in result.concise:
            print(f"- {factor.name}: {factor.level} ({factor.percent:.1f}%) - {factor.impact}")
        
        print("\n=== NORMALIZED FEATURES ===")
        for feature, value in result.features.items():
            print(f"- {feature}: {value:.3f}")
        
        print("\n=== MITIGATION STRATEGIES ===")
        for strategy, description in result.comprehensive.mitigation_strategies.items():
            print(f"- {strategy}: {description}")
        
        print("\nSUCCESS: Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Note: This test requires API keys to be set in backend/.env
    print("WARNING: Make sure you have created backend/.env with your API keys!")
    print("   Copy backend/.env.example to backend/.env and add your keys.")
    print()
    
    success = asyncio.run(test_orchestrator())
    sys.exit(0 if success else 1)
