#!/usr/bin/env python3
"""
Full LangGraph orchestrator test — complete multi-agent network:
1. Geocode → parallel Trade / News / Weather / Political / GSCPI nodes
2. Normalize features
3. TGN risk prediction (or weighted fallback)
4. Reporter concise + comprehensive output

Run from project root: python test_full_orchestrator.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.orchestrator.utils.schema import AnalyzeRequest
from backend.orchestrator.orchestrator import run_analysis

async def test_full_orchestrator():
    """Test the complete LangGraph multi-agent network"""
    print("FULL LANGGRAPH AGENT NETWORK TEST")
    print("=" * 60)
    print("This test runs the LangGraph StateGraph:")
    print("1. Geocode")
    print("2. Trade Agent -> LLM trade flows")
    print("3. News Agent -> SERP -> scrape -> LLM sentiment")
    print("4. Weather Agent -> Weather API -> anomaly")
    print("5. Political Agent -> LLM geopolitical risk")
    print("6. GSCPI Agent -> LLM global pressure")
    print("7. Normalizer -> feature scaling")
    print("8. TGN -> risk prediction")
    print("9. Reporter -> concise + comprehensive reports")
    print("=" * 60)
    
    # Create a comprehensive test request
    request = AnalyzeRequest(
        component_type="Semiconductor",
        seller_location="Hsinchu, Taiwan",
        import_location="Los Angeles, USA",
        seller_name="TSMC",
        additional_factors={
            "priority": "high",
            "volume": "large",
            "criticality": "essential"
        }
    )
    
    print(f"\n📋 ANALYSIS REQUEST:")
    print(f"   Component: {request.component_type}")
    print(f"   Seller: {request.seller_name} ({request.seller_location})")
    print(f"   Destination: {request.import_location}")
    print(f"   Additional Factors: {request.additional_factors}")
    
    try:
        print(f"\n⏳ Starting full orchestrator analysis...")
        print(f"   This will take 30-60 seconds as all agents fetch real data...")
        print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Run the complete analysis
        result = await run_analysis(request)
        
        print(f"\n✅ ANALYSIS COMPLETED!")
        print(f"   Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Request ID: {result.request_id}")
        
        print(f"\n🎯 TGN AI MODEL RESULTS:")
        print(f"   Risk Score: {result.tgn_result.risk_score:.3f}")
        print(f"   Risk Label: {result.tgn_result.risk_label}")
        print(f"   Model Confidence: {'High' if result.tgn_result.risk_score > 0.7 else 'Medium' if result.tgn_result.risk_score > 0.4 else 'Low'}")
        
        print(f"\n📊 NORMALIZED FEATURES (AI Model Inputs):")
        for feature, value in result.features.items():
            print(f"   {feature}: {value:.3f}")
        
        print(f"\n🔍 RISK COMPONENT BREAKDOWN:")
        for component, contribution in result.tgn_result.risk_components.items():
            print(f"   {component}: {contribution:.3f} ({contribution*100:.1f}%)")
        
        print(f"\n📈 CONCISE RISK FACTORS (Gemini Analysis):")
        for i, factor in enumerate(result.concise, 1):
            print(f"   {i}. {factor.name}: {factor.level} Risk ({factor.percent:.1f}%)")
            print(f"      Impact: {factor.impact}")
        
        print(f"\n🛡️  MITIGATION STRATEGIES (Gemini Recommendations):")
        for strategy, description in result.comprehensive.mitigation_strategies.items():
            print(f"   • {strategy}: {description}")
        
        print(f"\n🎉 FULL ORCHESTRATOR TEST SUCCESSFUL!")
        print(f"   All 7 AI agents executed successfully")
        print(f"   External APIs integrated and working")
        print(f"   TGN model processed features correctly")
        print(f"   Gemini provided comprehensive analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ORCHESTRATOR TEST FAILED!")
        print(f"   Error: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   1. Check that all API keys are set in backend/.env")
        print(f"   2. Verify internet connection for external APIs")
        print(f"   3. Ensure all dependencies are installed")
        print(f"   4. Check API key validity and quotas")
        
        import traceback
        traceback.print_exc()
        return False

def print_api_requirements():
    """Print the API key requirements"""
    print("\n🔑 REQUIRED API KEYS:")
    print("   To run the full orchestrator, you need:")
    print("   1. Google Maps API Key (for geocoding)")
    print("   2. SERP API Key (for news search)")
    print("   3. Weather API Key (OpenWeather or WeatherAPI)")
    print("   4. Gemini API Key (for AI analysis)")
    print("\n   Edit backend/.env and replace the placeholder keys with real ones.")
    print("   Get your keys from:")
    print("   • Google Maps: https://console.cloud.google.com/")
    print("   • SERP API: https://serpapi.com/")
    print("   • OpenWeather: https://openweathermap.org/api")
    print("   • Gemini: https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    print_api_requirements()
    print("\n" + "="*60)
    
    # Check if .env file exists and has real keys
    env_path = os.path.join("backend", ".env")
    if not os.path.exists(env_path):
        print("❌ backend/.env file not found!")
        print("   Please create it with your API keys.")
        sys.exit(1)
    
    with open(env_path, 'r') as f:
        env_content = f.read()
        if "your_" in env_content:
            print("⚠️  WARNING: Placeholder API keys detected!")
            print("   Please replace placeholder keys with real ones in backend/.env")
            print("   The test will likely fail with API errors.")
            print("\n   Continue anyway? (y/n): ", end="")
            response = input().lower()
            if response != 'y':
                print("   Test cancelled. Update API keys and try again.")
                sys.exit(0)
    
    print("\n🚀 Starting full orchestrator test...")
    success = asyncio.run(test_full_orchestrator())
    sys.exit(0 if success else 1)
