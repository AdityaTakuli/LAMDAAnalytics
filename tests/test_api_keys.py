#!/usr/bin/env python3
"""
Test individual API keys to verify they work
"""

import asyncio
import sys
import os
import aiohttp
import google.generativeai as genai

# Add backend to path
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path[:0] = [os.path.join(ROOT, 'backend'), ROOT]

from backend.config.settings import settings

async def test_gemini_api():
    """Test Gemini API key"""
    print("Testing Gemini API...")
    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, this is a test. Please respond with 'API working'.")
        print(f"✅ Gemini API: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return False

async def test_weather_api():
    """Test Weather API key"""
    print("Testing Weather API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={settings.weather_api_key}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Weather API: {data['name']} - {data['weather'][0]['description']}")
                    return True
                else:
                    print(f"❌ Weather API Error: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Weather API Error: {e}")
        return False

async def test_serp_api():
    """Test SERP API key"""
    print("Testing SERP API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://serpapi.com/search"
            params = {
                "q": "supply chain disruption",
                "api_key": settings.serp_api_key,
                "engine": "google"
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ SERP API: Found {len(data.get('organic_results', []))} results")
                    return True
                else:
                    print(f"❌ SERP API Error: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ SERP API Error: {e}")
        return False

async def test_maps_api():
    """Test Google Maps API key"""
    print("Testing Google Maps API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": "New York",
                "key": settings.google_maps_api_key
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'OK':
                        print(f"✅ Maps API: Found {data['results'][0]['formatted_address']}")
                        return True
                    else:
                        print(f"❌ Maps API Error: {data.get('status')}")
                        return False
                else:
                    print(f"❌ Maps API Error: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Maps API Error: {e}")
        return False

async def main():
    """Test all API keys"""
    print("=" * 50)
    print("API KEY VALIDATION TEST")
    print("=" * 50)
    
    results = await asyncio.gather(
        test_gemini_api(),
        test_weather_api(),
        test_serp_api(),
        test_maps_api()
    )
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    apis = ["Gemini", "Weather", "SERP", "Maps"]
    for api, result in zip(apis, results):
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"{api:10}: {status}")
    
    working_count = sum(results)
    print(f"\nWorking APIs: {working_count}/4")
    
    if working_count >= 2:
        print("✅ Enough APIs working to run the orchestrator!")
    else:
        print("❌ Need at least 2 working APIs for the orchestrator")

if __name__ == "__main__":
    asyncio.run(main())
