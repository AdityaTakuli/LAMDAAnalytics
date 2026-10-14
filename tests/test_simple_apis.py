#!/usr/bin/env python3
"""
Simple API key test without Unicode
"""

import asyncio
import sys
import os
import aiohttp

# Add backend to path
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path[:0] = [os.path.join(ROOT, 'backend'), ROOT]

from backend.config.settings import settings

async def test_weather_api():
    """Test Weather API key"""
    print("Testing Weather API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={settings.weather_api_key}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"SUCCESS Weather API: {data['name']} - {data['weather'][0]['description']}")
                    return True
                else:
                    print(f"FAILED Weather API: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"FAILED Weather API: {e}")
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
                    print(f"SUCCESS SERP API: Found {len(data.get('organic_results', []))} results")
                    return True
                else:
                    print(f"FAILED SERP API: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"FAILED SERP API: {e}")
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
                        print(f"SUCCESS Maps API: Found {data['results'][0]['formatted_address']}")
                        return True
                    else:
                        print(f"FAILED Maps API: {data.get('status')}")
                        return False
                else:
                    print(f"FAILED Maps API: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"FAILED Maps API: {e}")
        return False

async def main():
    """Test API keys"""
    print("=" * 50)
    print("API KEY VALIDATION TEST")
    print("=" * 50)
    
    results = await asyncio.gather(
        test_weather_api(),
        test_serp_api(),
        test_maps_api()
    )
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    apis = ["Weather", "SERP", "Maps"]
    for api, result in zip(apis, results):
        status = "WORKING" if result else "FAILED"
        print(f"{api:10}: {status}")
    
    working_count = sum(results)
    print(f"\nWorking APIs: {working_count}/3")
    
    if working_count >= 2:
        print("SUCCESS: Enough APIs working to run the orchestrator!")
        print("Note: Gemini API key needs to be valid for full functionality")
    else:
        print("FAILED: Need at least 2 working APIs for the orchestrator")

if __name__ == "__main__":
    asyncio.run(main())
