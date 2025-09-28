#!/usr/bin/env python3
"""
Test Google Gemini API connection
"""

import google.generativeai as genai
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.config.settings import settings

def test_gemini_api():
    """Test Gemini API with the provided key"""
    print("=" * 60)
    print("GOOGLE GEMINI API TEST")
    print("=" * 60)
    
    print(f"API Key: {'*' * 20}... (hidden for security)")
    print(f"Project Number: [HIDDEN]")
    print()
    
    try:
        # Configure the API
        genai.configure(api_key=settings.gemini_api_key)
        
        # List available models first
        print("Available Models:")
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
        print()
        
        # Try different model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                print(f"Testing model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello, this is a test. Please respond with 'Google API is working!'")
                print(f"SUCCESS: {model_name}")
                print(f"Response: {response.text}")
                print()
                return True
            except Exception as e:
                print(f"FAILED: {model_name} - {str(e)[:100]}...")
                print()
        
        print("All model attempts failed")
        return False
        
    except Exception as e:
        print(f"Configuration Error: {e}")
        return False

def test_google_maps_api():
    """Test Google Maps API"""
    print("=" * 60)
    print("GOOGLE MAPS API TEST")
    print("=" * 60)
    
    import aiohttp
    import asyncio
    
    async def test_maps():
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "address": "New York, NY",
                    "key": settings.google_maps_api_key
                }
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'OK':
                            print(f"SUCCESS: Maps API working")
                            print(f"Found: {data['results'][0]['formatted_address']}")
                            return True
                        else:
                            print(f"FAILED: Maps API - {data.get('status')}")
                            print(f"Error: {data.get('error_message', 'Unknown error')}")
                            return False
                    else:
                        print(f"FAILED: Maps API - HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"FAILED: Maps API - {e}")
            return False
    
    return asyncio.run(test_maps())

if __name__ == "__main__":
    print("Testing Google APIs...")
    print()
    
    gemini_success = test_gemini_api()
    print()
    maps_success = test_google_maps_api()
    
    print("=" * 60)
    print("GOOGLE API TEST RESULTS")
    print("=" * 60)
    print(f"Gemini API: {'SUCCESS' if gemini_success else 'FAILED'}")
    print(f"Maps API: {'SUCCESS' if maps_success else 'FAILED'}")
    
    if gemini_success and maps_success:
        print("\nAll Google APIs are working!")
    elif gemini_success:
        print("\nGemini API working, Maps API needs attention")
    elif maps_success:
        print("\nMaps API working, Gemini API needs attention")
    else:
        print("\nBoth Google APIs need attention")
        print("\nTroubleshooting:")
        print("1. Check if API keys are valid")
        print("2. Verify billing is enabled")
        print("3. Check API quotas and limits")
        print("4. Ensure APIs are enabled in Google Cloud Console")
