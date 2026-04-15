#!/usr/bin/env python3
"""Quick test of all API keys — Weather, SERP, Mappls (OAuth2), Groq"""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from config.settings import settings
import httpx

async def main():
    results = {}

    # Test 1: Weather API
    print("=" * 50)
    print("TEST 1: Weather API (OpenWeather)")
    print("=" * 50)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={"q": "London", "appid": settings.weather_api_key}
            )
            if r.status_code == 200:
                d = r.json()
                print("SUCCESS:", d["name"], "-", d["weather"][0]["description"])
                results["Weather"] = True
            else:
                print("FAILED: HTTP", r.status_code, "-", r.text[:200])
                results["Weather"] = False
    except Exception as e:
        print("FAILED:", e)
        results["Weather"] = False

    print()

    # Test 2: SERP API
    print("=" * 50)
    print("TEST 2: SERP API")
    print("=" * 50)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "q": "supply chain", "num": 1, "api_key": settings.serp_api_key}
            )
            if r.status_code == 200:
                d = r.json()
                cnt = len(d.get("organic_results", []))
                print("SUCCESS: Found", cnt, "results")
                results["SERP"] = True
            else:
                print("FAILED: HTTP", r.status_code, "-", r.text[:200])
                results["SERP"] = False
    except Exception as e:
        print("FAILED:", e)
        results["SERP"] = False

    print()

    # Test 3: Mappls Geocoding (OAuth2)
    print("=" * 50)
    print("TEST 3: Mappls Geocoding API (OAuth2)")
    print("=" * 50)
    mappls_ok = False
    # Step 1: Get OAuth token
    if settings.mappls_client_id and settings.mappls_client_secret:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    "https://outpost.mappls.com/api/security/oauth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": settings.mappls_client_id,
                        "client_secret": settings.mappls_client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if r.status_code == 200:
                    token = r.json().get("access_token")
                    print("OAuth token obtained:", token[:15] + "...")
                    # Step 2: Geocode
                    r2 = await client.get(
                        "https://atlas.mappls.com/api/places/geocode",
                        params={"address": "New Delhi"},
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if r2.status_code == 200:
                        d = r2.json()
                        cop = d.get("copResults", {})
                        print("SUCCESS:", cop.get("city", ""), cop.get("state", ""))
                        lat = cop.get("latitude", "?")
                        lng = cop.get("longitude", "?")
                        print(f"  Coordinates: lat={lat}, lng={lng}")
                        mappls_ok = True
                    else:
                        print("Geocode FAILED: HTTP", r2.status_code, r2.text[:200])
                else:
                    print("OAuth FAILED: HTTP", r.status_code, r.text[:200])
        except Exception as e:
            print("FAILED:", e)
    else:
        print("SKIPPED: MAPPLS_CLIENT_ID or MAPPLS_CLIENT_SECRET not set")

    # Fallback geocoder test
    from backend.orchestrator.utils.api_clients import _fallback_geocode
    fb = _fallback_geocode("Hsinchu, Taiwan")
    print(f"Fallback geocode 'Hsinchu, Taiwan' -> {fb}")
    results["Mappls"] = mappls_ok

    print()

    # Test 4: Groq API
    print("=" * 50)
    print("TEST 4: Groq API (LLM)")
    print("=" * 50)
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with exactly: API_OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        print("SUCCESS:", text)
        results["Groq"] = True
    except Exception as e:
        err = str(e)
        if "429" in err:
            print("RATE LIMITED: Key valid but quota exceeded. Wait and retry.")
            results["Groq"] = "rate_limited"
        elif "403" in err:
            print("ACCESS DENIED (403): Groq may be blocked in your region.")
            print("  Try using a VPN or check https://console.groq.com")
            results["Groq"] = False
        else:
            print("FAILED:", err[:200])
            results["Groq"] = False

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        if ok is True:
            icon = "OK"
        elif ok == "rate_limited":
            icon = "RATE LIMITED (key valid)"
        else:
            icon = "FAILED"
        print(f"  {name:12}: {icon}")
    
    working = sum(1 for v in results.values() if v is True or v == "rate_limited")
    print(f"\n  Working: {working}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
