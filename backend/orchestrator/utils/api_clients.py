import httpx
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.timeutils import utc_now_iso
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import settings

# Configure Groq client
_groq_client = Groq(api_key=settings.groq_api_key)
GROQ_MODEL = "llama-3.3-70b-versatile"

def llm_generate(prompt: str) -> str:
    """Send a prompt to Groq and return the text response."""
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""

# Keep backward compatibility alias
def get_gemini():
    """Deprecated: returns a wrapper that uses Groq instead of Gemini."""
    class _GroqWrapper:
        def generate_content(self, prompt, **kwargs):
            if isinstance(prompt, list):
                prompt = prompt[0].get("text", str(prompt[0])) if isinstance(prompt[0], dict) else str(prompt[0])
            class _Resp:
                def __init__(self, text):
                    self.text = text
            return _Resp(llm_generate(str(prompt)))
    return _GroqWrapper()

def http_client():
    return httpx.AsyncClient(timeout=settings.http_timeout_seconds)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def serp_search(query: str, num: int = 10) -> dict:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "api_key": settings.serp_api_key,
        "hl": "en",
        "gl": "us"
    }
    async with http_client() as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

# Fallback geocoding for common supply chain locations
_KNOWN_LOCATIONS = {
    "hsinchu": (24.8138, 120.9675), "taiwan": (25.0330, 121.5654),
    "los angeles": (34.0522, -118.2437), "new york": (40.7128, -74.0060),
    "houston": (29.7604, -95.3698), "san francisco": (37.7749, -122.4194),
    "london": (51.5074, -0.1278), "berlin": (52.5200, 13.4050),
    "paris": (48.8566, 2.3522), "rotterdam": (51.9244, 4.4777),
    "hamburg": (53.5511, 9.9937), "frankfurt": (50.1109, 8.6821),
    "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090), "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639),
    "shanghai": (31.2304, 121.4737), "beijing": (39.9042, 116.4074),
    "shenzhen": (22.5431, 114.0579), "guangzhou": (23.1291, 113.2644),
    "tokyo": (35.6762, 139.6503), "osaka": (34.6937, 135.5023),
    "busan": (35.1796, 129.0756), "seoul": (37.5665, 126.9780),
    "singapore": (1.3521, 103.8198), "dubai": (25.2048, 55.2708),
    "cairo": (30.0444, 31.2357), "lagos": (6.5244, 3.3792),
    "sao paulo": (-23.5505, -46.6333), "mexico city": (19.4326, -99.1332),
    "sydney": (-33.8688, 151.2093), "melbourne": (-37.8136, 144.9631),
    "bangkok": (13.7563, 100.5018), "jakarta": (-6.2088, 106.8456),
    "hong kong": (22.3193, 114.1694), "durban": (-29.8587, 31.0218),
    "usa": (39.8283, -98.5795), "china": (35.8617, 104.1954),
    "india": (20.5937, 78.9629), "germany": (51.1657, 10.4515),
    "japan": (36.2048, 138.2529), "uk": (55.3781, -3.4360),
    "brazil": (-14.2350, -51.9253), "australia": (-25.2744, 133.7751),
}

def _fallback_geocode(address: str) -> tuple[float, float] | None:
    """Try to match address against known supply chain locations."""
    addr_lower = address.lower().strip()
    for key, coords in _KNOWN_LOCATIONS.items():
        if key in addr_lower or addr_lower in key:
            return coords
    return None

# --- Mappls OAuth2 token cache ---
_mappls_token_cache = {"token": None, "expires_at": 0}

async def _get_mappls_token() -> str | None:
    """Get Mappls OAuth2 access token using client_id + client_secret."""
    import time
    # Return cached token if still valid
    if _mappls_token_cache["token"] and time.time() < _mappls_token_cache["expires_at"]:
        return _mappls_token_cache["token"]
    
    if not settings.mappls_client_id or not settings.mappls_client_secret:
        return None
    
    try:
        async with http_client() as client:
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
                data = r.json()
                token = data.get("access_token")
                expires_in = data.get("expires_in", 86400)
                _mappls_token_cache["token"] = token
                _mappls_token_cache["expires_at"] = time.time() + expires_in - 60
                return token
    except Exception:
        pass
    return None

async def geocode(address: str) -> tuple[float, float] | None:
    """Geocode an address using Mappls OAuth2 API → REST key → fallback."""
    # Strategy 1: OAuth2 token-based geocoding
    token = await _get_mappls_token()
    if token:
        try:
            async with http_client() as client:
                r = await client.get(
                    "https://atlas.mappls.com/api/places/geocode",
                    params={"address": address, "itemCount": 1},
                    headers={"Authorization": f"Bearer {token}"},
                )
                if r.status_code == 200:
                    data = r.json()
                    coords = _extract_coords(data)
                    if coords:
                        return coords
        except Exception:
            pass
    
    # Strategy 2: REST key URL pattern
    try:
        base = f"https://apis.mappls.com/advancedmaps/v1/{settings.mappls_api_key}/geocode"
        async with http_client() as client:
            r = await client.get(base, params={"address": address, "itemCount": 1})
            if r.status_code == 200:
                coords = _extract_coords(r.json())
                if coords:
                    return coords
    except Exception:
        pass
    
    # Strategy 3: Fallback to known locations
    return _fallback_geocode(address)

def _extract_coords(data) -> tuple[float, float] | None:
    """Extract lat/lng from various Mappls response formats."""
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
    elif isinstance(data, dict):
        results = data.get("copResults") or data.get("results") or []
        if isinstance(results, list) and results:
            item = results[0]
        elif isinstance(results, dict):
            item = results
        else:
            item = data
    else:
        return None
    lat = float(item.get("latitude") or item.get("lat", 0))
    lng = float(item.get("longitude") or item.get("lng", 0))
    if lat != 0 and lng != 0:
        return (lat, lng)
    return None

async def fetch_openweather(lat: float, lon: float) -> dict | None:
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon, "appid": settings.weather_api_key,
        "exclude": "minutely,hourly,alerts"
    }
    async with http_client() as client:
        r = await client.get(url, params=params)
        if r.status_code == 200:
            return r.json()
    return None

async def fetch_weatherapi(lat: float, lon: float) -> dict | None:
    # Map lat/lon to a query string as provider expects city names; here we pass coord
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": settings.weather_api_key, "q": f"{lat},{lon}", "days": 7, "aqi": "no", "alerts": "no"
    }
    async with http_client() as client:
        r = await client.get(url, params=params)
        if r.status_code == 200:
            return r.json()
    return None

async def fetch_weather(lat: float, lon: float) -> dict | None:
    if settings.weather_provider.lower() == "weatherapi":
        return await fetch_weatherapi(lat, lon)
    return await fetch_openweather(lat, lon)

async def gemini_structured(prompt: str) -> dict:
    """Ask LLM (Groq) to return strict JSON. If parsing fails, return {}."""
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only, no markdown or extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        import json
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        # Fallback: try without JSON mode
        try:
            text = llm_generate(prompt)
            import json, re
            text = re.sub(r"```json|```", "", text).strip()
            return json.loads(text)
        except Exception:
            return {}
