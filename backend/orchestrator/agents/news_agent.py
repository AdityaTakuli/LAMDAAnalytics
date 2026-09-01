from bs4 import BeautifulSoup
import os
from config.settings import settings
from ..utils.api_clients import serp_search, http_client, get_gemini, secret_configured
from ..utils.schema import NewsFeatures
from ..utils.cache import get_cached, set_cached, news_cache

SEARCH_TEMPLATES = [
    "{component} {seller} {loc} strike OR protest OR stoppage",
    "{component} {seller} {loc} delay OR disruption OR outage",
    "{component} {seller} {loc} supply chain bottleneck",
    "{component} {loc} semiconductor fab shutdown OR maintenance",
    "{seller} {loc} port congestion OR customs backlog"
]

async def _fetch_pages(urls: list[str]) -> list[str]:
    htmls = []
    async with http_client() as client:
        for u in urls[:8]:
            try:
                r = await client.get(u, follow_redirects=True)
                if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                    htmls.append(r.text)
            except Exception:
                continue
    return htmls

async def analyze_news(component_type: str, seller_loc: str, import_loc: str, seller_name: str | None) -> NewsFeatures:
    if os.environ.get("LAMDA_FORCE_NEWS_FAIL") == "1":
        return NewsFeatures(
            news_vol_7d=0,
            neg_tone_frac_3d=0.0,
            strike_flag_7d=0,
            sources=["LAMDA_FORCE_NEWS_FAIL"],
        )

    key = f"{component_type}|{seller_loc}|{import_loc}|{seller_name or ''}"
    cached = get_cached(news_cache, key)
    if cached:
        return cached

    queries = []
    for t in SEARCH_TEMPLATES:
        queries.append(t.format(component=component_type, seller=seller_name or "", loc=seller_loc))
        queries.append(t.format(component=component_type, seller=seller_name or "", loc=import_loc))

    urls = []
    if secret_configured(settings.serp_api_key):
        for q in queries:
            try:
                res = await serp_search(q, num=6)
                for item in res.get("organic_results", []):
                    link = item.get("link")
                    if link and link not in urls:
                        urls.append(link)
            except Exception:
                continue

    htmls = await _fetch_pages(urls)
    texts = []
    for h in htmls:
        try:
            soup = BeautifulSoup(h, "lxml")
            txt = soup.get_text(separator=" ", strip=True)
            texts.append(txt[:5000])  # cap per page
        except Exception:
            continue

    # Ask Gemini to compute our three features from texts
    model = get_gemini()
    prompt = f"""
You are analyzing news snippets about supply-chain disruptions. From the provided texts, compute:
- news_vol_7d: count of distinct relevant disruption mentions in last 7 days
- neg_tone_frac_3d: fraction [0..1] of negative-toned mentions in last 3 days
- strike_flag_7d: 1 if any strike/unrest detected in last 7 days else 0
Return JSON with fields exactly: {{"news_vol_7d": int, "neg_tone_frac_3d": float, "strike_flag_7d": int}}
Texts:
{texts[:10]}
"""
    try:
        resp = model.generate_content(prompt)
        import json, re
        s = (resp.text or "{}")
        s = re.sub(r"```json|```", "", s).strip()
        data = json.loads(s)
        feats = NewsFeatures(
            news_vol_7d=int(data.get("news_vol_7d", 0)),
            neg_tone_frac_3d=float(data.get("neg_tone_frac_3d", 0.0)),
            strike_flag_7d=int(data.get("strike_flag_7d", 0)),
            sources=urls[:10]
        )
    except Exception:
        feats = NewsFeatures(news_vol_7d=0, neg_tone_frac_3d=0.0, strike_flag_7d=0, sources=urls[:10])

    set_cached(news_cache, key, feats)
    return feats
