#!/usr/bin/env python3
"""End-to-end test: POST to /analyze and print results."""
import httpx, json

r = httpx.post("http://127.0.0.1:8007/analyze", json={
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC"
}, timeout=120)

print("Status:", r.status_code)
data = r.json()

if r.status_code == 200:
    tgn = data.get("tgn_result", {})
    print("Risk Score:", tgn.get("risk_score"))
    print("Risk Label:", tgn.get("risk_label"))
    
    print("\nRisk Factors:")
    for f in data.get("concise", []):
        print("  -", f["name"], ":", f["level"], "(", f["percent"], "%)")
    
    print("\nFeatures:")
    for k, v in data.get("features", {}).items():
        print("  -", k, ":", v)
    
    print("\nMitigation Strategies:")
    comp = data.get("comprehensive", {})
    for k, v in comp.get("mitigation_strategies", {}).items():
        print("  -", k, ":", v)
    
    print("\nSUCCESS: Full pipeline works!")
else:
    print("Response:", json.dumps(data, indent=2)[:500])
