# API Documentation

Backend analysis is executed by a **LangGraph** multi-agent `StateGraph` (see `ORCHESTRATOR_README.md`). The HTTP contract below is unchanged.

## Base URL
```
http://127.0.0.1:8007
```

## Authentication
Currently, the API does not require authentication. In production, implement proper API key authentication or OAuth2.

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-28T04:29:19.555688Z",
  "version": "3.0.0"
}
```

### 2. Model Information

**GET** `/model/info`

Get information about the loaded TGN model.

**Response:**
```json
{
  "model_name": "TGN",
  "model_class": "FixedTGN",
  "f1_score": 0.45392434594234293,
  "loaded_at": "2025-09-28T07:34:29.408680"
}
```

### 3. Analytics Overview

**GET** `/analytics/overview`

Get analytics overview data for the dashboard.

**Response:**
```json
{
  "total_suppliers": 10000,
  "active_routes": 24,
  "active_alerts": 7,
  "reliability_score": 89.2,
  "risk_trend": "decreasing",
  "last_updated": "2024-01-01T00:00:00Z"
}
```

### 4. Supply Chain Risk Analysis

**POST** `/analyze`

Run the complete LangGraph multi-agent network analysis (geocode → parallel feature agents → normalize → TGN → report).

**Request Body:**
```json
{
  "component_type": "Semiconductor",
  "seller_location": "Hsinchu, Taiwan",
  "import_location": "Los Angeles, USA",
  "seller_name": "TSMC",
  "additional_factors": {
    "priority": "high",
    "volume": "large",
    "criticality": "essential"
  }
}
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `component_type` | string | Yes | Type of component (e.g., "Semiconductor", "Battery", "Pharmaceutical") |
| `seller_location` | string | Yes | Location of the seller/supplier |
| `import_location` | string | Yes | Destination location |
| `seller_name` | string | Yes | Name of the seller/supplier |
| `additional_factors` | object | No | Additional risk factors |

**Response:**
```json
{
  "request_id": "f2a72415-2f9c-4c0a-9789-7384b66996ed",
  "created_at": "2025-09-28T04:29:19.555688Z",
  "inputs": {
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC",
    "additional_factors": {
      "priority": "high",
      "volume": "large",
      "criticality": "essential"
    }
  },
  "features": {
    "inventory_days": 0.5,
    "past_delay_days": 0.33,
    "news_vol_7d": 0.5,
    "neg_tone_frac_3d": 0.5,
    "strike_flag_7d": 0.5,
    "weather_anomaly_7d": 0.5,
    "global_risk": 0.33
  },
  "tgn_result": {
    "risk_score": 0.458,
    "risk_label": "Medium",
    "risk_components": {
      "inventory_days": 0.1,
      "past_delay_days": 0.066,
      "news_vol_7d": 0.075,
      "neg_tone_frac_3d": 0.075,
      "strike_flag_7d": 0.075,
      "weather_anomaly_7d": 0.05,
      "global_risk": 0.017
    }
  },
  "concise": [
    {
      "name": "inventory_days",
      "level": "Low",
      "percent": 10.0,
      "impact": "Inventory coverage may be tightening"
    },
    {
      "name": "news_vol_7d",
      "level": "Low",
      "percent": 7.5,
      "impact": "High volume of disruption mentions"
    },
    {
      "name": "neg_tone_frac_3d",
      "level": "Low",
      "percent": 7.5,
      "impact": "Negative news sentiment trending"
    },
    {
      "name": "strike_flag_7d",
      "level": "Low",
      "percent": 7.5,
      "impact": "Labor unrest may affect ports/logistics"
    },
    {
      "name": "past_delay_days",
      "level": "Low",
      "percent": 6.6,
      "impact": "Historical delays suggest risk carryover"
    },
    {
      "name": "weather_anomaly_7d",
      "level": "Low",
      "percent": 5.0,
      "impact": "Recent weather anomaly near route/plant"
    },
    {
      "name": "global_risk",
      "level": "Low",
      "percent": 1.6,
      "impact": "Global chain pressure elevated"
    }
  ],
  "comprehensive": {
    "risk_distribution": [
      {
        "name": "inventory_days",
        "level": "Low",
        "percent": 10.0,
        "impact": "Inventory coverage may be tightening"
      }
    ],
    "mitigation_strategies": {
      "Weather Risk Mitigation": "Plan alternate routes and maintain buffer inventory during severe weather.",
      "Labor Strike Contingency": "Pre-negotiate slots with alternative ports and carriers.",
      "Sanctions Compliance": "Continuously monitor regulations; pre-qualify alternate suppliers."
    }
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique identifier for the analysis request |
| `created_at` | string | ISO timestamp of when the analysis was created |
| `inputs` | object | Original input parameters |
| `features` | object | Normalized features used by the TGN model |
| `tgn_result` | object | TGN model prediction results |
| `concise` | array | Simplified risk factor analysis |
| `comprehensive` | object | Detailed analysis and mitigation strategies |

### 5. Monitoring Alerts

**GET** `/monitoring/alerts`

Get real-time monitoring alerts.

**Response:**
```json
[
  {
    "id": "alert_001",
    "severity": "high",
    "message": "Supply chain disruption detected in Taiwan",
    "timestamp": "2025-09-28T04:29:19.555688Z",
    "route": "Hsinchu, Taiwan → Los Angeles, USA",
    "component": "Semiconductor"
  }
]
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "component_type"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred during analysis"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement rate limiting based on:
- API key
- IP address
- User account

Recommended limits:
- 100 requests per minute per API key
- 1000 requests per hour per API key

## CORS

The API supports CORS for the following origins:
- `http://localhost:5173`
- `http://localhost:5174`
- `http://localhost:5175`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:5174`
- `http://127.0.0.1:5175`

## WebSocket Support

Currently not implemented. Future versions will include WebSocket support for real-time updates.

## SDK Examples

### Python

```python
import requests
import json

# Analyze supply chain risk
def analyze_supply_chain(component_type, seller_location, import_location, seller_name):
    url = "http://127.0.0.1:8007/analyze"
    payload = {
        "component_type": component_type,
        "seller_location": seller_location,
        "import_location": import_location,
        "seller_name": seller_name,
        "additional_factors": {
            "priority": "high",
            "volume": "large"
        }
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = analyze_supply_chain(
    "Semiconductor",
    "Hsinchu, Taiwan",
    "Los Angeles, USA",
    "TSMC"
)

print(f"Risk Score: {result['tgn_result']['risk_score']}")
print(f"Risk Label: {result['tgn_result']['risk_label']}")
```

### JavaScript

```javascript
// Analyze supply chain risk
async function analyzeSupplyChain(componentType, sellerLocation, importLocation, sellerName) {
    const url = 'http://127.0.0.1:8007/analyze';
    const payload = {
        component_type: componentType,
        seller_location: sellerLocation,
        import_location: importLocation,
        seller_name: sellerName,
        additional_factors: {
            priority: 'high',
            volume: 'large'
        }
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}

// Example usage
analyzeSupplyChain('Semiconductor', 'Hsinchu, Taiwan', 'Los Angeles, USA', 'TSMC')
    .then(result => {
        console.log(`Risk Score: ${result.tgn_result.risk_score}`);
        console.log(`Risk Label: ${result.tgn_result.risk_label}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

### cURL

```bash
# Analyze supply chain risk
curl -X POST "http://127.0.0.1:8007/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC",
    "additional_factors": {
      "priority": "high",
      "volume": "large"
    }
  }'

# Get model info
curl -X GET "http://127.0.0.1:8007/model/info"

# Get analytics overview
curl -X GET "http://127.0.0.1:8007/analytics/overview"

# Health check
curl -X GET "http://127.0.0.1:8007/health"
```

## Testing

### Postman Collection

Import the following collection for testing:

```json
{
  "info": {
    "name": "LAMDA Analytics API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "http://127.0.0.1:8007/health",
          "protocol": "http",
          "host": ["127", "0", "0", "1"],
          "port": "8007",
          "path": ["health"]
        }
      }
    },
    {
      "name": "Analyze Supply Chain",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"component_type\": \"Semiconductor\",\n  \"seller_location\": \"Hsinchu, Taiwan\",\n  \"import_location\": \"Los Angeles, USA\",\n  \"seller_name\": \"TSMC\",\n  \"additional_factors\": {\n    \"priority\": \"high\",\n    \"volume\": \"large\"\n  }\n}"
        },
        "url": {
          "raw": "http://127.0.0.1:8007/analyze",
          "protocol": "http",
          "host": ["127", "0", "0", "1"],
          "port": "8007",
          "path": ["analyze"]
        }
      }
    }
  ]
}
```

## Performance

### Response Times

| Endpoint | Average Response Time | 95th Percentile |
|----------|----------------------|-----------------|
| `/health` | 50ms | 100ms |
| `/model/info` | 10ms | 20ms |
| `/analytics/overview` | 20ms | 50ms |
| `/analyze` | 26s | 60s |
| `/monitoring/alerts` | 100ms | 200ms |

### Throughput

- **Concurrent Requests**: Up to 10 simultaneous `/analyze` requests
- **Rate Limit**: 100 requests per minute (recommended)
- **Queue**: Requests are queued if capacity is exceeded

## Monitoring

### Metrics

The API exposes Prometheus metrics at `/metrics`:

- `requests_total`: Total number of requests
- `request_duration_seconds`: Request duration histogram
- `active_connections`: Number of active connections
- `error_rate`: Error rate percentage

### Health Checks

Use the `/health` endpoint for:
- Load balancer health checks
- Kubernetes liveness/readiness probes
- Monitoring system checks

## Security

### API Key Authentication (Future)

```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
```

### Rate Limiting (Future)

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request, ...):
    # Implementation
    pass
```

---

This API documentation provides comprehensive information about all available endpoints, request/response formats, and usage examples. For interactive API exploration, visit `http://127.0.0.1:8007/docs` when the backend is running.
