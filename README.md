# LAMDA Supply Chain Risk Analysis System

A comprehensive AI-powered supply chain risk analysis system featuring a complete 7-agent AI pipeline that combines PyTorch machine learning models with real-time external API integration and a modern React frontend.

## 🚀 Features

* **Complete AI Agent Pipeline**: 7 specialized AI agents for comprehensive risk analysis
* **Real-time External API Integration**: Gemini AI, SERP API, Weather API, Google Maps
* **TGN Model Integration**: PyTorch-based risk prediction with fallback mechanisms
* **Async Orchestration**: Parallel execution of all AI agents for optimal performance
* **Real-time Monitoring**: Live tracking of supply chain disruptions and alerts
* **Interactive Dashboard**: Modern React-based interface with real-time updates
* **Comprehensive Risk Analysis**: Multi-factor risk assessment with actionable insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  PyTorch TGN   │
│   (Port 5175)   │◄──►│   (Port 8007)   │◄──►│  (tgn_model.pth)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  AI Agent       │
                    │  Orchestrator   │
                    │  (7 Agents)     │
                    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ Gemini   │ │ SERP API │ │ Weather  │
            │ AI API   │ │          │ │ API      │
            └──────────┘ └──────────┘ └──────────┘
```

## 🤖 AI Agent Pipeline

The system features a complete 7-agent AI pipeline:

1. **🔍 Trade Agent** → Gemini AI trade flow analysis
2. **📰 News Agent** → SERP API → Web scraping → Gemini sentiment analysis
3. **🌤️ Weather Agent** → Weather API → Anomaly detection
4. **🏛️ Political Agent** → Gemini geopolitical risk assessment
5. **🌍 GSCPI Agent** → Gemini global supply chain pressure analysis
6. **⚖️ Normalizer Agent** → Feature normalization and scaling
7. **🤖 TGN Model** → Trained AI risk prediction
8. **📊 Reporter Agent** → Gemini report generation and simplification

## 📋 Prerequisites

* Python 3.13+
* Node.js 16+
* npm or yarn
* API Keys for external services:
  - Google Gemini API
  - Google Maps API
  - SERP API
  - Weather API (OpenWeather or WeatherAPI)

## 🛠️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/iareARiES/LAMDAAnalytics.git
cd LAMDAAnalytics
```

### 2. Backend Setup

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_MAPS_API_KEY=your_key
# SERP_API_KEY=your_key
# WEATHER_API_KEY=your_key
# GEMINI_API_KEY=your_key

# Start the backend server
python -m uvicorn main:app --host 127.0.0.1 --port 8007
```

The backend will be available at: `http://127.0.0.1:8007`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd project2

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at: `http://localhost:5175`

### 4. Quick Start (Windows)

```bash
# Start backend (in one terminal)
python start_backend.py

# Start frontend (in another terminal)
start_frontend.bat
```

## 🔧 API Endpoints

### Core Analysis

* `POST /analyze` - Complete AI agent pipeline analysis
* `GET /model/info` - Get TGN model information
* `GET /health` - Health check

### Monitoring & Analytics

* `GET /monitoring/alerts` - Get real-time alerts
* `GET /analytics/overview` - Get analytics overview

## 📊 Usage

1. **Open the Dashboard**: Navigate to `http://localhost:5175/dashboard`
2. **Configure Analysis**:  
   * Select component type (semiconductors, batteries, etc.)  
   * Enter seller location (e.g., "Hsinchu, Taiwan")  
   * Enter import location (e.g., "Los Angeles, USA")
   * Enter seller name (e.g., "TSMC")
3. **Run Analysis**: Click "Analyze Supply Chain Risk"
4. **View Results**:  
   * Risk score and level (0-1 scale)
   * Detailed risk factors with percentages
   * AI-generated mitigation strategies
   * Real-time monitoring data

## 🧠 AI Model Integration

The system integrates with your PyTorch TGN model (`tgn_model.pth`) which provides:

* **Risk Prediction**: Neural network-based risk scoring
* **Feature Processing**: 7 normalized input features
* **Component Analysis**: Individual risk factor contributions
* **Fallback Mechanism**: Weighted scoring when model unavailable

### Input Features (Normalized 0-1)

* `inventory_days`: Inventory coverage analysis
* `past_delay_days`: Historical delay patterns
* `news_vol_7d`: News volume in last 7 days
* `neg_tone_frac_3d`: Negative sentiment fraction
* `strike_flag_7d`: Labor unrest detection
* `weather_anomaly_7d`: Weather anomaly detection
* `global_risk`: Global supply chain pressure

## 🔍 Risk Analysis Features

* **Real-time News Monitoring**: SERP API integration for disruption detection
* **Weather Risk Assessment**: Climate-based anomaly detection
* **Political Risk Evaluation**: Geopolitical and sanctions monitoring
* **Trade Flow Analysis**: Gemini AI-powered trade pattern analysis
* **Global Pressure Monitoring**: GSCPI integration for macro trends

## 📈 Analytics Dashboard

* **Real-time Risk Scoring**: Live risk assessment with confidence levels
* **Interactive Route Visualization**: Supply chain map with risk overlays
* **Alert Management**: Real-time disruption notifications
* **Performance Metrics**: Model accuracy and execution times
* **Mitigation Strategies**: AI-generated actionable recommendations

## 🚀 Performance

* **Execution Time**: ~26 seconds for complete analysis
* **Parallel Processing**: All 7 agents execute concurrently
* **Caching**: 15-minute TTL for news and weather data
* **Error Handling**: Graceful degradation with mock responses
* **Scalability**: Designed for high-throughput production use

## 🧪 Testing

### Run Complete Pipeline Test

```bash
# Test the full AI agent pipeline
python run_orchestrator_realtime.py

# Test individual API connections
python test_google_api.py

# Test with mock data
python test_simple_orchestrator.py
```

### API Testing

```bash
# Test the complete analysis endpoint
curl -X POST http://127.0.0.1:8007/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC"
  }'
```

## 🚨 Troubleshooting

### Backend Issues

* Ensure Python dependencies are installed: `pip install -r backend/requirements.txt`
* Check if model file exists: `tgn_model.pth`
* Verify API keys are set in `backend/.env`
* Check if port 8007 is available

### Frontend Issues

* Ensure Node.js dependencies are installed: `npm install`
* Check if port 5175 is available
* Verify backend is running on port 8007

### API Issues

* Verify all API keys are valid and have proper permissions
* Check API quotas and billing status
* Ensure internet connectivity for external API calls

### Model Issues

* Model file should be in the backend directory
* Ensure PyTorch is properly installed
* Check model file integrity

## 📁 Project Structure

```
LAMDAAnalytics/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # API keys configuration
│   ├── tgn_model.pth          # PyTorch TGN model
│   ├── config/
│   │   └── settings.py        # Configuration management
│   ├── orchestrator/
│   │   ├── orchestrator.py    # Main orchestrator
│   │   ├── agents/            # AI agent implementations
│   │   └── utils/             # Utility functions
│   └── models/
│       └── tgn_model.py       # TGN model wrapper
├── project2/
│   ├── src/
│   │   ├── Dashboard.jsx      # Main dashboard
│   │   ├── SupplyChainMap.jsx # Route visualization
│   │   └── services/
│   │       └── api.js         # API communication
│   └── package.json           # Frontend dependencies
├── run_orchestrator_realtime.py # Pipeline test script
├── test_google_api.py         # API testing script
└── README.md                  # This file
```

## 🔮 Future Enhancements

* **Real-time Data Feeds**: Integration with live supply chain data
* **Advanced Visualization**: 3D route mapping and risk heatmaps
* **Machine Learning**: Model retraining and improvement
* **Multi-language Support**: Internationalization
* **Mobile App**: React Native mobile application
* **Blockchain Integration**: Supply chain transparency
* **IoT Integration**: Real-time sensor data

## 📝 License

This project is part of the LAMDA Analytics system for supply chain risk management.

## 👥 Team

* **Devansh Behl**: Full Stack Development
* **Mayan Sharma**: AI/ML Engineering  
* **Aditya Takuli**: Data Engineering & Analytics
* **Lay Gupta**: Product & Business Model

## 📞 Support

For support or questions:
* API Documentation: `http://127.0.0.1:8007/docs` (when backend is running)
* GitHub Issues: [Create an issue](https://github.com/iareARiES/LAMDAAnalytics/issues)
* Email: mayan25sharma@gmail.com, devanshbhel@gmail.com

---

**Status**: ✅ Fully Operational with Real API Integration  
**Last Updated**: September 2025  
**Version**: 2.0 - Complete AI Agent Pipeline
