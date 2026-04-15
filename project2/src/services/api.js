// API service for communicating with FastAPI backend
const API_BASE_URL = 'http://127.0.0.1:8000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Get model information
  async getModelInfo() {
    return this.request('/model/info');
  }

  // Analyze supply chain risk - updated for new orchestrator format
  async analyzeSupplyChain(data) {
    return this.request('/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get real-time alerts
  async getAlerts() {
    return this.request('/monitoring/alerts');
  }

  // Get analytics overview
  async getAnalyticsOverview() {
    return this.request('/analytics/overview');
  }
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService;
