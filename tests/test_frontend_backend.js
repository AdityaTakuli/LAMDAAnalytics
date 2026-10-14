// Simple test to verify frontend-backend connection
// Run this in browser console on http://localhost:5174

async function testBackendConnection() {
    try {
        console.log('Testing backend connection...');
        
        // Test model info endpoint
        const modelResponse = await fetch('http://127.0.0.1:8002/model/info');
        const modelData = await modelResponse.json();
        console.log('Model Info:', modelData);
        
        // Test analytics endpoint
        const analyticsResponse = await fetch('http://127.0.0.1:8002/analytics/overview');
        const analyticsData = await analyticsResponse.json();
        console.log('Analytics Data:', analyticsData);
        
        // Verify expected properties exist
        if (modelData.model_name && modelData.f1_score && modelData.loaded_at) {
            console.log('✅ Model info has expected properties');
        } else {
            console.log('❌ Model info missing expected properties');
        }
        
        if (analyticsData.active_routes && analyticsData.active_alerts && analyticsData.reliability_score) {
            console.log('✅ Analytics data has expected properties');
        } else {
            console.log('❌ Analytics data missing expected properties');
        }
        
        console.log('✅ Backend connection test completed successfully!');
        
    } catch (error) {
        console.error('❌ Backend connection test failed:', error);
    }
}

// Run the test
testBackendConnection();
