import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Dashboard from './Dashboard';

// Components
const FloatingParticles = () => (
  <div className="floating-particles">
    {[...Array(12)].map((_, i) => (
      <div key={i} className="particle"></div>
    ))}
  </div>
);

const Navbar = () => (
  <nav className="navbar">
    <div className="nav-container">
      <div className="logo">
        LAMDA <span>Analytics</span>
      </div>
      <div className="nav-links">
        <a href="#architecture">Architecture</a>
        <a href="#agents">Agents</a>
        <a href="#features">Features</a>
        <a href="#team">Team</a>
      </div>
      <Link to="/dashboard" className="cta-button">Launch Dashboard</Link>
    </div>
  </nav>
);

const HeroSection = ({ stats }) => (
  <section className="hero-section">
    <div className="container">
      <div className="hero-content">
        <h1 className="hero-title">
          <span className="gradient-text">LAMDA</span><br />
          Supply Chain Risk Analysis System
        </h1>
        <p className="hero-subtitle">
          A comprehensive, production-ready system that combines multi-agent data collection, 
          real-time risk analysis, and intelligent monitoring for supply chain risk 
          prediction. Built with 8 specialized agents and advanced analytics capabilities.
        </p>
        
        
          
      </div>
    </div>
  </section>
);

const ArchitectureSection = () => (
  <section id="architecture" className="section">
    <div className="container">
      <h2 className="section-title">System Architecture</h2>
      <p className="section-subtitle">
        Production-ready multi-agent architecture with FastAPI backend, React frontend, 
        MongoDB database, and Prefect orchestration for scalable supply chain analytics.
      </p>
      
      <div className="architecture-flow">
        <div className="flow-stage">
          <div className="flow-card">
            <div className="flow-icon">Data</div>
            <h3>Data Sources</h3>
            <p>RSS Feeds, SERP API, Social Media, Weather Data, Geographic Intelligence</p>
          </div>
        </div>
        
        <div className="flow-arrow">→</div>
        
        <div className="flow-stage">
          <div className="flow-card">
            <div className="flow-icon">AI</div>
            <h3>Multi-Agent Processing</h3>
            <p>8 Specialized agents for data collection, analysis, and feature engineering</p>
          </div>
        </div>
        
        <div className="flow-arrow">→</div>
        
        <div className="flow-stage">
          <div className="flow-card">
            <div className="flow-icon">Brain</div>
            <h3>Risk Analysis</h3>
            <p>Advanced analytics, predictive modeling with comprehensive risk assessment</p>
          </div>
        </div>
        
        <div className="flow-arrow">→</div>
        
        <div className="flow-stage">
          <div className="flow-card">
            <div className="flow-icon">Web</div>
            <h3>Web Interface</h3>
            <p>FastAPI backend with comprehensive API endpoints and data visualization</p>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const AgentGrid = () => {
  const agents = [
    {
      id: "Agent 0",
      title: "Registry Normalizer",
      description: "Builds canonical supplier registry with Google Maps geocoding and location intelligence",
      icon: "Target",
      status: "active"
    },
    {
      id: "Agent 1", 
      title: "Social/X Fetcher",
      description: "Fetches X/Twitter posts for suppliers and news outlets with sentiment analysis",
      icon: "Social",
      status: "processing"
    },
    {
      id: "Agent 2",
      title: "News Fetcher", 
      description: "Collects news from RSS feeds and SERP API for comprehensive media monitoring",
      icon: "News",
      status: "active"
    },
    {
      id: "Agent 3",
      title: "Deep Crawl Extraction",
      description: "Deep crawls URLs with crawl4ai and extracts structured events from content",
      icon: "Crawl",
      status: "standby"
    },
    {
      id: "Agent 4",
      title: "Weather Anomalies",
      description: "Detects weather anomalies for each location using meteorological data",
      icon: "Weather",
      status: "active"
    },
    {
      id: "Agent 5",
      title: "Feature Builder",
      description: "Builds advanced features from all data sources with comprehensive analytics",
      icon: "Build",
      status: "processing"
    },
    {
      id: "Agent 6",
      title: "Features Builder",
      description: "Additional feature engineering and data validation for analysis pipeline",
      icon: "Engineer",
      status: "active"
    },
    {
      id: "Agent 7",
      title: "Export CSV",
      description: "Validates and exports final CSV for analysis pipeline with structured schema",
      icon: "Export",
      status: "ready"
    }
  ];

  return (
    <section id="agents" className="section">
      <div className="container">
        <h2 className="section-title">8 Specialized AI Agents</h2>
        <p className="section-subtitle">
          Each agent handles specific aspects of data collection, processing, and analysis 
          in our coordinated multi-agent pipeline architecture.
        </p>
        
        <div className="agents-grid">
          {agents.map((agent, index) => (
            <div key={agent.id} className="agent-card" style={{animationDelay: `${index * 0.1}s`}}>
              <div className="agent-status">
                <span className={`status-indicator ${agent.status}`}></span>
                <span className="status-text">{agent.status}</span>
              </div>
              <div className="agent-icon">{agent.icon}</div>
              <h3 className="agent-title">{agent.title}</h3>
              <p className="agent-description">{agent.description}</p>
              <div className="agent-id">{agent.id}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const TechStack = () => {
  const technologies = [
    { category: "Backend", items: ["FastAPI", "Python", "MongoDB", "Prefect"] },
    { category: "Frontend", items: ["React", "JavaScript", "CSS3", "WebSocket"] },
    { category: "AI/ML", items: ["crawl4ai", "DistilBERT", "Pandas", "Scikit-learn"] },
    { category: "APIs", items: ["Google Maps", "SERP API", "Weather API", "Twitter API"] }
  ];

  return (
    <section id="features" className="section">
      <div className="container">
        <h2 className="section-title">Production Technology Stack</h2>
        <p className="section-subtitle">
          Enterprise-grade technologies ensuring scalability, reliability, and performance 
          for mission-critical supply chain operations.
        </p>
        
        <div className="tech-grid">
          {technologies.map((tech, index) => (
            <div key={tech.category} className="tech-category" style={{animationDelay: `${index * 0.1}s`}}>
              <h3 className="tech-category-title">{tech.category}</h3>
              <div className="tech-items">
                {tech.items.map((item, itemIndex) => (
                  <span key={item} className="tech-item" style={{animationDelay: `${(index * 0.1) + (itemIndex * 0.05)}s`}}>
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const KeyFeatures = () => (
  <section className="section">
    <div className="container">
      <h2 className="section-title">Revolutionary Features</h2>
      <p className="section-subtitle">
        Industry-first capabilities that redefine supply chain risk management 
        through advanced AI technology and multi-agent intelligence.
      </p>
      
      <div className="motivation-grid">
        <div className="motivation-card">
          <span className="motivation-icon">Rocket</span>
          <h3>Real-Time Risk Detection</h3>
          <p>
            Advanced risk identification with rapid response capabilities, 
            delivering alerts faster than traditional enterprise systems.
          </p>
        </div>
        
        <div className="motivation-card">
          <span className="motivation-icon">Brain</span>
          <h3>Multi-Agent Intelligence</h3>
          <p>
            8 specialized AI agents working in coordination, each optimized for 
            specific supply chain intelligence tasks and data sources.
          </p>
        </div>
        
        <div className="motivation-card">
          <span className="motivation-icon">Global</span>
          <h3>Global Network Coverage</h3>
          <p>
            Comprehensive monitoring of suppliers across all continents 
            with seamless data integration and analysis capabilities.
          </p>
        </div>
        
        <div className="motivation-card">
          <span className="motivation-icon">Cost</span>
          <h3>Cost-Effective Solution</h3>
          <p>
            Significant cost reduction compared to traditional enterprise solutions 
            while delivering superior performance and comprehensive capabilities.
          </p>
        </div>
        
        <div className="motivation-card">
          <span className="motivation-icon">Security</span>
          <h3>Enterprise Security</h3>
          <p>
            Robust security architecture ensuring data protection 
            with comprehensive encryption and compliance standards.
          </p>
        </div>
        
        <div className="motivation-card">
          <span className="motivation-icon">Analytics</span>
          <h3>Predictive Analytics</h3>
          <p>
            Advanced analytical models with high accuracy in predicting supply chain 
            disruptions and providing actionable insights.
          </p>
        </div>
      </div>
    </div>
  </section>
);

const TeamSection = () => {
  const team = [
    {
      name: "Devansh Behl",
      role: "Full Stack",
      university: "Computer Science",
      expertise: "React • FastAPI • MongoDB • Multi-Agent Systems",
      initial: "DB"
    },
    {
      name: "Mayan Sharma", 
      role: "AI ML",
      university: "Data Science",
      expertise: "Python • TensorFlow • NLP • Feature Engineering",
      initial: "MS"
    },
    {
      name: "Aditya Takuli",
      role: "Data Engineer & Analytics",
      university: "Software Engineering", 
      expertise: "ETL • MongoDB • APIs • Real-time Processing",
      initial: "AT"
    },
    {
      name: "Lay Gupta",
      role: "Product and Business Model",
      university: "Business Technology",
      expertise: "Product Strategy • Market Analysis • Business Development",
      initial: "LG"
    }
  ];

  return (
    <section id="team" className="section">
      <div className="container">
        <h2 className="section-title">Team LAMDA</h2>
        <p className="section-subtitle">
          Passionate innovators combining technical expertise with business acumen 
          to build production-ready supply chain intelligence systems.
        </p>
        
        <div className="team-grid">
          {team.map((member, index) => (
            <div key={member.name} className="team-card" style={{animationDelay: `${index * 0.1}s`}}>
              <div className="team-photo">{member.initial}</div>
              <h3 className="team-name">{member.name}</h3>
              <p className="team-role">{member.role}</p>
              <p className="team-university">{member.university}</p>
              <p className="team-expertise">{member.expertise}</p>
              <div className="team-glow"></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const ProjectOverview = () => (
  <section className="section">
    <div className="container">
      <h2 className="section-title">Project Overview</h2>
      <p className="section-subtitle">
        LAMDA transforms supply chain risk management through innovative multi-agent architecture 
        and advanced analytics for comprehensive business intelligence.
      </p>
      
      <div className="project-card">
        <div className="project-description">
          <h3>Supply Chain Intelligence Revolution</h3>
          <p>
            LAMDA represents a breakthrough in supply chain risk management, combining 
            8 specialized AI agents with production-ready infrastructure. Our system 
            delivers real-time risk analysis, comprehensive supplier monitoring, and 
            predictive analytics to prevent disruptions and optimize operations.
          </p>
        </div>
        
        <div className="project-details">
          <div className="detail-group">
            <div className="detail-label">Architecture</div>
            <div className="detail-value">8 AI Agents + FastAPI</div>
          </div>
          <div className="detail-group">
            <div className="detail-label">Data Sources</div>
            <div className="detail-value">Multi-Source Intelligence</div>
          </div>
          <div className="detail-group">
            <div className="detail-label">Analytics</div>
            <div className="detail-value">Real-Time Processing</div>
          </div>
          <div className="detail-group">
            <div className="detail-label">Scalability</div>
            <div className="detail-value">Enterprise-Ready</div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const Footer = () => (
  <footer className="footer">
    <div className="container">
      <div className="footer-content">
        <div className="footer-logo">
          LAMDA <span>Analytics</span>
        </div>
        <p className="footer-text">
          Production-Ready Supply Chain Risk Analysis System
        </p>
        <div className="footer-links">
          <a href="#github">GitHub Repository</a>
          <a href="#docs">Documentation</a>
          <a href="#api">API Reference</a>
          <a href="#demo">Live Demo</a>
          <a href="#contact">Contact</a>
        </div>
      </div>
      <div className="footer-bottom">
        <p className="copyright">
          © 2025 Team LAMDA. Built with ❤️ for supply chain risk management.
        </p>
      </div>
    </div>
  </footer>
);

// Landing Page Component
const LandingPage = ({ stats }) => (
  <div className="app">
    <FloatingParticles />
    <Navbar />
    <HeroSection stats={stats} />
    <ArchitectureSection />
    <AgentGrid />
    <TechStack />
    <KeyFeatures />
    <ProjectOverview />
    <TeamSection />
    <Footer />
  </div>
);

// Main App Component
function App() {
  const [stats, setStats] = useState({
    suppliers: 0,
    agents: 8,
    accuracy: 0,
    countries: 0
  });

  useEffect(() => {
    const animateValue = (start, end, duration, callback) => {
      const startTime = performance.now();
      const animate = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const value = start + (end - start) * progress;
        callback(value);
        if (progress < 1) requestAnimationFrame(animate);
      };
      requestAnimationFrame(animate);
    };

    setTimeout(() => {
      animateValue(0, 10000, 2000, (val) => setStats(prev => ({ ...prev, suppliers: Math.floor(val) })));
      animateValue(0, 94.7, 2200, (val) => setStats(prev => ({ ...prev, accuracy: val.toFixed(1) })));
      animateValue(0, 50, 1800, (val) => setStats(prev => ({ ...prev, countries: Math.floor(val) })));
    }, 1000);
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage stats={stats} />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
