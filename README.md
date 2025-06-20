# STAYIN ALIVE - Disaster Preparedness Platform Backend

🎯 **Afet Hazırlık ve Senaryo Platformu** - Gelişmiş AI destekli backend sistemi

## 🏗️ Architecture Overview

**STAYIN ALIVE** is a comprehensive disaster preparedness platform with advanced AI capabilities, built on a robust FastAPI + PostgreSQL + Redis stack.

### 🔧 Tech Stack
- **Backend**: FastAPI + Python 3.11
- **Database**: PostgreSQL + Redis
- **AI/LLM**: OpenAI (gpt-4.1-mini), Anthropic Claude
- **Authentication**: JWT + OAuth2
- **Real-time**: WebSockets + Server-Sent Events
- **Containerization**: Docker + Docker Compose
- **Task Queue**: Celery + Redis

## 🎉 System Status

### ✅ **Production Ready** (346+ Active API Endpoints)

- **🏢 Profile Management**: 120+ endpoints (10 categories)
- **🤖 AI/LLM Services**: Full OpenAI & Anthropic integration
- **⚙️ Admin Configuration**: Dynamic model management
- **📊 Analytics & Monitoring**: Real-time system metrics
- **🔧 Cross-System Integration**: Unified data flow

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### 🔨 Installation

1. **Clone Repository**
```bash
git clone https://github.com/ETekinYasar/stayin-alive-scenario-backend.git
cd stayin-alive-scenario-backend
```

2. **Environment Setup**
```bash
cp .env.example .env
# Configure your environment variables
```

3. **Start Services**
```bash
docker-compose up -d
```

4. **Database Migration**
```bash
docker-compose exec api alembic upgrade head
```

### 🌐 API Access
- **API Documentation**: http://localhost:8000/docs
- **Admin Panel**: http://localhost:8000/admin
- **Health Check**: http://localhost:8000/health

## 📚 API Documentation

### 🔑 Authentication
```bash
# Register User
POST /api/v1/auth/register
{
  \"email\": \"user@email.com\",
  \"password\": \"SecurePass123\",
  \"full_name\": \"User Name\"
}

# Login
POST /api/v1/auth/login
{
  \"email\": \"user@email.com\",
  \"password\": \"SecurePass123\"
}
```

### 🤖 AI/LLM Features
```bash
# AI Scenario Generation
POST /api/v1/ai/scenarios/generate
Authorization: Bearer {token}
{
  \"scenario_type\": \"earthquake\",
  \"location\": \"Istanbul\",
  \"complexity\": \"high\"
}
```

### 📊 System Capabilities

#### **Profile Management System**
- **120+ Endpoints** across 10 categories
- Dynamic profile templates
- Resource management
- Location-based profiles
- People relationship management
- AI-assisted profile generation

#### **AI/LLM Integration**
- **OpenAI GPT-4.1-mini** support
- **Anthropic Claude** integration
- Admin-controlled model selection
- Cost optimization & budgeting
- Response caching

#### **Admin Configuration**
- Dynamic LLM model management
- Feature flags & A/B testing
- System-wide configuration
- Audit trails & logging

## 🔧 Development

### 📁 Project Structure
```
app/
├── api/v1/              # API endpoints
│   ├── auth.py          # Authentication
│   ├── profiles/        # Profile management (120+ endpoints)
│   ├── ai/              # AI/LLM services
│   └── admin/           # Admin configuration
├── core/                # Core configuration
├── models/              # Database models
├── services/            # Business logic
├── schemas/             # Pydantic models
└── utils/               # Utilities
```

### 🧪 Testing
```bash
# Run comprehensive tests
pytest

# Test specific module
pytest tests/test_ai_integration.py

# Coverage report
pytest --cov=app tests/
```

### 🛠️ Development Workflow
```bash
# Code formatting
black app/
isort app/

# Type checking
mypy app/

# Database migration
alembic revision --autogenerate -m \"description\"
alembic upgrade head
```

## 🌟 Key Features

### 🎯 **Real User Testing Ready**
- Comprehensive API endpoint coverage
- Real OpenAI API integration
- Live monitoring & debugging
- Production-grade error handling

### 🔐 **Enterprise Security**
- JWT authentication
- Role-based access control
- Rate limiting & throttling
- Security audit logging

### 📈 **Scalability**
- Horizontal scaling support
- Database connection pooling
- Redis caching layer
- Background task processing

### 🤖 **AI-Powered Features**
- Dynamic scenario generation
- Profile analysis & recommendations
- Contextual decision making
- Multi-language support

## 📊 System Integration Status

| Component | Status | Endpoints | Description |
|-----------|--------|-----------|-------------|
| Profile Management | ✅ | 120+ | Complete profile system |
| AI/LLM Services | ✅ | 30+ | OpenAI & Anthropic integration |
| Admin Configuration | ✅ | 25+ | Dynamic system management |
| User Analytics | ✅ | 40+ | Real-time user tracking |
| Core Systems | ✅ | 130+ | Foundation services |

## 🔄 Environment Configuration

### Required Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/stayinalive
REDIS_URL=redis://localhost:6379

# AI/LLM
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256

# Optional Features
ENABLE_WEBSOCKETS=true
ENABLE_REAL_TIME_ANALYTICS=true
```

## 📝 API Usage Examples

### Profile Creation with AI
```python
import httpx

# Create enhanced profile
response = httpx.post(
    \"http://localhost:8000/api/v1/profiles/create-enhanced\",
    headers={\"Authorization\": f\"Bearer {token}\"},
    json={
        \"name\": \"Emergency Coordinator\",
        \"type\": \"emergency_responder\",
        \"location\": \"Istanbul\",
        \"ai_enhancement\": True
    }
)
```

### AI Scenario Generation
```python
# Generate disaster scenario
response = httpx.post(
    \"http://localhost:8000/api/v1/ai/scenarios/generate\",
    headers={\"Authorization\": f\"Bearer {token}\"},
    json={
        \"disaster_type\": \"earthquake\",
        \"magnitude\": \"7.2\",
        \"location\": \"Kahramanmaraş\",
        \"complexity_level\": \"high\",
        \"include_psychological_factors\": True
    }
)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 About

**STAYIN ALIVE** is designed to provide comprehensive disaster preparedness training through realistic scenario simulations powered by advanced AI technology.

---

🎯 **Generated with [Claude Code](https://claude.ai/code)**

Co-Authored-By: Claude <noreply@anthropic.com>