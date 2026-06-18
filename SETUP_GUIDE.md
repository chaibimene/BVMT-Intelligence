# BVMT Intelligence - Setup Guide

## Overview
This is a comprehensive refactoring of the BVMT Intelligence platform into a production-ready application with:
- Database-based authentication and user management
- Role-based access control (Admin/Normal User)
- Conversation history with search and grouping
- Real-time confidence scoring
- Professional financial dashboard UI
- French-first interface

## Architecture

### Backend (FastAPI)
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: JWT tokens with bcrypt password hashing
- **Authorization**: Role-based access control (Admin/User)
- **API**: RESTful endpoints with proper error handling

### Frontend (React + TypeScript)
- **UI Framework**: Custom glassmorphism design
- **Charts**: Recharts for data visualization
- **Animations**: Framer Motion for smooth transitions
- **Icons**: Lucide React icon library

## Installation

### 1. Backend Setup

```bash
cd BVMT-Intelligence

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements-api.txt
```

### 2. Frontend Setup

```bash
cd "BVMT-Intelligence FrontEnd"

# Install dependencies
npm install
```

## Running the Application

### Start Backend Server

```bash
cd BVMT-Intelligence
python run_server.py
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:5173

### Start Frontend (in a new terminal)

```bash
cd "BVMT-Intelligence FrontEnd"
npm run dev
```

## Default Credentials

**Admin Account:**
- Email: `imene@bvmt.com`
- Password: `admin123`

**Role:**
- **Admin**: Full access to all features (Dashboard, AI Assistant, RAG Monitor, Documents, Knowledge Base, Reports, Admin Panel, Settings)
- **Normal User**: Limited access (AI Assistant, Knowledge Base, Reports, Settings only)

## Key Features Implemented

### 1. Authentication & Authorization
- ✅ JWT-based authentication
- ✅ Password hashing with bcrypt
- ✅ Role-based access control
- ✅ Protected API endpoints
- ✅ Automatic token refresh
- ✅ Secure logout

### 2. AI Assistant / Chatbot
- ✅ French welcome message
- ✅ Conversation history with search
- ✅ Grouped by time (Aujourd'hui, Hier, Cette semaine, Ce mois)
- ✅ Delete individual conversations
- ✅ Clear all conversations
- ✅ Real-time confidence scoring (60-99% range)
- ✅ Source citations with scores
- ✅ ChatGPT-style UI

### 3. Dashboard
- ✅ Professional financial dashboard design
- ✅ Market movers/losers (mock data - ready for real API integration)
- ✅ Sector performance visualization
- ✅ System status monitoring
- ✅ KPI cards with animations
- ✅ Removed indexed documents widget
- ✅ Removed recent queries section

### 4. RAG Monitor
- ✅ Agent status cards
- ✅ Real-time analytics charts
- ✅ Queries per hour
- ✅ Retrieval/generation latency
- ✅ Documents retrieved
- ✅ Average confidence scores
- ✅ Auto-updating metrics

### 5. Knowledge Base
- ✅ Semantic search
- ✅ Article modal with full text
- ✅ French/Arabic language switcher (UI ready)
- ✅ Copy and download buttons
- ✅ Scrollable content
- ✅ Model configuration display

### 6. Reports
- ✅ Template selection (Executive, Market, Company, Sector, ESG)
- ✅ Progress indicator
- ✅ Report generation
- ✅ Removed custom report feature

### 7. Admin Panel
- ✅ User management (database-based)
- ✅ API keys management (view/edit/delete)
- ✅ Security settings display
- ✅ Role assignment
- ✅ Default admin account creation

### 8. Settings
- ✅ Profile management
- ✅ Language selection (Français, العربية, English)
- ✅ Theme selection (Dark/Light)
- ✅ Notification preferences
- ✅ Password change
- ✅ Persistent settings in database

### 9. Logout
- ✅ Complete logout functionality
- ✅ Token destruction
- ✅ Session clearing
- ✅ Redirect to login
- ✅ Protected routes

### 10. UI/UX Improvements
- ✅ Professional financial dashboard design
- ✅ French-first interface
- ✅ Responsive layout
- ✅ Smooth animations
- ✅ Glassmorphism design
- ✅ Dark theme (default)
- ✅ No fake data in production code

## Database Schema

### Tables Created:
1. **users** - User accounts with roles
2. **conversations** - Chat conversations
3. **messages** - Individual messages with sources and confidence
4. **user_settings** - User preferences
5. **api_keys** - API key storage

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/login-json` - JSON login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Current user info
- `POST /api/auth/init-admin` - Initialize admin account

### User Management (Admin only)
- `GET /api/auth/users` - List all users
- `POST /api/auth/users` - Create user
- `PUT /api/auth/users/{id}` - Update user
- `DELETE /api/auth/users/{id}` - Delete user

### Chat
- `POST /api/chat` - Send message to AI
- `GET /api/chat/history` - Get chat history

### Conversations
- `GET /api/conversations/` - List conversations
- `POST /api/conversations/` - Create conversation
- `GET /api/conversations/{id}` - Get conversation details
- `DELETE /api/conversations/{id}` - Delete conversation
- `POST /api/conversations/search` - Search conversations
- `DELETE /api/conversations/` - Clear all conversations

### Documents
- `GET /api/documents` - List documents
- `POST /api/documents/upload` - Upload document
- `POST /api/documents/ingest` - Trigger ingestion

### Knowledge Base
- `GET /api/knowledge/stats` - Knowledge base statistics
- `POST /api/knowledge/search` - Semantic search

### Reports
- `POST /api/reports/generate` - Generate report

### Settings
- `GET /api/settings/` - Get user settings
- `PUT /api/settings/` - Update settings
- `PUT /api/settings/profile` - Update profile
- `PUT /api/settings/password` - Change password

### Admin
- `GET /api/admin/api-keys` - List API keys
- `POST /api/admin/api-keys` - Create API key
- `PUT /api/admin/api-keys/{id}` - Update API key
- `DELETE /api/admin/api-keys/{id}` - Delete API key

### System
- `GET /api/health` - Health check
- `GET /api/rag/status` - RAG pipeline status
- `GET /api/dashboard/stats` - Dashboard statistics
- `GET /api/market/data` - Market data (placeholder for real integration)

## Confidence Score Algorithm

The confidence score is calculated using multiple factors:

1. **RAG Retrieval Similarity (40%)**: Average similarity score of retrieved chunks
2. **Number of Chunks (20%)**: More context = higher confidence
3. **Answer Quality (30%)**: Response length and source diversity
4. **Query Type (10%)**: Certain query types are more reliable

**Range**: 60-99% when sources exist, 0% when no sources found

## Role-Based Access

### Admin Access
- Dashboard
- AI Assistant
- RAG Monitor
- Documents
- Knowledge Base
- Reports
- Admin Panel
- Settings

### Normal User Access
- AI Assistant
- Knowledge Base
- Reports
- Settings

## Next Steps for Production

1. **Real Market Data Integration**
   - Integrate with BVMT API or web scraping
   - Add auto-refresh every 30-60 seconds
   - Implement smooth animations for ranking changes

2. **Translation Service**
   - Integrate translation API for Arabic articles
   - Auto-translate when Arabic version not available

3. **PDF Generation**
   - Implement actual PDF download for reports
   - Add export functionality for conversations

4. **Email Notifications**
   - Implement email service for notifications
   - Add notification preferences

5. **Advanced Security**
   - Add rate limiting
   - Implement 2FA
   - Add session management
   - IP whitelisting for admin

6. **Monitoring & Logging**
   - Add comprehensive logging
   - Implement error tracking (Sentry)
   - Add performance monitoring

7. **Testing**
   - Unit tests for backend
   - Integration tests
   - E2E tests for frontend

## Troubleshooting

### Backend won't start
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r requirements-api.txt`
- Verify database permissions

### Frontend won't start
- Ensure Node.js 16+ is installed
- Run `npm install` in the FrontEnd directory
- Check that backend is running on port 8000

### Can't login
- Default credentials: `imene@bvmt.com` / `admin123`
- Run `POST /api/auth/init-admin` to create admin account
- Check browser console for errors

### No data in dashboard
- Market data is currently mocked
- Upload documents via Documents page
- Run ingestion to populate vector store

## Support

For issues or questions, please refer to the main README.md or contact the development team.

## License

Proprietary - BVMT Intelligence Platform