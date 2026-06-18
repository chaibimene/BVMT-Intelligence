#!/usr/bin/env python3
"""
BVMT Intelligence - Server Startup Script
Starts the FastAPI backend server
"""

import uvicorn
import os
import sys

def main():
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    print("=" * 60)
    print("BVMT Intelligence API Server")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:8000")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Frontend: http://localhost:5173")
    print("=" * 60)
    print("\nDefault admin credentials:")
    print("  Email: imene@bvmt.com")
    print("  Password: admin123")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Initialize database and create default admin
    from api.database import init_db, SessionLocal, User, UserRole
    from api.auth import get_password_hash
    from api.main import app
    from sqlalchemy.orm import Session
    
    init_db()
    
    # Create default admin if not exists
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == "imene@bvmt.com").first()
        if not admin:
            admin = User(
                email="imene@bvmt.com",
                name="Admin BVMT",
                hashed_password=get_password_hash("admin123"),
                role=UserRole.ADMIN,
                is_active=True
            )
            db.add(admin)
            db.commit()
            print("OK - Default admin account created: imene@bvmt.com / admin123\n")
    finally:
        db.close()
    
    # Start server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()