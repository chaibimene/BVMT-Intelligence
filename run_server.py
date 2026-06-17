"""Entry point to run the BVMT Intelligence API server."""
import sys
import os

# Add the project root to Python's path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)