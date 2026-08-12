import sys
import os
import uvicorn

# 1. Force Python to add your current root folder to its search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 2. Import the app directly instead of using a string
from src.api.app import app

if __name__ == "__main__":
    # 3. Run the server using the actual app object
    uvicorn.run(app, host="127.0.0.1", port=8000)