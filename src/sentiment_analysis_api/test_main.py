import os
import sys  # <--- This was missing!

# --- START PATH SETUP ---
# 1. Get the folder of this file (src/sentiment_analysis_api)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to get the project root (SentimentAnalysisAPI)
#    (current_dir -> src -> root)
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# 3. Add the project root to the system path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END PATH SETUP ---

# Now this import will work because Python can see the 'src' folder
from fastapi.testclient import TestClient
from src.sentiment_analysis_api.main import app

client = TestClient(app)

def test_positive_sentiment():
    response = client.post("/v1/sentiment", json={"text": "Det var en god lærer."})
    assert response.status_code == 200
    assert response.json() == {"score": 3}

def test_negative_sentiment():
    response = client.post("/v1/sentiment", json={"text": "It was a bad course"})
    assert response.status_code == 200
    assert response.json() == {"score": -3}

if __name__ == "__main__":
    test_positive_sentiment()
    test_negative_sentiment()
    print("All tests passed!")