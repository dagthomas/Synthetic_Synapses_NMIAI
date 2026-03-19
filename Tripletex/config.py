import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"
MAX_AGENT_TURNS = 25
