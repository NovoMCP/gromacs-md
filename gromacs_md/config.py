"""Configuration loaded from environment variables."""

import os

PORT = int(os.environ.get("PORT", "8024"))
API_KEY = os.environ.get("API_KEY", "")
MAX_CONCURRENT_SIMS = int(os.environ.get("MAX_CONCURRENT_SIMS", "3"))
NTOMP = int(os.environ.get("NTOMP", "8"))  # OpenMP threads per simulation

# S3 (results bucket). main.py reads MD_RESULTS_BUCKET directly; this
# constant is retained for any legacy consumer that still imports it.

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
