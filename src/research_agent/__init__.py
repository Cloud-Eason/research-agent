"""LLM Research Agent: multi-hop research with RAG, crash recovery and evaluation."""

from dotenv import load_dotenv

# Export .env into os.environ as early as possible. pydantic-settings only fills
# our Settings object; libraries such as huggingface_hub read os.environ directly
# (HF_HUB_OFFLINE, HF_HOME, HF_TOKEN), and some read it once at import time.
# override=False: variables already set in the shell still win.
load_dotenv(override=False)

__version__ = "0.1.0"
