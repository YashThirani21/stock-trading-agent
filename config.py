"""Load config.yaml — single source of truth for prompts and model settings."""

import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

MODEL_ORCHESTRATOR = _config["models"]["orchestrator"]
MODEL_SPECIALIST = _config["models"]["specialist"]
MAX_TURNS_ORCHESTRATOR = _config["max_turns"]["orchestrator"]
MAX_TURNS_SPECIALIST = _config["max_turns"]["specialist"]

ORCHESTRATOR_PROMPT = _config["prompts"]["orchestrator"]
MARKET_ANALYST_PROMPT = _config["prompts"]["market_analyst"]
NEWS_ANALYST_PROMPT = _config["prompts"]["news_analyst"]
RISK_MANAGER_PROMPT = _config["prompts"]["risk_manager"]
TRADER_PROMPT = _config["prompts"]["trader"]
