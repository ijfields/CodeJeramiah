from __future__ import annotations

import os
from typing import Dict, Any

from dotenv import load_dotenv


load_dotenv()

IS_BACKTESTING: bool = os.getenv("IS_BACKTESTING", "true").lower() == "true"

TRADIER_CONFIG: Dict[str, Any] = {
    "API_KEY": os.getenv("TRADIER_API_KEY"),
    "ACCOUNT_ID": os.getenv("TRADIER_ACCOUNT_ID"),
}

TRADELOCKER_CONFIG: Dict[str, Any] = {
    "ENVIRONMENT": os.getenv("TRADELOCKER_ENV", "https://demo.tradelocker.com"),
    "USERNAME": os.getenv("TRADELOCKER_USERNAME"),
    "PASSWORD": os.getenv("TRADELOCKER_PASSWORD"),
    "SERVER": os.getenv("TRADELOCKER_SERVER"),
}

DATA_ROOT: str | None = os.getenv("DATA_ROOT")

