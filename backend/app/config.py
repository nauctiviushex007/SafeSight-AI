"""Central configuration. Values come from environment variables (.env) with safe defaults."""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./safesight.db")
MIN_PERSON_CONF = float(os.getenv("MIN_PERSON_CONF", "0.50"))
MIN_PPE_CONF = float(os.getenv("MIN_PPE_CONF", "0.40"))
MIN_PERSON_HEIGHT_PX = int(os.getenv("MIN_PERSON_HEIGHT_PX", "80"))

# Configurable severity rules: category -> severity
SEVERITY_RULES = {
    "no_helmet": "high",
    "no_vest": "medium",
    "restricted_zone": "critical",
}
CATEGORIES = list(SEVERITY_RULES.keys())
