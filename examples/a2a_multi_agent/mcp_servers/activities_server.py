"""
MCP server exposing a `suggest_activities` tool backed by a static lookup table.

Used by the `trip_planner_agent` pipeline as its private toolbox.
Deterministic on purpose, so the demo output is reproducible.

Run with:
    python mcp_servers/activities_server.py
"""

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

# Activities per weather bucket; the generic ones are always suggested.
ACTIVITIES: dict[str, list[str]] = {
    "clear": [
        "Rent a bike and explore the city parks",
        "Take a walking tour of the historic center",
        "Have a picnic or visit a rooftop bar at sunset",
    ],
    "cloudy": [
        "Visit the main museums and galleries",
        "Explore covered markets and food halls",
        "Join a guided city tour",
    ],
    "rain": [
        "Spend the day in museums or science centers",
        "Find a cozy café and try local pastries",
        "Watch a show or visit an indoor market",
    ],
    "snow": [
        "Visit a thermal bath or spa",
        "Enjoy a warm café with a view of the snow",
        "Check out indoor attractions and museums",
    ],
}

GENERIC_ACTIVITIES = [
    "Try a restaurant serving local cuisine",
    "Visit the city's most famous landmark",
]
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("activities_mcp_server")

mcp = FastMCP(
    "activities-tools",
    host=os.getenv("ACTIVITIES_MCP_HOST", "localhost"),
    port=int(os.getenv("ACTIVITIES_MCP_PORT", "8002")),
)


def log_event(message: str) -> None:
    logger.info("activities_mcp_server | %s", message)


def _weather_bucket(weather_condition: str) -> str:
    condition = weather_condition.lower()
    if any(word in condition for word in ("rain", "drizzle", "thunderstorm", "shower")):
        return "rain"
    if any(word in condition for word in ("snow", "freezing", "rime")):
        return "snow"
    if any(word in condition for word in ("cloud", "overcast", "fog")):
        return "cloudy"
    return "clear"


@mcp.tool()
def suggest_activities(city: str, weather_condition: str) -> str:
    """
    Suggest activities for a city given the current weather conditions.

    Args:
        city: City name, e.g. 'Berlin'
        weather_condition: Short weather description, e.g. 'Light rain' or 'Clear sky'
    """
    bucket = _weather_bucket(weather_condition)
    log_event(
        f"suggest_activities called with city={city!r}, weather_condition={weather_condition!r}; bucket={bucket!r}"
    )
    return json.dumps(
        {
            "city": city,
            "weather_bucket": bucket,
            "activities": ACTIVITIES[bucket] + GENERIC_ACTIVITIES,
        }
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
