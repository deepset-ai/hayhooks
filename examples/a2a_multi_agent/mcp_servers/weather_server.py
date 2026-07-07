"""
MCP server exposing a `get_weather` tool backed by the free Open-Meteo API.

Used by the `weather_agent` pipeline as its private toolbox.

Run with:
    python mcp_servers/weather_server.py
"""

import json
import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("weather_mcp_server")

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

mcp = FastMCP(
    "weather-tools",
    host=os.getenv("WEATHER_MCP_HOST", "localhost"),
    port=int(os.getenv("WEATHER_MCP_PORT", "8001")),
)


def log_event(message: str) -> None:
    logger.info("weather_mcp_server | %s", message)


@mcp.tool()
def get_weather(location: str) -> str:
    """
    Get current weather for a city. Returns temperature, humidity, wind speed, and conditions.

    Args:
        location: City name, e.g. 'Berlin'
    """
    log_event(f"get_weather called with location={location!r}")
    with httpx.Client(timeout=10) as client:
        log_event("resolving location with Open-Meteo geocoding")
        geo = client.get(GEOCODING_URL, params={"name": location, "count": 1, "language": "en"})
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            log_event(f"location not found: {location!r}")
            return json.dumps({"error": f"Location '{location}' not found"})

        place = results[0]
        log_event(f"fetching current forecast for {place.get('name', location)!r}")
        forecast = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            },
        )
        forecast.raise_for_status()
        current = forecast.json().get("current", {})

    weather_code = current.get("weather_code", -1)
    log_event(f"returning weather: {current.get('temperature_2m')} C, {WMO_WEATHER_CODES.get(weather_code, 'Unknown')}")
    return json.dumps(
        {
            "location": place.get("name", location),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather_code": weather_code,
            "weather_description": WMO_WEATHER_CODES.get(weather_code, "Unknown"),
        }
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
