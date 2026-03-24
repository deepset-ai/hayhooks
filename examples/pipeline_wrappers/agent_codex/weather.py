import json

import httpx
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

from hayhooks import log

WEATHER_AGENT_MODEL = "gpt-4o-mini"
NO_WEATHER_CONTEXT = "NO_WEATHER_CONTEXT"

_WEATHER_KEYWORDS = (
    "weather", "temperature", "forecast", "rain", "snow",
    "wind", "humidity", "sunny", "cloudy",
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

SERVER_WEATHER_SYSTEM_PROMPT = (
    "You are a weather lookup specialist. "
    "If the user asks for current weather conditions, call the server_get_weather tool exactly once and "
    "return a concise weather summary. "
    f"If the user request is not about weather, respond exactly with {NO_WEATHER_CONTEXT}."
)


def looks_like_weather_request(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in _WEATHER_KEYWORDS)


def get_weather(location: str) -> str:
    """Fetch current weather for a location via Open-Meteo (JSON string)."""
    log.opt(colors=True).info("<yellow>[weather]</yellow> server_get_weather called for '{}'", location)

    try:
        with httpx.Client(timeout=10) as client:
            geocoding = client.get(GEOCODING_URL, params={"name": location, "count": 1, "language": "en"})
            geocoding.raise_for_status()
            results = geocoding.json().get("results")
            if not results:
                log.opt(colors=True).warning("<yellow>[weather]</yellow> Location not found: '{}'", location)
                return json.dumps({"error": f"Location '{location}' not found"})

            place = results[0]
            lat, lon = place["latitude"], place["longitude"]
            resolved_name = place.get("name", location)
            log.opt(colors=True).debug("<yellow>[weather]</yellow> Geocoded '{}' -> '{}' ({}, {})", location, resolved_name, lat, lon)

            forecast = client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                },
            )
            forecast.raise_for_status()
            current = forecast.json().get("current", {})
    except httpx.HTTPError as exc:
        log.opt(colors=True).warning("<yellow>[weather]</yellow> Request failed for '{}': {}", location, exc)
        return json.dumps({"error": f"Weather API request failed: {exc}"})
    except Exception as exc:
        log.opt(colors=True).warning("<yellow>[weather]</yellow> Unexpected error for '{}': {}", location, exc)
        return json.dumps({"error": f"Unexpected weather lookup error: {exc}"})

    weather_code = current.get("weather_code", -1)
    payload = {
        "location": resolved_name,
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "weather_code": weather_code,
        "weather_description": WMO_WEATHER_CODES.get(weather_code, "Unknown"),
    }
    log.opt(colors=True).info("<yellow>[weather]</yellow> Success for '{}': {} C, {}", payload["location"], payload["temperature"], payload["weather_description"])
    return json.dumps(payload)


server_weather_tool = Tool(
    name="server_get_weather",
    description="Get current weather for a city via Open-Meteo. Returns temperature, humidity, wind speed and condition.",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name, e.g. 'Berlin'"}},
        "required": ["location"],
    },
    function=get_weather,
)


def create_weather_agent() -> Agent:
    return Agent(
        chat_generator=OpenAIChatGenerator(model=WEATHER_AGENT_MODEL),
        system_prompt=SERVER_WEATHER_SYSTEM_PROMPT,
        tools=[server_weather_tool],
        max_agent_steps=4,
        exit_conditions=["text"],
    )
