import json
from collections.abc import AsyncGenerator

import httpx
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, async_streaming_generator
from hayhooks.chainlit_events import create_custom_element_event
from hayhooks.events import PipelineEvent, create_notification_event, create_status_event

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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


def get_weather(location: str) -> str:
    """
    Fetch current weather for a location using the Open-Meteo API.

    Returns a JSON string with weather data, or an error message.
    """
    with httpx.Client(timeout=10) as client:
        geo = client.get(GEOCODING_URL, params={"name": location, "count": 1, "language": "en"})
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return json.dumps({"error": f"Location '{location}' not found"})

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = place.get("name", location)

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

    weather_code = current.get("weather_code", -1)
    return json.dumps(
        {
            "location": resolved_name,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather_code": weather_code,
            "weather_description": WMO_WEATHER_CODES.get(weather_code, "Unknown"),
        }
    )


weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city. Returns temperature, humidity, wind speed, and conditions.",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name, e.g. 'Berlin'"}},
        "required": ["location"],
    },
    function=get_weather,
)


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=(
                "You are a helpful weather assistant. "
                "When the user asks about the weather, use the get_weather tool. "
                "After receiving the result, provide a friendly summary of the conditions."
            ),
            tools=[weather_tool],
        )

    def on_tool_call_start(
        self,
        tool_name: str,  # noqa: ARG002
        arguments: dict,
        id: str,  # noqa: ARG002, A002
    ) -> list[PipelineEvent]:
        location = arguments.get("location", "unknown")
        return [
            create_status_event(description=f"Fetching weather for {location}..."),
            create_notification_event(notification_type="info", content=f"Looking up weather for {location}"),
        ]

    def on_tool_call_end(
        self,
        tool_name: str,
        arguments: dict,  # noqa: ARG002
        result: str,
        error: bool,
    ) -> list[PipelineEvent]:
        events: list[PipelineEvent] = [
            create_status_event(description=f"Weather data received from {tool_name}", done=True),
        ]

        if error:
            events.append(create_notification_event(notification_type="error", content="Failed to fetch weather data"))
            return events

        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return events

        if "error" not in data:
            events.append(
                create_custom_element_event(
                    name="WeatherCard",
                    props={
                        "location": data.get("location", ""),
                        "temperature": data.get("temperature"),
                        "humidity": data.get("humidity"),
                        "wind_speed": data.get("wind_speed"),
                        "weather_code": data.get("weather_code", -1),
                        "weather_description": data.get("weather_description", ""),
                    },
                )
            )

        return events

    async def run_chat_completion_async(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator:
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        return async_streaming_generator(
            on_tool_call_start=self.on_tool_call_start,
            on_tool_call_end=self.on_tool_call_end,
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
