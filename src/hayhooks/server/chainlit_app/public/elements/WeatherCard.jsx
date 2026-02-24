import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import {
  Thermometer,
  Droplets,
  Wind,
  Sun,
  Cloud,
  CloudRain,
  CloudSnow,
  CloudFog,
  CloudLightning,
  CloudDrizzle,
} from "lucide-react";

// Map WMO weather codes to icon + label
// https://open-meteo.com/en/docs#weathervariables
function weatherInfo(code) {
  if (code === 0) return { icon: Sun, label: "Clear sky" };
  if (code <= 3) return { icon: Cloud, label: "Partly cloudy" };
  if (code <= 48) return { icon: CloudFog, label: "Fog" };
  if (code <= 55) return { icon: CloudDrizzle, label: "Drizzle" };
  if (code <= 57) return { icon: CloudDrizzle, label: "Freezing drizzle" };
  if (code <= 65) return { icon: CloudRain, label: "Rain" };
  if (code <= 67) return { icon: CloudRain, label: "Freezing rain" };
  if (code <= 77) return { icon: CloudSnow, label: "Snow" };
  if (code <= 82) return { icon: CloudRain, label: "Rain showers" };
  if (code <= 86) return { icon: CloudSnow, label: "Snow showers" };
  if (code <= 99) return { icon: CloudLightning, label: "Thunderstorm" };
  return { icon: Cloud, label: "Unknown" };
}

export default function WeatherCard() {
  const {
    location = "Unknown",
    temperature = "--",
    humidity = "--",
    wind_speed = "--",
    weather_code = -1,
    weather_description = "",
  } = props;

  const { icon: WeatherIcon, label: autoLabel } = weatherInfo(weather_code);
  const conditionLabel = weather_description || autoLabel;

  return (
    <Card className="w-full max-w-sm my-2">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{location}</CardTitle>
          <WeatherIcon className="h-6 w-6 text-muted-foreground" />
        </div>
        <CardDescription>{conditionLabel}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-end gap-1 mb-4">
          <span className="text-4xl font-bold">{temperature}</span>
          <span className="text-xl text-muted-foreground mb-1">°C</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Droplets className="h-4 w-4" />
            <span>Humidity: {humidity}%</span>
          </div>
          <div className="flex items-center gap-2">
            <Wind className="h-4 w-4" />
            <span>Wind: {wind_speed} km/h</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
