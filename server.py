from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from astronomy_solver import (
    solar_longitude,
    subsolar_point,
    find_equinox,
    find_season_events,
    find_sunset_utc,
    get_default_kernel_name
)

app = FastAPI(title="HPC Astronomy Authority")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "defaultKernel": get_default_kernel_name()
    }

@app.get("/equinox/{year}")
def equinox(year: str):
    try:
        year_int = int(year)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year")
    dt_str, kernel = find_equinox(year_int)
    if dt_str is None:
        raise HTTPException(status_code=404, detail="Equinox not found")
    return {
        "year": year_int,
        "kernel": kernel,
        "equinoxUTC": dt_str
    }

@app.get("/equinox-bce/{year}")
def equinox_bce(year: int):
    # 1 BC = astro year 0, 2 BC = astro year -1, 1451 BC = astro year -1450
    astro_year = -(year - 1)
    dt_str, kernel = find_equinox(astro_year)
    if dt_str is None:
        raise HTTPException(status_code=404, detail="Equinox not found")
    return {
        "year": astro_year,
        "bceYear": year,
        "kernel": kernel,
        "equinoxUTC": dt_str
    }

@app.get("/season-events")
def season_events(year: int):
    events, kernel = find_season_events(year)
    required_keys = [
        "spring_equinox",
        "summer_solstice",
        "autumn_equinox",
        "winter_solstice",
    ]
    if not all(key in events for key in required_keys):
        raise HTTPException(status_code=404, detail="One or more season events not found")
    return {
        "year": year,
        "kernel": kernel,
        "events": {
            "spring_equinox": {
                "utc": events["spring_equinox"],
                "eventType": "spring_equinox",
            },
            "summer_solstice": {
                "utc": events["summer_solstice"],
                "eventType": "summer_solstice",
            },
            "autumn_equinox": {
                "utc": events["autumn_equinox"],
                "eventType": "autumn_equinox",
            },
            "winter_solstice": {
                "utc": events["winter_solstice"],
                "eventType": "winter_solstice",
            },
        },
    }

@app.get("/season-events-bce/{year}")
def season_events_bce(year: int):
    astro_year = -(year - 1)
    events, kernel = find_season_events(astro_year)
    required_keys = [
        "spring_equinox",
        "summer_solstice",
        "autumn_equinox",
        "winter_solstice",
    ]
    if not all(key in events for key in required_keys):
        raise HTTPException(status_code=404, detail="One or more season events not found")
    return {
        "year": astro_year,
        "bceYear": year,
        "kernel": kernel,
        "events": {
            "spring_equinox": {
                "utc": events["spring_equinox"],
                "eventType": "spring_equinox",
            },
            "summer_solstice": {
                "utc": events["summer_solstice"],
                "eventType": "summer_solstice",
            },
            "autumn_equinox": {
                "utc": events["autumn_equinox"],
                "eventType": "autumn_equinox",
            },
            "winter_solstice": {
                "utc": events["winter_solstice"],
                "eventType": "winter_solstice",
            },
        },
    }

@app.get("/solar_longitude")
def solar(date: str):
    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    result = solar_longitude(dt)
    return {
        "date": date,
        "kernel": result["kernel"],
        "solarLongitude": result["solarLongitude"],
        "subsolarLatitude": result["subsolarLatitude"],
        "subsolarLongitude": result["subsolarLongitude"]
    }

@app.get("/subsolar-point")
def subsolar(date: str):
    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    point = subsolar_point(dt)
    return {
        "date": date,
        "subsolarLatitude": point["latitude"],
        "subsolarLongitude": point["longitude"]
    }

@app.get("/sunset")
def sunset(date: str, latitude: float, longitude: float):
    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    sunset_dt, kernel = find_sunset_utc(dt, latitude, longitude)
    if sunset_dt is None:
        raise HTTPException(status_code=404, detail="Sunset not found")
    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "kernel": kernel,
        "sunsetUTC": sunset_dt.isoformat()
    }