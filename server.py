from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from astronomy_solver import (
    solar_longitude,
    subsolar_point,
    find_equinox,
    find_season_events,
    find_sunset_utc,
    find_next_sunset_after_utc,
    get_default_kernel_name,
    get_delta_t,
)

app = FastAPI(title="HPC Astronomy Authority")


def parse_utc_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "defaultKernel": get_default_kernel_name(),
    }


@app.get("/delta-t/{year}")
def delta_t(year: int):
    try:
        return get_delta_t(year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/delta-t-bce/{year}")
def delta_t_bce(year: int):
    try:
        astro_year = -(year - 1)
        result = get_delta_t(astro_year)
        result["bceYear"] = year
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        "equinoxUTC": dt_str,
    }


@app.get("/equinox-bce/{year}")
def equinox_bce(year: int):
    astro_year = -(year - 1)

    dt_str, kernel = find_equinox(astro_year)

    if dt_str is None:
        raise HTTPException(status_code=404, detail="Equinox not found")

    return {
        "year": astro_year,
        "bceYear": year,
        "kernel": kernel,
        "equinoxUTC": dt_str,
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
        raise HTTPException(
            status_code=404,
            detail="One or more season events not found",
        )

    return {
        "year": year,
        "kernel": kernel,
        "events": {
            key: {
                "utc": events[key],
                "eventType": key,
            }
            for key in required_keys
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
        raise HTTPException(
            status_code=404,
            detail="One or more season events not found",
        )

    return {
        "year": astro_year,
        "bceYear": year,
        "kernel": kernel,
        "events": {
            key: {
                "utc": events[key],
                "eventType": key,
            }
            for key in required_keys
        },
    }


@app.get("/solar_longitude")
def solar(date: str):
    dt = parse_utc_datetime(date)
    result = solar_longitude(dt)

    return {
        "date": date,
        "kernel": result["kernel"],
        "solarLongitude": result["solarLongitude"],
        "subsolarLatitude": result["subsolarLatitude"],
        "subsolarLongitude": result["subsolarLongitude"],
    }


@app.get("/subsolar-point")
def subsolar(date: str):
    dt = parse_utc_datetime(date)
    point = subsolar_point(dt)

    return {
        "date": date,
        "subsolarLatitude": point["latitude"],
        "subsolarLongitude": point["longitude"],
    }


@app.get("/sunset")
def sunset(date: str, latitude: float, longitude: float):
    dt = parse_utc_datetime(date)

    sunset_dt, kernel = find_sunset_utc(dt, latitude, longitude)

    if sunset_dt is None:
        raise HTTPException(status_code=404, detail="Sunset not found")

    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "kernel": kernel,
        "sunsetUTC": sunset_dt.isoformat(),
    }


@app.get("/sunset-after")
def sunset_after(afterUTC: str, latitude: float, longitude: float):
    after_dt = parse_utc_datetime(afterUTC)

    sunset_dt, kernel = find_next_sunset_after_utc(
        after_dt,
        latitude,
        longitude,
    )

    if sunset_dt is None:
        raise HTTPException(status_code=404, detail="Next sunset not found")

    return {
        "afterUTC": afterUTC,
        "latitude": latitude,
        "longitude": longitude,
        "kernel": kernel,
        "sunsetUTC": sunset_dt.isoformat(),
    }