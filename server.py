import math

from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone

from runtime_enforcement import verify_runtime_scientific_components

# Certify the scientific runtime before any astronomy is loaded.
# astronomy_solver constructs its Skyfield timescale from the bundled
# Delta-T table at module import, so the gate must run first: an
# uncertified runtime must never build a timescale. The astronomy import
# below is therefore deliberately late.
verify_runtime_scientific_components()

from astronomy_solver import (  # noqa: E402 - deliberate: gate runs first
    solar_longitude,
    subsolar_point,
    find_equinox,
    find_season_events,
    find_sunset_utc,
    find_next_sunset_after_utc,
    get_default_kernel_name,
    get_delta_t,
)
from sunset_cursor import (  # noqa: E402 - deliberate: gate runs first
    LATITUDE_DOMAIN,
    LONGITUDE_DOMAIN,
    SunsetCursorError,
    encode_sunset_cursor,
)

app = FastAPI(title="HPC Astronomy Authority")


def parse_utc_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


CURSOR_UNAVAILABLE_OBSERVER = "OBSERVER_OUT_OF_CURSOR_DOMAIN"
CURSOR_UNAVAILABLE_ENCODING = "CURSOR_ENCODING_FAILED"


def observer_is_cursor_representable(latitude: float, longitude: float) -> bool:
    """Predict whether the cursor encoder can represent this observer.

    The domains are imported from sunset_cursor rather than restated here,
    so this prediction cannot drift from the encoder it predicts.

    This is a prediction, NOT request validation. No sunset request is
    accepted or rejected on its result, and the coordinate handling of
    /sunset and /sunset-after is unchanged.
    """
    for value, (low, high) in (
        (latitude, LATITUDE_DOMAIN),
        (longitude, LONGITUDE_DOMAIN),
    ):
        if not math.isfinite(value) or value < low or value > high:
            return False

    return True


def cursor_fields(tt, latitude: float, longitude: float) -> dict:
    """Build the additive continuation-witness fields for a sunset response.

    Emission is additive. An observer the cursor encoder cannot represent
    is reported as an explicit unavailability rather than as an error, and
    a SunsetCursorError raised while encoding is converted into the same
    explicit representation: ``cursor`` is null and a reason is returned.

    No claim is made that every possible runtime failure is converted.
    Only SunsetCursorError is handled here; anything that sunset_cursor
    does not normalize to it propagates unchanged.

    This governs response emission only. It is not coordinate validation:
    no request is accepted or rejected here, and the sunset fields of a
    successful determination remain the governing response content.

    ``cursorUnavailableReason`` is present only when ``cursor`` is null.
    """
    if not observer_is_cursor_representable(latitude, longitude):
        return {
            "cursor": None,
            "cursorUnavailableReason": CURSOR_UNAVAILABLE_OBSERVER,
        }

    try:
        cursor = encode_sunset_cursor(
            tt=tt,
            latitude=latitude,
            longitude=longitude,
        )
    except SunsetCursorError:
        return {
            "cursor": None,
            "cursorUnavailableReason": CURSOR_UNAVAILABLE_ENCODING,
        }

    return {"cursor": cursor}


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

    determination = find_sunset_utc(dt, latitude, longitude)

    if determination.utc is None:
        raise HTTPException(status_code=404, detail="Sunset not found")

    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "kernel": determination.kernel,
        "sunsetUTC": determination.utc.isoformat(),
        **cursor_fields(determination.tt, latitude, longitude),
    }


@app.get("/sunset-after")
def sunset_after(afterUTC: str, latitude: float, longitude: float):
    after_dt = parse_utc_datetime(afterUTC)

    determination = find_next_sunset_after_utc(
        after_dt,
        latitude,
        longitude,
    )

    if determination.utc is None:
        raise HTTPException(status_code=404, detail="Next sunset not found")

    return {
        "afterUTC": afterUTC,
        "latitude": latitude,
        "longitude": longitude,
        "kernel": determination.kernel,
        "sunsetUTC": determination.utc.isoformat(),
        **cursor_fields(determination.tt, latitude, longitude),
    }