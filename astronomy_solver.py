import os
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
from skyfield import almanac

PRIMARY_KERNEL = os.getenv("HPC_EPHEMERIS_PRIMARY", "de440.bsp")
ANCIENT_KERNEL = os.getenv("HPC_EPHEMERIS_ANCIENT", "de441_part-1.bsp")
FUTURE_KERNEL = os.getenv("HPC_EPHEMERIS_FUTURE", "de441_part-2.bsp")

PRIMARY_START_YEAR = 1550
PRIMARY_END_YEAR = 2650
EPHEMERIS_DIR = os.path.join(os.path.dirname(__file__), "ephemeris")

ts = load.timescale()


@lru_cache(maxsize=8)
def load_kernel(kernel_name: str):
    path = os.path.join(EPHEMERIS_DIR, kernel_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ephemeris kernel not found: {path}. "
            f"Download the BSP file into the ephemeris folder."
        )

    return load(path)


def choose_kernel_name(year: int) -> str:
    if PRIMARY_START_YEAR <= year <= PRIMARY_END_YEAR:
        return PRIMARY_KERNEL

    if year < PRIMARY_START_YEAR:
        return ANCIENT_KERNEL

    return FUTURE_KERNEL


def get_default_kernel_name() -> str:
    return PRIMARY_KERNEL


def get_eph_for_year(year: int):
    return load_kernel(choose_kernel_name(year))


def get_eph_for_datetime(dt: datetime):
    return load_kernel(choose_kernel_name(dt.year))


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def subsolar_point(dt: datetime):
    dt = ensure_utc(dt)

    eph = get_eph_for_datetime(dt)
    earth = eph["earth"]
    sun = eph["sun"]

    t = ts.from_datetime(dt)
    apparent = earth.at(t).observe(sun).apparent()
    subpoint = wgs84.subpoint(apparent)

    return {
        "latitude": subpoint.latitude.degrees,
        "longitude": subpoint.longitude.degrees,
    }


def solar_longitude(dt: datetime):
    dt = ensure_utc(dt)

    eph = get_eph_for_datetime(dt)
    earth = eph["earth"]
    sun = eph["sun"]

    t = ts.from_datetime(dt)
    apparent = earth.at(t).observe(sun).apparent()

    _lat, lon, _dist = apparent.ecliptic_latlon()
    subpoint = wgs84.subpoint(apparent)

    return {
        "solarLongitude": lon.degrees % 360,
        "subsolarLatitude": subpoint.latitude.degrees,
        "subsolarLongitude": subpoint.longitude.degrees,
        "kernel": choose_kernel_name(dt.year),
    }


def format_skyfield_time(t) -> str:
    """
    Safely format a Skyfield time object to ISO UTC string.
    Works for modern and BCE astronomical years.
    """
    try:
        return t.utc_strftime("%Y-%m-%dT%H:%M:%S") + "+00:00"
    except ValueError:
        cal = t.ut1_calendar()

        year = int(cal[0])
        month = int(cal[1])
        day = int(cal[2])
        hour = int(cal[3])
        minute = int(cal[4])
        second = int(cal[5])

        if year <= 0:
            bce_year = abs(year) + 1
            return (
                f"-{bce_year:04d}-{month:02d}-{day:02d}"
                f"T{hour:02d}:{minute:02d}:{second:02d}+00:00"
            )

        return (
            f"{year:04d}-{month:02d}-{day:02d}"
            f"T{hour:02d}:{minute:02d}:{second:02d}+00:00"
        )


def find_equinox(year: int):
    eph = get_eph_for_year(year)

    sky_year = year if year > 0 else year + 1

    t0 = ts.utc(sky_year, 1, 1)
    t1 = ts.utc(sky_year, 12, 31)

    f = almanac.seasons(eph)
    times, events = almanac.find_discrete(t0, t1, f)

    for t, event in zip(times, events):
        if int(event) == 0:
            return format_skyfield_time(t), choose_kernel_name(year)

    return None, choose_kernel_name(year)


def find_season_events(year: int):
    eph = get_eph_for_year(year)

    sky_year = year if year > 0 else year + 1

    t0 = ts.utc(sky_year, 1, 1)
    t1 = ts.utc(sky_year, 12, 31)

    f = almanac.seasons(eph)
    times, events = almanac.find_discrete(t0, t1, f)

    season_map = {
        0: "spring_equinox",
        1: "summer_solstice",
        2: "autumn_equinox",
        3: "winter_solstice",
    }

    results = {}

    for t, event in zip(times, events):
        event_name = season_map.get(int(event))

        if event_name is not None:
            results[event_name] = format_skyfield_time(t)

    return results, choose_kernel_name(year)


def find_sunset_utc(date_utc: datetime, latitude: float, longitude: float):
    """
    HPC-aware local-date sunset finder.

    Given a requested civil date and location, returns the evening sunset
    that closes that local civil day.

    This avoids the UTC-midnight crossover bug where western longitudes
    may have a sunset just after 00:00 UTC and another near 23:xx UTC.

    Strategy:
    - Anchor the search at approximate local solar noon.
    - Search forward 24 hours.
    - Return the first sunset after local solar noon.
    """
    date_utc = ensure_utc(date_utc)

    eph = get_eph_for_datetime(date_utc)
    location = wgs84.latlon(latitude, longitude)

    longitude_offset_hours = longitude / 15.0

    local_noon_utc = datetime(
        date_utc.year,
        date_utc.month,
        date_utc.day,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    ) - timedelta(hours=longitude_offset_hours)

    t0 = ts.from_datetime(local_noon_utc)
    t1 = ts.from_datetime(local_noon_utc + timedelta(hours=24))

    f = almanac.sunrise_sunset(eph, location)
    times, events = almanac.find_discrete(t0, t1, f)

    for t, is_sun_up in zip(times, events):
        if not bool(is_sun_up):
            return t.utc_datetime(), choose_kernel_name(date_utc.year)

    return None, choose_kernel_name(date_utc.year)


def find_next_sunset_after_utc(after_utc: datetime, latitude: float, longitude: float):
    after_utc = ensure_utc(after_utc)

    eph = get_eph_for_datetime(after_utc)
    location = wgs84.latlon(latitude, longitude)

    start_dt = after_utc - timedelta(hours=6)
    end_dt = after_utc + timedelta(hours=72)

    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(end_dt)

    f = almanac.sunrise_sunset(eph, location)
    times, events = almanac.find_discrete(t0, t1, f)

    after_timestamp = after_utc.timestamp()
    min_gap_seconds = 3600  # 1 hour

    for t, is_sun_up in zip(times, events):
        if not bool(is_sun_up):
            sunset_dt = t.utc_datetime()

            if sunset_dt.timestamp() > after_timestamp + min_gap_seconds:
                return sunset_dt, choose_kernel_name(after_utc.year)

    return None, choose_kernel_name(after_utc.year)

def get_delta_t(year: int) -> dict:
    """
    Returns Delta T for research transparency.

    Skyfield applies Delta T internally during ephemeris calculations.
    This endpoint exposes the value so the HPC system can document the
    time-scale correction used in astronomical calculations.
    """
    sky_year = year if year > 0 else year + 1

    t = ts.utc(sky_year, 6, 15)
    delta_t_seconds = float(t.delta_t)

    if year >= 1900:
        model = "IERS observed data"
        accuracy = "high — sub-second accuracy"
    elif year >= -500:
        model = "Morrison-Stephenson 2004 polynomial"
        accuracy = "moderate — uncertainty ±few minutes"
    else:
        model = "Morrison-Stephenson 2004 polynomial (extrapolated)"
        accuracy = "approximate — uncertainty increases with age"

    hours = abs(delta_t_seconds) / 3600
    window_impact = hours > 1.0

    return {
        "year": year,
        "deltaTSeconds": round(delta_t_seconds, 3),
        "deltaTMinutes": round(delta_t_seconds / 60, 3),
        "deltaTHours": round(hours, 4),
        "model": model,
        "accuracy": accuracy,
        "windowImpactPossible": window_impact,
        "note": (
            "Delta T is applied automatically by Skyfield in all HPC calculations. "
            "This value is provided for research transparency. "
            "For ancient dates, Delta T can affect precise UTC event timing."
        ),
    }
