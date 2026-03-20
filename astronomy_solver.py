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
    elif year < PRIMARY_START_YEAR:
        return ANCIENT_KERNEL
    else:
        return FUTURE_KERNEL

def get_default_kernel_name() -> str:
    return PRIMARY_KERNEL

def get_eph_for_year(year: int):
    return load_kernel(choose_kernel_name(year))

def get_eph_for_datetime(dt: datetime):
    return load_kernel(choose_kernel_name(dt.year))

def subsolar_point(dt: datetime):
    eph = get_eph_for_datetime(dt)
    earth = eph["earth"]
    sun = eph["sun"]
    t = ts.from_datetime(dt.astimezone(timezone.utc))
    apparent = earth.at(t).observe(sun).apparent()
    subpoint = wgs84.subpoint(apparent)
    return {
        "latitude": subpoint.latitude.degrees,
        "longitude": subpoint.longitude.degrees
    }

def solar_longitude(dt: datetime):
    eph = get_eph_for_datetime(dt)
    earth = eph["earth"]
    sun = eph["sun"]
    t = ts.from_datetime(dt.astimezone(timezone.utc))
    apparent = earth.at(t).observe(sun).apparent()
    lat, lon, dist = apparent.ecliptic_latlon()
    subpoint = wgs84.subpoint(apparent)
    return {
        "solarLongitude": lon.degrees % 360,
        "subsolarLatitude": subpoint.latitude.degrees,
        "subsolarLongitude": subpoint.longitude.degrees,
        "kernel": choose_kernel_name(dt.year)
    }

def format_skyfield_time(t) -> str:
    """
    Safely format a Skyfield time object to ISO string.
    Works for all years including BCE (negative astronomical years).
    """
    try:
        return t.utc_strftime('%Y-%m-%dT%H:%M:%S') + '+00:00'
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
            return f'-{bce_year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00'
        return f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00'

def find_equinox(year: int):
    eph = get_eph_for_year(year)
    sky_year = year if year > 0 else year + 1
    t0 = ts.utc(sky_year, 1, 1)
    t1 = ts.utc(sky_year, 12, 31)
    f = almanac.seasons(eph)
    times, events = almanac.find_discrete(t0, t1, f)
    for t, e in zip(times, events):
        if e == 0:
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
    for t, e in zip(times, events):
        event_name = season_map.get(int(e))
        if event_name is not None:
            results[event_name] = format_skyfield_time(t)
    return results, choose_kernel_name(year)
    results = {}
    for t, e in zip(times, events):
        event_name = season_map.get(int(e))
        if event_name is not None:
            # Use utc_strftime to avoid Python datetime negative year limitation
           results[event_name] = t.utc_strftime('%Y-%m-%dT%H:%M:%S') + '+00:00' 
    return results, choose_kernel_name(year)

def find_sunset_utc(date_utc: datetime, latitude: float, longitude: float):
    eph = get_eph_for_datetime(date_utc)
    location = wgs84.latlon(latitude, longitude)
    start_dt = datetime(
        date_utc.year,
        date_utc.month,
        date_utc.day,
        0, 0, 0,
        tzinfo=timezone.utc
    )
    end_dt = start_dt + timedelta(days=2)
    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(end_dt)
    f = almanac.sunrise_sunset(eph, location)
    times, events = almanac.find_discrete(t0, t1, f)
    for t, is_sun_up in zip(times, events):
        if not bool(is_sun_up):
            return t.utc_datetime(), choose_kernel_name(date_utc.year)
    return None, choose_kernel_name(date_utc.year)

def get_delta_t(year: int) -> dict:
    """
    Returns the Delta T value for a given year.

    Delta T (ΔT) is the difference between Terrestrial Time (TT) and
    Universal Time (UT1). It accounts for the gradual slowing of Earth's
    rotation due to tidal friction from the moon.

    For modern dates (post-1900): IERS observed values — highly accurate.
    For ancient dates: Morrison-Stephenson polynomial extrapolation.

    Impact on HPC calendar:
    - Modern dates (2019+): ΔT ≈ 70 seconds — negligible
    - 1451 BC (Exodus): ΔT ≈ 16,800 seconds (~4.7 hours)
    - 967 BC (Solomon): ΔT ≈ 14,200 seconds (~3.9 hours)

    Note: Skyfield applies ΔT automatically in all ephemeris calculations.
    This endpoint exposes the value for research transparency.
    """
    sky_year = year if year > 0 else year + 1
    t = ts.utc(sky_year, 6, 15)  # Mid-year sample for representative value
    delta_t_seconds = float(t.delta_t)

    # Determine model used
    if year >= 1900:
        model = "IERS observed data"
        accuracy = "high — sub-second accuracy"
    elif year >= -500:
        model = "Morrison-Stephenson 2004 polynomial"
        accuracy = "moderate — uncertainty ±few minutes"
    else:
        model = "Morrison-Stephenson 2004 polynomial (extrapolated)"
        accuracy = "approximate — uncertainty increases with age"

    # Flag if ΔT is large enough to potentially affect observable window
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
            "ΔT is applied automatically by Skyfield in all HPC calculations. "
            "This value is provided for research transparency only. "
            "For ancient dates ΔT affects the precise UTC time of the equinox "
            "but rarely changes the observable window assignment."
        )
    }