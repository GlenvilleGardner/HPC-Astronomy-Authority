import math
import os
from collections import namedtuple
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
from skyfield import almanac

from scientific_environment import ScientificEnvironmentError

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


# ---------------------------------------------------------------------------
# A1b - tolerance-free sunset successor continuation.
#
# Continues a sunset search from the exact Terrestrial Time solver state
# carried by a validated continuation witness (A1a). The witness records a
# solver start value; it is NOT physical-event identity, and A1a does not
# and cannot decide whether that value actually sits on the post-transition
# side of a crossing. That astronomical question is answered here, before
# any search is permitted to run.
#
# NO TIME-GAP TOLERANCE IS USED OR NEEDED.
#
# find_discrete reports only points at which the sampled predicate changes
# value. The first sample of the successor bracket is the continuation state
# itself, and it has already been proven sun-down. The crossing that
# produced it therefore presents no sign change inside the bracket and
# cannot be rediscovered. Exclusion of the originating root is a structural
# consequence of the validated post-transition start - not a minimum gap,
# not an epsilon, not a time-distance identity test.
#
# The 3600-second guard in find_next_sunset_after_utc is deliberately NOT
# reused, NOT extended and NOT removed. It governs /sunset-after, which
# resumes from an arbitrary caller-supplied instant with no proof of which
# side of a crossing it lies on, and it remains that route's business.
#
# SUCCESSOR_SEARCH_SPAN_DAYS is a search horizon, not an event-identity
# tolerance. It equals the forward reach already used by
# find_next_sunset_after_utc, so polar and no-event behavior is preserved
# rather than changed: an observer with no sunset inside the horizon yields
# no successor instead of a fabricated one.
# ---------------------------------------------------------------------------

SUCCESSOR_SEARCH_SPAN_DAYS = 3.0

REASON_CURSOR_STATE_INVALID = "CURSOR_STATE_INVALID"
REASON_CURSOR_NOT_POST_TRANSITION = "CURSOR_NOT_POST_TRANSITION"

SunsetSuccessor = namedtuple("SunsetSuccessor", ("utc", "tt", "kernel"))


class SunsetSuccessorError(ScientificEnvironmentError):
    """Sunset successor continuation failed closed.

    ``reason`` carries a stable code so a later route can map every
    rejection onto one HTTP status with a distinguishing detail. No HTTP
    semantics are decided here.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def find_sunset_successor(tt_value: float, latitude: float, longitude: float):
    """Return the next distinct sunset after an exact continuation state.

    ``tt_value`` is the exact binary64 Terrestrial Time Julian Date of a
    previously determined sunset, as carried by a validated continuation
    witness. It is reconstructed with ts.tt_jd and used as the solver start
    state directly: it is never routed through datetime, ISO text, Unix
    seconds or milliseconds, so the value searched from is bit-identical to
    the value supplied.

    The continuation state must be post-transition - the Sun must be down
    for the bound observer at that exact instant. If it is not, the state
    does not describe the far side of a sunset, and this function fails
    closed rather than returning a crossing that would silently answer a
    different question. Passing that check proves only that the supplied
    state lies on the sun-down side of the predicate. It is not proof of
    origin and not proof of physical-event identity.

    Returns a SunsetSuccessor, or None when the observer has no sunset
    within the search horizon.

    Kernel selection follows the ordinary repository routing rule for the
    period in which this search is performed. It is deliberately NOT
    inherited from whatever kernel produced the continuation state: the
    witness binds no kernel filename, because kernel identity is
    computation provenance and not continuation compatibility. The same
    kernel is used for the post-transition check and for the search, so a
    state near a kernel boundary is validated under the kernel that will
    actually perform the search.

    Observer coordinate domains are NOT re-validated here; observer
    binding is owned by the continuation witness.
    """
    if isinstance(tt_value, bool) or not isinstance(tt_value, (int, float)):
        raise SunsetSuccessorError(
            REASON_CURSOR_STATE_INVALID,
            "SUNSET CONTINUATION STATE INVALID - tt must be a real number, "
            "received %s" % type(tt_value).__name__,
        )

    try:
        tt = float(tt_value)
    except OverflowError as error:
        raise SunsetSuccessorError(
            REASON_CURSOR_STATE_INVALID,
            "SUNSET CONTINUATION STATE INVALID - tt is too large to represent "
            "as a float",
        ) from error

    if not math.isfinite(tt):
        raise SunsetSuccessorError(
            REASON_CURSOR_STATE_INVALID,
            "SUNSET CONTINUATION STATE INVALID - tt must be finite, "
            "received %r" % (tt_value,),
        )

    t_cursor = ts.tt_jd(tt)

    kernel_name = choose_kernel_name(int(t_cursor.utc.year))
    eph = load_kernel(kernel_name)

    is_sun_up = almanac.sunrise_sunset(eph, wgs84.latlon(latitude, longitude))

    if bool(is_sun_up(t_cursor)):
        raise SunsetSuccessorError(
            REASON_CURSOR_NOT_POST_TRANSITION,
            "SUNSET CONTINUATION STATE NOT POST-TRANSITION - the Sun is up "
            "for the bound observer at the supplied continuation state, so "
            "it does not lie on the far side of a sunset; refusing to "
            "continue",
        )

    t_horizon = ts.tt_jd(tt + SUCCESSOR_SEARCH_SPAN_DAYS)

    times, events = almanac.find_discrete(t_cursor, t_horizon, is_sun_up)

    for t, sun_is_up in zip(times, events):
        if not bool(sun_is_up):
            return SunsetSuccessor(
                utc=t.utc_datetime(),
                tt=float(t.tt),
                kernel=kernel_name,
            )

    return None

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
