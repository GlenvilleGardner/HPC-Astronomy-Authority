import math
import os
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
from skyfield import almanac
from skyfield.errors import EphemerisRangeError

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


@dataclass(frozen=True)
class SunsetDetermination:
    """A sunset determination and the exact state it was found at.

    ``tt`` is float(t.tt) read directly from the Skyfield Time object the
    solver selected. It is never derived from the UTC datetime, ISO text,
    Unix seconds or milliseconds, so a continuation witness encoded from
    it is bit-identical to the state the root was found at.

    The fields are named and this is deliberately not a tuple: no call
    site can unpack a determination positionally, so a future change to
    field order cannot silently transpose utc, tt and kernel.

    When no sunset exists inside the search window, ``utc`` and ``tt`` are
    both None and ``kernel`` still reports the routing decision, preserving
    the previous (None, kernel) outcome.
    """

    utc: datetime | None
    tt: float | None
    kernel: str


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
            return SunsetDetermination(
                utc=t.utc_datetime(),
                tt=float(t.tt),
                kernel=choose_kernel_name(date_utc.year),
            )

    return SunsetDetermination(
        utc=None,
        tt=None,
        kernel=choose_kernel_name(date_utc.year),
    )


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
                return SunsetDetermination(
                    utc=sunset_dt,
                    tt=float(t.tt),
                    kernel=choose_kernel_name(after_utc.year),
                )

    return SunsetDetermination(
        utc=None,
        tt=None,
        kernel=choose_kernel_name(after_utc.year),
    )


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


# ---------------------------------------------------------------------------
# A2-1 - certified kernel coverage substrate.
#
# WHAT THIS OPERATION ADDS
#
# The ability to ask a pinned NASA/JPL artifact what it actually contains,
# read from that artifact's own segment metadata. Nothing else. No kernel is
# selected here, no interval is solved, no event is determined and no route
# is served; each of those is a separate governed operation.
#
# SCIENTIFIC AUTHORITY
#
# The kernel content is the authority on where a genuine topocentric solar
# event is actually supported. kernel_coverage_tt reads that support from
# the published segment descriptors through the existing certified
# load_kernel path. No declared value, no manifest entry, no configuration
# setting and no civil-year constant contributes to it.
#
# choose_kernel_name and the civil-year constants it is built on are frozen
# legacy compatibility routing: they state which file existing code opens.
# Nothing in this block reads them, and nothing here modifies them or any
# existing route.
#
# TIME SCALES
#
# BSP segment metadata is published in JD(TDB). KernelCoverage preserves the
# raw TDB bounds unaltered for provenance and carries the TT bounds that the
# certified timescale derives from them, because solver intervals are
# expressed in TT. No bound is widened or narrowed to absorb the difference
# between the two scales, and a raw TDB bound is never compared against a TT
# interval.
# ---------------------------------------------------------------------------

REASON_INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_KERNEL_SEGMENTS_UNAVAILABLE = "KERNEL_SEGMENTS_UNAVAILABLE"

# The exact vector chain a topocentric sunset consumes:
#
#     (0, 3)     solar system barycentre -> Earth-Moon barycentre
#     (3, 399)   Earth-Moon barycentre   -> Earth
#     (0, 10)    solar system barycentre -> Sun
#
# eph["earth"] resolves through the first two and eph["sun"] through the
# third. A kernel that does not publish all three cannot answer the
# question at all, whatever else it contains.
REQUIRED_SOLAR_SEGMENT_PAIRS = ((0, 3), (3, 399), (0, 10))


class SunsetChronologyError(ScientificEnvironmentError):
    """Sunset chronology failed closed.

    ``reason`` carries a stable code so a later route can map every
    rejection onto one HTTP status with a distinguishing detail. No HTTP
    semantics are decided here.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class KernelCoverage:
    """What one pinned NASA/JPL kernel actually contains.

    ``tdb_start`` and ``tdb_end`` are the published segment bounds exactly
    as the BSP records them, in JD(TDB), preserved for provenance.
    ``tt_start`` and ``tt_end`` are those same two instants expressed in
    Terrestrial Time by the certified timescale, which is the scale solver
    intervals are compared in. Neither pair is rounded or adjusted.

    The bounds are the intersection across REQUIRED_SOLAR_SEGMENT_PAIRS,
    computed rather than assumed, so they describe the interval over which
    a topocentric sunset is actually supported - not the widest interval
    any one segment happens to reach.

    Named fields, deliberately not a tuple: no call site can unpack a
    coverage record positionally, so a later change to field order cannot
    silently transpose the TDB and TT bounds.
    """

    kernel: str
    tdb_start: float
    tdb_end: float
    tt_start: float
    tt_end: float


def _exact_finite_tt(value, field):
    """Accept only an exact, finite, real Terrestrial Time value.

    Introduced here as approved A2-1 substrate. It has no caller in this
    operation; it is first consumed by the later governed operations that
    accept an anchor instant.

    bool is rejected before the numeric check because it is a subclass of
    int and would otherwise be silently accepted as 0.0 or 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SUNSET CHRONOLOGY STATE INVALID - %s must be a real number, "
            "received %s" % (field, type(value).__name__),
        )

    try:
        number = float(value)
    except OverflowError as error:
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SUNSET CHRONOLOGY STATE INVALID - %s is too large to represent "
            "as a float" % field,
        ) from error

    if not math.isfinite(number):
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SUNSET CHRONOLOGY STATE INVALID - %s must be finite, received %r"
            % (field, value),
        )

    return number


@lru_cache(maxsize=8)
def kernel_coverage_tt(kernel_name: str) -> KernelCoverage:
    """Return what ``kernel_name`` actually covers, read from the BSP itself.

    The kernel is opened through the existing certified load_kernel path;
    no file is parsed independently and no coverage is taken from
    configuration, from a manifest or from any declared value. Only the
    segment metadata published inside the artifact is consulted.

    The TT bounds come from each segment's own time_range against the
    shared certified timescale, so the TDB-to-TT conversion is performed by
    the certified machinery rather than by arithmetic here.

    Cached because the answer is a property of an immutable pinned
    artifact. A failure is not cached: lru_cache stores return values only,
    so a kernel missing a required segment is re-diagnosed on every call.

    SINGLE-SEGMENT INVARIANT

    Each required pair must be represented by exactly one segment, and
    that is enforced rather than assumed. Every certified kernel currently
    publishes 14 segments carrying 14 distinct center/target pairs, so no
    pair is duplicated and the invariant holds with margin.

    It is enforced because the certified machinery, not this reader,
    decides which segment applies when several share a pair: jplephem's
    SPK.__getitem__ returns the last matching segment, while Skyfield
    composes a Stack across the matches and selects per epoch. A duplicate
    would therefore make applicable coverage a question this bounds reader
    cannot answer as a single interval without reimplementing certified
    selection logic. It fails closed instead of guessing.
    """
    eph = load_kernel(kernel_name)

    tdb_start = None
    tdb_end = None
    tt_start = None
    tt_end = None
    missing = []
    ambiguous = []

    for pair in REQUIRED_SOLAR_SEGMENT_PAIRS:
        matches = [
            candidate
            for candidate in eph.segments
            if (candidate.center, candidate.target) == pair
        ]

        if not matches:
            missing.append(pair)
            continue

        if len(matches) > 1:
            ambiguous.append((pair, len(matches)))
            continue

        segment = matches[0]

        spk_segment = segment.spk_segment
        t_start, t_end = segment.time_range(ts)

        segment_tdb_start = float(spk_segment.start_jd)
        segment_tdb_end = float(spk_segment.end_jd)
        segment_tt_start = float(t_start.tt)
        segment_tt_end = float(t_end.tt)

        if tdb_start is None:
            tdb_start = segment_tdb_start
            tdb_end = segment_tdb_end
            tt_start = segment_tt_start
            tt_end = segment_tt_end
        else:
            tdb_start = max(tdb_start, segment_tdb_start)
            tdb_end = min(tdb_end, segment_tdb_end)
            tt_start = max(tt_start, segment_tt_start)
            tt_end = min(tt_end, segment_tt_end)

    if missing:
        raise SunsetChronologyError(
            REASON_KERNEL_SEGMENTS_UNAVAILABLE,
            "KERNEL SEGMENTS UNAVAILABLE - %s does not publish every solar "
            "segment a topocentric sunset requires; missing %s. No sunset "
            "can be determined from this kernel." % (kernel_name, list(missing)),
        )

    if ambiguous:
        raise SunsetChronologyError(
            REASON_KERNEL_SEGMENTS_UNAVAILABLE,
            "KERNEL SEGMENTS UNAVAILABLE - %s represents a required solar "
            "segment more than once %s. Which segment applies is decided by "
            "the certified ephemeris machinery per epoch, not by a single "
            "coverage interval, so this reader refuses to report bounds "
            "rather than guess." % (kernel_name, sorted(ambiguous)),
        )

    if not tt_start < tt_end:
        raise SunsetChronologyError(
            REASON_KERNEL_SEGMENTS_UNAVAILABLE,
            "KERNEL SEGMENTS UNAVAILABLE - the required solar segments of %s "
            "do not share a non-empty interval; intersection TT %r .. %r"
            % (kernel_name, tt_start, tt_end),
        )

    return KernelCoverage(
        kernel=kernel_name,
        tdb_start=tdb_start,
        tdb_end=tdb_end,
        tt_start=tt_start,
        tt_end=tt_end,
    )


# ---------------------------------------------------------------------------
# A2-2 - data-authoritative kernel selection.
#
# Answers one question: which pinned NASA/JPL artifact has governed
# precedence for a given exact TT state or interval under its actual
# certified BSP coverage? The answer comes from actual
# certified BSP coverage under a fixed governed precedence, and from
# nothing else. No civil year, no Gregorian date, no UTC year, no timezone
# and no nominal duration participates.
#
# SCOPE OF THE CLAIM
#
# The selector determines precedence among pinned authoritative NASA/JPL
# artifacts according to declared certified coverage. It does not certify
# that every later computation is evaluable throughout that interval.
#
# Declared coverage and computation-specific evaluability are different
# statements. A computation that resolves light time reads the ephemeris
# before the instant it is asked about, so a state at an artifact's exact
# lower bound is declared-covered yet cannot serve such a computation.
# That reach is a property of the computation, not of the data. It is
# deliberately not modelled here, and no margin, lookback or slack of any
# size is applied. Handling it belongs to the solver.
#
# In the opposite direction the certified stack may evaluate beyond a
# segment's declared end without raising. The containment test below is
# therefore the binding upper guard, and an ephemeris range error must
# never be relied upon to supply one.
# ---------------------------------------------------------------------------

# Fixed, order-sensitive scientific precedence.
#
# DE440 is preferred wherever its actual authoritative BSP coverage
# contains the required TT state or interval; DE441 supplies extended and
# deep-time declared coverage outside DE440.
#
# JPL states inside the kernels themselves that DE441 is less accurate
# than DE440 for the current century, so this ordering is the product
# architecture rather than an HPC convention.
PINNED_KERNEL_PRECEDENCE = (PRIMARY_KERNEL, ANCIENT_KERNEL, FUTURE_KERNEL)


def select_kernel_containing_instant(tt):
    """Return the selected pinned NASA/JPL artifact whose coverage contains ``tt``.

    Selection evaluates the fixed governed precedence and returns the
    first pinned artifact whose certified coverage contains the state.
    Containment is closed and inclusive, compared exactly against the
    certified bounds. No bound is widened or narrowed.

    This reports declared certified coverage. It does not certify that a
    later computation is evaluable at the instant.

    Returns None when no pinned artifact covers the instant. That is a
    coverage-query result, not a request failure: nothing is fabricated,
    and None cannot be mistaken for an artifact name.
    """
    instant = _exact_finite_tt(tt, "instant tt")

    for kernel_name in PINNED_KERNEL_PRECEDENCE:
        coverage = kernel_coverage_tt(kernel_name)
        if coverage.tt_start <= instant <= coverage.tt_end:
            return kernel_name

    return None


def select_kernel_for_interval(tt_lo, tt_hi):
    """Return the selected pinned NASA/JPL artifact containing the whole interval.

    Selection evaluates the same fixed governed precedence. The entire
    closed interval must lie inside one artifact's certified coverage: a
    state covered only by stitching two artifacts together is not returned
    here, because no single pinned artifact declares coverage for the
    whole interval.

    A zero-width interval is accepted and agrees with
    select_kernel_containing_instant. A reversed interval is malformed
    input and fails closed; it is not reported as absent coverage.

    This reports declared certified coverage. It does not certify that a
    later computation is evaluable throughout the interval.

    Returns None when no pinned artifact contains the whole interval.
    """
    lo = _exact_finite_tt(tt_lo, "interval start tt")
    hi = _exact_finite_tt(tt_hi, "interval end tt")

    if lo > hi:
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SUNSET CHRONOLOGY INTERVAL INVALID - the interval start must "
            "not follow the interval end, received TT %r .. %r" % (lo, hi),
        )

    for kernel_name in PINNED_KERNEL_PRECEDENCE:
        coverage = kernel_coverage_tt(kernel_name)
        if coverage.tt_start <= lo and hi <= coverage.tt_end:
            return kernel_name

    return None


# ---------------------------------------------------------------------------
# A2-3a - supported directional search frontier.
#
# WHAT THIS OPERATION ADDS
#
# The ability to ask, for one directional search, which pinned NASA/JPL
# artifact can actually support it and over exactly what interval. Nothing
# else. No event is solved, no crossing is selected, no ordering rule is
# applied and no route is served; each of those is a separate governed
# operation.
#
# TWO QUESTIONS, KEPT SEPARATE
#
# A2-1 and A2-2 answer the first question: does authoritative NASA/JPL data
# DECLARE coverage for the required state or interval? They deliberately
# make no claim about the second: can the actual topocentric solar
# computation be EVALUATED there?
#
# The two are not the same statement, and neither one alone is sufficient.
#
#   Declared coverage is not sufficient. earth.at(t).observe(sun) resolves
#   light time, so it reads the Sun, and the deflecting bodies, at t minus
#   the one-way light time. A state at an artifact's exact declared start is
#   therefore declared-covered while the observation cannot be formed there.
#
#   Evaluating without raising is not sufficient either. The certified
#   computational stack has been observed to return evaluations beyond
#   declared BSP coverage rather than refusing them, so a successful
#   computation cannot substitute for explicit declared-coverage
#   containment. Silence above a declared bound is evidence of nothing, and
#   the declared containment test below - not an ephemeris range error - is
#   the binding upper guard.
#
# Support is consequently defined as the conjunction: an artifact must
# DECLARE the whole bracket and the actual certified predicate must
# EVALUATE at the bracket's endpoints. Neither half is trusted alone.
#
# NO MARGIN IS ENCODED
#
# The lower reach is the converged one-way light time from the Sun at that
# epoch. It is a physical quantity that varies with the Earth-Sun distance,
# not a constant, and it differs between artifacts because their declared
# starts fall at different points in the orbit. No fixed lookback, slack,
# safety margin or second-count of any size appears in this block. The
# reach is never modelled and never predicted: it is established by
# performing the actual computation and observing whether the certified
# machinery can complete it.
#
# CONTIGUITY
#
# Exactly one bracket is returned, anchored at the caller's anchor state and
# examined by exactly one artifact. There is no second bracket, no
# resumption and no stitching, so no unsupported temporal gap can be
# crossed, skipped, or resumed beyond. The certified root finder refines by
# convex combination of existing grid points and therefore samples nowhere
# outside the bracket it is given, so a bracket proven supported at its
# endpoints bounds the entire examined frontier.
#
# The anchor is never moved. A frontier may stop short of the requested
# horizon, but it always still touches the anchor.
#
# A TRUNCATED FRONTIER IS NOT AN OUTCOME
#
# When authoritative or evaluable territory ends before the requested
# horizon, the frontier is returned SHORT rather than refused, carrying
# complete=False and the reason its territory ran out. Whether that is a
# complete answer or an exhaustion report is not decided here: an event
# found inside a truncated frontier was reached across continuously examined
# authoritative territory and is a complete answer to the nearest-event
# question, while an empty truncated frontier is an exhaustion report and
# must never be presented as an absence of sunset. That determination
# belongs to the solver, which is a separate governed operation.
#
# complete=True is the only state from which a caller may conclude that the
# entire requested directional horizon was authoritatively covered and
# actually evaluable.
# ---------------------------------------------------------------------------

REASON_EPHEMERIS_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_EPHEMERIS_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"

# The governed geodetic observer domain, stated here as scientific input to
# the topocentric solar computation.
#
# It is deliberately NOT imported from sunset_cursor: that module owns a
# transport and serialization domain and sits above this one, and importing
# it here would invert the layering by making the solver depend on the
# continuation-witness encoding. The two domains must agree, and a later
# certification operation asserts that they do rather than one silently
# deriving from the other.
OBSERVER_LATITUDE_DOMAIN = (-90.0, 90.0)
OBSERVER_LONGITUDE_DOMAIN = (-180.0, 180.0)


@dataclass(frozen=True)
class SupportedSearchFrontier:
    """The interval one directional search may actually be performed over.

    ``kernel`` is the pinned NASA/JPL artifact that both declares the
    interval and was observed to evaluate the certified topocentric solar
    computation across it.

    ``tt_lo`` and ``tt_hi`` are the frontier in ascending Terrestrial Time,
    ready to bound a search directly. Ascending order is a property of the
    record, not a statement about direction: which endpoint is the anchor is
    known to the caller that supplied it, and the anchor is never moved.

    ``complete`` is True only when the frontier is the entire requested
    horizon. False means authoritative or evaluable territory ended first
    and the frontier was shortened to where it genuinely ends - the anchor
    side is unchanged and the examined territory remains contiguous from it.

    ``truncation_reason`` is None when ``complete`` is True, and otherwise
    carries the stable code describing why the territory ran out, so a
    caller that finds no event inside a shortened frontier can report that
    exhaustion faithfully instead of reporting an absence of sunset.

    Named fields, deliberately not a tuple: no call site can unpack a
    frontier positionally, so a later change to field order cannot silently
    transpose the bounds or invert the completeness flag.
    """

    kernel: str
    tt_lo: float
    tt_hi: float
    complete: bool
    truncation_reason: str | None


def _governed_observer(value, field, domain):
    """Accept only a real, finite observer coordinate inside its domain.

    An observer coordinate is not an instant state, so it never reports
    INSTANT_STATE_INVALID. It has its own stable reason.

    This validation is a precondition of the failure taxonomy rather than
    ordinary input hygiene. A non-finite latitude reaches the ephemeris as
    an invalid cast and surfaces as an ephemeris range error - the exact
    signal this module reads as exhausted computational reach - so an
    unvalidated observer would be reported as a scientific limit of the
    NASA/JPL data. A latitude outside the geodetic domain is worse: it
    computes silently and would be reported as a genuine absence of sunset.

    bool is rejected before the numeric check because it is a subclass of
    int and would otherwise be silently accepted as 0.0 or 1.0.

    Signed zero is deliberately NOT canonicalized. That rule belongs to the
    continuation witness, where it makes a serialized observer binding
    unique; -0.0 and +0.0 are the same coordinate to the geometry here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SunsetChronologyError(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            "SUNSET CHRONOLOGY OBSERVER OUT OF DOMAIN - %s must be a real "
            "number, received %s" % (field, type(value).__name__),
        )

    try:
        number = float(value)
    except OverflowError as error:
        raise SunsetChronologyError(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            "SUNSET CHRONOLOGY OBSERVER OUT OF DOMAIN - %s is too large to "
            "represent as a float" % field,
        ) from error

    if not math.isfinite(number):
        raise SunsetChronologyError(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            "SUNSET CHRONOLOGY OBSERVER OUT OF DOMAIN - %s must be finite, "
            "received %r" % (field, value),
        )

    low, high = domain
    if number < low or number > high:
        raise SunsetChronologyError(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            "SUNSET CHRONOLOGY OBSERVER OUT OF DOMAIN - %s must lie "
            "inclusively within [%r, %r], received %r"
            % (field, low, high, number),
        )

    return number


def _observation_is_evaluable(state_probe, tt):
    """Report whether the certified computation actually completes at ``tt``.

    ``state_probe`` is the certified computation itself, as a callable of a
    Time that raises an ephemeris range error exactly where it cannot be
    formed. Which computation that is belongs to the caller: this helper
    reports evaluability and asserts nothing about what is being evaluated.

    The question is answered by performing that real computation, not by
    predicting it. Only an ephemeris range error is treated as an answer: it
    is the certified machinery reporting that a required record lies outside
    the published coefficients, which is conclusive proof that the
    observation cannot be formed.

    Nothing else is caught. Any other failure is not a statement about
    ephemeris support and must propagate rather than be recorded here as
    absent reach.

    A True result is never used on its own. Every caller has already
    required the state to lie inside declared certified coverage, because
    the certified stack has been observed to return evaluations beyond
    declared coverage rather than refusing them, so completing without
    raising above a declared bound proves nothing.
    """
    try:
        state_probe(ts.tt_jd(tt))
    except EphemerisRangeError:
        return False

    return True


def _first_evaluable_state(state_probe, unevaluable_tt, evaluable_tt):
    """Return the exact earliest TT state at which the computation completes.

    ``state_probe`` is the certified computation whose reach is being
    located, as a callable of a Time that raises an ephemeris range error
    exactly where it cannot be formed. The boundary this returns belongs to
    that computation and to no other: two computations that read the
    ephemeris differently have different reaches, and one may not be used to
    certify the other.

    PRECONDITION, required of every caller:

        unevaluable_tt < evaluable_tt

    ``unevaluable_tt`` must be a state the computation was ACTUALLY OBSERVED
    to fail at, and ``evaluable_tt`` a later state it was ACTUALLY OBSERVED
    to complete at. Neither is assumed, predicted or modelled. Calling this
    with the temporal or evaluability roles reversed asks a question the
    evidence does not support and is not permitted; each call site
    establishes both roles by observation immediately beforehand.

    The transition is located by interrogating the actual computation,
    halving the interval until no representable binary64 value lies strictly
    between the two ends. Termination is exhaustion of the representation
    itself: there is no epsilon, no tolerance, no convergence threshold, no
    iteration limit and no second-count, because once no representable state
    remains between them there is nothing left to examine.

    The result is exact rather than approximate. The returned state is
    evaluable and its immediate binary64 predecessor is not, so no evaluable
    state is discarded, and the answer does not depend on how wide the
    starting interval was.

    This is sound because the failing region is a contiguous prefix. A read
    fails on the low side only when it falls below the segment start, and the
    deepest read trails the requested state by the one-way solar light time,
    which changes far more slowly than the state itself. The requested state
    minus that lookback is therefore strictly increasing, so evaluability
    never resumes and never lapses again.
    """
    bad = unevaluable_tt
    good = evaluable_tt

    while True:
        mid = bad + (good - bad) / 2.0

        if mid == bad or mid == good:
            return good

        if _observation_is_evaluable(state_probe, mid):
            good = mid
        else:
            bad = mid


def _admit_bracket(tt_lo, tt_hi, latitude, longitude):
    """Return the artifact that both declares and can evaluate the bracket.

    The governed precedence is evaluated in order and BOTH conditions are
    required of each candidate. An artifact that declares the bracket but
    cannot evaluate the computation at its endpoints does not end the scan:
    the search continues to the next artifact, which may be able to. That
    is a real and reachable case, not a defensive branch - a bracket
    beginning at DE440's exact declared start is declared by DE440 and
    cannot be evaluated under it, while DE441 part 1 declares the same
    bracket and evaluates it. Stopping at the first declaring artifact would
    refuse a fully supported search.

    The declared containment test is exact, closed and inclusive, and
    matches select_kernel_for_interval. It is restated rather than reused
    because that function returns only the FIRST declaring artifact and
    offers no way to continue past one, which is precisely what this scan
    must do.

    WHY TWO ENDPOINT PROBES CERTIFY THE WHOLE BRACKET

    Under the certified runtime and pinned artifact set, every ephemeris
    read the certified predicate makes at an observation state falls between
    that state's own instant and that instant less the one-way solar light
    time; the deflection reads are bounded to the same depth and go no
    deeper. Both ends of that read window advance with the observation
    state, because the light time changes by a fraction of a second per day
    while the state advances by a day per day. The deepest read over an
    interval is therefore made at its lower endpoint and the shallowest at
    its upper endpoint, so an interval whose endpoints both evaluate has no
    interior state that reads outside what those two already proved. The
    certified root finder samples only inside the interval it is given, and
    evaluates through the same path as these probes, so what is proven here
    is what the search will actually require.

    This is a property of the certified computational environment, not a
    timeless one. A different runtime or artifact set could read
    differently, so it is verified rather than assumed.

    Returns None when no pinned artifact supports the whole bracket. That is
    a support-query result, not a request failure: nothing is fabricated,
    and None cannot be mistaken for an artifact name.
    """
    location = wgs84.latlon(latitude, longitude)

    for kernel_name in PINNED_KERNEL_PRECEDENCE:
        coverage = kernel_coverage_tt(kernel_name)

        if not (coverage.tt_start <= tt_lo and tt_hi <= coverage.tt_end):
            continue

        is_sun_up = almanac.sunrise_sunset(load_kernel(kernel_name), location)

        if (_observation_is_evaluable(is_sun_up, tt_lo)
                and _observation_is_evaluable(is_sun_up, tt_hi)):
            return kernel_name

    return None


def supported_search_frontier(anchor_tt, horizon_tt, latitude, longitude):
    """Return the frontier a directional sunset search may be performed over.

    ``anchor_tt`` is the exact binary64 Terrestrial Time state the search
    starts from and is never moved: the returned frontier always touches it,
    so examined territory is contiguous from the anchor. ``horizon_tt`` is
    the exact TT state the search would like to reach. Direction is derived
    from their order rather than declared, so a backward frontier cannot be
    requested with the endpoints transposed. Both are exact TT states; no
    civil year, Gregorian date, timezone or nominal duration participates.

    An anchor equal to its horizon has no direction and fails closed as
    malformed interval state.

    Two outcomes are returned rather than raised:

        complete=True   the entire requested horizon is declared by one
                        artifact and the computation evaluates across it
        complete=False  authoritative or evaluable territory ended first;
                        the frontier reaches where it genuinely ends and
                        carries the reason

    Two outcomes fail closed:

        EPHEMERIS_COVERAGE_EXHAUSTED  no pinned artifact declares any
                                      directional territory beyond the
                                      anchor
        EPHEMERIS_REACH_EXHAUSTED     an artifact declares the territory,
                                      but the certified topocentric solar
                                      computation cannot be evaluated from
                                      the anchor onward

    The second is deliberately not reported as inconsistent kernel coverage.
    The BSP metadata is not inconsistent - the segments are exactly what JPL
    published. The limit belongs to a computation that resolves light time,
    not to the data.

    No HTTP semantics are decided here.
    """
    anchor = _exact_finite_tt(anchor_tt, "anchor tt")
    horizon = _exact_finite_tt(horizon_tt, "horizon tt")
    latitude = _governed_observer(
        latitude, "latitude", OBSERVER_LATITUDE_DOMAIN
    )
    longitude = _governed_observer(
        longitude, "longitude", OBSERVER_LONGITUDE_DOMAIN
    )

    if anchor == horizon:
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SUNSET CHRONOLOGY FRONTIER INVALID - the anchor and the horizon "
            "are the same state, TT %r, so the requested frontier has no "
            "direction" % (anchor,),
        )

    backward = horizon < anchor
    requested_lo = horizon if backward else anchor
    requested_hi = anchor if backward else horizon

    kernel_name = _admit_bracket(
        requested_lo, requested_hi, latitude, longitude
    )

    if kernel_name is not None:
        return SupportedSearchFrontier(
            kernel=kernel_name,
            tt_lo=requested_lo,
            tt_hi=requested_hi,
            complete=True,
            truncation_reason=None,
        )

    holder = select_kernel_containing_instant(anchor)

    if holder is None:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - no pinned NASA/JPL artifact "
            "declares coverage for the anchor state TT %r, so no directional "
            "search frontier exists there" % (anchor,),
        )

    coverage = kernel_coverage_tt(holder)

    # Shortened to the authoritative bound, never widened past what was
    # requested. The anchor side is untouched.
    if backward:
        frontier_lo = max(requested_lo, coverage.tt_start)
        frontier_hi = requested_hi
        collapsed = not frontier_lo < anchor
    else:
        frontier_lo = requested_lo
        frontier_hi = min(requested_hi, coverage.tt_end)
        collapsed = not anchor < frontier_hi

    if collapsed:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - the declared coverage of %s ends "
            "at the anchor state TT %r, so no authoritative territory exists "
            "in the requested direction" % (holder, anchor),
        )

    kernel_name = _admit_bracket(
        frontier_lo, frontier_hi, latitude, longitude
    )

    if kernel_name is not None:
        return SupportedSearchFrontier(
            kernel=kernel_name,
            tt_lo=frontier_lo,
            tt_hi=frontier_hi,
            complete=False,
            truncation_reason=REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
        )

    if select_kernel_for_interval(frontier_lo, frontier_hi) is None:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - no pinned NASA/JPL artifact "
            "declares the whole frontier TT %r .. %r"
            % (frontier_lo, frontier_hi),
        )

    # The frontier is declared, but the computation could not be evaluated
    # across all of it.
    #
    # Exactly one recovery is authorized, and only for the geometry the
    # evidence established: a BACKWARD frontier whose far endpoint sits in
    # the unevaluable prefix above an artifact's declared start. There the
    # far state lies strictly below the anchor, the far state is observed
    # unevaluable, the anchor is observed evaluable, and one artifact
    # declares the whole interval - so a first evaluable state exists
    # strictly between them and is located exactly rather than estimated.
    # Territory that is genuinely authoritative and genuinely evaluable is
    # kept instead of being discarded.
    #
    # Nothing else is recovered, and the temporal roles are never reversed.
    # A forward request whose anchor is itself unevaluable has no supported
    # territory beginning at that anchor, and the anchor is never moved, so
    # it falls through and fails closed below.
    if backward:
        location = wgs84.latlon(latitude, longitude)

        for kernel_name in PINNED_KERNEL_PRECEDENCE:
            coverage = kernel_coverage_tt(kernel_name)

            if not (coverage.tt_start <= frontier_lo
                    and frontier_hi <= coverage.tt_end):
                continue

            is_sun_up = almanac.sunrise_sunset(
                load_kernel(kernel_name), location
            )

            if _observation_is_evaluable(is_sun_up, frontier_lo):
                continue

            if not _observation_is_evaluable(is_sun_up, anchor):
                continue

            return SupportedSearchFrontier(
                kernel=kernel_name,
                tt_lo=_first_evaluable_state(is_sun_up, frontier_lo, anchor),
                tt_hi=frontier_hi,
                complete=False,
                truncation_reason=REASON_EPHEMERIS_REACH_EXHAUSTED,
            )

    raise SunsetChronologyError(
        REASON_EPHEMERIS_REACH_EXHAUSTED,
        "EPHEMERIS REACH EXHAUSTED - the frontier TT %r .. %r is declared by "
        "a pinned NASA/JPL artifact, but the certified topocentric solar "
        "computation cannot be evaluated from the anchor state TT %r onward, "
        "and the anchor is not moved" % (frontier_lo, frontier_hi, anchor),
    )


# ---------------------------------------------------------------------------
# A2-3b - strict sunset predecessor.
#
# WHAT THIS OPERATION ADDS
#
# One astronomical question: what is the latest genuine observer-local
# astronomical topocentric apparent sunset strictly before an arbitrary exact
# Terrestrial Time anchor? Nothing else. No successor direction, no route, no
# transport representation and no HTTP semantics are decided here.
#
# The anchor is arbitrary. It need not be a sunset, need not be
# post-transition, and carries no proof of which side of a crossing it lies
# on. That is why this is a separate question from the frozen A1b successor,
# which continues from a state already proven to sit on the far side of a
# crossing and is neither modified nor consulted here.
#
# WHAT IT DELEGATES
#
# Which pinned NASA/JPL artifact may answer, over exactly what interval, is
# not decided here. A2-3b asks A2-3a for a backward supported frontier and
# searches inside exactly what it is given, using exactly the artifact it
# names. Observer validation, declared certified coverage, computation
# evaluability, governed precedence, anchor immobility and the contiguity of
# the examined territory are all owned by that substrate. None of it is
# re-derived, re-checked against a second artifact, or worked around.
#
# STRICT ORDERING
#
# A crossing qualifies only when its exact binary64 TT is strictly less than
# the anchor. One comparison decides it. There is no epsilon, no tolerance,
# no minimum gap, no event-identity rule, no nearest-event heuristic and no
# nominal-day arithmetic.
#
# A crossing exactly equal to the anchor belongs to neither direction and is
# excluded. This is not a special case in the code: the same strict
# comparison covers it. An anchor that is itself a previously determined
# sunset root is re-reported by the certified root finder as exactly that
# root, so it fails the comparison and the sunset before it is returned.
#
# Roots found under different search brackets may differ in their final
# bits. That is an inherited numerical property of the certified root finder,
# not something this block repairs; repairing it would require an
# event-identity tolerance, which is precisely what is forbidden.
#
# ABSENCE IS NOT EXHAUSTION
#
# None means the entire requested directional horizon was authoritatively
# covered, actually evaluable, and contained no qualifying sunset. It is a
# scientific answer.
#
# A frontier that stopped short and contained no qualifying sunset is not
# that answer, and must never be reported as one. It raises the reason its
# territory ran out. A qualifying sunset found inside a shortened frontier IS
# a complete answer to the nearest-event question, because the territory from
# the anchor through that sunset was examined continuously.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SunsetEvent:
    """A determined sunset and the artifact that actually produced it.

    ``tt`` is float(t.tt) read directly from the Skyfield Time object the
    certified root finder returned. It is never derived from a datetime, ISO
    text, Unix seconds or milliseconds, so it is the exact state the crossing
    was found at.

    ``kernel`` is the pinned NASA/JPL artifact the search actually ran under,
    as selected from actual certified coverage and evaluability. It is
    computation provenance, not a routing decision.

    No transport representation is carried, deliberately. Terrestrial Time is
    the astronomical state; a calendar rendering is a presentation concern
    belonging to a later layer, and one that cannot always be formed - the
    standard library cannot represent a datetime for the deep-time events
    this solver legitimately returns, so a datetime field would make a
    genuine answer impossible to construct. ts.tt_jd(event.tt) reconstructs
    the exact Time whenever a representation is needed.

    Named fields, deliberately not a tuple: no call site can unpack an event
    positionally, so a later change to field order cannot silently transpose
    the state and its provenance.
    """

    tt: float
    kernel: str


def find_sunset_predecessor(tt, latitude, longitude):
    """Return the latest genuine sunset strictly before an arbitrary anchor.

    ``tt`` is an exact binary64 Terrestrial Time state. It is arbitrary: it
    need not be a sunset and need not lie on any particular side of a
    crossing.

    Returns a SunsetEvent, or None when the entire requested directional
    horizon was authoritatively supported and contained no sunset - the
    genuine polar outcome.

    Fails closed, preserving the substrate's own stable reason, when the
    anchor state is malformed, the observer lies outside the governed
    geodetic domain, or authoritative coverage or computational reach ends
    before the horizon without a qualifying sunset having been found first.

    No HTTP semantics are decided here.
    """
    anchor = _exact_finite_tt(tt, "anchor tt")

    frontier = supported_search_frontier(
        anchor, anchor - SUCCESSOR_SEARCH_SPAN_DAYS, latitude, longitude
    )

    is_sun_up = almanac.sunrise_sunset(
        load_kernel(frontier.kernel), wgs84.latlon(latitude, longitude)
    )

    # The frontier was admitted by probing this same computation at both of
    # its endpoints, so a failure in here is not expected. If one occurs the
    # examined territory is no longer whole, and the only honest response is
    # to fail closed: the frontier is never abandoned for another artifact,
    # never resumed past the failure, and never stitched to a second
    # interval, because any of those would answer a different question.
    try:
        times, events = almanac.find_discrete(
            ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi), is_sun_up
        )
    except EphemerisRangeError as error:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_REACH_EXHAUSTED,
            "EPHEMERIS REACH EXHAUSTED - the certified topocentric solar "
            "computation failed inside the supported frontier TT %r .. %r "
            "under %s; the examined territory is not whole, so no sunset is "
            "reported"
            % (frontier.tt_lo, frontier.tt_hi, frontier.kernel),
        ) from error

    latest = None

    for t, sun_is_up in zip(times, events):
        if bool(sun_is_up):
            continue

        event_tt = float(t.tt)

        if event_tt < anchor:
            latest = event_tt

    if latest is not None:
        return SunsetEvent(tt=latest, kernel=frontier.kernel)

    if frontier.complete:
        return None

    raise SunsetChronologyError(
        frontier.truncation_reason,
        "SUNSET PREDECESSOR UNRESOLVED - no sunset lies strictly before the "
        "anchor state TT %r inside the supported frontier TT %r .. %r, and "
        "that frontier stopped short of the requested horizon, so the "
        "absence of a sunset is not established"
        % (anchor, frontier.tt_lo, frontier.tt_hi),
    )


# ---------------------------------------------------------------------------
# A2-3c - strict arbitrary-instant sunset successor.
#
# WHAT THIS OPERATION ADDS
#
# One astronomical question: what is the earliest genuine observer-local
# astronomical topocentric apparent sunset strictly after an arbitrary exact
# Terrestrial Time anchor? It is the directional mirror of A2-3b and answers
# the forward half of the same chronology.
#
# WHY THIS IS NOT THE FROZEN A1b SUCCESSOR
#
# A1b continues from a state already proven to lie on the far side of a
# known crossing: it validates that the Sun is down at the supplied state
# and refuses otherwise, because a continuation witness that is not
# post-transition does not describe the far side of a sunset. That
# precondition is what makes its search safe, and it is also what makes it a
# different question from this one.
#
# The anchor here is arbitrary. It may fall anywhere inside a local
# sunset-bounded day, including while the Sun is up, where A1b correctly
# refuses to continue. A2-3c therefore inherits none of A1b's machinery: no
# minimum-gap rule, no continuation-witness semantics, no post-transition
# assumption, no civil-year kernel selection and no route behavior. A1b is
# neither modified nor consulted.
#
# WHAT IT DELEGATES
#
# Which pinned NASA/JPL artifact may answer, and over exactly what interval,
# is not decided here. A2-3c asks A2-3a for a forward supported frontier and
# searches inside exactly what it is given, under exactly the artifact it
# names. Observer validation, declared certified coverage, computation
# evaluability, governed precedence, anchor immobility and the contiguity of
# the examined territory are all owned by that substrate. None of it is
# re-derived, re-checked against a second artifact, or worked around.
#
# STRICT ORDERING
#
# A crossing qualifies only when its exact binary64 TT is strictly greater
# than the anchor. One comparison decides it. There is no epsilon, no
# tolerance, no minimum gap, no event-identity rule, no nearest-event
# heuristic and no nominal-day arithmetic.
#
# A crossing exactly equal to the anchor belongs to neither direction and is
# excluded by that same comparison. An anchor that is itself a determined
# sunset root needs no special handling at all: the certified root finder
# returns the upper end of its converged bracket, at which the Sun is
# already down, so the crossing that produced such an anchor presents no
# sign change inside this frontier and cannot be rediscovered. The sunset
# after it is returned.
#
# The certified root finder reports crossings in ascending time, so the
# first qualifying crossing is the earliest one.
#
# ABSENCE IS NOT EXHAUSTION
#
# None means the entire requested directional horizon was authoritatively
# covered, actually evaluable, and contained no qualifying sunset. It is a
# scientific answer.
#
# A frontier that stopped short and contained no qualifying sunset is not
# that answer and raises the reason its territory ran out. A qualifying
# sunset found inside a shortened frontier IS a complete answer to the
# nearest-event question, because the territory from the anchor through that
# sunset was examined continuously.
# ---------------------------------------------------------------------------


def find_sunset_from_instant(tt, latitude, longitude):
    """Return the earliest genuine sunset strictly after an arbitrary anchor.

    ``tt`` is an exact binary64 Terrestrial Time state. It is arbitrary: it
    need not be a sunset, and unlike a continuation witness it need not lie
    on the far side of one.

    Returns a SunsetEvent, or None when the entire requested directional
    horizon was authoritatively supported and contained no sunset - the
    genuine polar outcome.

    Fails closed, preserving the substrate's own stable reason, when the
    anchor state is malformed, the observer lies outside the governed
    geodetic domain, or authoritative coverage or computational reach ends
    before the horizon without a qualifying sunset having been found first.

    No HTTP semantics are decided here.
    """
    anchor = _exact_finite_tt(tt, "anchor tt")

    frontier = supported_search_frontier(
        anchor, anchor + SUCCESSOR_SEARCH_SPAN_DAYS, latitude, longitude
    )

    is_sun_up = almanac.sunrise_sunset(
        load_kernel(frontier.kernel), wgs84.latlon(latitude, longitude)
    )

    # The frontier was admitted by probing this same computation at both of
    # its endpoints, so a failure in here is not expected. If one occurs the
    # examined territory is no longer whole, and the only honest response is
    # to fail closed: the frontier is never abandoned for another artifact,
    # never resumed past the failure, and never stitched to a second
    # interval, because any of those would answer a different question.
    try:
        times, events = almanac.find_discrete(
            ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi), is_sun_up
        )
    except EphemerisRangeError as error:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_REACH_EXHAUSTED,
            "EPHEMERIS REACH EXHAUSTED - the certified topocentric solar "
            "computation failed inside the supported frontier TT %r .. %r "
            "under %s; the examined territory is not whole, so no sunset is "
            "reported"
            % (frontier.tt_lo, frontier.tt_hi, frontier.kernel),
        ) from error

    for t, sun_is_up in zip(times, events):
        if bool(sun_is_up):
            continue

        event_tt = float(t.tt)

        if event_tt > anchor:
            return SunsetEvent(tt=event_tt, kernel=frontier.kernel)

    if frontier.complete:
        return None

    raise SunsetChronologyError(
        frontier.truncation_reason,
        "SUNSET SUCCESSOR UNRESOLVED - no sunset lies strictly after the "
        "anchor state TT %r inside the supported frontier TT %r .. %r, and "
        "that frontier stopped short of the requested horizon, so the "
        "absence of a sunset is not established"
        % (anchor, frontier.tt_lo, frontier.tt_hi),
    )


# ---------------------------------------------------------------------------
# A2-4 - atomic sunset bracket.
#
# WHAT THIS OPERATION ADDS
#
# One composed astronomical answer: the pair of genuine observer-local
# sunset boundaries that surround an arbitrary exact Terrestrial Time
# state.
#
#     previous sunset  <  anchor  <  next sunset
#
# It performs no astronomy. Both boundaries come from the published strict
# directional solvers, which own every scientific decision between them -
# observer validation, declared certified coverage, computation
# evaluability, governed precedence, anchor immobility, contiguity of the
# examined territory, and the exact binary64 ordering that makes each side
# strict. Nothing here re-solves, re-checks or second-guesses any of it,
# and no third astronomical solution exists in this block.
#
# STILL ASTRONOMY, NOT CALENDAR
#
# This identifies the local astronomical sunset boundaries surrounding an
# instant. It decides no weekday, no ordinal, no month or day ownership, no
# annual grid, no year and no feast date. Those are separate questions for
# a later governed layer, and none of them is implied here.
#
# ONE ANCHOR, ONE OBSERVER, TWO PROVENANCES
#
# The anchor is validated once and the resulting exact value is handed to
# both directions, so the two sides are answered for a bit-identical state
# rather than for two states that merely look alike. The same observer is
# passed to both.
#
# The two sides are NOT required to share an artifact. Each direction
# selects from actual certified coverage and evaluability independently, so
# an anchor lying within a directional reach of a coverage boundary
# legitimately yields one boundary from one pinned NASA/JPL artifact and
# the other from another. Each event keeps its own provenance and the
# bracket asserts no single-artifact ownership of the interval between
# them.
#
# ABSENCE IS NOT FAILURE, AND FAILURE IS NOT ABSENCE
#
# Both directions are computed unconditionally, before any conclusion is
# drawn. A governed failure from either side propagates unchanged: if a
# direction could not be completed then this composed question was not
# completed either, and saying so is the honest answer.
#
# None is returned only when BOTH directions completed scientifically and
# at least one of them established that no sunset exists inside its fully
# supported horizon. At high latitude that is the ordinary, correct answer
# rather than an error: an instant inside a polar day or polar night is not
# bounded by sunsets within the certified directional horizon, so no
# bracket exists to report. A failure is never converted into that claim.
#
# EXACT-ROOT ANCHORS
#
# An anchor that is itself a determined sunset root is not special-cased.
# The published directional contracts exclude a crossing exactly equal to
# the anchor from both directions, so such an anchor is bracketed by the
# surrounding pair. That is the definition working, not an edge case to
# repair, and introducing an event-identity rule to change it is precisely
# what is forbidden.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SunsetBracket:
    """The two genuine sunset boundaries surrounding an exact TT state.

    ``anchor`` is the exact binary64 Terrestrial Time state the bracket was
    established for, carried through unaltered - never via a datetime, ISO
    text, Unix seconds or milliseconds. It is part of the record because it
    makes the claim self-contained and checkable: previous.tt < anchor <
    next.tt can be verified from the record alone, and a bracket cannot be
    silently reused for an instant it does not contain.

    ``previous`` and ``next`` each carry their own artifact provenance.
    They may name different pinned NASA/JPL artifacts when the anchor lies
    near a coverage boundary, and that is correct rather than a defect: no
    bracket-level kernel is recorded, because no single artifact owns the
    interval in that case.

    The observer is deliberately absent. Binding an observer into a
    transportable record is the continuation witness's responsibility, not
    this one's.

    Named fields, deliberately not a tuple: no call site can unpack a
    bracket positionally, so a later change to field order cannot silently
    transpose the two boundaries or the state they surround.
    """

    anchor: float
    previous: SunsetEvent
    next: SunsetEvent


def find_sunset_bracket(tt, latitude, longitude):
    """Return the sunset boundaries surrounding an arbitrary exact anchor.

    ``tt`` is an exact binary64 Terrestrial Time state. It is arbitrary: it
    need not be a sunset and need not lie on any particular side of one.

    Returns a SunsetBracket, or None when both directions completed
    scientifically and at least one of them found no sunset inside its
    fully supported horizon - the genuine polar outcome, where an instant
    is simply not bounded by sunsets within the certified directional
    horizon.

    Fails closed by propagating the directional solvers' own stable
    reasons, unchanged, when the anchor state is malformed, the observer
    lies outside the governed geodetic domain, or authoritative coverage or
    computational reach ran out before a boundary could be established. No
    governed failure is caught or reinterpreted here, and none is ever
    converted into an absence.

    No HTTP semantics are decided here.
    """
    anchor = _exact_finite_tt(tt, "anchor tt")

    # Both directions are computed before anything is concluded. A failure
    # on either side propagates from here unchanged, so an incomplete
    # direction can never be reported as an absence of sunset.
    previous = find_sunset_predecessor(anchor, latitude, longitude)
    next_sunset = find_sunset_from_instant(anchor, latitude, longitude)

    if previous is None or next_sunset is None:
        return None

    return SunsetBracket(
        anchor=anchor, previous=previous, next=next_sunset
    )


# ---------------------------------------------------------------------------
# A3a - exact astronomical event record.
#
# WHAT THIS OPERATION ADDS
#
# The record an astronomical event instant is carried in, and the governed
# identities for the four solar-longitude crossings. Substrate only: nothing
# is computed, nothing is validated, nothing is served, and no existing
# caller is migrated onto it.
#
# WHY A RECORD AT ALL
#
# An event that leaves this module as formatted text is no longer the event
# that was computed. Rendering a crossing to whole seconds moves it by up to
# half a second, and the text cannot reconstruct the state it came from, so
# a downstream consumer handed only that text is holding a different instant
# and has no way to know it. The exact Terrestrial Time of the root is the
# scientific state; a calendar rendering is a view of it, produced later and
# elsewhere, and never the authority.
#
# TT IS SUFFICIENT
#
# Barycentric Dynamical Time is recoverable from Terrestrial Time through
# the certified timescale, so carrying it here would duplicate derived data
# rather than preserve source data. That is the opposite of the coverage
# substrate, which preserves raw JD(TDB) because BSP segment metadata is
# published in that scale and is genuinely the source there. An event is a
# computed root whose native output is TT.
#
# NO TRANSPORT REPRESENTATION
#
# No datetime, ISO text, Unix value or civil field appears. The standard
# library cannot represent a datetime for the deep-time events this
# Authority legitimately determines, so a transport field would make a
# genuine event impossible to construct. ts.tt_jd(event.tt) reconstructs
# the exact Time whenever a representation is wanted.
#
# IDENTITIES CARRY NO CALENDAR AND NO HEMISPHERE
#
# The four crossings are identified by the apparent geocentric solar
# ecliptic longitude that defines them. Season names are hemisphere
# conventions - the 180 degree crossing opens spring in Sydney and autumn in
# New York - and month names are Gregorian. Neither can be the identity used
# by a data-first astronomical authority, so neither is used here.
# ---------------------------------------------------------------------------

SOLAR_LONGITUDE_000 = "SOLAR_LONGITUDE_000"
SOLAR_LONGITUDE_090 = "SOLAR_LONGITUDE_090"
SOLAR_LONGITUDE_180 = "SOLAR_LONGITUDE_180"
SOLAR_LONGITUDE_270 = "SOLAR_LONGITUDE_270"


@dataclass(frozen=True)
class AstronomicalEvent:
    """A determined astronomical instant and what produced it.

    ``tt`` is the exact binary64 Terrestrial Time of the root, read directly
    from the Skyfield Time object the certified search returned. It is never
    derived from a datetime, ISO text, Unix seconds or milliseconds, so it is
    the state the event was actually found at rather than a rendering of it.
    It is the authoritative event state for this record.

    ``kind`` is one of the governed solar-longitude identities above. It
    states the astronomical condition the crossing satisfies and carries no
    hemisphere, month, year or other calendar meaning.

    ``kernel`` is the pinned NASA/JPL artifact the determination actually ran
    under. It is computation provenance, not a routing decision.

    Named fields, deliberately not a tuple: no call site can unpack an event
    positionally, so a later change to field order cannot silently transpose
    the state, its identity and its provenance.
    """

    tt: float
    kind: str
    kernel: str


# ---------------------------------------------------------------------------
# A3b-i - geocentric supported search frontier.
#
# WHAT THIS OPERATION ADDS
#
# The ability to ask, for one directional interval, which pinned NASA/JPL
# artifact can actually support the geocentric solar-longitude computation
# and over exactly what contiguous interval. Nothing else. No crossing is
# solved, no event kind is accepted, no horizon is defined and no route is
# served; each of those is a separate governed operation.
#
# WHY THIS CANNOT BORROW THE SUNSET FRONTIER
#
# The topocentric sunset computation and the geocentric solar-longitude
# computation do not have interchangeable scientific support, and that was
# measured rather than assumed. Each artifact's first evaluable state
# differs between the two computations by a few milliseconds, and the
# direction of the difference is not constant: under DE440 the geocentric
# boundary is the lower of the two, while under both DE441 parts it is the
# higher. A frontier admitted for the sunset computation can therefore
# contain states at which this computation raises, and the reverse is also
# possible. Support certified for one is not support for the other.
#
# So the sunset frontier is neither reused, wrapped nor called here, and no
# observer is invented in order to borrow it. The observer is not merely
# unnecessary for this question - it is absent from it. What IS shared is
# the observer-free coverage and precedence substrate, which answers the
# same declared-data question for both.
#
# TWO QUESTIONS, KEPT SEPARATE
#
# Declared coverage and computation-specific evaluability remain distinct.
# Neither is sufficient alone. A state at an artifact's declared start is
# declared-covered yet cannot be computed there, because this computation
# resolves light time and reads the Sun before the instant it is asked
# about. And the certified stack has been observed to return evaluations
# beyond declared coverage rather than refusing them, so a successful
# computation can never substitute for explicit declared containment.
# Support is the conjunction of both, and nothing here relies on either
# half by itself.
#
# NO MARGIN IS ENCODED
#
# The lower reach is a physical quantity that varies with the Earth-Sun
# distance and differs between artifacts. No fixed lookback, slack or
# second-count of any size appears in this block; the reach is established
# by performing the actual computation, and where it must be located
# exactly it is located by the neutral binary64 boundary search rather than
# estimated.
#
# CONTIGUITY
#
# Exactly one interval is returned, anchored at the caller's anchor state
# and examined by exactly one artifact. There is no second interval, no
# resumption and no stitching, so no unsupported temporal gap can be
# crossed or skipped. The anchor is never moved; only the far side may
# shorten.
#
# A SHORTENED FRONTIER IS NOT AN OUTCOME
#
# When authoritative or evaluable territory ends before the requested
# horizon, the frontier is returned SHORT rather than refused, carrying
# complete=False and the reason its territory ran out. Whether that is a
# complete answer or an exhaustion report is not decided here; that belongs
# to the solver that consumes it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeocentricSearchFrontier:
    """The interval one geocentric solar-longitude search may run over.

    ``kernel`` is the pinned NASA/JPL artifact that both declares the
    interval and was observed to evaluate the geocentric solar-longitude
    computation at its endpoints.

    ``tt_lo`` and ``tt_hi`` are the frontier in ascending Terrestrial Time,
    ready to bound a search directly. Ascending order is a property of the
    record, not a statement about direction: which endpoint is the anchor is
    known to the caller that supplied it, and the anchor is never moved.

    ``complete`` is True only when the frontier is the entire requested
    horizon. False means authoritative or evaluable territory ended first
    and the frontier was shortened to where it genuinely ends.

    ``truncation_reason`` is None when ``complete`` is True, and otherwise
    carries the stable code describing why the territory ran out.

    This is deliberately NOT the sunset frontier record, and the two are not
    interchangeable. They certify different computations whose measured
    reaches differ, so a value of one type must never be accepted where the
    other is required. Keeping them distinct makes that structural rather
    than a matter of documentation.

    Named fields, deliberately not a tuple: no call site can unpack a
    frontier positionally, so a later change to field order cannot silently
    transpose the bounds or invert the completeness flag.
    """

    kernel: str
    tt_lo: float
    tt_hi: float
    complete: bool
    truncation_reason: str | None


def _admit_geocentric_bracket(tt_lo, tt_hi):
    """Return the artifact that declares and can compute the whole bracket.

    The governed precedence is evaluated in order and BOTH conditions are
    required of each candidate. An artifact that declares the bracket but
    cannot evaluate this computation at its endpoints does not end the scan:
    the search continues to the next artifact, which may be able to.

    The declared containment test is exact, closed and inclusive, and
    matches select_kernel_for_interval. It is restated rather than reused
    because that function returns only the FIRST declaring artifact and
    offers no way to continue past one, which is precisely what this scan
    must do.

    Evaluability is decided by invoking the real geocentric computation
    through the neutral probe, so what is certified here is the same
    callable a later search will actually run. No observer participates,
    because none appears in the computation.

    WHY TWO ENDPOINT PROBES CERTIFY THE WHOLE BRACKET

    Under the certified runtime and pinned artifact set, every ephemeris
    read this computation makes at an observation state falls between that
    state's own instant and that instant less the one-way solar light time;
    the deflection reads are bounded to the same depth and go no deeper.
    Both ends of that read window advance with the observation state,
    because the light time changes by a fraction of a second per day while
    the state advances by a day per day. The deepest read over an interval
    is therefore made at its lower endpoint and the shallowest at its upper
    endpoint, so an interval whose endpoints both evaluate has no interior
    state that reads outside what those two already proved.

    This is a property of the certified computational environment, not a
    timeless one. A different runtime or artifact set could read
    differently, so it is verified rather than assumed.

    Returns None when no pinned artifact supports the whole bracket. That is
    a support-query result, not a request failure: nothing is fabricated,
    and None cannot be mistaken for an artifact name.
    """
    for kernel_name in PINNED_KERNEL_PRECEDENCE:
        coverage = kernel_coverage_tt(kernel_name)

        if not (coverage.tt_start <= tt_lo and tt_hi <= coverage.tt_end):
            continue

        season_at = almanac.seasons(load_kernel(kernel_name))

        if (_observation_is_evaluable(season_at, tt_lo)
                and _observation_is_evaluable(season_at, tt_hi)):
            return kernel_name

    return None


def geocentric_search_frontier(anchor_tt, horizon_tt):
    """Return the frontier a geocentric solar-longitude search may run over.

    ``anchor_tt`` is the exact binary64 Terrestrial Time state the search
    starts from and is never moved: the returned frontier always touches it,
    so examined territory is contiguous from the anchor. ``horizon_tt`` is
    the exact TT state the search would like to reach. Direction is derived
    from their order rather than declared, so a backward frontier cannot be
    requested with the endpoints transposed. Both are exact TT states; no
    civil year, Gregorian date, timezone or nominal duration participates,
    and no horizon is assumed - the caller supplies it.

    An anchor equal to its horizon has no direction and fails closed as
    malformed interval state.

    Two outcomes are returned rather than raised:

        complete=True   the entire requested horizon is declared by one
                        artifact and this computation evaluates across it
        complete=False  authoritative or evaluable territory ended first;
                        the frontier reaches where it genuinely ends and
                        carries the reason

    Two outcomes fail closed:

        EPHEMERIS_COVERAGE_EXHAUSTED  no pinned artifact declares any
                                      directional territory beyond the
                                      anchor
        EPHEMERIS_REACH_EXHAUSTED     an artifact declares the territory,
                                      but this computation cannot be
                                      evaluated across it

    The second is deliberately not reported as inconsistent kernel coverage.
    The BSP metadata is not inconsistent - the segments are exactly what JPL
    published. The limit belongs to a computation that resolves light time,
    not to the data.

    No observer is accepted, no event kind is accepted, and no HTTP
    semantics are decided here.
    """
    anchor = _exact_finite_tt(anchor_tt, "anchor tt")
    horizon = _exact_finite_tt(horizon_tt, "horizon tt")

    if anchor == horizon:
        raise SunsetChronologyError(
            REASON_INSTANT_STATE_INVALID,
            "SEARCH FRONTIER INVALID - the anchor and the horizon are the "
            "same state, TT %r, so the requested frontier has no direction"
            % (anchor,),
        )

    backward = horizon < anchor
    requested_lo = horizon if backward else anchor
    requested_hi = anchor if backward else horizon

    kernel_name = _admit_geocentric_bracket(requested_lo, requested_hi)

    if kernel_name is not None:
        return GeocentricSearchFrontier(
            kernel=kernel_name,
            tt_lo=requested_lo,
            tt_hi=requested_hi,
            complete=True,
            truncation_reason=None,
        )

    holder = select_kernel_containing_instant(anchor)

    if holder is None:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - no pinned NASA/JPL artifact "
            "declares coverage for the anchor state TT %r, so no directional "
            "search frontier exists there" % (anchor,),
        )

    coverage = kernel_coverage_tt(holder)

    # Shortened to the authoritative bound, never widened past what was
    # requested. The anchor side is untouched.
    if backward:
        frontier_lo = max(requested_lo, coverage.tt_start)
        frontier_hi = requested_hi
        collapsed = not frontier_lo < anchor
    else:
        frontier_lo = requested_lo
        frontier_hi = min(requested_hi, coverage.tt_end)
        collapsed = not anchor < frontier_hi

    if collapsed:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - the declared coverage of %s ends "
            "at the anchor state TT %r, so no authoritative territory exists "
            "in the requested direction" % (holder, anchor),
        )

    kernel_name = _admit_geocentric_bracket(frontier_lo, frontier_hi)

    if kernel_name is not None:
        return GeocentricSearchFrontier(
            kernel=kernel_name,
            tt_lo=frontier_lo,
            tt_hi=frontier_hi,
            complete=False,
            truncation_reason=REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
        )

    if select_kernel_for_interval(frontier_lo, frontier_hi) is None:
        raise SunsetChronologyError(
            REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            "EPHEMERIS COVERAGE EXHAUSTED - no pinned NASA/JPL artifact "
            "declares the whole frontier TT %r .. %r"
            % (frontier_lo, frontier_hi),
        )

    # The frontier is declared, but this computation could not be evaluated
    # across all of it.
    #
    # Exactly one recovery is authorized, and only for the geometry the
    # evidence establishes: a BACKWARD frontier whose far endpoint sits in
    # the unevaluable prefix above an artifact's declared start. There the
    # far state lies strictly below the anchor, the far state is observed
    # unevaluable, the anchor is observed evaluable, and one artifact
    # declares the whole interval - so a first evaluable state exists
    # strictly between them and is located exactly rather than estimated.
    # Territory that is genuinely authoritative and genuinely evaluable is
    # kept instead of being discarded.
    #
    # The boundary is located by the neutral binary64 search, handed this
    # computation's own probe. That matters: the boundary it returns belongs
    # to the computation it was given and to no other, which is why the
    # sunset frontier's boundary could not have been borrowed.
    #
    # Nothing else is recovered, and the temporal roles are never reversed.
    # A forward request whose anchor is itself unevaluable has no supported
    # territory beginning at that anchor, and the anchor is never moved, so
    # it falls through and fails closed below.
    if backward:
        for kernel_name in PINNED_KERNEL_PRECEDENCE:
            coverage = kernel_coverage_tt(kernel_name)

            if not (coverage.tt_start <= frontier_lo
                    and frontier_hi <= coverage.tt_end):
                continue

            season_at = almanac.seasons(load_kernel(kernel_name))

            if _observation_is_evaluable(season_at, frontier_lo):
                continue

            if not _observation_is_evaluable(season_at, anchor):
                continue

            return GeocentricSearchFrontier(
                kernel=kernel_name,
                tt_lo=_first_evaluable_state(
                    season_at, frontier_lo, anchor
                ),
                tt_hi=frontier_hi,
                complete=False,
                truncation_reason=REASON_EPHEMERIS_REACH_EXHAUSTED,
            )

    raise SunsetChronologyError(
        REASON_EPHEMERIS_REACH_EXHAUSTED,
        "EPHEMERIS REACH EXHAUSTED - the frontier TT %r .. %r is declared by "
        "a pinned NASA/JPL artifact, but the geocentric solar-longitude "
        "computation cannot be evaluated from the anchor state TT %r onward, "
        "and the anchor is not moved" % (frontier_lo, frontier_hi, anchor),
    )
