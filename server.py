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
    ANCIENT_KERNEL,
    EPHEMERIS_DIR,
    FUTURE_KERNEL,
    PRIMARY_KERNEL,
    SunsetChronologyError,
    SunsetSuccessorError,
    count_sunsets_in_interval,
    determine_solar_regime,
    solar_longitude,
    subsolar_point,
    find_equinox,
    find_night_start_after,
    find_season_events,
    find_solar_crossing,
    find_sunset_utc,
    find_next_sunset_after_utc,
    find_solar_longitude_event_after,
    find_solar_longitude_event_before,
    find_solar_longitude_event_in_year,
    find_sunset_bracket,
    find_sunset_from_instant,
    find_sunset_successor,
    get_default_kernel_name,
    get_delta_t,
)
from astronomical_event_transport import (  # noqa: E402 - gate runs first
    project_astronomical_event,
    project_night_start,
    project_solar_crossing,
    project_solar_regime,
    project_sunset_bracket,
    project_sunset_count,
    project_sunset_event,
    reason_detail,
)
from exact_time_transport import (  # noqa: E402 - gate runs first
    ExactTimeTransportError,
    decode_tt_bits,
)
from sunset_cursor import (  # noqa: E402 - deliberate: gate runs first
    LATITUDE_DOMAIN,
    LONGITUDE_DOMAIN,
    SunsetCursorError,
    decode_sunset_cursor,
    encode_sunset_cursor,
)
from scientific_provenance import (  # noqa: E402 - gate runs first
    establish_scientific_provenance,
    scientific_provenance,
)
from earth_rotation import (  # noqa: E402 - deliberate: gate runs first
    EarthRotationError,
    earth_rotation_evidence,
)
from civil_instant import (  # noqa: E402 - deliberate: gate runs first
    REASON_CIVIL_INSTANT_INVALID,
    CivilInstantError,
    resolve_civil_instant,
)

# Establish this process's scientific identity before it can serve anything.
#
# THE SECOND STARTUP GATE, AND IT IS DELIBERATELY HERE RATHER THAN BESIDE THE
# FIRST. The A0.3 gate above runs before any astronomy is imported, because an
# uncertified runtime must never build a timescale. This one cannot: the
# canonical ephemeris manifest binds each routing role to the bytes of the
# artifact it routes to, and the routing names belong to the astronomy module
# imported above. So it runs at the earliest point where the material it
# describes exists, and still before the application object below and
# therefore before any route can be reached.
#
# The two gates answer different questions. A0.3 asks whether this runtime is
# the certified one and refuses to start if not. This asks what this process
# actually IS, and records the answer. Neither substitutes for the other.
#
# Roughly nineteen seconds of that is hashing the pinned NASA/JPL artifact set,
# and it is paid once, on purpose. An Authority that cannot state its own
# scientific identity must not answer scientific questions, so a failure here
# raises out of import and the process does not start. There is no degraded
# mode and no partially initialised provenance endpoint.
establish_scientific_provenance(
    ephemeris_dir=EPHEMERIS_DIR,
    routing_names={
        "ancient": ANCIENT_KERNEL,
        "future": FUTURE_KERNEL,
        "primary": PRIMARY_KERNEL,
    },
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


@app.get("/sunset-successor")
def sunset_successor(cursor: str, latitude: float, longitude: float):
    """Continue a sunset sequence from a validated continuation witness.

    Additive. /sunset and /sunset-after are unchanged: no existing field
    is renamed, removed or reinterpreted, and no existing status behavior
    is altered.

    The witness is decoded and validated by A1a, and the exact binary64
    Terrestrial Time state it carries is handed to the A1b solver
    directly. No astronomy is performed here, and the continuation state
    is never routed through a datetime, an ISO string or a Unix value.

    The observer that governs the search is the one the witness is bound
    to. The request coordinates are supplied to the decoder so that a
    cursor presented for a different observer fails closed rather than
    silently answering a different question.

    Every A1a and A1b fail-closed rejection is reported as HTTP 400 with
    its stable reason code in the detail. A checksum mismatch is NOT an
    authentication or authorization outcome - the checksum detects
    accidental corruption only - so 401 and 403 are deliberately unused.

    HTTP 404 reports that the bound observer has no sunset inside the
    certified successor horizon. That is an astronomical absence, not a
    rejected request.

    The legacy one-hour guard in find_next_sunset_after_utc governs
    /sunset-after and remains that route's business. It is not reused,
    not extended and not removed here, and this route introduces no gap,
    epsilon or event-identity tolerance of its own.
    """
    try:
        decoded = decode_sunset_cursor(
            cursor,
            latitude=latitude,
            longitude=longitude,
        )
        successor = find_sunset_successor(
            decoded.tt,
            decoded.latitude,
            decoded.longitude,
        )
    except (SunsetCursorError, SunsetSuccessorError) as error:
        raise HTTPException(
            status_code=400,
            detail="%s: %s" % (error.reason, error),
        )

    if successor is None:
        raise HTTPException(
            status_code=404,
            detail="Sunset successor not found",
        )

    return {
        "latitude": decoded.latitude,
        "longitude": decoded.longitude,
        "kernel": successor.kernel,
        "sunsetUTC": successor.utc.isoformat(),
        **cursor_fields(
            successor.tt,
            decoded.latitude,
            decoded.longitude,
        ),
    }


# ---------------------------------------------------------------------------
# A3c-1 - exact solar-longitude event routes.
#
# WHAT THESE ROUTES ADD
#
# The first HTTP surface for the exact astronomical substrate. Two questions,
# already answered by the published solver: which governed solar-longitude
# crossing is the nearest one strictly before an exact Terrestrial Time state,
# and which is the nearest one strictly after it.
#
# NO ASTRONOMY HAPPENS HERE
#
# Neither route searches, selects an artifact, validates an event kind or
# decides anything scientific. Each decodes an exact state, hands it to the
# published solver, and projects whatever comes back. Every scientific
# decision - the frontier, the artifact, the sampling, the canonical event
# identity, the strict ordering, the fail-closed taxonomy - stays where it was
# certified, and none of it is re-derived, re-checked or worked around.
#
# THE ANCHOR IS BITS, NOT A DECIMAL
#
# The anchor is supplied as ttBits, the IEEE-754 spelling of an exact binary64
# state. A decimal query parameter would be a different contract: it would
# invite a client to round, reformat or re-parse the value under its own rule
# and silently ask about a different instant. There is deliberately no tt
# parameter, no civil year, no Gregorian date, no ISO instant and no observer
# - none of those can address an exact astronomical state, and three of them
# cannot address most of the authoritative domain at all.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a TRANSPORT failure: the exact state was never
# received, so no scientific question was asked and none was refused. It
# reports the transport reason. A state that WAS received and then refused by
# the solver reports the solver's own stable reason, unchanged. Collapsing the
# two would tell a caller that the Authority rejected an instant it never
# actually had.
#
# Only those two exception types are caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# NO ABSENCE OUTCOME EXISTS
#
# The published solver never returns None: a complete horizon always contains
# the requested crossing, and a shortened one fails closed with the reason its
# territory ran out. There is therefore no 404 here and no empty success -
# every request either yields an event or reports why it could not.
#
# ADDITIVE
#
# No existing route is touched. The structured detail below is introduced for
# these two routes only; every published route keeps its own error shape.
# ---------------------------------------------------------------------------


@app.get("/solar-longitude-event-before")
def solar_longitude_event_before(ttBits: str, kind: str):
    """Return the nearest governed crossing strictly before an exact state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial
    Time anchor. ``kind`` is one of the governed solar-longitude identities;
    it is passed to the solver unaltered and is not second-guessed here.

    The anchor is arbitrary: it need not be a crossing, and when it is one it
    is excluded by the solver's strict ordering.

    Returns the exact event projection. Fails closed with HTTP 400 carrying a
    stable reason, either for a malformed anchor or for the solver's own
    governed refusal. There is no 404 and no empty success.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        event = find_solar_longitude_event_before(anchor, kind)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_astronomical_event(event)


@app.get("/solar-longitude-event-after")
def solar_longitude_event_after(ttBits: str, kind: str):
    """Return the nearest governed crossing strictly after an exact state.

    The directional mirror of the route above, and identical in every respect
    except which published solver answers it. The two are written out
    separately rather than sharing a direction argument, because the scientific
    API they expose has no direction parameter and neither should its
    transport.

    Returns the exact event projection. Fails closed with HTTP 400 carrying a
    stable reason, either for a malformed anchor or for the solver's own
    governed refusal. There is no 404 and no empty success.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        event = find_solar_longitude_event_after(anchor, kind)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_astronomical_event(event)


# ---------------------------------------------------------------------------
# A3c-2 - exact sunset bracket route.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for the published atomic sunset bracket: the two genuine
# observer-local sunset boundaries surrounding an exact Terrestrial Time
# state, with the state they surround.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route decodes an exact anchor, hands it and the observer to the
# published bracket function, and projects whatever comes back. It runs no
# search, selects no artifact, stitches nothing, and re-derives none of the
# scientific decisions - the directional frontiers, the artifact selection,
# the strict ordering, the anchor immobility - which stay where they were
# certified.
#
# TWO PROVENANCES, NOT ONE
#
# The bracket's two boundaries are determined independently and may genuinely
# have been produced under different pinned artifacts when the anchor lies
# near a coverage boundary. That is preserved: each boundary reports its own
# kernel, and no bracket-level artifact is invented. The anchor reports none
# at all, because the caller supplied it and no computation produced it.
#
# ABSENCE IS AN ANSWER, NOT A FAILURE
#
# At high latitude an instant inside a polar day or polar night is simply not
# bounded by sunsets within the certified directional horizon. The published
# function returns that as an absence rather than an error, and it is
# reported here as 404 - the same treatment /sunset-successor already gives
# an astronomical absence, and distinct from the 400 that reports a request
# the Authority refused to answer.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a transport failure: the exact state was never
# received. Anything the bracket function refuses - a malformed anchor, an
# observer outside the governed geodetic domain, exhausted coverage or
# exhausted computational reach - reports that function's own stable reason,
# unchanged. Only those two exception types are caught; an unexpected failure
# stays visible.
#
# ADDITIVE
#
# No existing route is touched, and the observer domain, its validation and
# its stable reason all remain the substrate's.
# ---------------------------------------------------------------------------

REASON_SUNSET_BRACKET_ABSENT = "SUNSET_BRACKET_ABSENT"


@app.get("/sunset-bracket")
def sunset_bracket(ttBits: str, latitude: float, longitude: float):
    """Return the sunset boundaries surrounding an exact anchor state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial
    Time anchor. It is arbitrary: it need not be a sunset and need not lie on
    any particular side of one. ``latitude`` and ``longitude`` are the
    observer, validated by the published substrate against its own governed
    geodetic domain and not re-validated or normalized here.

    Returns the exact bracket projection: the anchor, and the two surrounding
    sunsets each with its own artifact provenance.

    Fails closed with HTTP 400 carrying a stable reason, either for a
    malformed anchor or for the substrate's own governed refusal. Reports
    HTTP 404 when the observer genuinely has no sunset boundaries around that
    instant, which is an astronomical absence rather than a rejected request.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        bracket = find_sunset_bracket(anchor, latitude, longitude)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    if bracket is None:
        raise HTTPException(
            status_code=404,
            detail=reason_detail(
                REASON_SUNSET_BRACKET_ABSENT,
                "SUNSET BRACKET ABSENT - the bound observer has no sunset "
                "boundaries surrounding the anchor state inside the certified "
                "directional horizon; the instant is not bracketed by sunsets "
                "there",
            ),
        )

    return project_sunset_bracket(bracket)


# ---------------------------------------------------------------------------
# A3c-3 - year-addressed exact solar-longitude event route.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for the published year-addressed operation: which governed
# solar-longitude crossing the Authority determines for a given astronomical
# year, returned as the exact event projection every other exact route
# already emits.
#
# It is the one addressing form a consumer that has no exact Terrestrial Time
# state can use. Without it such a consumer would have to construct a TT
# state itself, which is scientific timescale work and is this Authority's.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route hands the year and the kind to the published operation and
# projects whatever comes back. It forms no addressing state, runs no search,
# selects no artifact, validates no event kind, decides no domain and
# re-derives none of the scientific decisions, all of which stay where they
# were certified.
#
# THE YEAR IS AN ADDRESS, NOT AN INSTANT
#
# ``year`` selects which crossing is meant and is never an event time. It is
# an astronomical-year integer - 1 is 1 CE, 0 is 1 BCE, -1 is 2 BCE - and it
# is deliberately the ONLY temporal parameter. There is no month, day, hour,
# date, ISO instant, timezone or decimal TT parameter, because this route
# addresses a year and must not become a civil-time conversion surface. There
# is no observer, because the crossing is geocentric, and no kernel, because
# artifact selection is the substrate's.
#
# The addressing year is not echoed back. The response is the exact event and
# only the exact event: a consumer that received its own input alongside the
# result could mistake the address for the answer.
#
# THIS IS NOT THE LEGACY EQUINOX ROUTE
#
# /equinox/{year} is frozen, published and untouched. It renders at whole
# seconds and applies its own +1 shift to years at or below zero. This route
# is additive, exact, and uses astronomical year numbering without that
# shift. The two agree on which crossing is meant for every year above zero
# and diverge at or below zero; neither claim is left implicit, and both are
# certified.
#
# ONE FAILURE SHAPE
#
# No ttBits is decoded here, so there is no transport failure to keep apart
# from a scientific one. Every governed refusal - a year the timescale cannot
# express, an ungoverned event kind, or a year the pinned artifacts do not
# support - is the substrate's own, reported as HTTP 400 with its stable
# reason unchanged.
#
# Only SunsetChronologyError is caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# There is no absence outcome and therefore no 404: the published operation
# never returns None, so every request either yields an event or reports why
# it could not.
#
# ADDITIVE
#
# No existing route is touched.
# ---------------------------------------------------------------------------


@app.get("/solar-longitude-event-in-year")
def solar_longitude_event_in_year(year: int, kind: str):
    """Return the governed crossing of ``kind`` addressed by a year.

    ``year`` is an astronomical-year integer: 1 is 1 CE, 0 is 1 BCE, -1 is
    2 BCE. It is an addressing selector for a geocentric crossing, not an
    HPC/SCE year and not part of the response. ``kind`` is one of the
    governed solar-longitude identities; it is passed to the published
    operation unaltered and is not second-guessed here.

    Returns the exact event projection: the canonical Terrestrial Time state
    of the crossing, its governed identity, and the artifact provenance of
    the search that determined it. The ``utc`` field is reference
    information and is null wherever the instant lies outside the span a
    calendar datetime can express.

    Fails closed with HTTP 400 carrying the substrate's stable reason. There
    is no 404 and no empty success.
    """
    try:
        event = find_solar_longitude_event_in_year(year, kind)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_astronomical_event(event)


# ---------------------------------------------------------------------------
# A3c-4 - exact sunset successor route.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for one published directional question: which genuine
# observer-local sunset is the earliest one strictly later than an exact
# Terrestrial Time state.
#
# It is the continuation primitive the exact layer was missing. Every other
# way to continue a sunset sequence across this Authority either resumes from
# a rendered timestamp or from an opaque witness; this one resumes from the
# exact state itself, so a consumer walking a sequence never leaves exact
# scientific state and never has to hold anything else.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route decodes an exact anchor, hands it and the observer to the
# published directional solver, and projects whatever comes back. It runs no
# search, selects no artifact, stitches nothing, and re-derives none of the
# scientific decisions - the supported frontier, the artifact selection, the
# strict binary64 ordering, the anchor immobility - which stay where they
# were certified.
#
# Exactly one directional search is performed. The atomic bracket is
# deliberately NOT used: it determines a preceding boundary as well, and a
# caller asking which sunset comes next has not asked for that one. Answering
# a narrower question with a wider computation would discard half of every
# result and would make a sequence walk cost twice what the question does.
#
# STRICTLY LATER, AND NOTHING ELSE
#
# The contract is the earliest genuine sunset STRICTLY AFTER the supplied
# state. It is not the nearest sunset, not the sunset whose day contains the
# instant, and not the sunset falling on any civil date. An anchor that is
# itself a determined sunset root is not special-cased: the published solver
# excludes it by strict ordering, and the certification establishes that the
# result is the FOLLOWING physical sunset from crossing topology rather than
# from any comparison between two determined roots.
#
# THE ANCHOR IS BITS, NOT A DECIMAL
#
# The anchor is supplied as ttBits, the IEEE-754 spelling of an exact
# binary64 state. A decimal query parameter would be a different contract: it
# would invite a client to round, reformat or re-parse the value under its
# own rule and silently ask about a different instant. There is deliberately
# no tt parameter, no UTC, no civil date, no civil year, no kernel selector,
# no direction control and no continuation witness.
#
# The anchor is not returned. The atomic bracket returns its anchor because
# the claim it makes - previous < anchor < next - is only checkable with it;
# a single directional result makes no such claim, so echoing the caller's
# own input back would add a field nothing reads.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a TRANSPORT failure: the exact state was never
# received, so no scientific question was asked and none was refused. It
# reports the transport reason. A state that WAS received and then refused by
# the substrate - a malformed instant, an observer outside the governed
# geodetic domain, exhausted coverage or exhausted computational reach -
# reports the substrate's own stable reason, unchanged.
#
# Only those two exception types are caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# ABSENCE IS AN ANSWER, NOT A FAILURE
#
# At high latitude an instant inside a polar day or polar night is simply not
# followed by a sunset within the certified directional horizon. The
# published solver returns that as an absence rather than an error, and only
# when the examined frontier was complete, so it can never be confused with
# territory that ran out. It is reported here as 404 - the same treatment
# /sunset-successor and /sunset-bracket already give an astronomical absence,
# and distinct from the 400 that reports a request the Authority refused to
# answer.
#
# SCIENTIFIC ENVIRONMENT
#
# This route is stateless, exactly as the rest of the exact layer is. It
# carries no environment fingerprint, because the certified runtime is
# verified once at startup before any astronomy is loaded, and a request
# answered by this process is answered under that verified runtime.
#
# A consumer performing a MULTI-REQUEST walk across a restart or redeploy is
# a different matter: consistency of the scientific environment across that
# walk is a deployment-provenance concern and is NOT established by this
# route or by any other route in the exact layer. It must be addressed before
# final production acceptance. It is deliberately not addressed here, because
# a per-response generation field would change a published projection shape
# to carry a property that belongs to the deployment rather than to the
# event.
#
# ADDITIVE
#
# No existing route is touched, and the observer domain, its validation and
# its stable reason all remain the substrate's.
# ---------------------------------------------------------------------------

REASON_SUNSET_EVENT_ABSENT = "SUNSET_EVENT_ABSENT"


@app.get("/sunset-event-after")
def sunset_event_after(ttBits: str, latitude: float, longitude: float):
    """Return the earliest genuine sunset strictly after an exact state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial
    Time anchor. It is arbitrary: it need not be a sunset and need not lie on
    any particular side of one. ``latitude`` and ``longitude`` are the
    observer, validated by the published substrate against its own governed
    geodetic domain and not re-validated or normalized here.

    Returns the exact sunset projection: the state of the crossing and the
    artifact provenance of the search that determined it. ``utc`` is
    reference information and is null wherever the instant lies outside the
    span a calendar datetime can express.

    Fails closed with HTTP 400 carrying a stable reason, either for a
    malformed anchor or for the substrate's own governed refusal. Reports
    HTTP 404 when the observer genuinely has no sunset after that instant
    inside the certified directional horizon, which is an astronomical
    absence rather than a rejected request.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        event = find_sunset_from_instant(anchor, latitude, longitude)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    if event is None:
        raise HTTPException(
            status_code=404,
            detail=reason_detail(
                REASON_SUNSET_EVENT_ABSENT,
                "SUNSET EVENT ABSENT - the supplied observer has no sunset "
                "strictly after the anchor state inside the certified "
                "directional horizon; the instant is not followed by a "
                "sunset there",
            ),
        )

    return project_sunset_event(event)


# ---------------------------------------------------------------------------
# A4 - ABC-1 bulk sunset count route.
#
# The marker above obeys the repository's governed block grammar
# (^# A\d+[a-z]?(?:-[0-9a-z]+)? - ). That is load-bearing: the published route
# source guards delimit each block from its own marker to the NEXT one, so a
# terminal block written under an unrecognised marker would silently be read
# as part of the block before it.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for one published question: how many genuine
# observer-local sunset transitions occur inside an exact Terrestrial Time
# interval. The answer is an integer.
#
# It exists because the only way to obtain that integer across this Authority
# today is to walk the directional successor route once per sunset. For a
# year-scale interval that is roughly three hundred and sixty-six requests,
# three hundred and sixty-six frontier admissions and three hundred and
# sixty-six searches, to produce one number. This asks the same certified
# question once.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route decodes two exact bounds, hands them and the observer to the
# published count solver, and projects whatever comes back. It runs no
# search, selects no artifact, admits no frontier, applies no resource bound
# of its own and re-derives none of the scientific decisions - the supported
# frontier, the artifact selection, the half-open interval contract, the
# governed span admission, the strict binary64 comparisons - which all stay
# where they were certified.
#
# Exactly one solver invocation. There is no loop over sunsets here, no
# accumulation, no second call and no retry: a count assembled by this layer
# from several answers would be a different quantity than the one the solver
# certifies, and nothing in this block could establish that it was right.
#
# HOW MANY, NEVER WHICH
#
# The response carries an integer, the interval it belongs to, the artifact
# that produced it and nothing else. No crossing, no root, no array and no
# individual event identity is published, because the solver does not retain
# any. A consumer needing a boundary asks a directional route for it.
#
# No calendar meaning is produced either: no weekday, no ordinal, no month,
# no year, no year type, no 365/366 classification. This is an astronomy
# count primitive and it stays one.
#
# THE BOUNDS ARE BITS, NOT DECIMALS
#
# Both bounds are supplied as ttBits, the IEEE-754 spelling of an exact
# binary64 state, exactly as every other route in the exact layer requires. A
# decimal query parameter would be a different contract: it would invite a
# client to round, reformat or re-parse the value under its own rule and
# silently ask about a different interval. There is deliberately no tt
# parameter, no UTC, no civil date, no civil year and no kernel selector.
#
# BOTH BOUNDS ARE RETURNED, AND SO IS WHAT WAS ACTUALLY COVERED
#
# Unlike a single directional result, a count MAKES A CLAIM ABOUT AN
# INTERVAL, so the interval is part of the claim and is echoed. The covered
# pair is published alongside it because the two can legitimately differ: a
# frontier shortened by exhausted coverage answers for less territory than
# was asked about, and reporting only the requested pair would attribute to
# the whole interval a number belonging to part of it.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a TRANSPORT failure: the exact state was never
# received, so no scientific question was asked and none was refused. It
# reports the transport reason. A state that WAS received and then refused by
# the substrate - a malformed instant, a non-ascending interval, a span
# beyond the governed admission bound, an observer outside the geodetic
# domain, exhausted coverage or exhausted computational reach - reports the
# substrate's own stable reason, unchanged.
#
# Only those two exception types are caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# The governed span refusal is projected exactly like every other
# substrate-domain refusal, at 400 with its stable reason. It is a statement
# that the Authority declines to answer a question of that size, which is the
# same category of answer as declining a malformed instant, and inventing a
# second HTTP policy for it would make one governed refusal look unlike the
# rest for no scientific reason.
#
# ZERO IS AN ANSWER, NOT AN ABSENCE
#
# A complete frontier containing no sunset counts zero and returns 200. This
# deliberately differs from the directional routes, which report 404 for an
# astronomical absence: they were asked WHICH sunset and there is none, while
# this was asked HOW MANY and the answer exists and is zero. Reporting that
# as an absence would discard a successful measurement, and would make a
# genuine polar-night count indistinguishable from a failure to obtain one.
#
# An INCOMPLETE frontier is likewise not an error. It is a smaller true
# answer, and it is returned with complete=false and the substrate's own
# truncation reason so a consumer can tell the two apart. Nothing here
# promotes a partial result to a whole one, discards it, fabricates coverage,
# stitches a second artifact or retries to manufacture completeness.
#
# SCIENTIFIC ENVIRONMENT
#
# This route is stateless, exactly as the rest of the exact layer is. It
# carries no environment fingerprint, because the certified runtime is
# verified once at startup before any astronomy is loaded, and a request
# answered by this process is answered under that verified runtime.
#
# ADDITIVE
#
# No existing route is touched, and the observer domain, its validation, the
# interval contract and every stable reason all remain the substrate's.
# ---------------------------------------------------------------------------


@app.get("/sunset-count")
def sunset_count(
    ttLoBits: str, ttHiBits: str, latitude: float, longitude: float
):
    """Return how many genuine sunsets lie in an exact TT interval.

    ``ttLoBits`` and ``ttHiBits`` are the IEEE-754 spellings of the exact
    binary64 Terrestrial Time bounds. Neither need be a sunset. The interval
    is half-open below and closed above - a sunset is counted when
    ``lo < sunset.tt <= hi`` - and that contract belongs to the published
    solver, not to this route. ``latitude`` and ``longitude`` are the
    observer, validated by the substrate against its own governed geodetic
    domain and not re-validated or normalized here.

    Returns the count projection: the integer, the requested interval, the
    interval actually covered, the completeness of that coverage, whether a
    counted sunset coincides exactly with the requested upper bound, and the
    artifact provenance of the single search that determined it. No sunset
    identity is published.

    Fails closed with HTTP 400 carrying a stable reason, for a malformed
    bound or for the substrate's own governed refusal - including a requested
    span beyond the governed admission bound.

    A count of zero is a successful answer, not an absence, and a frontier
    that could not be fully admitted returns its partial count explicitly
    marked incomplete rather than failing.
    """
    try:
        interval_lo = decode_tt_bits(ttLoBits)
        interval_hi = decode_tt_bits(ttHiBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        record = count_sunsets_in_interval(
            interval_lo, interval_hi, latitude, longitude
        )
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_sunset_count(record)


# ---------------------------------------------------------------------------
# A5 - scientific environment provenance route.
#
# The marker above obeys the repository's governed block grammar
# (^# A\d+[a-z]?(?:-[0-9a-z]+)? - ). That is load-bearing: the published route
# source guards delimit each block from its own marker to the NEXT one, so a
# terminal block written under an unrecognised marker would silently be read
# as part of the block before it.
#
# WHAT THIS ROUTE ADDS
#
# One question, and it is not astronomical: what scientific environment is
# this Authority actually running under?
#
# Every astronomical response already carries the pinned artifact a
# determination ran under. That is per-computation provenance and it remains
# the only statement about WHICH kernel answered. It is not enough to decide
# whether a result obtained last month is still equivalent to one obtained
# today: a changed Skyfield, numpy, jplephem, IERS table or artifact content
# would leave that filename identical while the science moved underneath it.
#
# This route publishes the identity that does change in those cases, so a
# consumer persisting certified results can invalidate them correctly instead
# of trusting them indefinitely.
#
# NO ASTRONOMY, AND NOTHING DISCOVERED HERE
#
# The request path loads no kernel, builds no timescale, solves nothing,
# hashes nothing, opens no file and inspects no installation. Every value was
# collected once at startup, before this application object existed, and this
# reads it. That is not an optimization: recomputing the identity per request
# would rehash 3.3 GB of pinned artifacts to re-derive a constant.
#
# NO FAILURE MODE OF ITS OWN
#
# Establishment happens at startup and a failure there stops the process, so a
# reachable route implies an established identity. There is deliberately no
# 503, no partially initialised state and no self-reported unavailability: a
# running Authority either knows what it is or does not exist.
#
# THE IDENTITY IS CHECKABLE, NOT MERELY ASSERTED
#
# The response carries the canonical environment, the derived identifier, and
# the canonical ephemeris manifest. A consumer can therefore verify both
# derivations independently rather than taking either on trust:
#
#     canonical_digest(environment)        == scientificEnvironmentId
#     derive_ephemeris_data_set_id(manifest) == environment.ephemerisDataSetId
#
# The manifest is the identifier's PREIMAGE, not a duplicate of it. Nothing
# else is added: no field already represented canonically is repeated, and no
# second vocabulary is invented for an identifier A0.1 has already named.
#
# WHAT IS DELIBERATELY ABSENT
#
# No filesystem path, no host, no port, no process identifier, no environment
# variable, no credential and no operational internal. Artifact routing names
# are filenames every astronomical response already publishes as `kernel`, and
# artifact hashes are content identity. Nothing here discloses where anything
# lives.
#
# Sampling policy is also absent, deliberately. The certified predicate's grid
# is a property of the Skyfield version already named here, and a deliberate
# Authority change to it is a scientific behaviour change that must advance
# authoritySolverGeneration rather than appear as a new field.
#
# ADDITIVE
#
# No existing route is touched and no astronomical contract changes.
# ---------------------------------------------------------------------------


@app.get("/scientific-environment")
def scientific_environment():
    """Return the scientific identity this Authority established at startup.

    ``environment`` is the canonical scientific environment record.
    ``scientificEnvironmentId`` is its canonical digest.
    ``ephemerisManifest`` is the canonical manifest that
    ``environment.ephemerisDataSetId`` was derived from, published so the
    derivation can be reproduced independently.

    Takes no parameters, performs no astronomy, and cannot fail: the identity
    was established before this route became reachable.
    """
    return scientific_provenance()


# ---------------------------------------------------------------------------
# PTC-I1 - Earth rotation evidence route.
#
# WHAT THIS PUBLISHES
#
# The time-scale and Earth-orientation quantities this certified runtime has
# always been able to establish, exposed for the first time: UT1, Delta T,
# UT1-UTC where it is meaningful, and the Earth Rotation Angle under a named
# convention, each with the provenance of the source that actually produced it.
#
# WHY IT TAKES NO OBSERVER
#
# Earth Rotation Angle is a property of the Earth's orientation as a body and
# carries no observer term. Accepting a latitude or longitude would suggest one
# were required, and at the exact geographic poles no unique meridian exists to
# supply. The scientific question this route answers genuinely has no observer
# in it.
#
# WHAT IT DOES NOT PUBLISH
#
# No mean-solar phase, no continuity cell, no cell ordinal, no day count and no
# calendar position of any kind. Deriving a local phase requires a convention
# about where a day begins and, away from the poles, a meridian - neither of
# which this Authority owns. UT1 is published instead, and the arithmetic is
# the consumer's, carrying the consumer's conventions openly.
#
# ADDITIVE
#
# No existing route is touched and no astronomical contract changes.
# ---------------------------------------------------------------------------


@app.get("/earth-rotation")
def earth_rotation(ttBits: str):
    """Return the Earth-rotation evidence for an exact Terrestrial Time state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial Time
    anchor, identical to every other exact route. It is arbitrary: it need not
    be an event and need not lie near one. No observer is accepted, because the
    published quantities do not have one.

    Returns the requested instant in the standard exact projection, together
    with ``ut1JulianDate``, ``deltaTSeconds``, ``dut1Seconds``,
    ``eraRotations``, the governing ``rotationConvention`` and the provenance
    regime of each derived quantity. ``dut1Seconds`` is null outside the pinned
    table's span, where UT1-UTC is extrapolation arithmetic rather than an
    Earth-orientation measurement.

    Fails closed with HTTP 400 carrying a stable reason, either for a malformed
    anchor or for a state the certified time-scale model cannot evaluate.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        return earth_rotation_evidence(anchor)
    except EarthRotationError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error


# ---------------------------------------------------------------------------
# PTC-I2 - certified solar regime and event-absence evidence route.
#
# The marker deliberately does NOT obey the repository's governed block
# grammar (^# A\d+[a-z]?(?:-[0-9a-z]+)? - ). A5 asserts that it is the
# terminal governed block, and a PTC marker written in that grammar would
# falsify that claim rather than extend it.
#
# WHAT THIS ROUTE ADDS
#
# The first contract on this Authority that can say what the Sun does over a
# declared interval rather than when it next does something. Every existing
# sunset route answers "when", inside a bounded search, so its silence is a
# statement about the search horizon and not about the sky.
#
# THE DISTINCTION IT EXISTS TO PRESERVE
#
# Three answers, never two. A crossing proven present; absence certified over
# the whole declared interval; or the Authority unable to establish which,
# because the geometry sits inside the resolution guard or because the
# examined territory fell short of what was asked. A consumer that cannot
# tell the third from the second will eventually read a three-day search
# that found nothing as a sky in which nothing happens.
#
# WHAT IT IS NOT
#
# Not a polar contract, not a continuity contract and not a calendar
# contract. It states an astronomical fact about an interval and an observer.
# No event is synthesized and no event instant is published: a sunset here is
# the same certified apparent event the directional routes determine, and
# they remain the only place this Authority identifies one.
#
# ADDITIVE
#
# No existing route is touched, no existing solver is modified and no
# astronomical contract changes.
# ---------------------------------------------------------------------------


@app.get("/solar-regime")
def solar_regime(
    ttLoBits: str, ttHiBits: str, latitude: float, longitude: float
):
    """Return the certified solar regime over an exact TT interval.

    ``ttLoBits`` and ``ttHiBits`` are the IEEE-754 spellings of the exact
    binary64 Terrestrial Time bounds. Neither need be an event. The interval
    is closed - a regime is a statement about territory, so both endpoints are
    examined - and that contract belongs to the published solver, not to this
    route. ``latitude`` and ``longitude`` are the observer, validated by the
    substrate against its own governed geodetic domain and not re-validated or
    normalized here.

    Returns the regime projection: the classification, whether a crossing is
    present, absent or unestablished, how many crossings the certified root
    finder resolved and whether that enumeration is whole, the requested and
    covered intervals with the completeness of that coverage, the event
    threshold and convention the answer is about, the altitude-margin extremes
    and the resolution guard they were judged against, and the role of the
    pinned artifact that answered. No event identity is published.

    Fails closed with HTTP 400 carrying a stable reason, for a malformed bound
    or for the substrate's own governed refusal - including a reversed or
    zero-width interval, an observer outside the geodetic domain, and a
    requested span beyond the governed admission bound.

    A certified continuous regime is a successful answer, not an absence of
    one, and an interval whose territory could not be fully examined returns
    INDETERMINATE explicitly rather than reporting that nothing happens in it.
    """
    try:
        interval_lo = decode_tt_bits(ttLoBits)
        interval_hi = decode_tt_bits(ttHiBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        record = determine_solar_regime(
            interval_lo, interval_hi, latitude, longitude
        )
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_solar_regime(record)


# ---------------------------------------------------------------------------
# DT-R4-A - certified night-start route.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for one published question: where does the first night
# strictly after an exact Terrestrial Time state begin, for one observer? It
# is the astronomical evidence the ratified R4 Creation-Week anchor rule
# consumes. The answer is a kind and a certified bracket: a genuine sunset,
# or - where polar day has removed that night - the altitude minimum the
# vanished night collapsed into, or an explicitly unresolved tangency.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route decodes an exact anchor, hands it and the observer to the
# published solver, and projects whatever comes back. It runs no search,
# selects no artifact, enforces no domain of its own and re-derives none of
# the scientific decisions, which all stay where they were certified.
#
# NO CALENDAR MEANING
#
# No Earth-rotation index, local mean time, weekday, Creation week, Sabbath
# or HPC day is produced, and a vanished night's minimum is never published
# as a sunset. Which Carrier-B rotation contains a night-start is decided by
# a consumer, which owns that convention.
#
# THE ANCHOR IS BITS, NOT A DECIMAL
#
# Exactly as /sunset-event-after: ttBits is the IEEE-754 spelling of the
# exact binary64 state, and there is deliberately no tt parameter, no UTC,
# no civil date and no kernel selector.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a TRANSPORT failure and reports the transport
# reason. A state that was received and then refused by the substrate - a
# malformed instant, an observer outside the geodetic domain or outside the
# narrower certified R4 latitude domain, exhausted coverage or computational
# reach, a Delta-T knot, a detection ambiguity or inconsistency, an instant
# inside a night-start bracket, polar night, or a frontier holding no
# night-start - reports the substrate's own stable reason, unchanged, at 400.
#
# Only those two exception types are caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# SCIENTIFIC ENVIRONMENT
#
# Stateless, exactly as the rest of the exact layer is.
#
# ADDITIVE
#
# No existing route is touched, no existing solver is modified and no
# existing astronomical answer changes.
# ---------------------------------------------------------------------------


@app.get("/night-start-after")
def night_start_after(ttBits: str, latitude: float, longitude: float):
    """Return the certified first night-start strictly after an exact state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial
    Time anchor. ``latitude`` and ``longitude`` are the observer, validated
    by the substrate against the governed geodetic domain and the certified
    R4 latitude domain, and not re-validated or normalized here.

    Returns the night-start projection: its kind, its certified bracket, the
    sunset event for a genuine sunset, the sampled altitude-margin evidence,
    the event threshold and convention, the artifact and its role, and the
    certified latitude limit.

    Fails closed with HTTP 400 carrying a stable reason, for a malformed
    anchor or for the substrate's own governed refusal.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        record = find_night_start_after(anchor, latitude, longitude)
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_night_start(record)


# ---------------------------------------------------------------------------
# PTC-A1 - certified directional solar-crossing route.
#
# The marker deliberately does NOT obey the repository's governed block
# grammar, exactly as the PTC-I1, PTC-I2 and DT-R4-A markers above do not:
# the A5 block remains the terminal governed A block.
#
# WHAT THIS ROUTE ADDS
#
# The HTTP surface for one published question: for one observer, where is
# the requested genuine solar horizon crossing - RISING or SETTING - AFTER or
# BEFORE an exact Terrestrial Time state, inside a governed horizon? It is the
# first published sunrise, and the first certified backward search.
#
# NO ASTRONOMY HAPPENS HERE
#
# The route decodes an exact anchor, hands it, the observer, the orientation,
# the direction and the horizon to the published solver, and projects
# whatever comes back. It runs no search, selects no artifact, applies no
# resource bound or domain policy of its own and re-derives none of the
# scientific decisions, which all stay where they were certified.
#
# ABSENCE IS AN ANSWER, AND IT IS 200
#
# A complete frontier holding no requested crossing is a successful
# determination, reported with a null event. This deliberately differs from
# /sunset-event-after, whose 404 is unchanged: this contract states its
# frontier and its completeness on every answer, so "certifiably none here"
# is a fact about the searched territory rather than a missing resource. An
# incomplete frontier holding none is never reported that way - the solver
# refuses it with the reason its territory ran out.
#
# TWO FAILURES, KEPT APART
#
# A malformed ttBits is a TRANSPORT failure and reports the transport
# reason. Anything the substrate refuses - a malformed instant, an
# unpublished orientation or direction, an invalid or over-long horizon, an
# observer outside the geodetic domain, exhausted coverage or computational
# reach, a Delta-T knot, a detection ambiguity or inconsistency - reports the
# substrate's own stable reason, unchanged, at 400.
#
# Only those two exception types are caught. An unexpected failure is not a
# governed rejection and must stay visible rather than be relabelled as one.
#
# NO CALENDAR MEANING
#
# No rotation, weekday, Creation week, Sabbath, HPC day, polar regime or
# continuity state is produced. Which polar event a crossing is belongs to a
# consumer.
#
# ADDITIVE
#
# No existing route is touched, no existing solver is modified and no
# existing astronomical answer changes.
# ---------------------------------------------------------------------------


@app.get("/solar-crossing")
def solar_crossing(
    ttBits: str,
    latitude: float,
    longitude: float,
    orientation: str,
    direction: str,
    horizonDays: float,
):
    """Return the certified directional solar crossing for an exact state.

    ``ttBits`` is the IEEE-754 spelling of the exact binary64 Terrestrial
    Time anchor. ``latitude`` and ``longitude`` are the observer, validated
    by the substrate against the governed geodetic domain and not
    re-validated or normalized here. ``orientation`` is RISING or SETTING,
    ``direction`` AFTER or BEFORE, and ``horizonDays`` the horizon searched
    in that direction.

    Returns the crossing projection: the requested crossing or a null event
    over a complete frontier, the request, the frontier covered and its
    completeness, the event threshold and convention, and the artifact.

    Fails closed with HTTP 400 carrying a stable reason, for a malformed
    anchor or for the substrate's own governed refusal.
    """
    try:
        anchor = decode_tt_bits(ttBits)
    except ExactTimeTransportError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    try:
        record = find_solar_crossing(
            anchor, latitude, longitude, orientation, direction, horizonDays
        )
    except SunsetChronologyError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error

    return project_solar_crossing(record)


# ---------------------------------------------------------------------------
# PTC-T2 - exact civil-instant resolver route.
#
# The marker deliberately does NOT obey the repository's governed block
# grammar, for the same reason as PTC-I2: A5 asserts that it is the terminal
# governed block.
#
# WHAT THIS ROUTE ADDS
#
# The exact Terrestrial Time identity of one civil UTC instant, so that a
# presentation client holding only a civil clock reading can call the exact
# routes without owning any time-scale science. The conversion is the one the
# UTC-input routes above have always performed; only its result is new on the
# wire.
#
# WHAT IT IS NOT
#
# Not a Deep-Time, BCE or chronology converter. Instants outside the civil span
# certified by the pinned time-scale artifact are refused, not extrapolated.
#
# ADDITIVE
#
# No existing route is touched, no existing solver is modified and no
# astronomical answer changes.
# ---------------------------------------------------------------------------


@app.get("/civil-instant")
def civil_instant(utc: str):
    """Return the exact Terrestrial Time identity of a civil UTC instant.

    ``utc`` is an ISO 8601 civil instant, read by the same parser as every
    other UTC-input route: an offset is honoured and normalized to UTC, and an
    instant without one is read as UTC.

    Returns the resolved instant in the standard exact projection, the instant
    as parsed, the inclusive certified civil span, and the scientific
    environment that performed the conversion.

    Fails closed with HTTP 400 carrying a stable reason, for an unparseable
    instant or for one outside the certified civil span.
    """
    try:
        requested = parse_utc_datetime(utc)
    except (ValueError, OverflowError) as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(
                REASON_CIVIL_INSTANT_INVALID,
                "CIVIL INSTANT INVALID - the instant is not a representable "
                "ISO 8601 civil timestamp",
            ),
        ) from error

    try:
        return resolve_civil_instant(requested)
    except CivilInstantError as error:
        raise HTTPException(
            status_code=400,
            detail=reason_detail(error.reason, str(error)),
        ) from error
