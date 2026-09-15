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
    SunsetChronologyError,
    SunsetSuccessorError,
    solar_longitude,
    subsolar_point,
    find_equinox,
    find_season_events,
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
    project_sunset_bracket,
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
