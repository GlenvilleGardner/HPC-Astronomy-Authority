"""HPC Astronomy Authority - Exact Civil-Instant Resolution.

PTC-T2 - the civil UTC to exact Terrestrial Time conversion this Authority has
always performed internally, published for presentation clients.

WHAT THIS MODULE IS

One question: which exact binary64 Terrestrial Time state does this certified
runtime assign to one civil UTC instant?

Exact-instant consumers take the published ttBits identity and nothing else. A
presentation client holds only a civil clock reading. This module closes that
gap with the conversion the Authority already uses wherever a UTC datetime
enters its astronomy - the shared timescale's from_datetime - so a client never
has to own time-scale science and no second conversion exists anywhere.

IT PERFORMS NO ASTRONOMY

No ephemeris is opened, no kernel is selected, no observer is accepted and no
event is solved. Delta T, UT1 and Earth orientation take no part in UTC to TT
conversion, so none is evaluated and none is reported.

IT KNOWS NOTHING ABOUT CALENDARS

No HPC day, year, month, weekday or continuity state, and no projection onto
any grid. The resolved state is passed by the client, unchanged, to whichever
exact-instant capability it is calling.

THIS IS NOT A GENERAL CIVIL-TIME CONVERTER

The year-addressed solar-longitude operation refuses to become a general civil
to Terrestrial Time surface, and that refusal stands. This module is narrower
than such a surface in the way that matters: it resolves only instants inside
the span the pinned time-scale artifact actually certifies, and refuses every
other instant. It is not a Deep-Time, BCE, Creation-chronology or SCE
conversion and must never be used as one.

THE CERTIFIED CIVIL SPAN IS DERIVED FROM THE ARTIFACT, NOT FROM A YEAR

The runtime will numerically convert any representable datetime: before the
first leap-table entry it applies the earliest offset indefinitely, and after
the last tabulated day it applies the latest offset indefinitely. Neither
extrapolation is certified, so neither is published.

- Lower bound: the first explicit leap-table entry of the shared timescale,
  ``ts.leap_dates[0]``, a UTC Julian Date, converted by the same timescale.
- Upper bound: the end of the pinned IERS table span, read through the same
  function that governs the Earth-rotation TABULATED regime, with the same
  inclusive Terrestrial Time semantics.

Both bounds are inclusive and both are compared in Terrestrial Time, the scale
the table span is indexed by. When the certified artifact is re-certified the
span moves with it, and no source change is required or permitted.

The span does not distinguish observed IERS rows from predicted ones. The
pinned artifact does not preserve that distinction, so nothing here claims it:
an instant is inside the certified span or outside it, and nothing finer.

LEAP SECONDS

A Python datetime cannot represent a :60 civil second, so a leap second cannot
be requested and none is fabricated. The certified leap table is applied on
either side of every leap boundary by the shared timescale; nothing here
compensates for a leap second by hand.

PRECISION

A civil instant is accepted at the resolution of the existing UTC parser. The
resolved state is a binary64 Julian Date, whose spacing near the present epoch
is roughly forty microseconds, so instants closer than that may share one
state. That state is the identity; the ``utc`` rendering beside it is reference
information produced by the existing projection and reproduces the input to
the millisecond. Inside a leap second that rendering cannot spell :60 either,
which is a property of the existing projection and not of the resolved state.

FAILS CLOSED

An instant outside the certified span is refused with a stable reason. When the
running timescale exposes no table from which to derive the span, every
instant is refused. There is no fallback, no substituted scale and no nominal
offset arithmetic.
"""

from astronomical_event_transport import project_exact_instant
from astronomy_solver import ensure_utc, ts
from earth_rotation import _tabulated_span
from scientific_environment import ScientificEnvironmentError
from scientific_provenance import scientific_provenance

REASON_CIVIL_INSTANT_INVALID = "CIVIL_INSTANT_INVALID"
REASON_CIVIL_INSTANT_OUTSIDE_CERTIFIED_SPAN = (
    "CIVIL_INSTANT_OUTSIDE_CERTIFIED_SPAN"
)
REASON_CERTIFIED_SPAN_UNAVAILABLE = "CERTIFIED_SPAN_UNAVAILABLE"

# The Modified Julian Date origin, 1858-11-17 00:00 UTC, as a Julian Date. The
# leap table stores UTC Julian Dates and the timescale accepts calendar fields,
# so a table date is presented to it as a day count from this origin - the same
# form the runtime itself uses when it builds its tables from IERS data.
_MJD_ORIGIN_JULIAN_DATE = 2400000.5


class CivilInstantError(ScientificEnvironmentError):
    """Exact civil-instant resolution failed closed."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _certified_civil_span():
    """The inclusive Terrestrial Time span this artifact certifies for UTC.

    Returns (lower, upper) as binary64 Terrestrial Time Julian Dates, or None
    when the running timescale exposes no leap table or no IERS table, in
    which case no instant can be certified.
    """
    span = _tabulated_span()

    if span is None:
        return None

    try:
        first_entry = float(ts.leap_dates[0])
    except (IndexError, TypeError, ValueError):
        return None

    lower = float(
        ts.utc(1858, 11, 17.0 + (first_entry - _MJD_ORIGIN_JULIAN_DATE)).tt
    )

    return lower, span[1]


_CERTIFIED_CIVIL_SPAN = _certified_civil_span()


def resolve_civil_instant(value):
    """Return the exact Terrestrial Time identity of one civil UTC instant.

    ``value`` is a datetime already parsed by the existing UTC parser. It is
    normalized to UTC by the existing normalization and converted by the
    shared timescale, exactly as every UTC-input astronomy path converts it.

    The returned record carries the resolved instant in the same projection
    every exact route publishes, the instant as parsed, the certified civil
    span it was admitted against, and the scientific environment that
    performed the conversion.

    Fails closed with CivilInstantError when the instant lies outside the
    certified span or no span can be derived.
    """
    if _CERTIFIED_CIVIL_SPAN is None:
        raise CivilInstantError(
            REASON_CERTIFIED_SPAN_UNAVAILABLE,
            "CERTIFIED SPAN UNAVAILABLE - the running time-scale artifact "
            "exposes no table from which a certified civil span can be derived",
        )

    requested = ensure_utc(value)
    tt = float(ts.from_datetime(requested).tt)
    lower, upper = _CERTIFIED_CIVIL_SPAN

    if not lower <= tt <= upper:
        raise CivilInstantError(
            REASON_CIVIL_INSTANT_OUTSIDE_CERTIFIED_SPAN,
            "CIVIL INSTANT OUTSIDE CERTIFIED SPAN - %s lies outside the civil "
            "span certified by the pinned time-scale artifact"
            % requested.isoformat(timespec="microseconds"),
        )

    provenance = scientific_provenance()

    record = project_exact_instant(tt)

    record["requestedUtc"] = requested.isoformat(timespec="microseconds")
    record["certifiedSpan"] = {
        "lower": project_exact_instant(lower),
        "upper": project_exact_instant(upper),
    }
    record["scientificEnvironmentId"] = provenance["scientificEnvironmentId"]
    record["iersDataSha256"] = provenance["environment"]["iersDataSha256"]
    record["skyfieldVersion"] = provenance["environment"]["skyfieldVersion"]

    return record
