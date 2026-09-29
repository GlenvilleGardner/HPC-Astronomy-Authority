"""HPC Astronomy Authority - Earth Rotation Evidence.

PTC-I1 - the time-scale and Earth-orientation quantities this Authority can
already establish, published for the first time.

WHAT THIS MODULE IS

One question: at one exact Terrestrial Time state, what does this certified
runtime know about the rotation of the Earth?

It performs NO astronomy in the ephemeris sense. No BSP artifact is opened, no
kernel is selected, no observer is validated, no predicate is evaluated and no
event is solved. It reads the certified timescale and nothing else, which is
why it needs neither an ephemeris nor a geodetic domain.

IT KNOWS NOTHING ABOUT CALENDARS. No HPC year, no month, no day, no Telma, no
Creation week, no Sabbath, no polar continuity interval and no projection onto
anyone's grid. Those belong to consumers of this evidence and would be a
category error here: the Earth's rotation is not a calendar and does not become
one by being measured.

WHY AN OBSERVER IS ABSENT

Earth Rotation Angle is the geocentric angle between the Celestial Intermediate
Origin and the Terrestrial Intermediate Origin. It is a property of the Earth's
orientation as a body and contains no observer term, so an observer at the
equator and an observer at the geographic pole share it exactly. Accepting a
latitude or a longitude here would therefore invite the belief that one was
needed, and at the exact poles no unique meridian exists to supply.

Local mean-solar phase DOES require a meridian. That is precisely why it is not
computed here - see WHAT IS DELIBERATELY NOT COMPUTED below.

THE INSTANT IS AN EXACT TT STATE, NOT A DATE

The input is the published ttBits encoding, identical to every other exact
route. A civil date string would reintroduce the questions this Authority
already settled: which timezone, which rounding, which calendar reform. An
exact binary64 TT state has none of them.

UT1 IS RECONSTRUCTIBLE EXACTLY, NOT MERELY REPORTED

``ut1JulianDate`` is a binary64 Julian Date and therefore resolves to roughly a
tenth of a millisecond near the present epoch. That is ample for its stated
purpose, but it is not the exact carrier. The exact carrier is the pair
(``ttBits``, ``deltaTSeconds``): UT1 = TT - Delta T, computed by the consumer
from the same bits this Authority was given. A consumer needing full precision
reconstructs rather than reads, and cannot be silently degraded by a decimal
projection.

CONVENTIONS, STATED RATHER THAN ASSUMED

Earth Rotation Angle follows IAU 2000 Resolution B1.8, under which UT1 is
DEFINED by a conventionally adopted linear proportionality to ERA, with the
relation published in the IERS Conventions. ``rotationConvention`` carries that
identity on the wire so a consumer never has to infer which definition produced
the number.

ERA is reported in ROTATIONS on [0, 1), which is the form the certified runtime
computes. Degrees and radians are omitted deliberately: a second spelling of
one quantity is a second thing to keep consistent, and multiplying by 360 is
not a service this Authority needs to provide.

PROVENANCE IS DERIVED FROM DATA, NOT ASSERTED FROM A YEAR

The bundled IERS/Delta-T table covers a finite span. Inside it, Delta T is read
from tabulated values; outside it, the runtime evaluates a long-term polynomial
model. This module reports which of those actually applied by comparing the
requested state against the table's own extent, so the regime moves when the
pinned table moves and cannot drift out of step with a hardcoded year.

WHAT THIS MODULE REFUSES TO CLAIM

The bundled table does not distinguish OBSERVED Earth-orientation measurements
from IERS PREDICTIONS - both are simply rows. This module therefore does not
publish those labels, because it cannot establish them. Reporting a predicted
value as observed would be a stronger claim than the evidence supports, and the
honest coarser statement is the one made here.

DUT1 IS WITHHELD OUTSIDE THE TABULATED SPAN, DELIBERATELY

UT1-UTC is an Earth-orientation quantity only where UTC exists and the table
covers the state. Outside that span the runtime will still return a difference,
but it is an artifact of extrapolating Delta T rather than a measurement of the
Earth: at three thousand years before the common era the same arithmetic yields
tens of thousands of seconds. Publishing that number under the name DUT1 would
be false, so it is reported as unavailable with the reason carried alongside.

WHAT IS DELIBERATELY NOT COMPUTED

No mean-solar phase, no continuity cell, no cell ordinal, no retained fraction,
no day count and no calendar position. Polar Temporal Continuity research
established a working cadence of one UT1 mean-solar day with a continuously
retained fractional phase, and none of it is computed here, for two reasons.

It would require a CONVENTION this Authority does not own - where a mean solar
day begins - and at every location except the exact poles it would additionally
require a MERIDIAN, which is a consumer's property and not an astronomical
fact. UT1 is published instead, from which any such phase follows by
arithmetic that carries its own conventions openly.

FAILS CLOSED

A malformed exact state is refused by the published transport codec before this
module is reached. A state the certified time-scale model cannot evaluate is
refused here, with a stable reason. There is no fallback, no substituted scale,
no nominal 86400-second arithmetic and no system clock.
"""

from astronomical_event_transport import project_exact_instant
from astronomy_solver import ts
from scientific_environment import ScientificEnvironmentError

from skyfield.earthlib import earth_rotation_angle

# ---------------------------------------------------------------------------
# Convention identity
# ---------------------------------------------------------------------------

# The definition the published Earth Rotation Angle follows.
#
# IAU 2000 Resolution B1.8 established the Celestial and Terrestrial
# Intermediate Origins and redefined UT1 as a linear function of the angle
# between them; the IERS Conventions publish the adopted relation. Carrying the
# identity on the wire is what lets a consumer verify against the same source
# rather than against this implementation.
ROTATION_CONVENTION = "IAU 2000 Resolution B1.8 (ERA); IERS Conventions"

# ---------------------------------------------------------------------------
# Provenance regimes
# ---------------------------------------------------------------------------

# Delta T was read from the pinned IERS/Delta-T table.
REGIME_TABULATED = "TABULATED"

# Delta T was evaluated from the runtime's long-term polynomial model, because
# the requested state lies outside the pinned table.
REGIME_MODELLED = "MODELLED"

# UT1-UTC is not an Earth-orientation quantity for this state.
REGIME_UNSUPPORTED = "UNSUPPORTED"

REASON_ROTATION_STATE_UNEVALUABLE = "ROTATION_STATE_UNEVALUABLE"


class EarthRotationError(ScientificEnvironmentError):
    """Earth rotation evidence failed closed."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _tabulated_span():
    """The extent of the pinned Delta-T table, in the scale it is indexed by.

    Read once, from the timescale this process actually built, so the published
    regime boundary is a property of the pinned artifact rather than a constant
    maintained beside it.

    Returns None when the running Skyfield exposes no table, in which case no
    state can be claimed as tabulated and every state is reported as modelled.
    That is the conservative direction: it understates confidence.
    """
    table = getattr(ts, "delta_t_table", None)

    if table is None:
        return None

    try:
        jd = table[0]
        return float(jd[0]), float(jd[-1])
    except (IndexError, TypeError, ValueError):
        return None


_TABULATED_SPAN = _tabulated_span()


def delta_t_regime(tt):
    """Which Delta-T source actually applies to this exact TT state."""
    if _TABULATED_SPAN is None:
        return REGIME_MODELLED

    start, end = _TABULATED_SPAN

    return REGIME_TABULATED if start <= tt <= end else REGIME_MODELLED


def earth_rotation_evidence(tt):
    """Return the Earth-rotation evidence for one exact Terrestrial Time state.

    ``tt`` is an exact binary64 Terrestrial Time state, already decoded from its
    published transport encoding. It is arbitrary: it need not be an event and
    need not lie near one.

    The returned record carries the requested instant in the same projection
    every other exact route publishes, the time-scale quantities this runtime
    established for it, the Earth Rotation Angle under the named convention, and
    the provenance of each.

    Fails closed with EarthRotationError when the certified time-scale model
    cannot evaluate the state. No value is estimated, substituted or defaulted.
    """
    try:
        instant = ts.tt_jd(tt)
        delta_t_seconds = float(instant.delta_t)
        ut1_julian_date = float(instant.ut1)
        era_rotations = float(earth_rotation_angle(instant.ut1))
    except Exception as error:
        raise EarthRotationError(
            REASON_ROTATION_STATE_UNEVALUABLE,
            "ROTATION STATE UNEVALUABLE - the certified time-scale model could "
            "not evaluate Terrestrial Time state %r" % (tt,),
        ) from error

    regime = delta_t_regime(tt)

    # UT1-UTC is published only where it is an Earth-orientation quantity.
    # Outside the tabulated span the runtime still returns a difference, but it
    # is extrapolation arithmetic rather than a measurement, so the field is
    # withheld and the reason travels with it.
    if regime == REGIME_TABULATED:
        try:
            dut1_seconds = float(instant.dut1)
            dut1_regime = REGIME_TABULATED
        except Exception:
            dut1_seconds = None
            dut1_regime = REGIME_UNSUPPORTED
    else:
        dut1_seconds = None
        dut1_regime = REGIME_UNSUPPORTED

    evidence = project_exact_instant(tt)

    evidence["ut1JulianDate"] = ut1_julian_date
    evidence["deltaTSeconds"] = delta_t_seconds
    evidence["deltaTProvenance"] = regime
    evidence["dut1Seconds"] = dut1_seconds
    evidence["dut1Provenance"] = dut1_regime
    evidence["eraRotations"] = era_rotations
    evidence["rotationConvention"] = ROTATION_CONVENTION

    return evidence
