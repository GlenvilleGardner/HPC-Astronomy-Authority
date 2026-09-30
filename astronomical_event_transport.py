"""HPC Astronomy Authority - Astronomical Event Transport.

A3c-0 - the wire projection of a determined astronomical event.

WHAT THIS MODULE IS

The projection that turns an exact astronomical determination into something
a consumer can read, without letting the readable form become the authority.
It performs NO astronomy: no search is run, no crossing is solved, no kernel
is selected, no frontier is consulted and no event kind is validated. It
receives determinations that are already made and writes them down.

No HTTP semantics are decided here. Status codes, routes and request
handling belong to the server, and this module is usable without one.

THE AUTHORITY IS THE BITS

``ttBits`` is the transport identity of an event: the exact IEEE-754
spelling of the Terrestrial Time the determination was actually made at.
``tt`` is the same value as a JSON number - a numerical projection OF those
bits, never a second opinion about them. The two can never disagree, because
one is produced from the other rather than alongside it.

The invariant a consumer may rely on, and which the certification asserts,
is exactly:

    decode_tt_bits(projection["ttBits"]) is bit-identical to projection["tt"]

UTC IS A REFERENCE, NOT AN EVENT

``utc`` is derived, human-facing reference information. It is never event
identity, never a search input, never continuation state, never an input to
kernel selection and never HPC calendar authority. It is produced from the
event AFTER the event exists, and nothing derived from it is ever fed back
into a computation.

It is also, for most of the authoritative domain, impossible. The certified
artifacts span roughly 11.5 million TT days, while a calendar datetime can
only be formed between year 1 and year 9999 - under a third of that reach.
Every event outside that window is a genuine determination with no calendar
rendering, so ``utc`` is None there.

The key is ALWAYS present. A consumer must be able to distinguish "this
event has no calendar rendering" from "this projection forgot to include
one", and a conditionally absent key cannot express that difference.

PRECISION

A representable reference is emitted with exactly six fractional digits.
Nothing is truncated to whole seconds and nothing is normalized to
milliseconds: those are the two losses that made a calendar rendering
unusable as identity in the first place, and neither is reintroduced here.
The fixed width is deliberate, so the field does not silently change shape
when a determination happens to land on an exact second.
"""

from astronomy_solver import ts
from exact_time_transport import decode_tt_bits, encode_tt_bits


def utc_reference(tt):
    """Return a calendar reference rendering of ``tt``, or None.

    The exact state is never altered, and nothing produced here is ever
    consulted by a computation. This is the last step of presentation, not
    the first step of anything.

    Only ValueError is caught, and it is caught because that is precisely
    what the certified machinery raises when the instant lies outside the
    span a calendar datetime can represent - below year 1 or above year
    9999. Repository precedent is the published BCE rendering fallback,
    which catches the same single exception for the same reason.

    Nothing else is caught. Any other failure is not a statement about
    calendar representability and must propagate rather than be recorded
    here as an unrepresentable instant.
    """
    try:
        return ts.tt_jd(tt).utc_datetime().isoformat(timespec="microseconds")
    except ValueError:
        return None


def project_exact_instant(tt):
    """Return the wire projection of one exact Terrestrial Time state.

    ``ttBits`` is the identity. ``tt`` is derived from those same bits
    rather than from the argument, so the two fields cannot drift apart:
    whatever a consumer decodes from ``ttBits`` is exactly the number it was
    given. ``utc`` is a reference rendering or None.

    The three keys are always present, in this order.
    """
    tt_bits = encode_tt_bits(tt)
    exact = decode_tt_bits(tt_bits)

    return {
        "ttBits": tt_bits,
        "tt": exact,
        "utc": utc_reference(exact),
    }


def project_astronomical_event(event):
    """Return the wire projection of a determined astronomical event.

    ``event`` is a published AstronomicalEvent. Its exact state, governed
    identity and artifact provenance are carried through unchanged; nothing
    is recomputed, reinterpreted or renamed.

    ``kind`` is the governed internal scientific identity exactly as the
    solver reported it. No calendar or season terminology is produced here:
    the 180 degree crossing opens spring in Sydney and autumn in New York,
    so a hemisphere-bearing name is a presentation decision belonging to a
    consumer, not a fact this Authority can state.

    ``kernel`` is the pinned NASA/JPL artifact the determination actually
    ran under. It is computation provenance, not a routing decision.

    The record itself is not modified, and no field is added to it.
    """
    projection = project_exact_instant(event.tt)
    projection["kind"] = event.kind
    projection["kernel"] = event.kernel

    return projection


def project_sunset_event(event):
    """Return the wire projection of a determined sunset.

    ``event`` is a published SunsetEvent. Its exact state and artifact
    provenance are carried through unchanged; nothing is recomputed or
    reinterpreted, and the record itself is not modified.

    ``kernel`` is the pinned NASA/JPL artifact this sunset was actually
    determined under. It is carried per event rather than per bracket
    because the two boundaries surrounding one instant may genuinely have
    been determined under different artifacts, and reporting a single
    bracket-level artifact would assert provenance the Authority never
    established.

    This is deliberately the same shape as an astronomical event projection
    minus the governed kind, because a sunset has no governed kind: it is
    identified by the observer and the instant, not by a longitude the Sun
    attains.
    """
    projection = project_exact_instant(event.tt)
    projection["kernel"] = event.kernel

    return projection


def project_sunset_bracket(bracket):
    """Return the wire projection of an atomic sunset bracket.

    ``anchor`` carries no artifact provenance, deliberately. It is the
    caller's own exact state rather than something the Authority determined,
    so there is no computation whose artifact could be reported; inventing
    one - by copying a boundary's kernel, say - would claim provenance for a
    value the Authority never computed.

    ``previous`` and ``next`` each keep their own. They may legitimately
    name different artifacts when the anchor lies near a coverage boundary,
    and that is preserved exactly rather than flattened.

    The observer is absent, exactly as it is from the record. Binding an
    observer into a transportable record is the continuation witness's
    responsibility, and repeating it here would create a second, unchecked
    place where an observer binding could drift from the one that actually
    governed the search.
    """
    return {
        "anchor": project_exact_instant(bracket.anchor),
        "previous": project_sunset_event(bracket.previous),
        "next": project_sunset_event(bracket.next),
    }


def reason_detail(reason, message):
    """Return the structured body of a governed failure.

    A stable reason code crossing a wire as prose inside a sentence cannot
    be read by a machine without parsing English. This carries the code as
    its own field so a consumer can branch on it, with the human-readable
    message alongside rather than instead.

    No HTTP status is decided here, and none is carried. Which status a
    given reason deserves is a routing question, and this shape is equally
    usable by a caller that has no HTTP at all.
    """
    return {
        "reason": reason,
        "message": message,
    }


def project_sunset_count(record):
    """Return the wire projection of a bulk observer-local sunset count.

    ``record`` is a published SunsetCount. Its integer, its bounds, its
    completeness and its artifact provenance are carried through unchanged;
    nothing is recomputed, reinterpreted or renamed, and the record itself is
    not modified.

    HOW MANY, NEVER WHICH

    A count answers a different question from an event, and this projection
    keeps it that way. No crossing, no root, no array of instants and no
    individual event identity appears here, and none may ever be added: the
    solver deliberately does not retain them, so there is nothing to leak
    even by accident. A consumer needing boundaries must ask the published
    directional routes for them, one at a time, which is the only place the
    Authority will state a sunset's identity.

    Nor is any calendar meaning produced. There is no weekday, no day
    ordinal, no month, no year, no year type and no 365/366 classification.
    The integer is an astronomical count of transitions; what a calendar
    makes of it is a question for a consumer that this Authority does not
    answer.

    FOUR BOUNDS, BECAUSE TWO WOULD BE A CLAIM

    ``requested`` is what the caller asked about. ``covered`` is the frontier
    the search actually examined. They are both published because they can
    legitimately differ: authoritative coverage or evaluable reach may end
    before the requested upper bound, and a count reported against only the
    requested pair would silently attribute to the whole interval a number
    that belongs to part of it.

    Each of the four is projected through the same exact-instant projection
    every other route uses, so ``ttBits`` remains the identity, ``tt`` is
    produced from those same bits rather than alongside them, and ``utc`` is
    reference information that is null wherever no calendar rendering exists.

    ``complete`` and ``truncationReason`` are one statement in two fields:
    the reason is null exactly when the frontier was whole. Together they are
    what stops a partial count from being read as a total, and a count of
    zero over a COMPLETE frontier - the ordinary polar answer - from being
    confused with a search that ran out of territory.

    ``boundaryCoincident`` reports one physical fact: a counted sunset's
    exact state equals the requested upper bound at binary64 equality. It
    decides nothing. Which interval such a crossing belongs to is a calendar
    question, and the Authority states the fact rather than resolving it.

    ``kernel`` is the single pinned NASA/JPL artifact the one admitted
    frontier ran under. It is singular because the search was: nothing here
    spans two artifacts, so no per-bound provenance exists to report.
    """
    return {
        "sunsetCount": record.count,
        "requested": {
            "lo": project_exact_instant(record.requested_lo),
            "hi": project_exact_instant(record.requested_hi),
        },
        "covered": {
            "lo": project_exact_instant(record.covered_lo),
            "hi": project_exact_instant(record.covered_hi),
        },
        "complete": record.complete,
        "truncationReason": record.truncation_reason,
        "boundaryCoincident": record.boundary_coincident,
        "kernel": record.kernel,
    }


def project_solar_regime(record):
    """Return the wire projection of a certified solar regime determination.

    ``record`` is a published SolarRegime. Its classification, its bounds, its
    completeness, its evidence and its artifact role are carried through
    unchanged; nothing is recomputed, reinterpreted or renamed, and the record
    itself is not modified.

    THREE ANSWERS, NEVER TWO

    ``crossingPresent`` is the field this whole contract exists for. True
    means a crossing is proven to lie in the declared interval. False means
    absence is certified across the whole of it. Null means the Authority
    cannot establish which - because the geometry sits inside the resolution
    guard, or because the examined territory was not the whole interval.

    A consumer must be able to distinguish those three, because collapsing the
    third into the second is exactly how a bounded search gets mistaken for an
    empty sky. ``regime`` names which of the five cases produced the answer,
    and ``complete`` with ``truncationReason`` says whether the territory was
    whole.

    EVIDENCE, NOT JUST A VERDICT

    ``minimumAltitudeMarginDegrees`` and ``maximumAltitudeMarginDegrees`` are
    how far the Sun's apparent altitude stayed from the certified threshold at
    its extremes, and ``resolutionGuardDegrees`` is how far it had to stay for
    continuity to be certifiable at ``sampleStepDays``. Published together
    they let a consumer see how near the call was, and let an auditor rederive
    the classification instead of believing it.

    ``eventThresholdDegrees`` and ``eventConvention`` name the event itself.
    Without them a regime is a verdict about an unstated predicate.

    NO EVENT IDENTITY

    No crossing, no root and no instant appears. There is no PTC sunset and no
    polar sunset: a sunset here is the same certified apparent event every
    other route determines, and its instant is stated by the published
    directional routes, one at a time, which is the only place this Authority
    identifies one. ``sunsets`` and ``sunrises`` are how many, never which,
    and ``crossingEnumerationAgrees`` marks the intervals where the certified
    root finder under-resolves grazing crossings and those counts are a lower
    bound rather than a total.

    FOUR BOUNDS, BECAUSE TWO WOULD BE A CLAIM

    ``requested`` is what the caller asked about; ``covered`` is the frontier
    actually examined. They can legitimately differ, and a regime reported
    against only the requested pair would attribute to a whole interval a
    classification that belongs to part of it. Each of the four is projected
    through the same exact-instant projection every other route uses.

    ``ephemerisRole`` is artifact provenance without an artifact filename. A
    consumer needing the artifact's identity resolves the role through the
    published scientific environment, which names it with its content digest.

    NO OBSERVER

    The observer is absent, exactly as it is from the record and from every
    other projection in this module. It governed the search and is known to
    the caller that supplied it; echoing it here would create a second,
    unchecked place where an observer binding could drift from the one the
    geometry actually ran on. The route still accepts a latitude and a
    longitude - they simply do not become part of what is transported.

    No calendar meaning is produced. There is no weekday, no day ordinal, no
    month, no year, no Telma, no Creation week, no Sabbath and no continuity
    interval. A regime is an astronomical fact about an interval; what a
    calendar makes of it is a question for a consumer that this Authority does
    not answer.
    """
    return {
        "regime": record.regime,
        "crossingPresent": record.crossing_present,
        "sunsets": record.sunsets,
        "sunrises": record.sunrises,
        "crossingEnumerationAgrees": record.enumeration_agrees,
        "requested": {
            "lo": project_exact_instant(record.requested_lo),
            "hi": project_exact_instant(record.requested_hi),
        },
        "covered": {
            "lo": project_exact_instant(record.covered_lo),
            "hi": project_exact_instant(record.covered_hi),
        },
        "complete": record.complete,
        "truncationReason": record.truncation_reason,
        "eventThresholdDegrees": record.event_threshold_degrees,
        "eventConvention": record.event_convention,
        "minimumAltitudeMarginDegrees": record.minimum_margin_degrees,
        "maximumAltitudeMarginDegrees": record.maximum_margin_degrees,
        "resolutionGuardDegrees": record.guard_degrees,
        "sampleStepDays": record.step_days,
        "ephemerisRole": record.ephemeris_role,
    }
