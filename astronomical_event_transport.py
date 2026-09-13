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
