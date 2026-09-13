"""HPC Astronomy Authority - Exact Time Transport.

A3c-0 - the canonical wire encoding of an exact binary64 Terrestrial Time
state.

WHAT THIS MODULE IS

One question: how is an exact TT state written down, and read back, without
losing a single bit? Nothing else.

It is deliberately the smallest module in the repository. It performs NO
astronomy: it imports no ephemeris, builds no timescale, evaluates no
predicate, reads no BSP artifact, touches no network and writes no file. It
knows nothing about cursors, observers, events, kernels, HTTP, Gregorian
years or calendars.

WHY IT MUST NOT KNOW ABOUT ASTRONOMY

astronomy_solver builds its certified timescale at module import. The A0.3
runtime gate therefore has to run before that import, and the continuation
witness guarantees it never builds a timescale at all. A codec that both the
witness and the event transport depend on must consequently sit BELOW both
and import neither, so that importing it can never pull a timescale into
existence. That is an architectural constraint, not a preference.

WHY BITS AND NOT DECIMAL TEXT

An exact TT state is a binary64 value. Written as a decimal string it is
readable but not obviously exact, and a consumer that reformats, rounds or
re-parses it with a different rule silently answers a different question.
Written as its IEEE-754 bit pattern it is exact by construction, and the
question of whether a decimal spelling round-trips does not arise at all.

The 16 uppercase hexadecimal characters are the transport identity. A
decimal projection of the same value may be carried alongside for
convenience, but it is a projection of these bits and never their
replacement.

SIGNED ZERO IS PRESERVED, NOT CANONICALIZED

+0.0 and -0.0 are distinct binary64 states and encode to distinct patterns.
This module transports the state it is given and does not decide whether a
state is scientifically admissible; that is the solver's question, asked
separately and afterwards. The continuation witness canonicalizes signed
zero for OBSERVER COORDINATES, where it makes a serialized binding unique,
and deliberately does not do so for the time state. Nothing here changes
either rule.

WHAT IS AND IS NOT VALIDATED HERE

This module answers a TRANSPORT question: is this a well-formed exact-TT
wire value? It does not answer the SCIENTIFIC question of whether the
decoded state is an admissible solver anchor. A caller that obtains a value
from here still submits it to the published exact-finite-TT validation
before any computation, and the two failures stay distinct: a malformed wire
value is TT_BITS_INVALID, an inadmissible solver state is the solver's own
reason.

Non-finite values are refused in both directions, because an infinity or a
NaN is not a time.
"""

import math
import struct

from scientific_environment import ScientificEnvironmentError

# An IEEE-754 binary64 value is eight bytes, so its hexadecimal spelling is
# exactly sixteen characters. Neither shorter nor longer text is a valid
# encoding of one, whatever it decodes to.
TT_BITS_LENGTH = 16

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"

# Uppercase only. A hexadecimal value has two spellings per digit and only
# one of them is canonical here, so a lowercase or mixed-case string is
# refused rather than accepted as an alternate encoding of the same bits.
_HEX_UPPER = frozenset("0123456789ABCDEF")


class ExactTimeTransportError(ScientificEnvironmentError):
    """An exact time state could not be written down or read back.

    ``reason`` carries a stable code so a later route can map every
    rejection onto one HTTP status with a distinguishing detail. No HTTP
    semantics are decided here.

    The reason is a TRANSPORT reason and must never be presented as an
    astronomical one: it says the wire value was malformed, not that any
    scientific question was asked and refused.

    The message deliberately carries no uppercase banner of its own. Every
    caller of this codec sits inside some larger operation - a continuation
    witness, an event projection, a route - and states which one in its own
    failure. A banner here would either duplicate that or contradict it, and
    the stable reason already identifies this layer precisely.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _invalid(message):
    return ExactTimeTransportError(REASON_TT_BITS_INVALID, message)


def encode_tt_bits(tt):
    """Return the exact IEEE-754 spelling of a Terrestrial Time state.

    ``tt`` is an exact binary64 Terrestrial Time Julian Date. The result is
    its big-endian IEEE-754 bit pattern as exactly TT_BITS_LENGTH uppercase
    hexadecimal characters, so a decoded value is bit-identical to the
    encoded one.

    bool is rejected before the numeric check because it is a subclass of
    int and would otherwise be silently accepted as 0.0 or 1.0.

    An integer too large to become a float raises OverflowError during
    conversion, which is a refusal to represent rather than a value, and is
    reported as such.

    Signed zero and subnormal values are preserved exactly. Non-finite
    values are refused: an infinity or a NaN is not a time state.
    """
    if isinstance(tt, bool) or not isinstance(tt, (int, float)):
        raise _invalid(
            "tt must be a real number, received %s" % type(tt).__name__
        )

    try:
        number = float(tt)
    except OverflowError as error:
        raise _invalid(
            "tt is too large to represent as a float: %s" % error
        ) from error

    if not math.isfinite(number):
        raise _invalid("tt must be finite, received %r" % (tt,))

    return struct.pack(">d", number).hex().upper()


def decode_tt_bits(value):
    """Return the exact Terrestrial Time state an IEEE-754 spelling denotes.

    ``value`` must be a string of exactly TT_BITS_LENGTH uppercase
    hexadecimal characters. Nothing is repaired, defaulted, trimmed or
    normalized: a lowercase spelling, a "0x" prefix, surrounding whitespace,
    a short or long string and a non-hexadecimal character are all refused
    rather than interpreted, because each of them is a different encoding
    than the one this module defines.

    The grammar is checked before the bytes are unpacked, so a malformed
    string is never partially interpreted.

    A pattern that denotes an infinity or a NaN is refused after unpacking.
    Such a pattern is well formed as hexadecimal and still does not denote a
    time state.

    Signed zero is preserved: 0000000000000000 decodes to +0.0 and
    8000000000000000 to -0.0, which are distinct binary64 states even though
    they compare equal.
    """
    if not isinstance(value, str):
        raise _invalid(
            "ttBits must be a string, received %s" % type(value).__name__
        )

    if len(value) != TT_BITS_LENGTH or not set(value) <= _HEX_UPPER:
        raise _invalid(
            "ttBits must be %d uppercase hex characters, received %r"
            % (TT_BITS_LENGTH, value)
        )

    number = struct.unpack(">d", bytes.fromhex(value))[0]

    if not math.isfinite(number):
        raise _invalid(
            "ttBits decodes to a non-finite value: %r" % (number,)
        )

    return number
