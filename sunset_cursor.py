"""HPC Astronomy Authority - Sunset Continuation Witness.

A1a - deterministic serialization and fail-closed validation of the sunset
continuation witness ("cursor").

WHAT THIS MODULE IS

A continuation witness records the exact solver start value from which a
subsequent sunset search may resume, together with the observer and the
scientific environment under which it was created. It is deterministic
serialization and validation only.

This module performs NO astronomy. It imports no ephemeris, builds no
timescale, evaluates no predicate, reads no BSP artifact, touches no
network and writes no file.

WHAT THE CURSOR IS NOT

The cursor is NOT physical-event identity. The same physical crossing can
yield different cursors under different search brackets, so cursor equality
is NOT proof of event equality.

The cursor is opaque at the API-contract level. Base64URL is transport
encoding, not secrecy: the cursor is NOT encrypted and NOT secret, and its
internal payload fields are NOT part of any public API contract.

The checksum detects ACCIDENTAL CORRUPTION only. It is not authentication,
not authorization, not signing, not proof of origin, and no protection
against any party able to recompute it. A cursor that validates is a cursor
whose bindings match the current environment - nothing more.

Whether a decoded cursor actually sits on the post-transition side of a
crossing is an astronomical question. It is NOT answered here; that
predicate belongs to the A1b solver wiring.

REPRESENTATION

    payload    11 string-valued fields (see CURSOR_PAYLOAD_FIELDS)
    envelope   {"checksum": <digest of payload>, "payload": {...}}
    transport  canonical JSON -> UTF-8 -> unpadded URL-safe Base64

Transport is canonical, not merely well formed. A cursor must be the exact
canonical unpadded Base64URL spelling of its own bytes, and those bytes must
be the exact canonical JSON encoding of the parsed envelope. Alternate
spellings that decode to identical content - unused Base64 padding bits,
added whitespace, different key ordering - are rejected as malformed rather
than treated as checksum failures.

The TT continuation value is carried as ttBits, the IEEE-754 binary64 bit
pattern in 16 uppercase hex characters. It is never converted through
datetime, ISO text, Unix milliseconds or decimal rounding, so a decoded
value equals the encoded value bit for bit.

Observer coordinates are carried as Python canonical float repr strings.
Signed zero is canonicalized to +0.0, so -0.0 and +0.0 denote the same
coordinate. A decoded coordinate string must equal repr of its own
canonical value, which makes the serialized observer binding unique:
"40.740600", "4.07406e1" and "-0.0" are all rejected. No other rounding,
truncation, quantization or normalization is applied.

No kernel filename is bound. Kernel selection remains per-computation
provenance, not cursor compatibility identity.

EXCEPTION BOUNDARY

Malformed cursor parsing, representation and validation failures are
normalized to SunsetCursorError. Duplicate JSON keys, non-standard JSON
constants, recursion limits and canonicalization failures all surface with
reason MALFORMED_CURSOR, and the original exception is preserved as
__cause__ where one exists.

BaseException and process- or resource-level failures are not intentionally
swallowed. KeyboardInterrupt, SystemExit and conditions such as MemoryError
propagate unchanged; no claim is made that every possible runtime failure is
converted.
"""

import base64
import binascii
import json
import math
import struct
from collections import namedtuple

from runtime_enforcement import CERTIFIED_RUNTIME_SCIENTIFIC_COMPONENTS
from scientific_environment import (
    AUTHORITY_SOLVER_GENERATION,
    DIGEST_PREFIX,
    HORIZON_MODEL_GENERATION,
    ScientificEnvironmentError,
    canonical_digest,
    canonical_json_bytes,
)

CURSOR_SCHEMA_GENERATION = "hpc-sunset-cursor-v1"

# Sorted, so every diagnostic and every comparison is deterministic.
CURSOR_PAYLOAD_FIELDS = (
    "authoritySolverGeneration",
    "cursorSchemaGeneration",
    "horizonModelGeneration",
    "iersDataSha256",
    "jplephemVersion",
    "latitude",
    "longitude",
    "numpyVersion",
    "pythonVersion",
    "skyfieldVersion",
    "ttBits",
)

CURSOR_ENVELOPE_FIELDS = ("checksum", "payload")

REASON_MALFORMED_CURSOR = "MALFORMED_CURSOR"
REASON_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
REASON_INCOMPATIBLE_ENVIRONMENT = "INCOMPATIBLE_ENVIRONMENT"
REASON_INCOMPATIBLE_OBSERVER = "INCOMPATIBLE_OBSERVER"

LATITUDE_DOMAIN = (-90.0, 90.0)
LONGITUDE_DOMAIN = (-180.0, 180.0)

TT_BITS_LENGTH = 16
_CHECKSUM_BODY_LENGTH = 64
_HEX_UPPER = frozenset("0123456789ABCDEF")
_HEX_LOWER = frozenset("0123456789abcdef")

# Unpadded URL-safe Base64 alphabet. Enforced explicitly before decoding so
# that "+", "/", "=", whitespace and every other character are rejected on
# syntax rather than relying on the decoder to notice.
_BASE64URL_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-_"
)

# Generation bindings owned or governed elsewhere. The five certified
# runtime components are taken wholesale from runtime_enforcement so that
# no scientific identity literal is duplicated here.
_GENERATION_BINDINGS = {
    "authoritySolverGeneration": AUTHORITY_SOLVER_GENERATION,
    "cursorSchemaGeneration": CURSOR_SCHEMA_GENERATION,
    "horizonModelGeneration": HORIZON_MODEL_GENERATION,
}

DecodedSunsetCursor = namedtuple(
    "DecodedSunsetCursor", ("tt", "latitude", "longitude")
)


class SunsetCursorError(ScientificEnvironmentError):
    """Sunset continuation-witness validation failed closed.

    ``reason`` carries a stable code so a later route can map every
    rejection onto one HTTP status with a distinguishing detail.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class _CursorParseError(ValueError):
    """Internal signal raised by the JSON hooks.

    Subclasses ValueError so the single parse boundary catches it alongside
    JSONDecodeError. It never escapes this module.
    """


def _malformed(message):
    return SunsetCursorError(
        REASON_MALFORMED_CURSOR, "SUNSET CURSOR MALFORMED - " + message
    )


def current_cursor_bindings():
    """Return the environment/generation bindings a cursor must carry."""
    bindings = dict(_GENERATION_BINDINGS)
    bindings.update(CERTIFIED_RUNTIME_SCIENTIFIC_COMPONENTS)
    return bindings


def _require_exact_fields(mapping, expected, what):
    missing = sorted(set(expected) - set(mapping))
    unexpected = sorted(set(mapping) - set(expected))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing %s" % missing)
        if unexpected:
            detail.append("unexpected %s" % unexpected)
        raise _malformed(
            "%s field set is not the expected set: %s" % (what, "; ".join(detail))
        )


def _to_float(value, field):
    """Convert to float, failing closed on type and overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(
            "%s must be a real number, received %s" % (field, type(value).__name__)
        )
    try:
        return float(value)
    except OverflowError as error:
        raise _malformed(
            "%s is too large to represent as a float: %s" % (field, error)
        ) from error


def _canonical_coordinate(value, field, domain):
    """Validate a coordinate and canonicalize signed zero to +0.0."""
    number = _to_float(value, field)

    if not math.isfinite(number):
        raise _malformed("%s must be finite, received %r" % (field, value))

    low, high = domain
    if number < low or number > high:
        raise _malformed(
            "%s must lie inclusively within [%r, %r], received %r"
            % (field, low, high, number)
        )

    if number == 0.0:
        return 0.0

    return number


def _parse_bound_coordinate(text, field, domain):
    """Parse a bound coordinate string and require canonical text."""
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise _malformed(
            "%s must be a decimal string, received %r" % (field, text)
        ) from error
    except OverflowError as error:
        raise _malformed(
            "%s is too large to represent as a float: %s" % (field, error)
        ) from error

    canonical = _canonical_coordinate(number, field, domain)

    if text != repr(canonical):
        raise _malformed(
            "%s must be the canonical representation %r, received %r"
            % (field, repr(canonical), text)
        )

    return canonical


def _encode_tt_bits(tt):
    number = _to_float(tt, "tt")

    if not math.isfinite(number):
        raise _malformed("tt must be finite, received %r" % (tt,))

    return struct.pack(">d", number).hex().upper()


def _decode_tt_bits(value):
    if len(value) != TT_BITS_LENGTH or not set(value) <= _HEX_UPPER:
        raise _malformed(
            "ttBits must be %d uppercase hex characters, received %r"
            % (TT_BITS_LENGTH, value)
        )

    number = struct.unpack(">d", bytes.fromhex(value))[0]

    if not math.isfinite(number):
        raise _malformed("ttBits decodes to a non-finite value: %r" % (number,))

    return number


def _require_checksum_form(value):
    if not isinstance(value, str):
        raise _malformed(
            "checksum must be a string, received %s" % type(value).__name__
        )
    if not value.startswith(DIGEST_PREFIX):
        raise _malformed(
            "checksum must start with %r, received %r" % (DIGEST_PREFIX, value)
        )

    body = value[len(DIGEST_PREFIX):]
    if len(body) != _CHECKSUM_BODY_LENGTH or not set(body) <= _HEX_LOWER:
        raise _malformed(
            "checksum must carry %d lowercase hex characters, received %r"
            % (_CHECKSUM_BODY_LENGTH, value)
        )


def _reject_duplicate_keys(pairs):
    """object_pairs_hook: refuse last-key-wins parsing at any nesting level."""
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise _CursorParseError("duplicate JSON key %r" % key)
        seen.add(key)
    return dict(pairs)


def _reject_json_constant(name):
    """parse_constant: refuse NaN, Infinity and -Infinity."""
    raise _CursorParseError("non-standard JSON constant %r is not permitted" % name)


def _safe_canonical_json_bytes(obj):
    """Canonicalize received content, mapping any serialization failure."""
    try:
        return canonical_json_bytes(obj)
    except SunsetCursorError:
        raise
    except (ScientificEnvironmentError, TypeError, ValueError,
            UnicodeEncodeError, RecursionError) as error:
        raise _malformed(
            "cursor content cannot be canonicalized: %s: %s"
            % (type(error).__name__, error)
        ) from error


def _safe_canonical_digest(payload):
    """Digest received payload, mapping any serialization failure."""
    try:
        return canonical_digest(payload)
    except SunsetCursorError:
        raise
    except (ScientificEnvironmentError, TypeError, ValueError,
            UnicodeEncodeError, RecursionError) as error:
        raise _malformed(
            "cursor payload cannot be canonicalized: %s: %s"
            % (type(error).__name__, error)
        ) from error


def _base64url_encode(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _base64url_decode(cursor):
    if not isinstance(cursor, str):
        raise _malformed(
            "cursor must be a string, received %s" % type(cursor).__name__
        )
    if not cursor:
        raise _malformed("cursor must not be empty")

    invalid = sorted(set(cursor) - _BASE64URL_ALPHABET)
    if invalid:
        raise _malformed(
            "cursor contains characters outside the unpadded Base64URL "
            "alphabet: %s" % invalid
        )

    if len(cursor) % 4 == 1:
        raise _malformed(
            "cursor length %d is not a valid unpadded Base64URL length" % len(cursor)
        )

    padded = cursor + ("=" * (-len(cursor) % 4))

    try:
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise _malformed("cursor is not valid Base64URL: %s" % error) from error

    if _base64url_encode(raw) != cursor:
        raise _malformed(
            "cursor is not the canonical unpadded Base64URL spelling of its "
            "own content"
        )

    return raw


def encode_sunset_cursor(*, tt, latitude, longitude):
    """Encode a continuation witness for the supplied solver state.

    ``tt`` is the exact binary64 Terrestrial Time Julian Date produced by
    the solver. It is stored as its IEEE-754 bit pattern, never as decimal
    text, so a decoded value is bit-identical to the encoded one.
    """
    payload = current_cursor_bindings()
    payload["latitude"] = repr(
        _canonical_coordinate(latitude, "latitude", LATITUDE_DOMAIN)
    )
    payload["longitude"] = repr(
        _canonical_coordinate(longitude, "longitude", LONGITUDE_DOMAIN)
    )
    payload["ttBits"] = _encode_tt_bits(tt)

    envelope = {"checksum": _safe_canonical_digest(payload), "payload": payload}

    return _base64url_encode(canonical_json_bytes(envelope))


def decode_sunset_cursor(cursor, *, latitude, longitude):
    """Decode and validate a continuation witness, failing closed.

    Validation order is fixed so that malformed structure always takes
    precedence over checksum classification:

        transport -> strict Base64URL syntax -> decode -> canonical
        Base64URL verification -> UTF-8 -> JSON hooks -> envelope/object
        structure -> payload object -> checksum lexical form -> canonical
        envelope JSON verification -> checksum recomputation -> payload
        schema/types -> values -> environment -> observer

    Nothing is repaired, defaulted or normalized beyond the documented
    signed-zero canonicalization.
    """
    raw = _base64url_decode(cursor)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _malformed("cursor content is not valid UTF-8: %s" % error) from error

    try:
        envelope = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError) as error:
        raise _malformed(
            "cursor content is not acceptable JSON: %s: %s"
            % (type(error).__name__, error)
        ) from error

    if not isinstance(envelope, dict):
        raise _malformed(
            "cursor envelope must be a JSON object, received %s"
            % type(envelope).__name__
        )

    _require_exact_fields(envelope, CURSOR_ENVELOPE_FIELDS, "cursor envelope")

    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise _malformed(
            "cursor payload must be a JSON object, received %s"
            % type(payload).__name__
        )

    checksum = envelope["checksum"]
    _require_checksum_form(checksum)

    if raw != _safe_canonical_json_bytes(envelope):
        raise _malformed(
            "cursor content is not the canonical JSON encoding of its own "
            "envelope"
        )

    if _safe_canonical_digest(payload) != checksum:
        raise SunsetCursorError(
            REASON_CHECKSUM_MISMATCH,
            "SUNSET CURSOR CHECKSUM MISMATCH - the payload does not match its "
            "recorded checksum; refusing to continue",
        )

    _require_exact_fields(payload, CURSOR_PAYLOAD_FIELDS, "cursor payload")

    for field in CURSOR_PAYLOAD_FIELDS:
        if not isinstance(payload[field], str):
            raise _malformed(
                "%s must be a string, received %s"
                % (field, type(payload[field]).__name__)
            )

    tt = _decode_tt_bits(payload["ttBits"])
    bound_latitude = _parse_bound_coordinate(
        payload["latitude"], "latitude", LATITUDE_DOMAIN
    )
    bound_longitude = _parse_bound_coordinate(
        payload["longitude"], "longitude", LONGITUDE_DOMAIN
    )
    requested_latitude = _canonical_coordinate(
        latitude, "requested latitude", LATITUDE_DOMAIN
    )
    requested_longitude = _canonical_coordinate(
        longitude, "requested longitude", LONGITUDE_DOMAIN
    )

    expected = current_cursor_bindings()
    mismatches = []
    for field in sorted(expected):
        if payload[field] != expected[field]:
            mismatches.append(
                "%s: cursor %r, current %r"
                % (field, payload[field], expected[field])
            )

    if mismatches:
        raise SunsetCursorError(
            REASON_INCOMPATIBLE_ENVIRONMENT,
            "SUNSET CURSOR ENVIRONMENT INCOMPATIBLE - the cursor was encoded "
            "under a different scientific environment; refusing to continue. "
            + "; ".join(mismatches),
        )

    if bound_latitude != requested_latitude or bound_longitude != requested_longitude:
        raise SunsetCursorError(
            REASON_INCOMPATIBLE_OBSERVER,
            "SUNSET CURSOR OBSERVER MISMATCH - the cursor is bound to "
            "latitude %r longitude %r, requested latitude %r longitude %r; "
            "refusing to continue"
            % (
                bound_latitude,
                bound_longitude,
                requested_latitude,
                requested_longitude,
            ),
        )

    return DecodedSunsetCursor(
        tt=tt, latitude=bound_latitude, longitude=bound_longitude
    )
