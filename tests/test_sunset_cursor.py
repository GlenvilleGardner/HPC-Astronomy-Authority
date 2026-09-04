"""A1a verification for sunset_cursor.py.

Standard library only: unittest.

INDEPENDENT ORACLE DISCIPLINE

Expected field names, generation tokens and certified runtime values are
declared as test-local literals. They are deliberately NOT imported from
sunset_cursor or runtime_enforcement: importing them would make the tests
agree with a defective constant instead of detecting it.

Tampered cursors are built with test-local canonical JSON and Base64URL
machinery rather than by calling the production encoder, so a defect in the
production encoding cannot confirm itself. The encoder is additionally held
against that independent construction character-for-character.

SCOPE

Astronomy-free. No ephemeris is loaded, no BSP artifact is read, no
timescale is built, no network is used and no file is written. A decoded
cursor is not asserted to be a valid post-transition witness; that predicate
belongs to A1b.
"""

import base64
import hashlib
import json
import struct
import unittest

import sunset_cursor
from sunset_cursor import (
    DecodedSunsetCursor,
    SunsetCursorError,
    decode_sunset_cursor,
    encode_sunset_cursor,
)
from scientific_environment import ScientificEnvironmentError


EXPECTED_PAYLOAD_FIELDS = (
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
EXPECTED_ENVELOPE_FIELDS = ("checksum", "payload")

EXPECTED_CURSOR_SCHEMA_GENERATION = "hpc-sunset-cursor-v1"
EXPECTED_AUTHORITY_SOLVER_GENERATION = "hpc-authority-solver-v1"
EXPECTED_HORIZON_MODEL_GENERATION = "hpc-apparent-sunset-v1"

EXPECTED_PYTHON_VERSION = "CPython 3.13.7"
EXPECTED_SKYFIELD_VERSION = "1.54"
EXPECTED_NUMPY_VERSION = "2.4.3"
EXPECTED_JPLEPHEM_VERSION = "2.24"
EXPECTED_IERS_DATA_SHA256 = (
    "C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C"
)

EXPECTED_RUNTIME_BINDINGS = {
    "iersDataSha256": EXPECTED_IERS_DATA_SHA256,
    "jplephemVersion": EXPECTED_JPLEPHEM_VERSION,
    "numpyVersion": EXPECTED_NUMPY_VERSION,
    "pythonVersion": EXPECTED_PYTHON_VERSION,
    "skyfieldVersion": EXPECTED_SKYFIELD_VERSION,
}

# MEASURED - the TT value of the NYC 2027-03-20 sunset crossing, and its
# IEEE-754 binary64 bit pattern.
REFERENCE_TT = 2461485.4645259595
REFERENCE_TT_BITS = "4142C796BB75962E"

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9586

# Yonkers - roughly 21 km away. Previously MEASURED to produce a plausible
# 24.0141 h separation that an interval sanity check would not reject,
# which is why observer binding must reject it here.
ADJACENT_LATITUDE = 40.9312
ADJACENT_LONGITUDE = -73.8988


def independent_canonical_bytes(obj):
    """Re-implement the canonical encoding, independently of production."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def independent_digest(obj):
    return "sha256:" + hashlib.sha256(independent_canonical_bytes(obj)).hexdigest()


def independent_encode_bytes(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def reference_payload(**overrides):
    payload = {
        "authoritySolverGeneration": EXPECTED_AUTHORITY_SOLVER_GENERATION,
        "cursorSchemaGeneration": EXPECTED_CURSOR_SCHEMA_GENERATION,
        "horizonModelGeneration": EXPECTED_HORIZON_MODEL_GENERATION,
        "latitude": repr(REFERENCE_LATITUDE),
        "longitude": repr(REFERENCE_LONGITUDE),
        "ttBits": REFERENCE_TT_BITS,
    }
    payload.update(EXPECTED_RUNTIME_BINDINGS)
    payload.update(overrides)
    return payload


def independent_cursor(payload=None, checksum=None):
    """Build a cursor without using the production encoder."""
    body = reference_payload() if payload is None else payload
    envelope = {
        "checksum": independent_digest(body) if checksum is None else checksum,
        "payload": body,
    }
    return independent_encode_bytes(independent_canonical_bytes(envelope))


def cursor_from_text(text):
    """Encode arbitrary JSON text, bypassing canonical construction."""
    return independent_encode_bytes(text.encode("utf-8"))


def decode_reference(cursor):
    return decode_sunset_cursor(
        cursor, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
    )


class RejectionMixin:
    def assert_rejected(self, cursor, reason, **observer):
        coordinates = {
            "latitude": REFERENCE_LATITUDE,
            "longitude": REFERENCE_LONGITUDE,
        }
        coordinates.update(observer)
        with self.assertRaises(SunsetCursorError) as caught:
            decode_sunset_cursor(cursor, **coordinates)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception


class TestCursorContract(unittest.TestCase):

    def test_error_is_a_scientific_environment_error(self):
        self.assertTrue(issubclass(SunsetCursorError, ScientificEnvironmentError))

    def test_payload_field_set_matches_independent_literals(self):
        self.assertEqual(sunset_cursor.CURSOR_PAYLOAD_FIELDS, EXPECTED_PAYLOAD_FIELDS)
        self.assertEqual(len(EXPECTED_PAYLOAD_FIELDS), 11)

    def test_envelope_field_set_matches_independent_literals(self):
        self.assertEqual(sunset_cursor.CURSOR_ENVELOPE_FIELDS, EXPECTED_ENVELOPE_FIELDS)


class TestEncode(unittest.TestCase):

    def encoded(self):
        return encode_sunset_cursor(
            tt=REFERENCE_TT,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )

    def envelope_of(self, cursor):
        padded = cursor + ("=" * (-len(cursor) % 4))
        return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))

    def test_encode_is_deterministic(self):
        self.assertEqual(self.encoded(), self.encoded())

    def test_encode_matches_independent_construction(self):
        # Stronger than encoding twice: proves the production encoder agrees
        # with independently implemented canonical JSON + digest + Base64URL.
        self.assertEqual(self.encoded(), independent_cursor())

    def test_checksum_is_deterministic(self):
        envelope = self.envelope_of(self.encoded())
        self.assertEqual(envelope["checksum"], independent_digest(envelope["payload"]))

    def test_all_five_runtime_bindings_present(self):
        payload = self.envelope_of(self.encoded())["payload"]
        for field, value in EXPECTED_RUNTIME_BINDINGS.items():
            with self.subTest(field=field):
                self.assertEqual(payload[field], value)

    def test_generation_bindings_present(self):
        payload = self.envelope_of(self.encoded())["payload"]
        self.assertEqual(
            payload["cursorSchemaGeneration"], EXPECTED_CURSOR_SCHEMA_GENERATION
        )
        self.assertEqual(
            payload["authoritySolverGeneration"],
            EXPECTED_AUTHORITY_SOLVER_GENERATION,
        )
        self.assertEqual(
            payload["horizonModelGeneration"], EXPECTED_HORIZON_MODEL_GENERATION
        )

    def test_no_kernel_field_is_bound(self):
        self.assertNotIn("kernel", self.envelope_of(self.encoded())["payload"])


class TestRoundTrip(unittest.TestCase):

    def encoded(self, tt=REFERENCE_TT, latitude=REFERENCE_LATITUDE,
                longitude=REFERENCE_LONGITUDE):
        return encode_sunset_cursor(tt=tt, latitude=latitude, longitude=longitude)

    def test_round_trip_returns_decoded_cursor(self):
        self.assertIsInstance(decode_reference(self.encoded()), DecodedSunsetCursor)

    def test_tt_is_preserved_bit_for_bit(self):
        decoded = decode_reference(self.encoded())
        self.assertEqual(
            struct.pack(">d", decoded.tt), struct.pack(">d", REFERENCE_TT)
        )
        self.assertEqual(decoded.tt, REFERENCE_TT)

    def test_observer_is_preserved_exactly(self):
        decoded = decode_reference(self.encoded())
        self.assertEqual(decoded.latitude, REFERENCE_LATITUDE)
        self.assertEqual(decoded.longitude, REFERENCE_LONGITUDE)

    def test_signed_zero_observer_is_canonicalized(self):
        negative = self.encoded(latitude=-0.0, longitude=-0.0)
        positive = self.encoded(latitude=0.0, longitude=0.0)
        self.assertEqual(negative, positive)

        decoded = decode_sunset_cursor(negative, latitude=0.0, longitude=0.0)
        self.assertEqual(struct.pack(">d", decoded.latitude), struct.pack(">d", 0.0))
        self.assertEqual(struct.pack(">d", decoded.longitude), struct.pack(">d", 0.0))

        # A cursor encoded at +0.0 must also accept a -0.0 request.
        decode_sunset_cursor(positive, latitude=-0.0, longitude=-0.0)


class TestTransportRejection(RejectionMixin, unittest.TestCase):

    def test_non_string_cursor_rejected(self):
        for bad in (None, 1, b"abc", []):
            with self.subTest(value=type(bad).__name__):
                self.assert_rejected(bad, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_empty_cursor_rejected(self):
        self.assert_rejected("", sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_base64_length_mod_four_of_one_rejected(self):
        self.assertEqual(len("A") % 4, 1)
        self.assert_rejected("A", sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_standard_base64_plus_rejected(self):
        self.assert_rejected("ab+d", sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_standard_base64_slash_rejected(self):
        self.assert_rejected("ab/d", sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_supplied_padding_rejected(self):
        padded = independent_cursor()
        padded = padded + ("=" * (-len(padded) % 4 or 4))
        self.assert_rejected(padded, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_whitespace_rejected(self):
        for bad in ("ab cd", "ab\ncd", "ab\tcd", " abcd"):
            with self.subTest(value=repr(bad)):
                self.assert_rejected(bad, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_other_invalid_characters_rejected(self):
        for bad in ("ab!d", "ab.d", "abéd"):
            with self.subTest(value=bad):
                self.assert_rejected(bad, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_noncanonical_base64_spelling_rejected(self):
        canonical = independent_encode_bytes(b"\x00")   # "AA"
        alternate = canonical[:-1] + "B"                # "AB"

        self.assertNotEqual(alternate, canonical)

        def raw_bytes(token):
            return base64.b64decode(
                (token + "=" * (-len(token) % 4)).encode("ascii"),
                altchars=b"-_",
                validate=True,
            )

        # Both spellings decode to identical bytes; only the unused trailing
        # Base64 bits differ.
        self.assertEqual(raw_bytes(alternate), raw_bytes(canonical))

        self.assert_rejected(alternate, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_invalid_utf8_rejected(self):
        self.assert_rejected(
            independent_encode_bytes(b"\xff\xfe\xfd\xfc"),
            sunset_cursor.REASON_MALFORMED_CURSOR,
        )


class TestJsonRejection(RejectionMixin, unittest.TestCase):

    def test_invalid_json_rejected(self):
        self.assert_rejected(
            cursor_from_text("{not json"), sunset_cursor.REASON_MALFORMED_CURSOR
        )

    def test_non_object_json_rejected(self):
        for text in ("[]", '"text"', "7", "null"):
            with self.subTest(value=text):
                self.assert_rejected(
                    cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
                )

    def test_duplicate_envelope_key_rejected(self):
        payload = json.dumps(reference_payload())
        text = (
            '{"checksum":"sha256:%s","checksum":"sha256:%s","payload":%s}'
            % ("0" * 64, "1" * 64, payload)
        )
        self.assert_rejected(
            cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
        )

    def test_duplicate_payload_key_rejected(self):
        text = (
            '{"checksum":"sha256:%s","payload":'
            '{"skyfieldVersion":"1.54","skyfieldVersion":"1.53"}}' % ("0" * 64)
        )
        self.assert_rejected(
            cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
        )

    def test_duplicate_key_rejection_preserves_cause(self):
        text = (
            '{"checksum":"sha256:%s","checksum":"sha256:%s","payload":{}}'
            % ("0" * 64, "1" * 64)
        )
        error = self.assert_rejected(
            cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
        )
        self.assertIsNotNone(error.__cause__)

    def test_json_nan_rejected(self):
        text = '{"checksum":"sha256:%s","payload":{"latitude":NaN}}' % ("0" * 64)
        self.assert_rejected(
            cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
        )

    def test_json_infinity_rejected(self):
        for literal in ("Infinity", "-Infinity"):
            with self.subTest(value=literal):
                text = (
                    '{"checksum":"sha256:%s","payload":{"latitude":%s}}'
                    % ("0" * 64, literal)
                )
                self.assert_rejected(
                    cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
                )

    def test_noncanonical_json_transport_rejected(self):
        payload = reference_payload()
        envelope = {"checksum": independent_digest(payload), "payload": payload}

        # Payload and checksum stay valid; only the transport spelling differs.
        for label, text in (
            ("whitespace", json.dumps(envelope, sort_keys=True,
                                      separators=(", ", ": "))),
            ("key order", json.dumps(envelope, sort_keys=False,
                                     separators=(",", ":"))),
        ):
            with self.subTest(variant=label):
                self.assert_rejected(
                    cursor_from_text(text), sunset_cursor.REASON_MALFORMED_CURSOR
                )


class TestEnvelopeRejection(RejectionMixin, unittest.TestCase):

    def test_envelope_shape_rejected(self):
        payload = reference_payload()
        for envelope in (
            {"payload": payload},
            {"checksum": independent_digest(payload)},
            {
                "checksum": independent_digest(payload),
                "payload": payload,
                "extra": "x",
            },
        ):
            with self.subTest(keys=sorted(envelope)):
                cursor = independent_encode_bytes(
                    independent_canonical_bytes(envelope)
                )
                self.assert_rejected(cursor, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_non_object_payload_rejected(self):
        envelope = {"checksum": "sha256:" + "0" * 64, "payload": "not an object"}
        cursor = independent_encode_bytes(independent_canonical_bytes(envelope))
        self.assert_rejected(cursor, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_non_string_checksum_rejected(self):
        for bad in (7, None, ["sha256:" + "0" * 64], {"digest": "x"}):
            with self.subTest(value=type(bad).__name__):
                envelope = {"checksum": bad, "payload": reference_payload()}
                cursor = independent_encode_bytes(
                    independent_canonical_bytes(envelope)
                )
                self.assert_rejected(cursor, sunset_cursor.REASON_MALFORMED_CURSOR)

    def test_malformed_checksum_field_rejected(self):
        payload = reference_payload()
        for bad in ("md5:" + "0" * 64, "0" * 64, "sha256:" + "0" * 63, "sha256:", ""):
            with self.subTest(value=bad):
                self.assert_rejected(
                    independent_cursor(payload, checksum=bad),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_uppercase_checksum_rejected(self):
        self.assert_rejected(
            independent_cursor(reference_payload(), checksum="sha256:" + "A" * 64),
            sunset_cursor.REASON_MALFORMED_CURSOR,
        )

    def test_checksum_corruption_rejected(self):
        wrong = independent_digest(reference_payload(ttBits="4142C79680000000"))
        self.assert_rejected(
            independent_cursor(reference_payload(), checksum=wrong),
            sunset_cursor.REASON_CHECKSUM_MISMATCH,
        )


class TestPayloadShapeRejection(RejectionMixin, unittest.TestCase):

    def test_missing_field_rejected(self):
        payload = reference_payload()
        payload.pop("skyfieldVersion")
        self.assert_rejected(
            independent_cursor(payload), sunset_cursor.REASON_MALFORMED_CURSOR
        )

    def test_unexpected_field_rejected(self):
        self.assert_rejected(
            independent_cursor(reference_payload(kernel="de440.bsp")),
            sunset_cursor.REASON_MALFORMED_CURSOR,
        )

    def test_wrong_field_type_rejected(self):
        for field, bad in (
            ("latitude", 40.7406),
            ("ttBits", 1),
            ("skyfieldVersion", None),
            ("numpyVersion", ["2.4.3"]),
        ):
            with self.subTest(field=field):
                self.assert_rejected(
                    independent_cursor(reference_payload(**{field: bad})),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_malformed_tt_bits_rejected(self):
        for bad in (
            "",
            "4142C796",
            "4142C796BB75962EAA",
            "4142c796bb75962e",
            "ZZZZZZZZZZZZZZZZ",
        ):
            with self.subTest(value=bad):
                self.assert_rejected(
                    independent_cursor(reference_payload(ttBits=bad)),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )


class TestValueRejection(RejectionMixin, unittest.TestCase):

    def assert_encode_rejected(self, **overrides):
        parameters = {
            "tt": REFERENCE_TT,
            "latitude": REFERENCE_LATITUDE,
            "longitude": REFERENCE_LONGITUDE,
        }
        parameters.update(overrides)
        with self.assertRaises(SunsetCursorError) as caught:
            encode_sunset_cursor(**parameters)
        self.assertEqual(
            caught.exception.reason, sunset_cursor.REASON_MALFORMED_CURSOR
        )
        return caught.exception

    def test_non_finite_tt_rejected_at_encode(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                self.assert_encode_rejected(tt=bad)

    def test_non_finite_reconstructed_tt_rejected_at_decode(self):
        for bits in ("7FF0000000000000", "FFF0000000000000", "7FF8000000000000"):
            with self.subTest(value=bits):
                self.assert_rejected(
                    independent_cursor(reference_payload(ttBits=bits)),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_huge_numeric_tt_overflow_rejected(self):
        self.assert_encode_rejected(tt=10 ** 400)

    def test_huge_numeric_coordinate_overflow_rejected(self):
        for field in ("latitude", "longitude"):
            with self.subTest(field=field):
                self.assert_encode_rejected(**{field: 10 ** 400})

    def test_inclusive_coordinate_endpoints_are_accepted(self):
        for latitude, longitude in (
            (-90.0, -180.0),
            (-90.0, 180.0),
            (90.0, -180.0),
            (90.0, 180.0),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                cursor = encode_sunset_cursor(
                    tt=REFERENCE_TT, latitude=latitude, longitude=longitude
                )
                decoded = decode_sunset_cursor(
                    cursor, latitude=latitude, longitude=longitude
                )
                self.assertEqual(decoded.latitude, latitude)
                self.assertEqual(decoded.longitude, longitude)

    def test_coordinates_outside_domain_rejected_at_encode(self):
        for latitude, longitude in (
            (90.0000001, 0.0),
            (-90.0000001, 0.0),
            (0.0, 180.0000001),
            (0.0, -180.0000001),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assert_encode_rejected(latitude=latitude, longitude=longitude)

    def test_coordinates_outside_domain_rejected_at_decode(self):
        for field, value in (("latitude", "90.5"), ("longitude", "-180.5")):
            with self.subTest(field=field):
                self.assert_rejected(
                    independent_cursor(reference_payload(**{field: value})),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_non_finite_observer_rejected_at_encode(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                self.assert_encode_rejected(latitude=bad)

    def test_non_numeric_bound_coordinate_rejected(self):
        self.assert_rejected(
            independent_cursor(reference_payload(latitude="north")),
            sunset_cursor.REASON_MALFORMED_CURSOR,
        )

    def test_noncanonical_latitude_representation_rejected(self):
        for bad in ("40.740600", "4.07406e1", "+40.7406", " 40.7406"):
            with self.subTest(value=bad):
                self.assert_rejected(
                    independent_cursor(reference_payload(latitude=bad)),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_noncanonical_longitude_representation_rejected(self):
        for bad in ("-73.958600", "-7.39586e1", "-73.9586000"):
            with self.subTest(value=bad):
                self.assert_rejected(
                    independent_cursor(reference_payload(longitude=bad)),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                )

    def test_negative_zero_observer_text_rejected(self):
        for field in ("latitude", "longitude"):
            with self.subTest(field=field):
                self.assert_rejected(
                    independent_cursor(reference_payload(**{field: "-0.0"})),
                    sunset_cursor.REASON_MALFORMED_CURSOR,
                    **{field: 0.0}
                )


class TestEnvironmentRejection(RejectionMixin, unittest.TestCase):

    def test_each_runtime_component_mismatch_rejected(self):
        for field in sorted(EXPECTED_RUNTIME_BINDINGS):
            with self.subTest(field=field):
                error = self.assert_rejected(
                    independent_cursor(reference_payload(**{field: "drifted-value"})),
                    sunset_cursor.REASON_INCOMPATIBLE_ENVIRONMENT,
                )
                self.assertIn(field, str(error))

    def test_solver_generation_mismatch_rejected(self):
        self.assert_rejected(
            independent_cursor(
                reference_payload(
                    authoritySolverGeneration="hpc-authority-solver-v2"
                )
            ),
            sunset_cursor.REASON_INCOMPATIBLE_ENVIRONMENT,
        )

    def test_horizon_generation_mismatch_rejected(self):
        self.assert_rejected(
            independent_cursor(
                reference_payload(horizonModelGeneration="hpc-apparent-sunset-v2")
            ),
            sunset_cursor.REASON_INCOMPATIBLE_ENVIRONMENT,
        )

    def test_cursor_schema_mismatch_rejected(self):
        self.assert_rejected(
            independent_cursor(
                reference_payload(cursorSchemaGeneration="hpc-sunset-cursor-v2")
            ),
            sunset_cursor.REASON_INCOMPATIBLE_ENVIRONMENT,
        )


class TestObserverRejection(RejectionMixin, unittest.TestCase):

    def test_adjacent_observer_rejected(self):
        self.assert_rejected(
            independent_cursor(),
            sunset_cursor.REASON_INCOMPATIBLE_OBSERVER,
            latitude=ADJACENT_LATITUDE,
            longitude=ADJACENT_LONGITUDE,
        )

    def test_exact_observer_accepted(self):
        decoded = decode_reference(independent_cursor())
        self.assertEqual(decoded.latitude, REFERENCE_LATITUDE)
        self.assertEqual(decoded.longitude, REFERENCE_LONGITUDE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
