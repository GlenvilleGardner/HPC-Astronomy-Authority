"""A3c-0 verification for exact_time_transport.py.

Covers encode_tt_bits, decode_tt_bits, the transport failure taxonomy, and
the certification that sunset_cursor.py's delegation left the published A1
continuation-witness wire contract unchanged.

Test framework: standard-library unittest.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE

The expected bit patterns below are LITERALS. They are not produced by
calling struct.pack in the test, because a codec validated against a
re-implementation of itself would agree with its own defects. They were
transcribed from the published pre-delegation cursor codec and are the exact
strings the A1 wire contract has always carried.

The reason code and the hex length are likewise declared here as test-local
literals rather than imported from the module under test.

WHY THE CURSOR CORPUS IS HERE

The only field of a continuation witness that this increment can affect is
payload.ttBits. The cursor's bytes are the canonical JSON of its envelope,
and its checksum is computed over the payload, so a ttBits string that is
byte-identical makes the payload, the checksum and the whole cursor
byte-identical. Pinning ttBits for a broad corpus therefore certifies the
published wire contract exactly, without pinning environment-dependent
cursor strings that would break on any certified-runtime change.
"""

import ast
import inspect
import math
import struct
import unittest

import exact_time_transport
import sunset_cursor
from exact_time_transport import (
    ExactTimeTransportError,
    decode_tt_bits,
    encode_tt_bits,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"
REASON_MALFORMED_CURSOR = "MALFORMED_CURSOR"
EXPECTED_TT_BITS_LENGTH = 16

# (label, exact TT value, the bit pattern the A1 wire contract carries)
PINNED_TT_BITS = (
    ("modern event", 2460754.8768300507, "4142C629703BF794"),
    ("sunset", 2460666.3976841737, "4142C5FD32E750A4"),
    ("positive zero", 0.0, "0000000000000000"),
    ("negative zero", -0.0, "8000000000000000"),
    ("min subnormal", 5e-324, "0000000000000001"),
    ("max subnormal", 2.225073858507201e-308, "000FFFFFFFFFFFFF"),
    ("min normal", 2.2250738585072014e-308, "0010000000000000"),
    ("max float", 1.7976931348623157e308, "7FEFFFFFFFFFFFFF"),
    ("de441-1 floor", -3100015.5, "C147A6B7C0000000"),
    ("de441-2 ceiling", 8000016.5, "415E848420000000"),
    ("deep time event", 700287.9296739673, "41255EFFDBFE39EB"),
    ("deep negative", -1000000.0, "C12E848000000000"),
    ("utc lower edge", 1721425.500489, "413A445180200C0F"),
    ("utc upper edge", 5373484.5008, "41547F8B200D1B71"),
    ("one", 1.0, "3FF0000000000000"),
    ("minus one", -1.0, "BFF0000000000000"),
)

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9

CURSOR_OBSERVERS = (
    (40.7406, -73.9),
    (0.0, 0.0),
    (-89.9, 179.9),
    (89.9, -179.9),
)


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def executable_source(text):
    """Return a module's executable logic, prose excluded.

    A governed module names the mechanisms it excludes, in order to state
    that it excludes them. Scanning raw source for a prohibited token would
    therefore convict a module of the very thing its documentation promises
    it does not do. The established repository mechanism strips docstrings
    through the AST and drops comment-only lines, so a structural guard
    inspects logic rather than prose.
    """
    lines = text.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            drop.update(range(first.lineno - 1, first.end_lineno))
    kept = (l for i, l in enumerate(lines) if i not in drop)
    return "".join(l for l in kept if not l.lstrip().startswith("#"))


class TransportAssertions(unittest.TestCase):
    def assert_transport_rejects(self, function, value):
        with self.assertRaises(ExactTimeTransportError) as caught:
            function(value)
        self.assertEqual(caught.exception.reason, REASON_TT_BITS_INVALID)
        return caught.exception


# --- 1. Encode / decode round trip -----------------------------------------


class TestExactRoundTrip(TransportAssertions):
    def test_every_pinned_value_encodes_to_its_pinned_pattern(self):
        for label, value, expected in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(encode_tt_bits(value), expected)

    def test_every_pinned_pattern_decodes_to_its_pinned_value(self):
        for label, value, pattern in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(bits(decode_tt_bits(pattern)), bits(value))

    def test_round_trip_is_bit_identical(self):
        for label, value, _pattern in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(bits(decode_tt_bits(encode_tt_bits(value))),
                                 bits(value))

    def test_round_trip_over_many_arbitrary_bit_patterns(self):
        """Exhaustive in spirit: every exponent, sign and mantissa shape."""
        checked = 0
        for exponent in range(1, 2047, 37):
            for mantissa in (0, 1, 0x5555555555, 0xFFFFFFFFFFFFF):
                for sign in (0, 1):
                    packed = (sign << 63) | (exponent << 52) | mantissa
                    value = struct.unpack(">d", struct.pack(">Q", packed))[0]
                    if not math.isfinite(value):
                        continue
                    self.assertEqual(
                        bits(decode_tt_bits(encode_tt_bits(value))), bits(value)
                    )
                    checked += 1
        self.assertGreater(checked, 200)

    def test_encoded_form_is_always_the_declared_length(self):
        for label, value, _pattern in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(len(encode_tt_bits(value)),
                                 EXPECTED_TT_BITS_LENGTH)


# --- 2. Signed zero and subnormals -----------------------------------------


class TestSignedZeroAndSubnormals(TransportAssertions):
    def test_positive_and_negative_zero_stay_bit_distinct(self):
        self.assertNotEqual(encode_tt_bits(0.0), encode_tt_bits(-0.0))
        self.assertEqual(encode_tt_bits(0.0), "0000000000000000")
        self.assertEqual(encode_tt_bits(-0.0), "8000000000000000")

    def test_signed_zero_survives_decoding(self):
        self.assertEqual(bits(decode_tt_bits("0000000000000000")), bits(0.0))
        self.assertEqual(bits(decode_tt_bits("8000000000000000")), bits(-0.0))
        # They compare equal as numbers, which is exactly why bits are the
        # identity and a numeric comparison is not.
        self.assertEqual(decode_tt_bits("8000000000000000"), 0.0)
        self.assertNotEqual(bits(decode_tt_bits("8000000000000000")), bits(0.0))

    def test_negative_zero_is_not_canonicalized(self):
        self.assertEqual(
            bits(decode_tt_bits(encode_tt_bits(-0.0))), bits(-0.0)
        )
        self.assertEqual(math.copysign(1.0, decode_tt_bits(encode_tt_bits(-0.0))),
                         -1.0)

    def test_subnormal_values_are_preserved(self):
        for value in (5e-324, 1e-320, 2.225073858507201e-308):
            with self.subTest(value=value):
                self.assertEqual(
                    bits(decode_tt_bits(encode_tt_bits(value))), bits(value)
                )


# --- 3. Encode rejection taxonomy ------------------------------------------


class TestEncodeRejection(TransportAssertions):
    def test_bool_is_rejected(self):
        for value in (True, False):
            with self.subTest(value=value):
                self.assert_transport_rejects(encode_tt_bits, value)

    def test_non_numeric_is_rejected(self):
        for value in (None, "2460754.0", b"2460754.0", [2460754.0],
                      {"tt": 1.0}, object()):
            with self.subTest(value=repr(value)):
                self.assert_transport_rejects(encode_tt_bits, value)

    def test_nan_is_rejected(self):
        self.assert_transport_rejects(encode_tt_bits, float("nan"))

    def test_infinities_are_rejected(self):
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assert_transport_rejects(encode_tt_bits, value)

    def test_overflowing_integer_is_rejected(self):
        self.assert_transport_rejects(encode_tt_bits, 10 ** 400)
        self.assert_transport_rejects(encode_tt_bits, -(10 ** 400))

    def test_plain_integers_are_accepted(self):
        self.assertEqual(encode_tt_bits(1), encode_tt_bits(1.0))
        self.assertEqual(encode_tt_bits(2460754), "4142C62900000000")


# --- 4. Decode grammar -----------------------------------------------------


class TestDecodeGrammar(TransportAssertions):
    def test_non_string_is_rejected(self):
        for value in (None, 1, 1.0, True, b"4142C629703BF794",
                      ["4142C629703BF794"]):
            with self.subTest(value=repr(value)):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_wrong_length_is_rejected(self):
        for value in ("", "4142C796", "4142C629703BF79",
                      "4142C629703BF7944", "4142C796BB75962EAA"):
            with self.subTest(value=value):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_lowercase_and_mixed_case_are_rejected(self):
        for value in ("4142c629703bf794", "4142C629703bf794",
                      "4142c629703BF794"):
            with self.subTest(value=value):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_non_hexadecimal_is_rejected(self):
        for value in ("ZZZZZZZZZZZZZZZZ", "4142C629703BF79G",
                      "4142 629703BF794", "4142-629703BF794"):
            with self.subTest(value=value):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_prefixed_forms_are_rejected(self):
        for value in ("0x4142C629703BF7", "0X4142C629703BF7",
                      "#4142C629703BF79"):
            with self.subTest(value=value):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_surrounding_whitespace_is_rejected(self):
        for value in (" 4142C629703BF794", "4142C629703BF794 ",
                      "\t4142C629703BF79"):
            with self.subTest(value=repr(value)):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_non_finite_patterns_are_rejected(self):
        for value in ("7FF0000000000000", "FFF0000000000000",
                      "7FF8000000000000", "FFF8000000000000",
                      "7FFFFFFFFFFFFFFF"):
            with self.subTest(value=value):
                self.assert_transport_rejects(decode_tt_bits, value)

    def test_the_grammar_is_checked_before_the_bytes_are_unpacked(self):
        """A malformed string is never partially interpreted."""
        error = self.assert_transport_rejects(decode_tt_bits, "ZZZZZZZZZZZZZZZZ")
        self.assertIn("uppercase hex", str(error))


# --- 5. Failure contract ---------------------------------------------------


class TestFailureContract(TransportAssertions):
    def test_error_is_a_scientific_environment_error(self):
        with self.assertRaises(ScientificEnvironmentError):
            encode_tt_bits(float("nan"))

    def test_every_failure_carries_the_transport_reason(self):
        for function, value in ((encode_tt_bits, float("inf")),
                                (encode_tt_bits, None),
                                (decode_tt_bits, "nope"),
                                (decode_tt_bits, "7FF0000000000000")):
            with self.subTest(function=function.__name__, value=repr(value)):
                self.assert_transport_rejects(function, value)

    def test_the_transport_reason_is_not_an_astronomical_reason(self):
        """TT_BITS_INVALID must never be mistaken for a solver outcome."""
        import astronomy_solver

        astronomical = {
            astronomy_solver.REASON_INSTANT_STATE_INVALID,
            astronomy_solver.REASON_EVENT_KIND_INVALID,
            astronomy_solver.REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            astronomy_solver.REASON_EPHEMERIS_REACH_EXHAUSTED,
            astronomy_solver.REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED,
        }
        self.assertNotIn(REASON_TT_BITS_INVALID, astronomical)
        self.assertEqual(
            exact_time_transport.REASON_TT_BITS_INVALID,
            REASON_TT_BITS_INVALID,
        )


# --- 6. Structural isolation of the codec ----------------------------------


class TestCodecIsolation(unittest.TestCase):
    def setUp(self):
        self.source = executable_source(
            inspect.getsource(exact_time_transport)
        )

    def test_module_performs_no_astronomy(self):
        for token in ("skyfield", "almanac", "astronomy_solver", "load(",
                      "timescale", "ephemeris", "tt_jd", "kernel", ".bsp",
                      "wgs84", "latlon", "observer"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_knows_nothing_about_cursors_or_http(self):
        for token in ("cursor", "checksum", "base64", "HTTPException",
                      "status_code", "detail", "route"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_knows_nothing_about_calendars(self):
        for token in ("datetime", "isoformat", "strftime", "utc", "year",
                      "Gregorian", "timezone", "timestamp", "round("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_imports_only_stdlib_and_the_error_base(self):
        import ast

        tree = ast.parse(self.source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"math", "struct", "scientific_environment"})

    def test_importing_the_codec_builds_no_timescale(self):
        """The whole reason this module exists separately."""
        import sys

        self.assertIn("exact_time_transport", sys.modules)
        module = sys.modules["exact_time_transport"]
        self.assertFalse(hasattr(module, "ts"))
        self.assertFalse(hasattr(module, "load"))


# --- 7. A1 continuation-witness wire contract is unchanged ------------------


class TestCursorWireContractPreserved(unittest.TestCase):
    """The delegation must be invisible to every published cursor consumer."""

    def test_cursor_carries_the_pinned_bits_for_every_corpus_value(self):
        for label, value, expected in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(sunset_cursor._encode_tt_bits(value), expected)

    def test_cursor_decodes_the_pinned_bits_exactly(self):
        for label, value, pattern in PINNED_TT_BITS:
            with self.subTest(case=label):
                self.assertEqual(
                    bits(sunset_cursor._decode_tt_bits(pattern)), bits(value)
                )

    def test_public_cursor_round_trip_is_bit_identical(self):
        for label, value, _pattern in PINNED_TT_BITS:
            for latitude, longitude in CURSOR_OBSERVERS:
                with self.subTest(case=label, observer=(latitude, longitude)):
                    cursor = sunset_cursor.encode_sunset_cursor(
                        tt=value, latitude=latitude, longitude=longitude
                    )
                    decoded = sunset_cursor.decode_sunset_cursor(
                        cursor, latitude=latitude, longitude=longitude
                    )
                    self.assertEqual(bits(decoded.tt), bits(value))

    def test_a_cursor_is_reproducible_byte_for_byte(self):
        """Same state in, same bytes out - the wire contract itself."""
        for label, value, _pattern in PINNED_TT_BITS:
            with self.subTest(case=label):
                first = sunset_cursor.encode_sunset_cursor(
                    tt=value,
                    latitude=REFERENCE_LATITUDE,
                    longitude=REFERENCE_LONGITUDE,
                )
                second = sunset_cursor.encode_sunset_cursor(
                    tt=value,
                    latitude=REFERENCE_LATITUDE,
                    longitude=REFERENCE_LONGITUDE,
                )
                self.assertEqual(first, second)

    def test_cursor_signed_zero_states_produce_different_cursors(self):
        positive = sunset_cursor.encode_sunset_cursor(
            tt=0.0, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
        )
        negative = sunset_cursor.encode_sunset_cursor(
            tt=-0.0, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
        )
        self.assertNotEqual(positive, negative)

    def test_cursor_failures_keep_the_cursor_reason(self):
        """No generic transport reason may leak through the cursor API."""
        for value in (float("nan"), float("inf"), None, True, 10 ** 400):
            with self.subTest(value=repr(value)):
                with self.assertRaises(sunset_cursor.SunsetCursorError) as caught:
                    sunset_cursor.encode_sunset_cursor(
                        tt=value,
                        latitude=REFERENCE_LATITUDE,
                        longitude=REFERENCE_LONGITUDE,
                    )
                self.assertEqual(caught.exception.reason, REASON_MALFORMED_CURSOR)
                self.assertNotEqual(
                    caught.exception.reason, REASON_TT_BITS_INVALID
                )

    def test_cursor_decode_failures_keep_the_cursor_reason(self):
        for value in ("4142c629703bf794", "4142C796", "ZZZZZZZZZZZZZZZZ",
                      "7FF0000000000000"):
            with self.subTest(value=value):
                with self.assertRaises(sunset_cursor.SunsetCursorError) as caught:
                    sunset_cursor._decode_tt_bits(value)
                self.assertEqual(caught.exception.reason, REASON_MALFORMED_CURSOR)

    def test_cursor_failure_message_keeps_its_own_banner(self):
        with self.assertRaises(sunset_cursor.SunsetCursorError) as caught:
            sunset_cursor._encode_tt_bits(float("nan"))
        message = str(caught.exception)
        self.assertTrue(message.startswith("SUNSET CURSOR MALFORMED - "))
        self.assertEqual(message.count("SUNSET CURSOR MALFORMED"), 1)
        self.assertIn("tt must be finite", message)

    def test_the_shared_codec_is_actually_the_one_the_cursor_uses(self):
        """Anti-vacuity: prove delegation, not a coincidental match."""
        import inspect

        source = inspect.getsource(sunset_cursor)
        self.assertIn("from exact_time_transport import", source)
        self.assertIn("return encode_tt_bits(tt)", source)
        self.assertIn("return decode_tt_bits(value)", source)
        self.assertNotIn("struct.pack", source)
        self.assertNotIn("struct.unpack", source)

    def test_no_second_tt_codec_remains_in_the_repository(self):
        import inspect

        import astronomical_event_transport

        for module in (sunset_cursor, astronomical_event_transport):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn('struct.pack(">d"', source)
                self.assertNotIn('struct.unpack(">d"', source)


if __name__ == "__main__":
    unittest.main()
