"""PTC-T2 verification for the exact civil-instant resolver.

Covers civil_instant.py and GET /civil-instant in server.py: pinned exact
identities, the millisecond round trip, the 2016-12-31 / 2017-01-01 leap
boundary, refusal of malformed and uncertified instants, encode/decode
identity, the published provenance, derivation of the certified civil span
from the pinned artifact, and structural isolation from any second time-scale
implementation.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest tests.test_ptc_t2_civil_instant -v

ROUTE SEAM

The route function is invoked directly, following PTC-I1. httpx is not
installed, so starlette's TestClient is unavailable.

ORACLES

The pinned ttBits below were produced by the certified runtime (Skyfield 1.54,
iers.npz C7D7536D...935C) and are recorded as literals so that a change of
runtime is a failure rather than a silent re-pin. They are additionally
compared with the year-addressing path ts.utc(...).tt, the other UTC-input
conversion this Authority already uses, so the resolver cannot drift from it.

The leap-boundary oracle uses no offset table and no TT - TAI constant: one
civil second across the 2016-12-31 boundary must span two Terrestrial Time
seconds, and one civil second across an ordinary midnight must span one.
"""

import inspect
import io
import textwrap
import tokenize
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
from fastapi import HTTPException
from skyfield.timelib import Timescale

import civil_instant
import earth_rotation
import server
from astronomical_event_transport import utc_reference
from exact_time_transport import decode_tt_bits, encode_tt_bits

ROUTE_PATH = "/civil-instant"

EXPECTED_KEYS = (
    "ttBits",
    "tt",
    "utc",
    "requestedUtc",
    "certifiedSpan",
    "scientificEnvironmentId",
    "iersDataSha256",
    "skyfieldVersion",
)
EXPECTED_INSTANT_KEYS = ("ttBits", "tt", "utc")

CERTIFIED_IERS_SHA256 = (
    "C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C"
)
CERTIFIED_SKYFIELD_VERSION = "1.54"

# Exact identities produced by the certified runtime.
PINNED = {
    "2016-12-31T23:59:59Z": "4142C04D40197AEB",
    "2017-01-01T00:00:00Z": "4142C04D401A3D1A",
    "2019-03-20T21:58:26Z": "4142C1E1B54BEBF0",
    "2026-10-06T12:34:56.789Z": "4142C7440335771F",
}

# The certified span of the pinned artifact, as resolved by the runtime.
PINNED_LOWER_BITS = "4142A08DC01060C0"
PINNED_UPPER_BITS = "4142C77A401A3D1A"

INVALID_REASON = "CIVIL_INSTANT_INVALID"
OUTSIDE_REASON = "CIVIL_INSTANT_OUTSIDE_CERTIFIED_SPAN"
UNAVAILABLE_REASON = "CERTIFIED_SPAN_UNAVAILABLE"

SECONDS_PER_DAY = 86400.0


def resolve(utc):
    return server.civil_instant(utc=utc)


def refusal(utc):
    with unittest.TestCase().assertRaises(HTTPException) as caught:
        resolve(utc)
    return caught.exception


def instant(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def executable(target):
    """Source of a module or function with docstrings and comments removed.

    Tokens are joined by single spaces, so guards spell operators spaced.
    """
    source = textwrap.dedent(inspect.getsource(target))
    kept = []
    previous = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous in (
            tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL
        ):
            continue
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.COMMENT):
            previous = token.type
    return " ".join(kept)


def registered_routes():
    found = {}
    for route in server.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        found[route.path] = (
            set(route.methods),
            endpoint.__name__,
            list(inspect.signature(endpoint).parameters),
        )
    return found


# --- 1. Published response ----------------------------------------------------


class TestPublishedResponse(unittest.TestCase):
    def test_the_route_is_registered_and_takes_only_a_civil_instant(self):
        methods, name, parameters = registered_routes()[ROUTE_PATH]
        self.assertEqual(methods, {"GET"})
        self.assertEqual(name, "civil_instant")
        self.assertEqual(parameters, ["utc"])

    def test_the_shape_is_exactly_the_published_contract(self):
        body = resolve("2026-10-06T12:34:56.789Z")
        self.assertEqual(tuple(body), EXPECTED_KEYS)
        self.assertEqual(tuple(body["certifiedSpan"]), ("lower", "upper"))
        for bound in ("lower", "upper"):
            self.assertEqual(tuple(body["certifiedSpan"][bound]),
                             EXPECTED_INSTANT_KEYS)

    def test_the_response_is_deterministic(self):
        self.assertEqual(resolve("2024-07-04T06:30:00Z"),
                         resolve("2024-07-04T06:30:00Z"))


# --- 2. Pinned exact identities (A) -------------------------------------------


class TestPinnedIdentities(unittest.TestCase):
    def test_pinned_instants_resolve_to_their_certified_bits(self):
        for utc, bits in PINNED.items():
            with self.subTest(utc=utc):
                self.assertEqual(resolve(utc)["ttBits"], bits)

    def test_the_resolver_agrees_with_the_year_addressing_conversion(self):
        cases = {
            "2017-01-01T00:00:00Z": (2017, 1, 1),
            "2019-03-20T21:58:26Z": (2019, 3, 20, 21, 58, 26),
            "2024-07-04T06:30:00Z": (2024, 7, 4, 6, 30, 0),
        }
        for utc, parts in cases.items():
            with self.subTest(utc=utc):
                self.assertEqual(
                    resolve(utc)["ttBits"],
                    encode_tt_bits(float(civil_instant.ts.utc(*parts).tt)),
                )

    def test_offsets_and_bare_instants_follow_the_existing_parser(self):
        reference = resolve("2026-10-06T12:34:56.789Z")
        for spelling in ("2026-10-06T12:34:56.789+00:00",
                         "2026-10-06T08:34:56.789-04:00",
                         "2026-10-06T12:34:56.789"):
            with self.subTest(spelling=spelling):
                body = resolve(spelling)
                self.assertEqual(body["ttBits"], reference["ttBits"])
                self.assertEqual(body["requestedUtc"],
                                 "2026-10-06T12:34:56.789000+00:00")


# --- 3. Round trip (B) ------------------------------------------------------


class TestRoundTrip(unittest.TestCase):
    MILLISECOND_INSTANTS = (
        "1972-07-01T00:00:00.000Z",
        "1999-12-31T23:59:59.999Z",
        "2016-12-31T23:59:59.999Z",
        "2017-01-01T00:00:00.000Z",
        "2017-01-01T00:00:00.001Z",
        "2024-02-29T12:00:00.500Z",
        "2026-10-06T12:34:56.789Z",
        "2027-01-23T00:00:00.000Z",
    )

    def test_utc_reference_reproduces_the_input_to_the_millisecond(self):
        for utc in self.MILLISECOND_INSTANTS:
            with self.subTest(utc=utc):
                body = resolve(utc)
                rendered = datetime.fromisoformat(utc_reference(body["tt"]))
                difference = abs((rendered - instant(utc)).total_seconds())
                self.assertLess(difference, 0.0005)
                self.assertEqual(body["utc"], utc_reference(body["tt"]))


# --- 4. Leap boundary (C) -----------------------------------------------------


class TestLeapBoundary(unittest.TestCase):
    def tt_seconds(self, utc):
        return resolve(utc)["tt"] * SECONDS_PER_DAY

    def test_the_2016_leap_second_is_applied_from_the_certified_table(self):
        across = (self.tt_seconds("2017-01-01T00:00:00Z")
                  - self.tt_seconds("2016-12-31T23:59:59Z"))
        self.assertAlmostEqual(across, 2.0, delta=0.0001)

    def test_an_ordinary_midnight_carries_no_leap(self):
        across = (self.tt_seconds("2016-12-31T00:00:00Z")
                  - self.tt_seconds("2016-12-30T23:59:59Z"))
        self.assertAlmostEqual(across, 1.0, delta=0.0001)

    def test_representable_instants_either_side_are_strictly_ordered(self):
        spellings = ("2016-12-31T23:59:58.999Z", "2016-12-31T23:59:59Z",
                     "2016-12-31T23:59:59.999Z", "2017-01-01T00:00:00Z",
                     "2017-01-01T00:00:00.001Z")
        states = [resolve(utc)["tt"] for utc in spellings]
        self.assertEqual(states, sorted(set(states)))

    def test_a_literal_leap_second_is_not_representable_and_is_refused(self):
        error = refusal("2016-12-31T23:59:60Z")
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.detail["reason"], INVALID_REASON)


# --- 5. Invalid input and the certified span (D) -------------------------------


class TestRefusals(unittest.TestCase):
    def test_malformed_instants_are_refused(self):
        for value in ("", "nonsense", "2026-13-01T00:00:00Z",
                      "2026-10-06T25:00:00Z", "2026-10-06 12:00 GMT",
                      "0001-01-01T00:00:00+01:00"):
            with self.subTest(utc=value):
                error = refusal(value)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.detail["reason"], INVALID_REASON)

    def test_instants_outside_the_certified_span_are_refused(self):
        for value in ("0001-01-01T00:00:00Z", "1900-01-01T00:00:00Z",
                      "1972-01-01T00:00:00Z", "2040-01-01T00:00:00Z",
                      "9999-12-31T23:59:59Z"):
            with self.subTest(utc=value):
                error = refusal(value)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.detail["reason"], OUTSIDE_REASON)

    def test_the_pinned_artifact_bounds_are_inclusive(self):
        span = resolve("2026-10-06T00:00:00Z")["certifiedSpan"]
        self.assertEqual(span["lower"]["ttBits"], PINNED_LOWER_BITS)
        self.assertEqual(span["upper"]["ttBits"], PINNED_UPPER_BITS)
        self.assertEqual(resolve("1972-07-01T00:00:00Z")["ttBits"],
                         PINNED_LOWER_BITS)
        self.assertEqual(resolve("2027-01-23T00:00:00Z")["ttBits"],
                         PINNED_UPPER_BITS)
        for value in ("1972-06-30T23:59:59.999Z", "2027-01-23T00:00:00.001Z"):
            with self.subTest(utc=value):
                self.assertEqual(refusal(value).detail["reason"],
                                 OUTSIDE_REASON)

    def test_no_extrapolated_offset_is_certified(self):
        # The runtime itself converts these; the resolver must not.
        for value in ("1960-01-01T00:00:00", "2030-01-01T00:00:00"):
            with self.subTest(utc=value):
                float(civil_instant.ts.from_datetime(
                    instant(value).replace(tzinfo=timezone.utc)).tt)
                self.assertEqual(refusal(value + "Z").detail["reason"],
                                 OUTSIDE_REASON)

    def test_a_refusal_discloses_nothing_beyond_the_governed_envelope(self):
        for value in ("nonsense", "1900-01-01T00:00:00Z"):
            with self.subTest(utc=value):
                detail = refusal(value).detail
                self.assertEqual(tuple(detail), ("reason", "message"))
                for leak in ("C:\\", "/home", "Traceback", "site-packages",
                             ".npz", ".bsp", "File \""):
                    self.assertNotIn(leak, repr(detail))

    def test_malformed_input_is_not_echoed(self):
        self.assertNotIn("<script>", repr(refusal("<script>").detail))


# --- 6. Exact transport (E) ---------------------------------------------------


class TestExactTransport(unittest.TestCase):
    def test_returned_bits_decode_to_the_returned_state_bit_for_bit(self):
        for utc in list(PINNED) + ["2024-07-04T06:30:00Z"]:
            with self.subTest(utc=utc):
                body = resolve(utc)
                decoded = decode_tt_bits(body["ttBits"])
                self.assertEqual(encode_tt_bits(decoded), body["ttBits"])
                self.assertEqual(decoded.hex(), body["tt"].hex())
                for bound in body["certifiedSpan"].values():
                    self.assertEqual(
                        decode_tt_bits(bound["ttBits"]).hex(),
                        bound["tt"].hex(),
                    )

    def test_the_state_is_the_shared_timescale_conversion_unchanged(self):
        utc = "2026-10-06T12:34:56.789Z"
        expected = float(civil_instant.ts.from_datetime(instant(utc)).tt)
        self.assertEqual(resolve(utc)["tt"].hex(), expected.hex())


# --- 7. Provenance (F) --------------------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_the_environment_that_converted_is_identified(self):
        body = resolve("2026-10-06T12:34:56.789Z")
        published = server.scientific_environment()
        self.assertEqual(body["scientificEnvironmentId"],
                         published["scientificEnvironmentId"])
        self.assertEqual(body["iersDataSha256"], CERTIFIED_IERS_SHA256)
        self.assertEqual(body["iersDataSha256"],
                         published["environment"]["iersDataSha256"])
        self.assertEqual(body["skyfieldVersion"], CERTIFIED_SKYFIELD_VERSION)

    def test_no_unrelated_provenance_is_fabricated(self):
        body = resolve("2026-10-06T12:34:56.789Z")
        for key in body:
            for token in ("delta", "ut1", "dut1", "kernel", "ephemeris",
                          "observed", "predicted", "regime", "provenance"):
                with self.subTest(key=key, token=token):
                    self.assertNotIn(token, key.lower())


# --- 8. The span is derived from the artifact ---------------------------------


class TestSpanDerivation(unittest.TestCase):
    def test_the_lower_bound_is_the_first_leap_table_entry(self):
        lower, _upper = civil_instant._CERTIFIED_CIVIL_SPAN
        first = float(civil_instant.ts.leap_dates[0])
        # The first entry is a UTC Julian Date. Built as a datetime here, by a
        # route independent of the module's calendar-field construction, and
        # converted by the shared timescale, it must give the same state.
        civil = (datetime(1858, 11, 17, tzinfo=timezone.utc)
                 + timedelta(days=first - 2400000.5))
        expected = float(civil_instant.ts.from_datetime(civil).tt)
        self.assertEqual(lower.hex(), expected.hex())

    def test_the_upper_bound_is_the_earth_rotation_tabulated_end(self):
        _lower, upper = civil_instant._CERTIFIED_CIVIL_SPAN
        self.assertEqual(upper, earth_rotation._TABULATED_SPAN[1])
        self.assertEqual(earth_rotation.delta_t_regime(upper), "TABULATED")

    def test_a_recertified_artifact_moves_the_span_without_source_change(self):
        real = civil_instant.ts
        daily_tt, daily_delta_t = real.delta_t_table
        keep = len(daily_tt) - 400
        moved = Timescale(
            (np.array(daily_tt[:keep]), np.array(daily_delta_t[:keep])),
            np.array(real.leap_dates[2:]),
            np.array(real.leap_offsets[2:]),
        )
        with mock.patch.object(civil_instant, "ts", moved), \
                mock.patch.object(earth_rotation, "ts", moved):
            lower, upper = civil_instant._certified_civil_span()

        self.assertEqual(upper, float(daily_tt[keep - 1]))
        self.assertGreater(lower, civil_instant._CERTIFIED_CIVIL_SPAN[0])
        self.assertLess(upper, civil_instant._CERTIFIED_CIVIL_SPAN[1])

    def test_an_artifact_without_a_table_certifies_nothing(self):
        with mock.patch.object(civil_instant, "_CERTIFIED_CIVIL_SPAN", None):
            error = refusal("2026-10-06T12:34:56.789Z")
        self.assertEqual(error.detail["reason"], UNAVAILABLE_REASON)

    def test_no_calendar_year_is_written_into_the_resolver(self):
        source = executable(civil_instant)
        for literal in ("1972", "1973", "2027", "2026", "2017"):
            with self.subTest(literal=literal):
                self.assertNotIn(literal, source)


# --- 9. Structural isolation --------------------------------------------------


class TestStructuralIsolation(unittest.TestCase):
    def setUp(self):
        self.module = executable(civil_instant)
        self.route = executable(server.civil_instant)

    def test_the_conversion_is_the_shared_timescale_only(self):
        self.assertIn("from astronomy_solver import ensure_utc , ts",
                      self.module)
        self.assertEqual(self.module.count("ts . from_datetime ("), 1)
        self.assertNotIn("load . timescale", self.module)
        self.assertNotIn("Timescale", self.module)

    def test_no_second_time_scale_implementation(self):
        for token in ("32.184", "32 .184", "37", "69.184", "leap_offsets",
                      "_leap", "TAI", "tai", "delta_t", "ut1", "dut1",
                      "timedelta", "timestamp", "86400"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.module)

    def test_no_second_codec_or_projection(self):
        for token in ("struct", "pack", ". hex (", "encode_tt_bits",
                      "decode_tt_bits", "utc_datetime", "strftime"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.module)
        self.assertIn("project_exact_instant", self.module)

    def test_no_astronomy_and_no_calendar(self):
        for token in ("almanac", "load_kernel", "wgs84", "find_",
                      "hpc", "telma", "sabbath", "creation", "weekday",
                      "continuity", "polar", "sce", "month"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.module.lower())

    def test_the_route_reuses_the_existing_parser_and_catches_narrowly(self):
        self.assertIn("parse_utc_datetime ( utc )", self.route)
        self.assertIn("except ( ValueError , OverflowError )", self.route)
        self.assertIn("except CivilInstantError", self.route)
        self.assertIn("return resolve_civil_instant ( requested )", self.route)
        for token in ("except Exception", "except BaseException", "except :",
                      "ts .", "fromisoformat", "project_exact_instant"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)


# --- 10. Previously published routes preserved --------------------------------


class TestPublishedRoutesPreserved(unittest.TestCase):
    PREVIOUS = {
        "/health": "health",
        "/delta-t/{year}": "delta_t",
        "/delta-t-bce/{year}": "delta_t_bce",
        "/equinox/{year}": "equinox",
        "/equinox-bce/{year}": "equinox_bce",
        "/season-events": "season_events",
        "/season-events-bce/{year}": "season_events_bce",
        "/solar_longitude": "solar",
        "/subsolar-point": "subsolar",
        "/sunset": "sunset",
        "/sunset-after": "sunset_after",
        "/sunset-successor": "sunset_successor",
        "/solar-longitude-event-before": "solar_longitude_event_before",
        "/solar-longitude-event-after": "solar_longitude_event_after",
        "/sunset-bracket": "sunset_bracket",
        "/solar-longitude-event-in-year": "solar_longitude_event_in_year",
        "/sunset-event-after": "sunset_event_after",
        "/sunset-count": "sunset_count",
        "/scientific-environment": "scientific_environment",
        "/earth-rotation": "earth_rotation",
        "/solar-regime": "solar_regime",
        "/night-start-after": "night_start_after",
        "/solar-crossing": "solar_crossing",
    }

    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()
        for path, name in self.PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], name)

    def test_this_increment_added_exactly_one_route(self):
        published = {
            path for path in registered_routes()
            if not path.startswith(("/openapi", "/docs", "/redoc"))
        }
        self.assertEqual(published - set(self.PREVIOUS), {ROUTE_PATH})


if __name__ == "__main__":
    unittest.main()
