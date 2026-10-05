"""DT-R4-A verification for the certified night-start route.

Covers GET /night-start-after in server.py and project_night_start in
astronomical_event_transport.py: the request contract, exact transport, the
published projection for each night-start kind, bit identity with
/sunset-event-after, the failure taxonomy, and the structural isolation of
the route block.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, as every route suite here does:
httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted. Direct invocation
exercises the real route body, including the HTTPException mapping.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, kinds, the expected key order, the route path, the block marker
and the certified latitude limit are declared here as test-local literals.
"""

import inspect
import math
import re
import struct
import unittest

from fastapi import HTTPException

import astronomical_event_transport
import server
from astronomy_solver import NightStart, find_solar_longitude_event_in_year
from exact_time_transport import decode_tt_bits, encode_tt_bits

ROUTE_PATH = "/night-start-after"
BLOCK_MARKER = "# DT-R4-A - certified night-start route."
GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)

EXPECTED_KEYS = (
    "nightStartKind",
    "nightStartLo",
    "nightStartHi",
    "event",
    "minimumAltitudeMarginDegrees",
    "eventThresholdDegrees",
    "eventConvention",
    "kernel",
    "ephemerisRole",
    "certifiedDomainLimitDegrees",
)
EXPECTED_INSTANT_KEYS = ("ttBits", "tt", "utc")

LATITUDE_LIMIT = 89.739
NEW_YORK = (40.7406, -73.9586)


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def equinox_bits():
    return encode_tt_bits(
        find_solar_longitude_event_in_year(2019, "SOLAR_LONGITUDE_000").tt
    )


def night_start(ttBits, latitude, longitude):
    return server.night_start_after(
        ttBits=ttBits, latitude=latitude, longitude=longitude
    )


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


# --- 1. Request contract ----------------------------------------------------


class TestRequestContract(unittest.TestCase):
    def test_the_route_is_registered_once_with_exact_parameters(self):
        methods, name, parameters = registered_routes()[ROUTE_PATH]
        self.assertEqual(methods, {"GET"})
        self.assertEqual(name, "night_start_after")
        self.assertEqual(parameters, ["ttBits", "latitude", "longitude"])


# --- 2. Projections -----------------------------------------------------------


class TestGenuineSunsetProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchor = equinox_bits()
        cls.body = night_start(cls.anchor, *NEW_YORK)
        cls.sunset = server.sunset_event_after(
            ttBits=cls.anchor, latitude=NEW_YORK[0], longitude=NEW_YORK[1]
        )

    def test_keys_are_exactly_the_published_keys_in_order(self):
        self.assertEqual(tuple(self.body), EXPECTED_KEYS)
        for key in ("nightStartLo", "nightStartHi", "event"):
            with self.subTest(key=key):
                self.assertEqual(tuple(self.body[key]), EXPECTED_INSTANT_KEYS)

    def test_the_event_is_the_sunset_event_after_answer_bit_for_bit(self):
        self.assertEqual(self.body["nightStartKind"], "GENUINE_SUNSET")
        self.assertEqual(self.body["event"]["ttBits"], self.sunset["ttBits"])
        self.assertEqual(self.body["kernel"], self.sunset["kernel"])
        self.assertEqual(bits(self.body["event"]["tt"]),
                         bits(decode_tt_bits(self.sunset["ttBits"])))

    def test_the_bracket_collapses_onto_the_event(self):
        self.assertEqual(self.body["nightStartLo"], self.body["event"])
        self.assertEqual(self.body["nightStartHi"], self.body["event"])
        self.assertIsNone(self.body["minimumAltitudeMarginDegrees"])

    def test_provenance_is_carried(self):
        self.assertEqual(self.body["kernel"], "de440.bsp")
        self.assertEqual(self.body["ephemerisRole"], "primary")
        self.assertEqual(self.body["eventThresholdDegrees"], -0.8333)
        self.assertEqual(self.body["certifiedDomainLimitDegrees"],
                         LATITUDE_LIMIT)


class TestVanishedNightProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchor = equinox_bits()
        cls.body = night_start(cls.anchor, 89.5, 0.0)

    def test_a_vanished_night_publishes_no_event(self):
        self.assertEqual(self.body["nightStartKind"], "VANISHED_NIGHT_MINIMUM")
        self.assertIsNone(self.body["event"])
        self.assertGreater(self.body["minimumAltitudeMarginDegrees"], 0.0)

    def test_the_bracket_is_exact_ordered_and_after_the_anchor(self):
        lo = decode_tt_bits(self.body["nightStartLo"]["ttBits"])
        hi = decode_tt_bits(self.body["nightStartHi"]["ttBits"])
        self.assertEqual(bits(lo), bits(self.body["nightStartLo"]["tt"]))
        self.assertEqual(bits(hi), bits(self.body["nightStartHi"]["tt"]))
        self.assertLess(decode_tt_bits(self.anchor), lo)
        self.assertLessEqual(lo, hi)


class TestProjectionFunction(unittest.TestCase):
    def test_the_route_returns_exactly_the_published_projection(self):
        record = NightStart(
            kind="TANGENCY_UNRESOLVED", lo=2458654.0, hi=2458654.001,
            event_tt=None, minimum_margin_degrees=1e-9, kernel="de440.bsp",
            ephemeris_role="primary", event_threshold_degrees=-0.8333,
            event_convention="convention", certified_domain_limit_degrees=89.739,
        )
        body = astronomical_event_transport.project_night_start(record)
        self.assertEqual(tuple(body), EXPECTED_KEYS)
        self.assertEqual(body["nightStartKind"], "TANGENCY_UNRESOLVED")
        self.assertIsNone(body["event"])
        self.assertEqual(body["nightStartLo"]["ttBits"],
                         encode_tt_bits(2458654.0))


# --- 3. Failure taxonomy ------------------------------------------------------


class TestFailures(unittest.TestCase):
    def rejection(self, ttBits, latitude, longitude):
        with self.assertRaises(HTTPException) as caught:
            night_start(ttBits, latitude, longitude)
        return caught.exception

    def test_a_malformed_anchor_is_a_transport_failure(self):
        error = self.rejection("ZZZZZZZZZZZZZZZZ", *NEW_YORK)
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.detail["reason"], "TT_BITS_INVALID")

    def test_the_uncertified_latitudes_are_refused(self):
        for latitude in (90.0, -90.0, math.nextafter(LATITUDE_LIMIT, 90.0)):
            with self.subTest(latitude=latitude):
                error = self.rejection(equinox_bits(), latitude, 0.0)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.detail["reason"],
                                 "NIGHT_START_LATITUDE_UNCERTIFIED")

    def test_a_geodetic_domain_failure_keeps_its_reason(self):
        error = self.rejection(equinox_bits(), 91.0, 0.0)
        self.assertEqual(error.detail["reason"], "OBSERVER_OUT_OF_DOMAIN")

    def test_the_detail_is_the_shared_structured_shape(self):
        error = self.rejection(equinox_bits(), 90.0, 0.0)
        self.assertEqual(tuple(error.detail), ("reason", "message"))


# --- 4. Structural isolation --------------------------------------------------


class TestSourceGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = inspect.getsource(server)
        cls.route = inspect.getsource(server.night_start_after)

    def test_the_block_marker_is_present_once_and_outside_the_a_grammar(self):
        """The A-series closed-world claims stay true: the A5 block is still
        the last governed A block, as it was after PTC-I1 and PTC-I2."""
        self.assertEqual(self.source.count(BLOCK_MARKER), 1)
        own = self.source.find(BLOCK_MARKER)
        self.assertIsNone(GOVERNED_BLOCK_MARKER.match(self.source, own))
        self.assertGreater(self.source.find("def night_start_after("), own)

    def test_the_route_delegates_and_hand_builds_nothing(self):
        for token in ("decode_tt_bits(", "find_night_start_after(",
                      "project_night_start(", "reason_detail("):
            with self.subTest(token=token):
                self.assertIn(token, self.route)
        for token in ("find_sunset_from_instant", "almanac", "ts.", "{\""):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_only_the_two_governed_exceptions_are_caught(self):
        caught = re.findall(r"except (\w+)", self.route)
        self.assertEqual(caught, ["ExactTimeTransportError",
                                  "SunsetChronologyError"])


if __name__ == "__main__":
    unittest.main()
