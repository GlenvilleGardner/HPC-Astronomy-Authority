"""PTC-A1 verification for the certified directional solar-crossing route.

Covers GET /solar-crossing in server.py and project_solar_crossing in
astronomical_event_transport.py: the request contract, exact transport, the
published projection for a found crossing and for certified absence, the
overlap with /sunset-event-after, the failure taxonomy, the unchanged
existing routes, the published surface, and the structural isolation of the
route, the projection and the solver block.

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
exercises the real route body, including the HTTPException mapping. A body
returned without an HTTPException is the route's 200.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, spellings, the expected key order, the route path, the block
markers and the fixtures are declared here as test-local literals.
"""

import inspect
import re
import struct
import unittest

from fastapi import HTTPException

import astronomical_event_transport
import astronomy_solver
import scientific_environment
import server
from astronomy_solver import SolarCrossing, find_solar_longitude_event_in_year
from exact_time_transport import decode_tt_bits, encode_tt_bits

ROUTE_PATH = "/solar-crossing"
ROUTE_BLOCK_MARKER = "# PTC-A1 - certified directional solar-crossing route."
SOLVER_BLOCK_MARKER = "# PTC-A1 - certified directional solar crossing."
GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)

EXPECTED_KEYS = (
    "orientation",
    "direction",
    "event",
    "requested",
    "covered",
    "complete",
    "truncationReason",
    "eventThresholdDegrees",
    "eventConvention",
    "kernel",
    "ephemerisRole",
)
EXPECTED_REQUESTED_KEYS = ("anchor", "horizon", "horizonDays")
EXPECTED_COVERED_KEYS = ("lo", "hi")
EXPECTED_INSTANT_KEYS = ("ttBits", "tt", "utc")

NEW_YORK = (40.7406, -73.9586)
FINAL_SUNSET_70N = "4142C1FDBCD4FDAC"
SOLVER_GENERATION = "hpc-authority-solver-v2"

# Every application route published before PTC-A1, with its endpoint.
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
}


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def equinox_bits():
    return encode_tt_bits(
        find_solar_longitude_event_in_year(2019, "SOLAR_LONGITUDE_000").tt
    )


def solar_crossing(ttBits, latitude, longitude, orientation, direction,
                   horizonDays):
    return server.solar_crossing(
        ttBits=ttBits, latitude=latitude, longitude=longitude,
        orientation=orientation, direction=direction, horizonDays=horizonDays,
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


def executable(function):
    """Source with the docstring removed, so prose never fails a scan."""
    return inspect.getsource(function).split('"""')[-1]


# --- 1. Request contract ----------------------------------------------------


class TestRequestContract(unittest.TestCase):
    def test_the_route_is_registered_once_with_exact_parameters(self):
        methods, name, parameters = registered_routes()[ROUTE_PATH]
        self.assertEqual(methods, {"GET"})
        self.assertEqual(name, "solar_crossing")
        self.assertEqual(
            parameters,
            ["ttBits", "latitude", "longitude", "orientation", "direction",
             "horizonDays"],
        )

    def test_the_route_takes_no_civil_or_decimal_anchor(self):
        parameters = registered_routes()[ROUTE_PATH][2]
        for forbidden in ("tt", "date", "year", "utc", "kernel"):
            with self.subTest(parameter=forbidden):
                self.assertNotIn(forbidden, parameters)


# --- 2. Projections -----------------------------------------------------------


class TestFoundCrossingProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchor = equinox_bits()
        cls.body = solar_crossing(cls.anchor, *NEW_YORK, "SETTING", "AFTER", 3.0)
        cls.sunset = server.sunset_event_after(
            ttBits=cls.anchor, latitude=NEW_YORK[0], longitude=NEW_YORK[1]
        )

    def test_keys_are_exactly_the_published_keys_in_order(self):
        self.assertEqual(tuple(self.body), EXPECTED_KEYS)
        self.assertEqual(tuple(self.body["requested"]), EXPECTED_REQUESTED_KEYS)
        self.assertEqual(tuple(self.body["covered"]), EXPECTED_COVERED_KEYS)
        for instant in (self.body["event"], self.body["requested"]["anchor"],
                        self.body["requested"]["horizon"],
                        self.body["covered"]["lo"], self.body["covered"]["hi"]):
            self.assertEqual(tuple(instant), EXPECTED_INSTANT_KEYS)

    def test_the_event_is_the_sunset_event_after_answer_bit_for_bit(self):
        self.assertEqual(self.body["event"]["ttBits"], self.sunset["ttBits"])
        self.assertEqual(self.body["kernel"], self.sunset["kernel"])

    def test_every_instant_is_exact_transport(self):
        for instant in (self.body["event"], self.body["covered"]["lo"],
                        self.body["covered"]["hi"]):
            self.assertEqual(bits(decode_tt_bits(instant["ttBits"])),
                             bits(instant["tt"]))

    def test_the_request_and_frontier_are_echoed(self):
        self.assertEqual(self.body["orientation"], "SETTING")
        self.assertEqual(self.body["direction"], "AFTER")
        self.assertEqual(self.body["requested"]["anchor"]["ttBits"], self.anchor)
        self.assertEqual(self.body["requested"]["horizonDays"], 3.0)
        self.assertEqual(self.body["covered"]["lo"]["ttBits"], self.anchor)
        self.assertEqual(
            bits(self.body["covered"]["hi"]["tt"]),
            bits(decode_tt_bits(self.anchor) + 3.0),
        )
        self.assertTrue(self.body["complete"])
        self.assertIsNone(self.body["truncationReason"])

    def test_provenance_is_carried(self):
        self.assertEqual(self.body["kernel"], "de440.bsp")
        self.assertEqual(self.body["ephemerisRole"], "primary")
        self.assertEqual(self.body["eventThresholdDegrees"], -0.8333)
        self.assertTrue(self.body["eventConvention"].startswith(
            "USNO apparent sunrise/sunset"))

    def test_a_sunrise_is_published_through_the_same_shape(self):
        body = solar_crossing(self.anchor, *NEW_YORK, "RISING", "BEFORE", 3.0)
        self.assertEqual(tuple(body), EXPECTED_KEYS)
        self.assertEqual(body["orientation"], "RISING")
        self.assertLess(decode_tt_bits(body["event"]["ttBits"]),
                        decode_tt_bits(self.anchor))


class TestCertifiedAbsenceProjection(unittest.TestCase):
    """A complete frontier with no crossing is a 200 with a null event."""

    @classmethod
    def setUpClass(cls):
        final = decode_tt_bits(FINAL_SUNSET_70N)
        onset = astronomy_solver.find_solar_crossing(
            final, 70.0, 0.0, "RISING", "AFTER", 3.0
        ).event_tt
        cls.body = solar_crossing(encode_tt_bits(onset), 70.0, 0.0, "SETTING",
                                  "AFTER", 60.0)

    def test_the_event_key_is_present_and_null(self):
        self.assertEqual(tuple(self.body), EXPECTED_KEYS)
        self.assertIn("event", self.body)
        self.assertIsNone(self.body["event"])

    def test_absence_is_stated_over_a_complete_frontier(self):
        self.assertTrue(self.body["complete"])
        self.assertIsNone(self.body["truncationReason"])
        self.assertEqual(self.body["requested"]["horizonDays"], 60.0)

    def test_the_existing_sunset_route_still_reports_absence_as_404(self):
        with self.assertRaises(HTTPException) as caught:
            server.sunset_event_after(ttBits=FINAL_SUNSET_70N, latitude=70.0,
                                      longitude=0.0)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail["reason"], "SUNSET_EVENT_ABSENT")


class TestProjectionFunction(unittest.TestCase):
    def test_the_projection_carries_the_record_unchanged(self):
        record = SolarCrossing(
            orientation="RISING", direction="BEFORE", event_tt=None,
            anchor=2458654.5, horizon_days=2.0, requested_bound=2458652.5,
            covered_lo=2458652.75, covered_hi=2458654.5, complete=False,
            truncation_reason="EPHEMERIS_REACH_EXHAUSTED", kernel="de440.bsp",
            ephemeris_role="primary", event_threshold_degrees=-0.8333,
            event_convention="convention",
        )
        body = astronomical_event_transport.project_solar_crossing(record)
        self.assertEqual(tuple(body), EXPECTED_KEYS)
        self.assertIsNone(body["event"])
        self.assertEqual(body["requested"]["anchor"]["ttBits"],
                         encode_tt_bits(2458654.5))
        self.assertEqual(body["requested"]["horizon"]["ttBits"],
                         encode_tt_bits(2458652.5))
        self.assertEqual(body["covered"]["lo"]["ttBits"],
                         encode_tt_bits(2458652.75))
        self.assertFalse(body["complete"])
        self.assertEqual(body["truncationReason"], "EPHEMERIS_REACH_EXHAUSTED")


# --- 3. Failure taxonomy ------------------------------------------------------


class TestFailures(unittest.TestCase):
    def rejection(self, ttBits=None, latitude=NEW_YORK[0],
                  longitude=NEW_YORK[1], orientation="SETTING",
                  direction="AFTER", horizonDays=3.0):
        with self.assertRaises(HTTPException) as caught:
            solar_crossing(equinox_bits() if ttBits is None else ttBits,
                           latitude, longitude, orientation, direction,
                           horizonDays)
        return caught.exception

    def test_a_malformed_anchor_is_a_transport_failure(self):
        error = self.rejection(ttBits="ZZZZZZZZZZZZZZZZ")
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.detail["reason"], "TT_BITS_INVALID")

    def test_governed_refusals_keep_their_reasons(self):
        for overrides, reason in (
            ({"orientation": "sunrise"}, "SOLAR_CROSSING_ORIENTATION_INVALID"),
            ({"direction": "LATER"}, "SOLAR_CROSSING_DIRECTION_INVALID"),
            ({"horizonDays": 0.0}, "SOLAR_CROSSING_HORIZON_INVALID"),
            ({"horizonDays": float("nan")}, "SOLAR_CROSSING_HORIZON_INVALID"),
            ({"horizonDays": 401.0}, "SOLAR_CROSSING_HORIZON_TOO_LONG"),
            ({"latitude": 91.0}, "OBSERVER_OUT_OF_DOMAIN"),
        ):
            with self.subTest(reason=reason, overrides=overrides):
                error = self.rejection(**overrides)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.detail["reason"], reason)

    def test_a_tangency_is_refused_not_answered(self):
        june = find_solar_longitude_event_in_year(2019, "SOLAR_LONGITUDE_090").tt
        error = self.rejection(ttBits=encode_tt_bits(june - 2.0),
                               latitude=65.74293204295881, longitude=0.0)
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.detail["reason"], "SUNSET_DETECTION_AMBIGUOUS")

    def test_an_incomplete_frontier_without_a_crossing_is_refused(self):
        end = astronomy_solver.kernel_coverage_tt(
            astronomy_solver.FUTURE_KERNEL
        ).tt_end
        found = astronomy_solver.find_solar_crossing(
            end - 1.0, *NEW_YORK, "SETTING", "AFTER", 10.0
        ).event_tt
        error = self.rejection(ttBits=encode_tt_bits(found), horizonDays=10.0)
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.detail["reason"], "EPHEMERIS_COVERAGE_EXHAUSTED")

    def test_the_detail_is_the_shared_structured_shape(self):
        error = self.rejection(orientation="sunrise")
        self.assertEqual(tuple(error.detail), ("reason", "message"))


# --- 4. Published surface -----------------------------------------------------


class TestPublishedSurface(unittest.TestCase):
    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()
        for path, name in PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], name)

    # Application routes added by governed increments AFTER PTC-A1. Each is
    # owned by the increment that added it, whose own suite owns the closed
    # world as of that increment; subtracting them keeps PTC-A1's claim
    # exactly as strong as it was.
    POST_PTC_A1_ROUTES = frozenset({"/civil-instant"})

    def test_this_increment_added_exactly_one_route(self):
        published = {
            path for path in registered_routes()
            if not path.startswith(("/openapi", "/docs", "/redoc"))
        } - self.POST_PTC_A1_ROUTES
        self.assertEqual(published - set(PREVIOUS), {ROUTE_PATH})

    def test_the_scientific_environment_generation_is_unchanged(self):
        self.assertEqual(scientific_environment.AUTHORITY_SOLVER_GENERATION,
                         SOLVER_GENERATION)


# --- 5. Structural isolation --------------------------------------------------


class TestSourceGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_source = inspect.getsource(server)
        cls.solver_source = inspect.getsource(astronomy_solver)
        cls.route = executable(server.solar_crossing)

    def test_the_route_marker_is_present_once_and_outside_the_a_grammar(self):
        """The A5 block stays the last governed A block in the server."""
        self.assertEqual(self.server_source.count(ROUTE_BLOCK_MARKER), 1)
        own = self.server_source.find(ROUTE_BLOCK_MARKER)
        self.assertIsNone(GOVERNED_BLOCK_MARKER.match(self.server_source, own))
        self.assertIsNone(GOVERNED_BLOCK_MARKER.search(self.server_source, own))
        self.assertGreater(self.server_source.find("def solar_crossing("), own)

    def test_the_solver_marker_is_present_once_and_outside_the_a_grammar(self):
        """The A4 block stays the last governed A block in the solver."""
        self.assertEqual(self.solver_source.count(SOLVER_BLOCK_MARKER), 1)
        own = self.solver_source.find(SOLVER_BLOCK_MARKER)
        self.assertIsNone(GOVERNED_BLOCK_MARKER.match(self.solver_source, own))
        self.assertIsNone(GOVERNED_BLOCK_MARKER.search(self.solver_source, own))
        self.assertGreater(self.solver_source.find("def find_solar_crossing("),
                           own)

    def test_the_route_delegates_and_hand_builds_nothing(self):
        for token in ("decode_tt_bits(", "find_solar_crossing(",
                      "project_solar_crossing(", "reason_detail("):
            with self.subTest(token=token):
                self.assertIn(token, self.route)
        for token in ("find_sunset", "almanac", "ts.", "{\"", "400.0",
                      "SOLAR_CROSSING_MAX_HORIZON_DAYS", "abs(", "status_code=404"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_only_the_two_governed_exceptions_are_caught(self):
        caught = re.findall(r"except (\w+)", self.route)
        self.assertEqual(caught, ["ExactTimeTransportError",
                                  "SunsetChronologyError"])

    def test_the_projection_performs_no_astronomy(self):
        body = executable(astronomical_event_transport.project_solar_crossing)
        for token in ("almanac", "find_discrete", "load_kernel", "wgs84",
                      "find_solar_crossing", "round("):
            with self.subTest(token=token):
                self.assertNotIn(token, body)
        self.assertEqual(body.count("project_exact_instant"), 5)

    def test_the_backward_search_is_not_built_on_the_uncertified_predecessor(self):
        for function in (
            astronomy_solver.find_solar_crossing,
            astronomy_solver._certified_last_crossing_before,
            astronomy_solver._clear_crossing_anchor_before,
        ):
            with self.subTest(function=function.__name__):
                body = executable(function)
                self.assertNotIn("find_sunset_predecessor", body)
                self.assertNotIn("find_sunset_bracket", body)

    def test_the_new_block_reuses_the_certified_structure(self):
        for function in (
            astronomy_solver._certified_first_crossing_after,
            astronomy_solver._certified_last_crossing_before,
        ):
            with self.subTest(function=function.__name__):
                self.assertIn("_certified_sunset_structure(", executable(function))

    def test_the_new_block_produces_no_calendar_meaning(self):
        own = self.solver_source.find(SOLVER_BLOCK_MARKER)
        block = self.solver_source[own:]
        executable_block = re.sub(r'"""[\s\S]*?"""', "", block)
        executable_block = "\n".join(
            line.split("#", 1)[0] for line in executable_block.split("\n")
        )
        self.assertGreater(len(executable_block), 2000)
        for token in ("weekday", "sabbath", "creation", "hpc", "carrier",
                      "rotation", "polar", "regime"):
            with self.subTest(token=token):
                self.assertNotIn(token, executable_block.lower())


if __name__ == "__main__":
    unittest.main()
