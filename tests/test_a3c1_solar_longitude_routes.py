"""A3c-1 verification for the exact solar-longitude event routes.

Covers GET /solar-longitude-event-before and GET /solar-longitude-event-after
in server.py: the request pipeline, the success projection, the transport and
scientific failure envelopes, exception discipline, absence semantics,
structural isolation and legacy-route preservation.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route functions are invoked directly, following A1c and A1d. httpx is not
installed, so starlette's TestClient is unavailable, and installing a
dependency in order to test is not permitted. Direct invocation exercises the
real route body, including the HTTPException mapping, which is asserted
through the exception object itself.

HANDOFF SEAM

The route -> solver handoff is certified by observation rather than inferred
from the answer. server.find_solar_longitude_event_* is temporarily replaced
by a recorder that DELEGATES to the real published solver, so the
astronomical result stays real while the exact call arguments become visible.
Both routes resolve the solver as a module global of server, so patching
server.<name> intercepts the real call site.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, governed event kinds, pinned artifact filenames, the expected
response key set and the legacy route inventory are declared here as
test-local literals. They are deliberately NOT imported from the modules
under test. The bit-level invariant is checked with struct.pack directly.

The published projection IS called, as certified A3c-0 infrastructure, to
assert that the route returns exactly it and does not hand-build a competing
response.
"""

import ast
import inspect
import struct
import unittest
from unittest import mock

from fastapi import HTTPException

import server
from astronomy_solver import (
    SunsetChronologyError,
    find_solar_longitude_event_after,
    find_solar_longitude_event_before,
)
from astronomical_event_transport import project_astronomical_event
from exact_time_transport import ExactTimeTransportError, encode_tt_bits

# --- Test-local oracles ----------------------------------------------------

BEFORE_PATH = "/solar-longitude-event-before"
AFTER_PATH = "/solar-longitude-event-after"

K000 = "SOLAR_LONGITUDE_000"
K090 = "SOLAR_LONGITUDE_090"
K180 = "SOLAR_LONGITUDE_180"
K270 = "SOLAR_LONGITUDE_270"
GOVERNED_KINDS = (K000, K090, K180, K270)

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"
PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

EXPECTED_RESPONSE_KEYS = ("ttBits", "tt", "utc", "kind", "kernel")
EXPECTED_QUERY_PARAMETERS = ["ttBits", "kind"]

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"
REASON_INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_EVENT_KIND_INVALID = "EVENT_KIND_INVALID"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED = "SOLAR_LONGITUDE_EVENT_UNRESOLVED"

SCIENTIFIC_REASONS = (
    REASON_INSTANT_STATE_INVALID,
    REASON_EVENT_KIND_INVALID,
    REASON_COVERAGE_EXHAUSTED,
    REASON_REACH_EXHAUSTED,
    REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED,
)

MODERN_ANCHOR = 2460678.0
DEEP_TIME_ANCHOR = 700000.0
DEEPER_TIME_ANCHOR = -1000000.0
FAR_FUTURE_ANCHOR = 5000000.0
BEYOND_COVERAGE = 9.9e6
BELOW_COVERAGE = -9.9e6

MALFORMED_TT_BITS = (
    ("lowercase", "4142c629703bf794"),
    ("mixed case", "4142C629703bf794"),
    ("too short", "4142C796"),
    ("too long", "4142C796BB75962EAA"),
    ("empty", ""),
    ("non-hex", "ZZZZZZZZZZZZZZZZ"),
    ("0x prefix", "0x4142C629703BF7"),
    ("nan bits", "7FF8000000000000"),
    ("positive infinity bits", "7FF0000000000000"),
    ("negative infinity bits", "FFF0000000000000"),
)

# The published surface this increment must not disturb.
LEGACY_ROUTES = (
    ("/health", "health", []),
    ("/delta-t/{year}", "delta_t", ["year"]),
    ("/delta-t-bce/{year}", "delta_t_bce", ["year"]),
    ("/equinox/{year}", "equinox", ["year"]),
    ("/equinox-bce/{year}", "equinox_bce", ["year"]),
    ("/season-events", "season_events", ["year"]),
    ("/season-events-bce/{year}", "season_events_bce", ["year"]),
    ("/solar_longitude", "solar", ["date"]),
    ("/subsolar-point", "subsolar", ["date"]),
    ("/sunset", "sunset", ["date", "latitude", "longitude"]),
    ("/sunset-after", "sunset_after", ["afterUTC", "latitude", "longitude"]),
    ("/sunset-successor", "sunset_successor",
     ["cursor", "latitude", "longitude"]),
)

# Routes FastAPI registers automatically for its own documentation surface.
# They are NOT Authority application routes and are not defined anywhere in
# server.py. They were demonstrated to exist at the published pre-A3c-1 HEAD,
# so an application-route comparison must exclude them or it would attribute
# framework behavior to this increment.
FRAMEWORK_ROUTES = frozenset(
    {
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)

A3C1_BLOCK_MARKER = "# A3c-1 - exact solar-longitude event routes."


# --- Independent helpers ---------------------------------------------------


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def before_response(ttBits, kind):
    return server.solar_longitude_event_before(ttBits=ttBits, kind=kind)


def after_response(ttBits, kind):
    return server.solar_longitude_event_after(ttBits=ttBits, kind=kind)


ROUTES = (
    ("before", before_response, find_solar_longitude_event_before,
     "find_solar_longitude_event_before"),
    ("after", after_response, find_solar_longitude_event_after,
     "find_solar_longitude_event_after"),
)


def registered_routes():
    """Path -> (methods, endpoint name, parameter names) from the live app."""
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


def executable_source(text):
    """Return a block's executable logic, prose excluded."""
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


class RouteAssertions(unittest.TestCase):
    def assert_route_rejects(self, call, ttBits, kind, reason):
        with self.assertRaises(HTTPException) as caught:
            call(ttBits, kind)
        error = caught.exception
        self.assertEqual(error.status_code, 400)
        self.assertIsInstance(error.detail, dict)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"], reason)
        return error

    def assert_is_projection(self, response, kind):
        self.assertIsNotNone(response)
        self.assertIsInstance(response, dict)
        self.assertEqual(tuple(response), EXPECTED_RESPONSE_KEYS)
        self.assertEqual(response["kind"], kind)
        self.assertIn(response["kernel"], PINNED_ARTIFACTS)
        self.assertIsInstance(response["tt"], float)
        self.assertIsInstance(response["ttBits"], str)
        self.assertTrue(response["utc"] is None
                        or isinstance(response["utc"], str))


# --- 1. Surface ------------------------------------------------------------


class TestRouteSurface(RouteAssertions):
    def test_both_paths_are_registered(self):
        found = registered_routes()
        self.assertIn(BEFORE_PATH, found)
        self.assertIn(AFTER_PATH, found)

    def test_both_routes_are_get_only(self):
        found = registered_routes()
        for path in (BEFORE_PATH, AFTER_PATH):
            with self.subTest(path=path):
                self.assertEqual(found[path][0] - {"HEAD"}, {"GET"})

    def test_query_parameters_are_exactly_ttbits_and_kind(self):
        found = registered_routes()
        for path in (BEFORE_PATH, AFTER_PATH):
            with self.subTest(path=path):
                self.assertEqual(found[path][2], EXPECTED_QUERY_PARAMETERS)

    def test_endpoint_names_are_the_expected_ones(self):
        found = registered_routes()
        self.assertEqual(found[BEFORE_PATH][1], "solar_longitude_event_before")
        self.assertEqual(found[AFTER_PATH][1], "solar_longitude_event_after")

    def test_no_year_date_direction_observer_or_kernel_parameter(self):
        forbidden = {"year", "date", "direction", "observer", "latitude",
                     "longitude", "kernel", "tt", "utc", "afterUTC",
                     "timezone", "tz", "hemisphere", "cursor"}
        found = registered_routes()
        for path in (BEFORE_PATH, AFTER_PATH):
            with self.subTest(path=path):
                self.assertEqual(set(found[path][2]) & forbidden, set())

    def test_no_third_new_route_was_introduced(self):
        found = set(registered_routes()) - FRAMEWORK_ROUTES
        expected = {path for path, _n, _p in LEGACY_ROUTES}
        expected |= {BEFORE_PATH, AFTER_PATH}
        self.assertEqual(found, expected)


# --- 2. Success pipeline ---------------------------------------------------


class TestSuccessPipeline(RouteAssertions):
    def test_response_is_exactly_the_published_projection(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for label, call, solver, _name in ROUTES:
            for kind in GOVERNED_KINDS:
                with self.subTest(direction=label, kind=kind):
                    expected = project_astronomical_event(
                        solver(MODERN_ANCHOR, kind)
                    )
                    self.assertEqual(call(anchor_bits, kind), expected)

    def test_five_keys_exactly_and_in_order(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for label, call, _solver, _name in ROUTES:
            for kind in GOVERNED_KINDS:
                with self.subTest(direction=label, kind=kind):
                    self.assert_is_projection(call(anchor_bits, kind), kind)

    def test_ttbits_and_tt_are_bit_identical(self):
        from exact_time_transport import decode_tt_bits

        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for label, call, _solver, _name in ROUTES:
            for kind in GOVERNED_KINDS:
                with self.subTest(direction=label, kind=kind):
                    response = call(anchor_bits, kind)
                    self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                                     bits(response["tt"]))

    def test_solver_receives_the_exact_decoded_state_once(self):
        """Observed at the real handoff, not inferred from the answer."""
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for label, call, solver, name in ROUTES:
            for kind in GOVERNED_KINDS:
                with self.subTest(direction=label, kind=kind):
                    calls = []

                    def recorder(tt, event_kind, _real=solver, _log=calls):
                        _log.append((tt, event_kind))
                        return _real(tt, event_kind)

                    with mock.patch.object(server, name, recorder):
                        call(anchor_bits, kind)

                    self.assertEqual(len(calls), 1)
                    supplied_tt, supplied_kind = calls[0]
                    self.assertEqual(bits(supplied_tt), bits(MODERN_ANCHOR))
                    self.assertIs(supplied_kind, kind)

    def test_each_route_uses_its_own_solver(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)

        with mock.patch.object(
            server, "find_solar_longitude_event_after",
            side_effect=AssertionError("before route called the after solver"),
        ):
            before_response(anchor_bits, K000)

        with mock.patch.object(
            server, "find_solar_longitude_event_before",
            side_effect=AssertionError("after route called the before solver"),
        ):
            after_response(anchor_bits, K000)

    def test_kernel_provenance_is_preserved(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for label, call, solver, _name in ROUTES:
            with self.subTest(direction=label):
                self.assertEqual(call(anchor_bits, K000)["kernel"],
                                 solver(MODERN_ANCHOR, K000).kernel)

    def test_strict_bracket_through_the_routes(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                earlier = before_response(anchor_bits, kind)["tt"]
                later = after_response(anchor_bits, kind)["tt"]
                self.assertLess(earlier, MODERN_ANCHOR)
                self.assertGreater(later, MODERN_ANCHOR)


# --- 3. Real scientific territory ------------------------------------------


class TestRealScientificTerritory(RouteAssertions):
    TERRITORY = (
        ("DE440 modern", MODERN_ANCHOR, DE440),
        ("DE441 part 1 deep time", DEEP_TIME_ANCHOR, DE441_PART_1),
        ("DE441 part 1 deeper time", DEEPER_TIME_ANCHOR, DE441_PART_1),
        ("DE441 part 2 far future", FAR_FUTURE_ANCHOR, DE441_PART_2),
    )

    def test_every_territory_and_kind_answers(self):
        for label, anchor, expected_kernel in self.TERRITORY:
            anchor_bits = encode_tt_bits(anchor)
            for direction, call, _solver, _name in ROUTES:
                for kind in GOVERNED_KINDS:
                    with self.subTest(territory=label, direction=direction,
                                      kind=kind):
                        response = call(anchor_bits, kind)
                        self.assert_is_projection(response, kind)
                        self.assertEqual(response["kernel"], expected_kernel)

    def test_all_three_artifacts_are_exercised(self):
        seen = set()
        for _label, anchor, _kernel in self.TERRITORY:
            seen.add(after_response(encode_tt_bits(anchor), K000)["kernel"])
        self.assertEqual(seen, set(PINNED_ARTIFACTS))

    def test_all_four_kinds_are_exercised(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        seen = {after_response(anchor_bits, k)["kind"] for k in GOVERNED_KINDS}
        self.assertEqual(seen, set(GOVERNED_KINDS))

    def test_deep_time_reference_is_null_not_missing(self):
        anchor_bits = encode_tt_bits(DEEP_TIME_ANCHOR)
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                response = after_response(anchor_bits, kind)
                self.assertIn("utc", response)
                self.assertIsNone(response["utc"])
                self.assertIsInstance(response["ttBits"], str)

    def test_representable_reference_is_present(self):
        response = after_response(encode_tt_bits(MODERN_ANCHOR), K000)
        self.assertIsInstance(response["utc"], str)
        self.assertTrue(response["utc"].endswith("+00:00"))

    def test_canonical_root_anchor_is_excluded_in_both_directions(self):
        """Strictness survives the route layer."""
        for label, anchor, _kernel in self.TERRITORY:
            for kind in GOVERNED_KINDS:
                with self.subTest(territory=label, kind=kind):
                    root = after_response(encode_tt_bits(anchor), kind)
                    root_bits = root["ttBits"]

                    following = after_response(root_bits, kind)
                    preceding = before_response(root_bits, kind)

                    self.assertNotEqual(following["ttBits"], root_bits)
                    self.assertNotEqual(preceding["ttBits"], root_bits)
                    self.assertGreater(following["tt"], root["tt"])
                    self.assertLess(preceding["tt"], root["tt"])


# --- 4. Transport failure matrix -------------------------------------------


class TestTransportFailures(RouteAssertions):
    def test_every_malformed_anchor_is_rejected(self):
        for label, value in MALFORMED_TT_BITS:
            for direction, call, _solver, _name in ROUTES:
                with self.subTest(case=label, direction=direction):
                    self.assert_route_rejects(
                        call, value, K000, REASON_TT_BITS_INVALID
                    )

    def test_transport_rejection_never_invokes_the_solver(self):
        for label, value in MALFORMED_TT_BITS:
            for direction, call, _solver, name in ROUTES:
                with self.subTest(case=label, direction=direction):
                    with mock.patch.object(
                        server, name,
                        side_effect=AssertionError("solver invoked"),
                    ) as solver:
                        with self.assertRaises(HTTPException):
                            call(value, K000)
                    self.assertEqual(solver.call_count, 0)

    def test_transport_reason_is_never_translated(self):
        error = self.assert_route_rejects(
            after_response, "4142c629703bf794", K000, REASON_TT_BITS_INVALID
        )
        self.assertNotEqual(error.detail["reason"],
                            REASON_INSTANT_STATE_INVALID)
        self.assertNotIn("CURSOR", error.detail["reason"])

    def test_transport_message_is_carried_and_cause_preserved(self):
        with self.assertRaises(HTTPException) as caught:
            after_response("ZZZZZZZZZZZZZZZZ", K000)
        error = caught.exception
        self.assertIsInstance(error.detail["message"], str)
        self.assertIn("ttBits", error.detail["message"])
        self.assertIsInstance(error.__cause__, ExactTimeTransportError)
        self.assertEqual(error.__cause__.reason, REASON_TT_BITS_INVALID)

    def test_a_valid_anchor_is_not_rejected(self):
        """Anti-vacuity: the matrix above must not pass by rejecting all."""
        self.assert_is_projection(
            after_response(encode_tt_bits(MODERN_ANCHOR), K000), K000
        )


# --- 5. Scientific reason matrix -------------------------------------------


class TestScientificFailures(RouteAssertions):
    def test_invalid_kind_is_reported_by_the_solver(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for kind in ("", " ", "000", "Spring", "spring_equinox",
                     "solar_longitude_000", "SOLAR_LONGITUDE_45"):
            for direction, call, _solver, _name in ROUTES:
                with self.subTest(kind=kind, direction=direction):
                    self.assert_route_rejects(
                        call, anchor_bits, kind, REASON_EVENT_KIND_INVALID
                    )

    def test_coverage_exhaustion_is_reported(self):
        for anchor in (BEYOND_COVERAGE, BELOW_COVERAGE):
            for direction, call, _solver, _name in ROUTES:
                with self.subTest(anchor=anchor, direction=direction):
                    self.assert_route_rejects(
                        call, encode_tt_bits(anchor), K000,
                        REASON_COVERAGE_EXHAUSTED,
                    )

    def test_every_governed_reason_crosses_unchanged(self):
        """Injected at the solver seam, including the unreachable ones."""
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for reason in SCIENTIFIC_REASONS:
            message = "GOVERNED FAILURE - %s detail text" % reason
            for direction, call, _solver, name in ROUTES:
                with self.subTest(reason=reason, direction=direction):
                    injected = SunsetChronologyError(reason, message)
                    with mock.patch.object(server, name, side_effect=injected):
                        with self.assertRaises(HTTPException) as caught:
                            call(anchor_bits, K000)
                    error = caught.exception
                    self.assertEqual(error.status_code, 400)
                    self.assertIsInstance(error.detail, dict)
                    self.assertEqual(error.detail["reason"], reason)
                    self.assertEqual(error.detail["message"], message)
                    self.assertIs(error.__cause__, injected)

    def test_reason_is_not_parsed_out_of_message_text(self):
        """A reason that does not appear in the message still crosses."""
        injected = SunsetChronologyError(
            REASON_REACH_EXHAUSTED, "a message mentioning no code at all"
        )
        with mock.patch.object(
            server, "find_solar_longitude_event_after", side_effect=injected
        ):
            with self.assertRaises(HTTPException) as caught:
                after_response(encode_tt_bits(MODERN_ANCHOR), K000)
        self.assertEqual(caught.exception.detail["reason"],
                         REASON_REACH_EXHAUSTED)
        self.assertNotIn(REASON_REACH_EXHAUSTED,
                         caught.exception.detail["message"])


# --- 6. Exception discipline -----------------------------------------------


class TestExceptionDiscipline(RouteAssertions):
    UNEXPECTED = (ValueError("unexpected seam"), RuntimeError("boom"),
                  MemoryError(), KeyError("missing"))

    def test_unexpected_solver_failures_propagate(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for injected in self.UNEXPECTED:
            for direction, call, _solver, name in ROUTES:
                with self.subTest(error=type(injected).__name__,
                                  direction=direction):
                    with mock.patch.object(server, name,
                                           side_effect=injected):
                        with self.assertRaises(type(injected)):
                            call(anchor_bits, K000)

    def test_unexpected_failures_are_not_converted_to_http(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for injected in self.UNEXPECTED:
            with self.subTest(error=type(injected).__name__):
                with mock.patch.object(
                    server, "find_solar_longitude_event_after",
                    side_effect=injected,
                ):
                    try:
                        after_response(anchor_bits, K000)
                    except HTTPException:
                        self.fail("unexpected failure relabelled as governed")
                    except type(injected):
                        pass

    def test_a_governed_failure_causes_exactly_one_solver_call(self):
        injected = SunsetChronologyError(REASON_EVENT_KIND_INVALID, "text")
        for direction, call, _solver, name in ROUTES:
            with self.subTest(direction=direction):
                with mock.patch.object(server, name,
                                       side_effect=injected) as solver:
                    with self.assertRaises(HTTPException):
                        call(encode_tt_bits(MODERN_ANCHOR), K000)
                self.assertEqual(solver.call_count, 1)

    def test_no_alternate_solver_is_attempted_after_a_failure(self):
        injected = SunsetChronologyError(REASON_COVERAGE_EXHAUSTED, "text")
        with mock.patch.object(
            server, "find_solar_longitude_event_after", side_effect=injected
        ), mock.patch.object(
            server, "find_solar_longitude_event_before",
            side_effect=AssertionError("the other solver was attempted"),
        ):
            with self.assertRaises(HTTPException):
                after_response(encode_tt_bits(MODERN_ANCHOR), K000)

    def test_no_kernel_selection_is_performed_by_the_route(self):
        with mock.patch.object(
            server, "get_default_kernel_name",
            side_effect=AssertionError("route selected a kernel"),
        ) as selector:
            after_response(encode_tt_bits(MODERN_ANCHOR), K000)
            before_response(encode_tt_bits(MODERN_ANCHOR), K000)
        self.assertEqual(selector.call_count, 0)


# --- 7. Absence semantics --------------------------------------------------


class TestAbsenceSemantics(RouteAssertions):
    def test_only_200_and_400_are_produced(self):
        anchors = (MODERN_ANCHOR, DEEP_TIME_ANCHOR, DEEPER_TIME_ANCHOR,
                   FAR_FUTURE_ANCHOR, BEYOND_COVERAGE, BELOW_COVERAGE)
        statuses = set()
        for anchor in anchors:
            anchor_bits = encode_tt_bits(anchor)
            for _direction, call, _solver, _name in ROUTES:
                for kind in GOVERNED_KINDS:
                    try:
                        self.assertIsNotNone(call(anchor_bits, kind))
                        statuses.add(200)
                    except HTTPException as error:
                        statuses.add(error.status_code)
        self.assertEqual(statuses, {200, 400})
        self.assertNotIn(404, statuses)

    def test_route_bodies_contain_no_404(self):
        for _direction, _call, _solver, name in ROUTES:
            endpoint = getattr(server, name.replace(
                "find_solar_longitude_event", "solar_longitude_event"))
            with self.subTest(route=endpoint.__name__):
                body = executable_source(inspect.getsource(endpoint))
                self.assertNotIn("404", body)
                self.assertNotIn("status_code=404", body)

    def test_success_is_never_none(self):
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        for _direction, call, _solver, _name in ROUTES:
            for kind in GOVERNED_KINDS:
                with self.subTest(kind=kind):
                    self.assertIsNotNone(call(anchor_bits, kind))

    def test_missing_parameters_are_framework_business(self):
        """FastAPI's 422 binding failure is not relabelled by the route."""
        for _direction, _call, _solver, name in ROUTES:
            endpoint = getattr(server, name.replace(
                "find_solar_longitude_event", "solar_longitude_event"))
            with self.subTest(route=endpoint.__name__):
                with self.assertRaises(TypeError):
                    endpoint()


# --- 8. Structural isolation -----------------------------------------------


class TestA3c1StructuralIsolation(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(server)
        index = source.find(A3C1_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3c-1 block marker not found")
        self.block = source[index:]
        self.preamble = source[:index]
        self.executable = executable_source(
            inspect.getsource(server.solar_longitude_event_before)
            + "\n"
            + inspect.getsource(server.solar_longitude_event_after)
        )

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def solar_longitude_event_before", self.block)
        self.assertIn("def solar_longitude_event_after", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        for token in ("decode_tt_bits", "find_solar_longitude_event_before",
                      "find_solar_longitude_event_after",
                      "project_astronomical_event", "reason_detail",
                      "ExactTimeTransportError", "SunsetChronologyError",
                      "HTTPException", "status_code=400", "from error"):
            with self.subTest(token=token):
                self.assertIn(token, self.executable)

    def test_no_civil_or_gregorian_routing(self):
        for token in ("choose_kernel_name", "PRIMARY_START_YEAR",
                      "PRIMARY_END_YEAR", "year", "Gregorian", "civil",
                      "parse_utc_datetime", "datetime", "fromisoformat",
                      "isoformat", "strftime", "timezone", "timestamp"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_no_observer_machinery(self):
        for token in ("observer", "wgs84", "latlon", "cursor_fields",
                      "LATITUDE_DOMAIN", "LONGITUDE_DOMAIN"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

        # A bare "longitude" cannot be scanned literally here: it necessarily
        # occurs inside the governed scientific identifier
        # solar_longitude_event_*, so a substring guard would convict the
        # route of the very thing its name describes. The absence of an
        # observer coordinate is asserted on the signatures instead, which is
        # authoritative rather than textual: a parameter that does not exist
        # cannot be supplied by any request.
        for name in ("solar_longitude_event_before",
                     "solar_longitude_event_after"):
            with self.subTest(route=name):
                parameters = list(
                    inspect.signature(getattr(server, name)).parameters
                )
                self.assertEqual(parameters, EXPECTED_QUERY_PARAMETERS)

    def test_no_route_level_astronomy_or_kernel_selection(self):
        for token in ("almanac", "find_discrete", "seasons", "load_kernel",
                      "get_default_kernel_name", "select_kernel",
                      "kernel_coverage", "geocentric_search_frontier",
                      "ts.tt_jd", "ts.utc"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_no_seasonal_terminology(self):
        for token in ("Spring", "Summer", "Autumn", "Winter", "equinox",
                      "solstice", "season"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_no_broad_exception_handling(self):
        for token in ("except Exception", "except ScientificEnvironmentError",
                      "except BaseException", "except:"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_exactly_two_exception_types_are_caught(self):
        self.assertEqual(
            self.executable.count("except ExactTimeTransportError"), 2
        )
        self.assertEqual(
            self.executable.count("except SunsetChronologyError"), 2
        )

    def test_no_second_tt_codec_and_no_hand_built_response(self):
        for token in ("struct", 'pack(">d"', "hex()", '"ttBits":', '"kind":',
                      '"kernel":', '"utc":'):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_no_tt_query_parameter_or_direction_dispatch(self):
        for token in ("direction", "def _solar_longitude_event",
                      "solver=", "dispatch"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_response_comes_only_from_the_published_projection(self):
        self.assertEqual(
            self.executable.count("return project_astronomical_event(event)"), 2
        )
        self.assertEqual(self.executable.count("return {"), 0)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in ("verify_runtime_scientific_components()",
                      "def parse_utc_datetime", "def cursor_fields",
                      "def health", "def sunset", "def sunset_after",
                      "def sunset_successor", "def equinox",
                      "def season_events"):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)

    def test_runtime_gate_still_precedes_every_astronomy_import(self):
        """The A0.3 ordering must survive the new imports."""
        source = inspect.getsource(server)
        gate = source.index("verify_runtime_scientific_components()\n")
        for module in ("from astronomy_solver import",
                       "from astronomical_event_transport import",
                       "from exact_time_transport import",
                       "from sunset_cursor import"):
            with self.subTest(module=module):
                self.assertGreater(source.index(module), gate)


# --- 9. Legacy preservation ------------------------------------------------


class TestLegacyRoutesUnchanged(unittest.TestCase):
    def test_every_legacy_route_is_still_registered_unchanged(self):
        found = registered_routes()
        for path, name, parameters in LEGACY_ROUTES:
            with self.subTest(path=path):
                self.assertIn(path, found)
                methods, endpoint_name, endpoint_parameters = found[path]
                self.assertEqual(methods - {"HEAD"}, {"GET"})
                self.assertEqual(endpoint_name, name)
                self.assertEqual(endpoint_parameters, parameters)

    def test_legacy_count_is_exactly_twelve(self):
        found = set(registered_routes()) - FRAMEWORK_ROUTES
        legacy = found - {BEFORE_PATH, AFTER_PATH}
        self.assertEqual(len(legacy), len(LEGACY_ROUTES))
        self.assertEqual(legacy, {path for path, _n, _p in LEGACY_ROUTES})

    def test_legacy_failure_still_carries_string_detail(self):
        """Proof the structured envelope was not globalized."""
        with self.assertRaises(HTTPException) as caught:
            server.equinox("not-a-year")
        error = caught.exception
        self.assertEqual(error.status_code, 400)
        self.assertIsInstance(error.detail, str)
        self.assertNotIsInstance(error.detail, dict)

    def test_legacy_health_is_unchanged(self):
        response = server.health()
        self.assertEqual(set(response), {"status", "defaultKernel"})
        self.assertEqual(response["status"], "ok")

    def test_new_structured_detail_is_not_used_by_legacy_routes(self):
        legacy_sources = "\n".join(
            inspect.getsource(getattr(server, name))
            for _path, name, _parameters in LEGACY_ROUTES
        )
        self.assertNotIn("reason_detail", legacy_sources)


if __name__ == "__main__":
    unittest.main()
