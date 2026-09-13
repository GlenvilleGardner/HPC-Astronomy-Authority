"""A3c-2 verification for the exact sunset bracket route.

Covers GET /sunset-bracket in server.py and the two projections added to
astronomical_event_transport.py: the request pipeline, the success
projection, independent boundary provenance, the transport and scientific
failure envelopes, astronomical absence, exception discipline, structural
isolation and legacy-route preservation.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d and A3c-1. httpx
is not installed, so starlette's TestClient is unavailable, and installing a
dependency in order to test is not permitted. Direct invocation exercises the
real route body, including the HTTPException mapping, which is asserted
through the exception object itself.

HANDOFF SEAM

The route -> substrate handoff is certified by observation rather than
inferred from the answer. server.find_sunset_bracket is temporarily replaced
by a recorder that DELEGATES to the real published function, so the
astronomical result stays real while the exact call arguments become visible.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the expected key sets, the observer
domains and the legacy route inventory are declared here as test-local
literals. The bit-level invariant is checked with struct.pack directly. The
published projections ARE called, as certified A3c-0/A3c-2 infrastructure, to
assert that the route returns exactly them and hand-builds nothing.

EVIDENCE DISCIPLINE

MEASURED  - observed against the real artifacts at test time.
TAXONOMY  - certified by injecting a governed exception, where no ordinary
            input witness exists. Labelled as such and never presented as a
            real-input observation.
"""

import ast
import inspect
import struct
import unittest
from unittest import mock

from fastapi import HTTPException

import astronomical_event_transport
import server
from astronomy_solver import (
    SunsetChronologyError,
    find_sunset_bracket,
    kernel_coverage_tt,
)
from astronomical_event_transport import (
    project_exact_instant,
    project_sunset_bracket,
    project_sunset_event,
)
from exact_time_transport import ExactTimeTransportError, decode_tt_bits, encode_tt_bits

# --- Test-local oracles ----------------------------------------------------

BRACKET_PATH = "/sunset-bracket"

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"
PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

EXPECTED_BRACKET_KEYS = ("anchor", "previous", "next")
EXPECTED_ANCHOR_KEYS = ("ttBits", "tt", "utc")
EXPECTED_BOUNDARY_KEYS = ("ttBits", "tt", "utc", "kernel")
EXPECTED_QUERY_PARAMETERS = ["ttBits", "latitude", "longitude"]

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"
REASON_INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_SUNSET_BRACKET_ABSENT = "SUNSET_BRACKET_ABSENT"

SCIENTIFIC_REASONS = (
    REASON_INSTANT_STATE_INVALID,
    REASON_OBSERVER_OUT_OF_DOMAIN,
    REASON_COVERAGE_EXHAUSTED,
    REASON_REACH_EXHAUSTED,
)

# A mid-latitude observer with ordinary sunsets, and a polar observer whose
# instants are genuinely not bracketed by sunsets.
NYC = {"latitude": 40.7406, "longitude": -73.9}
POLAR = {"latitude": 78.0, "longitude": 15.0}

MODERN_ANCHOR = 2460678.0
DEEP_TIME_ANCHOR = 700000.0
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

# The 14 application routes published before this increment.
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
    ("/solar-longitude-event-before", "solar_longitude_event_before",
     ["ttBits", "kind"]),
    ("/solar-longitude-event-after", "solar_longitude_event_after",
     ["ttBits", "kind"]),
)

# Registered automatically by FastAPI; not Authority application routes and
# not defined anywhere in server.py.
FRAMEWORK_ROUTES = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)

A3C2_BLOCK_MARKER = "# A3c-2 - exact sunset bracket route."


# --- Independent helpers ---------------------------------------------------


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def bracket_response(ttBits, latitude=NYC["latitude"],
                     longitude=NYC["longitude"]):
    return server.sunset_bracket(
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


def artifact_boundary_anchor():
    """An anchor whose two sunset boundaries fall under different artifacts.

    MEASURED: located just inside DE440's declared ceiling, where the
    backward search stays in DE440 and the forward search must move to
    DE441 part 2.
    """
    ceiling = kernel_coverage_tt(DE440).tt_end
    for offset in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        candidate = ceiling - offset
        result = find_sunset_bracket(
            candidate, NYC["latitude"], NYC["longitude"]
        )
        if result is not None and result.previous.kernel != result.next.kernel:
            return candidate, result
    return None, None


class BracketAssertions(unittest.TestCase):
    def assert_rejects(self, ttBits, reason, status=400, **observer):
        parameters = dict(NYC)
        parameters.update(observer)
        with self.assertRaises(HTTPException) as caught:
            bracket_response(ttBits, **parameters)
        error = caught.exception
        self.assertEqual(error.status_code, status)
        self.assertIsInstance(error.detail, dict)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"], reason)
        return error

    def assert_is_bracket(self, response):
        self.assertIsNotNone(response)
        self.assertIsInstance(response, dict)
        self.assertEqual(tuple(response), EXPECTED_BRACKET_KEYS)
        self.assertEqual(tuple(response["anchor"]), EXPECTED_ANCHOR_KEYS)
        for side in ("previous", "next"):
            self.assertEqual(tuple(response[side]), EXPECTED_BOUNDARY_KEYS)
            self.assertIn(response[side]["kernel"], PINNED_ARTIFACTS)
        for key in EXPECTED_BRACKET_KEYS:
            event = response[key]
            self.assertEqual(bits(decode_tt_bits(event["ttBits"])),
                             bits(event["tt"]))
            self.assertTrue(event["utc"] is None
                            or isinstance(event["utc"], str))
        return response


# --- 1. Route surface ------------------------------------------------------


class TestRouteSurface(BracketAssertions):
    def test_path_is_registered(self):
        self.assertIn(BRACKET_PATH, registered_routes())

    def test_route_is_get_only(self):
        methods = registered_routes()[BRACKET_PATH][0]
        self.assertEqual(methods - {"HEAD"}, {"GET"})

    def test_parameters_are_exactly_ttbits_latitude_longitude(self):
        self.assertEqual(registered_routes()[BRACKET_PATH][2],
                         EXPECTED_QUERY_PARAMETERS)

    def test_endpoint_name(self):
        self.assertEqual(registered_routes()[BRACKET_PATH][1],
                         "sunset_bracket")

    def test_no_year_date_timezone_kernel_direction_or_cursor_parameter(self):
        forbidden = {"year", "date", "timezone", "tz", "kernel", "direction",
                     "cursor", "kind", "tt", "utc", "afterUTC", "hemisphere"}
        self.assertEqual(
            set(registered_routes()[BRACKET_PATH][2]) & forbidden, set()
        )

    def test_exactly_one_application_route_was_added(self):
        found = set(registered_routes()) - FRAMEWORK_ROUTES
        expected = {path for path, _n, _p in LEGACY_ROUTES} | {BRACKET_PATH}
        self.assertEqual(found, expected)
        self.assertEqual(len(found), 15)


# --- 2. Projection helpers -------------------------------------------------


class TestProjections(BracketAssertions):
    @classmethod
    def setUpClass(cls):
        cls.bracket = find_sunset_bracket(
            MODERN_ANCHOR, NYC["latitude"], NYC["longitude"]
        )

    def test_sunset_event_projection_extends_the_exact_instant(self):
        event = self.bracket.previous
        base = project_exact_instant(event.tt)
        projected = project_sunset_event(event)
        self.assertEqual(tuple(projected), EXPECTED_BOUNDARY_KEYS)
        for key in EXPECTED_ANCHOR_KEYS:
            self.assertEqual(projected[key], base[key])
        self.assertEqual(projected["kernel"], event.kernel)

    def test_bracket_projection_uses_the_shared_projections(self):
        projected = project_sunset_bracket(self.bracket)
        self.assertEqual(projected["anchor"],
                         project_exact_instant(self.bracket.anchor))
        self.assertEqual(projected["previous"],
                         project_sunset_event(self.bracket.previous))
        self.assertEqual(projected["next"],
                         project_sunset_event(self.bracket.next))

    def test_anchor_has_exactly_three_keys_and_no_kernel(self):
        projected = project_sunset_bracket(self.bracket)
        self.assertEqual(tuple(projected["anchor"]), EXPECTED_ANCHOR_KEYS)
        self.assertNotIn("kernel", projected["anchor"])

    def test_boundaries_have_exactly_four_keys(self):
        projected = project_sunset_bracket(self.bracket)
        for side in ("previous", "next"):
            with self.subTest(side=side):
                self.assertEqual(tuple(projected[side]),
                                 EXPECTED_BOUNDARY_KEYS)

    def test_every_projected_instant_is_bit_identical(self):
        projected = project_sunset_bracket(self.bracket)
        for key, source in (("anchor", self.bracket.anchor),
                            ("previous", self.bracket.previous.tt),
                            ("next", self.bracket.next.tt)):
            with self.subTest(key=key):
                self.assertEqual(bits(projected[key]["tt"]), bits(source))
                self.assertEqual(
                    bits(decode_tt_bits(projected[key]["ttBits"])),
                    bits(source),
                )

    def test_the_record_is_not_modified(self):
        before = (self.bracket.anchor, self.bracket.previous.tt,
                  self.bracket.previous.kernel, self.bracket.next.tt,
                  self.bracket.next.kernel)
        project_sunset_bracket(self.bracket)
        after = (self.bracket.anchor, self.bracket.previous.tt,
                 self.bracket.previous.kernel, self.bracket.next.tt,
                 self.bracket.next.kernel)
        self.assertEqual(before, after)


# --- 3. Success pipeline ---------------------------------------------------


class TestSuccessPipeline(BracketAssertions):
    def test_response_is_exactly_the_published_projection(self):
        expected = project_sunset_bracket(
            find_sunset_bracket(MODERN_ANCHOR, NYC["latitude"],
                                NYC["longitude"])
        )
        self.assertEqual(bracket_response(encode_tt_bits(MODERN_ANCHOR)),
                         expected)

    def test_substrate_receives_the_exact_state_and_observer_once(self):
        """Observed at the real handoff, not inferred from the answer."""
        calls = []
        real = find_sunset_bracket

        def recorder(tt, latitude, longitude):
            calls.append((tt, latitude, longitude))
            return real(tt, latitude, longitude)

        with mock.patch.object(server, "find_sunset_bracket", recorder):
            bracket_response(encode_tt_bits(MODERN_ANCHOR))

        self.assertEqual(len(calls), 1)
        supplied_tt, supplied_lat, supplied_lon = calls[0]
        self.assertEqual(bits(supplied_tt), bits(MODERN_ANCHOR))
        self.assertEqual(supplied_lat, NYC["latitude"])
        self.assertEqual(supplied_lon, NYC["longitude"])

    def test_observer_is_passed_through_unaltered(self):
        for latitude, longitude in ((0.0, 0.0), (-0.0, -0.0), (89.9, 179.9),
                                    (-40.7406, 73.9)):
            with self.subTest(observer=(latitude, longitude)):
                calls = []
                real = find_sunset_bracket

                def recorder(tt, lat, lon, _log=calls, _real=real):
                    _log.append((lat, lon))
                    return _real(tt, lat, lon)

                with mock.patch.object(server, "find_sunset_bracket", recorder):
                    try:
                        bracket_response(encode_tt_bits(MODERN_ANCHOR),
                                         latitude=latitude, longitude=longitude)
                    except HTTPException:
                        pass
                self.assertEqual(calls[0], (latitude, longitude))

    def test_anchor_is_the_supplied_state(self):
        response = bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(bits(response["anchor"]["tt"]), bits(MODERN_ANCHOR))

    def test_bracket_ordering_holds_in_the_response(self):
        response = self.assert_is_bracket(
            bracket_response(encode_tt_bits(MODERN_ANCHOR))
        )
        self.assertLess(response["previous"]["tt"], response["anchor"]["tt"])
        self.assertLess(response["anchor"]["tt"], response["next"]["tt"])


# --- 4. Real scientific territory ------------------------------------------


class TestRealScientificTerritory(BracketAssertions):
    def test_ordinary_modern_bracket(self):
        response = self.assert_is_bracket(
            bracket_response(encode_tt_bits(MODERN_ANCHOR))
        )
        self.assertEqual(response["previous"]["kernel"], DE440)
        self.assertEqual(response["next"]["kernel"], DE440)
        self.assertIsInstance(response["anchor"]["utc"], str)

    def test_arbitrary_non_sunset_anchor(self):
        """An anchor with no particular relationship to a crossing."""
        for anchor in (MODERN_ANCHOR + 0.123456, MODERN_ANCHOR - 0.31415,
                       MODERN_ANCHOR + 0.5):
            with self.subTest(anchor=anchor):
                response = self.assert_is_bracket(
                    bracket_response(encode_tt_bits(anchor))
                )
                self.assertLess(response["previous"]["tt"], anchor)
                self.assertGreater(response["next"]["tt"], anchor)

    def test_exact_sunset_anchor_is_excluded_from_both_sides(self):
        first = bracket_response(encode_tt_bits(MODERN_ANCHOR))
        sunset_bits = first["next"]["ttBits"]
        second = self.assert_is_bracket(bracket_response(sunset_bits))
        self.assertEqual(second["anchor"]["ttBits"], sunset_bits)
        self.assertNotEqual(second["previous"]["ttBits"], sunset_bits)
        self.assertNotEqual(second["next"]["ttBits"], sunset_bits)
        self.assertLess(second["previous"]["tt"], second["anchor"]["tt"])
        self.assertLess(second["anchor"]["tt"], second["next"]["tt"])

    def test_deep_time_bracket(self):
        response = self.assert_is_bracket(
            bracket_response(encode_tt_bits(DEEP_TIME_ANCHOR))
        )
        self.assertEqual(response["previous"]["kernel"], DE441_PART_1)
        self.assertEqual(response["next"]["kernel"], DE441_PART_1)
        for key in EXPECTED_BRACKET_KEYS:
            with self.subTest(key=key):
                self.assertIn("utc", response[key])
                self.assertIsNone(response[key]["utc"])

    def test_far_future_bracket(self):
        response = self.assert_is_bracket(
            bracket_response(encode_tt_bits(FAR_FUTURE_ANCHOR))
        )
        self.assertEqual(response["previous"]["kernel"], DE441_PART_2)
        self.assertIsInstance(response["previous"]["utc"], str)

    def test_real_artifact_boundary_bracket_keeps_both_provenances(self):
        """MEASURED, not structural: a real bracket spanning two artifacts."""
        anchor, record = artifact_boundary_anchor()
        self.assertIsNotNone(anchor, "no artifact-boundary anchor located")
        response = self.assert_is_bracket(
            bracket_response(encode_tt_bits(anchor))
        )
        self.assertNotEqual(response["previous"]["kernel"],
                            response["next"]["kernel"])
        self.assertEqual(response["previous"]["kernel"], record.previous.kernel)
        self.assertEqual(response["next"]["kernel"], record.next.kernel)
        self.assertEqual(response["previous"]["kernel"], DE440)
        self.assertEqual(response["next"]["kernel"], DE441_PART_2)

    def test_no_bracket_level_kernel_is_reported(self):
        anchor, _record = artifact_boundary_anchor()
        response = bracket_response(encode_tt_bits(anchor))
        self.assertNotIn("kernel", response)
        self.assertNotIn("kernel", response["anchor"])

    def test_all_three_artifacts_are_exercised(self):
        seen = set()
        for anchor in (MODERN_ANCHOR, DEEP_TIME_ANCHOR, FAR_FUTURE_ANCHOR):
            response = bracket_response(encode_tt_bits(anchor))
            seen.add(response["previous"]["kernel"])
            seen.add(response["next"]["kernel"])
        self.assertEqual(seen, set(PINNED_ARTIFACTS))


# --- 5. Transport failures -------------------------------------------------


class TestTransportFailures(BracketAssertions):
    def test_every_malformed_anchor_is_rejected(self):
        for label, value in MALFORMED_TT_BITS:
            with self.subTest(case=label):
                self.assert_rejects(value, REASON_TT_BITS_INVALID)

    def test_transport_rejection_never_invokes_the_substrate(self):
        for label, value in MALFORMED_TT_BITS:
            with self.subTest(case=label):
                with mock.patch.object(
                    server, "find_sunset_bracket",
                    side_effect=AssertionError("substrate invoked"),
                ) as substrate:
                    with self.assertRaises(HTTPException):
                        bracket_response(value)
                self.assertEqual(substrate.call_count, 0)

    def test_transport_cause_is_preserved(self):
        with self.assertRaises(HTTPException) as caught:
            bracket_response("ZZZZZZZZZZZZZZZZ")
        error = caught.exception
        self.assertIsInstance(error.__cause__, ExactTimeTransportError)
        self.assertEqual(error.__cause__.reason, REASON_TT_BITS_INVALID)
        self.assertIn("ttBits", error.detail["message"])

    def test_transport_reason_is_never_a_scientific_reason(self):
        error = self.assert_rejects("4142c629703bf794", REASON_TT_BITS_INVALID)
        self.assertNotIn(error.detail["reason"], SCIENTIFIC_REASONS)

    def test_a_valid_anchor_is_not_rejected(self):
        """Anti-vacuity for the matrix above."""
        self.assert_is_bracket(bracket_response(encode_tt_bits(MODERN_ANCHOR)))


# --- 6. Scientific failures ------------------------------------------------


class TestScientificFailures(BracketAssertions):
    def test_observer_out_of_domain_from_real_coordinates(self):
        """MEASURED from real invalid coordinates."""
        for latitude, longitude in ((91.0, 0.0), (-91.0, 0.0), (0.0, 181.0),
                                    (0.0, -181.0), (float("nan"), 0.0)):
            with self.subTest(observer=(latitude, longitude)):
                self.assert_rejects(
                    encode_tt_bits(MODERN_ANCHOR),
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    latitude=latitude, longitude=longitude,
                )

    def test_coverage_exhausted_from_real_unsupported_anchor(self):
        """MEASURED from a real anchor outside all pinned coverage."""
        for anchor in (BEYOND_COVERAGE, BELOW_COVERAGE):
            with self.subTest(anchor=anchor):
                self.assert_rejects(encode_tt_bits(anchor),
                                    REASON_COVERAGE_EXHAUSTED)

    def test_every_governed_reason_crosses_unchanged(self):
        """TAXONOMY: injected, covering the reasons with no ordinary witness.

        INSTANT_STATE_INVALID is structurally unreachable through this route
        because the codec already guarantees a finite float, and
        EPHEMERIS_REACH_EXHAUSTED has no ordinary input witness, so both are
        certified here by injecting the governed exception rather than by
        fabricating a real-input case.
        """
        for reason in SCIENTIFIC_REASONS:
            message = "GOVERNED FAILURE - %s detail text" % reason
            with self.subTest(reason=reason):
                injected = SunsetChronologyError(reason, message)
                with mock.patch.object(server, "find_sunset_bracket",
                                       side_effect=injected):
                    with self.assertRaises(HTTPException) as caught:
                        bracket_response(encode_tt_bits(MODERN_ANCHOR))
                error = caught.exception
                self.assertEqual(error.status_code, 400)
                self.assertIsInstance(error.detail, dict)
                self.assertEqual(error.detail["reason"], reason)
                self.assertEqual(error.detail["message"], message)
                self.assertIs(error.__cause__, injected)

    def test_reason_is_not_parsed_from_message_text(self):
        injected = SunsetChronologyError(
            REASON_REACH_EXHAUSTED, "a message mentioning no code at all"
        )
        with mock.patch.object(server, "find_sunset_bracket",
                               side_effect=injected):
            with self.assertRaises(HTTPException) as caught:
                bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(caught.exception.detail["reason"],
                         REASON_REACH_EXHAUSTED)
        self.assertNotIn(REASON_REACH_EXHAUSTED,
                         caught.exception.detail["message"])

    def test_scientific_failure_is_never_reported_as_absence(self):
        injected = SunsetChronologyError(REASON_COVERAGE_EXHAUSTED, "text")
        with mock.patch.object(server, "find_sunset_bracket",
                               side_effect=injected):
            with self.assertRaises(HTTPException) as caught:
                bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertNotEqual(caught.exception.detail["reason"],
                            REASON_SUNSET_BRACKET_ABSENT)


# --- 7. Astronomical absence -----------------------------------------------


class TestAstronomicalAbsence(BracketAssertions):
    def test_the_polar_case_really_returns_none(self):
        """MEASURED: the substrate itself reports absence, not the route."""
        self.assertIsNone(
            find_sunset_bracket(MODERN_ANCHOR, POLAR["latitude"],
                                POLAR["longitude"])
        )

    def test_absence_is_reported_as_404_with_a_stable_reason(self):
        with self.assertRaises(HTTPException) as caught:
            bracket_response(encode_tt_bits(MODERN_ANCHOR), **POLAR)
        error = caught.exception
        self.assertEqual(error.status_code, 404)
        self.assertIsInstance(error.detail, dict)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"],
                         REASON_SUNSET_BRACKET_ABSENT)
        self.assertIsInstance(error.detail["message"], str)

    def test_absence_is_not_reported_as_400(self):
        with self.assertRaises(HTTPException) as caught:
            bracket_response(encode_tt_bits(MODERN_ANCHOR), **POLAR)
        self.assertNotEqual(caught.exception.status_code, 400)

    def test_absence_reason_is_not_a_scientific_or_transport_reason(self):
        with self.assertRaises(HTTPException) as caught:
            bracket_response(encode_tt_bits(MODERN_ANCHOR), **POLAR)
        reason = caught.exception.detail["reason"]
        self.assertNotIn(reason, SCIENTIFIC_REASONS)
        self.assertNotEqual(reason, REASON_TT_BITS_INVALID)

    def test_absence_fabricates_no_event(self):
        """No success body is produced and no boundary is invented."""
        with self.assertRaises(HTTPException) as caught:
            bracket_response(encode_tt_bits(MODERN_ANCHOR), **POLAR)
        detail = caught.exception.detail
        for token in ("ttBits", "tt", "previous", "next", "anchor", "kernel"):
            with self.subTest(token=token):
                self.assertNotIn(token, detail)

    def test_absence_reason_lives_in_the_route_layer_only(self):
        import astronomy_solver

        self.assertEqual(server.REASON_SUNSET_BRACKET_ABSENT,
                         REASON_SUNSET_BRACKET_ABSENT)
        self.assertFalse(
            any(getattr(astronomy_solver, name, None)
                == REASON_SUNSET_BRACKET_ABSENT
                for name in dir(astronomy_solver))
        )

    def test_a_non_polar_observer_still_succeeds(self):
        """Anti-vacuity: absence must not be the answer everywhere."""
        self.assert_is_bracket(bracket_response(encode_tt_bits(MODERN_ANCHOR)))


# --- 8. Exception discipline -----------------------------------------------


class TestExceptionDiscipline(BracketAssertions):
    UNEXPECTED = (ValueError("unexpected seam"), RuntimeError("boom"),
                  MemoryError(), KeyError("missing"))

    def test_unexpected_failures_propagate(self):
        for injected in self.UNEXPECTED:
            with self.subTest(error=type(injected).__name__):
                with mock.patch.object(server, "find_sunset_bracket",
                                       side_effect=injected):
                    with self.assertRaises(type(injected)):
                        bracket_response(encode_tt_bits(MODERN_ANCHOR))

    def test_unexpected_failures_are_not_converted_to_http(self):
        for injected in self.UNEXPECTED:
            with self.subTest(error=type(injected).__name__):
                with mock.patch.object(server, "find_sunset_bracket",
                                       side_effect=injected):
                    try:
                        bracket_response(encode_tt_bits(MODERN_ANCHOR))
                    except HTTPException:
                        self.fail("unexpected failure relabelled as governed")
                    except type(injected):
                        pass

    def test_exactly_one_substrate_call_on_a_governed_failure(self):
        injected = SunsetChronologyError(REASON_COVERAGE_EXHAUSTED, "text")
        with mock.patch.object(server, "find_sunset_bracket",
                               side_effect=injected) as substrate:
            with self.assertRaises(HTTPException):
                bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(substrate.call_count, 1)

    def test_exactly_one_substrate_call_on_success(self):
        real = find_sunset_bracket
        with mock.patch.object(server, "find_sunset_bracket",
                               side_effect=real) as substrate:
            bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(substrate.call_count, 1)

    def test_no_alternate_solver_or_kernel_selection(self):
        with mock.patch.object(
            server, "get_default_kernel_name",
            side_effect=AssertionError("route selected a kernel"),
        ) as selector, mock.patch.object(
            server, "find_sunset_utc",
            side_effect=AssertionError("alternate sunset solver used"),
        ) as alternate:
            bracket_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(selector.call_count, 0)
        self.assertEqual(alternate.call_count, 0)


# --- 9. Structural isolation -----------------------------------------------


class TestA3c2StructuralIsolation(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(server)
        index = source.find(A3C2_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3c-2 block marker not found")
        self.block = source[index:]
        self.preamble = source[:index]
        self.route = executable_source(
            inspect.getsource(server.sunset_bracket)
        )
        self.projections = executable_source(
            inspect.getsource(
                astronomical_event_transport.project_sunset_event)
            + "\n"
            + inspect.getsource(
                astronomical_event_transport.project_sunset_bracket)
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        for token in ("decode_tt_bits", "find_sunset_bracket",
                      "project_sunset_bracket", "reason_detail",
                      "ExactTimeTransportError", "SunsetChronologyError",
                      "HTTPException", "status_code=400", "status_code=404",
                      "from error", "REASON_SUNSET_BRACKET_ABSENT"):
            with self.subTest(token=token):
                self.assertIn(token, self.route)

    def test_route_has_no_civil_or_gregorian_routing(self):
        for token in ("choose_kernel_name", "PRIMARY_START_YEAR",
                      "PRIMARY_END_YEAR", "parse_utc_datetime", "datetime",
                      "fromisoformat", "isoformat", "strftime", "timezone",
                      "timestamp", "Gregorian", "civil", "year"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_route_performs_no_astronomy_or_kernel_selection(self):
        for token in ("almanac", "find_discrete", "seasons", "load_kernel",
                      "get_default_kernel_name", "select_kernel",
                      "kernel_coverage", "supported_search_frontier",
                      "find_sunset_utc", "find_sunset_predecessor",
                      "find_sunset_from_instant", "ts.tt_jd", "wgs84"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_route_duplicates_no_tt_codec_and_builds_no_response(self):
        for token in ("struct", "pack(", "unpack(", "hex()", "encode_tt_bits",
                      '"ttBits":', '"anchor":', '"previous":', '"next":'):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_route_has_no_broad_exception_handler(self):
        for token in ("except Exception", "except ScientificEnvironmentError",
                      "except BaseException", "except:", "except ValueError",
                      "except RuntimeError"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.route)

    def test_route_catches_exactly_two_governed_types(self):
        self.assertEqual(self.route.count("except ExactTimeTransportError"), 1)
        self.assertEqual(self.route.count("except SunsetChronologyError"), 1)
        self.assertEqual(self.route.count("from error"), 2)

    def test_route_calls_each_step_once(self):
        self.assertEqual(self.route.count("decode_tt_bits("), 1)
        self.assertEqual(self.route.count("find_sunset_bracket("), 1)
        self.assertEqual(self.route.count("project_sunset_bracket("), 1)

    def test_route_returns_only_the_published_projection(self):
        self.assertIn("return project_sunset_bracket(bracket)", self.route)
        self.assertEqual(self.route.count("return "), 1)

    def test_projections_reuse_the_shared_primitive(self):
        self.assertIn("project_exact_instant(", self.projections)
        for token in ("struct", "hex()", "utc_datetime", "isoformat",
                      "datetime", "load_kernel", "almanac"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.projections)

    def test_projections_do_not_invent_anchor_provenance(self):
        self.assertNotIn('"anchor"]["kernel"', self.projections)
        self.assertEqual(self.projections.count('"kernel"'), 1)
        self.assertIn('projection["kernel"] = event.kernel', self.projections)

    def test_transport_module_defines_the_expected_symbols(self):
        tree = ast.parse(inspect.getsource(astronomical_event_transport))
        self.assertEqual(
            [n.name for n in tree.body if isinstance(n, ast.FunctionDef)],
            ["utc_reference", "project_exact_instant",
             "project_astronomical_event", "project_sunset_event",
             "project_sunset_bracket", "reason_detail"],
        )
        self.assertEqual(
            [n for n in tree.body if isinstance(n, ast.Assign)], []
        )

    def test_runtime_gate_still_precedes_every_astronomy_import(self):
        source = inspect.getsource(server)
        gate = source.index("verify_runtime_scientific_components()\n")
        for module in ("from astronomy_solver import",
                       "from astronomical_event_transport import",
                       "from exact_time_transport import",
                       "from sunset_cursor import"):
            with self.subTest(module=module):
                self.assertGreater(source.index(module), gate)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in ("verify_runtime_scientific_components()",
                      "def health", "def sunset", "def sunset_after",
                      "def sunset_successor", "def equinox",
                      "def season_events",
                      "def solar_longitude_event_before",
                      "def solar_longitude_event_after"):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


# --- 10. Legacy preservation -----------------------------------------------


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

    def test_legacy_count_is_exactly_fourteen(self):
        found = set(registered_routes()) - FRAMEWORK_ROUTES
        legacy = found - {BRACKET_PATH}
        self.assertEqual(len(legacy), len(LEGACY_ROUTES))
        self.assertEqual(legacy, {path for path, _n, _p in LEGACY_ROUTES})

    def test_legacy_failure_still_carries_string_detail(self):
        with self.assertRaises(HTTPException) as caught:
            server.equinox("not-a-year")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIsInstance(caught.exception.detail, str)

    def test_a3c1_routes_still_carry_structured_detail(self):
        with self.assertRaises(HTTPException) as caught:
            server.solar_longitude_event_after(ttBits="nope", kind="x")
        self.assertIsInstance(caught.exception.detail, dict)
        self.assertEqual(caught.exception.detail["reason"],
                         REASON_TT_BITS_INVALID)

    def test_legacy_routes_do_not_use_the_structured_envelope(self):
        legacy_sources = "\n".join(
            inspect.getsource(getattr(server, name))
            for _path, name, _parameters in LEGACY_ROUTES
            if name not in ("solar_longitude_event_before",
                            "solar_longitude_event_after")
        )
        self.assertNotIn("reason_detail", legacy_sources)

    def test_legacy_health_is_unchanged(self):
        response = server.health()
        self.assertEqual(set(response), {"status", "defaultKernel"})
        self.assertEqual(response["status"], "ok")


if __name__ == "__main__":
    unittest.main()
