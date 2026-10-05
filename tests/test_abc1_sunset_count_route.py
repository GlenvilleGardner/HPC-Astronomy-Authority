"""ABC-1 Operation 2 verification for the bulk sunset count route.

Covers GET /sunset-count in server.py and project_sunset_count in
astronomical_event_transport.py: the request contract, exact transport of
both bounds, observer handling, the projected response schema, exact-state
preservation, completeness and truncation projection, boundary coincidence,
the governed resource refusal, the failure taxonomy, single-invocation
discipline, structural isolation of the route, and preservation of every
previously published route.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d, A3c-1, A3c-2,
A3c-3 and A3c-4. httpx is not installed, so starlette's TestClient is
unavailable, and installing a dependency in order to test is not permitted.
Direct invocation exercises the real route body, including the HTTPException
mapping, which is asserted through the exception object itself.

WHAT THIS FILE DOES NOT RE-CERTIFY

The astronomy. How many sunsets an interval contains, the half-open interval
contract, the governed span bound, frontier admission, truncation semantics
and binary64 comparison are all certified in tests/test_abc1_sunset_count.py
against the published solver. This file certifies that the HTTP and transport
layers carry that result faithfully and add nothing to it.

INDEPENDENT ORACLE DISCIPLINE

Stable reason codes, the route path, the expected key sets, the pinned
artifact filenames and the observer inventory are declared here as test-local
literals. They are deliberately NOT imported from the modules under test:
importing a constant to check that same constant would make the test agree
with a defective value instead of detecting it.
"""

import inspect
import math
import re
import struct
import unittest
from unittest import mock

from fastapi import HTTPException

import astronomy_solver
import server
from astronomical_event_transport import project_sunset_count
from astronomy_solver import (
    count_sunsets_in_interval,
    kernel_coverage_tt,
    ts,
)
from exact_time_transport import decode_tt_bits, encode_tt_bits


# --- Test-local literals. Never imported from the modules under test. ------

COUNT_PATH = "/sunset-count"

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"
REASON_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_INTERVAL_TOO_LONG = "COUNT_INTERVAL_TOO_LONG"

PRIMARY_ARTIFACT = "de440.bsp"
PINNED_ARTIFACTS = ("de440.bsp", "de441_part-1.bsp", "de441_part-2.bsp")

EXPECTED_TOP_LEVEL_KEYS = (
    "sunsetCount",
    "requested",
    "covered",
    "complete",
    "truncationReason",
    "boundaryCoincident",
    "kernel",
)
EXPECTED_BOUND_PAIR_KEYS = ("lo", "hi")
EXPECTED_INSTANT_KEYS = ("ttBits", "tt", "utc")

NEW_YORK = {"latitude": 40.7128, "longitude": -74.0060}
SYDNEY = {"latitude": -33.8688, "longitude": 151.2093}

MAX_SPAN_DAYS = 400.0


def tt_of(year, month, day):
    return float(ts.utc(year, month, day).tt)


def bits_of(tt):
    """IEEE-754 spelling, built with struct directly rather than through the
    module under test, so a defective encoder cannot agree with itself."""
    return struct.pack(">d", tt).hex().upper()


def count_response(lo_tt, hi_tt, observer=None):
    observer = observer or NEW_YORK
    return server.sunset_count(
        ttLoBits=bits_of(lo_tt),
        ttHiBits=bits_of(hi_tt),
        latitude=observer["latitude"],
        longitude=observer["longitude"],
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


def outermost_declared_end():
    return max(
        kernel_coverage_tt(name).tt_end
        for name in astronomy_solver.PINNED_KERNEL_PRECEDENCE
    )


class CountRouteAssertions(unittest.TestCase):
    """Shared schema assertions for every projected count."""

    def assert_schema(self, body):
        self.assertIsInstance(body, dict)
        self.assertEqual(tuple(body), EXPECTED_TOP_LEVEL_KEYS)

        self.assertIsInstance(body["sunsetCount"], int)
        self.assertGreaterEqual(body["sunsetCount"], 0)
        self.assertIsInstance(body["complete"], bool)
        self.assertIsInstance(body["boundaryCoincident"], bool)
        self.assertIn(body["kernel"], PINNED_ARTIFACTS)

        for pair_name in ("requested", "covered"):
            pair = body[pair_name]

            self.assertIsInstance(pair, dict)
            self.assertEqual(tuple(pair), EXPECTED_BOUND_PAIR_KEYS)

            for bound in EXPECTED_BOUND_PAIR_KEYS:
                instant = pair[bound]

                self.assertEqual(tuple(instant), EXPECTED_INSTANT_KEYS)
                self.assertIsInstance(instant["ttBits"], str)
                self.assertIsInstance(instant["tt"], float)
                self.assertTrue(
                    instant["utc"] is None
                    or isinstance(instant["utc"], str)
                )

        # Completeness and its reason are one statement, never two.
        if body["complete"]:
            self.assertIsNone(body["truncationReason"])
        else:
            self.assertIsInstance(body["truncationReason"], str)


# --- 1. The request contract -----------------------------------------------


class TestRequestContract(CountRouteAssertions):
    def test_the_route_is_registered_with_the_governed_signature(self):
        routes = registered_routes()

        self.assertIn(COUNT_PATH, routes)

        methods, name, parameters = routes[COUNT_PATH]

        self.assertIn("GET", methods)
        self.assertEqual(name, "sunset_count")
        self.assertEqual(
            parameters, ["ttLoBits", "ttHiBits", "latitude", "longitude"]
        )

    def test_there_is_no_decimal_tt_or_civil_date_parameter(self):
        _, _, parameters = registered_routes()[COUNT_PATH]

        for forbidden in (
            "tt", "ttLo", "ttHi", "utc", "date", "afterUTC", "year",
            "cursor", "kernel",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_valid_bits_decode_through_the_established_mechanism(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 4, 21)
        body = count_response(lo, hi)

        self.assert_schema(body)

        # The bounds that came back are the exact states the bits denoted,
        # decoded by the published transport rather than re-parsed here.
        self.assertEqual(decode_tt_bits(body["requested"]["lo"]["ttBits"]), lo)
        self.assertEqual(decode_tt_bits(body["requested"]["hi"]["ttBits"]), hi)

    def test_a_complete_year_projects_count_and_provenance(self):
        body = count_response(tt_of(2027, 3, 21), tt_of(2028, 3, 21))

        self.assert_schema(body)
        self.assertTrue(body["complete"])
        self.assertIsNone(body["truncationReason"])
        self.assertEqual(body["kernel"], PRIMARY_ARTIFACT)
        self.assertGreater(body["sunsetCount"], 300)

    def test_the_route_agrees_with_the_solver_exactly(self):
        lo, hi = tt_of(2027, 5, 1), tt_of(2027, 7, 1)

        for observer in (NEW_YORK, SYDNEY):
            with self.subTest(observer=observer):
                record = count_sunsets_in_interval(
                    lo, hi, observer["latitude"], observer["longitude"]
                )
                body = count_response(lo, hi, observer)

                self.assertEqual(body["sunsetCount"], record.count)
                self.assertEqual(body["complete"], record.complete)
                self.assertEqual(body["kernel"], record.kernel)
                self.assertEqual(
                    body["boundaryCoincident"], record.boundary_coincident
                )

    def test_repeated_requests_are_deterministic(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 5, 21)

        self.assertEqual(count_response(lo, hi), count_response(lo, hi))


# --- 2. The failure taxonomy -----------------------------------------------


class TestFailureProjection(unittest.TestCase):
    def assert_refused(self, reason, **kwargs):
        with self.assertRaises(HTTPException) as caught:
            server.sunset_count(**kwargs)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(tuple(caught.exception.detail), ("reason", "message"))
        self.assertEqual(caught.exception.detail["reason"], reason)
        self.assertTrue(caught.exception.detail["message"])

        return caught.exception

    def base(self, **overrides):
        request = {
            "ttLoBits": bits_of(tt_of(2027, 3, 21)),
            "ttHiBits": bits_of(tt_of(2027, 4, 21)),
            "latitude": NEW_YORK["latitude"],
            "longitude": NEW_YORK["longitude"],
        }
        request.update(overrides)
        return request

    def test_malformed_bit_identities_fail_closed(self):
        for spelling in (
            "",
            "zzzz",
            "4142C796C01A3D1",          # short
            "4142C796C01A3D1AA",        # long
            "4142c796c01a3d1a",         # lowercase
            "0x4142C796C01A3D1A",       # prefixed
            " 4142C796C01A3D1A",        # padded
            "4142C796C01A3D1G",         # non-hex
        ):
            with self.subTest(spelling=spelling):
                self.assert_refused(
                    REASON_TT_BITS_INVALID,
                    **self.base(ttLoBits=spelling),
                )
                self.assert_refused(
                    REASON_TT_BITS_INVALID,
                    **self.base(ttHiBits=spelling),
                )

    def test_non_finite_bit_identities_fail_closed(self):
        for value in (math.inf, -math.inf, math.nan):
            spelling = bits_of(value)

            with self.subTest(value=value):
                self.assert_refused(
                    REASON_TT_BITS_INVALID,
                    **self.base(ttLoBits=spelling),
                )
                self.assert_refused(
                    REASON_TT_BITS_INVALID,
                    **self.base(ttHiBits=spelling),
                )

    def test_a_non_ascending_interval_fails_closed(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 4, 21)

        # Zero width.
        self.assert_refused(
            REASON_STATE_INVALID,
            **self.base(ttLoBits=bits_of(lo), ttHiBits=bits_of(lo)),
        )
        # Reversed.
        self.assert_refused(
            REASON_STATE_INVALID,
            **self.base(ttLoBits=bits_of(hi), ttHiBits=bits_of(lo)),
        )

    def test_the_governed_span_bound_projects_its_own_reason(self):
        lo = tt_of(2027, 3, 21)

        self.assert_refused(
            REASON_INTERVAL_TOO_LONG,
            **self.base(
                ttLoBits=bits_of(lo),
                ttHiBits=bits_of(lo + MAX_SPAN_DAYS + 0.1),
            ),
        )

    def test_exactly_the_governed_span_is_admitted(self):
        lo = tt_of(2027, 3, 21)
        body = server.sunset_count(**self.base(
            ttLoBits=bits_of(lo), ttHiBits=bits_of(lo + MAX_SPAN_DAYS)
        ))

        self.assertTrue(body["complete"])

    def test_an_out_of_domain_observer_projects_the_substrate_reason(self):
        for latitude, longitude in (
            (90.5, 0.0), (-90.5, 0.0), (0.0, 180.5), (0.0, -180.5),
        ):
            with self.subTest(observer=(latitude, longitude)):
                self.assert_refused(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    **self.base(latitude=latitude, longitude=longitude),
                )

    def test_a_lower_bound_outside_all_coverage_projects_its_reason(self):
        outer_start = min(
            kernel_coverage_tt(name).tt_start
            for name in astronomy_solver.PINNED_KERNEL_PRECEDENCE
        )

        self.assert_refused(
            REASON_COVERAGE_EXHAUSTED,
            **self.base(
                ttLoBits=bits_of(outer_start - 30.0),
                ttHiBits=bits_of(outer_start + 30.0),
            ),
        )

    def test_transport_and_substrate_failures_stay_apart(self):
        """A malformed bound never reports a scientific reason, and a
        governed refusal never reports a transport one."""
        transport = self.assert_refused(
            REASON_TT_BITS_INVALID, **self.base(ttLoBits="nope")
        )
        substrate = self.assert_refused(
            REASON_OBSERVER_OUT_OF_DOMAIN, **self.base(latitude=91.0)
        )

        self.assertNotEqual(
            transport.detail["reason"], substrate.detail["reason"]
        )

    def test_the_original_exception_is_preserved_as_cause(self):
        with self.assertRaises(HTTPException) as caught:
            server.sunset_count(**self.base(ttLoBits="nope"))

        self.assertIsNotNone(caught.exception.__cause__)

    def test_only_governed_failures_are_translated(self):
        """An unexpected failure must stay visible, not become a 400."""
        boom = RuntimeError("synthetic: unexpected")

        with mock.patch.object(
            server, "count_sunsets_in_interval", side_effect=boom
        ):
            with self.assertRaises(RuntimeError):
                server.sunset_count(**self.base())


# --- 3. Completeness, truncation and zero ----------------------------------


class TestCompletenessProjection(CountRouteAssertions):
    def test_a_complete_zero_count_is_a_successful_answer(self):
        """Polar night. Zero is measured, not absent - so it is 200, not 404."""
        body = count_response(
            tt_of(2027, 12, 5), tt_of(2027, 12, 20),
            {"latitude": 80.0, "longitude": 20.0},
        )

        self.assert_schema(body)
        self.assertEqual(body["sunsetCount"], 0)
        self.assertTrue(body["complete"])
        self.assertIsNone(body["truncationReason"])

    def test_an_incomplete_frontier_stays_explicitly_incomplete(self):
        outer_end = outermost_declared_end()
        body = count_response(outer_end - 30.0, outer_end + 30.0)

        self.assert_schema(body)
        self.assertFalse(body["complete"])
        self.assertEqual(body["truncationReason"], REASON_COVERAGE_EXHAUSTED)
        self.assertGreater(body["sunsetCount"], 0)

    def test_an_incomplete_result_publishes_the_actual_covered_interval(self):
        outer_end = outermost_declared_end()
        lo = outer_end - 30.0
        hi = outer_end + 30.0
        body = count_response(lo, hi)

        # The requested pair is echoed unchanged...
        self.assertEqual(decode_tt_bits(body["requested"]["lo"]["ttBits"]), lo)
        self.assertEqual(decode_tt_bits(body["requested"]["hi"]["ttBits"]), hi)

        # ...and the covered pair says what was really examined: the lower
        # bound never moves, and the upper stops at real coverage.
        self.assertEqual(decode_tt_bits(body["covered"]["lo"]["ttBits"]), lo)
        self.assertEqual(
            decode_tt_bits(body["covered"]["hi"]["ttBits"]), outer_end
        )
        self.assertLess(
            decode_tt_bits(body["covered"]["hi"]["ttBits"]),
            decode_tt_bits(body["requested"]["hi"]["ttBits"]),
        )

    def test_zero_complete_and_partial_incomplete_are_distinguishable(self):
        empty = count_response(
            tt_of(2027, 12, 5), tt_of(2027, 12, 20),
            {"latitude": 80.0, "longitude": 20.0},
        )
        outer_end = outermost_declared_end()
        partial = count_response(outer_end - 30.0, outer_end + 30.0)

        self.assertTrue(empty["complete"])
        self.assertFalse(partial["complete"])
        self.assertIsNone(empty["truncationReason"])
        self.assertIsNotNone(partial["truncationReason"])

    def test_the_route_never_promotes_a_partial_result(self):
        """complete is carried, never decided, by the HTTP layer."""
        source = inspect.getsource(server.sunset_count)

        for fabricated in ("complete", "truncation", "True", "False"):
            self.assertNotIn(fabricated, source.split('"""')[-1])


# --- 4. Boundary coincidence -----------------------------------------------


class TestBoundaryCoincidenceProjection(CountRouteAssertions):
    """Controlled transition evidence.

    A naturally occurring sunset exactly equal to a requested bound cannot be
    relied upon: the scan re-solves its roots under its own bracket and the
    Authority does not guarantee those bits. The transitions are supplied
    directly so the projected flag is exercised at exactly the intended
    states and nothing else can explain the outcome.
    """

    LO = None
    HI = None

    def setUp(self):
        self.LO = tt_of(2027, 3, 21)
        self.HI = tt_of(2027, 3, 25)

    def _staged(self, sunset_tt):
        return mock.Mock(return_value=(ts.tt_jd([sunset_tt]), [False]))

    def _staged_structure(self, sunset_tt):
        # DT-A1: a reported sunset is accepted only inside a certified setting
        # bracket, so the certified structure is staged consistently with the
        # staged transition and the projected flag stays the thing under test.
        setting = ()
        if self.LO < sunset_tt <= self.HI:
            setting = ((sunset_tt - 0.01, sunset_tt),)

        return mock.Mock(
            return_value=astronomy_solver._SunsetStructure(setting, (), ())
        )

    def _body_with(self, sunset_tt):
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", self._staged(sunset_tt)
        ), mock.patch.object(
            astronomy_solver, "_certified_sunset_structure",
            self._staged_structure(sunset_tt),
        ):
            return count_response(self.LO, self.HI)

    def test_false_projects_exactly_for_an_ordinary_interval(self):
        body = count_response(tt_of(2027, 3, 21), tt_of(2027, 4, 21))

        self.assert_schema(body)
        self.assertIs(body["boundaryCoincident"], False)

    def test_true_projects_exactly_when_a_sunset_is_on_the_bound(self):
        body = self._body_with(self.HI)

        self.assert_schema(body)
        self.assertIs(body["boundaryCoincident"], True)
        self.assertEqual(body["sunsetCount"], 1)

    def test_one_ulp_below_the_bound_is_not_coincidence(self):
        body = self._body_with(math.nextafter(self.HI, -math.inf))

        self.assertIs(body["boundaryCoincident"], False)
        self.assertEqual(body["sunsetCount"], 1)

    def test_one_ulp_above_the_bound_is_excluded_entirely(self):
        body = self._body_with(math.nextafter(self.HI, math.inf))

        self.assertIs(body["boundaryCoincident"], False)
        self.assertEqual(body["sunsetCount"], 0)

    def test_coincidence_is_not_an_error_and_not_a_calendar_decision(self):
        body = self._body_with(self.HI)

        self.assertTrue(body["complete"])
        self.assertIsNone(body["truncationReason"])

        # The Authority states the fact and assigns no ownership.
        self.assertEqual(tuple(body), EXPECTED_TOP_LEVEL_KEYS)


# --- 5. Exact state survives projection ------------------------------------


class TestExactStatePreservation(CountRouteAssertions):
    def test_every_bound_round_trips_bit_for_bit(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 6, 21)
        body = count_response(lo, hi)

        for pair_name, expected in (("requested", (lo, hi)),):
            for bound, value in zip(EXPECTED_BOUND_PAIR_KEYS, expected):
                instant = body[pair_name][bound]

                # The spelling denotes the exact state the caller supplied.
                self.assertEqual(instant["ttBits"], bits_of(value))
                # tt is produced FROM those bits, never alongside them.
                self.assertEqual(
                    instant["tt"], decode_tt_bits(instant["ttBits"])
                )
                self.assertEqual(
                    struct.pack(">d", instant["tt"]).hex().upper(),
                    instant["ttBits"],
                )

    def test_no_bound_is_reconstructed_from_its_utc_rendering(self):
        """A civil rendering is reference only and is never round-tripped."""
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 6, 21)
        body = count_response(lo, hi)

        for pair_name in ("requested", "covered"):
            for bound in EXPECTED_BOUND_PAIR_KEYS:
                instant = body[pair_name][bound]
                rendering = instant["utc"]

                if rendering is None:
                    continue

                # Microsecond text cannot carry a binary64 TT. If the value
                # had been rebuilt from this string it would have lost bits.
                self.assertRegex(rendering, r"\.\d{6}\+00:00$")

        # Decisive: the route never mentions a rendering at all.
        source = inspect.getsource(server.sunset_count)
        for token in ("utc", "datetime", "isoformat", "strftime"):
            self.assertNotIn(token, source.split('"""')[-1])

    def test_deep_time_bounds_project_with_a_null_rendering(self):
        """Beyond the calendar's reach the key is present and null."""
        outer_end = outermost_declared_end()
        body = count_response(outer_end - 30.0, outer_end - 1.0)

        for pair_name in ("requested", "covered"):
            for bound in EXPECTED_BOUND_PAIR_KEYS:
                instant = body[pair_name][bound]

                self.assertIn("utc", instant)
                self.assertIsNone(instant["utc"])
                self.assertIsInstance(instant["ttBits"], str)

    def test_kernel_identity_survives_projection(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 4, 21)
        record = count_sunsets_in_interval(lo, hi, **NEW_YORK)
        body = count_response(lo, hi)

        self.assertEqual(body["kernel"], record.kernel)
        self.assertIn(body["kernel"], PINNED_ARTIFACTS)


# --- 6. Nothing beyond a count leaks ---------------------------------------


class TestNoIdentityOrCalendarLeak(CountRouteAssertions):
    @staticmethod
    def walk(body, path=()):
        """Yield every (path, key, value) in the projected body."""
        if isinstance(body, dict):
            for key, value in body.items():
                yield path, key, value
                yield from TestNoIdentityOrCalendarLeak.walk(
                    value, path + (key,)
                )

    def test_no_sunset_event_or_root_identity_appears(self):
        body = count_response(tt_of(2027, 3, 21), tt_of(2027, 4, 21))

        # No array or sequence of any kind can carry a set of crossings.
        for path, key, value in self.walk(body):
            self.assertNotIsInstance(
                value, (list, tuple, set),
                "sequence at %s.%s could carry event identities"
                % (".".join(path), key),
            )

        # The only exact instants published are the four BOUNDS, which are
        # the caller's own states and the frontier, never a determined sunset.
        instants = [
            path for path, key, _ in self.walk(body) if key == "ttBits"
        ]
        self.assertEqual(len(instants), 4)
        self.assertEqual(
            sorted(instants),
            [("covered", "hi"), ("covered", "lo"),
             ("requested", "hi"), ("requested", "lo")],
        )

        for _, key, _ in self.walk(body):
            for forbidden in (
                "sunset", "sunsets", "event", "events", "crossing",
                "crossings", "root", "roots", "times", "boundaries",
            ):
                self.assertNotEqual(key, forbidden)

    def test_no_calendar_or_hpc_semantics_appear(self):
        body = count_response(tt_of(2027, 3, 21), tt_of(2027, 4, 21))

        for _, key, _ in self.walk(body):
            lowered = key.lower()
            for forbidden in (
                "weekday", "sabbath", "abib", "month", "day", "year",
                "hpc", "sce", "feast", "yeartype", "ordinal", "season",
            ):
                self.assertNotIn(forbidden, lowered)

    def test_the_count_is_never_normalized(self):
        """Any legal integer survives, including values a 365/366
        classification would have rewritten."""
        cases = (
            (tt_of(2027, 3, 21), tt_of(2027, 4, 21)),      # about 31
            (tt_of(2027, 3, 21), tt_of(2028, 3, 21)),      # about 366
            (tt_of(2027, 12, 5), tt_of(2027, 12, 20)),     # polar zero
        )

        observed = set()

        for lo, hi in cases:
            observer = (
                {"latitude": 80.0, "longitude": 20.0}
                if lo == tt_of(2027, 12, 5) else NEW_YORK
            )
            record = count_sunsets_in_interval(
                lo, hi, observer["latitude"], observer["longitude"]
            )
            body = count_response(lo, hi, observer)

            self.assertEqual(body["sunsetCount"], record.count)
            observed.add(body["sunsetCount"])

        # Non-vacuity: genuinely different integers passed through, so the
        # agreement above is not an accident of one repeated value.
        self.assertGreaterEqual(len(observed), 3)
        self.assertIn(0, observed)


# --- 7. Single invocation, no per-sunset loop ------------------------------


class TestSingleInvocation(unittest.TestCase):
    def test_one_request_invokes_the_bulk_solver_exactly_once(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2028, 3, 21)
        recorder = mock.Mock(wraps=count_sunsets_in_interval)

        with mock.patch.object(
            server, "count_sunsets_in_interval", recorder
        ):
            body = count_response(lo, hi)

        self.assertEqual(recorder.call_count, 1)
        self.assertGreater(body["sunsetCount"], 300)

    def test_one_request_performs_exactly_one_search(self):
        """A year of sunsets, one find_discrete. No per-sunset walk."""
        lo, hi = tt_of(2027, 3, 21), tt_of(2028, 3, 21)
        real = astronomy_solver.almanac.find_discrete
        recorder = mock.Mock(wraps=real)

        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", recorder
        ):
            body = count_response(lo, hi)

        self.assertEqual(recorder.call_count, 1)
        self.assertGreater(body["sunsetCount"], 300)

    def test_one_request_admits_exactly_one_frontier_and_artifact(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2028, 3, 21)
        recorder = mock.Mock(wraps=astronomy_solver.supported_search_frontier)

        with mock.patch.object(
            astronomy_solver, "supported_search_frontier", recorder
        ):
            body = count_response(lo, hi)

        self.assertEqual(recorder.call_count, 1)
        self.assertIsInstance(body["kernel"], str)

    def test_the_route_body_contains_no_loop(self):
        source = inspect.getsource(server.sunset_count)
        executable = source.split('"""')[-1]

        for token in ("for ", "while ", "sum(", "len(", "range("):
            self.assertNotIn(token, executable)


# --- 8. Structural controls ------------------------------------------------


class TestRouteSourceGuard(unittest.TestCase):
    BLOCK_MARKER = "# A4 - ABC-1 bulk sunset count route."
    GOVERNED_BLOCK_MARKER = re.compile(
        r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ", re.MULTILINE
    )

    @property
    def server_source(self):
        return inspect.getsource(server)

    def test_the_marker_obeys_the_governed_grammar(self):
        source = self.server_source

        markers = self.GOVERNED_BLOCK_MARKER.findall(source)
        self.assertGreater(len(markers), 3)

        own = source.find(self.BLOCK_MARKER)
        self.assertNotEqual(own, -1, "A4 route block marker not found")
        self.assertIsNotNone(self.GOVERNED_BLOCK_MARKER.match(source, own))

        # The block owns its own symbols: nothing governed intervenes between
        # this marker and the route it introduces. A LATER governed block may
        # follow - it must simply begin after that route, not inside it.
        own_function = source.find("def sunset_count(")
        self.assertGreater(own_function, own)

        following = self.GOVERNED_BLOCK_MARKER.search(source, own + 1)
        if following is not None:
            self.assertGreater(following.start(), own_function)

    def test_the_preceding_governed_block_is_still_delimited(self):
        """A3c-4 must still be a block, now terminated by this one."""
        source = self.server_source
        previous = source.find("# A3c-4 - exact sunset successor route.")

        self.assertNotEqual(previous, -1)
        self.assertLess(previous, source.find(self.BLOCK_MARKER))

        following = self.GOVERNED_BLOCK_MARKER.search(source, previous + 1)

        self.assertIsNotNone(following)
        self.assertEqual(following.start(), source.find(self.BLOCK_MARKER))

    def test_the_route_performs_no_astronomy(self):
        executable = inspect.getsource(server.sunset_count).split('"""')[-1]

        for token in (
            "almanac", "find_discrete", "sunrise_sunset", "wgs84",
            "load_kernel", "kernel_coverage", "supported_search_frontier",
            "step_days", "ts.tt_jd", "de440", "de441",
        ):
            self.assertNotIn(token, executable)

    def test_the_route_applies_no_resource_or_domain_policy_of_its_own(self):
        executable = inspect.getsource(server.sunset_count).split('"""')[-1]

        for token in ("400.0", "COUNT_MAX_SPAN_DAYS", "90.0", "180.0", "abs("):
            self.assertNotIn(token, executable)

    def test_the_route_calls_the_solver_and_the_projection_once_each(self):
        executable = inspect.getsource(server.sunset_count).split('"""')[-1]

        self.assertEqual(executable.count("count_sunsets_in_interval"), 1)
        self.assertEqual(executable.count("project_sunset_count"), 1)
        self.assertEqual(executable.count("decode_tt_bits"), 2)

    def test_the_source_scanner_is_non_vacuous(self):
        executable = inspect.getsource(server.sunset_count).split('"""')[-1]

        self.assertGreater(len(executable), 200)
        self.assertIn("count_sunsets_in_interval", executable)
        self.assertIn("HTTPException", executable)

        # A planted violation would be seen.
        planted = "    total = sum(1 for s in sunsets)\n"
        self.assertIn("sum(", planted)

    def test_the_transport_projection_performs_no_astronomy(self):
        executable = inspect.getsource(project_sunset_count).split('"""')[-1]

        for token in (
            "almanac", "find_discrete", "load_kernel", "wgs84",
            "count_sunsets_in_interval", "round(", "//", "365", "366",
        ):
            self.assertNotIn(token, executable)

        # It really is the projection, reusing the established instant shape.
        self.assertEqual(executable.count("project_exact_instant"), 4)

    def test_the_solver_acquires_no_transport_or_http_dependency(self):
        """Dependency direction: route -> transport -> solver."""
        solver_source = inspect.getsource(astronomy_solver)
        executable = "\n".join(
            line.split("#", 1)[0] for line in solver_source.split("\n")
        )
        imports = re.findall(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
            executable,
            re.MULTILINE,
        )
        found = {name for pair in imports for name in pair if name}

        self.assertIn("scientific_environment", found)   # non-vacuity
        for upward in (
            "server", "fastapi", "astronomical_event_transport",
            "exact_time_transport",
        ):
            self.assertNotIn(upward, found)


# --- 9. Every previously published route is preserved ----------------------


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
    }

    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()

        for path, endpoint in self.PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], endpoint)

    # Application routes added by governed increments AFTER A4.
    #
    # A4's closed-world claim is about the surface AS OF A4: that this
    # increment added exactly one route. A later additive increment does not
    # weaken that claim, but it does make an unqualified count of the whole
    # application factually wrong. Subtracting the later additions keeps the
    # claim exactly as strong as it was while letting it stay true.
    POST_A4_ROUTES = frozenset({
        "/scientific-environment", "/earth-rotation", "/solar-regime",
    })

    def test_exactly_one_route_was_added(self):
        routes = registered_routes()
        published = {
            path for path in routes
            if not path.startswith("/openapi")
            and not path.startswith("/docs")
            and not path.startswith("/redoc")
        } - self.POST_A4_ROUTES

        self.assertEqual(
            published - set(self.PREVIOUS), {COUNT_PATH}
        )

    def test_an_existing_route_still_answers_unchanged(self):
        anchor = tt_of(2027, 6, 1)
        body = server.sunset_event_after(
            ttBits=bits_of(anchor), **NEW_YORK
        )

        self.assertEqual(tuple(body), ("ttBits", "tt", "utc", "kernel"))
        self.assertGreater(decode_tt_bits(body["ttBits"]), anchor)

    def test_the_health_route_still_answers(self):
        self.assertIsInstance(server.health(), dict)


if __name__ == "__main__":
    unittest.main()
