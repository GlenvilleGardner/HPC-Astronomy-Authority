"""A3c-4 verification for the exact sunset successor route.

Covers GET /sunset-event-after in server.py: the request contract, exact
transport, observer handling, strict-after semantics, exact-root
continuation, deep-time and cross-artifact operation, per-event provenance,
governed absence, the failure taxonomy, structural isolation of the route,
and preservation of every previously published route.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d, A3c-1, A3c-2 and
A3c-3. httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted. Direct invocation
exercises the real route body, including the HTTPException mapping, which is
asserted through the exception object itself.

Direct invocation also means FastAPI performs no parameter coercion, so the
substrate's own observer guard is exercised by the values a caller actually
supplies rather than by whatever a framework would have converted first. The
framework's own 422 behavior is a separate concern and is asserted to be
distinct rather than reproduced.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the expected key sets, the governed
directional reach, the route path and the observer inventory are declared
here as test-local literals. They are deliberately NOT imported from
astronomy_solver: importing a constant to check that same constant would make
the test agree with a defective value instead of detecting it.

The bit-level invariant is checked with struct.pack directly. Crossings are
enumerated here against skyfield.almanac over the real frontier rather than
through any production helper. The published projection IS called, as
certified A3c-0 infrastructure, to assert that the route returns exactly it
and hand-builds nothing.

EVIDENCE DISCIPLINE

PROVED    - follows structurally from the code under test.
MEASURED  - observed against the real pinned artifacts and certified runtime
            at test time.
TAXONOMY  - certified by injecting a governed exception, where no ordinary
            input witness exists. Labelled as such and never presented as a
            real-input observation.

WHAT IS DELIBERATELY NOT ENCODED

No epsilon, tolerance, minimum gap or light-time second-count governs event
identity anywhere in this file.

Where this file must establish that a result is the FOLLOWING physical sunset
rather than a numerical re-report of the anchor's own crossing, it does so by
CROSSING TOPOLOGY: sunrises and sunsets strictly alternate, so exactly one
sunrise inside the spanned interval proves exactly one night ended. The
production result is used only as an interval ENDPOINT. Its value is never
asserted equal to, or different from, any independently determined root.

Two independently bracketed or independently artifact-routed searches may
legitimately report one physical sunset at slightly different binary64
values. Nothing here requires them to agree - or to disagree - bit for bit.

No HPC/SCE year, epoch, weekday, month, Telma, year classification or
calendar semantics appears. A3c-4 is Astronomy Authority.
"""

import ast
import inspect
import math
import re
import struct
import unittest
from unittest import mock

from fastapi import HTTPException
from skyfield import almanac
from skyfield.api import wgs84

import server
from astronomy_solver import (
    SunsetChronologyError,
    find_sunset_from_instant,
    kernel_coverage_tt,
    load_kernel,
    supported_search_frontier,
    ts,
)
from astronomical_event_transport import project_sunset_event
from exact_time_transport import decode_tt_bits, encode_tt_bits

# --- Test-local oracles ----------------------------------------------------

EVENT_AFTER_PATH = "/sunset-event-after"

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"
PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

EXPECTED_EVENT_KEYS = ("ttBits", "tt", "utc", "kernel")
EXPECTED_QUERY_PARAMETERS = ["ttBits", "latitude", "longitude"]

REASON_TT_BITS_INVALID = "TT_BITS_INVALID"
REASON_INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_SUNSET_EVENT_ABSENT = "SUNSET_EVENT_ABSENT"

SCIENTIFIC_REASONS = (
    REASON_INSTANT_STATE_INVALID,
    REASON_OBSERVER_OUT_OF_DOMAIN,
    REASON_COVERAGE_EXHAUSTED,
    REASON_REACH_EXHAUSTED,
)

# The governed directional reach of one sunset search, in TT days.
# Re-declared rather than imported.
SPAN_DAYS = 3.0

NYC = {"latitude": 40.7406, "longitude": -73.9}

OBSERVERS = (
    (40.7406, -73.9, "new-york"),
    (0.0, 0.0, "null-island"),
    (-33.9, 151.2, "sydney"),
    (35.7, 139.7, "tokyo"),
    (64.1, -21.9, "reykjavik"),
    (-54.8, -68.3, "ushuaia"),
    (0.0, 180.0, "antimeridian"),
)

# MEASURED: instants inside a polar day or polar night, where a complete
# supported frontier genuinely contains no sunset.
POLAR_CASES = (
    (78.2, 15.6, 2460850.0, "svalbard polar day"),
    (78.2, 15.6, 2460680.0, "svalbard polar night"),
    (-80.0, 0.0, 2460680.0, "antarctic polar day"),
)

MODERN_ANCHOR = 2460678.0
DEEP_TIME_ANCHOR = 700000.0

# MEASURED: must resolve beyond the span a calendar datetime can express, so
# that the nullable-UTC certification is actually exercised. The certified
# timescale places 10000-01-01 at TT 5373484.500800741, so an anchor below
# that resolves to a REPRESENTABLE year and would yield a utc string rather
# than null. This value resolves to calendar year 11715 and remains inside
# de441 part 2 coverage (TT 2440400.5 .. 8000016.5) with the whole governed
# forward frontier to spare.
FAR_FUTURE_ANCHOR = 6000000.0
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

A3C4_BLOCK_MARKER = "# A3c-4 - exact sunset successor route."

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)


# --- Independent helpers ---------------------------------------------------


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def event_response(ttBits, latitude=NYC["latitude"],
                   longitude=NYC["longitude"]):
    return server.sunset_event_after(
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


def predicate(kernel_name, latitude, longitude):
    return almanac.sunrise_sunset(
        load_kernel(kernel_name), wgs84.latlon(latitude, longitude)
    )


def oracle_transitions(kernel_name, tt_lo, tt_hi, latitude, longitude):
    """Independent enumeration of every sunrise/sunset transition.

    Returns (tt, sun_is_up) pairs in ascending time. Used to examine the
    CROSSING TOPOLOGY of an interval rather than the value of any root.
    """
    times, events = almanac.find_discrete(
        ts.tt_jd(tt_lo), ts.tt_jd(tt_hi),
        predicate(kernel_name, latitude, longitude),
    )
    return [
        (float(t.tt), bool(sun_is_up))
        for t, sun_is_up in zip(times, events)
    ]


def oracle_sunsets(kernel_name, tt_lo, tt_hi, latitude, longitude):
    return [tt for tt, up in oracle_transitions(
        kernel_name, tt_lo, tt_hi, latitude, longitude) if not up]


def forward_frontier(anchor, latitude, longitude):
    """The frontier the successor call is bound to."""
    return supported_search_frontier(
        anchor, anchor + SPAN_DAYS, latitude, longitude
    )


def artifact_transition_anchor():
    """An anchor whose successor search must move to a later artifact.

    MEASURED: located just inside DE440's declared ceiling, where the
    forward three-day frontier exceeds it.
    """
    ceiling = kernel_coverage_tt(DE440).tt_end
    for offset in (2.5, 2.0, 1.5, 1.0, 0.5):
        candidate = ceiling - offset
        event = find_sunset_from_instant(
            candidate, NYC["latitude"], NYC["longitude"]
        )
        if event is not None and event.kernel != DE440:
            return candidate, event
    return None, None


class EventAssertions(unittest.TestCase):
    def assert_rejects(self, ttBits, reason, status=400, **observer):
        parameters = dict(NYC)
        parameters.update(observer)
        with self.assertRaises(HTTPException) as caught:
            event_response(ttBits, **parameters)
        error = caught.exception
        self.assertEqual(error.status_code, status)
        self.assertIsInstance(error.detail, dict)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"], reason)
        self.assertIsInstance(error.detail["message"], str)
        return error

    def assert_is_event(self, response):
        self.assertIsInstance(response, dict)
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)
        self.assertIn(response["kernel"], PINNED_ARTIFACTS)
        self.assertIsInstance(response["tt"], float)
        self.assertTrue(math.isfinite(response["tt"]))
        self.assertIsInstance(response["ttBits"], str)
        # ttBits is the identity; tt is produced from those same bits.
        self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                         bits(response["tt"]))
        self.assertTrue(response["utc"] is None
                        or isinstance(response["utc"], str))
        return response


# --- 1. Route surface and request contract ---------------------------------


class TestRouteSurface(EventAssertions):
    def test_path_is_registered(self):
        self.assertIn(EVENT_AFTER_PATH, registered_routes())

    def test_route_is_get_only(self):
        methods = registered_routes()[EVENT_AFTER_PATH][0]
        self.assertEqual(methods - {"HEAD"}, {"GET"})

    def test_endpoint_name(self):
        self.assertEqual(registered_routes()[EVENT_AFTER_PATH][1],
                         "sunset_event_after")

    def test_parameters_are_exactly_ttbits_latitude_longitude(self):
        self.assertEqual(registered_routes()[EVENT_AFTER_PATH][2],
                         EXPECTED_QUERY_PARAMETERS)

    def test_no_civil_kernel_cursor_or_direction_parameter(self):
        forbidden = {
            "tt", "utc", "date", "afterUTC", "instant", "year", "month",
            "day", "hour", "timezone", "tz", "offset", "era", "bceYear",
            "kernel", "ephemeris", "artifact", "direction", "cursor",
            "witness", "retry", "attempts", "hemisphere", "kind", "season",
        }
        self.assertEqual(
            set(registered_routes()[EVENT_AFTER_PATH][2]) & forbidden, set()
        )

    def test_path_is_distinct_from_the_legacy_sunset_after_route(self):
        """The exact route must not be confusable with the legacy one."""
        found = registered_routes()
        self.assertIn("/sunset-after", found)
        self.assertNotEqual(EVENT_AFTER_PATH, "/sunset-after")
        self.assertEqual(found["/sunset-after"][2],
                         ["afterUTC", "latitude", "longitude"])


# --- 2. Success projection and exact transport -----------------------------


class TestSuccessProjection(EventAssertions):
    def test_response_is_exactly_the_published_projection(self):
        expected = project_sunset_event(
            find_sunset_from_instant(
                MODERN_ANCHOR, NYC["latitude"], NYC["longitude"]
            )
        )
        self.assertEqual(event_response(encode_tt_bits(MODERN_ANCHOR)),
                         expected)

    def test_response_shape_and_bit_consistency(self):
        self.assert_is_event(event_response(encode_tt_bits(MODERN_ANCHOR)))

    def test_ttbits_round_trips_bit_identically(self):
        response = event_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                         bits(response["tt"]))

    def test_anchor_is_not_returned(self):
        """A single directional result makes no claim needing the anchor."""
        response = event_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)
        self.assertNotIn("anchor", response)
        self.assertNotEqual(bits(response["tt"]), bits(MODERN_ANCHOR))

    def test_no_cursor_is_returned(self):
        response = event_response(encode_tt_bits(MODERN_ANCHOR))
        for key in ("cursor", "cursorUnavailableReason", "witness"):
            with self.subTest(key=key):
                self.assertNotIn(key, response)

    def test_repeated_calls_are_deterministic(self):
        anchor = encode_tt_bits(MODERN_ANCHOR)
        self.assertEqual(event_response(anchor), event_response(anchor))

    def test_the_substrate_receives_the_exact_state_and_observer_once(self):
        """PROVED by observation of the delegated call."""
        captured = []
        real = server.find_sunset_from_instant

        def recorder(tt, latitude, longitude):
            captured.append((tt, latitude, longitude))
            return real(tt, latitude, longitude)

        with mock.patch.object(
            server, "find_sunset_from_instant", recorder
        ):
            event_response(encode_tt_bits(MODERN_ANCHOR))

        self.assertEqual(len(captured), 1)
        self.assertEqual(bits(captured[0][0]), bits(MODERN_ANCHOR))
        self.assertEqual(captured[0][1:], (NYC["latitude"], NYC["longitude"]))

    def test_observer_is_passed_through_unaltered(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                captured = []
                real = server.find_sunset_from_instant

                def recorder(tt, lat, lon):
                    captured.append((lat, lon))
                    return real(tt, lat, lon)

                with mock.patch.object(
                    server, "find_sunset_from_instant", recorder
                ):
                    event_response(encode_tt_bits(MODERN_ANCHOR),
                                   latitude=latitude, longitude=longitude)
                self.assertEqual(captured, [(latitude, longitude)])


# --- 3. Strict-after semantics ---------------------------------------------


class TestStrictAfterSemantics(EventAssertions):
    def test_result_is_strictly_later_for_the_global_observer_matrix(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                response = self.assert_is_event(event_response(
                    encode_tt_bits(MODERN_ANCHOR),
                    latitude=latitude, longitude=longitude,
                ))
                self.assertGreater(response["tt"], MODERN_ANCHOR)

    def test_result_is_a_genuine_sunset_in_its_own_frontier(self):
        """Independently enumerated over the frontier the search was bound to."""
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                response = event_response(
                    encode_tt_bits(MODERN_ANCHOR),
                    latitude=latitude, longitude=longitude,
                )
                frontier = forward_frontier(
                    MODERN_ANCHOR, latitude, longitude
                )
                self.assertIn(
                    response["tt"],
                    oracle_sunsets(frontier.kernel, frontier.tt_lo,
                                   frontier.tt_hi, latitude, longitude),
                )

    def test_arbitrary_non_sunset_anchors_are_accepted(self):
        for offset in (0.123456, -0.31415, 0.5, 0.0009765625):
            anchor = MODERN_ANCHOR + offset
            with self.subTest(anchor=anchor):
                response = self.assert_is_event(
                    event_response(encode_tt_bits(anchor))
                )
                self.assertGreater(response["tt"], anchor)

    def test_no_sunset_is_skipped_between_anchor_and_result(self):
        """Crossing topology, over an arbitrary mid-day anchor.

        An anchor with the Sun up is followed by exactly one sunset and no
        sunrise: the very next transition is the result's own crossing.
        """
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                response = event_response(
                    encode_tt_bits(MODERN_ANCHOR),
                    latitude=latitude, longitude=longitude,
                )
                spanned = oracle_transitions(
                    response["kernel"], MODERN_ANCHOR, response["tt"],
                    latitude, longitude,
                )
                sunsets = [tt for tt, up in spanned if not up]
                self.assertLessEqual(
                    len(sunsets), 1,
                    "only the result's own crossing may lie in the span",
                )


# --- 4. Exact-root anchor continuation -------------------------------------


class TestExactRootContinuation(EventAssertions):
    """The result of an exact-root anchor is the FOLLOWING physical sunset.

    Established by CROSSING TOPOLOGY, never by comparing root values.
    Sunrises and sunsets strictly alternate, so exactly one sunrise inside
    the spanned interval proves exactly one night ended: the result is
    neither a re-report of the anchor's own crossing, which would leave the
    interval inside a single night, nor a sunset reached by skipping one,
    which would end two nights or more.

    The production result appears only as an interval ENDPOINT.

    Asserting that the anchor is ABSENT from an independently enumerated
    crossing set is an absence/membership claim about that one enumeration.
    It is NOT a requirement that two independently solved representations of
    the same event compare bit-identically, and it must never be rewritten
    as a cross-solver assertNotEqual.
    """

    def determined_roots(self, latitude, longitude, count=2):
        roots = []
        anchor = MODERN_ANCHOR
        for _ in range(count):
            event = find_sunset_from_instant(anchor, latitude, longitude)
            if event is None:
                return roots
            roots.append(event)
            anchor = event.tt
        return roots

    def test_every_returned_root_is_post_transition(self):
        """MEASURED, under the event's OWN reported kernel.

        The structural reason a forward search beginning at a determined
        root cannot re-detect that root's crossing: the predicate presents
        no sign change there.
        """
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                response = event_response(
                    encode_tt_bits(MODERN_ANCHOR),
                    latitude=latitude, longitude=longitude,
                )
                sun_is_up = predicate(
                    response["kernel"], latitude, longitude
                )
                self.assertFalse(bool(sun_is_up(ts.tt_jd(response["tt"]))))

    def test_exact_root_anchor_is_absent_from_its_own_frontier(self):
        for latitude, longitude, label in OBSERVERS:
            for event in self.determined_roots(latitude, longitude):
                with self.subTest(observer=label, tt=event.tt):
                    frontier = forward_frontier(
                        event.tt, latitude, longitude
                    )
                    self.assertNotIn(
                        event.tt,
                        oracle_sunsets(frontier.kernel, frontier.tt_lo,
                                       frontier.tt_hi, latitude, longitude),
                    )

    def test_exact_root_anchor_yields_the_following_physical_sunset(self):
        """Exactly one night ends between the anchor and the result."""
        for latitude, longitude, label in OBSERVERS:
            for event in self.determined_roots(latitude, longitude):
                with self.subTest(observer=label, tt=event.tt):
                    response = event_response(
                        encode_tt_bits(event.tt),
                        latitude=latitude, longitude=longitude,
                    )
                    self.assertGreater(response["tt"], event.tt)

                    spanned = oracle_transitions(
                        response["kernel"], event.tt, response["tt"],
                        latitude, longitude,
                    )
                    sunrises = [tt for tt, up in spanned if up]
                    sunsets = [tt for tt, up in spanned if not up]

                    self.assertEqual(
                        len(sunrises), 1,
                        "exactly one night must end between an exact-root "
                        "anchor and the sunset reported after it",
                    )
                    self.assertLessEqual(len(sunsets), 1)

    def test_a_walk_composes_on_returned_ttbits_alone(self):
        """Ordinal only. No cursor, no witness, no timestamp participates."""
        anchor_bits = encode_tt_bits(MODERN_ANCHOR)
        walked = []
        for _ in range(5):
            response = event_response(anchor_bits)
            walked.append(response["tt"])
            anchor_bits = response["ttBits"]
        self.assertEqual(len(walked), 5)
        self.assertEqual(walked, sorted(walked))
        self.assertEqual(len(set(walked)), 5)


# --- 5. Deep time and artifact provenance ----------------------------------


class TestDeepTimeAndProvenance(EventAssertions):
    def test_de441_part_1_operation(self):
        response = self.assert_is_event(
            event_response(encode_tt_bits(DEEP_TIME_ANCHOR))
        )
        self.assertEqual(response["kernel"], DE441_PART_1)
        self.assertGreater(response["tt"], DEEP_TIME_ANCHOR)

    def test_de440_operation(self):
        response = self.assert_is_event(
            event_response(encode_tt_bits(MODERN_ANCHOR))
        )
        self.assertEqual(response["kernel"], DE440)

    def test_de441_part_2_operation(self):
        response = self.assert_is_event(
            event_response(encode_tt_bits(FAR_FUTURE_ANCHOR))
        )
        self.assertEqual(response["kernel"], DE441_PART_2)
        self.assertGreater(response["tt"], FAR_FUTURE_ANCHOR)

    def test_all_three_artifacts_are_exercised(self):
        seen = {
            event_response(encode_tt_bits(anchor))["kernel"]
            for anchor in (DEEP_TIME_ANCHOR, MODERN_ANCHOR, FAR_FUTURE_ANCHOR)
        }
        self.assertEqual(seen, set(PINNED_ARTIFACTS))

    def test_utc_is_null_in_deep_time_while_tt_remains_exact(self):
        for anchor in (DEEP_TIME_ANCHOR, FAR_FUTURE_ANCHOR):
            with self.subTest(anchor=anchor):
                response = event_response(encode_tt_bits(anchor))
                self.assertIsNone(response["utc"])
                self.assertTrue(math.isfinite(response["tt"]))
                self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                                 bits(response["tt"]))

    def test_utc_is_present_in_the_representable_span(self):
        response = event_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertIsInstance(response["utc"], str)

    def test_the_utc_key_is_always_present(self):
        for anchor in (DEEP_TIME_ANCHOR, MODERN_ANCHOR, FAR_FUTURE_ANCHOR):
            with self.subTest(anchor=anchor):
                self.assertIn("utc", event_response(encode_tt_bits(anchor)))

    def test_provenance_matches_the_frontier_the_search_was_bound_to(self):
        for anchor in (DEEP_TIME_ANCHOR, MODERN_ANCHOR, FAR_FUTURE_ANCHOR):
            with self.subTest(anchor=anchor):
                response = event_response(encode_tt_bits(anchor))
                frontier = forward_frontier(
                    anchor, NYC["latitude"], NYC["longitude"]
                )
                self.assertEqual(response["kernel"], frontier.kernel)

    def test_reported_artifact_really_declares_the_returned_state(self):
        for anchor in (DEEP_TIME_ANCHOR, MODERN_ANCHOR, FAR_FUTURE_ANCHOR):
            with self.subTest(anchor=anchor):
                response = event_response(encode_tt_bits(anchor))
                coverage = kernel_coverage_tt(response["kernel"])
                self.assertLessEqual(coverage.tt_start, response["tt"])
                self.assertLessEqual(response["tt"], coverage.tt_end)


class TestCrossArtifactContinuation(EventAssertions):
    def test_an_anchor_may_precede_an_event_from_another_artifact(self):
        """MEASURED at the DE440 ceiling, where the frontier must move on."""
        anchor, event = artifact_transition_anchor()
        self.assertIsNotNone(anchor, "no artifact transition anchor located")
        coverage = kernel_coverage_tt(DE440)
        # The anchor itself lies inside DE440's declared coverage...
        self.assertLessEqual(coverage.tt_start, anchor)
        self.assertLessEqual(anchor, coverage.tt_end)
        # ...while the search that answers for it runs under a later one.
        response = event_response(encode_tt_bits(anchor))
        self.assertEqual(response["kernel"], DE441_PART_2)
        self.assertNotEqual(response["kernel"], DE440)
        self.assertGreater(response["tt"], anchor)

    def test_consecutive_successor_calls_may_change_artifact(self):
        """A walk is not entitled to assume a shared kernel.

        MEASURED: a walk begun below the DE440 ceiling reports DE440 while
        its forward frontier still fits, then legitimately reports DE441
        part 2 once it no longer does. Both are truthful.
        """
        anchor_bits = encode_tt_bits(kernel_coverage_tt(DE440).tt_end - 5.0)
        kernels = []
        for _ in range(5):
            response = event_response(anchor_bits)
            kernels.append(response["kernel"])
            anchor_bits = response["ttBits"]
        self.assertGreater(
            len(set(kernels)), 1,
            "this probe requires a walk that crosses an artifact boundary",
        )
        self.assertEqual(kernels[0], DE440)
        self.assertEqual(kernels[-1], DE441_PART_2)

    def test_no_shared_kernel_assumption_is_expressed(self):
        """The result carries its own provenance and nothing else's."""
        response = event_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)
        self.assertEqual(
            [k for k in response if "kernel" in k.lower()], ["kernel"]
        )

    def test_the_caller_supplies_no_artifact_and_none_is_echoed(self):
        self.assertNotIn("kernel", registered_routes()[EVENT_AFTER_PATH][2])


# --- 6. Governed absence ----------------------------------------------------


class TestGovernedAbsence(EventAssertions):
    def test_polar_cases_really_have_no_successor(self):
        """PRECONDITION, established without the route."""
        for latitude, longitude, anchor, label in POLAR_CASES:
            with self.subTest(case=label):
                self.assertIsNone(
                    find_sunset_from_instant(anchor, latitude, longitude)
                )

    def test_absence_is_reported_as_404_with_a_stable_reason(self):
        for latitude, longitude, anchor, label in POLAR_CASES:
            with self.subTest(case=label):
                self.assert_rejects(
                    encode_tt_bits(anchor), REASON_SUNSET_EVENT_ABSENT,
                    status=404, latitude=latitude, longitude=longitude,
                )

    def test_absence_is_not_reported_as_400(self):
        latitude, longitude, anchor, _label = POLAR_CASES[0]
        with self.assertRaises(HTTPException) as caught:
            event_response(encode_tt_bits(anchor),
                           latitude=latitude, longitude=longitude)
        self.assertEqual(caught.exception.status_code, 404)

    def test_absence_reason_is_not_a_scientific_or_transport_reason(self):
        self.assertNotIn(REASON_SUNSET_EVENT_ABSENT, SCIENTIFIC_REASONS)
        self.assertNotEqual(REASON_SUNSET_EVENT_ABSENT, REASON_TT_BITS_INVALID)

    def test_absence_fabricates_no_event(self):
        latitude, longitude, anchor, _label = POLAR_CASES[0]
        with self.assertRaises(HTTPException) as caught:
            event_response(encode_tt_bits(anchor),
                           latitude=latitude, longitude=longitude)
        self.assertEqual(tuple(caught.exception.detail), ("reason", "message"))
        self.assertNotIn("ttBits", caught.exception.detail)

    def test_a_non_polar_observer_at_the_same_instant_still_succeeds(self):
        _lat, _lon, anchor, _label = POLAR_CASES[0]
        self.assert_is_event(event_response(encode_tt_bits(anchor)))

    def test_absence_reason_lives_in_the_route_layer_only(self):
        import astronomy_solver
        self.assertNotIn(REASON_SUNSET_EVENT_ABSENT,
                         inspect.getsource(astronomy_solver))

    def test_scientific_failure_is_never_reported_as_absence(self):
        """TAXONOMY: injected, because no ordinary input witness exists."""
        governed = SunsetChronologyError(
            REASON_REACH_EXHAUSTED, "injected governed failure"
        )
        with mock.patch.object(
            server, "find_sunset_from_instant", side_effect=governed
        ):
            error = self.assert_rejects(
                encode_tt_bits(MODERN_ANCHOR), REASON_REACH_EXHAUSTED
            )
        self.assertEqual(error.status_code, 400)


# --- 7. Failure taxonomy ----------------------------------------------------


class TestFailureTaxonomy(EventAssertions):
    def test_every_malformed_ttbits_is_a_transport_rejection(self):
        for label, value in MALFORMED_TT_BITS:
            with self.subTest(case=label):
                self.assert_rejects(value, REASON_TT_BITS_INVALID)

    def test_transport_rejection_never_invokes_the_substrate(self):
        with mock.patch.object(
            server, "find_sunset_from_instant"
        ) as substrate:
            with self.assertRaises(HTTPException):
                event_response("not-bits")
        substrate.assert_not_called()

    def test_transport_reason_is_never_a_scientific_reason(self):
        self.assertNotIn(REASON_TT_BITS_INVALID, SCIENTIFIC_REASONS)

    def test_transport_cause_is_preserved(self):
        from exact_time_transport import ExactTimeTransportError
        with self.assertRaises(HTTPException) as caught:
            event_response("not-bits")
        self.assertIsInstance(caught.exception.__cause__,
                              ExactTimeTransportError)

    def test_observer_out_of_domain_from_real_coordinates(self):
        for latitude, longitude, label in (
            (91.0, 0.0, "latitude above domain"),
            (-91.0, 0.0, "latitude below domain"),
            (0.0, 181.0, "longitude above domain"),
            (0.0, -181.0, "longitude below domain"),
            (float("nan"), 0.0, "non-finite latitude"),
            (float("inf"), 0.0, "infinite latitude"),
            (0.0, float("nan"), "non-finite longitude"),
        ):
            with self.subTest(case=label):
                self.assert_rejects(
                    encode_tt_bits(MODERN_ANCHOR),
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    latitude=latitude, longitude=longitude,
                )

    def test_coverage_exhausted_from_real_unsupported_anchors(self):
        for anchor in (BEYOND_COVERAGE, BELOW_COVERAGE):
            with self.subTest(anchor=anchor):
                self.assert_rejects(encode_tt_bits(anchor),
                                    REASON_COVERAGE_EXHAUSTED)

    def test_every_governed_reason_crosses_unchanged(self):
        """TAXONOMY: injected for the reasons with no ordinary witness."""
        for reason in SCIENTIFIC_REASONS:
            with self.subTest(reason=reason):
                governed = SunsetChronologyError(reason, "injected")
                with mock.patch.object(
                    server, "find_sunset_from_instant", side_effect=governed
                ):
                    self.assert_rejects(
                        encode_tt_bits(MODERN_ANCHOR), reason
                    )

    def test_reason_is_a_field_not_parsed_from_prose(self):
        error = self.assert_rejects("not-bits", REASON_TT_BITS_INVALID)
        self.assertEqual(tuple(error.detail), ("reason", "message"))

    def test_unexpected_failures_propagate_unchanged(self):
        """TAXONOMY: an unexpected failure is not a governed rejection."""
        with mock.patch.object(
            server, "find_sunset_from_instant",
            side_effect=RuntimeError("not governed"),
        ):
            with self.assertRaises(RuntimeError):
                event_response(encode_tt_bits(MODERN_ANCHOR))

    def test_framework_coercion_is_distinct_from_governed_failure(self):
        """FastAPI owns request shape; Authority owns scientific domain.

        A non-numeric coordinate never reaches the substrate through the
        framework, so it is a 422 there rather than a governed 400. Invoked
        directly, the same value reaches the substrate and earns the
        governed observer reason. Both behaviours are correct and must stay
        distinguishable.
        """
        parameters = registered_routes()[EVENT_AFTER_PATH][2]
        self.assertEqual(parameters, EXPECTED_QUERY_PARAMETERS)
        annotations = inspect.signature(server.sunset_event_after).parameters
        self.assertIs(annotations["latitude"].annotation, float)
        self.assertIs(annotations["longitude"].annotation, float)
        self.assertIs(annotations["ttBits"].annotation, str)
        self.assert_rejects(
            encode_tt_bits(MODERN_ANCHOR), REASON_OBSERVER_OUT_OF_DOMAIN,
            latitude="not-a-number",
        )


# --- 8. Observer validation ownership --------------------------------------


class TestObserverValidationOwnership(EventAssertions):
    def test_the_route_declares_no_geodetic_domain(self):
        source = executable_source(
            inspect.getsource(server.sunset_event_after)
        )
        for token in ("90", "180", "-90", "-180", "DOMAIN", "isfinite",
                      "math."):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_the_route_normalizes_no_coordinate(self):
        source = executable_source(
            inspect.getsource(server.sunset_event_after)
        )
        for token in ("abs(", "round(", "% 360", "min(", "max("):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_the_substrate_owns_the_rejection(self):
        """The reason is the substrate's own, unchanged at the route."""
        with self.assertRaises(SunsetChronologyError) as caught:
            find_sunset_from_instant(MODERN_ANCHOR, 91.0, 0.0)
        self.assertEqual(caught.exception.reason,
                         REASON_OBSERVER_OUT_OF_DOMAIN)
        error = self.assert_rejects(
            encode_tt_bits(MODERN_ANCHOR), REASON_OBSERVER_OUT_OF_DOMAIN,
            latitude=91.0,
        )
        self.assertEqual(error.detail["reason"], caught.exception.reason)

    def test_domain_edges_are_accepted(self):
        for latitude, longitude, label in (
            (90.0, 0.0, "north pole"), (-90.0, 0.0, "south pole"),
            (0.0, 180.0, "antimeridian"), (0.0, -180.0, "antimeridian west"),
        ):
            with self.subTest(case=label):
                try:
                    event_response(encode_tt_bits(MODERN_ANCHOR),
                                   latitude=latitude, longitude=longitude)
                except HTTPException as error:
                    # A pole may legitimately have no sunset; it must never
                    # be rejected as an out-of-domain observer.
                    self.assertEqual(error.status_code, 404)
                    self.assertEqual(error.detail["reason"],
                                     REASON_SUNSET_EVENT_ABSENT)


# --- 9. One-direction search only ------------------------------------------


class TestExactlyOneDirectionalSearch(EventAssertions):
    def test_route_calls_the_directional_solver_exactly_once(self):
        calls = []
        real = server.find_sunset_from_instant

        def recorder(tt, latitude, longitude):
            calls.append((tt, latitude, longitude))
            return real(tt, latitude, longitude)

        with mock.patch.object(
            server, "find_sunset_from_instant", recorder
        ):
            event_response(encode_tt_bits(MODERN_ANCHOR))
        self.assertEqual(len(calls), 1)

    def test_route_never_invokes_the_bracket_or_any_other_solver(self):
        import astronomy_solver
        watched = (
            "find_sunset_bracket",
            "find_sunset_predecessor",
            "find_sunset_successor",
            "find_next_sunset_after_utc",
            "find_sunset_utc",
        )
        with mock.patch.multiple(
            astronomy_solver,
            **{name: mock.DEFAULT for name in watched}
        ) as patched:
            with mock.patch.multiple(
                server,
                **{name: mock.DEFAULT for name in ("find_sunset_bracket",
                                                   "find_sunset_successor",
                                                   "find_sunset_utc",
                                                   "find_next_sunset_after_utc")}
            ) as server_patched:
                event_response(encode_tt_bits(MODERN_ANCHOR))
            for name, stub in patched.items():
                with self.subTest(callee=name):
                    stub.assert_not_called()
            for name, stub in server_patched.items():
                with self.subTest(callee="server." + name):
                    stub.assert_not_called()

    def test_route_source_references_only_the_directional_solver(self):
        tree = ast.parse(
            inspect.getsource(server.sunset_event_after).strip()
        )
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("find_sunset_from_instant", called)
        for forbidden in ("find_sunset_bracket", "find_sunset_predecessor",
                          "find_sunset_successor", "find_sunset_utc",
                          "find_next_sunset_after_utc",
                          "supported_search_frontier", "load_kernel",
                          "choose_kernel_name"):
            with self.subTest(callee=forbidden):
                self.assertNotIn(forbidden, called)


# --- 10. Route structural isolation ----------------------------------------


class TestRouteSourceGuard(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(server)
        index = source.find(A3C4_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3c-4 block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3C4_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.executable = executable_source(
            inspect.getsource(server.sunset_event_after)
        )

    def test_marker_obeys_the_governed_grammar(self):
        self.assertTrue(GOVERNED_BLOCK_MARKER.match(A3C4_BLOCK_MARKER))

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def sunset_event_after", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A3C4_BLOCK_MARKER)),
            "the A3c-4 scan reaches into a later governed block",
        )

    def test_route_performs_no_astronomy(self):
        # ".observe(" rather than a bare "observe": the forbidden thing is
        # Skyfield's vector call, and the noun "observer" occurs
        # legitimately in the executable absence message. The call form
        # cannot false-positive on it.
        for token in ("almanac", "find_discrete", "seasons", "wgs84",
                      "sunrise_sunset", "ts.", "tt_jd", ".observe("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_route_performs_no_artifact_selection(self):
        for token in ("load_kernel", "choose_kernel_name", "kernel_coverage",
                      "select_kernel", "de440", "de441", ".bsp"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_route_duplicates_no_codec_and_builds_no_response(self):
        for token in ("struct", "encode_tt_bits", "return {", "float(",
                      "isfinite"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)
        self.assertIn("decode_tt_bits", self.executable)
        self.assertIn("project_sunset_event", self.executable)

    def test_route_catches_exactly_two_governed_types(self):
        tree = ast.parse(
            inspect.getsource(server.sunset_event_after).strip()
        )
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
        self.assertEqual(len(handlers), 2)
        caught = [h.type.id for t in handlers for h in t.handlers]
        self.assertEqual(caught,
                         ["ExactTimeTransportError", "SunsetChronologyError"])

    def test_route_has_no_broad_exception_handler(self):
        for token in ("except Exception", "except BaseException", "except:"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_route_has_no_cursor_or_witness_machinery(self):
        for token in ("cursor", "witness", "encode_sunset_cursor",
                      "decode_sunset_cursor", "sunset_cursor"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_route_has_no_civil_or_calendar_machinery(self):
        for token in ("datetime", "parse_utc_datetime", "isoformat",
                      "strftime", "timezone", "timedelta", "sky_year",
                      "year", "hpc", "sce", "telma", "weekday", "equinox"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_route_introduces_no_epsilon_or_tolerance(self):
        for token in ("epsilon", "tolerance", "atol", "rtol", "isclose",
                      "min_gap", "3600"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_route_declares_exactly_one_reason_constant(self):
        tree = ast.parse(self.block)
        self.assertEqual(
            [n.targets[0].id for n in tree.body
             if isinstance(n, ast.Assign)],
            ["REASON_SUNSET_EVENT_ABSENT"],
        )

    def test_block_records_the_deployment_provenance_concern(self):
        """Cross-restart environment consistency is out of scope here.

        The governance ruling requires that the boundary be recorded in
        prose rather than silently assumed, so the prose is asserted.
        """
        lowered = self.block.lower()
        self.assertIn("deployment-provenance", lowered)
        self.assertIn("production acceptance", lowered)

    def test_runtime_gate_still_precedes_every_astronomy_import(self):
        source = inspect.getsource(server)
        gate = source.find("verify_runtime_scientific_components()")
        self.assertNotEqual(gate, -1)
        self.assertLess(gate, source.find("from astronomy_solver import"))


# --- 11. Previously published contracts unchanged --------------------------


class TestExistingContractsUnchanged(unittest.TestCase):
    EXACT_ROUTES = (
        ("/solar-longitude-event-before", "solar_longitude_event_before",
         ["ttBits", "kind"]),
        ("/solar-longitude-event-after", "solar_longitude_event_after",
         ["ttBits", "kind"]),
        ("/sunset-bracket", "sunset_bracket",
         ["ttBits", "latitude", "longitude"]),
        ("/solar-longitude-event-in-year", "solar_longitude_event_in_year",
         ["year", "kind"]),
    )
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
        ("/sunset-after", "sunset_after",
         ["afterUTC", "latitude", "longitude"]),
        ("/sunset-successor", "sunset_successor",
         ["cursor", "latitude", "longitude"]),
    )
    FRAMEWORK_ROUTES = frozenset(
        {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    )

    def test_every_preexisting_route_is_registered_unchanged(self):
        found = registered_routes()
        for path, name, parameters in self.LEGACY_ROUTES + self.EXACT_ROUTES:
            with self.subTest(path=path):
                self.assertIn(path, found)
                methods, endpoint_name, endpoint_parameters = found[path]
                self.assertEqual(methods - {"HEAD"}, {"GET"})
                self.assertEqual(endpoint_name, name)
                self.assertEqual(endpoint_parameters, parameters)

    def test_a3c4_added_exactly_one_application_route(self):
        found = set(registered_routes()) - self.FRAMEWORK_ROUTES
        expected = {
            path for path, _n, _p in self.LEGACY_ROUTES + self.EXACT_ROUTES
        } | {EVENT_AFTER_PATH}
        self.assertEqual(found, expected)
        self.assertEqual(len(found), 17)

    def test_legacy_sunset_successor_still_takes_a_cursor(self):
        self.assertEqual(registered_routes()["/sunset-successor"][2],
                         ["cursor", "latitude", "longitude"])

    def test_exact_routes_still_reject_civil_parameters(self):
        for path, _name, parameters in self.EXACT_ROUTES:
            if path == "/solar-longitude-event-in-year":
                continue
            with self.subTest(path=path):
                self.assertNotIn("year", parameters)
                self.assertNotIn("date", parameters)

    def test_a3c1_structured_detail_is_unchanged(self):
        with self.assertRaises(HTTPException) as caught:
            server.solar_longitude_event_after(ttBits="nope", kind="x")
        self.assertIsInstance(caught.exception.detail, dict)
        self.assertEqual(tuple(caught.exception.detail), ("reason", "message"))

    def test_a3c3_structured_detail_is_unchanged(self):
        with self.assertRaises(HTTPException) as caught:
            server.solar_longitude_event_in_year(year=2020.5, kind="x")
        self.assertIsInstance(caught.exception.detail, dict)
        self.assertEqual(caught.exception.detail["reason"],
                         "EVENT_YEAR_INVALID")

    def test_legacy_failure_still_carries_string_detail(self):
        with self.assertRaises(HTTPException) as caught:
            server.equinox("not-a-year")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIsInstance(caught.exception.detail, str)

    def test_legacy_health_is_unchanged(self):
        response = server.health()
        self.assertEqual(set(response), {"status", "defaultKernel"})
        self.assertEqual(response["status"], "ok")


if __name__ == "__main__":
    unittest.main()
