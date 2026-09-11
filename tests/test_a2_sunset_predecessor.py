"""A2-3b verification for the strict sunset predecessor.

Covers SunsetEvent and find_sunset_predecessor in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the directional reach and the
2650 defect window are declared as test-local literals. They are
deliberately NOT imported from astronomy_solver: importing a constant to
check that same constant would make the test agree with a defective value
instead of detecting it.

Sunset crossings are enumerated here by calling skyfield.almanac directly
over an independently constructed bracket. find_sunset_predecessor is never
used to produce its own expected answer.

supported_search_frontier IS used, as certified A2-3a infrastructure, to
recompute the frontier a call must have been bound to. It is not the unit
under test here; it has its own focused suite.

SCOPE

This file verifies one astronomical question: the latest genuine
observer-local topocentric apparent sunset strictly before an arbitrary
exact TT anchor. It asserts nothing about the successor direction, about
any route, or about HTTP behavior, none of which exist yet.

WHAT IS DELIBERATELY NOT ENCODED

No epsilon, tolerance, minimum gap or light-time second-count appears as an
expected value. Sunset identity is never established by numeric closeness:
where a relationship to the frozen A1b successor is exercised it is
asserted ordinally, never by bit equality across differently bracketed
searches and never against skyfield's internal search convergence value.

Python datetime behavior is not made part of the astronomical contract.
The deep-time test establishes that a determined event remains correct and
constructible whether or not a calendar rendering happens to exist.
"""

import ast
import dataclasses
import inspect
import math
import re
import unittest
from unittest import mock

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.errors import EphemerisRangeError

import astronomy_solver
from astronomy_solver import (
    choose_kernel_name,
    find_sunset_predecessor,
    find_sunset_successor,
    kernel_coverage_tt,
    load_kernel,
    supported_search_frontier,
    ts,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"

# The directional reach, declared test-locally.
SPAN_DAYS = 3.0

EXPECTED_EVENT_FIELDS = ("tt", "kernel")

# An ordinary interior anchor, well inside DE440's certified support.
INTERIOR_TT = 2460678.0

# Inside the window the frozen civil routing sends to DE440 even though
# DE440 publishes no data there.
DEFECT_WINDOW_TT = 2689118.000800741
DEFECT_WINDOW_CIVIL_YEAR = 2650

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9

OBSERVERS = (
    (40.7406, -73.9, "new-york"),
    (0.0, 0.0, "null-island"),
    (-33.9, 151.2, "sydney"),
    (35.7, 139.7, "tokyo"),
    (64.1, -21.9, "reykjavik"),
    (-54.8, -68.3, "ushuaia"),
    (0.0, 180.0, "antimeridian"),
)

A2_3B_BLOCK_MARKER = "# A2-3b - strict sunset predecessor."

# The shape of a governed production block header, used only to find where
# the A2-3b block ends. It matches the header FORM and nothing else.
GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-\d+[a-z]?)? - ",
                                   re.MULTILINE)


# --- Independent helpers ---------------------------------------------------


def coverage(kernel_name):
    return kernel_coverage_tt(kernel_name)


def union_bounds():
    lo = min(coverage(name).tt_start for name in PINNED_ARTIFACTS)
    hi = max(coverage(name).tt_end for name in PINNED_ARTIFACTS)
    return lo, hi


def predicate(kernel_name, latitude, longitude):
    return almanac.sunrise_sunset(
        load_kernel(kernel_name), wgs84.latlon(latitude, longitude)
    )


def frontier_for(anchor, latitude, longitude):
    """The backward frontier a predecessor call must be bound to."""
    return supported_search_frontier(
        anchor, anchor - SPAN_DAYS, latitude, longitude
    )


def oracle_sunsets(kernel_name, tt_lo, tt_hi, latitude, longitude):
    """Independent enumeration of sunset crossings inside a bracket.

    Built from skyfield.almanac directly. find_sunset_predecessor is never
    consulted, so it cannot produce its own expected answer.
    """
    times, events = almanac.find_discrete(
        ts.tt_jd(tt_lo),
        ts.tt_jd(tt_hi),
        predicate(kernel_name, latitude, longitude),
    )
    return [
        float(t.tt)
        for t, sun_is_up in zip(times, events)
        if not bool(sun_is_up)
    ]


def oracle_predecessor(anchor, latitude, longitude):
    """The expected answer, derived without the unit under test."""
    frontier = frontier_for(anchor, latitude, longitude)
    qualifying = [
        tt
        for tt in oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            latitude, longitude,
        )
        if tt < anchor
    ]
    return (qualifying[-1] if qualifying else None), frontier


def deep_time_anchor(offset_days):
    """An anchor above the absolute DE441 part 1 declared floor."""
    return coverage(DE441_PART_1).tt_start + offset_days


def executable_source(block):
    """Return a block's executable logic, prose excluded.

    An AST node's ``body`` attribute is not always a list - ``ast.IfExp``
    carries a single expression there - so the list check is required
    rather than defensive.
    """
    lines = block.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(ast.parse(block)):
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


class PredecessorAssertions(unittest.TestCase):
    def assert_fails_closed(self, reason, anchor,
                            latitude=REFERENCE_LATITUDE,
                            longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            find_sunset_predecessor(anchor, latitude, longitude)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def assert_event_invariants(self, event, anchor, latitude, longitude):
        self.assertIsInstance(event, astronomy_solver.SunsetEvent)
        self.assertIsInstance(event.tt, float)
        self.assertTrue(math.isfinite(event.tt))
        self.assertIn(event.kernel, PINNED_ARTIFACTS)

        # Strict ordering, exact binary64.
        self.assertLess(event.tt, anchor)

        frontier = frontier_for(anchor, latitude, longitude)
        self.assertEqual(event.kernel, frontier.kernel)
        self.assertGreaterEqual(event.tt, frontier.tt_lo)
        self.assertLessEqual(event.tt, frontier.tt_hi)

        # It is a genuine sunset under independent enumeration.
        self.assertIn(
            event.tt,
            oracle_sunsets(
                frontier.kernel, frontier.tt_lo, frontier.tt_hi,
                latitude, longitude,
            ),
        )


# --- 1. Strict predecessor semantics ---------------------------------------


class TestStrictPredecessor(PredecessorAssertions):
    def test_interior_anchor_returns_a_genuine_sunset(self):
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_result_matches_the_independent_oracle(self):
        expected, _frontier = oracle_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, expected)

    def test_strict_inequality_holds_for_the_global_observer_matrix(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                event = find_sunset_predecessor(
                    INTERIOR_TT, latitude, longitude
                )
                self.assertIsNotNone(event)
                self.assertLess(event.tt, INTERIOR_TT)
                self.assert_event_invariants(
                    event, INTERIOR_TT, latitude, longitude
                )

    def test_exact_sunset_root_anchor_excludes_that_root(self):
        """An anchor that IS a sunset returns the sunset before it."""
        frontier = supported_search_frontier(
            INTERIOR_TT, INTERIOR_TT + SPAN_DAYS,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        roots = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        root = roots[0]

        event = find_sunset_predecessor(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertNotEqual(
            event.tt, root, "the anchor's own crossing must be excluded"
        )
        self.assertLess(event.tt, root)

        # And it is the sunset immediately before it, by independent
        # enumeration rather than by any distance measure.
        expected, _ = oracle_predecessor(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, expected)

    def test_latest_qualifying_sunset_is_returned(self):
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        sunsets = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        qualifying = [tt for tt in sunsets if tt < INTERIOR_TT]
        self.assertGreater(
            len(qualifying), 1, "this probe needs several qualifying sunsets"
        )

        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, max(qualifying))

    def test_repeated_calls_are_deterministic(self):
        results = {
            find_sunset_predecessor(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
            for _ in range(3)
        }
        self.assertEqual(len(results), 1)

    def test_event_record_is_frozen_and_minimal(self):
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(event)),
            EXPECTED_EVENT_FIELDS,
        )
        with self.assertRaises(Exception):
            event.tt = 0.0


# --- 2. Absence versus exhaustion ------------------------------------------


class TestAbsenceVersusExhaustion(PredecessorAssertions):
    POLAR_CASES = (
        (78.2, 15.6, 2460850.0, "svalbard polar day"),
        (78.2, 15.6, 2460680.0, "svalbard polar night"),
        (-80.0, 0.0, 2460680.0, "antarctic polar day"),
        (-75.0, 120.0, 2460860.0, "antarctic polar night"),
    )

    def test_genuine_polar_absence_returns_none(self):
        for latitude, longitude, anchor, label in self.POLAR_CASES:
            with self.subTest(case=label):
                frontier = frontier_for(anchor, latitude, longitude)
                self.assertTrue(
                    frontier.complete,
                    "this probe requires a complete frontier",
                )
                self.assertEqual(
                    oracle_sunsets(
                        frontier.kernel, frontier.tt_lo, frontier.tt_hi,
                        latitude, longitude,
                    ),
                    [],
                    "this probe requires a genuinely sunset-free horizon",
                )
                self.assertIsNone(
                    find_sunset_predecessor(anchor, latitude, longitude)
                )

    def test_none_is_only_possible_when_the_frontier_is_complete(self):
        checked = 0
        for latitude, longitude, anchor, label in self.POLAR_CASES:
            with self.subTest(case=label):
                result = find_sunset_predecessor(anchor, latitude, longitude)
                if result is None:
                    frontier = frontier_for(anchor, latitude, longitude)
                    self.assertTrue(frontier.complete)
                    self.assertIsNone(frontier.truncation_reason)
                    checked += 1
        self.assertGreater(checked, 0, "no None outcome was exercised")

    def test_truncated_frontier_with_a_qualifying_sunset_returns_it(self):
        anchor = deep_time_anchor(1.0)
        frontier = frontier_for(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertFalse(frontier.complete, "this probe needs truncation")
        self.assertEqual(frontier.truncation_reason, REASON_REACH_EXHAUSTED)

        event = find_sunset_predecessor(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_truncated_frontier_without_a_sunset_raises_its_reason(self):
        for offset in (0.1, 0.02):
            with self.subTest(offset=offset):
                anchor = deep_time_anchor(offset)
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertFalse(frontier.complete)
                self.assertEqual(
                    [
                        tt
                        for tt in oracle_sunsets(
                            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
                            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                        )
                        if tt < anchor
                    ],
                    [],
                    "this probe requires an empty truncated frontier",
                )

                error = self.assert_fails_closed(
                    frontier.truncation_reason, anchor
                )
                self.assertEqual(error.reason, REASON_REACH_EXHAUSTED)

    def test_empty_truncated_frontier_never_returns_none(self):
        anchor = deep_time_anchor(0.1)
        with self.assertRaises(ScientificEnvironmentError):
            find_sunset_predecessor(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )


# --- 3. Data-authoritative coverage ----------------------------------------


class TestDataAuthoritativeCoverage(PredecessorAssertions):
    def test_de440_lower_transition_uses_de441_part_1(self):
        anchor = coverage(DE440).tt_start + 1.0
        event = find_sunset_predecessor(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_1)

    def test_de440_upper_transition_uses_de441_part_2(self):
        anchor = coverage(DE440).tt_end + 1.0
        event = find_sunset_predecessor(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_2)

    def test_interior_anchor_uses_de440(self):
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE440)

    def test_2650_legacy_defect_present_in_legacy_but_bypassed_here(self):
        # The frozen civil selector still points at the artifact with no data.
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)

        event = find_sunset_predecessor(
            DEFECT_WINDOW_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, DEFECT_WINDOW_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_2)

    def test_de441_part_1_absolute_lower_frontier(self):
        floor = coverage(DE441_PART_1).tt_start

        # Exactly at the declared floor there is no backward territory.
        self.assert_fails_closed(REASON_COVERAGE_EXHAUSTED, floor)

        # Above it, the recovered frontier yields a genuine event.
        anchor = deep_time_anchor(1.0)
        event = find_sunset_predecessor(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_1)
        self.assertLess(event.tt, anchor)
        self.assertGreater(event.tt, floor)

    def test_anchor_outside_the_authoritative_union_fails_closed(self):
        lo, hi = union_bounds()
        for anchor in (lo - 1.0, lo - 1e6, hi + 1.0, hi + 1e6):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(REASON_COVERAGE_EXHAUSTED, anchor)

    def test_event_kernel_and_tt_belong_to_the_admitted_frontier(self):
        anchors = (
            INTERIOR_TT,
            coverage(DE440).tt_start + 1.0,
            coverage(DE440).tt_end + 1.0,
            DEFECT_WINDOW_TT,
            deep_time_anchor(1.0),
        )
        for anchor in anchors:
            with self.subTest(anchor=anchor):
                event = find_sunset_predecessor(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(event.kernel, frontier.kernel)
                self.assertGreaterEqual(event.tt, frontier.tt_lo)
                self.assertLessEqual(event.tt, frontier.tt_hi)


# --- 4. Input and failure propagation --------------------------------------


class TestFailurePropagation(PredecessorAssertions):
    BAD_TT = (
        True, False, "2461485.0", None, [2461485.0], {}, object(),
        complex(1, 0), float("nan"), float("inf"), float("-inf"), 10 ** 400,
    )

    BAD_OBSERVERS = (
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        (95.0, 0.0),
        (-90.5, 0.0),
        (0.0, 181.0),
        (0.0, -180.5),
        (True, 0.0),
        ("40.0", 0.0),
        (None, 0.0),
    )

    def test_malformed_anchor_reports_instant_state_invalid(self):
        for value in self.BAD_TT:
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(REASON_INSTANT_INVALID, value)

    def test_invalid_observer_reports_observer_out_of_domain(self):
        for latitude, longitude in self.BAD_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    INTERIOR_TT, latitude, longitude,
                )

    def test_malformed_observer_never_masquerades_as_reach_exhaustion(self):
        """A NaN latitude reaches the ephemeris as an invalid cast.

        Unguarded it surfaces as an ephemeris range error, which is the
        signal A2-3 reads as exhausted computational reach.
        """
        # The hazard is real: the raw computation genuinely raises.
        with self.assertRaises(EphemerisRangeError):
            predicate(DE440, float("nan"), 0.0)(ts.tt_jd(INTERIOR_TT))

        for latitude, longitude in self.BAD_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                error = self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    INTERIOR_TT, latitude, longitude,
                )
                self.assertNotEqual(error.reason, REASON_REACH_EXHAUSTED)
                self.assertNotEqual(error.reason, REASON_COVERAGE_EXHAUSTED)


class TestInteriorFailureBackstop(PredecessorAssertions):
    """An unexpected failure inside an admitted frontier must fail closed."""

    @staticmethod
    def _range_error():
        return EphemerisRangeError(
            "stub segment range", None, None, None, None
        )

    def test_interior_ephemeris_failure_becomes_reach_exhausted(self):
        error = self._range_error()
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=error
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                find_sunset_predecessor(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertEqual(caught.exception.reason, REASON_REACH_EXHAUSTED)

    def test_original_exception_is_preserved_as_cause(self):
        error = self._range_error()
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=error
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                find_sunset_predecessor(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertIs(caught.exception.__cause__, error)

    def test_there_is_no_retry_kernel_switch_or_continuation(self):
        recorder = mock.Mock(side_effect=self._range_error())
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", recorder
        ):
            with self.assertRaises(ScientificEnvironmentError):
                find_sunset_predecessor(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertEqual(
            recorder.call_count, 1,
            "the search must not be retried after an interior failure",
        )

    def test_no_alternative_artifact_is_ever_selected_in_the_block(self):
        """Structural proof that no second artifact can be reached."""
        body = executable_source(
            inspect.getsource(astronomy_solver.find_sunset_predecessor)
        )
        self.assertNotIn("PINNED_KERNEL_PRECEDENCE", body)
        self.assertNotIn("select_kernel_containing_instant", body)
        self.assertNotIn("select_kernel_for_interval", body)
        self.assertNotIn("_admit_bracket", body)
        self.assertEqual(
            body.count("find_discrete"), 1,
            "exactly one search may be performed",
        )
        self.assertEqual(
            body.count("supported_search_frontier"), 1,
            "exactly one frontier may be requested",
        )
        self.assertEqual(body.count("load_kernel"), 1)


# --- 5. Search containment -------------------------------------------------


class TestSearchContainment(PredecessorAssertions):
    """The search must never examine TT outside the admitted frontier."""

    def anchors(self):
        return (
            ("interior", INTERIOR_TT),
            ("de440 lower", coverage(DE440).tt_start + 1.0),
            ("de440 upper", coverage(DE440).tt_end + 1.0),
            ("2650 defect", DEFECT_WINDOW_TT),
            ("deep-time truncated", deep_time_anchor(1.0)),
        )

    def run_instrumented(self, anchor, latitude, longitude):
        """Record every bracket and every sampled TT the search touches."""
        real_find_discrete = almanac.find_discrete
        brackets = []
        sampled = []

        def spy(t0, t1, f):
            brackets.append((float(t0.tt), float(t1.tt)))

            def watched(t):
                sampled.extend(
                    float(x)
                    for x in np.atleast_1d(np.asarray(t.tt)).ravel()
                )
                return f(t)

            watched.step_days = f.step_days
            return real_find_discrete(t0, t1, watched)

        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", spy
        ):
            event = find_sunset_predecessor(anchor, latitude, longitude)

        return event, brackets, sampled

    def test_search_bracket_equals_the_admitted_frontier(self):
        for label, anchor in self.anchors():
            with self.subTest(case=label):
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                _event, brackets, _sampled = self.run_instrumented(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(
                    brackets, [(frontier.tt_lo, frontier.tt_hi)],
                    "exactly one search, bounded by the frontier",
                )

    def test_no_sample_falls_outside_the_frontier(self):
        for label, anchor in self.anchors():
            with self.subTest(case=label):
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                _event, _brackets, sampled = self.run_instrumented(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertTrue(sampled, "the search sampled nothing")
                self.assertGreaterEqual(min(sampled), frontier.tt_lo)
                self.assertLessEqual(max(sampled), frontier.tt_hi)

    def test_unsupported_territory_below_the_frontier_is_never_examined(self):
        """At the deep-time edge the frontier stops above the declared floor."""
        anchor = deep_time_anchor(1.0)
        frontier = frontier_for(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        declared_floor = coverage(DE441_PART_1).tt_start
        self.assertGreater(
            frontier.tt_lo, declared_floor,
            "this probe requires a reach-truncated frontier",
        )

        _event, _brackets, sampled = self.run_instrumented(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertGreaterEqual(min(sampled), frontier.tt_lo)
        self.assertGreater(min(sampled), declared_floor)


# --- 6. Deep-time record ---------------------------------------------------


class TestDeepTimeRecord(PredecessorAssertions):
    """A determined event stays correct whether or not a calendar exists.

    The astronomical contract is the exact TT state and its provenance.
    Whether some presentation layer can render that state as a civil
    calendar value is a property of that layer, asserted here only as the
    reason the record carries no transport field - never as astronomy.
    """

    def setUp(self):
        self.anchor = deep_time_anchor(1.0)
        self.event = find_sunset_predecessor(
            self.anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_a_genuine_deep_time_predecessor_is_returned(self):
        self.assert_event_invariants(
            self.event, self.anchor,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        self.assertEqual(self.event.kernel, DE441_PART_1)

    def test_event_carries_no_transport_field(self):
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(self.event)),
            EXPECTED_EVENT_FIELDS,
        )
        for forbidden in ("utc", "iso", "isoformat", "datetime", "unix"):
            with self.subTest(field=forbidden):
                self.assertFalse(hasattr(self.event, forbidden))

    def test_exact_state_round_trips_through_the_certified_timescale(self):
        self.assertEqual(float(ts.tt_jd(self.event.tt).tt), self.event.tt)

    def test_event_is_correct_regardless_of_calendar_renderability(self):
        """The record is constructible either way; astronomy is unaffected."""
        t = ts.tt_jd(self.event.tt)
        try:
            t.utc_datetime()
            renderable = True
        except (ValueError, OverflowError):
            renderable = False

        # Whatever the presentation layer can or cannot do, the determined
        # event is unchanged and still satisfies every astronomical claim.
        self.assertIsInstance(renderable, bool)
        self.assertLess(self.event.tt, self.anchor)
        self.assertEqual(self.event.kernel, DE441_PART_1)
        self.assertEqual(float(ts.tt_jd(self.event.tt).tt), self.event.tt)


# --- 7. Relationship to the frozen A1b successor ---------------------------


class TestA1SuccessorRelationship(PredecessorAssertions):
    """Ordinal only. A1 composition is supporting evidence, not identity.

    No HPC-authored tolerance, no skyfield search convergence value and no
    bit equality across differently bracketed searches is used or implied.
    """

    def test_successor_of_the_predecessor_is_strictly_later(self):
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        successor = find_sunset_successor(
            event.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsNotNone(successor)
        self.assertGreater(successor.tt, event.tt)

    def test_predecessor_of_that_successor_is_strictly_earlier(self):
        event = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        successor = find_sunset_successor(
            event.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        back = find_sunset_predecessor(
            successor.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsNotNone(back)
        self.assertLess(back.tt, successor.tt)

    def test_frozen_successor_surface_is_untouched(self):
        self.assertTrue(callable(find_sunset_successor))
        self.assertIn(
            "min_gap_seconds",
            inspect.getsource(astronomy_solver.find_next_sunset_after_utc),
        )


# --- 8. Structural guards --------------------------------------------------


class TestA2_3bStructuralIndependence(unittest.TestCase):
    """The A2-3b block must not inherit civil routing or legacy guards.

    Scoped to the A2-3b block's executable logic. The same tokens exist
    legitimately elsewhere in astronomy_solver.py, in frozen legacy code
    this increment does not touch, and in prose that names an excluded
    mechanism in order to state that it is excluded.
    """

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A2_3B_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A2-3b block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A2_3B_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("class SunsetEvent", self.block)
        self.assertIn("def find_sunset_predecessor", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        """The scan must cover A2-3b and end there, not annex what follows."""
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A2_3B_BLOCK_MARKER)),
            "the A2-3b scan reaches into a later governed block",
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        self.assertIn("def find_sunset_predecessor", self.executable)
        self.assertIn("_exact_finite_tt", self.executable)
        self.assertIn("supported_search_frontier", self.executable)
        self.assertIn("frontier.kernel", self.executable)
        self.assertIn("frontier.tt_lo", self.executable)
        self.assertIn("frontier.tt_hi", self.executable)
        self.assertIn("almanac.sunrise_sunset", self.executable)
        self.assertIn("EphemerisRangeError", self.executable)
        self.assertIn("frontier.complete", self.executable)
        self.assertIn("frontier.truncation_reason", self.executable)
        self.assertIn("event_tt < anchor", self.executable)
        self.assertGreater(len(self.executable), 500)

        # Prose really was excluded.
        self.assertNotIn("STRICT ORDERING", self.executable)
        self.assertNotIn("ABSENCE IS NOT EXHAUSTION", self.executable)

    def test_block_does_not_use_civil_routing_or_legacy_guards(self):
        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "ts.utc(",
            "utc_datetime",
            "utc_strftime",
            "isoformat",
            "timestamp",
            "3600",
            "min_gap",
            "epsilon",
            "tolerance",
            "timedelta",
            "/ 15",
            "datetime",
            "weekday",
            "strftime",
            "lru_cache",
            "cache",
            "HTTPException",
            "status_code",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_encodes_no_light_time_margin(self):
        for token in ("480", "489", "490", "500", "507", "510", "520"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_observer_parameters_are_not_prohibited(self):
        """`latitude` and `longitude` are legitimate observer parameters."""
        self.assertIn("latitude", self.executable)
        self.assertIn("longitude", self.executable)

    def test_block_introduces_no_new_constant_or_import(self):
        tree = ast.parse(self.block)
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)]
        imports = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(assignments, [])
        self.assertEqual(imports, [])

    def test_block_defines_exactly_the_authorized_symbols(self):
        tree = ast.parse(self.block)
        defined = [
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(defined, ["SunsetEvent", "find_sunset_predecessor"])

    def test_frozen_legacy_code_is_still_present_ahead_of_the_block(self):
        """Guards against 'passing' by having deleted the legacy code."""
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def find_sunset_successor",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def select_kernel_for_interval",
            "def supported_search_frontier",
            "def _first_evaluable_state",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
