"""A2-3c verification for the strict arbitrary-instant sunset successor.

Covers find_sunset_from_instant in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the directional reach and the 2650
defect window are declared as test-local literals. They are deliberately
NOT imported from astronomy_solver: importing a constant to check that same
constant would make the test agree with a defective value instead of
detecting it.

Sunset crossings are enumerated here by calling skyfield.almanac directly
over an independently constructed bracket. find_sunset_from_instant is
never used to produce its own expected answer.

supported_search_frontier IS used, as certified A2-3a infrastructure, to
recompute the frontier a call must have been bound to. find_sunset_
predecessor is used only as certified A2-3b infrastructure for the ordinal
composition check. Neither is the unit under test here; both have their own
focused suites.

SCOPE

This file verifies one astronomical question: the earliest genuine
observer-local topocentric apparent sunset strictly after an arbitrary
exact TT anchor. It asserts nothing about any route or HTTP behavior.

WHAT IS DELIBERATELY NOT ENCODED

No epsilon, tolerance, minimum gap or light-time second-count appears as an
expected value. Sunset identity is never established by numeric closeness,
by skyfield's internal search convergence value, or by bit equality across
differently bracketed searches.

The relationship to frozen A1b is certified as an ARCHITECTURAL
distinction - A1b requires a post-transition continuation state and refuses
without one, while A2-3c accepts an arbitrary instant. Any bit-identical
agreement observed where brackets and kernels happen to coincide is
recorded as diagnostic evidence only and is never asserted.
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
    SunsetSuccessorError,
    choose_kernel_name,
    find_sunset_from_instant,
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

# A1b's own fail-closed reason when its continuation precondition is absent.
REASON_CURSOR_NOT_POST_TRANSITION = "CURSOR_NOT_POST_TRANSITION"

# The directional reach, declared test-locally.
SPAN_DAYS = 3.0

EXPECTED_EVENT_FIELDS = ("tt", "kernel")

INTERIOR_TT = 2460678.0

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

A2_3C_BLOCK_MARKER = "# A2-3c - strict arbitrary-instant sunset successor."

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
    """The forward frontier a successor call must be bound to."""
    return supported_search_frontier(
        anchor, anchor + SPAN_DAYS, latitude, longitude
    )


def oracle_sunsets(kernel_name, tt_lo, tt_hi, latitude, longitude):
    """Independent enumeration of sunset crossings inside a bracket."""
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


def oracle_successor(anchor, latitude, longitude):
    """The expected answer, derived without the unit under test."""
    frontier = frontier_for(anchor, latitude, longitude)
    qualifying = [
        tt
        for tt in oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            latitude, longitude,
        )
        if tt > anchor
    ]
    return (qualifying[0] if qualifying else None), frontier


def ceiling_anchor(offset_days):
    """An anchor below the absolute DE441 part 2 declared ceiling."""
    return coverage(DE441_PART_2).tt_end - offset_days


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


class SuccessorAssertions(unittest.TestCase):
    def assert_fails_closed(self, reason, anchor,
                            latitude=REFERENCE_LATITUDE,
                            longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            find_sunset_from_instant(anchor, latitude, longitude)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def assert_event_invariants(self, event, anchor, latitude, longitude):
        self.assertIsInstance(event, astronomy_solver.SunsetEvent)
        self.assertIsInstance(event.tt, float)
        self.assertTrue(math.isfinite(event.tt))
        self.assertIn(event.kernel, PINNED_ARTIFACTS)

        # Strict ordering, exact binary64.
        self.assertGreater(event.tt, anchor)

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


# --- 1. Strict successor semantics -----------------------------------------


class TestStrictSuccessor(SuccessorAssertions):
    def test_interior_anchor_returns_a_genuine_sunset(self):
        event = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_result_matches_the_independent_oracle(self):
        expected, _frontier = oracle_successor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        event = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, expected)

    def test_strict_inequality_holds_for_the_global_observer_matrix(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                event = find_sunset_from_instant(
                    INTERIOR_TT, latitude, longitude
                )
                self.assertIsNotNone(event)
                self.assertGreater(event.tt, INTERIOR_TT)
                self.assert_event_invariants(
                    event, INTERIOR_TT, latitude, longitude
                )

    def test_exact_sunset_root_anchor_excludes_that_root(self):
        """An anchor that IS a sunset returns the sunset after it."""
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        roots = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        root = roots[0]

        event = find_sunset_from_instant(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertNotEqual(
            event.tt, root, "the anchor's own crossing must be excluded"
        )
        self.assertGreater(event.tt, root)

        expected, _ = oracle_successor(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, expected)

    def test_earliest_qualifying_sunset_is_returned(self):
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        sunsets = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        qualifying = [tt for tt in sunsets if tt > INTERIOR_TT]
        self.assertGreater(
            len(qualifying), 1, "this probe needs several qualifying sunsets"
        )

        event = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.tt, min(qualifying))

    def test_repeated_calls_are_deterministic(self):
        results = {
            find_sunset_from_instant(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
            for _ in range(3)
        }
        self.assertEqual(len(results), 1)

    def test_event_record_is_reused_frozen_and_minimal(self):
        event = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(event)),
            EXPECTED_EVENT_FIELDS,
        )
        with self.assertRaises(Exception):
            event.tt = 0.0


# --- 2. Composition with published A2-3b -----------------------------------


class TestPredecessorSuccessorComposition(SuccessorAssertions):
    """Ordinal only. No tolerance, no cross-bracket identity rule."""

    def test_anchor_is_strictly_bracketed_by_both_directions(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                before = find_sunset_predecessor(
                    INTERIOR_TT, latitude, longitude
                )
                after = find_sunset_from_instant(
                    INTERIOR_TT, latitude, longitude
                )
                self.assertIsNotNone(before)
                self.assertIsNotNone(after)
                self.assertLess(before.tt, INTERIOR_TT)
                self.assertLess(INTERIOR_TT, after.tt)

    def test_both_directions_agree_on_provenance(self):
        before = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        after = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(before.kernel, DE440)
        self.assertEqual(after.kernel, DE440)

    def test_each_direction_is_adjacent_within_its_own_bracket(self):
        """Adjacency is asserted per direction, never across brackets.

        Within the successor's own frontier the returned event is the
        earliest crossing after the anchor, so nothing separates them. The
        same holds for the predecessor within its own frontier. A combined
        claim is deliberately not made: no single bracket contains both
        results, and any third bracket re-reports the same physical
        crossings at slightly different binary64 values, so such a claim
        would silently require differently bracketed searches to agree bit
        for bit - the cross-bracket event-identity rule this file refuses
        to encode.
        """
        after = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        forward = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(
            [
                tt
                for tt in oracle_sunsets(
                    forward.kernel, forward.tt_lo, forward.tt_hi,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )
                if INTERIOR_TT < tt < after.tt
            ],
            [],
        )

        before = find_sunset_predecessor(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        backward = supported_search_frontier(
            INTERIOR_TT, INTERIOR_TT - SPAN_DAYS,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        self.assertEqual(
            [
                tt
                for tt in oracle_sunsets(
                    backward.kernel, backward.tt_lo, backward.tt_hi,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )
                if before.tt < tt < INTERIOR_TT
            ],
            [],
        )


# --- 3. Absence versus exhaustion ------------------------------------------


class TestAbsenceVersusExhaustion(SuccessorAssertions):
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
                    find_sunset_from_instant(anchor, latitude, longitude)
                )

    def test_none_is_only_possible_when_the_frontier_is_complete(self):
        checked = 0
        for latitude, longitude, anchor, label in self.POLAR_CASES:
            with self.subTest(case=label):
                result = find_sunset_from_instant(anchor, latitude, longitude)
                if result is None:
                    frontier = frontier_for(anchor, latitude, longitude)
                    self.assertTrue(frontier.complete)
                    self.assertIsNone(frontier.truncation_reason)
                    checked += 1
        self.assertGreater(checked, 0, "no None outcome was exercised")

    def test_truncated_frontier_with_a_qualifying_sunset_returns_it(self):
        for offset in (1.0, 0.4):
            with self.subTest(offset=offset):
                anchor = ceiling_anchor(offset)
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertFalse(
                    frontier.complete, "this probe needs truncation"
                )
                self.assertEqual(
                    frontier.truncation_reason, REASON_COVERAGE_EXHAUSTED
                )

                event = find_sunset_from_instant(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assert_event_invariants(
                    event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

    def test_truncated_frontier_without_a_sunset_raises_coverage_exhausted(self):
        for offset in (0.05, 0.005):
            with self.subTest(offset=offset):
                anchor = ceiling_anchor(offset)
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
                        if tt > anchor
                    ],
                    [],
                    "this probe requires an empty truncated frontier",
                )

                error = self.assert_fails_closed(
                    REASON_COVERAGE_EXHAUSTED, anchor
                )
                self.assertEqual(error.reason, frontier.truncation_reason)

    def test_exact_absolute_ceiling_fails_closed(self):
        self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, coverage(DE441_PART_2).tt_end
        )

    def test_empty_truncated_frontier_never_returns_none(self):
        anchor = ceiling_anchor(0.05)
        with self.assertRaises(ScientificEnvironmentError):
            find_sunset_from_instant(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )


# --- 4. Coverage and provenance --------------------------------------------


class TestCoverageAndProvenance(SuccessorAssertions):
    def test_de440_lower_transition_uses_de441_part_1(self):
        for anchor in (
            coverage(DE440).tt_start - 1.0,
            coverage(DE440).tt_start,
        ):
            with self.subTest(anchor=anchor):
                event = find_sunset_from_instant(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assert_event_invariants(
                    event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(event.kernel, DE441_PART_1)

    def test_de440_upper_transition_uses_de441_part_2(self):
        for anchor in (
            coverage(DE440).tt_end - 1.0,
            coverage(DE440).tt_end + 1.0,
        ):
            with self.subTest(anchor=anchor):
                event = find_sunset_from_instant(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assert_event_invariants(
                    event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(event.kernel, DE441_PART_2)

    def test_interior_anchor_uses_de440(self):
        event = find_sunset_from_instant(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE440)

    def test_de441_part_2_absolute_upper_edge(self):
        ceiling = coverage(DE441_PART_2).tt_end

        event = find_sunset_from_instant(
            ceiling - 1.0, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_2)
        self.assertGreater(event.tt, ceiling - 1.0)
        self.assertLessEqual(
            event.tt, ceiling,
            "no event may lie beyond the declared certified ceiling",
        )

    def test_2650_legacy_defect_present_in_legacy_but_bypassed_here(self):
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)

        event = find_sunset_from_instant(
            DEFECT_WINDOW_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, DEFECT_WINDOW_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(event.kernel, DE441_PART_2)

    def test_anchor_outside_the_authoritative_union_fails_closed(self):
        lo, hi = union_bounds()
        for anchor in (lo - 1.0, lo - 1e6, hi + 1.0, hi + 1e6):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(REASON_COVERAGE_EXHAUSTED, anchor)

    def test_event_belongs_to_the_admitted_frontier(self):
        anchors = (
            INTERIOR_TT,
            coverage(DE440).tt_start - 1.0,
            coverage(DE440).tt_end + 1.0,
            DEFECT_WINDOW_TT,
            ceiling_anchor(1.0),
        )
        for anchor in anchors:
            with self.subTest(anchor=anchor):
                event = find_sunset_from_instant(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                frontier = frontier_for(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(event.kernel, frontier.kernel)
                self.assertGreaterEqual(event.tt, frontier.tt_lo)
                self.assertLessEqual(event.tt, frontier.tt_hi)
                self.assertTrue(
                    coverage(frontier.kernel).tt_start <= event.tt
                    <= coverage(frontier.kernel).tt_end,
                    "the event must lie inside declared certified coverage",
                )


# --- 5. Input and failure propagation --------------------------------------


class TestFailurePropagation(SuccessorAssertions):
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

    def test_malformed_observer_never_masquerades_as_exhaustion(self):
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


class TestInteriorFailureBackstop(SuccessorAssertions):
    """An unexpected failure inside an admitted frontier must fail closed."""

    @staticmethod
    def _range_error():
        return EphemerisRangeError(
            "stub segment range", None, None, None, None
        )

    def test_interior_ephemeris_failure_becomes_reach_exhausted(self):
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete",
            side_effect=self._range_error(),
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                find_sunset_from_instant(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertEqual(caught.exception.reason, REASON_REACH_EXHAUSTED)

    def test_original_exception_is_preserved_as_cause(self):
        error = self._range_error()
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=error
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                find_sunset_from_instant(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertIs(caught.exception.__cause__, error)

    def test_exactly_one_search_with_no_retry(self):
        recorder = mock.Mock(side_effect=self._range_error())
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", recorder
        ):
            with self.assertRaises(ScientificEnvironmentError):
                find_sunset_from_instant(
                    INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )

        self.assertEqual(
            recorder.call_count, 1,
            "the search must not be retried after an interior failure",
        )

    def test_no_alternative_artifact_can_be_reached(self):
        """Structural proof that no kernel switch or stitching is possible."""
        body = executable_source(
            inspect.getsource(astronomy_solver.find_sunset_from_instant)
        )
        self.assertNotIn("PINNED_KERNEL_PRECEDENCE", body)
        self.assertNotIn("select_kernel_containing_instant", body)
        self.assertNotIn("select_kernel_for_interval", body)
        self.assertNotIn("_admit_bracket", body)
        self.assertEqual(body.count("find_discrete"), 1)
        self.assertEqual(body.count("supported_search_frontier"), 1)
        self.assertEqual(body.count("load_kernel"), 1)


# --- 6. Architectural distinction from frozen A1b --------------------------


class TestA1bArchitecturalDistinction(SuccessorAssertions):
    """A1b requires a post-transition continuation state; A2-3c does not.

    Bit-identical agreement where brackets and kernels happen to coincide
    is NOT asserted here. It is recorded in the diagnostic test below as
    evidence only, with no equality requirement.
    """

    def midday_anchor(self):
        """An arbitrary instant at which the Sun is up for the observer."""
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        root = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )[0]
        return root - 0.25, frontier.kernel

    def test_the_chosen_anchor_really_has_the_sun_up(self):
        anchor, kernel = self.midday_anchor()
        self.assertTrue(
            bool(
                predicate(
                    kernel, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )(ts.tt_jd(anchor))
            ),
            "the probe requires an anchor where the Sun is up",
        )

    def test_frozen_a1b_refuses_an_arbitrary_midday_anchor(self):
        anchor, _kernel = self.midday_anchor()
        with self.assertRaises(SunsetSuccessorError) as caught:
            find_sunset_successor(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        self.assertEqual(
            caught.exception.reason, REASON_CURSOR_NOT_POST_TRANSITION
        )

    def test_a2_3c_succeeds_from_that_same_arbitrary_anchor(self):
        anchor, _kernel = self.midday_anchor()
        event = find_sunset_from_instant(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_event_invariants(
            event, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertGreater(event.tt, anchor)

    def test_a1b_precondition_is_the_distinguishing_mechanism(self):
        """A1b accepts the post-transition state it was designed for."""
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        root = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )[0]
        self.assertFalse(
            bool(
                predicate(
                    frontier.kernel, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )(ts.tt_jd(root))
            ),
            "a determined root is post-transition",
        )
        self.assertIsNotNone(
            find_sunset_successor(
                root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        )

    def test_diagnostic_only_agreement_is_not_a_contract(self):
        """Both are strictly after the anchor. Equality is NOT required."""
        frontier = frontier_for(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        root = oracle_sunsets(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )[0]
        a1b = find_sunset_successor(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        a23c = find_sunset_from_instant(
            root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertGreater(a1b.tt, root)
        self.assertGreater(a23c.tt, root)

    def test_frozen_a1b_surfaces_remain_untouched(self):
        self.assertIn(
            "min_gap_seconds",
            inspect.getsource(astronomy_solver.find_next_sunset_after_utc),
        )
        self.assertIn(
            "REASON_CURSOR_NOT_POST_TRANSITION",
            inspect.getsource(astronomy_solver.find_sunset_successor),
        )


# --- 7. Search containment -------------------------------------------------


class TestSearchContainment(SuccessorAssertions):
    def anchors(self):
        return (
            ("interior", INTERIOR_TT),
            ("de440 lower", coverage(DE440).tt_start - 1.0),
            ("de440 upper", coverage(DE440).tt_end + 1.0),
            ("2650 defect", DEFECT_WINDOW_TT),
            ("truncated ceiling", ceiling_anchor(1.0)),
        )

    def run_instrumented(self, anchor, latitude, longitude):
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
            event = find_sunset_from_instant(anchor, latitude, longitude)

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

    def test_no_sample_exceeds_the_declared_ceiling_when_truncated(self):
        anchor = ceiling_anchor(1.0)
        frontier = frontier_for(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        ceiling = coverage(DE441_PART_2).tt_end
        self.assertFalse(
            frontier.complete, "this probe requires a truncated frontier"
        )

        _event, _brackets, sampled = self.run_instrumented(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertLessEqual(max(sampled), ceiling)
        self.assertLessEqual(max(sampled), frontier.tt_hi)


# --- 8. Structural guards --------------------------------------------------


class TestA2_3cStructuralIndependence(unittest.TestCase):
    """The A2-3c block must not inherit civil routing or legacy guards."""

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A2_3C_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A2-3c block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A2_3C_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def find_sunset_from_instant", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A2_3C_BLOCK_MARKER)),
            "the A2-3c scan reaches into a later governed block",
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        self.assertIn("def find_sunset_from_instant", self.executable)
        self.assertIn("_exact_finite_tt", self.executable)
        self.assertIn("supported_search_frontier", self.executable)
        self.assertIn("SUCCESSOR_SEARCH_SPAN_DAYS", self.executable)
        self.assertIn("frontier.kernel", self.executable)
        self.assertIn("frontier.tt_lo", self.executable)
        self.assertIn("frontier.tt_hi", self.executable)
        self.assertIn("almanac.sunrise_sunset", self.executable)
        self.assertIn("almanac.find_discrete", self.executable)
        self.assertIn("EphemerisRangeError", self.executable)
        self.assertIn("frontier.complete", self.executable)
        self.assertIn("frontier.truncation_reason", self.executable)
        self.assertIn("SunsetEvent", self.executable)
        self.assertIn("event_tt > anchor", self.executable)
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
        self.assertIn("latitude", self.executable)
        self.assertIn("longitude", self.executable)

    def test_block_defines_exactly_the_authorized_symbol(self):
        tree = ast.parse(self.block)
        defined = [
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(defined, ["find_sunset_from_instant"])

    def test_block_introduces_no_constant_import_record_or_helper(self):
        tree = ast.parse(self.block)
        self.assertEqual(
            [n for n in tree.body if isinstance(n, ast.Assign)], []
        )
        self.assertEqual(
            [
                n for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ],
            [],
        )
        self.assertEqual(
            [n for n in tree.body if isinstance(n, ast.ClassDef)], []
        )
        self.assertEqual(
            len([n for n in tree.body if isinstance(n, ast.FunctionDef)]), 1
        )

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
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
            "class SunsetEvent",
            "def find_sunset_predecessor",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
